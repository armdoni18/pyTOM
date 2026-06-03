"""
F0_Main_PM_Source.py
====================

Assembly of the permanent-magnet contribution f_pm to the
right-hand side of Eq. (2):

    K(nu) * A = f + f_pm

For a uniformly magnetized region with constant residual flux
density B_r, the volume integral of curl(nu * B_r) reduces by
Stokes' theorem to a boundary integral. For each triangular
element belonging to a PM domain, this function loops over the
three edges, determines the outward unit normal, and accumulates
the contribution

    I_edge = K_bz * length(edge)

with

    K_bz = H_cx * n_y - H_cy * n_x ,
    H_c  = B_r / mu_0 ,

lumped in equal halves to the two endpoint nodes. The orientation
of the outward normal is determined from the signed area of the
triangle, so the convention is consistent for arbitrary mesh
node orderings.

Multiple PM domains with different magnitudes and orientations
are supported through parallel lists in the input dictionary:

    inputs["PM"] = {
        "domIDs": [7, 8, ...],     # domain identifiers
        "Br"    : [0.2, 1.4, ...], # residual flux densities (T)
        "theta" : [180.0, 90.0,..] # magnetization angle (deg)
    }
"""

import numpy as np

def F0_Main_PM_Source(fem, inputs):
    """
    Assemble the permanent-magnet load vector f_pm of Eq. (2).

    Parameters
    ----------
    fem : dict
        Finite-element data structure. Must contain ``IX``, ``X``,
        and ``ndof``.
    inputs : dict
        Problem input dictionary. The ``PM`` sub-dictionary
        described above is consulted; if absent or empty, a zero
        vector is returned.

    Returns
    -------
    T_pm : ndarray, shape (ndof,)
        Lumped nodal load vector representing the PM contribution.
    """

# Allocate the PM load vector; return zeros if no PM data is supplied
    ndof = int(fem["ndof"])
    T_pm = np.zeros(ndof, dtype=float)

    PM = inputs.get("PM", None)
    if PM is None:
        return T_pm

    # Read the parallel PM description lists: domain ID, Br, and angle.
    domIDs    = np.asarray(PM.get("domIDs", []), dtype=int).reshape(-1)
    Br_vals   = np.asarray(PM.get("Br",     []), dtype=float).reshape(-1)
    theta_deg = np.asarray(PM.get("theta",  []), dtype=float).reshape(-1)

    if domIDs.size == 0:
        return T_pm
    if not (len(domIDs) == len(Br_vals) == len(theta_deg)):
        raise ValueError("PM.domIDs, PM.Br, PM.theta must have same length")

    # Coercive field H_c = B_r / mu_0 for each PM domain.
    mu0     = 4.0 * np.pi * 1e-7
    Hc_vals = Br_vals / mu0

    # Resolve the (H_cx, H_cy) components from the magnetization angle once
    Hc_map = {}
    for k in range(len(domIDs)):
        th  = np.deg2rad(theta_deg[k])
        Hcx = Hc_vals[k] * np.cos(th)        # x-component of H_c
        Hcy = Hc_vals[k] * np.sin(th)        # y-component of H_c
        Hc_map[int(domIDs[k])] = (Hcx, Hcy)

    IX = fem["IX"][:, :4]
    X  = fem["X"]
    eps = np.finfo(float).eps

    # Assemble the boundary integral for each PM domain independently.
    for dom_id, (Hcx, Hcy) in Hc_map.items():
        mask = (IX[:, 3] == dom_id)
        if not np.any(mask):
            continue

        elems = IX[mask, :3]                 # PM triangle node ids (n_pm, 3), 1-based

        # Enumerate the three directed edges of each triangle: a->b, b->c, c->a.
        na_all = np.concatenate([elems[:, 0], elems[:, 1], elems[:, 2]])   # edge start nodes
        nb_all = np.concatenate([elems[:, 1], elems[:, 2], elems[:, 0]])   # edge end nodes

        xa = X[na_all - 1, 0];  ya = X[na_all - 1, 1]   # start coordinates
        xb = X[nb_all - 1, 0];  yb = X[nb_all - 1, 1]   # end coordinates

        vx = xb - xa                         # edge vector x
        vy = yb - ya                         # edge vector y
        l  = np.hypot(vx, vy)                # edge length
        valid = l > eps                      # skip degenerate edges

        # Signed area (x2) of each triangle; its sign fixes the normal orientation.
        n1 = elems[:, 0]; n2 = elems[:, 1]; n3 = elems[:, 2]
        x1 = X[n1-1,0]; y1 = X[n1-1,1]
        x2 = X[n2-1,0]; y2 = X[n2-1,1]
        x3 = X[n3-1,0]; y3 = X[n3-1,1]

        area2 = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        orient = np.where(np.concatenate([area2, area2, area2]) > 0, 1.0, -1.0)  # one orientation sign per edge

        # Edge tangent and outward normal.
        tx = np.where(valid, vx / (l + eps), 0.0)
        ty = np.where(valid, vy / (l + eps), 0.0)
        nx_out =  orient * ty                # outward normal x
        ny_out = -orient * tx                # outward normal y

        # Equivalent edge current K_bz = H_cx*n_y - H_cy*n_x, multiplied by length.
        Kbz   = Hcx * ny_out - Hcy * nx_out
        I_edge = np.where(valid, Kbz * l, 0.0)
        I2     = 0.5 * I_edge                # lump half to each endpoint node

        # Scatter the lumped contributions into the global PM load vector.
        np.add.at(T_pm, na_all - 1, I2)
        np.add.at(T_pm, nb_all - 1, I2)

    return T_pm
