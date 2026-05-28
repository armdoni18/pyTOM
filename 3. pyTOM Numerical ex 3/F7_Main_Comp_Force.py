"""
F7_Main_Comp_Force.py
=====================

Electromagnetic force on the plunger via Maxwell stress tensor integration along a closed air-side path that surrounds the plunger.

Theory link
-----------
The Maxwell stress tensor of Eq. (7),

    T = nu * (B o B - (1/2) |B|^2 I) ,

is integrated over a closed path Gamma (Eq. (8)) to give the electromagnetic force on the plunger. The closed path lies entirely in the air domain so that T is evaluated with the linear reluctivity nu_air;
integrating through the iron region would give an incorrect force because of the material nonlinearity gradient.

The discrete approximation of Eq. (9),

    F approx sum_k T_k n_k Delta s_k ,

is implemented as a vectorized sum over the edges of the closed air loop. The construction of that closed loop takes three steps:

  1. Identify the boundary edges of the plunger domain.
  2. Collect the air-domain triangles that share at least one node with the plunger boundary.
  3. Extract the outer envelope of those air triangles and remove the plunger-boundary edges themselves, leaving a closed loop on the air side of the plunger.

For each edge, the field quantities needed for T are taken from the nearest element to a midpoint shifted slightly inward (eps_shift = 1e-3 in mesh length units).
The small inward shift biases the nearest-element search to land on an air-side triangle instead of a plunger-side triangle, which guarantees that the reluctivity used is nu_air and not nu_iron.

The outward-normal orientation is enforced by comparing the normal with the vector from the edge midpoint to the plunger centroid: if the two have a positive dot product, the normal is flipped so it points away from the plunger.
"""

import numpy as np

def F7_Main_Comp_Force(fem):
    """
    Evaluate the electromagnetic force on the plunger.

    Parameters
    ----------
    fem : dict
        Finite-element data with the converged field. Must contain ``IX``, ``X``, ``ne``, ``nu_e``, ``Bx``, ``By``.

    Returns
    -------
    Fx_total : float
        x-component of the total force on the plunger.
    Fy_total : float
        y-component of the total force on the plunger.
    fem : dict
        Updated in place with:
            ``plunger_boundary_edges``
            ``cleaned_air_loop_around_plunger``
            ``mst`` : dict with the integration path and per-edge force contributions, used by F9 plotting.
            ``Fx_total``, ``Fy_total``
    """
    IX = fem["IX"]
    X  = fem["X"]
    ne = int(fem["ne"])

    # ============================================================
    # STEP 1: Plunger boundary edges (domain 5)
    # ============================================================
    plunger_mask = (IX[:, 3] == 5)
    plunger_elems = IX[plunger_mask, 0:3]   # (nPL, 3)

    pl_edges = np.vstack([
        plunger_elems[:, [0, 1]],
        plunger_elems[:, [1, 2]],
        plunger_elems[:, [2, 0]]
    ])
    pl_edges = np.sort(pl_edges, axis=1)
    unique_edges, counts = np.unique(pl_edges, axis=0, return_counts=True)
    plunger_boundary_edges = unique_edges[counts == 1]
    fem["plunger_boundary_edges"] = plunger_boundary_edges

    # ============================================================
    # STEP 2: Air triangles (domain 1) touching plunger boundary
    # ============================================================
    air_mask    = (IX[:, 3] == 1)
    air_elems   = IX[air_mask, 0:3]   # (nAIR, 3)

    # Set of plunger boundary nodes
    pl_nodes_set = set(plunger_boundary_edges.flatten().tolist())

    # Air triangle is "connected" if any of its nodes is a plunger boundary node
    air_has_pl = np.any(
        np.isin(air_elems, list(pl_nodes_set)), axis=1
    )
    connected_air = air_elems[air_has_pl]

    if connected_air.shape[0] == 0:
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    # ============================================================
    # STEP 3: Outer air loop (boundary edges of connected_air)
    # ============================================================
    air_edges = np.vstack([
        connected_air[:, [0, 1]],
        connected_air[:, [1, 2]],
        connected_air[:, [2, 0]]
    ])
    air_edges = np.sort(air_edges, axis=1)
    uniq_aedges, cnt_air = np.unique(air_edges, axis=0, return_counts=True)
    outer_air_loop = uniq_aedges[cnt_air == 1]

    # Remove plunger boundary edges from outer air loop
    pl_set   = set(map(tuple, np.sort(plunger_boundary_edges, axis=1).tolist()))
    air_clean = np.array([
        e for e in outer_air_loop
        if tuple(e.tolist()) not in pl_set
    ], dtype=int)

    if air_clean.shape[0] == 0:
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    fem["cleaned_air_loop_around_plunger"] = air_clean

    # ============================================================
    # Plunger centroid
    # ============================================================
    plunger_nodes  = np.unique(IX[plunger_mask, 0:3])
    plunger_center = np.mean(X[plunger_nodes - 1, :], axis=0)

    # ============================================================
    # STEP 4: Precompute element centroids for fast nearest search
    # ============================================================
    nodes012   = IX[:, 0:3] - 1              # (ne, 3)
    centroids  = np.mean(X[nodes012, :], axis=1)  # (ne, 2)  vectorized

    # ============================================================
    # STEP 5: Force integration — vectorized over edges
    # ============================================================
    Nedge = air_clean.shape[0]

    n1 = air_clean[:, 0] - 1   # (Nedge,)
    n2 = air_clean[:, 1] - 1

    p1 = X[n1, :]   # (Nedge, 2)
    p2 = X[n2, :]

    seg = p2 - p1             # (Nedge, 2)
    ds  = np.linalg.norm(seg, axis=1)   # (Nedge,)

    valid = ds > 0.0

    # Outward normal (un-normalized)
    normal_raw = np.column_stack([ seg[:, 1], -seg[:, 0]])
    norm_mag   = np.linalg.norm(normal_raw, axis=1, keepdims=True) + 1e-30
    normal     = normal_raw / norm_mag   # (Nedge, 2)

    mid = 0.5 * (p1 + p2)   # (Nedge, 2)

    # Flip normal if pointing toward plunger
    to_pl     = plunger_center[None, :] - mid   # (Nedge, 2)
    dot_sign  = np.sum(normal * to_pl, axis=1)   # (Nedge,)
    flip_mask = dot_sign > 0.0
    normal[flip_mask, :] *= -1.0

    # Shifted midpoints (toward plunger, i.e. inward).
    # The small inward shift biases the nearest-element search to
    # land on an air-side triangle (domain = 1) instead of a
    # plunger-side triangle (domain = 5), so the reluctivity used
    # in the Maxwell stress tensor is nu_air and not nu_iron.
    eps_shift   = 1e-3
    shifted_mid = mid + eps_shift * normal   # (Nedge, 2)

    # Vectorized nearest element: find argmin distance from shifted_mid to centroids
    from scipy.spatial.distance import cdist
    dist_mat   = cdist(shifted_mid, centroids)   # (Nedge, ne)
    closest_e  = np.argmin(dist_mat, axis=1)     # (Nedge,)

    # Reluctivity at closest element
    nu = fem["nu_e"][closest_e]   # (Nedge,)

    # Magnetic flux density at closest element
    Bx = fem["Bx"][closest_e]   # (Nedge,)
    By = fem["By"][closest_e]

    nx = normal[:, 0]   # (Nedge,)
    ny = normal[:, 1]

    # MST using reluctivity nu:
    # Eq. (7) in component form for 2D (in-plane B only):
    #   T_xx = nu * 0.5 * (B_x^2 - B_y^2)
    #   T_xy = nu * B_x * B_y
    #   T_yy = nu * 0.5 * (B_y^2 - B_x^2)
    Txx = nu * 0.5 * (Bx**2 - By**2)
    Txy = nu * Bx * By
    Tyy = nu * 0.5 * (By**2 - Bx**2)

    # Eq. (9): per-edge contribution dF = T n ds
    dFx = (Txx * nx + Txy * ny) * ds   # (Nedge,)
    dFy = (Txy * nx + Tyy * ny) * ds

    dFx = np.where(valid, dFx, 0.0)
    dFy = np.where(valid, dFy, 0.0)

    Fx_total = float(np.sum(dFx))
    Fy_total = float(np.sum(dFy))

    fem["mst"] = {
        "path":  {"midpoints": mid},
        "force": {"dF": np.column_stack([dFx, dFy])}
    }
    fem["Fx_total"] = Fx_total
    fem["Fy_total"] = Fy_total

    print("Magnetic force by MST computation Done. ✅")
    return Fx_total, Fy_total, fem
