"""
F7_Main_Comp_Force.py
=====================

Electromagnetic force on the plunger via Maxwell stress tensor
integration along a closed air-side path that surrounds the
plunger.

Theory link
-----------
The Maxwell stress tensor of Eq. (10),

    T = nu * (B o B - (1/2) |B|^2 I)

is integrated over a closed path Gamma (Eq. (11)) to give the
electromagnetic force on the plunger. The closed path lies
entirely in the air domain so that T is evaluated with the linear
reluctivity nu_air. Integrating through the iron region would
give an incorrect force because of the material nonlinearity
gradient.

The discrete approximation of Eq. (12),

    F approx sum_k T_k n_k Delta s_k

is implemented as a vectorized sum over the edges of the closed
air loop. The construction of that closed loop takes three steps:

  1. Identify the boundary edges of the plunger domain.
  2. Collect the air-domain triangles that share at least one
     node with the plunger boundary.
  3. Extract the outer envelope of those air triangles and remove
     the plunger-boundary edges themselves, leaving a closed loop
     on the air side of the plunger.

For each edge, the field quantities needed for T are taken from
the nearest element to a midpoint shifted slightly along the
selected normal direction. This small shift biases the nearest-
element search toward the air-side triangle, so the reluctivity
used in the stress tensor is nu_air rather than nu_iron.

The outward-normal orientation is enforced by comparing the
normal with the vector from the edge midpoint to the plunger
centroid. If the two have a positive dot product, the normal is
flipped so it points away from the plunger.
"""

import numpy as np

def F7_Main_Comp_Force(fem):
    """Evaluate the electromagnetic force on the plunger.

    Parameters
    ----------
    fem : dict
        Finite-element data with the converged field. Must
        contain ``IX``, ``X``, ``ne``, ``nu_e``, ``Bx``, ``By``.

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
            ``mst`` : integration path and per-edge force data
            ``Fx_total``, ``Fy_total``
    """

    IX = fem["IX"]
    X  = fem["X"]
    ne = int(fem["ne"])

    # =====================================================================
    # STEP 1: PLUNGER BOUNDARY EDGES (domain 5)
    # =====================================================================
    # A boundary edge of the plunger appears in exactly one plunger triangle.
    plunger_mask = (IX[:, 3] == 5)           # plunger elements
    plunger_elems = IX[plunger_mask, 0:3]    # their node triples (nPL, 3)

    pl_edges = np.vstack([                   # all three edges of every plunger tri
        plunger_elems[:, [0, 1]],
        plunger_elems[:, [1, 2]],
        plunger_elems[:, [2, 0]]
    ])

    pl_edges = np.sort(pl_edges, axis=1)     # canonical (low,high) node ordering
    unique_edges, counts = np.unique(pl_edges, axis=0, return_counts=True)

    plunger_boundary_edges = unique_edges[counts == 1]   # edges used once = boundary
    fem["plunger_boundary_edges"] = plunger_boundary_edges

    # =====================================================================
    # STEP 2: AIR TRIANGLES (domain 1) TOUCHING THE PLUNGER BOUNDARY
    # =====================================================================
    air_mask    = (IX[:, 3] == 1)            # air elements
    air_elems   = IX[air_mask, 0:3]          # their node triples (nAIR, 3)

    pl_nodes_set = set(plunger_boundary_edges.flatten().tolist())   # plunger boundary nodes

    # Keep air triangles that share at least one node with the plunger boundary.
    air_has_pl = np.any(
        np.isin(air_elems, list(pl_nodes_set)), axis=1
    )
    connected_air = air_elems[air_has_pl]

    if connected_air.shape[0] == 0:          # no surrounding air -> zero force
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    # =====================================================================
    # STEP 3: AIR-SIDE INTEGRATION LOOP
    # =====================================================================
    air_edges = np.vstack([                  # all edges of the connected air strip
        connected_air[:, [0, 1]],
        connected_air[:, [1, 2]],
        connected_air[:, [2, 0]]
    ])
    air_edges = np.sort(air_edges, axis=1)
    uniq_aedges, cnt_air = np.unique(air_edges, axis=0, return_counts=True)

    # Boundary edges of the connected air strip form its outer envelope.
    outer_air_loop = uniq_aedges[cnt_air == 1]   # outer envelope of the strip

    # Remove plunger-boundary edges, leaving a closed air-side path.
    pl_set   = set(map(tuple, np.sort(plunger_boundary_edges, axis=1).tolist()))
    air_clean = np.array([
        e for e in outer_air_loop
        if tuple(e.tolist()) not in pl_set
    ], dtype=int)

    if air_clean.shape[0] == 0:              # degenerate loop -> zero force
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    fem["cleaned_air_loop_around_plunger"] = air_clean

    # =====================================================================
    # STEP 4: GEOMETRY FOR MST INTEGRATION
    # =====================================================================
    # Plunger centroid used to orient the edge normals.
    plunger_nodes  = np.unique(IX[plunger_mask, 0:3])
    plunger_center = np.mean(X[plunger_nodes - 1, :], axis=0)

    # Element centroids used for nearest-element field lookup.
    nodes012   = IX[:, 0:3] - 1                     # (ne, 3) 0-based node indices
    centroids  = np.mean(X[nodes012, :], axis=1)    # (ne, 2) element centroids

    Nedge = air_clean.shape[0]
    n1 = air_clean[:, 0] - 1                 # edge endpoint 1 (0-based)
    n2 = air_clean[:, 1] - 1                 # edge endpoint 2

    p1 = X[n1, :]                            # endpoint coords (Nedge, 2)
    p2 = X[n2, :]

    seg = p2 - p1                            # edge vectors (Nedge, 2)
    ds  = np.linalg.norm(seg, axis=1)        # edge lengths Delta s_k
    valid = ds > 0.0                         # guard against degenerate edges

    # Edge normal = 90-degree rotation of the edge vector, then normalization.
    normal_raw = np.column_stack([ seg[:, 1], -seg[:, 0]])
    norm_mag   = np.linalg.norm(normal_raw, axis=1, keepdims=True) + 1e-30
    normal     = normal_raw / norm_mag       # unit normals (Nedge, 2)

    mid = 0.5 * (p1 + p2)                     # edge midpoints (Nedge, 2)

    # Orient every normal outward, away from the plunger centroid.
    to_pl     = plunger_center[None, :] - mid    # midpoint -> plunger vector
    dot_sign  = np.sum(normal * to_pl, axis=1)   # > 0 means normal points inward
    flip_mask = dot_sign > 0.0
    normal[flip_mask, :] *= -1.0

    # Shifted midpoints used for robust air-side field lookup.
    eps_shift   = 1e-3
    shifted_mid = mid + eps_shift * normal      # note: +normal points inward after flip

    # Nearest element to each shifted midpoint provides nu, Bx, and By.
    from scipy.spatial.distance import cdist
    dist_mat   = cdist(shifted_mid, centroids)   # (Nedge, ne) distances
    closest_e  = np.argmin(dist_mat, axis=1)     # index of nearest element per edge

    # =====================================================================
    # STEP 5: MAXWELL-STRESS INTEGRATION
    # =====================================================================
    nu = fem["nu_e"][closest_e]                 # reluctivity at the air-side element
    Bx = fem["Bx"][closest_e]                   # B_x at that element
    By = fem["By"][closest_e]                   # B_y at that element
    nx = normal[:, 0]                           # outward normal components
    ny = normal[:, 1]

    # Maxwell stress tensor components -- Eq. (10) in 2D (in-plane B only):
    #   T_xx = nu * 0.5 * (B_x^2 - B_y^2)
    #   T_xy = nu * B_x * B_y
    #   T_yy = nu * 0.5 * (B_y^2 - B_x^2)
    Txx = nu * 0.5 * (Bx**2 - By**2)
    Txy = nu * Bx * By
    Tyy = nu * 0.5 * (By**2 - Bx**2)

    # Per-edge force contribution dF = T n ds  -- Eq. (12)
    dFx = (Txx * nx + Txy * ny) * ds
    dFy = (Txy * nx + Tyy * ny) * ds
    dFx = np.where(valid, dFx, 0.0)          # zero-out degenerate edges
    dFy = np.where(valid, dFy, 0.0)

    Fx_total = float(np.sum(dFx))               # total x-force (sum over the loop)
    Fy_total = float(np.sum(dFy))               # total y-force

    fem["mst"] = {                              # path + per-edge force, for F9 plots
        "path":  {"midpoints": mid},
        "force": {"dF": np.column_stack([dFx, dFy])}
    }
    fem["Fx_total"] = Fx_total
    fem["Fy_total"] = Fy_total

    print("Magnetic force by MST computation Done. ✅")
    return Fx_total, Fy_total, fem
