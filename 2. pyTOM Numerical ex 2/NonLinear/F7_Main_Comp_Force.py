"""
F7_Main_Comp_Force.py — Numerical Example 2 (nonlinear actuator)
==================================================================

Maxwell-stress-tensor force evaluation for the nonlinear-material actuator of Section 5.2.

Same algorithmic role as the Example 3 version (``3. pyTOM Numerical ex 3/F7_Main_Comp_Force.py``), where the full module documentation is provided.
The closed air-side integration loop ensures that the reluctivity used in the MST is nu_air (linear);
evaluating in the iron domain would give an incorrect force due to the saturation gradient.
"""

import numpy as np

def F7_Main_Comp_Force(fem):

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
    # STEP 5: Force integration
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

    # Shifted midpoints (toward plunger, i.e. inward)
    eps_shift   = 1e-3
    shifted_mid = mid + eps_shift * normal   # (Nedge, 2)

    # find argmin distance from shifted_mid to centroids
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
    Txx = nu * 0.5 * (Bx**2 - By**2)
    Txy = nu * Bx * By
    Tyy = nu * 0.5 * (By**2 - Bx**2)

    # dF = T @ n * ds
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
