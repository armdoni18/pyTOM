import numpy as np
from scipy.spatial.distance import cdist


def F7_Main_Comp_Force(fem):

    IX = fem["IX"]
    X  = fem["X"]
    ne = int(fem["ne"])

    # ============================================================
    # STEP 1: Rotor (domain 5) boundary edges
    # ============================================================
    rot_mask  = (IX[:, 3] == 5)
    rot_elems = IX[rot_mask, 0:3]

    if rot_elems.shape[0] == 0:
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    rot_edges = np.vstack([
        rot_elems[:, [0, 1]],
        rot_elems[:, [1, 2]],
        rot_elems[:, [2, 0]],
    ])
    rot_edges = np.sort(rot_edges, axis=1)
    unique_edges, counts = np.unique(rot_edges, axis=0, return_counts=True)
    rot_boundary_edges = unique_edges[counts == 1]
    fem["plunger_boundary_edges"] = rot_boundary_edges

    # ============================================================
    # STEP 2: Air triangles (domain 1) touching the rotor boundary
    # ============================================================
    air_mask  = (IX[:, 3] == 1)
    air_elems = IX[air_mask, 0:3]

    rot_nodes_set = np.unique(rot_boundary_edges)

    air_has_rot = np.any(np.isin(air_elems, rot_nodes_set), axis=1)
    connected_air = air_elems[air_has_rot]

    if connected_air.shape[0] == 0:
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    # ============================================================
    # STEP 3: Outer air loop minus the rotor surface itself
    # ============================================================
    air_edges = np.vstack([
        connected_air[:, [0, 1]],
        connected_air[:, [1, 2]],
        connected_air[:, [2, 0]],
    ])
    air_edges = np.sort(air_edges, axis=1)
    uniq_aedges, cnt_air = np.unique(air_edges, axis=0, return_counts=True)
    outer_air_loop = uniq_aedges[cnt_air == 1]

    rot_set = set(map(tuple, np.sort(rot_boundary_edges, axis=1).tolist()))
    air_clean = np.array([
        e for e in outer_air_loop if tuple(e.tolist()) not in rot_set
    ], dtype=int)

    if air_clean.shape[0] == 0:
        fem["Fx_total"] = 0.0
        fem["Fy_total"] = 0.0
        return 0.0, 0.0, fem

    fem["cleaned_air_loop_around_plunger"] = air_clean

    # Rotor centroid (used for outward-normal orientation)
    rotor_nodes_unique = np.unique(IX[rot_mask, 0:3])
    rotor_center       = np.mean(X[rotor_nodes_unique - 1, :], axis=0)

    # ============================================================
    # STEP 4: Element centroids
    # ============================================================
    nodes012  = IX[:, 0:3] - 1
    centroids = np.mean(X[nodes012, :], axis=1)   # (ne, 2)

    # ============================================================
    # STEP 5: force integration
    # ============================================================
    n1 = air_clean[:, 0] - 1
    n2 = air_clean[:, 1] - 1

    p1 = X[n1, :]
    p2 = X[n2, :]

    seg = p2 - p1
    ds  = np.linalg.norm(seg, axis=1)
    valid = ds > 0.0

    # raw outward normal
    normal_raw = np.column_stack([seg[:, 1], -seg[:, 0]])
    norm_mag   = np.linalg.norm(normal_raw, axis=1, keepdims=True) + 1e-30
    normal     = normal_raw / norm_mag

    mid = 0.5 * (p1 + p2)

    # flip normals that point toward the rotor
    to_rot   = rotor_center[None, :] - mid
    dot_sign = np.sum(normal * to_rot, axis=1)
    flip     = dot_sign > 0.0
    normal[flip, :] *= -1.0

    # shifted point: into air (opposite of rotor)
    eps_shift   = 1e-3
    shifted_mid = mid + eps_shift * normal

    # nearest element to integration point
    dist_mat  = cdist(shifted_mid, centroids)
    closest_e = np.argmin(dist_mat, axis=1)

    nu = fem["nu_e"][closest_e]
    Bx = fem["Bx"][closest_e]
    By = fem["By"][closest_e]

    nx = normal[:, 0]
    ny = normal[:, 1]

    # MST in reluctivity form
    Txx = nu * 0.5 * (Bx ** 2 - By ** 2)
    Txy = nu * Bx * By
    Tyy = nu * 0.5 * (By ** 2 - Bx ** 2)

    dFx = (Txx * nx + Txy * ny) * ds
    dFy = (Txy * nx + Tyy * ny) * ds

    dFx = np.where(valid, dFx, 0.0)
    dFy = np.where(valid, dFy, 0.0)

    Fx_total = float(np.sum(dFx))
    Fy_total = float(np.sum(dFy))

    fem["mst"] = {
        "path":  {"midpoints": mid},
        "force": {"dF": np.column_stack([dFx, dFy])},
    }
    fem["Fx_total"] = Fx_total
    fem["Fy_total"] = Fy_total

    print("Magnetic force by MST computation Done. ✅")
    return Fx_total, Fy_total, fem
