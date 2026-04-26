import numpy as np

def F7_Main_Comp_Force(fem):
    Fx_total = 0.0
    Fy_total = 0.0

    IX = fem["IX"]          # 1-based node IDs
    X  = fem["X"]           # coordinates
    ne = int(fem["ne"])

    # ============================================================
    # STEP 1: Extract plunger boundary edges (domID = 5)
    # ============================================================
    target_mat = 5

    plunger_elements = IX[IX[:, 3] == target_mat, 0:3]

    plunger_edges = np.vstack([
        plunger_elements[:, [0, 1]],
        plunger_elements[:, [1, 2]],
        plunger_elements[:, [2, 0]]
    ])

    plunger_edges = np.sort(plunger_edges, axis=1)

    unique_edges, counts = np.unique(plunger_edges, axis=0, return_counts=True)
    plunger_boundary_edges = unique_edges[counts == 1]

    fem["plunger_boundary_edges"] = plunger_boundary_edges

    # ============================================================
    # STEP 2: Find AIR triangles (domID = 1)
    # ============================================================
    air_triangles = IX[IX[:, 3] == 1, 0:3]

    # =========================================== =================
    # STEP 3: Find AIR triangles touching plunger boundary
    # ============================================================
    connected_air = []

    for tri in air_triangles:
        for edge in plunger_boundary_edges:
            if np.any(np.isin(edge, tri)):
                connected_air.append(tri)
                break

    if len(connected_air) == 0:
        fem["Fy_total"] = 0.0
        return 0.0, fem

    connected_air = np.array(connected_air)

    # ============================================================
    # STEP 4: Build outer air loop
    # ============================================================
    air_edges = np.vstack([
        connected_air[:, [0, 1]],
        connected_air[:, [1, 2]],
        connected_air[:, [2, 0]]
    ])

    air_edges = np.sort(air_edges, axis=1)
    unique_aedges, counts_air = np.unique(air_edges, axis=0, return_counts=True)
    outer_air_loop = unique_aedges[counts_air == 1]

    # remove plunger edges
    plunger_sorted = np.sort(plunger_boundary_edges, axis=1)
    air_sorted     = np.sort(outer_air_loop, axis=1)

    cleaned_air_loop = np.array([
        edge for edge in air_sorted
        if not any(np.all(edge == p_edge) for p_edge in plunger_sorted)
    ])

    fem["cleaned_air_loop_around_plunger"] = cleaned_air_loop

    IntegPathEdges = cleaned_air_loop.shape[0]

    # ============================================================
    # Compute plunger centroid
    # ============================================================
    plunger_nodes  = np.unique(IX[IX[:, 3] == target_mat, 0:3])
    plunger_coords = X[plunger_nodes - 1, :]
    plunger_center = np.mean(plunger_coords, axis=0)

    # ============================================================
    # Force integration
    # ============================================================
    for k in range(IntegPathEdges):

        n1 = int(cleaned_air_loop[k, 0])
        n2 = int(cleaned_air_loop[k, 1])

        p1 = X[n1 - 1]
        p2 = X[n2 - 1]

        seg = p2 - p1
        ds  = np.linalg.norm(seg)

        if ds == 0:
            continue

        # outward normal
        normal = np.array([seg[1], -seg[0]])
        normal = normal / np.linalg.norm(normal)

        mid = 0.5 * (p1 + p2)
        to_plunger = plunger_center - mid

        if np.dot(normal, to_plunger) > 0:
            normal = -normal

        eps_shift   = 1e-3
        shifted_mid = mid + eps_shift * normal

        # find closest element
        min_dist  = np.inf
        closest_e = 0

        for e in range(ne):
            nodes_e = IX[e, 0:3]
            coords  = X[nodes_e - 1]
            centroid = np.mean(coords, axis=0)

            dist = np.linalg.norm(shifted_mid - centroid)

            if dist < min_dist:
                min_dist  = dist
                closest_e = e

        # magnetic field
        Bx = fem["Bx"][closest_e]
        By = fem["By"][closest_e]

        # convert reluctivity → permeability
        nu = fem["nu_e"][closest_e]

        # Maxwell Stress Tensor
        T = nu * np.array([
            [0.5 * (Bx**2 - By**2),  Bx * By],
            [Bx * By,               0.5 * (By**2 - Bx**2)]
        ])

        dF = T @ normal * ds

        Fx_total += dF[0]
        Fy_total += dF[1]

    fem["Fx_total"] = Fx_total
    fem["Fy_total"] = Fy_total

    print("Magnetic force by MST computation Done. ✅")

    return Fx_total, Fy_total, fem

