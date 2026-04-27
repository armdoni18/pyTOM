import numpy as np

def F2_Pre_FEM_Init(inputs, mesh):

    fem = {}
    fem["IX"]   = mesh["IX"]
    fem["X"]    = mesh["X"]
    fem["nn"]   = fem["X"].shape[0]
    fem["ndof"] = fem["nn"]
    fem["ne"]   = fem["IX"].shape[0]
    fem["edof"] = np.array([fem["IX"][:, 0],
                            fem["IX"][:, 1],
                            fem["IX"][:, 2]], dtype=int).T   # (ne, 3)

    fem["is"] = np.reshape(np.kron(fem["edof"], np.ones((1, 3), dtype=int)), 9 * fem["ne"])
    fem["js"] = np.reshape(np.kron(fem["edof"], np.ones((3, 1), dtype=int)), 9 * fem["ne"])
    fem["D"]  = np.eye(2, dtype=float)

    # -------------------------------------------------------
    # Node coordinates per element  (ne, 3)
    # -------------------------------------------------------
    IX  = fem["IX"]
    X   = fem["X"]

    n0 = IX[:, 0] - 1
    n1 = IX[:, 1] - 1
    n2 = IX[:, 2] - 1

    x0 = X[n0, 0]; y0 = X[n0, 1]
    x1 = X[n1, 0]; y1 = X[n1, 1]
    x2 = X[n2, 0]; y2 = X[n2, 1]

    fem["nx"] = np.column_stack([x0, x1, x2])
    fem["ny"] = np.column_stack([y0, y1, y2])

    # Element area
    Ae = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    fem["Ae"] = Ae

    # Shape function gradients
    b = np.column_stack([y1 - y2, y2 - y0, y0 - y1])   # (ne, 3)
    c = np.column_stack([x2 - x1, x0 - x2, x1 - x0])   # (ne, 3)

    inv4A = 1.0 / (4.0 * Ae)

    # Element stiffness
    Se_all = (
        np.einsum('ei,ej->eij', c, c) + np.einsum('ei,ej->eij', b, b)
    ) * inv4A[:, None, None]

    fem["S_S"] = Se_all.reshape(-1)   # (9*ne,)

    # -------------------------------------------------------
    # Boundary conditions (IPM motor — circular geometry)
    # -------------------------------------------------------
    cx   = float(inputs.get("bc_cx",   0.0))
    cy   = float(inputs.get("bc_cy",   0.0))
    Rout = float(inputs.get("bc_Rout", 25.0))
    Rin  = float(inputs.get("bc_Rin",  5.0))
    tol  = float(inputs.get("bc_tol",  1e-6))

    Xn = X[:, 0]
    Yn = X[:, 1]

    r = np.sqrt((Xn - cx) ** 2 + (Yn - cy) ** 2)

    x0_ind    = np.where(np.abs(Xn - cx)   < tol)[0]
    y0_ind    = np.where(np.abs(Yn - cy)   < tol)[0]
    outer_ind = np.where(np.abs(r - Rout)  < tol)[0]
    inner_ind = np.where(np.abs(r - Rin)   < tol)[0]

    bcdof0 = np.unique(np.concatenate([x0_ind, y0_ind, outer_ind, inner_ind])).astype(int)
    fem["bcdof"] = bcdof0 + 1                     # store as 1-based
    fem["bcval"] = np.zeros(bcdof0.size, dtype=float)

    # -------------------------------------------------------
    # Coil excitation  (current source vector)
    #   Coil1 (3): +J,  Coil2 (4): -J,  Coil3 (6): +J
    # -------------------------------------------------------
    J_val = float(inputs["J_am2"])

    Tdof_list = []
    Tval_list = []

    coil_specs = [(3, +1.0, "ncoil1", "c_A_all_coil1"),
                  (4, -1.0, "ncoil2", "c_A_all_coil2"),
                  (6, +1.0, "ncoil3", "c_A_all_coil3")]

    for coil_dom, sign, key, c_A_key in coil_specs:
        mask   = (IX[:, 3] == coil_dom)
        coil_e = IX[mask, :]

        if coil_e.shape[0] == 0:
            fem[key]      = np.zeros((0, 3), dtype=int)
            mesh[key]     = fem[key]
            fem[c_A_key]  = np.zeros(0, dtype=float)
            continue

        nodes_c = coil_e[:, 0:3].astype(int)   # (nc, 3), 1-based
        fem[key]  = nodes_c
        mesh[key] = nodes_c

        # Vectorized area
        na  = nodes_c[:, 0] - 1
        nb  = nodes_c[:, 1] - 1
        nc_ = nodes_c[:, 2] - 1

        v1 = X[na, :]
        v2 = X[nb, :]
        v3 = X[nc_, :]

        c_A = 0.5 * np.abs(
            (v1[:, 0] - v3[:, 0]) * (v2[:, 1] - v3[:, 1]) -
            (v1[:, 1] - v3[:, 1]) * (v2[:, 0] - v3[:, 0])
        )

        fem[c_A_key] = c_A

        val = sign * (c_A / 3.0) * J_val   # (nc,)

        # Each element contributes val to each of its 3 nodes
        Tdof_list.append(nodes_c[:, 0])
        Tdof_list.append(nodes_c[:, 1])
        Tdof_list.append(nodes_c[:, 2])
        Tval_list.append(val)
        Tval_list.append(val)
        Tval_list.append(val)

    if Tdof_list:
        fem["Tdof"] = np.concatenate(Tdof_list).astype(int)
        fem["Tval"] = np.concatenate(Tval_list).astype(float)
    else:
        fem["Tdof"] = np.zeros(0, dtype=int)
        fem["Tval"] = np.zeros(0, dtype=float)

    print("FEM Initialization Done ✅")
    return fem
