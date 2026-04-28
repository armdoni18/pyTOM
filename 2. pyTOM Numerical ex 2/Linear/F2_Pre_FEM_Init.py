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
    ne  = fem["ne"]

    # 0-based node indices for vectorized indexing
    n0 = IX[:, 0] - 1   # (ne,)
    n1 = IX[:, 1] - 1
    n2 = IX[:, 2] - 1

    x0 = X[n0, 0]; y0 = X[n0, 1]
    x1 = X[n1, 0]; y1 = X[n1, 1]
    x2 = X[n2, 0]; y2 = X[n2, 1]

    fem["nx"] = np.column_stack([x0, x1, x2])   # (ne, 3)
    fem["ny"] = np.column_stack([y0, y1, y2])   # (ne, 3)

    # -------------------------------------------------------
    # Element area  Ae = 0.5 * |det([1 x0 y0; 1 x1 y1; 1 x2 y2])|
    # -------------------------------------------------------
    Ae = 0.5 * np.abs(
        (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    )
    fem["Ae"] = Ae   # (ne,)

    # -------------------------------------------------------
    # Shape function gradients  b_i, c_i  (ne, 3)
    # -------------------------------------------------------
    # b0 = y1-y2, b1 = y2-y0, b2 = y0-y1
    # c0 = x2-x1, c1 = x0-x2, c2 = x1-x0
    b = np.column_stack([y1 - y2, y2 - y0, y0 - y1])   # (ne, 3)
    c = np.column_stack([x2 - x1, x0 - x2, x1 - x0])   # (ne, 3)

    inv2A = 1.0 / (2.0 * Ae)   # (ne,)

    # Outer products: (ne, 3, 3)
    Se_all = (
        np.einsum('ei,ej->eij', c, c) + np.einsum('ei,ej->eij', b, b)
    ) * (inv2A * 0.25)[:, None, None]   # multiply by 1/(4A)

    fem["S_S"] = Se_all.reshape(-1)   # (9*ne,)

    # -------------------------------------------------------
    # Boundary conditions  (Example 2 bounding box: [0, 390] x [0, 250])
    # -------------------------------------------------------
    tol = 1e-9
    X_   = np.asarray(mesh["X"], dtype=float)
    if X_.ndim == 1:
        X_ = X_.reshape(-1, 2)

    left0   = np.where(np.abs(X_[:, 0] - 0.0)   < tol)[0]
    right0  = np.where(np.abs(X_[:, 0] - 390.0) < tol)[0]
    top0    = np.where(np.abs(X_[:, 1] - 0.0)   < tol)[0]
    bottom0 = np.where(np.abs(X_[:, 1] - 250.0) < tol)[0]

    bcdof0      = np.unique(np.concatenate([left0, right0, top0, bottom0])).astype(int)
    fem["bcdof"] = bcdof0 + 1             # store as 1-based
    fem["bcval"] = np.zeros(bcdof0.size, dtype=float)

    # -------------------------------------------------------
    # Coil excitation (current source vector)
    # -------------------------------------------------------
    J_val = float(inputs["J_am2"])

    Tdof_list = []
    Tval_list = []

    for coil_dom, sign in [(3, +1.0), (4, -1.0)]:
        mask   = (IX[:, 3] == coil_dom)
        coil_e = IX[mask, :]                          # (nc, >=4)
        key    = "ncoil1" if coil_dom == 3 else "ncoil2"

        if coil_e.shape[0] == 0:
            fem[key] = np.zeros((0, 3), dtype=int)
            mesh[key] = fem[key]
            c_A_key = "c_A_all_coil1" if coil_dom == 3 else "c_A_all_coil2"
            fem[c_A_key] = np.zeros(0, dtype=float)
            continue

        nodes_c = coil_e[:, 0:3].astype(int)   # (nc, 3), 1-based
        fem[key]  = nodes_c
        mesh[key] = nodes_c

        # Vectorized area computation
        na = nodes_c[:, 0] - 1
        nb = nodes_c[:, 1] - 1
        nc_ = nodes_c[:, 2] - 1

        v1 = X[na, :]
        v2 = X[nb, :]
        v3 = X[nc_, :]

        c_A = 0.5 * np.abs(
            (v1[:, 0] - v3[:, 0]) * (v2[:, 1] - v3[:, 1]) -
            (v1[:, 1] - v3[:, 1]) * (v2[:, 0] - v3[:, 0])
        )

        c_A_key = "c_A_all_coil1" if coil_dom == 3 else "c_A_all_coil2"
        fem[c_A_key] = c_A

        val = sign * (c_A / 3.0) * J_val   # (nc,)

        # Each element contributes val to its 3 nodes
        Tdof_list.append(nodes_c[:, 0])
        Tdof_list.append(nodes_c[:, 1])
        Tdof_list.append(nodes_c[:, 2])
        Tval_list.append(val)
        Tval_list.append(val)
        Tval_list.append(val)

    fem["Tdof"] = np.concatenate(Tdof_list).astype(int)
    fem["Tval"] = np.concatenate(Tval_list).astype(float)

    print("FEM Initialization Done ✅")
    return fem
