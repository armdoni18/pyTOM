import numpy as np

def F2_Pre_FEM_Init(inputs, mesh):
    # Setting up the FEM variables
    fem = {}
    fem["IX"] = mesh["IX"]
    fem["X"]  = mesh["X"]
    fem["nn"] = fem["X"].shape[0]
    fem["ndof"] = fem["nn"]
    fem["ne"] = fem["IX"].shape[0]
    fem["edof"] = np.array([fem["IX"][:,0], fem["IX"][:,1], fem["IX"][:,2]], dtype=int).T
    fem["is"] = np.reshape(np.kron(fem["edof"],np.ones((1,3),dtype=int)),9*fem["ne"])
    fem["js"] = np.reshape(np.kron(fem["edof"],np.ones((3,1),dtype=int)),9*fem["ne"])
    fem["D"] = np.eye(2, dtype=float)

    # --- Building the solid reluctivity matrix (K_S) and area vector (Ve) ---
    nx = np.transpose(np.vstack([fem["X"][fem["IX"][:, 0] - 1, [0]],
                                 fem["X"][fem["IX"][:, 1] - 1, [0]],
                                 fem["X"][fem["IX"][:, 2] - 1, [0]]]))
    ny = np.transpose(np.vstack([fem["X"][fem["IX"][:, 0] - 1, [1]],
                                 fem["X"][fem["IX"][:, 1] - 1, [1]],
                                 fem["X"][fem["IX"][:, 2] - 1, [1]]]))
    fem["nx"] = nx
    fem["ny"] = ny

    # Initialization
    fem["S_S"]  = np.zeros((9 * fem["ne"]))
    fem["Ae"]   = np.zeros(fem["ne"])

    for e in range(fem["ne"]):
        px = np.array([nx[e, 0], nx[e, 1], nx[e, 2]]).reshape(-1,1)
        py = np.array([ny[e, 0], ny[e, 1], ny[e, 2]]).reshape(-1,1)

        A = np.abs(1./2 * np.linalg.det(np.vstack(([np.ones((1,3)),px.T,py.T]))))

        B = np.zeros((2, 3))
        for i in range(3):
            ind = [0, 1, 2]
            ind.remove(i)

            pm = [1, -1, 1]

            b_i = -pm[i] * np.linalg.det(np.vstack(([np.ones((1, 2)), py[ind].T])))
            c_i = pm[i] * np.linalg.det(np.vstack(([np.ones((1, 2)), px[ind].T])))

            B[:, i] = (1.0 / (2.0 * A)) * np.array([b_i, c_i])

        Se = np.matmul(np.matmul(B.T, fem["D"]), B) * A
        Ae = A

        fem["S_S"][e * 9:(e + 1) * 9] = Se.reshape(9)
        fem["Ae"][e] = Ae

    # Setting of FEM boundary condition index
    tol = 1e-9

    X = np.asarray(mesh["X"], dtype=float)

    if X.ndim == 1:
        if X.size % 2 != 0:
            raise ValueError(f"mesh['X'] 1D tapi ganjil: {X.size}. Harusnya 2*nn.")
        X = X.reshape(-1, 2)
    elif X.ndim == 2 and X.shape[1] != 2:
        raise ValueError(f"mesh['X'] harus (nn,2), tapi dapat {X.shape}")

    # === 0-based node indices ===
    left0   = np.where(np.abs(X[:, 0] - 0.0) < tol)[0]
    right0  = np.where(np.abs(X[:, 0] - 390.0) < tol)[0]
    top0    = np.where(np.abs(X[:, 1] - 0.0) < tol)[0]
    bottom0 = np.where(np.abs(X[:, 1] - 250.0) < tol)[0]

    bcdof0 = np.unique(np.concatenate([left0, right0, top0, bottom0])).astype(int)

    # === store as 1-based node IDs, because main code does (bcdof - 1) ===
    fem["bcdof"] = bcdof0 + 1
    fem["bcval"] = np.zeros(fem["bcdof"].size, dtype=float)

    # Setting up the current in coil-1 and coil-2 domains
    IX = mesh["IX"]          # expected shape: (ne, >=4), 1-based node ids in cols 0:3, domain id in col 3
    X  = fem["X"]            # node coordinates, shape: (nn, >=2)
    J  = float(inputs["J_am2"])

    # ------------------ Coil-1: collect elements ------------------
    ncoil1 = []
    for e in range(IX.shape[0]):
        if int(IX[e, 3]) == 3:
            ncoil1.append(IX[e, 0:3].astype(int))
    fem["ncoil1"] = np.array(ncoil1, dtype=int) if len(ncoil1) else np.zeros((0, 3), dtype=int)
    mesh["ncoil1"] = fem["ncoil1"]

    fem["c_A_all_coil1"] = np.zeros(fem["ncoil1"].shape[0], dtype=float)

    Tdof_list = []
    Tval_list = []

    # ------------------ Coil-1: build Tdof/Tval ------------------
    for j in range(fem["ncoil1"].shape[0]):
        n1, n2, n3 = fem["ncoil1"][j, :]  # still 1-based
        v1 = X[n1 - 1, 0:2]
        v2 = X[n2 - 1, 0:2]
        v3 = X[n3 - 1, 0:2]

        c_A = 0.5 * abs(np.linalg.det(np.vstack([v1 - v3, v2 - v3])))
        fem["c_A_all_coil1"][j] = c_A

        # Assign current density to each vertex
        val = (c_A / 3.0) * J
        Tdof_list.extend([n1, n2, n3])
        Tval_list.extend([val, val, val])

    # ------------------ Coil-2: collect elements ------------------
    ncoil2 = []
    for e in range(IX.shape[0]):
        if int(IX[e, 3]) == 4:
            ncoil2.append(IX[e, 0:3].astype(int))
    fem["ncoil2"] = np.array(ncoil2, dtype=int) if len(ncoil2) else np.zeros((0, 3), dtype=int)
    mesh["ncoil2"] = fem["ncoil2"]

    fem["c_A_all_coil2"] = np.zeros(fem["ncoil2"].shape[0], dtype=float)

    # ------------------ Coil-2: build Tdof/Tval ------------------
    for j in range(fem["ncoil2"].shape[0]):
        n1, n2, n3 = fem["ncoil2"][j, :]  # still 1-based
        v1 = X[n1 - 1, 0:2]
        v2 = X[n2 - 1, 0:2]
        v3 = X[n3 - 1, 0:2]

        c_A = 0.5 * abs(np.linalg.det(np.vstack([v1 - v3, v2 - v3])))
        fem["c_A_all_coil2"][j] = c_A

        # Assign current density to each vertex (negative sign for coil-2)
        val = -(c_A / 3.0) * J
        Tdof_list.extend([n1, n2, n3])
        Tval_list.extend([val, val, val])

    # ------------------ Store results ------------------
    fem["Tdof"] = np.asarray(Tdof_list, dtype=int)  # 1-based dof indices
    fem["Tval"] = np.asarray(Tval_list, dtype=float)

    print("FEM Initialization Done ✅")
    return fem
