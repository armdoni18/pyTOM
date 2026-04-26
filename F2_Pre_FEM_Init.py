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
    X = fem["X"][:, 0]  # x-coordinates
    Y = fem["X"][:, 1]  # y-coordinates

    cx = 0.0
    cy = 0.0
    Rout = 25.0
    Rin = 5.0

    tol = 1e-9

    x0_ind = np.where(np.abs(X - cx) < tol)[0] + 1
    y0_ind = np.where(np.abs(Y - cy) < tol)[0] + 1

    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    outer_ind = np.where(np.abs(r - Rout) < tol)[0] + 1
    inner_ind = np.where(np.abs(r - Rin) < tol)[0] + 1

    fem["bcdof"] = np.unique(np.concatenate([x0_ind, y0_ind, outer_ind, inner_ind])).astype(int)
    fem["bcval"] = np.zeros(len(fem["bcdof"]), dtype=float)

    # Setting up the current in coil-1 and coil-2 domains
    IX = mesh["IX"]          # expected shape: (ne, >=4), 1-based node ids in cols 0:3, domain id in col 3
    X  = fem["X"]            # node coordinates, shape: (nn, >=2)
    J  = float(inputs["J_am2"])

    # ============================================================
    # UNIVERSAL COIL HANDLING (Coil1, Coil2, Coil3, ...)
    # ============================================================

    IX = mesh["IX"]
    X  = fem["X"]
    J  = float(inputs["J_am2"])

    Tdof_list = []
    Tval_list = []

    # mapping domain → sign
    coil_domains = {
        3: +1.0,   # Coil1
        4: -1.0,   # Coil2
        9: +1.0    # Coil3 (you can adjust sign if needed)
    }

    for dom, sign in coil_domains.items():

        # collect elements
        elems = []
        for e in range(IX.shape[0]):
            if int(IX[e, 3]) == dom:
                elems.append(IX[e, 0:3].astype(int))

        elems = np.array(elems, dtype=int) if len(elems) else np.zeros((0, 3), dtype=int)

        # store (optional, for debug / plotting)
        fem[f"ncoil_dom{dom}"] = elems

        if elems.shape[0] == 0:
            continue  # skip if domain not present

        cA_all = np.zeros(elems.shape[0])

        for j in range(elems.shape[0]):
            n1, n2, n3 = elems[j, :]

            v1 = X[n1 - 1, 0:2]
            v2 = X[n2 - 1, 0:2]
            v3 = X[n3 - 1, 0:2]

            c_A = 0.5 * abs(np.linalg.det(np.vstack([v1 - v3, v2 - v3])))
            cA_all[j] = c_A

            val = sign * (c_A / 3.0) * J

            Tdof_list.extend([n1, n2, n3])
            Tval_list.extend([val, val, val])

        fem[f"c_A_all_dom{dom}"] = cA_all

    # ------------------ FINAL ASSEMBLY ------------------
    fem["Tdof"] = np.asarray(Tdof_list, dtype=int)
    fem["Tval"] = np.asarray(Tval_list, dtype=float)

    print("FEM Initialization Done ✅")
    return fem
