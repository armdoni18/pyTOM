import numpy as np
import scipy.sparse as sp

def F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e):

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    dnu_dB_e = np.asarray(dnu_dB_e, dtype=float).reshape(-1)
    if nu_e_all.size != ne or dnu_dB_e.size != ne:
        raise ValueError("nu_e_all and dnu_dB_e must have length fem['ne']")

    IX = fem["IX"]  # expects first 3 cols are node ids (1-based), 4th col domain id
    X  = fem["X"]

    # area per element: in your Func3 you stored fem["Ae"]; in MATLAB code it's fem.Ve
    # Use fem["Ae"] if present; else try fem["Ve"]
    if "Ae" in fem:
        Ve = np.asarray(fem["Ae"], dtype=float).reshape(-1)
    elif "Ve" in fem:
        Ve = np.asarray(fem["Ve"], dtype=float).reshape(-1)
    else:
        raise KeyError("Need fem['Ae'] or fem['Ve'] as element area")

    Bx = np.asarray(fem["Bx"], dtype=float).reshape(-1)
    By = np.asarray(fem["By"], dtype=float).reshape(-1)
    B  = np.asarray(fem["B"],  dtype=float).reshape(-1)

    # We'll assemble triplets for S and J
    rows_S = np.zeros(ne * 9, dtype=int)
    cols_S = np.zeros(ne * 9, dtype=int)
    data_S = np.zeros(ne * 9)

    rows_J = np.zeros(ne * 9, dtype=int)
    cols_J = np.zeros(ne * 9, dtype=int)
    data_J = np.zeros(ne * 9)

    counter = 0

    epsB = 1e-24

    for e in range(ne):
        # node ids (1-based in IX)
        i = int(IX[e, 0])
        j = int(IX[e, 1])
        k = int(IX[e, 2])
        nodes0 = np.array([i-1, j-1, k-1], dtype=int)  # 0-based for python matrices

        xi, yi = X[i-1, 0], X[i-1, 1]
        xj, yj = X[j-1, 0], X[j-1, 1]
        xk, yk = X[k-1, 0], X[k-1, 1]

        bi = yj - yk;  ci = xk - xj
        bj = yk - yi;  cj = xi - xk
        bk = yi - yj;  ck = xj - xi

        Ve_e = Ve[e]
        if Ve_e <= 0:
            raise ValueError(f"Non-positive element area at element {e}")

        # alpha for Bx, beta for By as in MATLAB
        alpha_i =  ci / (2.0 * Ve_e)
        alpha_j =  cj / (2.0 * Ve_e)
        alpha_k =  ck / (2.0 * Ve_e)

        beta_i  = -bi / (2.0 * Ve_e)
        beta_j  = -bj / (2.0 * Ve_e)
        beta_k  = -bk / (2.0 * Ve_e)

        # Base element matrix K0_e from fem["S_S"] (9 entries per element)
        idx0 = e * 9
        K0_e = np.asarray(fem["S_S"][idx0:idx0+9], dtype=float).reshape(3, 3)

        nu_e = float(nu_e_all[e])
        if nu_e <= 0:
            raise ValueError(f"nu_e <= 0 at element {e}")

        dnu_dB = float(dnu_dB_e[e])

        # stiffness element
        S_e = nu_e * K0_e

        # current B values
        Bx_e = float(Bx[e])
        By_e = float(By[e])
        B_e  = float(B[e])

        B_safe = np.sqrt(Bx_e*Bx_e + By_e*By_e + epsB)

        # grad B wrt Ai Aj Ak
        dB_dAi = (Bx_e * alpha_i + By_e * beta_i) / B_safe
        dB_dAj = (Bx_e * alpha_j + By_e * beta_j) / B_safe
        dB_dAk = (Bx_e * alpha_k + By_e * beta_k) / B_safe

        g = np.array([dB_dAi, dB_dAj, dB_dAk], dtype=float).reshape(3, 1)

        # Additional Jacobian term
        if (abs(B_e) > 0.0) and (dnu_dB != 0.0):
            J_extra = Ve_e * B_e * dnu_dB * (g @ g.T)
        else:
            J_extra = np.zeros((3, 3), dtype=float)

        J_e = S_e + J_extra

        # assemble S and J triplets
        for a in range(3):
            ra = nodes0[a]
            for b in range(3):
                cb = nodes0[b]

                rows_S[counter] = ra
                cols_S[counter] = cb
                data_S[counter] = S_e[a, b]

                rows_J[counter] = ra
                cols_J[counter] = cb
                data_J[counter] = J_e[a, b]

                counter += 1

    # Build sparse matrices
    S = sp.coo_matrix((data_S, (rows_S, cols_S)), shape=(ndof, ndof)).tocsc()
    J = sp.coo_matrix((data_J, (rows_J, cols_J)), shape=(ndof, ndof)).tocsc()

    # Symmetrize (like MATLAB)
    S = (S + S.T) * 0.5
    J = (J + J.T) * 0.5

    fem["S"] = S
    fem["J"] = J
    return fem, J
