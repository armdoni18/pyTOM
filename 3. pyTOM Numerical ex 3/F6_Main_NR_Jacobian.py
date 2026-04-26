import numpy as np
import scipy.sparse as sp

def F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e):

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    dnu_dB_e = np.asarray(dnu_dB_e, dtype=float).reshape(-1)

    IX = fem["IX"]
    X  = fem["X"]
    Ve = fem["Ae"]    # (ne,)

    Bx = fem["Bx"]   # (ne,)
    By = fem["By"]
    B  = fem["B"]

    # -------------------------------------------------------
    # Shape function gradient coefficients  (ne, 3)
    # -------------------------------------------------------
    i = IX[:, 0] - 1
    j = IX[:, 1] - 1
    k = IX[:, 2] - 1

    xi = X[i, 0]; yi = X[i, 1]
    xj = X[j, 0]; yj = X[j, 1]
    xk = X[k, 0]; yk = X[k, 1]

    bi = yj - yk;  ci = xk - xj
    bj = yk - yi;  cj = xi - xk
    bk = yi - yj;  ck = xj - xi

    inv2A = 1.0 / (2.0 * Ve)   # (ne,)

    # alpha = ci/(2A),  beta = -bi/(2A)   shape (ne,3)
    alpha = np.column_stack([ci, cj, ck]) * inv2A[:, None]   # dBx/dAnode
    beta  = np.column_stack([-bi, -bj, -bk]) * inv2A[:, None]  # dBy/dAnode

    K0 = fem["S_S"].reshape(ne, 3, 3)   # (ne, 3, 3)
    S_e = nu_e_all[:, None, None] * K0  # (ne, 3, 3)

    epsB  = 1e-24
    B_safe = np.sqrt(Bx**2 + By**2 + epsB)   # (ne,)

    g = (Bx[:, None] * alpha + By[:, None] * beta) / B_safe[:, None]  # (ne, 3)

    # scalar coefficient per element
    coeff = Ve * B * dnu_dB_e   # (ne,)
    coeff = np.where((np.abs(B) > 0.0) & (dnu_dB_e != 0.0), coeff, 0.0)

    J_extra = coeff[:, None, None] * np.einsum('ei,ej->eij', g, g)  # (ne,3,3)
    J_e     = S_e + J_extra   # (ne, 3, 3)

    # -------------------------------------------------------
    # Assemble triplets  (ne*9 entries)
    # -------------------------------------------------------
    nodes0 = np.column_stack([i, j, k])   # (ne, 3)

    # row and col indices for 3x3 block
    rows = np.repeat(nodes0, 3, axis=1).reshape(-1)   # (9*ne,)
    cols = np.tile(nodes0, (1, 3)).reshape(-1)         # (9*ne,)

    data_S = S_e.reshape(-1)
    data_J = J_e.reshape(-1)

    S = sp.coo_matrix((data_S, (rows, cols)), shape=(ndof, ndof)).tocsc()
    J = sp.coo_matrix((data_J, (rows, cols)), shape=(ndof, ndof)).tocsc()

    # Symmetrize
    S = (S + S.T) * 0.5
    J = (J + J.T) * 0.5

    fem["S"] = S
    fem["J"] = J
    return fem, J
