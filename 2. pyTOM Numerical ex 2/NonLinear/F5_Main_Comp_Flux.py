import numpy as np

def F5_Main_Comp_Flux(fem):

    IX = fem["IX"]
    X  = fem["X"]
    A  = fem["A"]
    Ae = fem["Ae"]    # (ne,)
    ne = fem["ne"]

    # 0-based node indices
    i = IX[:, 0] - 1   # (ne,)
    j = IX[:, 1] - 1
    k = IX[:, 2] - 1

    # Coordinates
    xi = X[i, 0]; yi = X[i, 1]
    xj = X[j, 0]; yj = X[j, 1]
    xk = X[k, 0]; yk = X[k, 1]

    # b_i, c_i terms (shape function gradient coefficients)
    bi = yj - yk;  ci = xk - xj
    bj = yk - yi;  cj = xi - xk
    bk = yi - yj;  ck = xj - xi

    # Nodal vector potentials
    Ai = A[i]; Aj = A[j]; Ak = A[k]

    inv2A = 1.0 / (2.0 * Ae)

    # Magnetic flux density components
    Bx = inv2A * (ci * Ai + cj * Aj + ck * Ak)
    By = inv2A * (-bi * Ai - bj * Aj - bk * Ak)
    B  = np.sqrt(Bx**2 + By**2)

    fem["Bx"]       = Bx
    fem["By"]       = By
    fem["B"]        = B

    print("Magnetic flux density computation Done. ✅")
    return fem
