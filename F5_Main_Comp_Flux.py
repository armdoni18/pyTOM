import numpy as np

def F5_Main_Comp_Flux(fem):
    ne = fem["ne"]

    Bx    = np.zeros(ne, dtype=float)
    By    = np.zeros(ne, dtype=float)
    B     = np.zeros(ne, dtype=float)
    MagEn = np.zeros(ne, dtype=float)

    IX = fem["IX"]
    X  = fem["X"]
    A  = fem["A"]
    Ae = fem["Ae"]

    for e in range(ne):

        # --- Node indices ---
        i = int(IX[e, 0]) - 1
        j = int(IX[e, 1]) - 1
        k = int(IX[e, 2]) - 1

        # --- Coordinates ---
        xi, yi = X[i, 0], X[i, 1]
        xj, yj = X[j, 0], X[j, 1]
        xk, yk = X[k, 0], X[k, 1]

        # --- b_i, c_i terms ---
        bi = yj - yk
        ci = xk - xj

        bj = yk - yi
        cj = xi - xk

        bk = yi - yj
        ck = xj - xi

        # --- Magnetic flux density ---
        Bx[e] = (1.0 / (2.0 * Ae[e])) * (ci * A[i] + cj * A[j] + ck * A[k])
        By[e] = (-1.0 / (2.0 * Ae[e])) * (bi * A[i] + bj * A[j] + bk * A[k])

        B[e] = np.sqrt(Bx[e]**2 + By[e]**2)

        # --- Magnetic energy ---
        nu_e = fem["nu_e"][e]
        MagEn[e] = 0.5 * nu_e * (B[e]**2) * Ae[e]

    # --- Store into fem ---
    fem["Bx"] = Bx
    fem["By"] = By
    fem["B"]  = B

    fem["MagEn"]      = MagEn
    fem["TotalMagEn"] = np.sum(MagEn)

    print("Magnetic flux density computation Done. ✅")
    return fem


