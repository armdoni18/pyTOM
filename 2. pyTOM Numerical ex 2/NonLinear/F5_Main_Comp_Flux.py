"""
F5_Main_Comp_Flux.py
====================

Element-wise computation of the magnetic flux density B = curl A from the nodal magnetic vector potential.

Theory link
-----------
The relation B = curl A underlies the field-dependent reluctivity in Eq. (3) and the Maxwell stress tensor in Eq. (7).
In 2D magnetostatics the vector potential A has only a z-component A_z, so the flux density has only in-plane components:

    B_x =  d A_z / d y
    B_y = - d A_z / d x

For linear (constant-strain) triangular elements the gradients are piecewise-constant and can be computed in closed form from the nodal coordinates and the element area,
using the shape-function gradient coefficients (b_i, c_i) defined in F2_Pre_FEM_Init.

Usage
-----
Called inside the Newton-Raphson loop (in ``Main_code_Mulpos.py``) to evaluate B from the previous iterate so that the reluctivity nu(|B|) can be updated,
and again once after convergence to provide B for the force evaluation (F7) and the sensitivity computation (F8).
"""

import numpy as np

def F5_Main_Comp_Flux(fem):
    """
    Compute the element-wise magnetic flux density from nodal A.

    Parameters
    ----------
    fem : dict
        Finite-element data structure. Must contain:
            "IX"  : (ne, >=4) element connectivity (cols 0-2 are one-based node indices, col 3 is domain ID).
            "X"   : (nn, 2)  nodal coordinates.
            "A"   : (ndof,)  current vector potential.
            "Ae"  : (ne,)    element areas.
            "ne"  : int      number of elements.

    Returns
    -------
    fem : dict
        Same dict with three new entries written in place:
            "Bx" : (ne,) x-component of B per element
            "By" : (ne,) y-component of B per element
            "B"  : (ne,) magnitude |B| per element
    """
    IX = fem["IX"]
    X  = fem["X"]
    A  = fem["A"]
    Ae = fem["Ae"]    # (ne,)
    ne = fem["ne"]

    # 0-based node indices for each triangle
    i = IX[:, 0] - 1
    j = IX[:, 1] - 1
    k = IX[:, 2] - 1

    # Nodal coordinates
    xi = X[i, 0]; yi = X[i, 1]
    xj = X[j, 0]; yj = X[j, 1]
    xk = X[k, 0]; yk = X[k, 1]

    # Shape function gradient coefficients
    #   b_i = y_j - y_k    (used for  d N_i / d x )
    #   c_i = x_k - x_j    (used for  d N_i / d y )
    bi = yj - yk;  ci = xk - xj
    bj = yk - yi;  cj = xi - xk
    bk = yi - yj;  ck = xj - xi

    # Nodal vector potentials
    Ai = A[i]; Aj = A[j]; Ak = A[k]

    inv2A = 1.0 / (2.0 * Ae)

    # Eq. for B from A in 2D:
    #   B_x =  d A_z / d y  =  (c_i A_i + c_j A_j + c_k A_k) / (2 A_e)
    #   B_y = -d A_z / d x  = -(b_i A_i + b_j A_j + b_k A_k) / (2 A_e)
    Bx = inv2A * (ci * Ai + cj * Aj + ck * Ak)
    By = inv2A * (-bi * Ai - bj * Aj - bk * Ak)
    B  = np.sqrt(Bx**2 + By**2)

    fem["Bx"]       = Bx
    fem["By"]       = By
    fem["B"]        = B

    print("Magnetic flux density computation Done. ✅")
    return fem
