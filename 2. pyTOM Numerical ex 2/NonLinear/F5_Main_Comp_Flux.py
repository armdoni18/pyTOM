"""
F5_Main_Comp_Flux.py
====================

Element-wise computation of the magnetic flux density B = curl A
from the nodal magnetic vector potential.

Theory link
-----------
The relation B = curl A underlies the field-dependent reluctivity
in Eq. (3) and the Maxwell stress tensor in Eq. (7). In 2D
magnetostatics the vector potential A has only a z-component A_z,
so the flux density has only in-plane components:

    B_x =  d A_z / d y
    B_y = - d A_z / d x

For linear triangular elements, the gradients are piecewise
constant and are computed in closed form from the nodal
coordinates, element area, and shape-function gradient
coefficients (b_i, c_i) defined in ``F2_Pre_FEM_Init.py``.

Usage
-----
Called inside the Newton-Raphson loop (in the main driver script)
to evaluate B from the previous iterate so that the reluctivity
nu(|B|) can be updated. It is also called after field convergence
to provide B for force evaluation (F7) and the sensitivity
computation (F8).
"""

import numpy as np

def F5_Main_Comp_Flux(fem):
    """Compute the element-wise magnetic flux density from nodal A.

    Parameters
    ----------
    fem : dict
        Finite-element data structure. Must contain:
            ``IX`` : (ne, >=4) element connectivity
            ``X``  : (nn, 2) nodal coordinates
            ``A``  : (ndof,) current vector potential
            ``Ae`` : (ne,) element areas
            ``ne`` : int number of elements

    Returns
    -------
    fem : dict
        Updated in place with:
            ``Bx`` : (ne,) x-component of B per element
            ``By`` : (ne,) y-component of B per element
            ``B``  : (ne,) magnitude |B| per element
    """

    # =====================================================================
    # UNPACK FEM DATA
    # =====================================================================
    IX = fem["IX"]                       # element connectivity (ne, >=4)
    X  = fem["X"]                        # nodal coordinates (nn, 2)
    A  = fem["A"]                        # current nodal vector potential (ndof,)
    Ae = fem["Ae"]                       # element areas (ne,)
    ne = fem["ne"]                       # number of elements

    # =====================================================================
    # ELEMENT GEOMETRY
    # =====================================================================
    # Element node indices (Gmsh 1-based -> Python 0-based).
    i = IX[:, 0] - 1                     # 1st node of each triangle
    j = IX[:, 1] - 1                     # 2nd node
    k = IX[:, 2] - 1                     # 3rd node

    # Nodal coordinates of the three triangle vertices.
    xi = X[i, 0]; yi = X[i, 1]           # vertex i
    xj = X[j, 0]; yj = X[j, 1]           # vertex j
    xk = X[k, 0]; yk = X[k, 1]           # vertex k

    # Shape-function gradient coefficients, consistent with F2.
    bi = yj - yk;  ci = xk - xj          # node i
    bj = yk - yi;  cj = xi - xk          # node j
    bk = yi - yj;  ck = xj - xi          # node k

    # =====================================================================
    # FLUX DENSITY COMPUTATION
    # =====================================================================
    Ai = A[i]; Aj = A[j]; Ak = A[k]
    inv2A = 1.0 / (2.0 * Ae)             # 1/(2 A_e) per element

    # B = curl A for A = A_z e_z:
    #   B_x =  dA_z/dy
    #   B_y = -dA_z/dx
    Bx = inv2A * (ci * Ai + cj * Aj + ck * Ak)
    By = inv2A * (-bi * Ai - bj * Aj - bk * Ak)
    B  = np.sqrt(Bx**2 + By**2)          # field magnitude |B|

    fem["Bx"] = Bx                       # store for F6 / F7 / F8
    fem["By"] = By
    fem["B"]  = B

    print("Magnetic flux density computation Done. ✅")
    return fem
