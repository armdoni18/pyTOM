"""
F5_Main_Comp_Flux.py — Numerical Example 1 (IPM motor benchmark)
==================================================================

Element-wise computation of B = curl A for the IPM motor of Section 5.1.

Same algorithmic role as the Example 3 version (``3. pyTOM Numerical ex 3/F5_Main_Comp_Flux.py``), where the full module documentation is provided.
Used inside the Newton-Raphson iteration to evaluate the field-dependent reluctivity, and after convergence for visualization.
"""

import numpy as np

def F5_Main_Comp_Flux(fem):

    IX = fem["IX"]
    X  = fem["X"]
    A  = fem["A"]
    Ae = fem["Ae"]
    ne = fem["ne"]
    nn = fem["nn"]

    i = IX[:, 0] - 1
    j = IX[:, 1] - 1
    k = IX[:, 2] - 1

    xi = X[i, 0]; yi = X[i, 1]
    xj = X[j, 0]; yj = X[j, 1]
    xk = X[k, 0]; yk = X[k, 1]

    bi = yj - yk;  ci = xk - xj
    bj = yk - yi;  cj = xi - xk
    bk = yi - yj;  ck = xj - xi

    Ai = A[i]; Aj = A[j]; Ak = A[k]

    inv2A = 1.0 / (2.0 * Ae)

    Bx = inv2A * ( ci * Ai + cj * Aj + ck * Ak)
    By = inv2A * (-bi * Ai - bj * Aj - bk * Ak)
    B  = np.sqrt(Bx ** 2 + By ** 2)

    fem["Bx"] = Bx
    fem["By"] = By
    fem["B"]  = B

    # ---- Nodal averaging ----
    Bx_node = np.zeros(nn, dtype=float)
    By_node = np.zeros(nn, dtype=float)
    weight  = np.zeros(nn, dtype=float)

    nodes = IX[:, 0:3] - 1   # (ne, 3)

    # broadcast Bx*Ae over 3 local nodes
    BxAe = (Bx * Ae)[:, None] * np.ones((1, 3))
    ByAe = (By * Ae)[:, None] * np.ones((1, 3))
    AeR  = Ae[:, None] * np.ones((1, 3))

    np.add.at(Bx_node, nodes.reshape(-1), BxAe.reshape(-1))
    np.add.at(By_node, nodes.reshape(-1), ByAe.reshape(-1))
    np.add.at(weight,  nodes.reshape(-1), AeR.reshape(-1))

    Bx_node /= (weight + 1e-12)
    By_node /= (weight + 1e-12)
    B_node = np.sqrt(Bx_node ** 2 + By_node ** 2)

    fem["Bx_node"] = Bx_node
    fem["By_node"] = By_node
    fem["B_node"]  = B_node

    print("Magnetic flux density computation Done. ✅")
    return fem
