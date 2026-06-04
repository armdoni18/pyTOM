"""
F6_Main_NR_Jacobian.py
======================

Consistent tangent (Jacobian) matrix assembly for Newton-Raphson
iteration of the nonlinear magnetostatic problem.

Theory link
-----------
The residual of Eq. (4) is

    R(A) = K(nu(B(A))) * A - (f + f_pm)

and the Newton-Raphson linearization of Eq. (5) requires its
Jacobian

    K_t = dR/dA
        = K(nu) + (dK/dA) * A

A linearized assembly would treat nu as constant and use only the
first term S_e = nu_e * K0_e on each element. Because nu depends
on the field through the Brauer model of Eq. (3), the residual has
an additional implicit dependence on A through |B(A)|. The
corresponding contribution, ``J_extra`` below, is included in the
consistent tangent matrix.

For each element this contribution is written as a rank-one update

    J_extra = A_e * |B| * (dnu/dB) * (g o g)

with

    g = (B_x * alpha + B_y * beta) / |B|

where alpha and beta contain the per-node derivatives of B_x and
B_y with respect to the nodal vector potential values. The full
element Jacobian is then J_e = S_e + J_extra, and the global
matrix K_t is assembled with the same COO triplet pattern used in
``F4_Main_Solve_VecPot.py``.

Numerical safeguards
--------------------
- A small ``epsB`` is added inside the square root defining
  ``B_safe`` to prevent division by zero when |B| is zero.
- The nonlinear tangent contribution is suppressed wherever
  |B| = 0 or dnu/dB = 0, so air, coil, and PM regions contribute
  only through the linear stiffness part.
- The assembled matrices are symmetrized to compensate for
  floating-point asymmetry from sparse COO accumulation.
"""

import numpy as np
import scipy.sparse as sp

def F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e):
    """Assemble the consistent tangent matrix of Eq. (5).

    Parameters
    ----------
    fem : dict
        Finite-element data. Must contain ``ndof``, ``ne``, ``IX``,
        ``X``, ``Ae``, ``Bx``, ``By``, ``B``, and ``S_S``.
        The field quantities ``Bx``, ``By``, and ``B`` are computed
        by ``F5_Main_Comp_Flux`` immediately before this call.
    nu_e_all : ndarray, shape (ne,)
        Element-wise reluctivity nu at the current iterate.
    dnu_dB_e : ndarray, shape (ne,)
        Element-wise derivative dnu/dB at the current iterate.
        Zero in field-independent regions and nonzero in nonlinear
        iron regions.

    Returns
    -------
    fem : dict
        Updated in place with:
            ``S`` : global stiffness matrix K(nu)
            ``J`` : global tangent Jacobian K_t
    J : csc_matrix
        The tangent Jacobian K_t, also returned directly for the
        adjoint solve in ``F8_Main_Comp_Sens``.
    """

    # =====================================================================
    # INPUT CHECKS AND CURRENT FIELD DATA
    # =====================================================================
    ndof = int(fem["ndof"])                                     # total number of DOFs
    ne   = int(fem["ne"])                                       # number of elements
    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)    # nu per element
    dnu_dB_e = np.asarray(dnu_dB_e, dtype=float).reshape(-1)    # dnu/dB per element

    IX = fem["IX"]                                  # element connectivity (ne, 4): nodes + domain id
    X  = fem["X"]                                   # nodal coordinates (nn, 2)
    Ve = fem["Ae"]                                  # element areas (ne,)
    Bx = fem["Bx"]                                  # B_x per element (from F5)
    By = fem["By"]                                  # B_y per element (from F5)
    B  = fem["B"]                                   # |B| per element  (from F5)

    # =====================================================================
    # SHAPE-FUNCTION GRADIENTS
    # =====================================================================
    # Element node indices (Gmsh 1-based -> Python 0-based).
    i = IX[:, 0] - 1                                # 1st node index (0-based)
    j = IX[:, 1] - 1                                # 2nd node index
    k = IX[:, 2] - 1                                # 3rd node index

    # Nodal coordinates of the three triangle vertices.
    xi = X[i, 0]; yi = X[i, 1]                      # coordinates of node i
    xj = X[j, 0]; yj = X[j, 1]                      # coordinates of node j
    xk = X[k, 0]; yk = X[k, 1]                      # coordinates of node k

    # Shape-function gradient coefficients, consistent with F2 and F5.
    bi = yj - yk;  ci = xk - xj                     # gradient coefficients of node i
    bj = yk - yi;  cj = xi - xk                     # gradient coefficients of node j
    bk = yi - yj;  ck = xj - xi                     # gradient coefficients of node k
    inv2A = 1.0 / (2.0 * Ve)                        # 1/(2*A_e) per element

    # alpha = dB_x/dA_node, beta = dB_y/dA_node.
    alpha = np.column_stack([ci, cj, ck]) * inv2A[:, None]      # (ne, 3)
    beta  = np.column_stack([-bi, -bj, -bk]) * inv2A[:, None]   # (ne, 3)

    # =====================================================================
    # LINEAR MATERIAL TANGENT
    # =====================================================================
    K0  = fem["S_S"].reshape(ne, 3, 3)              # pre-computed kernel (ne, 3, 3)
    S_e = nu_e_all[:, None, None] * K0              # linear stiffness block per element

    # =====================================================================
    # NONLINEAR MATERIAL TANGENT
    # =====================================================================
    epsB   = 1e-24                                  # tiny floor: avoids 0/0 in g when |B|=0
    B_safe = np.sqrt(Bx**2 + By**2 + epsB)          # regularized |B| (ne,)

    # d|B|/dA_node.
    g      = (Bx[:, None] * alpha + By[:, None] * beta) / B_safe[:, None]  # d|B|/dA_node (ne,3)

    coeff = Ve * B * dnu_dB_e                        # scalar prefactor A_e*|B|*dnu/dB (ne,)
    coeff = np.where((np.abs(B) > 0.0) & (dnu_dB_e != 0.0), coeff, 0.0)

    J_extra = coeff[:, None, None] * np.einsum('ei,ej->eij', g, g)   # rank-one update (ne,3,3)
    J_e     = S_e + J_extra                         # full element Jacobian block

    # =====================================================================
    # GLOBAL COO ASSEMBLY
    # =====================================================================
    nodes0 = np.column_stack([i, j, k])                      # element node table (ne, 3), 0-based
    rows = np.repeat(nodes0, 3, axis=1).reshape(-1)   # row indices of the 9 block entries
    cols = np.tile(nodes0, (1, 3)).reshape(-1)          # col indices of the 9 block entries
    data_S = S_e.reshape(-1)                                 # flattened linear blocks
    data_J = J_e.reshape(-1)                                 # flattened full Jacobian blocks

    S = sp.coo_matrix((data_S, (rows, cols)), shape=(ndof, ndof)).tocsc()  # K(nu)
    J = sp.coo_matrix((data_J, (rows, cols)), shape=(ndof, ndof)).tocsc()  # K_t

    # =====================================================================
    # SYMMETRIZATION AND STORAGE
    # =====================================================================
    S = (S + S.T) * 0.5
    J = (J + J.T) * 0.5

    fem["S"] = S                                    # store K(nu) (reused for the residual)
    fem["J"] = J                                    # store K_t   (reused by the adjoint solve)
    return fem, J
