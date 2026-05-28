"""
F6_Main_NR_Jacobian.py
======================

Consistent tangent (Jacobian) matrix assembly for Newton-Raphson iteration of the nonlinear magnetostatic problem.

Theory link
-----------
The residual of Eq. (4) is

    R(A) = K(nu(B(A))) * A - (f + f_pm)

and the Newton-Raphson linearization (Eq. (5)) requires its Jacobian

    K_t = d R / d A
        = K(nu) + (d K / d A) * A

A naive linearization would treat nu as constant and use only the first term S_e = nu_e * K0_e on each element.
Because nu depends on the field through the Brauer model (Eq. (3)), the residual has an additional implicit dependence on A via |B(A)|.
The corresponding contribution, ``J_extra`` below, is essential to recover quadratic NR convergence in deeply saturated regions.

For each element this contribution can be written as a rank-one update

    J_extra = A_e * |B| * (dnu/dB) * (g o g)

with the per-element row vector

    g = (B_x * alpha + B_y * beta) / |B|

where (alpha, beta) package the per-node shape-function gradient coefficients divided by 2*A_e.
The full element Jacobian is then J_e = S_e + J_extra, and the global K_t is assembled with the same COO triplet pattern as in ``F4_Main_Solve_VecPot.py``.

Numerical safeguards
--------------------
- A small ``epsB`` is added inside the square root that defines ``B_safe`` to prevent 0/0 in g for elements that happen to have exactly zero field (e.g. air interior in the first NR iterate).
- The scalar coefficient is suppressed wherever |B| = 0 or dnu/dB = 0 (air, coil, PM regions), so the rank-one term contributes nothing in those domains.
- The assembled J is symmetrized to compensate for floating-point asymmetries introduced by the COO accumulation order.
"""

import numpy as np
import scipy.sparse as sp

def F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e):
    """
    Assemble the consistent tangent matrix of Eq. (5).

    Parameters
    ----------
    fem : dict
        Finite-element data. Must contain ``ndof``, ``ne``, ``IX``, ``X``, ``Ae``, ``Bx``, ``By``, ``B`` (the latter three set by ``F5_Main_Comp_Flux`` immediately before this call) and ``S_S`` (the pre-computed element kernel from F2).
    nu_e_all : ndarray, shape (ne,)
        Element-wise reluctivity nu at the current iterate.
    dnu_dB_e : ndarray, shape (ne,)
        Element-wise derivative dnu/dB at the current iterate. Zero in regions where nu is field-independent (air, coil, PM); nonzero in iron domains via the Brauer model.

    Returns
    -------
    fem : dict
        Updated in place with:
            ``S`` : global stiffness K(nu)  (csc, ndof x ndof)
            ``J`` : global Jacobian K_t     (csc, ndof x ndof)
    J : csc_matrix
        The Jacobian K_t, also returned directly so the main driver can pass it to ``F8_Main_Comp_Sens`` for the adjoint solve.
    """
    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    dnu_dB_e = np.asarray(dnu_dB_e, dtype=float).reshape(-1)

    IX = fem["IX"]
    X  = fem["X"]
    Ve = fem["Ae"]    # element areas (ne,)

    Bx = fem["Bx"]
    By = fem["By"]
    B  = fem["B"]

    # -------------------------------------------------------
    # Shape-function gradient coefficients per element  (ne, 3)
    # b_i, c_i are the same coefficients used in F2 and F5.
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

    # alpha = c_i / (2 A_e)  ->  d B_x / d A_node
    # beta  = -b_i / (2 A_e) ->  d B_y / d A_node
    alpha = np.column_stack([ci, cj, ck]) * inv2A[:, None]
    beta  = np.column_stack([-bi, -bj, -bk]) * inv2A[:, None]

    # -------------------------------------------------------
    # Material part of the tangent: S_e = nu_e * K0_e
    # K0 is the pre-computed kernel reshaped to (ne, 3, 3).
    # -------------------------------------------------------
    K0  = fem["S_S"].reshape(ne, 3, 3)
    S_e = nu_e_all[:, None, None] * K0   # (ne, 3, 3)

    # -------------------------------------------------------
    # Nonlinearity contribution: J_extra = coeff * (g o g)
    # with coeff = A_e * |B| * dnu/dB, g shape (ne, 3).
    # -------------------------------------------------------
    epsB   = 1e-24                                   # avoid 0/0
    B_safe = np.sqrt(Bx**2 + By**2 + epsB)            # (ne,)
    g      = (Bx[:, None] * alpha + By[:, None] * beta) / B_safe[:, None]

    coeff = Ve * B * dnu_dB_e                         # (ne,)
    # Suppress the rank-one update wherever the field is exactly
    # zero or dnu/dB is exactly zero (air, coil, PM elements).
    coeff = np.where((np.abs(B) > 0.0) & (dnu_dB_e != 0.0), coeff, 0.0)

    J_extra = coeff[:, None, None] * np.einsum('ei,ej->eij', g, g)
    J_e     = S_e + J_extra   # (ne, 3, 3)

    # -------------------------------------------------------
    # COO triplet assembly: 9 entries per element
    # -------------------------------------------------------
    nodes0 = np.column_stack([i, j, k])   # (ne, 3) 0-based

    # Row and col index arrays for the 3x3 element blocks
    rows = np.repeat(nodes0, 3, axis=1).reshape(-1)   # (9*ne,)
    cols = np.tile(nodes0, (1, 3)).reshape(-1)         # (9*ne,)

    data_S = S_e.reshape(-1)
    data_J = J_e.reshape(-1)

    S = sp.coo_matrix((data_S, (rows, cols)), shape=(ndof, ndof)).tocsc()
    J = sp.coo_matrix((data_J, (rows, cols)), shape=(ndof, ndof)).tocsc()

    # Symmetrize against floating-point asymmetry in accumulation
    S = (S + S.T) * 0.5
    J = (J + J.T) * 0.5

    fem["S"] = S
    fem["J"] = J
    return fem, J
