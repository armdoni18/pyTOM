"""
F6_Main_NR_Jacobian.py
======================

Consistent tangent (Jacobian) matrix assembly for Newton-Raphson
iteration of the nonlinear magnetostatic problem.

Theory link
-----------
The residual of Eq. (4) is

    R(A) = K(nu(B(A))) * A - (f + f_pm)

and the Newton-Raphson linearization (Eq. (5)) requires its
Jacobian

    K_t = d R / d A
        = K(nu) + (d K / d A) * A

A naive linearization would treat nu as constant and use only the
first term S_e = nu_e * K0_e on each element. Because nu depends
on the field through the Brauer model (Eq. (3)), the residual has
an additional implicit dependence on A via |B(A)|. The
corresponding contribution, ``J_extra`` below, is essential to
recover quadratic NR convergence in deeply saturated regions.

For each element this contribution can be written as a rank-one
update

    J_extra = A_e * |B| * (dnu/dB) * (g o g)

with the per-element row vector

    g = (B_x * alpha + B_y * beta) / |B|

where (alpha, beta) package the per-node shape-function gradient
coefficients divided by 2*A_e. The full element Jacobian is then
J_e = S_e + J_extra, and the global K_t is assembled with the
same COO triplet pattern as in ``F4_Main_Solve_VecPot.py``.

Numerical safeguards
--------------------
- A small ``epsB`` is added inside the square root that defines
  ``B_safe`` to prevent 0/0 in g for elements that happen to have
  exactly zero field (e.g. air interior in the first NR iterate).
- The scalar coefficient is suppressed wherever |B| = 0 or
  dnu/dB = 0 (air, coil, PM regions), so the rank-one term
  contributes nothing in those domains.
- The assembled J is symmetrized to compensate for floating-point
  asymmetries introduced by the COO accumulation order.
"""

import numpy as np
import scipy.sparse as sp

def F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e):
    """
    Assemble the consistent tangent matrix of Eq. (5).

    Parameters
    ----------
    fem : dict
        Finite-element data. Must contain ``ndof``, ``ne``, ``IX``,
        ``X``, ``Ae``, ``Bx``, ``By``, ``B`` (the latter three set
        by ``F5_Main_Comp_Flux`` immediately before this call) and
        ``S_S`` (the pre-computed element kernel from F2).
    nu_e_all : ndarray, shape (ne,)
        Element-wise reluctivity nu at the current iterate.
    dnu_dB_e : ndarray, shape (ne,)
        Element-wise derivative dnu/dB at the current iterate.
        Zero in regions where nu is field-independent (air, coil,
        PM); nonzero in iron domains via the Brauer model.

    Returns
    -------
    fem : dict
        Updated in place with:
            ``S`` : global stiffness K(nu)  (csc, ndof x ndof)
            ``J`` : global Jacobian K_t     (csc, ndof x ndof)
    J : csc_matrix
        The Jacobian K_t, also returned directly so the main
        driver can pass it to ``F8_Main_Comp_Sens`` for the
        adjoint solve.
    """

# Read problem sizes and current-iterate material arrays
    ndof = int(fem["ndof"])                         # total number of DOFs
    ne   = int(fem["ne"])                           # number of elements
    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)   # nu per element
    dnu_dB_e = np.asarray(dnu_dB_e, dtype=float).reshape(-1)   # dnu/dB per element

    IX = fem["IX"]                                  # element connectivity (ne, 4): nodes + domain id
    X  = fem["X"]                                   # nodal coordinates (nn, 2)
    Ve = fem["Ae"]                                  # element areas (ne,)
    Bx = fem["Bx"]                                  # B_x per element (from F5)
    By = fem["By"]                                  # B_y per element (from F5)
    B  = fem["B"]                                   # |B| per element  (from F5)

# Shape-function gradient coefficients per element  (b_i, c_i)
    # Identical coefficients to those used in F2 and F5; recomputed
    # here so the Jacobian module is self-contained.
    i = IX[:, 0] - 1                                # 1st node index (0-based)
    j = IX[:, 1] - 1                                # 2nd node index
    k = IX[:, 2] - 1                                # 3rd node index
    xi = X[i, 0]; yi = X[i, 1]                      # coords of node i
    xj = X[j, 0]; yj = X[j, 1]                      # coords of node j
    xk = X[k, 0]; yk = X[k, 1]                      # coords of node k
    bi = yj - yk;  ci = xk - xj                     # gradient coeffs of node i
    bj = yk - yi;  cj = xi - xk                     # gradient coeffs of node j
    bk = yi - yj;  ck = xj - xi                     # gradient coeffs of node k
    inv2A = 1.0 / (2.0 * Ve)                        # 1/(2*A_e) per element

    # alpha = c_i/(2A_e) = d B_x / d A_node ;  beta = -b_i/(2A_e) = d B_y / d A_node
    alpha = np.column_stack([ci, cj, ck]) * inv2A[:, None]      # (ne, 3)
    beta  = np.column_stack([-bi, -bj, -bk]) * inv2A[:, None]   # (ne, 3)

# Material part of the tangent:  S_e = nu_e * K0_e
    K0  = fem["S_S"].reshape(ne, 3, 3)              # pre-computed kernel (ne, 3, 3)
    S_e = nu_e_all[:, None, None] * K0              # linear stiffness block per element

# Nonlinearity part:  J_extra = coeff * (g o g)   -- the saturation term
    epsB   = 1e-24                                  # tiny floor: avoids 0/0 in g when |B|=0
    B_safe = np.sqrt(Bx**2 + By**2 + epsB)          # regularized |B| (ne,)
    g      = (Bx[:, None] * alpha + By[:, None] * beta) / B_safe[:, None]  # d|B|/dA_node (ne,3)

    coeff = Ve * B * dnu_dB_e                        # scalar prefactor A_e*|B|*dnu/dB (ne,)
    # Switch the saturation term off in field-free or linear regions
    # (air/coil/PM): there |B|=0 or dnu/dB=0, so it must not contribute.
    coeff = np.where((np.abs(B) > 0.0) & (dnu_dB_e != 0.0), coeff, 0.0)

    J_extra = coeff[:, None, None] * np.einsum('ei,ej->eij', g, g)   # rank-one update (ne,3,3)
    J_e     = S_e + J_extra                         # full element Jacobian block

# COO triplet assembly: scatter the 9 entries of each 3x3 block
    nodes0 = np.column_stack([i, j, k])             # element node table (ne, 3), 0-based
    rows = np.repeat(nodes0, 3, axis=1).reshape(-1) # row indices of the 9 block entries
    cols = np.tile(nodes0, (1, 3)).reshape(-1)      # col indices of the 9 block entries
    data_S = S_e.reshape(-1)                        # flattened linear blocks
    data_J = J_e.reshape(-1)                        # flattened full Jacobian blocks

    S = sp.coo_matrix((data_S, (rows, cols)), shape=(ndof, ndof)).tocsc()  # K(nu)
    J = sp.coo_matrix((data_J, (rows, cols)), shape=(ndof, ndof)).tocsc()  # K_t

# Symmetrize to remove round-off asymmetry from the accumulation order
    S = (S + S.T) * 0.5
    J = (J + J.T) * 0.5

    fem["S"] = S                                    # store K(nu) (reused for the residual)
    fem["J"] = J                                    # store K_t   (reused by the adjoint solve)
    return fem, J
