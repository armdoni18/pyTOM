"""
F0_Main_Line_Search.py
======================

Energy-based backtracking line search for the damped Newton-Raphson
solution of the nonlinear magnetostatic problem.

Notation
--------
``Eq. (n)`` refers to the numbered equations of the manuscript.
``(En)`` refers to the auxiliary equations written out in this
docstring; they are the discrete or element-level counterparts of the
manuscript equations, and (E5) has no manuscript counterpart because it
is specific to this implementation.

Motivation and theory
---------------------
Solving the nonlinear magnetostatic system of Eq. (4),

    R(A) = K(nu(B(A))) A - (f + f_pm) = 0,

is equivalent to finding the vector potential A that minimizes the
convex magnetic energy functional

    E(A) = sum_e  Area_e * w_e(|B_e|)  -  A^T (f + f_pm),          (E1)

which is the element-assembled form of the energy functional of
Eq. (8), where the element energy density w_e is the co-energy
integral of the (monotone) B-H characteristic appearing inside Eq. (8),

    w(B) = int_0^B  nu(s) s ds.                                     (E2)

Stationarity of (E1) reproduces Eq. (2)/(4): d/dA [sum Area_e w_e]
= K(nu(B)) A, since dw/dB = nu(B) B. Because the Brauer curve of
Eq. (3) is monotone, E is convex and the Newton direction dA obtained
from Eq. (5) is a descent direction for E. A damped Newton update

    A^(k+1) = A^(k) + alpha_k dA,      alpha_k in (0, 1],       [Eq. (7)]

is therefore globally convergent when alpha_k is chosen by the
energy criterion

    alpha_k = max { 1, 1/2, 1/4, 1/8, ... }  such that
    E(A^(k) + alpha_k dA) < E(A^(k)),                               (E3)

which is the acceptance criterion of Eq. (9) in the manuscript,

and it recovers the full Newton step (alpha = 1) - and hence the
superlinear local convergence of Newton's method - as soon as the
iterate enters the region of attraction. Compared with a constant
damping factor, this both removes the non-convergence observed on
strongly saturated designs and lowers the iteration count, since a
fixed fraction of the Newton correction never exploits the fast local
convergence of Newton's method.

Closed-form energy density for the Brauer model
-----------------------------------------------
For the Brauer reluctivity of Eq. (3), nu(B) = a exp(b B^2) + c,
the integral (E2) evaluates in closed form, as stated in the text
following Eq. (8) of the manuscript:

    w_iron(B) = a/(2b) exp(b B^2) + c B^2 / 2.               (E4)

Consistently with the permeability bound mu >= mu_0 enforced in
``F0_Main_Mat_Nonlinear`` (i.e. nu <= nu_max = 1/mu_0), the density
switches to the linear vacuum branch above the clamping field B_c
defined by a exp(b B_c^2) + c = nu_max:

    w(B) = w_iron(B),                                   B <= B_c,
    w(B) = w_iron(B_c) + (nu_max/2) (B^2 - B_c^2),      B >  B_c.

Material interpolation
----------------------
Every element is described by two arrays supplied by the driver:

    nu_lin_e : constant (linear) reluctivity of the element, used
               for air, coil, and PM regions and as the "air" branch
               of the SIMP interpolation of Eq. (23);
    s_nl_e   : nonlinear mixing factor in [0, 1]:
                 0        -> purely linear element (air, coil, PM),
                 rho_e^p  -> design-domain element (SIMP, Eq. (23)),
                 1        -> fixed (non-design) iron.

so that the element energy density is

    w_e(B) = (1 - s_nl_e) * nu_lin_e * B^2 / 2 + s_nl_e * w(B).    (E5)

For s_nl_e = 0 this is the linear-material energy; for s_nl_e = 1 it
is the full Brauer energy; in between it is exactly the potential of
the SIMP-interpolated reluctivity of Eq. (23). This keeps the module
independent of the mesh-specific domain identifiers, so the same file
is shared by Examples 1, 2 (nonlinear), and 3.

Interface
---------
The module exposes two functions:

  * ``F0_Main_Energy``      : evaluate E(A) of (E1);
  * ``F0_Main_Line_Search`` : perform the backtracking search (E3)
                              and return the accepted iterate.

The flux density needed by the energy evaluation is recomputed
internally from B = curl A with the same vectorized expressions as
``F5_Main_Comp_Flux`` (kept local to avoid mutating the ``fem``
dictionary and printing inside the backtracking loop).

This module is invoked from:
  - the main driver scripts inside the Newton-Raphson loop
    (``Main_code_Ex1.py``, ``Main_code_Ex2_Nonlinear.py``,
    ``Main_code_Mulpos.py``), where it supplies the step size alpha_k of
    the damped update of Eq. (7).
"""

import numpy as np

# Vacuum permeability and Brauer coefficients (identical to
# F0_Main_Mat_Nonlinear.py; see Eq. (3) and Fig. 1 of the manuscript).
_MU0            = 4.0 * np.pi * 1e-7          # vacuum permeability [H/m]
_NU_MAX         = 1.0 / _MU0                  # reluctivity bound nu <= 1/mu0
_A_BR, _B_BR, _C_BR = 49.4, 1.46, 520.6       # fitted Brauer coefficients

# Clamping field B_c: a exp(b B_c^2) + c = nu_max  ->  linear branch above.
_BC2 = np.log((_NU_MAX - _C_BR) / _A_BR) / _B_BR   # B_c^2
_WC  = (_A_BR / (2.0 * _B_BR)) * np.exp(_B_BR * _BC2) \
       + 0.5 * _C_BR * _BC2                        # w_iron(B_c), see (E4)


def _flux_magnitude(fem, A):
    """Element-wise |B| from nodal A (silent local copy of F5's math).

    Parameters
    ----------
    fem : dict
        Finite-element data (uses ``IX``, ``X``, ``Ae``).
    A : ndarray, shape (ndof,)
        Trial nodal vector potential.

    Returns
    -------
    B : ndarray, shape (ne,)
        Flux-density magnitude |B| per element, B = |curl A|.
    """
    IX = fem["IX"]; X = fem["X"]; Ae = fem["Ae"]

    i = IX[:, 0] - 1; j = IX[:, 1] - 1; k = IX[:, 2] - 1
    xi = X[i, 0]; yi = X[i, 1]
    xj = X[j, 0]; yj = X[j, 1]
    xk = X[k, 0]; yk = X[k, 1]

    bi = yj - yk; ci = xk - xj
    bj = yk - yi; cj = xi - xk
    bk = yi - yj; ck = xj - xi

    inv2A = 1.0 / (2.0 * Ae)
    Bx = inv2A * ( ci * A[i] + cj * A[j] + ck * A[k])
    By = inv2A * (-bi * A[i] - bj * A[j] - bk * A[k])
    return np.sqrt(Bx ** 2 + By ** 2)


def _brauer_energy_density(B):
    """Clamped Brauer energy density w(B) of (E2) and (E4).

    Parameters
    ----------
    B : ndarray, shape (ne,)
        Flux-density magnitude per element [T].

    Returns
    -------
    w : ndarray, shape (ne,)
        Energy density int_0^B nu(s) s ds for the Brauer curve with
        the reluctivity bound nu <= 1/mu0 (linear branch for B > B_c).
    """
    B2 = B ** 2
    # Saturating (Brauer) branch, evaluated at min(B, B_c).
    B2c = np.minimum(B2, _BC2)
    w = (_A_BR / (2.0 * _B_BR)) * np.exp(_B_BR * B2c) + 0.5 * _C_BR * B2c
    # Linear vacuum branch above the clamping field.
    above = B2 > _BC2
    if np.any(above):
        w[above] = _WC + 0.5 * _NU_MAX * (B2[above] - _BC2)
    return w


def F0_Main_Energy(fem, A, T_rhs, nu_lin_e, s_nl_e):
    """Evaluate the magnetic energy functional E(A) of (E1).

    Parameters
    ----------
    fem : dict
        Finite-element data (uses ``IX``, ``X``, ``Ae``).
    A : ndarray, shape (ndof,)
        Nodal vector potential at which E is evaluated.
    T_rhs : ndarray, shape (ndof,)
        Total load vector f + f_pm (coil + permanent-magnet sources).
    nu_lin_e : ndarray, shape (ne,)
        Constant (linear) reluctivity per element; see module docstring.
    s_nl_e : ndarray, shape (ne,)
        Nonlinear mixing factor per element in [0, 1]; see module
        docstring (0 = linear, rho^p = SIMP design domain, 1 = iron).

    Returns
    -------
    E : float
        Value of the energy functional (E1). Only differences of E
        between trial iterates matter for the line search, so no
        depth/unit scaling is applied.
    """
    B  = _flux_magnitude(fem, A)
    B2 = B ** 2

    # Element energy density, (E5): linear part + Brauer part.
    w_lin = 0.5 * nu_lin_e * B2
    w_e   = w_lin.copy()
    nl    = s_nl_e > 0.0                      # skip w_iron on linear elements
    if np.any(nl):
        w_iron  = _brauer_energy_density(B[nl])
        w_e[nl] = (1.0 - s_nl_e[nl]) * w_lin[nl] + s_nl_e[nl] * w_iron

    # E(A) = field energy - load potential, (E1).
    return float(np.dot(fem["Ae"], w_e) - np.dot(A, T_rhs))


def F0_Main_Line_Search(fem, A_old, deltaA, T_rhs, nu_lin_e, s_nl_e,
                        fixdof, bcval, E_old=None, n_bisec_max=20):
    """Backtracking line search on the magnetic energy, (E3), i.e. the
    acceptance criterion of Eq. (9) of the manuscript.

    Starting from the full Newton step alpha = 1, the step size is
    halved until the energy functional decreases, i.e. the largest
    alpha in {1, 1/2, 1/4, ...} with E(A_old + alpha dA) < E(A_old)
    is accepted. Because the Newton direction is a descent direction
    of the convex functional (E1), such an alpha exists; the damped
    iteration is then globally convergent and recovers the full
    (superlinearly convergent) Newton step near the solution.

    Parameters
    ----------
    fem : dict
        Finite-element data (uses ``IX``, ``X``, ``Ae``).
    A_old : ndarray, shape (ndof,)
        Current Newton iterate A^(k) (Dirichlet values already set).
    deltaA : ndarray, shape (ndof,)
        Full Newton direction dA from Eq. (5) (zero on fixed dofs).
    T_rhs : ndarray, shape (ndof,)
        Total load vector f + f_pm.
    nu_lin_e, s_nl_e : ndarray, shape (ne,)
        Material-interpolation arrays; see module docstring.
    fixdof : ndarray of int
        Dirichlet (fixed) degrees of freedom (0-based).
    bcval : ndarray
        Prescribed values on ``fixdof``.
    E_old : float, optional
        E(A_old) if already available from the previous accepted step
        (saves one energy evaluation); recomputed when ``None``.
    n_bisec_max : int, optional
        Maximum number of step-size halvings (default 20, i.e.
        alpha_min ~ 1e-6). If no decrease is found - which cannot
        occur for a consistent tangent and a convex energy, but is
        guarded against numerically - the smallest trial step is
        returned with a warning.

    Returns
    -------
    A_new : ndarray, shape (ndof,)
        Accepted iterate A^(k+1) = A_old + alpha * deltaA.
    alpha : float
        Accepted step size.
    E_new : float
        Energy at the accepted iterate (pass back as ``E_old`` of the
        next call to avoid recomputation).
    n_trials : int
        Number of energy evaluations spent in the backtracking loop.
    """
    if E_old is None:
        E_old = F0_Main_Energy(fem, A_old, T_rhs, nu_lin_e, s_nl_e)

    alpha = 1.0
    for n_trials in range(1, n_bisec_max + 1):
        A_try = A_old + alpha * deltaA
        A_try[fixdof] = bcval                 # keep Dirichlet values exact
        E_try = F0_Main_Energy(fem, A_try, T_rhs, nu_lin_e, s_nl_e)
        if E_try < E_old:                     # energy decrease: accept (E3)
            return A_try, alpha, E_try, n_trials
        alpha *= 0.5                          # halve the step and retry

    # Numerical safeguard (not expected to trigger): accept smallest step.
    print("  [line search] WARNING: no energy decrease found; "
          f"accepting alpha = {alpha:.2e}")
    return A_try, alpha, E_try, n_bisec_max
