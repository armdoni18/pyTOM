"""
F0_Main_Mat_Derivative.py
=========================

Derivative of the nonlinear ferromagnetic material model with
respect to the magnetic flux density magnitude |B|.

Companion module to ``F0_Main_Mat_Nonlinear.py``. While the
manuscript expresses the nonlinear behavior in terms of the
reluctivity ``nu`` (Eq. (3)), this function returns the
derivative of the permeability mu = 1/nu, namely

    dmu/dB = - 2 * a * b * B * exp(b * B^2)
             --------------------------------
             (a * exp(b * B^2) + c)^2

The conversion to dnu/dB = -(dmu/dB)/mu^2 is performed in the
main driver immediately after this function is called. The
resulting dnu/dB enters the consistent tangent matrix of Eq. (5)
assembled in ``F6_Main_NR_Jacobian.py``.

The same overflow guard and the same coefficients as
``F0_Main_Mat_Nonlinear.py`` are used to keep the two functions
numerically consistent across all elements.
"""

import numpy as np

def F0_Main_Mat_Derivative(B):
    """Evaluate dmu/dB for the Brauer model.

    Parameters
    ----------
    B : ndarray, shape (ne,)
        Magnetic flux density magnitude per element (in Tesla).

    Returns
    -------
    dmu : ndarray, shape (ne,)
        Element-wise derivative of permeability with respect to
        |B|. Zeroed out wherever the safety bound mu >= mu_0 is
        active in ``F0_Main_Mat_Nonlinear.py``, so that the
        Jacobian contribution is consistent with the clamped
        permeability.

    Notes
    -----
    Used by ``F6_Main_NR_Jacobian.py`` through the chain dnu/dB
    to build the material-nonlinearity contribution to the
    consistent Jacobian of Eq. (5).
    """
    B = np.asarray(B, dtype=float)
    mu0 = 4.0 * np.pi * 1e-7

    # Same coefficients and overflow guard as in F0_Main_Mat_Nonlinear
    a, b, c = 49.4, 1.46, 520.6                          # same coefficients as F0_Main_Mat_Nonlinear
    z = np.clip(b * (B ** 2), 0.0, 700.0)   # overflow guard
    expterm = np.exp(z)                                  # exp(b |B|^2)

    # dmu/dB analytic form (chain through mu = 1/(a*exp(b*B^2)+c))
    denom   = (a * expterm + c) ** 2          # (a exp(b|B|^2)+c)^2
    dmu_raw = -(2.0 * a * b * B * expterm) / denom   # analytic dmu/d|B|

    # Suppress the derivative wherever the safety bound was active
    # (mu was clamped to mu0), so the Jacobian remains consistent.
    mu_raw = 1.0 / (a * expterm + c)          # recompute mu to test the clamp
    dmu = np.where(mu_raw > mu0, dmu_raw, 0.0)   # zero derivative where mu was clamped
    return dmu
