"""
F0_Main_Mat_Nonlinear.py
========================

Nonlinear ferromagnetic material model: Brauer's three-parameter saturation curve.

Implements Eq. (3) of the manuscript (Section 2.2):

    nu(|B|) = a * exp(b * |B|^2) + c

where ``nu`` is the magnetic reluctivity, ``|B|`` is the local magnetic flux density magnitude, and ``a``, ``b``, ``c`` are material coefficients obtained by curve-fitting to the B-H data shown in Fig. 1 of the manuscript.
In this implementation the coefficients are hard-coded as a=49.4, b=1.46, c=520.6 for the ferromagnetic material used in the numerical examples.

Convention note
---------------
Although Eq. (3) is written in terms of the reluctivity nu, this function returns the **permeability** mu = 1/nu. The conversion to reluctivity (and to its derivative through the chain rule dnu/dB = -(dmu/dB)/mu^2)
is performed in the main driver ``Main_code_Mulpos.py`` immediately after this function is called, together with the companion module ``F0_Main_Mat_Derivative.py``.

Returning mu directly is convenient because (i) the safety bound mu >= mu_0 (vacuum permeability) is naturally expressed in terms of mu and (ii) it matches the form in which manufacturer B-H data is usually provided.

This module is invoked from:
    - Main_code_Mulpos.py  (inside the Newton-Raphson loop, to update the field-dependent reluctivity)
    - F8_Main_Comp_Sens.py (to evaluate nu_iron at the converged field for SIMP sensitivity)
"""

import numpy as np

def F0_Main_Mat_Nonlinear(B):
    """
    Evaluate the Brauer saturation model.

    Parameters
    ----------
    B : ndarray, shape (ne,)
        Magnetic flux density magnitude per element (in Tesla).

    Returns
    -------
    mu : ndarray, shape (ne,)
        Element-wise magnetic permeability mu = 1/nu, where nu is the reluctivity of Eq. (3). Lower-bounded to mu_0 (vacuum permeability) to prevent unphysical values.

    Notes
    -----
    The exponent ``b * |B|^2`` is clipped to 700 to avoid float64 overflow in ``np.exp`` (which overflows near exp(709)). The clip threshold corresponds to |B| of about 22 T, which is far beyond any physically meaningful operating point
    for the ferromagnetic materials considered here, so this guard has no effect in practice.
    """
    B = np.asarray(B, dtype=float)
    mu0 = 4.0 * np.pi * 1e-7

    # Brauer coefficients from curve-fit to Fig. 1 of the manuscript
    a, b, c = 49.4, 1.46, 520.6

    # Overflow guard: exp(z) overflows for z >= 709 in float64
    z = np.clip(b * (B ** 2), 0.0, 700.0)
    expterm = np.exp(z)

    # Eq. (3) inverted: mu = 1 / (a*exp(b*B^2) + c)
    mu_raw = 1.0 / (a * expterm + c)

    # Safety bound: permeability must not fall below vacuum value
    mu = np.maximum(mu_raw, mu0)
    return mu
