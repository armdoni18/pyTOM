"""
F0_Main_Mat_Nonlinear.py
========================

Nonlinear ferromagnetic material model: Brauer's three-parameter
saturation curve.

Implements Eq. (3) of the manuscript (Section 2.2):

    nu(|B|) = a * exp(b * |B|^2) + c

where ``nu`` is the magnetic reluctivity, ``|B|`` is the local
magnetic flux density magnitude, and ``a``, ``b``, ``c`` are
material coefficients obtained by curve-fitting to the B-H data
shown in Fig. 1 of the manuscript. In this implementation the
coefficients are hard-coded as a=49.4, b=1.46, c=520.6 for the
ferromagnetic material used in the numerical examples.

Convention note
---------------
Although Eq. (3) is written in terms of the reluctivity nu, this
function returns the **permeability** mu = 1/nu. The conversion
to reluctivity and to its derivative through the chain rule
dnu/dB = -(dmu/dB)/mu^2 is performed in the main driver script
immediately after this function is called, together with the
companion module ``F0_Main_Mat_Derivative.py``.

Returning mu directly is convenient because (i) the safety bound
mu >= mu_0 (vacuum permeability) is naturally expressed in terms
of mu and (ii) it matches the form in which manufacturer B-H data
is usually provided.

This module is invoked from:
  - the main driver script inside the Newton-Raphson loop, to
    update the field-dependent reluctivity;
  - ``F8_Main_Comp_Sens.py`` to evaluate nu_iron at the converged
    field for SIMP sensitivity.
"""

import numpy as np

def F0_Main_Mat_Nonlinear(B):
    """Evaluate the nonlinear permeability mu(B) for the Brauer model.

    Parameters
    ----------
    B : ndarray, shape (ne,)
        Magnetic flux density magnitude per element (in Tesla).

    Returns
    -------
    mu : ndarray, shape (ne,)
        Element-wise nonlinear permeability. The returned value is
        bounded below by mu_0 to avoid nonphysical values below the
        vacuum permeability.
    """
    B = np.asarray(B, dtype=float)

    # Vacuum permeability.
    mu0 = 4.0 * np.pi * 1e-7

    # Brauer coefficients from curve-fit to Fig. 1 of the manuscript
    a, b, c = 49.4, 1.46, 520.6              # fitted Brauer coefficients

    # Overflow guard for exp(b*B^2).
    z = np.clip(b * (B ** 2), 0.0, 700.0)    # clamp the exponent argument
    expterm = np.exp(z)                       # exp(b |B|^2)

    # Brauer model permeability, mu = 1/(a*exp(b*B^2) + c) (inverse of Eq. (3))
    mu_raw = 1.0 / (a * expterm + c)          # mu = 1/nu, nu = a exp(b|B|^2)+c

    # Safety bound: permeability should not be below vacuum permeability.
    mu = np.maximum(mu_raw, mu0)              # enforce mu >= mu0
    return mu
