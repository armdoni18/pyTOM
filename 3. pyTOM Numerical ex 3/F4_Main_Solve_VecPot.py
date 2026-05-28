"""
F4_Main_Solve_VecPot.py
=======================

Assemble and solve the discrete magnetostatic system

    K(nu) * A = f + f_pm                              (Eq. (2))

corresponding to Section 2.1 of the manuscript. This module is called once per plunger position to provide an initial linear vector potential ``A``,
which then serves as the starting point of the Newton-Raphson iteration of Eqs. (4)-(6) in the main driver.
The same global system structure is also re-assembled inside ``F6_Main_NR_Jacobian.py`` during each NR iteration;
the two assemblies share the pre-computed element kernel ``S_S`` and COO triplet indices ``is``, ``js`` built in ``F2_Pre_FEM_Init.py``.

The right-hand side combines:
    - the coil contribution ``f`` (pre-computed in F2 as ``Tdof``, ``Tval``),
    - the permanent-magnet contribution ``f_pm`` (assembled here by calling ``F0_Main_PM_Source``).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from F0_Main_PM_Source import F0_Main_PM_Source

def F4_Main_Solve_VecPot(fem, inputs, nu_e_all):
    """
    Solve K(nu) A = f + f_pm for the magnetic vector potential.

    Parameters
    ----------
    fem : dict
        Finite-element data built by ``F2_Pre_FEM_Init``. Must contain: ``ne``, ``ndof``, ``S_S`` (flat element kernel), ``is``, ``js`` (COO triplet indices), ``bcdof``, ``bcval`` (Dirichlet info), ``Tdof``, ``Tval`` (coil source).
    inputs : dict
        Problem input dictionary. Used here to pass the ``PM`` sub-dictionary to ``F0_Main_PM_Source``.
    nu_e_all : ndarray, shape (ne,)
        Element-wise magnetic reluctivity nu.

    Returns
    -------
    fem : dict
        Updated in place with:
            ``A`` : (ndof,) solved vector potential
            ``S`` : (ndof, ndof) assembled global stiffness (csc)
            ``T`` : (ndof,) assembled right-hand side
    """

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    if nu_e_all.size != ne:
        raise ValueError("nu_e_all length must equal fem['ne']")

    T = np.zeros(ndof, dtype=float)

    # Store nu_e for later use in F7/F8 (force and sensitivity)
    fem["nu_e"] = nu_e_all.copy()

    # ---- Vectorized element stiffness assembly ----------------
    # Each element contributes 9 entries to the global matrix:
    #   nu_e * K0_e   (K0_e is the pre-computed kernel, shape 3x3)
    # By repeating nu_e_all 9 times and multiplying entry-wise
    # against the flat kernel fem["S_S"], we obtain the value
    # array for the COO triplet (I, J, values).
    Velist = np.repeat(nu_e_all, 9) * fem["S_S"]

    # COO assembly using the row/col index arrays built in F2
    I = fem["is"].astype(int) - 1   # to 0-based
    J = fem["js"].astype(int) - 1

    S = sp.coo_matrix((Velist, (I, J)), shape=(ndof, ndof)).tocsc()
    # Symmetrize against round-off introduced by accumulation order
    S = (S + S.T) * 0.5

    # ---- Dirichlet boundary conditions ------------------------
    # Standard penalty-free row/column-clearing technique:
    # zero the constrained rows and columns, place 1 on the
    # diagonal, and set the RHS entry to the prescribed value.
    bcdof = fem["bcdof"].astype(int) - 1
    bcval = fem["bcval"].astype(float)

    S = S.tolil()
    S[bcdof, :] = 0.0
    S[:, bcdof] = 0.0
    S[bcdof, bcdof] = 1.0
    S = S.tocsc()

    T[bcdof] = bcval

    # ---- Right-hand side: coil current source -----------------
    # Tdof/Tval were pre-computed in F2_Pre_FEM_Init and encode
    # the contribution J * A_e / 3 to each of the three nodes of
    # every coil element (with opposite signs for the two coils).
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)
    np.add.at(T, Tdof, Tval)

    # ---- Right-hand side: permanent magnet source -------------
    # Assembled here by F0_Main_PM_Source from the equivalent-
    # current representation of curl(nu * Br); see Section 2.1
    # and the appendix entry for F0_Main_PM_Source.
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm
    T[bcdof] = bcval   # re-enforce Dirichlet after additions

    # ---- Linear solve (SuperLU via scipy.sparse.linalg) -------
    A = spsolve(S, T)

    fem["A"] = A
    fem["S"] = S
    fem["T"] = T

    print("Vector potential computation Done. ✅")
    return fem
