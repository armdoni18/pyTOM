"""
F4_Main_Solve_VecPot.py — Numerical Example 1 (IPM motor benchmark)
==================================================================

Magnetostatic linear solver for the IPM motor of Section 5.1.

Same algorithmic role as the Example 3 version (``3. pyTOM Numerical ex 3/F4_Main_Solve_VecPot.py``), where the full module documentation is provided.
Solves Eq. (2) of Section 2.1 to obtain the initial linear vector potential, which is then refined by the Newton-Raphson loop for nonlinear-material cases.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from F0_Main_PM_Source import F0_Main_PM_Source


def F4_Main_Solve_VecPot(fem, inputs, nu_e_all):

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    if nu_e_all.size != ne:
        raise ValueError("nu_e_all length must equal fem['ne']")

    T = np.zeros(ndof, dtype=float)

    fem["nu_e"] = nu_e_all.copy()

    # Element-wise stiffness
    Velist = np.repeat(nu_e_all, 9) * fem["S_S"]

    # Assemble global stiffness
    I = fem["is"].astype(int) - 1
    J = fem["js"].astype(int) - 1

    S = sp.coo_matrix((Velist, (I, J)), shape=(ndof, ndof)).tocsc()
    S = (S + S.T) * 0.5

    # Dirichlet
    bcdof = fem["bcdof"].astype(int) - 1
    bcval = fem["bcval"].astype(float)

    S = S.tolil()
    S[bcdof, :]      = 0.0
    S[:, bcdof]      = 0.0
    S[bcdof, bcdof]  = 1.0
    S = S.tocsc()

    T[bcdof] = bcval

    # Coil current source
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)
    if Tdof.size > 0:
        np.add.at(T, Tdof, Tval)

    # PM source
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm
    T[bcdof] = bcval

    # Solve
    A = spsolve(S, T)

    fem["A"] = A
    fem["S"] = S
    fem["T"] = T

    print("Vector potential computation Done. ✅")
    return fem
