"""
F4_Main_Solve_VecPot.py — Numerical Example 2 (nonlinear actuator)
==================================================================

Magnetostatic linear solver for the nonlinear-material actuator of Section 5.2.

Same algorithmic role as the Example 3 version (``3. pyTOM Numerical ex 3/F4_Main_Solve_VecPot.py``), where the full module documentation is provided.

This solve produces the initial linear vector potential which is then refined by the Newton-Raphson iteration in the main driver.
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

    # Store nu into fem
    fem["nu_e"] = nu_e_all.copy()

    # Build element-wise stiffness: nu_e * K0_e
    Velist = np.repeat(nu_e_all, 9) * fem["S_S"]

    # Assemble global stiffness matrix S
    I = fem["is"].astype(int) - 1
    J = fem["js"].astype(int) - 1

    S = sp.coo_matrix((Velist, (I, J)), shape=(ndof, ndof)).tocsc()
    S = (S + S.T) * 0.5

    # Apply Dirichlet boundary conditions
    bcdof = fem["bcdof"].astype(int) - 1
    bcval = fem["bcval"].astype(float)

    S = S.tolil()
    S[bcdof, :] = 0.0
    S[:, bcdof] = 0.0
    S[bcdof, bcdof] = 1.0
    S = S.tocsc()

    T[bcdof] = bcval

    # Apply current source
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)
    np.add.at(T, Tdof, Tval)

    # Permanent magnet excitation
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm
    T[bcdof] = bcval

    # Solve for vector potential
    A = spsolve(S, T)

    fem["A"] = A
    fem["S"] = S
    fem["T"] = T

    print("Vector potential computation Done. ✅")
    return fem
