"""
F4_Main_Solve_VecPot.py
=======================

Assemble and solve the discrete magnetostatic system

    K(nu) * A = f + f_pm                              (Eq. (2))

corresponding to Section 2.1 of the manuscript. This module
provides an initial linear vector potential ``A`` once per
analysis, i.e. once per plunger position in the multi-position
case. For nonlinear analyses, this solution serves as the
starting point of the Newton-Raphson iteration of Eqs. (4)-(6)
in the main driver.

The same global system structure is also re-assembled inside
``F6_Main_NR_Jacobian.py`` during each NR iteration. The two
assemblies share the pre-computed element kernel ``S_S`` and
COO triplet indices ``is``, ``js`` built in ``F2_Pre_FEM_Init.py``.

The right-hand side combines:
    - the coil contribution ``f`` pre-computed in F2 as
      ``Tdof`` and ``Tval``;
    - the permanent-magnet contribution ``f_pm`` assembled here
      by calling ``F0_Main_PM_Source``.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from F0_Main_PM_Source import F0_Main_PM_Source

def F4_Main_Solve_VecPot(fem, inputs, nu_e_all):
    """Solve K(nu) A = f + f_pm for the magnetic vector potential.

    Parameters
    ----------
    fem : dict
        Finite-element data built by ``F2_Pre_FEM_Init``. Must
        contain ``ne``, ``ndof``, ``S_S``, ``is``, ``js``,
        ``bcdof``, ``bcval``, ``Tdof``, and ``Tval``.
    inputs : dict
        Problem input dictionary. Used here to pass the ``PM``
        sub-dictionary to ``F0_Main_PM_Source``.
    nu_e_all : ndarray, shape (ne,)
        Element-wise magnetic reluctivity nu.

    Returns
    -------
    fem : dict
        Updated in place with:
            ``A`` : (ndof,) solved vector potential
            ``S`` : (ndof, ndof) assembled global stiffness matrix
            ``T`` : (ndof,) assembled right-hand side
    """

    # =====================================================================
    # INPUT CHECKS AND INITIALIZATION
    # =====================================================================
    ndof = int(fem["ndof"])                 # total number of DOFs (= number of nodes)
    ne   = int(fem["ne"])                   # number of triangular elements

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)   # force 1-D float array
    if nu_e_all.size != ne:                 # one reluctivity value per element is required
        raise ValueError("nu_e_all length must equal fem['ne']")

    T = np.zeros(ndof, dtype=float)         # right-hand side vector f + f_pm (filled below)
    fem["nu_e"] = nu_e_all.copy()           # cache nu_e for reuse in F7 (force) and F8 (sens)

    # =====================================================================
    # GLOBAL STIFFNESS ASSEMBLY
    # =====================================================================
    # Assemble K(nu) of Eq. (2), left-hand side, from the reluctivity-free element kernel.
    Velist = np.repeat(nu_e_all, 9) * fem["S_S"]     # (9*ne,) triplet values
    I = fem["is"].astype(int) - 1                    # COO row indices (Gmsh 1-based -> 0-based)
    J = fem["js"].astype(int) - 1                    # COO column indices (1-based -> 0-based)
    S = sp.coo_matrix((Velist, (I, J)),
                      shape=(ndof, ndof)).tocsc()    # duplicate (i,j) entries are summed
    S = (S + S.T) * 0.5                              # symmetrize away accumulation round-off

    # =====================================================================
    # DIRICHLET BOUNDARY CONDITIONS
    # =====================================================================
    # Homogeneous Dirichlet condition A_z = 0 on the prescribed boundary.
    bcdof = fem["bcdof"].astype(int) - 1             # constrained DOF indices (0-based)
    bcval = fem["bcval"].astype(float)               # prescribed values (zero here)
    S = S.tolil()                                    # LIL allows cheap row/col edits
    S[bcdof, :] = 0.0                                # clear constrained rows
    S[:, bcdof] = 0.0                                # clear constrained columns
    S[bcdof, bcdof] = 1.0                            # unit diagonal on constrained DOFs
    S = S.tocsc()                                    # back to CSC for the solve
    T[bcdof] = bcval

    # =====================================================================
    # RIGHT-HAND SIDE: COIL SOURCE
    # =====================================================================
    # Add the coil-current contribution f of Eq. (2).
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1   # node indices (0-based)
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)     # per-node current load
    np.add.at(T, Tdof, Tval)                                    # scatter-add (handles repeated nodes)

    # =====================================================================
    # RIGHT-HAND SIDE: PERMANENT-MAGNET SOURCE
    # =====================================================================
    # Add the permanent-magnet contribution f_pm of Eq. (2).
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:                            # guard against a mesh/source mismatch
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm
    T[bcdof] = bcval                                 # re-enforce Dirichlet after the additions

    # =====================================================================
    # LINEAR SOLVE
    # =====================================================================
    A = spsolve(S, T)

    fem["A"] = A                                     # solved vector potential
    fem["S"] = S                                     # keep assembled K (reused by F6/F8)
    fem["T"] = T                                     # keep RHS for diagnostics

    print("Vector potential computation Done. ✅")
    return fem
