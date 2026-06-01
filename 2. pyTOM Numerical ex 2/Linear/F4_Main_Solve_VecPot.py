"""
F4_Main_Solve_VecPot.py
=======================

Assemble and solve the discrete magnetostatic system

    K(nu) * A = f + f_pm                              (Eq. (2))

corresponding to Section 2.1 of the manuscript. This module
provides an initial linear vector potential ``A`` (once per
analysis, i.e. once per plunger position in the multi-position
case), which then serves as the starting point of the
Newton-Raphson iteration of Eqs. (4)-(6) in the main driver. The same global system structure is also re-assembled
inside ``F6_Main_NR_Jacobian.py`` during each NR iteration; the
two assemblies share the pre-computed element kernel ``S_S`` and
COO triplet indices ``is``, ``js`` built in ``F2_Pre_FEM_Init.py``.

The right-hand side combines:
    - the coil contribution ``f`` (pre-computed in F2 as
      ``Tdof``, ``Tval``),
    - the permanent-magnet contribution ``f_pm`` (assembled here
      by calling ``F0_Main_PM_Source``).
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
        Finite-element data built by ``F2_Pre_FEM_Init``. Must
        contain: ``ne``, ``ndof``, ``S_S`` (flat element kernel),
        ``is``, ``js`` (COO triplet indices), ``bcdof``, ``bcval``
        (Dirichlet info), ``Tdof``, ``Tval`` (coil source).
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
            ``S`` : (ndof, ndof) assembled global stiffness (csc)
            ``T`` : (ndof,) assembled right-hand side
    """

# Read problem sizes and validate the reluctivity input
    ndof = int(fem["ndof"])                 # total number of DOFs (= number of nodes)
    ne   = int(fem["ne"])                   # number of triangular elements

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)   # force 1-D float array
    if nu_e_all.size != ne:                 # one reluctivity value per element is required
        raise ValueError("nu_e_all length must equal fem['ne']")

    T = np.zeros(ndof, dtype=float)         # right-hand side vector f + f_pm (filled below)
    fem["nu_e"] = nu_e_all.copy()           # cache nu_e for reuse in F7 (force) and F8 (sens)

# Assemble the global stiffness K(nu)  -- Eq. (2), left-hand side
    # Each element contributes nu_e * K0_e (9 entries of the 3x3 kernel).
    # Broadcasting nu_e over its 9 entries and multiplying the flat
    # pre-computed kernel S_S gives the COO value array in one shot.
    Velist = np.repeat(nu_e_all, 9) * fem["S_S"]     # (9*ne,) triplet values
    I = fem["is"].astype(int) - 1                    # COO row indices (Gmsh 1-based -> 0-based)
    J = fem["js"].astype(int) - 1                    # COO column indices (1-based -> 0-based)
    S = sp.coo_matrix((Velist, (I, J)),
                      shape=(ndof, ndof)).tocsc()    # duplicate (i,j) entries are summed
    S = (S + S.T) * 0.5                              # symmetrize away accumulation round-off

# Impose Dirichlet boundary conditions (A_z = 0 on the outer boundary)
    # Penalty-free row/column clearing: zero the constrained rows and
    # columns, set a unit diagonal, and write the prescribed value to T.
    bcdof = fem["bcdof"].astype(int) - 1             # constrained DOF indices (0-based)
    bcval = fem["bcval"].astype(float)               # prescribed values (zero here)
    S = S.tolil()                                    # LIL allows cheap row/col edits
    S[bcdof, :] = 0.0                                # clear constrained rows
    S[:, bcdof] = 0.0                                # clear constrained columns
    S[bcdof, bcdof] = 1.0                            # unit diagonal on constrained DOFs
    S = S.tocsc()                                    # back to CSC for the solve
    T[bcdof] = bcval

# Build the right-hand side: coil current source  -- f in Eq. (2)
    # Tdof/Tval (from F2) hold the per-node contribution J*A_e/3 of every
    # coil element, with opposite signs for the in/out current return path.
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1   # node indices (0-based)
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)     # per-node current load
    np.add.at(T, Tdof, Tval)                         # scatter-add (handles repeated nodes)

# Add the permanent-magnet source  -- f_pm in Eq. (2)
    # F0_Main_PM_Source returns the equivalent-current load of curl(nu*Br);
    # see Section 2.1 and the F0_Main_PM_Source appendix entry.
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:                            # guard against a mesh/source mismatch
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm
    T[bcdof] = bcval                                 # re-enforce Dirichlet after the additions

# Solve the linear system K A = f + f_pm  (SuperLU through spsolve)
    A = spsolve(S, T)

    fem["A"] = A                                     # solved vector potential
    fem["S"] = S                                     # keep assembled K (reused by F6/F8)
    fem["T"] = T                                     # keep RHS for diagnostics

    print("Vector potential computation Done. ✅")
    return fem
