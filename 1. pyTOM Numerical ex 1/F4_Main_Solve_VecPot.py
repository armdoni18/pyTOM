import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from F0_Main_PM_Source import F0_Main_PM_Source

def F4_Main_Solve_VecPot(fem, inputs, nu_e_all):

    # === 1) Initialize global system ===
    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])

    nu_e_all = np.asarray(nu_e_all, dtype=float).reshape(-1)
    if nu_e_all.size != ne:
        raise ValueError("nu_e_all length must equal fem['ne']")

    T = np.zeros(ndof, dtype=float)

    # === 2) Store nu into fem ===
    if fem["IX"].shape[1] >= 5:
        fem["IX"][:, 4] = nu_e_all

    fem["nu_e"] = nu_e_all.copy()

    # === 3) Build element-wise stiffness vector (Velist) ===
    Velist = np.kron(nu_e_all, np.ones(9, dtype=float)) * fem["S_S"]

    # === 4) Assemble global stiffness matrix S ===
    I = fem["is"].astype(int) - 1
    J = fem["js"].astype(int) - 1

    S = sp.coo_matrix((Velist, (I, J)), shape=(ndof, ndof)).tocsc()
    S = (S + S.T) * 0.5

    # === 5) Apply boundary conditions (Dirichlet) ===
    bcdof = fem["bcdof"].astype(int) - 1
    bcval = fem["bcval"].astype(float)

    S = S.tolil()
    S[bcdof, :] = 0.0
    S[:, bcdof] = 0.0
    S[bcdof, bcdof] = 1.0
    S = S.tocsc()

    T[bcdof] = bcval

    # === 6) Apply current source ===
    Tdof = np.asarray(fem["Tdof"], dtype=int).reshape(-1) - 1
    Tval = np.asarray(fem["Tval"], dtype=float).reshape(-1)
    np.add.at(T, Tdof, Tval)

    # === 7) Permanent Magnet excitation(multi - domain) ===
    T_pm = F0_Main_PM_Source(fem, inputs)
    T_pm = np.asarray(T_pm, dtype=float).reshape(-1)
    if T_pm.size != ndof:
        raise ValueError(f"T_pm size mismatch: {T_pm.size} vs ndof {ndof}")
    T += T_pm

    T[bcdof] = bcval

    # === 8) Solve for vector potential A ===
    A = spsolve(S, T)  # returns (ndof,)

    fem["A"] = A
    fem["S"] = S
    fem["T"] = T

    print("FEM Solved and Vector potential computation Done. ✅")
    return fem

