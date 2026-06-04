"""
F8_Main_Comp_Sens.py
====================

Adjoint sensitivity analysis for the force objective and the
volume constraint, with the chain rule extending through the
SIMP interpolation (Eq. (20)), the Heaviside projection
(Eq. (19)), and the Helmholtz filter (Eq. (18)).

Theory link
-----------
The objective is the magnetic force on the plunger, evaluated
from the Maxwell stress tensor integration of Eq. (9). The total
design sensitivity (Eq. (23)) contains explicit and implicit
dependencies on the design variables:

    dF / dphi
        = (partial F / partial A) (dA / dphi)
        + (partial F / partial phi)

where A is the converged magnetic vector potential and phi is
the filtered design variable.

The explicit term is obtained by differentiating Eq. (9) with
respect to A. The implicit term is evaluated by the adjoint
method through Eq. (24):

    K_t^T lambda = partial F / partial A

where K_t is the converged tangent Jacobian from
``F6_Main_NR_Jacobian.py``.

The total sensitivity then reduces to Eq. (25):

    dF / dphi
        = - lambda^T (partial R / partial phi)

with the residual derivative propagated through the SIMP
interpolation (Eq. (20)), Heaviside projection (Eq. (19)), and
Helmholtz filter (Eq. (18)). The volume sensitivity follows the
same chain and corresponds to Eq. (26).
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear


def _sech2(x: np.ndarray) -> np.ndarray:
    """Element-wise sech^2(x) = 1 / cosh(x)^2."""
    return 1.0 / (np.cosh(x) ** 2)

def F8_Main_Comp_Sens(fem, opt, J):
    """Compute adjoint sensitivities for the force and volume.

    Parameters
    ----------
    fem : dict
        Converged FEM state, including field, force, MST path, and
        material data.
    opt : dict
        Optimization state, including ``erho``, ``bt``, ``Ten``,
        and design DOFs.
    J : csc_matrix
        Converged Newton-Raphson Jacobian used for the adjoint solve.

    Returns
    -------
    tuple
        ``f``, ``g``, ``dfdx``, ``dgdx``, ``dfdrho_e``, ``lam``,
        and ``dfdA``.
    """

    # =====================================================================
    # OBJECTIVE (computed in F7) AND FIELD DATA
    # =====================================================================
    f = float(fem.get("Fy_total", 0.0))

    IX     = np.asarray(fem["IX"], dtype=int)
    X      = np.asarray(fem["X"],  dtype=float)
    A      = np.asarray(fem["A"],  dtype=float).reshape(-1)    # converged A_z
    Bx_all = np.asarray(fem["Bx"], dtype=float).reshape(-1)
    By_all = np.asarray(fem["By"], dtype=float).reshape(-1)
    Bmag   = np.asarray(fem["B"],  dtype=float).reshape(-1)
    nu_e   = np.asarray(fem["nu_e"], dtype=float).reshape(-1)

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])
    Ae_e = np.asarray(fem["Ae"], dtype=float).reshape(-1)
    penal = float(opt["penal"])

    # =====================================================================
    # STEP 1: EXPLICIT dF/dA FROM MST -- explicit derivative of Eq. (9)
    # =====================================================================
    dfdA = np.zeros(ndof, dtype=float)

    edges = np.asarray(fem["cleaned_air_loop_around_plunger"], dtype=int)  # same loop as F7
    Nedge = edges.shape[0]

    # Plunger centroid for normal orientation (identical to F7)
    plunger_mask   = (IX[:, 3] == 5)
    plunger_nodes  = np.unique(IX[plunger_mask, 0:3])
    plunger_center = np.mean(X[plunger_nodes - 1, :], axis=0)

    n1 = edges[:, 0] - 1                      # edge endpoints (0-based)
    n2 = edges[:, 1] - 1
    p1 = X[n1, :]
    p2 = X[n2, :]

    seg  = p2 - p1                            # edge vectors
    ds   = np.linalg.norm(seg, axis=1)        # edge lengths
    valid = ds > 0.0

    normal_raw = np.column_stack([seg[:, 1], -seg[:, 0]])
    norm_mag   = np.linalg.norm(normal_raw, axis=1, keepdims=True) + 1e-30
    normal     = normal_raw / norm_mag        # unit normals

    mid = 0.5 * (p1 + p2)                      # edge midpoints

    to_pl    = plunger_center[None, :] - mid  # orient normals outward (as in F7)
    dot_sign = np.sum(normal * to_pl, axis=1)
    normal[dot_sign > 0.0, :] *= -1.0

    eps_shift   = 1e-3                         # same inward shift as F7
    shifted_mid = mid + eps_shift * normal

    # Nearest air-side element to each edge (provides B and nu)
    nodes012  = IX[:, 0:3] - 1
    centroids = np.mean(X[nodes012, :], axis=1)
    dist_mat  = cdist(shifted_mid, centroids)
    closest_e = np.argmin(dist_mat, axis=1)

    # Shape-function gradient coefficients of each closest element
    ce_nodes = IX[closest_e, 0:3] - 1         # (Nedge, 3) node indices
    Ve_arr   = Ae_e[closest_e]                # element areas
    xi_c = X[ce_nodes[:, 0], 0]; yi_c = X[ce_nodes[:, 0], 1]
    xj_c = X[ce_nodes[:, 1], 0]; yj_c = X[ce_nodes[:, 1], 1]
    xk_c = X[ce_nodes[:, 2], 0]; yk_c = X[ce_nodes[:, 2], 1]
    bi_c = yj_c - yk_c;  ci_c = xk_c - xj_c
    bj_c = yk_c - yi_c;  cj_c = xi_c - xk_c
    bk_c = yi_c - yj_c;  ck_c = xj_c - xi_c
    inv2V = 1.0 / (2.0 * Ve_arr + 1e-30)

    # dB/dA_node for the three nodes (B = curl A on linear triangles)
    dBx_dA = np.column_stack([ci_c, cj_c, ck_c]) * inv2V[:, None]   # (Nedge, 3)
    dBy_dA = np.column_stack([-bi_c, -bj_c, -bk_c]) * inv2V[:, None]

    Bx_c = Bx_all[closest_e]                    # B at the integration element
    By_c = By_all[closest_e]
    nu_c = nu_e[closest_e]                      # reluctivity at the integration element
    nx = normal[:, 0]
    ny = normal[:, 1]

    # Derivatives of the stress-tensor components w.r.t. nodal A:
    #   dT_xy/dA = nu (By dBx/dA + Bx dBy/dA)
    #   dT_yy/dA = nu (By dBy/dA - Bx dBx/dA)
    dTxy_dA = nu_c[:, None] * (By_c[:, None] * dBx_dA + Bx_c[:, None] * dBy_dA)
    dTyy_dA = nu_c[:, None] * (By_c[:, None] * dBy_dA - Bx_c[:, None] * dBx_dA)

    # Per-edge dFy/dA (only the y-force enters the objective)
    dFy_edge = (dTxy_dA * nx[:, None] + dTyy_dA * ny[:, None]) * ds[:, None]   # (Nedge, 3)
    dFy_edge = np.where(valid[:, None], dFy_edge, 0.0)

    # Scatter each node's contribution into the global dF/dA vector
    for local in range(3):
        np.add.at(dfdA, ce_nodes[:, local], dFy_edge[:, local])

    # =====================================================================
    # STEP 2: ADJOINT SOLVE USING NR JACOBIAN -- Eq. (24): K_t^T lambda = dF/dA
    # =====================================================================
    # Adjoint solve  -- Eq. (24):  K_t^T lambda = dF/dA
    all_dofs = np.arange(ndof, dtype=int)
    fixdof   = np.asarray(fem["bcdof"], dtype=int).reshape(-1) - 1   # Dirichlet DOFs
    freedof  = np.setdiff1d(all_dofs, fixdof)                        # free DOFs

    lam = np.zeros(ndof, dtype=float)
    J_ff = J[freedof][:, freedof]             # free-free block of K_t
    lam[freedof] = spsolve(J_ff, dfdA[freedof])   # solve on free DOFs only

    # =====================================================================
    # STEP 3: dF/drho_e VIA NONLINEAR SIMP DIFFERENTIATION
    # =====================================================================
    dfdrho_e = np.zeros(ne, dtype=float)

    dd_mask = (IX[:, 3] == 2)                 # design-domain elements
    dd_idx  = np.where(dd_mask)[0]

    if dd_idx.size > 0:
        erho_dd   = np.asarray(opt["erho"], dtype=float).reshape(-1)[dd_idx]  # physical density
        nu_dd     = nu_e[dd_idx]              # converged nu(B) (unused below; kept for clarity)
        Bmag_dd   = Bmag[dd_idx]              # |B| at design elements

        # nu_air: read off any air element (fallback to 1/mu0 if none)
        air_elems = np.where(IX[:, 3] == 1)[0]
        nu_air = nu_e[int(air_elems[0])] if air_elems.size > 0 else 1.0 / (4.0 * np.pi * 1e-7)

        # nu_iron(|B|) at the converged field, from the Brauer model
        mu_iron_dd = F0_Main_Mat_Nonlinear(Bmag_dd)
        nu_iron_dd = 1.0 / mu_iron_dd

        # SIMP derivative -- Eq. (20):  dnu/drho = (nu_iron - nu_air) p rho^(p-1)
        dnu_drho_dd = (nu_iron_dd - nu_air) * penal * (erho_dd ** (penal - 1.0))

        # Element kernel K0_e for the design elements
        S_S_mat = fem["S_S"].reshape(ne, 3, 3)
        K0_dd   = S_S_mat[dd_idx]             # (ndd, 3, 3)
        nodes_dd = IX[dd_idx, 0:3] - 1        # design-element nodes (ndd, 3)
        Ae_vec  = A[nodes_dd]                 # nodal A per design element
        lam_vec = lam[nodes_dd]               # nodal adjoint per design element

        # Element sensitivity -- Eq. (25):  dF/drho_e = lambda^T (dnu/drho K0_e) A
        K0A = np.einsum('eij,ej->ei', K0_dd, Ae_vec)        # (K0_e A) per element
        dfdrho_e[dd_idx] = dnu_drho_dd * np.sum(lam_vec * K0A, axis=1)

    # =====================================================================
    # STEP 4: CHAIN TO NODAL DESIGN VARIABLES
    # =====================================================================
    Ten   = opt["Ten"]                        # element-to-nodal averaging (Eq. (17))
    bt    = float(opt["bt"])                  # projection sharpness beta
    fdv   = np.asarray(opt["fdv"], dtype=float).reshape(-1)   # filtered design field

    # Heaviside projection derivative -- Eq. (19): beta sech^2(beta fdv)/(2 tanh beta)
    denom        = 2.0 * np.tanh(bt) + 1e-30
    DnrhoDfdv    = _sech2(bt * fdv) * bt / denom              # (nn,)

    # Map element sensitivity back to nodes (Ten^T) and apply projection factor
    dfdx_all = np.asarray(Ten.T @ dfdrho_e).reshape(-1) * DnrhoDfdv

    dof_dd = np.asarray(opt["dof_dd"], dtype=int).reshape(-1) - 1   # design DOFs (0-based)
    dfdx   = dfdx_all[dof_dd].reshape(-1, 1)                        # restrict to design DOFs

    # =====================================================================
    # STEP 5: VOLUME CONSTRAINT AND SENSITIVITY -- Eq. (26)
    # =====================================================================
    VT      = float(opt["VT"])
    VND     = float(opt["VND"])
    volfrac = float(opt["volfrac"])

    erho = np.asarray(opt["erho"], dtype=float).reshape(-1)
    V_rho = float(np.dot(Ae_e, erho))                       # current material volume
    g     = (V_rho - VND) / (VT - VND + 1e-30) - volfrac    # normalized constraint

    dVdrho_e  = Ae_e / (VT + 1e-30)                                     # dV/drho_e = A_e / V_T
    dgdx_all  = np.asarray(Ten.T @ dVdrho_e).reshape(-1) * DnrhoDfdv    # same projection chain
    dgdx      = dgdx_all[dof_dd].reshape(1, -1)

    print("Nonlinear Sensitivity (NR-adjoint) Computation Done. ✅")
    return f, g, dfdx, dgdx, dfdrho_e.reshape(-1, 1), lam.reshape(-1, 1), dfdA.reshape(-1, 1)
