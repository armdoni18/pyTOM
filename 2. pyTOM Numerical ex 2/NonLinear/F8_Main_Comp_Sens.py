"""
F8_Main_Comp_Sens.py
====================

Adjoint sensitivity analysis for the force objective and the
volume constraint, with the chain rule extending through the
SIMP interpolation (Eq. (20)), the Heaviside projection (Eq.
(19)), and the Helmholtz filter (Eq. (18)).

Theory link
-----------
The total design sensitivity (Eq. (23)) for the force objective
combines an explicit term and an implicit term:

    dF / d phi  =  (partial F / partial A) (dA / d phi)
                 + (partial F / partial phi) .

The implicit term is evaluated efficiently by the adjoint method:
a single linear solve of Eq. (24) gives lambda from

    K_t^T lambda = partial F / partial A ,

where K_t is the converged Newton-Raphson Jacobian assembled in
``F6_Main_NR_Jacobian.py``. The sensitivity then reduces to the
compact form of Eq. (25),

    dF / d phi  =  - lambda^T (partial R / partial phi) ,

with the partial derivative of the residual flowing through the
SIMP interpolation, the projection, and the filter.

Implementation overview
-----------------------
STEP 1: dF / dA from MST edge integrals
    Differentiates Eq. (9) with respect to the nodal A_z values.
    For each edge of the closed air loop (same edges as in
    ``F7_Main_Comp_Force.py``), the chain rule combines the
    dependence of (T_xx, T_xy, T_yy) on B with the dependence
    of B on the three nodal A values of the neighboring element.

STEP 2: Adjoint solve
    Eq. (24) with the converged NR Jacobian (passed in as ``J``).

STEP 3: dF / drho_e via SIMP differentiation
    For design-domain elements only:
        dnu/drho = (nu_iron(|B|) - nu_air) * p * rho^(p-1) ,
    and the element-level sensitivity is
        dF/drho_e = lambda^T (dnu/drho * K0_e) A .

STEP 4: Chain to nodal design variables
    The projection derivative contributes the factor
        beta * sech^2(beta * phi_tilde) / (2 tanh beta)
    and the Helmholtz filter contributes a back-substitution
    against the cached LU factorization. The result is then
    restricted to the design-domain DOFs.

The volume sensitivity (Eq. (26)) follows the same projection and
filter chain with the constant dV/drho_e = A_e / V_T.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear


def _sech2(x: np.ndarray) -> np.ndarray:
    """Element-wise sech^2(x) = 1 / cosh(x)^2."""
    return 1.0 / (np.cosh(x) ** 2)

def F8_Main_Comp_Sens(fem, opt, J):
    """
    Compute adjoint sensitivities for the force and volume.

    Parameters
    ----------
    fem : dict
        Finite-element data at the converged NR state. Must
        contain the same entries used by F7, plus ``nu_e``.
    opt : dict
        Optimization state (filter LU factors, Ten matrix,
        current ``erho``, projection sharpness ``bt``, etc.).
    J : csc_matrix
        Converged Newton-Raphson Jacobian K_t from
        ``F6_Main_NR_Jacobian``.

    Returns
    -------
    f : float
        Scalar objective value (Fy total).
    g : float
        Volume-constraint value (V/VT - volfrac).
    dfdx : ndarray, shape (n_dd, 1)
        Sensitivity of f w.r.t. design DOFs.
    dgdx : ndarray, shape (1, n_dd)
        Sensitivity of g w.r.t. design DOFs.
    dfdrho_e : ndarray, shape (ne, 1)
        Element-level objective sensitivity (diagnostic).
    lam : ndarray, shape (ndof, 1)
        Adjoint vector (diagnostic).
    dfdA : ndarray, shape (ndof, 1)
        Explicit dF/dA contribution (diagnostic).
    """

# Objective value f = Fy on the plunger (computed in F7)
    f = float(fem.get("Fy_total", 0.0))

# Unpack converged-state fields
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

    # -------------------------------------------------------
    # STEP 1: dF/dA from MST  -- explicit derivative of Eq. (9)
    # -------------------------------------------------------
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

    Bx_c = Bx_all[closest_e]                  # B at the integration element
    By_c = By_all[closest_e]
    nu_c = nu_e[closest_e]                     # reluctivity at the integration element
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

    # -------------------------------------------------------
    # STEP 2: Adjoint solve  -- Eq. (24):  K_t^T lambda = dF/dA
    # -------------------------------------------------------
    all_dofs = np.arange(ndof, dtype=int)
    fixdof   = np.asarray(fem["bcdof"], dtype=int).reshape(-1) - 1   # Dirichlet DOFs
    freedof  = np.setdiff1d(all_dofs, fixdof)                        # free DOFs

    lam = np.zeros(ndof, dtype=float)
    J_ff = J[freedof][:, freedof]             # free-free block of K_t
    lam[freedof] = spsolve(J_ff, dfdA[freedof])   # solve on free DOFs only

    # -------------------------------------------------------
    # STEP 3: dF/drho_e via SIMP differentiation (design domain)
    # -------------------------------------------------------
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

    # -------------------------------------------------------
    # STEP 4: Chain to nodal design variables (projection + filter)
    # -------------------------------------------------------
    Ten   = opt["Ten"]                        # element-to-nodal averaging (Eq. (17))
    bt    = float(opt["bt"])                  # projection sharpness beta
    fdv   = np.asarray(opt["fdv"], dtype=float).reshape(-1)   # filtered design field

    # Heaviside projection derivative -- Eq. (19): beta sech^2(beta fdv)/(2 tanh beta)
    denom        = 2.0 * np.tanh(bt) + 1e-30
    DnrhoDfdv    = _sech2(bt * fdv) * bt / denom              # (nn,)

    # Map element sensitivity back to nodes (Ten^T) and apply projection factor
    dfdx_all = np.asarray(Ten.T @ dfdrho_e).reshape(-1) * DnrhoDfdv

    dof_dd = np.asarray(opt["dof_dd"], dtype=int).reshape(-1) - 1   # design DOFs (0-based)
    dfdx   = dfdx_all[dof_dd].reshape(-1, 1)  # restrict to design DOFs

    # -------------------------------------------------------
    # Volume constraint value and sensitivity  -- Eq. (26)
    # -------------------------------------------------------
    VT      = float(opt["VT"])
    VND     = float(opt["VND"])
    volfrac = float(opt["volfrac"])

    erho = np.asarray(opt["erho"], dtype=float).reshape(-1)
    V_rho = float(np.dot(Ae_e, erho))         # current material volume
    g     = (V_rho - VND) / (VT - VND + 1e-30) - volfrac   # normalized constraint

    dVdrho_e  = Ae_e / (VT + 1e-30)           # dV/drho_e = A_e / V_T
    dgdx_all  = np.asarray(Ten.T @ dVdrho_e).reshape(-1) * DnrhoDfdv   # same projection chain
    dgdx      = dgdx_all[dof_dd].reshape(1, -1)

    print("Nonlinear Sensitivity (NR-adjoint) Computation Done. ✅")
    return f, g, dfdx, dgdx, dfdrho_e.reshape(-1, 1), lam.reshape(-1, 1), dfdA.reshape(-1, 1)
