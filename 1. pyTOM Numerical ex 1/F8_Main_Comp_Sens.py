import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear


def _sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / (np.cosh(x) ** 2)


def F8_Main_Comp_Sens(fem, opt, J):

    f = float(fem.get("Fy_total", 0.0))

    IX     = np.asarray(fem["IX"], dtype=int)
    X      = np.asarray(fem["X"],  dtype=float)
    A      = np.asarray(fem["A"],  dtype=float).reshape(-1)
    Bx_all = np.asarray(fem["Bx"], dtype=float).reshape(-1)
    By_all = np.asarray(fem["By"], dtype=float).reshape(-1)
    Bmag   = np.asarray(fem["B"],  dtype=float).reshape(-1)
    nu_e   = np.asarray(fem["nu_e"], dtype=float).reshape(-1)

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])
    Ae_e = np.asarray(fem["Ae"], dtype=float).reshape(-1)
    penal = float(opt["penal"])

    # -------------------------------------------------------
    # STEP 1: dF/dA from MST
    # -------------------------------------------------------
    dfdA = np.zeros(ndof, dtype=float)

    edges = np.asarray(fem["cleaned_air_loop_around_plunger"], dtype=int)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)

    Nedge = edges.shape[0]

    # Rotor centroid (domain 5)
    rot_mask = (IX[:, 3] == 5)
    rotor_nodes_unique = np.unique(IX[rot_mask, 0:3])
    if rotor_nodes_unique.size > 0:
        rotor_center = np.mean(X[rotor_nodes_unique - 1, :], axis=0)
    else:
        rotor_center = np.array([0.0, 0.0])

    if Nedge > 0:
        n1 = edges[:, 0] - 1
        n2 = edges[:, 1] - 1

        p1 = X[n1, :]
        p2 = X[n2, :]

        seg = p2 - p1
        ds  = np.linalg.norm(seg, axis=1)
        valid = ds > 0.0

        normal_raw = np.column_stack([seg[:, 1], -seg[:, 0]])
        norm_mag   = np.linalg.norm(normal_raw, axis=1, keepdims=True) + 1e-30
        normal     = normal_raw / norm_mag

        mid = 0.5 * (p1 + p2)

        to_rot = rotor_center[None, :] - mid
        dot_sign = np.sum(normal * to_rot, axis=1)
        normal[dot_sign > 0.0, :] *= -1.0

        eps_shift   = 1e-3
        shifted_mid = mid + eps_shift * normal

        nodes012  = IX[:, 0:3] - 1
        centroids = np.mean(X[nodes012, :], axis=1)

        dist_mat  = cdist(shifted_mid, centroids)
        closest_e = np.argmin(dist_mat, axis=1)

        ce_nodes = IX[closest_e, 0:3] - 1
        Ve_arr   = Ae_e[closest_e]

        xi_c = X[ce_nodes[:, 0], 0]; yi_c = X[ce_nodes[:, 0], 1]
        xj_c = X[ce_nodes[:, 1], 0]; yj_c = X[ce_nodes[:, 1], 1]
        xk_c = X[ce_nodes[:, 2], 0]; yk_c = X[ce_nodes[:, 2], 1]

        bi_c = yj_c - yk_c;  ci_c = xk_c - xj_c
        bj_c = yk_c - yi_c;  cj_c = xi_c - xk_c
        bk_c = yi_c - yj_c;  ck_c = xj_c - xi_c

        inv2V = 1.0 / (2.0 * Ve_arr + 1e-30)

        dBx_dA = np.column_stack([ci_c, cj_c, ck_c]) * inv2V[:, None]
        dBy_dA = np.column_stack([-bi_c, -bj_c, -bk_c]) * inv2V[:, None]

        Bx_c = Bx_all[closest_e]
        By_c = By_all[closest_e]
        nu_c = nu_e[closest_e]

        nx = normal[:, 0]
        ny = normal[:, 1]

        dTxy_dA = nu_c[:, None] * (By_c[:, None] * dBx_dA + Bx_c[:, None] * dBy_dA)
        dTyy_dA = nu_c[:, None] * (By_c[:, None] * dBy_dA - Bx_c[:, None] * dBx_dA)

        dFy_edge = (dTxy_dA * nx[:, None] + dTyy_dA * ny[:, None]) * ds[:, None]
        dFy_edge = np.where(valid[:, None], dFy_edge, 0.0)

        for local in range(3):
            np.add.at(dfdA, ce_nodes[:, local], dFy_edge[:, local])

    # -------------------------------------------------------
    # STEP 2: Adjoint solve
    # -------------------------------------------------------
    all_dofs = np.arange(ndof, dtype=int)
    fixdof   = np.asarray(fem["bcdof"], dtype=int).reshape(-1) - 1
    freedof  = np.setdiff1d(all_dofs, fixdof)

    lam = np.zeros(ndof, dtype=float)
    J_ff = J[freedof][:, freedof]
    lam[freedof] = spsolve(J_ff, dfdA[freedof])

    # -------------------------------------------------------
    # STEP 3: df/drho_e for design domain
    # -------------------------------------------------------
    dfdrho_e = np.zeros(ne, dtype=float)

    dd_mask = (IX[:, 3] == 2)
    dd_idx  = np.where(dd_mask)[0]

    if dd_idx.size > 0:
        erho_dd = np.asarray(opt["erho"], dtype=float).reshape(-1)[dd_idx]
        Bmag_dd = Bmag[dd_idx]

        air_elems = np.where(IX[:, 3] == 1)[0]
        nu_air = nu_e[int(air_elems[0])] if air_elems.size > 0 else 1.0 / (4.0 * np.pi * 1e-7)

        mu_iron_dd = F0_Main_Mat_Nonlinear(Bmag_dd)
        nu_iron_dd = 1.0 / mu_iron_dd

        # SIMP:  nu(rho) = nu_air + (nu_iron - nu_air)*rho^p
        dnu_drho_dd = (nu_iron_dd - nu_air) * penal * (erho_dd ** (penal - 1.0))

        S_S_mat = fem["S_S"].reshape(ne, 3, 3)
        K0_dd   = S_S_mat[dd_idx]

        nodes_dd = IX[dd_idx, 0:3] - 1
        Ae_vec   = A[nodes_dd]
        lam_vec  = lam[nodes_dd]

        K0A = np.einsum('eij,ej->ei', K0_dd, Ae_vec)
        dfdrho_e[dd_idx] = dnu_drho_dd * np.sum(lam_vec * K0A, axis=1)

    # -------------------------------------------------------
    # STEP 4: Chain to nodal design variables
    # -------------------------------------------------------
    Ten = opt["Ten"]
    bt  = float(opt["bt"])
    fdv = np.asarray(opt["fdv"], dtype=float).reshape(-1)

    denom     = 2.0 * np.tanh(bt) + 1e-30
    DnrhoDfdv = _sech2(bt * fdv) * bt / denom

    dfdx_all = np.asarray(Ten.T @ dfdrho_e).reshape(-1) * DnrhoDfdv

    dof_dd = np.asarray(opt["dof_dd"], dtype=int).reshape(-1) - 1
    dfdx   = dfdx_all[dof_dd].reshape(-1, 1)

    # -------------------------------------------------------
    # Volume constraint
    # -------------------------------------------------------
    VT      = float(opt["VT"])
    VND     = float(opt["VND"])
    volfrac = float(opt["volfrac"])

    erho  = np.asarray(opt["erho"], dtype=float).reshape(-1)
    V_rho = float(np.dot(Ae_e, erho))
    g     = (V_rho - VND) / (VT - VND + 1e-30) - volfrac

    dVdrho_e = Ae_e / (VT + 1e-30)
    dgdx_all = np.asarray(Ten.T @ dVdrho_e).reshape(-1) * DnrhoDfdv
    dgdx     = dgdx_all[dof_dd].reshape(1, -1)

    print("Nonlinear Sensitivity (NR-adjoint) Computation Done. ✅")
    return f, g, dfdx, dgdx, dfdrho_e.reshape(-1, 1), lam.reshape(-1, 1), dfdA.reshape(-1, 1)
