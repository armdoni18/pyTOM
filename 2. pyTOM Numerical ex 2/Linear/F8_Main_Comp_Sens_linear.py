import numpy as np
from scipy.sparse.linalg import spsolve


def _sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / (np.cosh(x) ** 2)


def F8_Main_Comp_Sens_linear(fem, opt):

    # -----------------------
    # Objective value
    # -----------------------
    f = float(fem.get("Fy_total", 0.0))

    IX = np.asarray(fem["IX"], dtype=int)
    X  = np.asarray(fem["X"], dtype=float)
    A  = np.asarray(fem["A"], dtype=float).reshape(-1)
    Bx_all = np.asarray(fem["Bx"], dtype=float).reshape(-1)
    By_all = np.asarray(fem["By"], dtype=float).reshape(-1)
    nu_e = np.asarray(fem["nu_e"], dtype=float).reshape(-1)
    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])
    Ae_e = np.asarray(fem["Ae"], dtype=float).reshape(-1)
    penal = float(opt["penal"])

    # -----------------------
    # === Step 1: dF/dA from MST ===
    # -----------------------
    dfdA = np.zeros(ndof, dtype=float)
    edges = np.asarray(fem["cleaned_air_loop_around_plunger"], dtype=int)

    # plunger center (domain id 5)
    plunger_nodes = np.unique(IX[IX[:, 3] == 5, 0:3])
    plunger_coords = X[plunger_nodes - 1, :]
    plunger_center = np.mean(plunger_coords, axis=0)

    for k in range(edges.shape[0]):

        n1 = int(edges[k, 0])
        n2 = int(edges[k, 1])

        p1 = X[n1 - 1, :]
        p2 = X[n2 - 1, :]

        seg = p2 - p1
        ds = np.linalg.norm(seg)
        if ds <= 0.0:
            continue

        normal = np.array([seg[1], -seg[0]], dtype=float)
        normal /= (np.linalg.norm(normal) + 1e-30)

        mid = 0.5 * (p1 + p2)

        # keep normal outward from plunger
        if np.dot(normal, (plunger_center - mid)) > 0.0:
            normal = -normal

        # shifted point (toward plunger)
        eps_shift = 1e-3
        shifted_mid = mid + eps_shift * normal

        # find the closest element by centroid
        min_dist = np.inf
        closest_e = 0
        for e in range(ne):
            nodes_e = IX[e, 0:3]
            coords = X[nodes_e - 1, :]
            centroid = np.mean(coords, axis=0)
            dist = np.linalg.norm(shifted_mid - centroid)
            if dist < min_dist:
                min_dist = dist
                closest_e = e

        nodes_e = IX[closest_e, 0:3]
        nodes0  = nodes_e - 1
        Ve = Ae_e[closest_e]
        if Ve <= 0.0:
            continue

        # coords for bi, ci
        coords = X[nodes0, :]
        xi = coords[:, 0]
        yi = coords[:, 1]

        # linear triangular element coefficients
        bi = np.array([yi[1] - yi[2], yi[2] - yi[0], yi[0] - yi[1]], dtype=float)
        ci = np.array([xi[2] - xi[1], xi[0] - xi[2], xi[1] - xi[0]], dtype=float)

        Bx = Bx_all[closest_e]
        By = By_all[closest_e]

        nu_like = nu_e[closest_e]

        dBx_dA = (1.0 / (2.0 * Ve)) * ci
        dBy_dA = (-1.0 / (2.0 * Ve)) * bi

        dTxy_dA = nu_like * (By * dBx_dA + Bx * dBy_dA)
        dTyy_dA = nu_like * 0.5 * (2.0 * By * dBy_dA - 2.0 * Bx * dBx_dA)

        dFy_edge = (dTxy_dA * normal[0] + dTyy_dA * normal[1]) * ds

        dfdA[nodes0] += dFy_edge

    # -----------------------
    # === Step 2: Solve adjoint ===
    # S * lambda = dF/dA
    # -----------------------
    S = fem["S"]

    all_dofs = np.arange(ndof, dtype=int)
    fixdof = np.asarray(fem["bcdof"], dtype=int).reshape(-1) - 1
    freedof = np.setdiff1d(all_dofs, fixdof)

    lam = np.zeros(ndof, dtype=float)
    S_ff = S[freedof][:, freedof]
    rhs = dfdA[freedof]

    lam_free = spsolve(S_ff, rhs)

    lam[freedof] = lam_free
    lam[fixdof] = 0.0

    # -----------------------
    # === Step 3: df/drho_e (LINEAR) ===
    # -----------------------
    dfdrho_e = np.zeros(ne, dtype=float)

    S_S = np.asarray(fem["S_S"], dtype=float).reshape(-1)
    erho = np.asarray(opt["erho"], dtype=float).reshape(-1)

    V0 = np.min(nu_e)
    Vmin = np.max(nu_e)

    for e in range(ne):
        dom = int(IX[e, 3])

        # only design domain contributes
        if dom != 2:
            continue

        rhoe = float(erho[e])

        penal_term = penal * (V0 - Vmin) * (rhoe ** (penal - 1.0))

        K0 = S_S[e * 9:(e + 1) * 9].reshape(3, 3)

        nodes_e = IX[e, 0:3] - 1
        Ae_vec = A[nodes_e]
        lam_vec = lam[nodes_e]

        dfdrho_e[e] = lam_vec @ ((penal_term / V0) * (K0 @ Ae_vec))

    # -----------------------
    # === Step 4: chain rule to nodal design variables ===
    # -----------------------
    Ten = opt["Ten"]
    bt  = float(opt["bt"])
    fdv = np.asarray(opt["fdv"], dtype=float).reshape(-1)

    DerhoDnrho = Ten.T

    denom = 2.0 * np.tanh(bt) + 1e-30
    DnrhoDfdv = _sech2(bt * fdv) * bt / denom

    dfdx_all_nodes = np.asarray(DerhoDnrho @ dfdrho_e).reshape(-1)
    dfdx_nodes = dfdx_all_nodes * DnrhoDfdv

    dof_dd = np.asarray(opt["dof_dd"], dtype=int).reshape(-1) - 1
    dfdx = dfdx_nodes[dof_dd].reshape(-1, 1)

    # -----------------------
    # Volume constraint
    # -----------------------
    VT = float(opt["VT"])
    VND = float(opt["VND"])
    volfrac = float(opt["volfrac"])

    V_rho = float(np.dot(Ae_e, erho))
    g = (V_rho - VND) / (VT - VND + 1e-30) - volfrac

    dVdrho_e = Ae_e / (VT + 1e-30)
    dgdx_all_nodes = np.asarray(DerhoDnrho @ dVdrho_e).reshape(-1)
    dgdx_nodes = dgdx_all_nodes * DnrhoDfdv
    dgdx = dgdx_nodes[dof_dd].reshape(1, -1)

    print("Linear Sensitivity Computation Done. ✅")

    return f, g, dfdx, dgdx, dfdrho_e.reshape(-1, 1), lam.reshape(-1, 1), dfdA.reshape(-1, 1)
