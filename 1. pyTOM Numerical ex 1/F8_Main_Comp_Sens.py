import numpy as np
from scipy.sparse.linalg import spsolve

from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear


def _sech2(x: np.ndarray) -> np.ndarray:
    return 1.0 / (np.cosh(x) ** 2)


def F8_Main_Comp_Sens(fem, opt, J):

    # -----------------------
    # Objective value
    # -----------------------
    f = float(fem.get("Fy_total", 0.0))

    IX = np.asarray(fem["IX"], dtype=int)
    X  = np.asarray(fem["X"], dtype=float)
    A  = np.asarray(fem["A"], dtype=float).reshape(-1)

    Bx_all = np.asarray(fem["Bx"], dtype=float).reshape(-1)
    By_all = np.asarray(fem["By"], dtype=float).reshape(-1)
    Bmag   = np.asarray(fem["B"],  dtype=float).reshape(-1)
    nu_e   = np.asarray(fem["nu_e"], dtype=float).reshape(-1)

    ndof = int(fem["ndof"])
    ne   = int(fem["ne"])
    Ae_e = np.asarray(fem["Ae"], dtype=float).reshape(-1)

    penal = float(opt["penal"])

    # -----------------------
    # === Step 1: dF/dA (MST)
    # -----------------------
    dfdA = np.zeros(ndof)

    edges = np.asarray(fem["cleaned_air_loop_around_plunger"], dtype=int)
    centroids = fem["centroids"]

    # plunger center
    plunger_nodes = np.unique(IX[IX[:, 3] == 5, 0:3])
    plunger_center = np.mean(X[plunger_nodes - 1], axis=0)

    for k in range(edges.shape[0]):

        n1, n2 = edges[k]

        p1 = X[n1 - 1]
        p2 = X[n2 - 1]

        seg = p2 - p1
        ds  = np.linalg.norm(seg)

        if ds < 1e-12:
            continue

        normal = np.array([seg[1], -seg[0]])
        normal /= (np.linalg.norm(normal) + 1e-30)

        mid = 0.5 * (p1 + p2)

        if np.dot(normal, (plunger_center - mid)) > 0:
            normal = -normal

        shifted_mid = mid + 1e-3 * normal

        # FAST nearest element
        diff = centroids - shifted_mid
        closest_e = np.argmin(np.sum(diff**2, axis=1))

        nodes0 = IX[closest_e, 0:3] - 1
        Ve = Ae_e[closest_e]

        if Ve <= 0:
            continue

        xi = X[nodes0, 0]
        yi = X[nodes0, 1]

        bi = np.array([yi[1]-yi[2], yi[2]-yi[0], yi[0]-yi[1]])
        ci = np.array([xi[2]-xi[1], xi[0]-xi[2], xi[1]-xi[0]])

        Bx = Bx_all[closest_e]
        By = By_all[closest_e]
        nu_local = nu_e[closest_e]

        dBx_dA = ci / (2.0 * Ve)
        dBy_dA = -bi / (2.0 * Ve)

        dTxy_dA = nu_local * (By * dBx_dA + Bx * dBy_dA)
        dTyy_dA = nu_local * (By * dBy_dA - Bx * dBx_dA)

        dFy_edge = (dTxy_dA * normal[0] + dTyy_dA * normal[1]) * ds

        dfdA[nodes0] += dFy_edge

    # -----------------------
    # === Step 2: Adjoint solve
    # -----------------------
    all_dofs = np.arange(ndof)
    fixdof = np.asarray(fem["bcdof"], dtype=int) - 1
    freedof = np.setdiff1d(all_dofs, fixdof)

    lam = np.zeros(ndof)

    lam_free = spsolve(J[freedof][:, freedof], dfdA[freedof])
    lam[freedof] = lam_free

    # -----------------------
    # === Step 3: df/drho
    # -----------------------
    dfdrho_e = np.zeros(ne)

    erho = np.asarray(opt["erho"], dtype=float)
    S_S  = np.asarray(fem["S_S"], dtype=float)

    # ν_air
    air_elems = np.where(IX[:, 3] == 1)[0]
    nu_air = nu_e[air_elems[0]] if air_elems.size > 0 else np.min(nu_e)

    for e in range(ne):
        if int(IX[e, 3]) != 2:
            continue

        rhoe = erho[e]

        nu_iron = F0_Main_Mat_Nonlinear(Bmag[e])

        dnu_drho = penal * (nu_iron - nu_air) * (rhoe ** (penal - 1.0))

        K0 = S_S[e*9:(e+1)*9].reshape(3, 3)

        nodes = IX[e, 0:3] - 1

        dfdrho_e[e] = lam[nodes] @ (dnu_drho * (K0 @ A[nodes]))

    # -----------------------
    # === Step 4: chain rule
    # -----------------------
    Ten = opt["Ten"]
    bt  = float(opt["bt"])
    fdv = np.asarray(opt["fdv"], dtype=float)

    DerhoDnrho = Ten.T

    DnrhoDfdv = _sech2(bt * fdv) * bt / (2.0 * np.tanh(bt) + 1e-30)

    dfdx_nodes = (DerhoDnrho @ dfdrho_e) * DnrhoDfdv

    dof_dd = np.asarray(opt["dof_dd"], dtype=int) - 1
    dfdx = dfdx_nodes[dof_dd].reshape(-1, 1)

    # -----------------------
    # Volume constraint
    # -----------------------
    VT = float(opt["VT"])
    VND = float(opt["VND"])
    volfrac = float(opt["volfrac"])

    V_rho = np.dot(Ae_e, erho)
    g = (V_rho - VND) / (VT - VND + 1e-30) - volfrac

    dVdrho_e = Ae_e / (VT + 1e-30)

    dgdx_nodes = (DerhoDnrho @ dVdrho_e) * DnrhoDfdv
    dgdx = dgdx_nodes[dof_dd].reshape(1, -1)

    print("Nonlinear Sensitivity (NR-adjoint) Computation Done. ✅")

    return f, g, dfdx, dgdx, dfdrho_e.reshape(-1,1), lam.reshape(-1,1), dfdA.reshape(-1,1)
