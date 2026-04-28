import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mma import mmasub
import time
from F1_Pre_Mesh_Import   import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init      import F2_Pre_FEM_Init
from F3_Pre_Opt_Init      import F3_Pre_Opt_Init
from F4_Main_Solve_VecPot import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux    import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian  import F6_Main_NR_Jacobian
from F7_Main_Comp_Force   import F7_Main_Comp_Force
from F8_Main_Comp_Sens    import F8_Main_Comp_Sens
from F9_Post_Process_Plot import F9_Post_Process_Plot
from F0_Main_Mat_Nonlinear  import F0_Main_Mat_Nonlinear
from F0_Main_Mat_Derivative import F0_Main_Mat_Derivative

start_time = time.time()

# ===================== PRE-PROCESSING =====================
modelname = "Example_3_Actuator_MultiPos"
Npos = 1

inputs = {}
inputs["penal"]     = 3
inputs["initdv"]    = -0.5
inputs["VT"]        = 16800
inputs["VND"]       = 13200
inputs["VDD"]       = 3600
inputs["volfrac"]   = 0.30
inputs["mu0"]       = 4 * np.pi * 1e-7
inputs["mur_air"]   = 1
inputs["mur_coil1"] = 1
inputs["mur_coil2"] = 1
inputs["mur_iron"]  = 1500
inputs["mur_PM"]    = 1
inputs["J_am2"]     = 17900
inputs["conv"]      = 0.008
inputs["bt_init"]   = 0.1
inputs["bt_ic"]     = 1.5
inputs["bt_ns"]     = 4
inputs["bt_fn"]     = 20
inputs["MMA"]       = 1000
inputs["rmin"]      = 10
inputs["iterMax"]   = 400
inputs["scale"]     = 1000

inputs["PM"] = {
    "domIDs": [7],
    "Br":     [0.2],
    "theta":  [180.0]
}

# === Reluctivity inputs (nu = 1/mu) — consistent with manuscript ===
mu0      = inputs["mu0"]
inputs["nu_air"]   = 1.0 / (mu0 * inputs["mur_air"])
inputs["nu_coil1"] = 1.0 / (mu0 * inputs["mur_coil1"])
inputs["nu_coil2"] = 1.0 / (mu0 * inputs["mur_coil2"])
inputs["nu_iron"]  = 1.0 / (mu0 * inputs["mur_iron"])
inputs["nu_PM"]    = 1.0 / (mu0 * inputs["mur_PM"])

# === Load mesh and init FEM/OPT ===
mesh, IX_all = F1_Pre_Mesh_Import(modelname, Npos=Npos)
fem          = F2_Pre_FEM_Init(inputs, mesh)
opt, MMA     = F3_Pre_Opt_Init(inputs, fem)

MMA["c_input"] = inputs["MMA"]
# ===================== SCALING =====================
opt["mma_scale_initialized"] = False
opt["mma_obj_scale"] = 1.0

print("Pre-Processing Completed ✅")
elapsed_pre = time.time() - start_time
print("Elapsed time for Pre-Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(elapsed_pre)))

# ===================== MAIN PROCESSING =====================
force_profile_final = None
saved_iter          = -1

# Pre-build domain masks (vectorized, outside loop for speed)
ne = fem["ne"]
IX_base = fem["IX"]

# pm_domIDs set
pm_domIDs = set(inputs["PM"]["domIDs"])

while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

    # ── Filter + Projection ──────────────────────────────────────────────────
    opt["fdv"]  = spsolve(opt["Kft_sparse"],
                          sp.csc_matrix.dot(opt["Tft"], opt["nv"]))
    opt["nrho"] = np.maximum(
        np.minimum(
            np.tanh(opt["bt"] * opt["fdv"]) / (2 * np.tanh(opt["bt"])) + 0.5,
            1), -1)
    opt["erho"] = opt["Ten"].dot(opt["nrho"])

    # ── Multi-position loop ──────────────────────────────────────────────────
    plot_positions = [0, 15, 30]
    fields_pos  = {}
    f_pos       = []
    g_pos       = []
    dfdx_pos    = []
    dgdx_pos    = []
    mst_pos     = {}

    for j in range(Npos):

        # Swap domain IDs for this plunger position
        fem["IX"][:, 3] = IX_all[j][:, 3]
        IX = fem["IX"]

        # ── VECTORIZED nu_e_all initial guess ────────────────────────────────
        dom = IX[:, 3].astype(int)
        erho_vec = np.asarray(opt["erho"], dtype=float).reshape(-1)
        penal    = inputs["penal"]

        nu_e_all = np.full(ne, inputs["nu_air"])

        # Design domain (dom==2): SIMP interpolation
        dd_mask = (dom == 2)
        nu_e_all[dd_mask] = (inputs["nu_air"] +
                              (inputs["nu_iron"] - inputs["nu_air"]) *
                              erho_vec[dd_mask] ** penal)

        # Fixed iron domains
        iron_mask = (dom == 5) | (dom == 6)
        nu_e_all[iron_mask] = inputs["nu_iron"]

        # Coil domains
        nu_e_all[dom == 3] = inputs["nu_coil1"]
        nu_e_all[dom == 4] = inputs["nu_coil2"]

        # PM domains
        for pmid in pm_domIDs:
            nu_e_all[dom == pmid] = inputs["nu_PM"]

        # ── STEP 1: Initial linear solve ─────────────────────────────────────
        fem    = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)
        A_old  = fem["A"].copy()
        T_rhs  = fem["T"].copy()

        print(f"\nPos {j+1}: Initial linear solve done. Starting NR...")

        all_dofs = np.arange(fem["ndof"])
        fixdof   = fem["bcdof"].astype(int) - 1
        bcval    = fem["bcval"]
        freedof  = np.setdiff1d(all_dofs, fixdof)

        A_old[fixdof] = bcval

        # ── NEWTON–RAPHSON LOOP ───────────────────────────────────────────────
        NR_max = 30
        NR_tol = 1e-5

        for iterNR in range(NR_max):
            print(f"  NR iter {iterNR + 1}")

            A_old[fixdof] = bcval
            fem["A"]      = A_old

            # STEP 2: Compute B
            fem = F5_Main_Comp_Flux(fem)
            B   = fem["B"]

            # STEP 3: Update nu_e_all(B) — FULLY VECTORIZED ──────────────────
            dom_cur  = IX[:, 3].astype(int)
            dnu_dB_e = np.zeros(ne, dtype=float)

            # --- Nonlinear elements: design domain (2) + fixed iron (5, 6) ---
            # Compute mu(B) and dmu/dB for ALL elements once
            mu_all  = F0_Main_Mat_Nonlinear(B)    # (ne,)  -- mu(B) everywhere
            dmu_all = F0_Main_Mat_Derivative(B)   # (ne,)
            nu_nl   = 1.0 / mu_all                # nu(B)
            dnu_nl  = -dmu_all / (mu_all ** 2)    # dnu/dB

            # Fixed iron (domains 5, 6)
            fi_mask = (dom_cur == 5) | (dom_cur == 6)
            nu_e_all[fi_mask]  = nu_nl[fi_mask]
            dnu_dB_e[fi_mask]  = dnu_nl[fi_mask]

            # Design domain (domain 2): SIMP with nonlinear iron
            dd2_mask = (dom_cur == 2)
            nu_e_all[dd2_mask] = (inputs["nu_air"] +
                                   (nu_nl[dd2_mask] - inputs["nu_air"]) *
                                   erho_vec[dd2_mask] ** penal)
            dnu_dB_e[dd2_mask] = dnu_nl[dd2_mask] * erho_vec[dd2_mask] ** penal

            # Coil and PM: constant reluctivity, dnu/dB = 0
            nu_e_all[dom_cur == 3] = inputs["nu_coil1"]
            nu_e_all[dom_cur == 4] = inputs["nu_coil2"]
            for pmid in pm_domIDs:
                nu_e_all[dom_cur == pmid] = inputs["nu_PM"]

            fem["nu_e"] = nu_e_all.copy()

            # STEP 4: Assemble S and J
            fem, J_mat = F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e)

            # STEP 5: Residual
            R_full = fem["S"].dot(A_old) - T_rhs
            R      = R_full[freedof]

            # STEP 6: Newton update
            J_ff       = J_mat[freedof][:, freedof]
            deltaA_free = -spsolve(J_ff, R)

            deltaA          = np.zeros_like(A_old)
            deltaA[freedof] = deltaA_free

            alpha  = 0.2
            A_new  = A_old + alpha * deltaA
            A_new[fixdof] = bcval

            # STEP 7: Convergence check
            errA = (np.linalg.norm(deltaA[freedof]) /
                    (np.linalg.norm(A_new[freedof]) + 1e-12))
            print(f"     ||ΔA||/||A|| = {errA:.3e}")

            A_old = A_new
            if errA < NR_tol:
                print("NR converged.")
                break

        fem["A"] = A_old
        fem      = F5_Main_Comp_Flux(fem)

        if j in plot_positions:
            fields_pos[j + 1] = {
                "A": fem["A"].copy(),
                "B": fem["B"].copy(),
                "IX": fem["IX"].copy()
            }

        print("NR loop finished.")

        # ── Force & Sensitivity ──────────────────────────────────────────────
        Fx_total, Fy_total, fem = F7_Main_Comp_Force(fem)
        mst_pos[j + 1] = {
            "mst": fem["mst"].copy(),
            "IX":  fem["IX"].copy()
        }

        f, g, dfdx, dgdx, dfdrho_e, lam, dfdA = F8_Main_Comp_Sens(fem, opt, J_mat)

        f_pos.append(float(f))
        g_pos.append(float(g))
        dfdx_pos.append(np.asarray(dfdx).reshape(-1, 1))
        dgdx_pos.append(np.asarray(dgdx).reshape(1, -1))

    # ── Averaging across positions ───────────────────────────────────────────
    if ((opt["iter"] == inputs["iterMax"]) or
            (opt["deltaf"] < inputs["conv"])) and (opt["iter"] > saved_iter):
        force_profile_final = f_pos.copy()
        saved_iter = opt["iter"]

    f_avg = float(np.mean(f_pos))
    g_avg = float(np.mean(g_pos))
    dfdx_avg = np.mean(np.hstack(dfdx_pos), axis=1, keepdims=True)
    dgdx_avg = np.mean(np.vstack(dgdx_pos), axis=0, keepdims=True)

    # ===================== SCALING =====================
    if not opt["mma_scale_initialized"]:
        scale_factor = inputs["scale"] / (abs(f_avg) + 1e-12)
        opt["scale_factor"] = scale_factor
        MMA["c"] = MMA["c_input"] / scale_factor
        opt["mma_scale_initialized"] = True
    f_scaled = f_avg * opt["scale_factor"]
    dfdx_scaled = dfdx_avg * opt["scale_factor"]

    opt["f"].append(f_scaled)
    opt["g"].append(g_avg)
    opt["dfdx"] = dfdx_scaled
    opt["dgdx"] = dgdx_avg

    if opt["iter"] > 1:
        opt["deltaf"] = np.abs(
            (opt["f"][-1] - opt["f"][-2]) / (opt["f"][-2] + 1e-30))

    f_scaled = opt["f"][-1]
    print("iter : %3d\tf : %.4f\tVolume : %.4f\tdeltaf : %.5f\tbeta : %.2f" %
          (opt["iter"], f_scaled,
           opt["g"][-1] + inputs["volfrac"],
           opt["deltaf"], opt["bt"]))

    if force_profile_final is not None:
        opt["force_profile"] = force_profile_final

    F9_Post_Process_Plot(fem, opt, fields_pos, mst_pos=mst_pos)

    elapsed_iter = time.time() - start_time
    print("Total Elapsed Time: %s" %
          time.strftime('%H:%M:%S', time.gmtime(elapsed_iter)))

    f_mma = opt["f"][-1]
    dfdx_mma = opt["dfdx"]

    (opt["dvnew"], ymma, zmma, lam_mma, xsi, eta, mu_mma, zet, s,
     MMA["low"], MMA["upp"]) = mmasub(
        1, len(opt["dv"]), opt["iter"],
        opt["dv"], opt["dvmin"], opt["dvmax"],
        opt["dvold"], opt["dvolder"],
        f_mma, dfdx_mma,
        opt["g"][-1], opt["dgdx"],
        MMA["low"], MMA["upp"],
        MMA["a0"], MMA["a"], MMA["c"], MMA["d"], 1)

    opt["iter"]    += 1
    opt["dvolder"]  = opt["dvold"]
    opt["dvold"]    = opt["dv"]
    opt["dv"]       = opt["dvnew"]
    opt["nv"][opt["dof_dd"] - 1] = opt["dv"]

    # ── Continuation ─────────────────────────────────────────────────────────
    if (opt["cont_sw"] == 0) and (opt["deltaf"] < inputs["conv"]):
        opt["cont_sw"]   = 1
        opt["cont_iter"] = 0
        print("Continuation start")
    elif opt["cont_sw"] == 1:
        opt["cont_iter"] += 1
        if np.mod(opt["cont_iter"], inputs["bt_ns"]) == 1:
            opt["bt"] *= inputs["bt_ic"]

print('finish')

print("Main-Processing Completed ✅")
total_elapsed = time.time() - start_time
print("Elapsed time for Main Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(total_elapsed)))
