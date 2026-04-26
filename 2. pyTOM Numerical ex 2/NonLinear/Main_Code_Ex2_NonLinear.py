import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mma import mmasub
import time
from F1_Pre_Mesh_Import import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init import F2_Pre_FEM_Init
from F3_Pre_Opt_Init import F3_Pre_Opt_Init
from F4_Main_Solve_VecPot import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian import F6_Main_NR_Jacobian
from F7_Main_Comp_Force import F7_Main_Comp_Force
from F8_Main_Comp_Sens import F8_Main_Comp_Sens
from F9_Post_Process_Plot import F9_Post_Process_Plot
from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear
from F0_Main_Mat_Derivative import F0_Main_Mat_Derivative

start_time = time.time()        # TIMER START
# ===================== PRE-PROCESSING =====================
modelname = "Example_2_Actuator_NonLinear"

inputs = {}
inputs["penal"]     = 3
inputs["initdv"]    = 0.1
inputs["VT"]        = 97500
inputs["VND"]       = 53500
inputs["VDD"]       = 44000
inputs["volfrac"]   = 0.40
inputs["mu0"]       = 4 * np.pi * 1e-7
inputs["mur_air"]   = 1
inputs["mur_coil1"] = 1
inputs["mur_coil2"] = 1
inputs["mur_iron"]  = 1500
inputs["mur_PM"]    = 1
inputs["J_am2"]     = 1250
inputs["conv"]      = 0.008
inputs["bt_init"]   = 0.1
inputs["bt_ic"]     = 1.5
inputs["bt_ns"]     = 4
inputs["bt_fn"]     = 100
inputs["MMA_c"]     = 10000000
inputs["rmin"]      = 7.5
inputs["iterMax"]   = 3

# === Permanent Magnets ===
inputs["PM"] = {
    "domIDs": [7],
    "Br":     [0.2],
    "theta":  [0.0]
}

mu0      = inputs["mu0"]
mu_air   = mu0 * inputs["mur_air"]
mu_coil1 = mu0 * inputs["mur_coil1"]
mu_coil2 = mu0 * inputs["mur_coil2"]
mu_iron0 = mu0 * inputs["mur_iron"]
mu_PM    = mu0 * inputs["mur_PM"]
inputs["nu_air"]   = 1.0 / mu_air
inputs["nu_coil1"] = 1.0 / mu_coil1
inputs["nu_coil2"] = 1.0 / mu_coil2
inputs["nu_iron"]  = 1.0 / mu_iron0
inputs["nu_PM"]    = 1.0 / mu_PM

# === Load mesh and init FEM/OPT ===
(mesh)      = F1_Pre_Mesh_Import (modelname)
(fem)       = F2_Pre_FEM_Init(inputs, mesh)
(opt, MMA)  = F3_Pre_Opt_Init(inputs, fem)

print("Pre-Processing Completed ✅")
elapsed_pre = time.time() - start_time
print("Elapsed time for Pre-Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(elapsed_pre)))

# ===================== MAIN PROCESSING =====================
while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

    # -------------------------------------------------------
    # Filter + Projection
    # -------------------------------------------------------
    opt["fdv"] = spsolve(opt["Kft_sparse"],(sp.csc_matrix.dot(opt["Tft"],opt["nv"])))
    opt["nrho"] = np.maximum(np.minimum(np.tanh(np.dot(opt["bt"],opt["fdv"])) / (2 * np.tanh(opt["bt"])) + 0.5, 1), -1)
    opt["erho"] = opt["Ten"].dot(opt["nrho"])

    # =======================================================
    # STEP 0: Initial linear nu_e_all (starting guess)
    # =======================================================
    ne = fem["ne"]
    IX = fem["IX"]

    nu_e_all = np.zeros(ne)
    penal = inputs["penal"]

    pm_domIDs = set(inputs["PM"]["domIDs"])

    for e in range(ne):
        dom = int(IX[e, 3])

        if dom == 2:    # design domain
            rhoe = opt["erho"][e]
            nu_e_all[e] = inputs["nu_air"] + \
                          (inputs["nu_iron"] - inputs["nu_air"]) * rhoe ** penal

        elif dom == 5:  # non-design iron
            nu_e_all[e] = inputs["nu_iron"]
        elif dom == 6:  # fixed iron
            nu_e_all[e] = inputs["nu_iron"]
        elif dom == 3:
            nu_e_all[e] = inputs["nu_coil1"]
        elif dom == 4:
            nu_e_all[e] = inputs["nu_coil2"]
        elif dom in pm_domIDs:
            nu_e_all[e] = inputs["nu_PM"]
        else:
            nu_e_all[e] = inputs["nu_air"]

    # =======================================================
    # STEP 1: Initial linear solve (A_init, T)
    # =======================================================

    fem     = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)
    A_old   = fem["A"].copy()
    T_rhs   = fem["T"].copy()

    print("\nInitial linear solve completed. Starting NR loop...")

    # =======================================================
    # Setup BC DOFs
    # =======================================================

    all_dofs    = np.arange(fem["ndof"])
    fixdof      = fem["bcdof"].astype(int) - 1
    bcval       = fem["bcval"]
    freedof     = np.setdiff1d(all_dofs, fixdof)

    A_old[fixdof] = bcval

    # =======================================================
    # NEWTON–RAPHSON LOOP
    # =======================================================

    NR_max = 30
    NR_tol = 1e-5

    for iterNR in range(NR_max):

        print(f"  NR iter {iterNR + 1}")

        # -------------------------------------------
        # Enforce BC then compute B from the A_old
        # -------------------------------------------

        A_old[fixdof]   = bcval
        fem["A"]        = A_old

        # -------------------------------------------
        # STEP 2: Compute B
        # -------------------------------------------
        fem = F5_Main_Comp_Flux(fem)
        B   = fem["B"]

        # -------------------------------------------
        # STEP 3: Update nu_e_all(B)
        # -------------------------------------------
        dnu_dB_e = np.zeros(ne)

        for e in range(ne):
            dom = int(IX[e, 3])

            if dom == 2:  # DESIGN DOMAIN
                rhoe = opt["erho"][e]

                mu_iron = F0_Main_Mat_Nonlinear(B[e])
                dmu_iron = F0_Main_Mat_Derivative(B[e])

                nu_iron = 1.0 / mu_iron
                dnu_iron = -dmu_iron / (mu_iron ** 2)

                nu_e_all[e] = inputs["nu_air"] + \
                              (nu_iron - inputs["nu_air"]) * rhoe ** penal

                dnu_dB_e[e] = dnu_iron * rhoe ** penal

            elif dom in [5, 6]:  # NON-DESIGN IRON
                mu_iron = F0_Main_Mat_Nonlinear(B[e])
                dmu_iron = F0_Main_Mat_Derivative(B[e])

                nu_e_all[e] = 1.0 / mu_iron
                dnu_dB_e[e] = -dmu_iron / (mu_iron ** 2)

            elif dom in [3, 4]:
                nu_e_all[e] = inputs["nu_coil1"]
                dnu_dB_e[e] = 0

            elif dom in pm_domIDs:
                nu_e_all[e] = inputs["nu_PM"]
                dnu_dB_e[e] = 0.0

            else:
                nu_e_all[e] = inputs["nu_air"]
                dnu_dB_e[e] = 0

        # -------------------------------------------
        # STEP 4: Assemble S and J
        # -------------------------------------------
        fem, J = F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e)

        # -------------------------------------------
        # STEP 5: Residual
        # -------------------------------------------
        R_full  = fem["S"].dot(A_old) - T_rhs
        R       = R_full[freedof]

        # -------------------------------------------
        # STEP 6: Newton update
        # -------------------------------------------
        J_ff        = J[freedof][:, freedof]
        deltaA_free = -spsolve(J_ff, R)

        deltaA          = np.zeros_like(A_old)
        deltaA[freedof] = deltaA_free

        alpha           = 0.2
        A_new           = A_old + alpha * deltaA
        A_new[fixdof]   = bcval

        # -------------------------------------------
        # STEP 7: Convergence check
        # -------------------------------------------
        errA = np.linalg.norm(deltaA[freedof]) / \
               (np.linalg.norm(A_new[freedof]) + 1e-12)

        print(f"     ||ΔA||/||A|| = {errA:.3e}")

        if errA < NR_tol:
            print("NR converged.")
            A_old = A_new
            break

        A_old = A_new

    # Store final nonlinear solution
    fem["A"] = A_old

    print("NR loop finished.")

    # ================================
    # FORCE & SENSITIVITY
    # ================================
    Fx_total, Fy_total, fem                 = F7_Main_Comp_Force(fem)
    f, g, dfdx, dgdx, dfdrho_e, lam, dfdA   = F8_Main_Comp_Sens(fem, opt, J)

    opt["f"].append(float(f))
    opt["g"].append(float(g))
    opt["dfdx"] = dfdx
    opt["dgdx"] = dgdx

    if opt["iter"] > 1:
        opt["deltaf"] = np.abs((opt["f"][-1] - opt["f"][-2]) / opt["f"][-2])

    print("iter : %3d\tf : %.4f\tVolume : %.4f\tdeltaf : %.5f\tbeta : %.2f" % (opt["iter"], opt["f"][-1], opt["g"][-1] + inputs["volfrac"], opt["deltaf"], opt["bt"]))
    F9_Post_Process_Plot(fem, opt)

    # ---- Aggregate elapsed time since start ----
    elapsed_iter = time.time() - start_time
    print("Total Elapsed Time: %s" %
          time.strftime('%H:%M:%S', time.gmtime(elapsed_iter)))

    (opt["dvnew"], ymma, zmma, lam, xsi, eta, mu, zet, s, MMA["low"], MMA["upp"]) = mmasub(1, len(opt["dv"]),
                                                                                      opt["iter"],
                                                                                           opt["dv"], opt["dvmin"],
                                                                                           opt["dvmax"], opt["dvold"],
                                                                                           opt["dvolder"],
                                                                                           opt["f"][-1], opt["dfdx"],
                                                                                           opt["g"][-1], opt["dgdx"],
                                                                                           MMA["low"], MMA["upp"],
                                                                                           MMA["a0"], MMA["a"],
                                                                                           MMA["c"], MMA["d"], 1)

    opt["iter"] += 1
    opt["dvolder"] = opt["dvold"]
    opt["dvold"] = opt["dv"]
    opt["dv"] = opt["dvnew"]
    opt["nv"][opt["dof_dd"] - 1] = opt["dv"]

    # =======================================================
    # Continuation
    # =======================================================
    if (opt["cont_sw"] == 0) and (opt["deltaf"] < inputs["conv"]):
        opt["cont_sw"] = 1
        opt["cont_iter"] = 0
        print("Continuation start")
    elif (opt["cont_sw"] == 1):
        opt["cont_iter"] += 1
        if (np.mod(opt["cont_iter"], inputs["bt_ns"]) == 1):
            opt["bt"] *= inputs["bt_ic"]

print('finish')

print("Main-Processing Completed ✅")
total_elapsed = time.time() - start_time
print("Elapsed time for Main Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(total_elapsed)))
