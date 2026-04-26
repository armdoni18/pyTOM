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
from F7_Main_Comp_Force import F7_Main_Comp_Force
from F8_Main_Comp_Sens_linear import F8_Main_Comp_Sens_linear
from F9_Post_Process_Plot import F9_Post_Process_Plot

start_time = time.time()        # TIMER START
# ===================== PRE-PROCESSING =====================
modelname = "Example_2_Actuator_Linear"

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
inputs["iterMax"]   = 400

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
    fem     = F5_Main_Comp_Flux(fem)

    # ================================
    # FORCE & SENSITIVITY
    # ================================
    Fx_total, Fy_total, fem                 = F7_Main_Comp_Force(fem)
    f, g, dfdx, dgdx, dfdrho_e, lam, dfdA   = F8_Main_Comp_Sens_linear(fem, opt)

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

