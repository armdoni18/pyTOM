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
from F7_Main_Comp_Force   import F7_Main_Comp_Force
from F8_Main_Comp_Sens    import F8_Main_Comp_Sens
from F9_Post_Process_Plot import F9_Post_Process_Plot

start_time = time.time()

# ===================== PRE-PROCESSING =====================
modelname = "Example_2_Actuator_Linear"
Npos = 1   # static single-position problem (no plunger stroke)

inputs = {}
inputs["penal"]     = 3
inputs["initdv"]    = -0.5
# --- Volume parameters  ---
inputs["VT"]        = 97500     # bounding-box area
inputs["VND"]       = 53500     # non-design area = VT - VDD
inputs["VDD"]       = 44000     # design domain area
inputs["volfrac"]   = 0.40
inputs["mu0"]       = 4 * np.pi * 1e-7
inputs["mur_air"]   = 1
inputs["mur_coil1"] = 1
inputs["mur_coil2"] = 1
inputs["mur_iron"]  = 1500
inputs["mur_PM"]    = 1
inputs["J_am2"]     = 2500
inputs["conv"]      = 0.008
inputs["bt_init"]   = 0.1
inputs["bt_ic"]     = 1.5
inputs["bt_ns"]     = 4
inputs["bt_fn"]     = 1000
inputs["MMA"]       = 1000
inputs["rmin"]      = 20
inputs["iterMax"]   = 400
inputs["scale"]     = 100

# --- Permanent Magnet ---
inputs["PM"] = {
    "domIDs": [7],
    "Br":     [0.2],
    "theta":  [0.0]
}


# === Reluctivity inputs (nu = 1/mu) — LINEAR (constant per material) ===
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

# Pre-build domain masks (outside loop for speed)
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

    # ── Multi-position loop (Npos=1 for Example 2) ──────────────────────────
    plot_positions = [0]
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

        # ── nu_e_all (LINEAR — constant per material) ─────────────
        dom = IX[:, 3].astype(int)
        erho_vec = np.asarray(opt["erho"], dtype=float).reshape(-1)
        penal    = inputs["penal"]

        nu_e_all = np.full(ne, inputs["nu_air"])

        # Design domain (dom==2): SIMP with constant nu_iron
        dd_mask = (dom == 2)
        nu_e_all[dd_mask] = (inputs["nu_air"] +
                              (inputs["nu_iron"] - inputs["nu_air"]) *
                              erho_vec[dd_mask] ** penal)

        # Fixed iron domains (NonDesign iron = 5; FixIron = 6)
        iron_mask = (dom == 5) | (dom == 6)
        nu_e_all[iron_mask] = inputs["nu_iron"]

        # Coil domains
        nu_e_all[dom == 3] = inputs["nu_coil1"]
        nu_e_all[dom == 4] = inputs["nu_coil2"]

        # PM domains
        for pmid in pm_domIDs:
            nu_e_all[dom == pmid] = inputs["nu_PM"]

        # ── STEP 1: Linear solve (single solve, no NR) ───────────────────────
        fem = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)

        # ── STEP 2: Compute B ────────────────────────────────────────────────
        fem = F5_Main_Comp_Flux(fem)

        if j in plot_positions:
            fields_pos[j + 1] = {
                "A": fem["A"].copy(),
                "B": fem["B"].copy(),
                "IX": fem["IX"].copy()
            }

        # ── Force & Sensitivity (LINEAR) ─────────────────────────────────────
        Fx_total, Fy_total, fem = F7_Main_Comp_Force(fem)
        mst_pos[j + 1] = {
            "mst": fem["mst"].copy(),
            "IX":  fem["IX"].copy()
        }

        f, g, dfdx, dgdx, dfdrho_e, lam, dfdA = F8_Main_Comp_Sens(fem, opt, inputs)

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
