"""
Main_code_Ex1.py
================

Top-level driver script for Numerical Example 1:
validation of the magnetostatic vector-potential formulation against COMSOL on the one-quarter IPM motor (Section 5.1 of the manuscript, Fig. 3 and Table 2).

This script does NOT perform topology optimization. It performs a single nonlinear magnetostatic analysis (Newton-Raphson iteration on Eqs. (4)-(6))
for the IPM motor configuration and saves the nodal vector potential A and the per-element flux density (Bx, By, |B|) for comparison against the COMSOL reference solution.

The algorithmic structure mirrors the inner solver loop of ``Main_code_Mulpos.py`` (Example 3) but without the multi-position loop and without the outer topology-optimization loop.
See the Example 3 driver for the full per-iteration documentation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time

from F1_Pre_Mesh_Import      import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init         import F2_Pre_FEM_Init
from F3_Pre_Opt_Init         import F3_Pre_Opt_Init
from F4_Main_Solve_VecPot    import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux       import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian     import F6_Main_NR_Jacobian
from F7_Main_Comp_Force      import F7_Main_Comp_Force
from F9_Post_Process_Plot    import F9_Post_Process_Plot
from F0_Main_Mat_Nonlinear   import F0_Main_Mat_Nonlinear
from F0_Main_Mat_Derivative  import F0_Main_Mat_Derivative

start_time = time.time()

# =====================================================================
# PRE-PROCESSING — IPM Motor
# =====================================================================
modelname = "Example 1 IPM Motor"
Npos = 1   # static single-position problem (no rotor stroke)

inputs = {}
inputs["penal"]     = 3
inputs["initdv"]    = 1
inputs["VT"]        = 471.23
inputs["VND"]       = 362.77
inputs["VDD"]       = 108.46
inputs["volfrac"]   = 0.40
inputs["mu0"]       = 4 * np.pi * 1e-7
inputs["mur_air"]   = 1
inputs["mur_coil1"] = 1
inputs["mur_coil2"] = 1
inputs["mur_coil3"] = 1
inputs["mur_iron"]  = 1500
inputs["mur_PM"]    = 1
inputs["J_am2"]     = 10
inputs["conv"]      = 0.008
inputs["bt_init"]   = 0.1
inputs["bt_ic"]     = 1.5
inputs["bt_ns"]     = 4
inputs["bt_fn"]     = 20
inputs["MMA"]       = 1000
inputs["rmin"]      = 20
inputs["iterMax"]   = 1
inputs["scale"]     = 1

# --- Boundary condition geometry (IPM motor: circular) ---
inputs["bc_cx"]   = 0.0
inputs["bc_cy"]   = 0.0
inputs["bc_Rout"] = 25.0
inputs["bc_Rin"]  = 5.0
inputs["bc_tol"]  = 1e-6

# --- Permanent Magnets ---
inputs["PM"] = {
    "domIDs": [7, 8],
    "Br":     [0.2, 0.2],
    "theta":  [120.0, 150.0],
}

# --- Reluctivity inputs ---
mu0 = inputs["mu0"]
inputs["nu_air"]   = 1.0 / (mu0 * inputs["mur_air"])
inputs["nu_coil1"] = 1.0 / (mu0 * inputs["mur_coil1"])
inputs["nu_coil2"] = 1.0 / (mu0 * inputs["mur_coil2"])
inputs["nu_coil3"] = 1.0 / (mu0 * inputs["mur_coil3"])
inputs["nu_iron"]  = 1.0 / (mu0 * inputs["mur_iron"])
inputs["nu_PM"]    = 1.0 / (mu0 * inputs["mur_PM"])

# --- Load mesh and init FEM/OPT ---
mesh, IX_all = F1_Pre_Mesh_Import(modelname, Npos=Npos)
fem          = F2_Pre_FEM_Init(inputs, mesh)
opt, MMA     = F3_Pre_Opt_Init(inputs, fem)

print("Pre-Processing Completed ✅")
elapsed_pre = time.time() - start_time
print("Elapsed time for Pre-Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(elapsed_pre)))

# =====================================================================
# MAIN PROCESSING
# =====================================================================

ne      = fem["ne"]
IX_base = fem["IX"]

pm_domIDs = set(inputs["PM"]["domIDs"])

# SINGLE RUN (no optimization)

while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

    # ---- Filter + Projection ----
    opt["fdv"]  = spsolve(opt["Kft_sparse"],
                          sp.csc_matrix.dot(opt["Tft"], opt["nv"]))
    opt["nrho"] = np.maximum(
        np.minimum(np.tanh(opt["bt"] * opt["fdv"]) /
                   (2 * np.tanh(opt["bt"])) + 0.5, 1), -1)
    opt["erho"] = opt["Ten"].dot(opt["nrho"])

    # ---- Multi-position loop (Npos=1 for IPM static) ----
    plot_positions = [0]
    fields_pos = {}
    f_pos      = []
    g_pos      = []
    dfdx_pos   = []
    dgdx_pos   = []
    mst_pos    = {}

    for j in range(Npos):

        # Domain ids for this "position"
        fem["IX"][:, 3] = IX_all[j][:, 3]
        IX  = fem["IX"]
        dom = IX[:, 3].astype(int)

        erho_vec = np.asarray(opt["erho"], dtype=float).reshape(-1)
        penal    = inputs["penal"]

        # ---- Initial linear nu_e_all ----
        nu_e_all = np.full(ne, inputs["nu_air"])

        dd_mask = (dom == 2)
        nu_e_all[dd_mask] = (
            inputs["nu_air"]
            + (inputs["nu_iron"] - inputs["nu_air"]) * erho_vec[dd_mask] ** penal
        )

        # NonDesign iron rotor
        nu_e_all[dom == 5] = inputs["nu_iron"]

        # Coils
        nu_e_all[dom == 3] = inputs["nu_coil1"]
        nu_e_all[dom == 4] = inputs["nu_coil2"]
        nu_e_all[dom == 6] = inputs["nu_coil3"]

        # PMs
        for pmid in pm_domIDs:
            nu_e_all[dom == pmid] = inputs["nu_PM"]

        # ---- STEP 1: Initial linear solve ----
        fem   = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)
        A_old = fem["A"].copy()
        T_rhs = fem["T"].copy()

        print(f"\nPos {j+1}: Initial linear solve done. Starting NR...")

        all_dofs = np.arange(fem["ndof"])
        fixdof   = fem["bcdof"].astype(int) - 1
        bcval    = fem["bcval"]
        freedof  = np.setdiff1d(all_dofs, fixdof)

        A_old[fixdof] = bcval

        # ---- NEWTON–RAPHSON LOOP ----
        NR_max = 30
        NR_tol = 1e-5

        J_mat = None

        for iterNR in range(NR_max):
            print(f"  NR iter {iterNR + 1}")

            A_old[fixdof] = bcval
            fem["A"]      = A_old

            fem = F5_Main_Comp_Flux(fem)
            B   = fem["B"]

            # Update nu_e_all(B)
            dom_cur  = IX[:, 3].astype(int)
            dnu_dB_e = np.zeros(ne, dtype=float)

            mu_all  = F0_Main_Mat_Nonlinear(B)
            dmu_all = F0_Main_Mat_Derivative(B)
            nu_nl   = 1.0 / mu_all
            dnu_nl  = -dmu_all / (mu_all ** 2)

            # NonDesign iron (rotor) — domain 5
            fi_mask = (dom_cur == 5)
            nu_e_all[fi_mask] = nu_nl[fi_mask]
            dnu_dB_e[fi_mask] = dnu_nl[fi_mask]

            # Design domain — SIMP with nonlinear iron
            dd2_mask = (dom_cur == 2)
            nu_e_all[dd2_mask] = (
                inputs["nu_air"]
                + (nu_nl[dd2_mask] - inputs["nu_air"]) * erho_vec[dd2_mask] ** penal
            )
            dnu_dB_e[dd2_mask] = dnu_nl[dd2_mask] * erho_vec[dd2_mask] ** penal

            # Air, coils, PMs — constant reluctivity
            nu_e_all[dom_cur == 1] = inputs["nu_air"]
            nu_e_all[dom_cur == 3] = inputs["nu_coil1"]
            nu_e_all[dom_cur == 4] = inputs["nu_coil2"]
            nu_e_all[dom_cur == 6] = inputs["nu_coil3"]
            for pmid in pm_domIDs:
                nu_e_all[dom_cur == pmid] = inputs["nu_PM"]

            fem["nu_e"] = nu_e_all.copy()

            fem, J_mat = F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e)

            R_full = fem["S"].dot(A_old) - T_rhs
            R      = R_full[freedof]

            J_ff        = J_mat[freedof][:, freedof]
            deltaA_free = -spsolve(J_ff, R)

            deltaA          = np.zeros_like(A_old)
            deltaA[freedof] = deltaA_free

            alpha = 0.2
            A_new = A_old + alpha * deltaA
            A_new[fixdof] = bcval

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
                "A":  fem["A"].copy(),
                "B":  fem["B"].copy(),
                "IX": fem["IX"].copy(),
            }

        print("NR loop finished.")

        # ---- Force & Sensitivity ----
        Fx_total, Fy_total, fem = F7_Main_Comp_Force(fem)
        mst_pos[j + 1] = {"mst": fem["mst"].copy(), "IX": fem["IX"].copy()}

    F9_Post_Process_Plot(fem, opt, fields_pos, mst_pos=mst_pos)

    elapsed_iter = time.time() - start_time
    print("Total Elapsed Time: %s" %
          time.strftime('%H:%M:%S', time.gmtime(elapsed_iter)))

    opt["iter"]    += 1

print('finish')

print("Main-Processing Completed ✅")
total_elapsed = time.time() - start_time
print("Elapsed time for Main Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(total_elapsed)))
