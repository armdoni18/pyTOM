"""
Main_code_Ex1.py
================

This is the main driver script for Numerical Example 1 (IPM motor
field validation): validation of the magnetostatic vector-
potential formulation against COMSOL on the one-quarter IPM
motor (Section 5.1 of the manuscript, Fig. 4 and Table 3).

This driver performs the same nonlinear magnetostatic solve as
the innermost solver loop of ``Main_code_Mulpos.py`` (Example 3)
but without:
  - the topology-optimization loop,
  - the multi-position loop.

For the IPM motor configuration, the script performs a single
nonlinear magnetostatic analysis (Newton-Raphson iteration on
Eqs. (4)-(6)) and computes the nodal vector potential A and the
per-element flux density (Bx, By, |B|) for comparison against
the COMSOL result presented in Table 3.

See ``3. pyTOM Numerical ex 3/Main_code_Mulpos.py`` for the
full per-step documentation of the generic workflow.
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from F1_Pre_Mesh_Import      import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init         import F2_Pre_FEM_Init
from F3_Pre_Opt_Init         import F3_Pre_Opt_Init
from F4_Main_Solve_VecPot    import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux       import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian     import F6_Main_NR_Jacobian
from F9_Post_Process_Plot    import F9_Post_Process_Plot
from F0_Main_Mat_Nonlinear   import F0_Main_Mat_Nonlinear
from F0_Main_Mat_Derivative  import F0_Main_Mat_Derivative

start_time = time.time()

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Model/run settings ---
modelname = "Example_1_IPM_Motor"
Npos = 1   # static single-position problem (no rotor stroke)

inputs = {}

# --- Optimization / SIMP parameters ---
inputs["penal"]     = 3                       # SIMP penalization exponent p (Eq. (20))
inputs["initdv"]    = 1                       # initial (unfiltered) design-variable value

# --- Volume parameters ---
inputs["VT"]        = 471.23                  # total area of the model
inputs["VND"]       = 362.77                  # non-design area  (= VT - VDD)
inputs["VDD"]       = 108.46                  # design-domain area
inputs["volfrac"]   = 0.40                    # prescribed volume fraction V* (Eq. (13))

# --- Material properties (relative permeability) ---
inputs["mu0"]       = 4 * np.pi * 1e-7        # vacuum permeability mu_0 [H/m]
inputs["mur_air"]   = 1                       # relative permeability of air
inputs["mur_coil1"] = 1                       # relative permeability of coil 1
inputs["mur_coil2"] = 1                       # relative permeability of coil 2
inputs["mur_coil3"] = 1                       # relative permeability of coil 3 (IPM motor only)
inputs["mur_iron"]  = 1500                    # relative permeability of (linear-reference) iron
inputs["mur_PM"]    = 1                       # relative permeability of the PM region

# --- Coil excitation ---
inputs["J_am2"]     = 10                      # coil current density [A/m^2]

# --- Continuation schedule (Heaviside projection sharpness beta) ---
inputs["conv"]      = 0.008                   # objective-change tolerance to trigger continuation
inputs["bt_init"]   = 0.1                     # initial Heaviside projection sharpness beta (Eq. (19))
inputs["bt_ic"]     = 1.5                     # beta increase factor per continuation step
inputs["bt_ns"]     = 4                       # number of iterations between beta increases
inputs["bt_fn"]     = 20                      # final beta value (stops the outer TO loop)

# --- Solver / MMA optimizer settings ---
inputs["MMA"]       = 1000                    # MMA c-constant
inputs["rmin"]      = 20                      # filter radius r_min (mesh length units, Eq. (18))
inputs["iterMax"]   = 1                       # maximum number of TO iterations
inputs["scale"]     = 1                       # target objective magnitude for MMA scaling

# --- Boundary condition geometry (IPM motor: circular) ---
inputs["bc_cx"]   = 0.0                       # circular-boundary center x (IPM Dirichlet detection)
inputs["bc_cy"]   = 0.0                       # circular-boundary center y (IPM Dirichlet detection)
inputs["bc_Rout"] = 25.0                      # outer radius for Dirichlet detection (IPM)
inputs["bc_Rin"]  = 5.0                       # inner radius for Dirichlet detection (IPM)
inputs["bc_tol"]  = 1e-6                      # geometric tolerance for boundary-node detection (IPM)

# --- Permanent magnet settings ---
inputs["PM"] = {
    "domIDs": [7, 8],                         # mesh domain IDs assigned to permanent magnets
    "Br":     [0.2, 0.2],                     # remanent flux density B_r [T]
    "theta":  [120.0, 150.0],                 # magnetization direction angles [deg]
}

# =====================================================================
# PRE-PROCESSING
# =====================================================================

# --- Reluctivity inputs ---
mu0                = inputs["mu0"]
inputs["nu_air"]   = 1.0 / (mu0 * inputs["mur_air"])    # reluctivity of air (nu = 1/(mu0*mur))
inputs["nu_coil1"] = 1.0 / (mu0 * inputs["mur_coil1"])  # reluctivity of coil 1
inputs["nu_coil2"] = 1.0 / (mu0 * inputs["mur_coil2"])  # reluctivity of coil 2
inputs["nu_coil3"] = 1.0 / (mu0 * inputs["mur_coil3"])  # reluctivity of coil 3 (IPM motor only)
inputs["nu_iron"]  = 1.0 / (mu0 * inputs["mur_iron"])   # linear-reference reluctivity of iron
inputs["nu_PM"]    = 1.0 / (mu0 * inputs["mur_PM"])     # reluctivity of the PM region

# --- Load mesh and initialize FEM/OPT data ---
mesh, IX_all = F1_Pre_Mesh_Import(modelname, Npos=Npos) # import mesh from GMSH (nodes, elements, boundaries)
fem          = F2_Pre_FEM_Init(inputs, mesh)            # initialize FEM model (BCs, sources, matrices)
opt, MMA     = F3_Pre_Opt_Init(inputs, fem)             # initialize optimization variables (design variable, filters, MMA)

print("Pre-Processing Completed ✅")
elapsed_pre = time.time() - start_time
print("Elapsed time for Pre-Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(elapsed_pre)))

# =====================================================================
# MAIN PROCESSING
# =====================================================================

# Pre-build domain masks
ne      = fem["ne"]
IX_base = fem["IX"]

# pm_domIDs set
pm_domIDs = set(inputs["PM"]["domIDs"])

# SINGLE RUN (no optimization)

while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

    # ── Filter + Projection ──
    opt["fdv"]  = spsolve(opt["Kft_sparse"], sp.csc_matrix.dot(opt["Tft"], opt["nv"]))
    opt["nrho"] = np.maximum(np.minimum(np.tanh(opt["bt"] * opt["fdv"]) / (2 * np.tanh(opt["bt"])) + 0.5, 1), -1)
    opt["erho"] = opt["Ten"].dot(opt["nrho"])

    # ── Position loop (Npos=1 for IPM motor validation) ──
    plot_positions = [0]
    fields_pos = {}
    f_pos      = []
    g_pos      = []
    dfdx_pos   = []
    dgdx_pos   = []
    mst_pos    = {}

    for j in range(Npos):

        # Swap domain IDs for this position
        fem["IX"][:, 3] = IX_all[j][:, 3]
        IX  = fem["IX"]

        # ── Element reluctivity: initial linear guess ──
        dom = IX[:, 3].astype(int)
        erho_vec = np.asarray(opt["erho"], dtype=float).reshape(-1)
        penal    = inputs["penal"]

        nu_e_all = np.full(ne, inputs["nu_air"])

        # Design domain (dom==2): SIMP interpolation
        dd_mask = (dom == 2)
        nu_e_all[dd_mask] = (
            inputs["nu_air"]
            + (inputs["nu_iron"] - inputs["nu_air"]) *
            erho_vec[dd_mask] ** penal
        )

        # Fixed iron domain (rotor)
        nu_e_all[dom == 5] = inputs["nu_iron"]

        # Coil domains
        nu_e_all[dom == 3] = inputs["nu_coil1"]
        nu_e_all[dom == 4] = inputs["nu_coil2"]
        nu_e_all[dom == 6] = inputs["nu_coil3"]

        # Permanent-magnet domains
        for pmid in pm_domIDs:
            nu_e_all[dom == pmid] = inputs["nu_PM"]

        # STEP 1: Linear magnetostatic solve
        fem   = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)
        A_old = fem["A"].copy()
        T_rhs = fem["T"].copy()

        print(f"\nPos {j+1}: Initial linear solve done. Starting NR...")

        all_dofs = np.arange(fem["ndof"])
        fixdof   = fem["bcdof"].astype(int) - 1
        bcval    = fem["bcval"]
        freedof  = np.setdiff1d(all_dofs, fixdof)

        A_old[fixdof] = bcval

        # ── NEWTON–RAPHSON LOOP ───────────────────────────────────────────────
        NR_max = 30
        NR_tol = 1e-5

        J_mat = None

        for iterNR in range(NR_max):
            print(f"  NR iter {iterNR + 1}")

            A_old[fixdof] = bcval
            fem["A"]      = A_old

            # STEP 2: Compute magnetic flux density B
            fem = F5_Main_Comp_Flux(fem)
            B   = fem["B"]

            # STEP 3: Update field-dependent reluctivity nu_e_all(B)
            dom_cur  = IX[:, 3].astype(int)
            dnu_dB_e = np.zeros(ne, dtype=float)

            # --- Nonlinear elements: design domain (2) + fixed iron (5, 6) ---
            # Compute mu(B) and dmu/dB for ALL elements once
            mu_all  = F0_Main_Mat_Nonlinear(B)    # (ne,)  -- mu(B) everywhere
            dmu_all = F0_Main_Mat_Derivative(B)   # (ne,)
            nu_nl   = 1.0 / mu_all                # nu(B)
            dnu_nl  = -dmu_all / (mu_all ** 2)    # dnu/dB

            # Fixed iron (rotor domain)
            fi_mask = (dom_cur == 5)
            nu_e_all[fi_mask] = nu_nl[fi_mask]
            dnu_dB_e[fi_mask] = dnu_nl[fi_mask]

            # Design domain (domain 2): SIMP with nonlinear iron
            dd2_mask = (dom_cur == 2)
            nu_e_all[dd2_mask] = (inputs["nu_air"] +
                                  (nu_nl[dd2_mask] - inputs["nu_air"]) *
                                  erho_vec[dd2_mask] ** penal)
            dnu_dB_e[dd2_mask] = dnu_nl[dd2_mask] * erho_vec[dd2_mask] ** penal

            # Air, Coils and PMs: constant reluctivity (dnu/dB = 0)
            nu_e_all[dom_cur == 1] = inputs["nu_air"]
            nu_e_all[dom_cur == 3] = inputs["nu_coil1"]
            nu_e_all[dom_cur == 4] = inputs["nu_coil2"]
            nu_e_all[dom_cur == 6] = inputs["nu_coil3"]
            for pmid in pm_domIDs:
                nu_e_all[dom_cur == pmid] = inputs["nu_PM"]

            fem["nu_e"] = nu_e_all.copy()

            # STEP 4: Assemble S and the NR Jacobian K_t (Eq. (5))
            fem, J_mat = F6_Main_NR_Jacobian(fem, nu_e_all, dnu_dB_e)

            # STEP 5: Residual of Eq. (4):  R = S(nu) A - (f + f_pm)
            R_full = fem["S"].dot(A_old) - T_rhs
            R      = R_full[freedof]

            # STEP 6: Newton-Raphson linearization (Eq. (5)):
            #   K_t[free,free] * dA[free] = -R[free]
            J_ff        = J_mat[freedof][:, freedof]
            deltaA_free = -spsolve(J_ff, R)

            deltaA          = np.zeros_like(A_old)
            deltaA[freedof] = deltaA_free

            # STEP 7: Damped Newton update — Eq. (6) with damping
            #   A^(k+1) = A^(k) + alpha * dA
            alpha = 0.2
            A_new = A_old + alpha * deltaA
            A_new[fixdof] = bcval

            # STEP 8: Convergence check
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
