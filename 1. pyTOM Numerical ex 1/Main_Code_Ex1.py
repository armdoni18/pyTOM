import numpy as np
from scipy.sparse.linalg import spsolve
import time
from F1_Pre_Mesh_Import import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init import F2_Pre_FEM_Init
from F4_Main_Solve_VecPot import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian import F6_Main_NR_Jacobian
from F7_Main_Comp_Force import F7_Main_Comp_Force
from F9_Post_Process_Plot import F9_Post_Process_Plot
from F0_Main_Mat_Derivative import F0_Main_Mat_Derivative
from F0_Main_Mat_Nonlinear import F0_Main_Mat_Nonlinear

start_time = time.time()        # TIMER START
# ===================== PRE-PROCESSING =====================
modelname = "Example_1_IPM_Motor"

Npos = 1  # Single position (no multi-position loop)

inputs = {}

# Relative permeability
inputs["mur_air"]   = 1
inputs["mur_coil1"] = 1
inputs["mur_coil2"] = 1
inputs["mur_coil3"] = 1
inputs["mur_iron"]  = 1500
inputs["mur_PM"]    = 1

# Vacuum reluctivity
nu0 = 1.0 / (4 * np.pi * 1e-7)

# Reluctivity
inputs["nu_air"]   = nu0 / inputs["mur_air"]
inputs["nu_coil1"] = nu0 / inputs["mur_coil1"]
inputs["nu_coil2"] = nu0 / inputs["mur_coil2"]
inputs["nu_coil3"] = nu0 / inputs["mur_coil3"]
inputs["nu_iron"]  = nu0 / inputs["mur_iron"]
inputs["nu_PM"]    = nu0 / inputs["mur_PM"]

inputs["penal"]     = 3
inputs["VT"]        = 471.23
inputs["VND"]       = 362.77
inputs["VDD"]       = 108.46
inputs["J_am2"]     = 10

# === Permanent Magnets ===
inputs["PM"] = {
    "domIDs": [7, 8],
    "Br":     [0.2, 0.2],
    "theta":  [120.0, 150.0]
}

# === Load mesh and init FEM ===
mesh, IX_all    = F1_Pre_Mesh_Import(modelname, Npos=Npos)
(fem)           = F2_Pre_FEM_Init(inputs, mesh)

print("Pre-Processing Completed ✅")
elapsed_pre = time.time() - start_time
print("Elapsed time for Pre-Processing: %s" %
      time.strftime('%H:%M:%S', time.gmtime(elapsed_pre)))

# ===================== MAIN PROCESSING =====================

# Use first position
fem["IX"][:, 3] = IX_all[0][:, 3]

# =======================================================
# STEP 0: Initial linear nu_e_all (starting guess)
# =======================================================
ne = fem["ne"]
IX = fem["IX"]

nu_e_all = np.zeros(ne)
penal = inputs["penal"]

pm_domIDs = set(inputs["PM"]["domIDs"])

# Without TO: design domain (dom == 2) is treated as iron (rho = 1)
for e in range(ne):
    dom = int(IX[e, 3])

    if dom == 2:
        nu_e_all[e] = inputs["nu_iron"]
    elif dom == 5:
        nu_e_all[e] = inputs["nu_iron"]
    elif dom == 6:
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

        if dom == 2:
            # Without TO: full iron (rho = 1) in design domain
            nu_e_all[e] = F0_Main_Mat_Nonlinear(B[e])
            dnu_dB_e[e] = F0_Main_Mat_Derivative(B[e])

        elif dom in [5, 6]:
            nu_e_all[e] = F0_Main_Mat_Nonlinear(B[e])
            dnu_dB_e[e] = F0_Main_Mat_Derivative(B[e])

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

# Recompute B for final A
fem = F5_Main_Comp_Flux(fem)

print("NR loop finished.")

# ================================
# MAGNETIC FORCE CALCULATION
# ================================
Fx_total, Fy_total, fem = F7_Main_Comp_Force(fem)

print("\n========== RESULT ==========")
print(f"Fx_total = {Fx_total:.6e} N")
print(f"Fy_total = {Fy_total:.6e} N")
print("============================")

# ================================
# POST-PROCESSING PLOT
# ================================
# Build minimal fields_pos and mst_pos so F9 can plot the single position
fields_pos = {
    1: {
        "A":  fem["A"].copy(),
        "B":  fem["B"].copy(),
        "IX": fem["IX"].copy()
    }
}

mst_pos = {
    1: {
        "mst": fem["mst"].copy(),
        "IX":  fem["IX"].copy()
    }
}

# Minimal opt-like dict (no TO history, just placeholders so F9 doesn't break)
opt = {
    "iter":  1,
    "f":     [Fx_total],   # store force as the "objective" for plotting
    "g":     [0.0],
    "bt":    1.0,
    "nrho":  np.ones(ne),  # full material since no TO
    "erho":  np.ones(ne),
}

F9_Post_Process_Plot(fem, opt, fields_pos, mst_pos=mst_pos)

print("\nMain-Processing Completed ✅")
total_elapsed = time.time() - start_time
print("Total Elapsed Time: %s" %
      time.strftime('%H:%M:%S', time.gmtime(total_elapsed)))
