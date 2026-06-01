"""
Main_code_Mulpos.py
===================

main driver script for Numerical Example 3: topology
optimization of a magnetic actuator with multi-position evaluation
of the magnetic force (Section 5.3 of the manuscript, Figs. 7-9).

This is the sole driver of Example 3. Because the example compares
the optimized design across several plunger-stroke discretizations
(N_pos = 1, 11, 21), the topology-optimization procedure is executed
separately for each entry of ``NPOS_LIST`` within a single Python session.
Each run writes to its own ``Results/Npos_<n>/`` folder, and the
comparative force-profile plot ``Comparison_Force_Profile.png``
(Fig. 9) is produced at the end by ``plot_comparison``.

The body of ``Main_code_Mulpos`` performs the same outer
topology-optimization loop and the same Newton-Raphson inner solve
as ``Main_code_Ex2_Nonlinear.py`` (Example 2, nonlinear case); the
only structural addition is the multi-position loop over the
N_pos plunger configurations, whose per-position forces and
sensitivities are averaged according to Eqs. (15) and (22) before
each MMA design update.

Three nested loops are orchestrated for each run:

  1. Outer topology-optimization loop (driven by the MMA optimizer
     and by the continuation strategy on the projection sharpness
     beta of Eq. (19)).
  2. Multi-position loop over N_pos plunger configurations
     (the design variable is shared across positions).
  3. Innermost Newton-Raphson loop on the nonlinear magnetostatic
     problem of Eqs. (4)-(6) for each plunger position.

The implementation follows the workflow of Section 4.1 and
Fig. 2 of the manuscript. See Appendix A for per-module details.

User settings
-------------
    NPOS_LIST   : list of int   N_pos values to sweep (Figs. 8, 9)
    RESULTS_DIR : str           root output folder
    INPUTS_BASE : dict          base input dictionary (penalization,
                                volume fraction, MMA parameters,
                                continuation schedule, PM data, ...)
    STYLE_MAP   : dict          per-N_pos plot styles
"""

import os
import copy
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mma import mmasub
from F1_Pre_Mesh_Import     import F1_Pre_Mesh_Import
from F2_Pre_FEM_Init        import F2_Pre_FEM_Init
from F3_Pre_Opt_Init        import F3_Pre_Opt_Init
from F4_Main_Solve_VecPot   import F4_Main_Solve_VecPot
from F5_Main_Comp_Flux      import F5_Main_Comp_Flux
from F6_Main_NR_Jacobian    import F6_Main_NR_Jacobian
from F7_Main_Comp_Force     import F7_Main_Comp_Force
from F8_Main_Comp_Sens      import F8_Main_Comp_Sens
from F9_Post_Process_Plot   import F9_Post_Process_Plot
from F0_Main_Mat_Nonlinear  import F0_Main_Mat_Nonlinear
from F0_Main_Mat_Derivative import F0_Main_Mat_Derivative


# ===================== USER SETTINGS =====================
MODELNAME = "Example_3_Actuator_MultiPos"   # Gmsh model name (without .msh)

# Plunger-position counts to sweep (Figs. 8, 9)
NPOS_LIST   = [1, 11, 21]

# Output root; each N_pos run writes to Results/Npos_<n>/
RESULTS_DIR = "Results"

# --- Base input dictionary (deep-copied per run so runs stay independent) ---
INPUTS_BASE = {

    # --- Optimization / SIMP parameters ---
    "penal"    : 3,                           # SIMP penalization exponent p (Eq. (20))
    "initdv"   : -0.5,                        # initial (unfiltered) design-variable value

    # --- Volume parameters ---
    "VT"       : 16800,                       # total bounding-box area of the model
    "VND"      : 13200,                       # non-design area  (= VT - VDD)
    "VDD"      : 3600,                        # design-domain area
    "volfrac"  : 0.30,                        # prescribed volume fraction V* (Eq. (13))

    # --- Material properties (relative permeability) ---
    "mu0"      : 4 * np.pi * 1e-7,            # vacuum permeability mu_0 [H/m]
    "mur_air"  : 1,                           # relative permeability of air
    "mur_coil1": 1,                           # relative permeability of coil 1
    "mur_coil2": 1,                           # relative permeability of coil 2
    "mur_iron" : 1500,                        # relative permeability of (linear-reference) iron
    "mur_PM"   : 1,                           # relative permeability of the PM region

    # --- Coil excitation ---
    "J_am2"    : 17900,                       # coil current density [A/m^2]

    # --- Continuation schedule (Heaviside projection sharpness beta) ---
    "conv"     : 0.008,                       # objective-change tolerance to trigger continuation
    "bt_init"  : 0.1,                         # initial Heaviside projection sharpness beta (Eq. (19))
    "bt_ic"    : 1.5,                         # beta increase factor per continuation step
    "bt_ns"    : 4,                           # number of iterations between beta increases
    "bt_fn"    : 20,                          # final beta value (stops the outer TO loop)

    # --- Solver / MMA optimizer settings ---
    "MMA"      : 1000,                        # MMA c-constant (before objective scaling)
    "rmin"     : 10,                          # filter radius r_min (mesh length units, Eq. (18))
    "iterMax"  : 400,                         # maximum number of TO iterations
    "scale"    : 1000,                        # target objective magnitude for MMA scaling
    "PM"       : {
        "domIDs": [7],
        "Br"    : [0.2],
        "theta" : [180.0]
    }
}

# Per-N_pos plot styles for the comparison figure
STYLE_MAP = {
    1 : {"color": "#378ADD", "linestyle": "-",    "marker": "o", "label": r"$N_{pos}=1$"},
    3 : {"color": "#1D9E75", "linestyle": "--",   "marker": "s", "label": r"$N_{pos}=3$"},
    5 : {"color": "#D85A30", "linestyle": "-.",   "marker": "^", "label": r"$N_{pos}=5$"},
    7 : {"color": "#7F77DD", "linestyle": ":",    "marker": "D", "label": r"$N_{pos}=7$"},
    11: {"color": "#BA7517", "linestyle": (0, (5, 2, 1, 2)), "marker": "v", "label": r"$N_{pos}=11$"},
}

_FALLBACK_COLORS = ["#E24B4A", "#5DCAA5", "#D4537E", "#639922", "#EF9F27"]


# ===================== PER-N_pos DRIVER =====================

def run_single_npos(Npos, inputs_base, modelname, out_dir):
    """Run the full topology-optimization driver for one N_pos value.

    Mirrors ``Main_code_Ex2_Nonlinear.py`` with the multi-position
    loop activated (Npos plunger configurations per TO iteration).
    Returns the converged per-position force profile used by
    ``plot_comparison`` to assemble Fig. 9.
    """

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)

    start_time = time.time()
    print("\n" + "=" * 60)
    print(f"  START RUN  Npos = {Npos}")
    print(f"  Output    : {out_dir}")
    print("=" * 60)

    # ===================== PRE-PROCESSING =====================
    # Independent copy so concurrent/sequential runs do not share state
    inputs = copy.deepcopy(inputs_base)

    # === Reluctivity inputs (nu = 1/mu) — consistent with manuscript ===
    mu0 = inputs["mu0"]
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

    print(f"Pre-Processing Completed ✅  "
          f"({time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))})")

    # ===================== MAIN PROCESSING =====================
    force_profile_final = None
    saved_iter          = -1

    # Pre-build domain masks (outside loop for speed)
    ne = fem["ne"]
    IX_base = fem["IX"]

    # pm_domIDs set
    pm_domIDs = set(inputs["PM"]["domIDs"])

    # Positions saved for field plotting (does not affect the optimization)
    if Npos == 1:
        plot_positions = [0]
    elif Npos <= 3:
        plot_positions = list(range(Npos))
    else:
        step = max(1, (Npos - 1) // 2)
        plot_positions = [0, step, Npos - 1]

    while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

        # ── Filter + Projection ──────────────────────────────────────────────────
        # Eq. (18): Helmholtz filter via cached LU back-substitution.
        # The filtered nodal field opt["fdv"] is then projected by the
        # regularized Heaviside (Eq. (19)) and clipped to [-1, 1]
        # (numerical safeguard; see F3 docstring for the rationale).
        opt["fdv"]  = spsolve(opt["Kft_sparse"],
                              sp.csc_matrix.dot(opt["Tft"], opt["nv"]))
        opt["nrho"] = np.maximum(
            np.minimum(
                np.tanh(opt["bt"] * opt["fdv"]) / (2 * np.tanh(opt["bt"])) + 0.5,
                1), -1)
        opt["erho"] = opt["Ten"].dot(opt["nrho"])

        # ── Multi-position loop ──────────────────────────────────────────────────
        fields_pos  = {}
        f_pos       = []
        g_pos       = []
        dfdx_pos    = []
        dgdx_pos    = []
        mst_pos     = {}

        for j in range(Npos):

            # Swap domain IDs for this plunger position (domain-mapping strategy)
            fem["IX"][:, 3] = IX_all[j][:, 3]
            IX = fem["IX"]

            # ── nu_e_all initial guess ────────────────────────────────
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

            print(f"\nPos {j+1}/{Npos}: Initial linear solve done. Starting NR...")

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

                # STEP 3: Update nu_e_all(B) ──────────────────
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
                # alpha = 0.2 improves convergence robustness
                # in saturated regions during optimization.
                alpha  = 0.2
                A_new  = A_old + alpha * deltaA
                A_new[fixdof] = bcval

                # Convergence check
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

        # ── Averaging across positions (Eqs. (15), (22)) ────────────────────────
        # F_avg = (1/N_pos) * sum_i F^i and similarly for dF/dphi.
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
        print("iter : %3d\tf : %.4f\tVolume : %.4f\tdeltaf : %.5f\tbeta : %.2f\t[Npos=%d]" %
              (opt["iter"], f_scaled,
               opt["g"][-1] + inputs["volfrac"],
               opt["deltaf"], opt["bt"], Npos))

        if force_profile_final is not None:
            opt["force_profile"] = force_profile_final

        F9_Post_Process_Plot(fem, opt, fields_pos, mst_pos=mst_pos, out_dir=fig_dir)

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

        # ── Continuation strategy on projection sharpness beta ──────────────────
        # After convergence, beta is increased every `bt_ns`
        # iterations until `bt_fn` is reached, progressively
        # sharpening the Heaviside projection (Eq. (19)).
        if (opt["cont_sw"] == 0) and (opt["deltaf"] < inputs["conv"]):
            opt["cont_sw"]   = 1
            opt["cont_iter"] = 0
            print("Continuation start")
        elif opt["cont_sw"] == 1:
            opt["cont_iter"] += 1
            if np.mod(opt["cont_iter"], inputs["bt_ns"]) == 1:
                opt["bt"] *= inputs["bt_ic"]

    print('finish')

    print(f"Run Npos={Npos} Completed ✅  "
          f"({time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))})")

    if force_profile_final is None:
        force_profile_final = f_pos.copy()

    scale_factor = opt.get("scale_factor", 1.0)

    # ── Save the per-run force profile ────────────────────────────
    forces_scaled = np.array(force_profile_final) * scale_factor
    positions_mm  = np.arange(len(forces_scaled))   # position index (0, 1, 2, ...)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(positions_mm, forces_scaled, "-o", linewidth=2, color="#378ADD")
    ax.set_xlabel("Plunger Position (index)")
    ax.set_ylabel("Force")
    ax.set_title(f"Force Profile — Npos = {Npos}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Force_Profile.png"), dpi=300)
    plt.close(fig)

    return {
        "Npos"          : Npos,
        "force_profile" : forces_scaled,
        "positions"     : positions_mm,
        "scale_factor"  : scale_factor,
    }


# ===================== COMPARISON PLOT (Fig. 9) =====================

def plot_comparison(all_results, out_dir):
    """Overlay the force profiles of every N_pos run (Fig. 9)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    fallback_idx = 0
    for res in all_results:
        npos = res["Npos"]
        style = STYLE_MAP.get(npos, None)

        if style is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
            style = {
                "color"    : color,
                "linestyle": "--",
                "marker"   : "o",
                "label"    : f"Npos = {npos}"
            }

        ax.plot(
            res["positions"],
            res["force_profile"] / 1000.0,
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            markersize=6,
            label=style["label"]
        )

    ax.set_xlabel("Plunger Position (index)", fontsize=12)
    ax.set_ylabel("Force (kN)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    all_forces = []
    for res in all_results:
        all_forces.extend(res["force_profile"])

    all_forces_kn = [f / 1000.0 for f in all_forces]

    max_force = max(all_forces_kn)

    y_max = int(np.ceil(max_force / 50.0) * 50)

    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 50))

    fig.tight_layout()

    save_path = os.path.join(out_dir, "Comparison_Force_Profile.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"\n✅  Comparison plot saved to: {save_path}")


# ===================== ENTRY POINT =====================

if __name__ == "__main__":

    total_start = time.time()
    all_results = []

    for npos in NPOS_LIST:
        run_dir = os.path.join(RESULTS_DIR, f"Npos_{npos}")
        result  = run_single_npos(
            Npos        = npos,
            inputs_base = INPUTS_BASE,
            modelname   = MODELNAME,
            out_dir     = run_dir
        )
        all_results.append(result)

    # Comparison plot for all runs (Fig. 9)
    plot_comparison(all_results, out_dir=RESULTS_DIR)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  ALL RUNS COMPLETED ✅")
    print(f"  Total time: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    print(f"  Results in folder: {RESULTS_DIR}/")
    print("=" * 60)
