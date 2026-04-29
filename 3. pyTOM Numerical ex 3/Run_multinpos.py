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


# ================================================================
# USER SETTINGS
# ================================================================

MODELNAME = "Example_3_Actuator_MultiPos"

# list the desired Npos
NPOS_LIST  = [1,11,21]

# Folder root for every result
RESULTS_DIR = "Results"

# ── Parameter ──────────────────────────────────────────────
INPUTS_BASE = {
    "penal"    : 3,
    "initdv"   : -0.5,
    "VT"       : 16800,
    "VND"      : 13200,
    "VDD"      : 3600,
    "volfrac"  : 0.30,
    "mu0"      : 4 * np.pi * 1e-7,
    "mur_air"  : 1,
    "mur_coil1": 1,
    "mur_coil2": 1,
    "mur_iron" : 1500,
    "mur_PM"   : 1,
    "J_am2"    : 17900,
    "conv"     : 0.008,
    "bt_init"  : 0.1,
    "bt_ic"    : 1.5,
    "bt_ns"    : 4,
    "bt_fn"    : 20,
    "MMA"      : 1000,
    "rmin"     : 10,
    "iterMax"  : 400,
    "scale"    : 1000,
    "PM"       : {
        "domIDs": [7],
        "Br"    : [0.2],
        "theta" : [180.0]
    }
}

STYLE_MAP = {
    1 : {"color": "#378ADD", "linestyle": "-",    "marker": "o", "label": r"$N_{pos}=1$"},
    3 : {"color": "#1D9E75", "linestyle": "--",   "marker": "s", "label": r"$N_{pos}=3$"},
    5 : {"color": "#D85A30", "linestyle": "-.",   "marker": "^", "label": r"$N_{pos}=5$"},
    7 : {"color": "#7F77DD", "linestyle": ":",    "marker": "D", "label": r"$N_{pos}=7$"},
    11: {"color": "#BA7517", "linestyle": (0,(5,2,1,2)), "marker": "v", "label": r"$N_{pos}=11$"},
}

_FALLBACK_COLORS = ["#E24B4A", "#5DCAA5", "#D4537E", "#639922", "#EF9F27"]


# ================================================================
# Main-code run every Npos
# ================================================================

def run_single_npos(Npos, inputs_base, modelname, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)

    run_start = time.time()
    print("\n" + "="*60)
    print(f"  MULAI RUN  Npos = {Npos}")
    print(f"  Output    : {out_dir}")
    print("="*60)

    # ── copy inputs (each run independently) ──────────────────────
    inputs = copy.deepcopy(inputs_base)
    mu0    = inputs["mu0"]
    inputs["nu_air"]   = 1.0 / (mu0 * inputs["mur_air"])
    inputs["nu_coil1"] = 1.0 / (mu0 * inputs["mur_coil1"])
    inputs["nu_coil2"] = 1.0 / (mu0 * inputs["mur_coil2"])
    inputs["nu_iron"]  = 1.0 / (mu0 * inputs["mur_iron"])
    inputs["nu_PM"]    = 1.0 / (mu0 * inputs["mur_PM"])

    # ── Pre-processing ───────────────────────────────────────────
    mesh, IX_all = F1_Pre_Mesh_Import(modelname, Npos=Npos)
    fem          = F2_Pre_FEM_Init(inputs, mesh)
    opt, MMA     = F3_Pre_Opt_Init(inputs, fem)

    MMA["c_input"]                = inputs["MMA"]
    opt["mma_scale_initialized"]  = False
    opt["mma_obj_scale"]          = 1.0

    print(f"Pre-Processing Npos={Npos} selesai ✅  "
          f"({time.strftime('%H:%M:%S', time.gmtime(time.time()-run_start))})")

    # ── Setup variable tracking ──────────────────────────────────
    ne            = fem["ne"]
    pm_domIDs     = set(inputs["PM"]["domIDs"])
    force_profile_final = None
    saved_iter    = -1

    # Plotted position
    if Npos == 1:
        plot_positions = [0]
    elif Npos <= 3:
        plot_positions = list(range(Npos))
    else:
        step = max(1, (Npos - 1) // 2)
        plot_positions = [0, step, Npos - 1]

    # ── Main loop ────────────────────────────────────────────────
    while (opt["bt"] < inputs["bt_fn"]) and (opt["iter"] <= inputs["iterMax"]):

        # Filter + Projection
        opt["fdv"]  = spsolve(opt["Kft_sparse"],
                              sp.csc_matrix.dot(opt["Tft"], opt["nv"]))
        opt["nrho"] = np.maximum(
            np.minimum(
                np.tanh(opt["bt"] * opt["fdv"]) / (2 * np.tanh(opt["bt"])) + 0.5,
                1), -1)
        opt["erho"] = opt["Ten"].dot(opt["nrho"])

        # Multi-position loop
        fields_pos = {}
        f_pos      = []
        g_pos      = []
        dfdx_pos   = []
        dgdx_pos   = []
        mst_pos    = {}

        for j in range(Npos):

            fem["IX"][:, 3] = IX_all[j][:, 3]
            IX = fem["IX"]

            dom      = IX[:, 3].astype(int)
            erho_vec = np.asarray(opt["erho"], dtype=float).reshape(-1)
            penal    = inputs["penal"]

            # Initial nu_e_all
            nu_e_all = np.full(ne, inputs["nu_air"])
            dd_mask  = (dom == 2)
            nu_e_all[dd_mask] = (inputs["nu_air"] +
                                 (inputs["nu_iron"] - inputs["nu_air"]) *
                                 erho_vec[dd_mask] ** penal)
            iron_mask = (dom == 5) | (dom == 6)
            nu_e_all[iron_mask] = inputs["nu_iron"]
            nu_e_all[dom == 3]  = inputs["nu_coil1"]
            nu_e_all[dom == 4]  = inputs["nu_coil2"]
            for pmid in pm_domIDs:
                nu_e_all[dom == pmid] = inputs["nu_PM"]

            # Linear solve initially
            fem   = F4_Main_Solve_VecPot(fem, inputs, nu_e_all)
            A_old = fem["A"].copy()
            T_rhs = fem["T"].copy()

            print(f"\n  [Npos={Npos}] Pos {j+1}/{Npos}: linear solve selesai. NR start...")

            all_dofs = np.arange(fem["ndof"])
            fixdof   = fem["bcdof"].astype(int) - 1
            bcval    = fem["bcval"]
            freedof  = np.setdiff1d(all_dofs, fixdof)
            A_old[fixdof] = bcval

            # Newton-Raphson
            NR_max = 30
            NR_tol = 1e-5

            for iterNR in range(NR_max):
                A_old[fixdof] = bcval
                fem["A"]      = A_old

                fem  = F5_Main_Comp_Flux(fem)
                B    = fem["B"]

                dom_cur  = IX[:, 3].astype(int)
                dnu_dB_e = np.zeros(ne, dtype=float)

                mu_all  = F0_Main_Mat_Nonlinear(B)
                dmu_all = F0_Main_Mat_Derivative(B)
                nu_nl   = 1.0 / mu_all
                dnu_nl  = -dmu_all / (mu_all ** 2)

                fi_mask = (dom_cur == 5) | (dom_cur == 6)
                nu_e_all[fi_mask]  = nu_nl[fi_mask]
                dnu_dB_e[fi_mask]  = dnu_nl[fi_mask]

                dd2_mask = (dom_cur == 2)
                nu_e_all[dd2_mask] = (inputs["nu_air"] +
                                      (nu_nl[dd2_mask] - inputs["nu_air"]) *
                                      erho_vec[dd2_mask] ** penal)
                dnu_dB_e[dd2_mask] = dnu_nl[dd2_mask] * erho_vec[dd2_mask] ** penal

                nu_e_all[dom_cur == 3] = inputs["nu_coil1"]
                nu_e_all[dom_cur == 4] = inputs["nu_coil2"]
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
                print(f"     NR iter {iterNR+1}  ||ΔA||/||A|| = {errA:.3e}")

                A_old = A_new
                if errA < NR_tol:
                    print("  NR konvergen ✓")
                    break

            fem["A"] = A_old
            fem      = F5_Main_Comp_Flux(fem)

            # Saved field for the dedicated position
            if j in plot_positions:
                fields_pos[j + 1] = {
                    "A" : fem["A"].copy(),
                    "B" : fem["B"].copy(),
                    "IX": fem["IX"].copy()
                }

            # Force & sensitivity
            Fx_total, Fy_total, fem = F7_Main_Comp_Force(fem)
            mst_pos[j + 1] = {
                "mst": fem["mst"].copy(),
                "IX" : fem["IX"].copy()
            }

            f, g, dfdx, dgdx, dfdrho_e, lam, dfdA = F8_Main_Comp_Sens(
                fem, opt, J_mat)

            f_pos.append(float(f))
            g_pos.append(float(g))
            dfdx_pos.append(np.asarray(dfdx).reshape(-1, 1))
            dgdx_pos.append(np.asarray(dgdx).reshape(1, -1))

        # Saved force profile in the last iteration
        converged = (opt["deltaf"] < inputs["conv"])
        last_iter = (opt["iter"] == inputs["iterMax"])
        if (converged or last_iter) and (opt["iter"] > saved_iter):
            force_profile_final = f_pos.copy()
            saved_iter = opt["iter"]

        # average along the position
        f_avg    = float(np.mean(f_pos))
        g_avg    = float(np.mean(g_pos))
        dfdx_avg = np.mean(np.hstack(dfdx_pos), axis=1, keepdims=True)
        dgdx_avg = np.mean(np.vstack(dgdx_pos), axis=0, keepdims=True)

        # Scaling
        if not opt["mma_scale_initialized"]:
            scale_factor = inputs["scale"] / (abs(f_avg) + 1e-12)
            opt["scale_factor"] = scale_factor
            MMA["c"] = MMA["c_input"] / scale_factor
            opt["mma_scale_initialized"] = True

        f_scaled    = f_avg    * opt["scale_factor"]
        dfdx_scaled = dfdx_avg * opt["scale_factor"]

        opt["f"].append(f_scaled)
        opt["g"].append(g_avg)
        opt["dfdx"] = dfdx_scaled
        opt["dgdx"] = dgdx_avg

        if opt["iter"] > 1:
            opt["deltaf"] = np.abs(
                (opt["f"][-1] - opt["f"][-2]) / (opt["f"][-2] + 1e-30))

        print("iter:%3d  f:%.4f  Vol:%.4f  deltaf:%.5f  beta:%.2f  [Npos=%d]" % (
            opt["iter"], opt["f"][-1],
            opt["g"][-1] + inputs["volfrac"],
            opt["deltaf"], opt["bt"], Npos))

        # Plotting (save to fig_dir)
        F9_Post_Process_Plot(fem, opt, fields_pos,
                             mst_pos=mst_pos, out_dir=fig_dir)

        # MMA update
        (opt["dvnew"], ymma, zmma, lam_mma, xsi, eta, mu_mma, zet, s,
         MMA["low"], MMA["upp"]) = mmasub(
            1, len(opt["dv"]), opt["iter"],
            opt["dv"], opt["dvmin"], opt["dvmax"],
            opt["dvold"], opt["dvolder"],
            opt["f"][-1], opt["dfdx"],
            opt["g"][-1], opt["dgdx"],
            MMA["low"], MMA["upp"],
            MMA["a0"], MMA["a"], MMA["c"], MMA["d"], 1)

        opt["iter"]    += 1
        opt["dvolder"]  = opt["dvold"]
        opt["dvold"]    = opt["dv"]
        opt["dv"]       = opt["dvnew"]
        opt["nv"][opt["dof_dd"] - 1] = opt["dv"]

        # Continuation
        if (opt["cont_sw"] == 0) and (opt["deltaf"] < inputs["conv"]):
            opt["cont_sw"]   = 1
            opt["cont_iter"] = 0
            print("  >> Continuation mulai")
        elif opt["cont_sw"] == 1:
            opt["cont_iter"] += 1
            if np.mod(opt["cont_iter"], inputs["bt_ns"]) == 1:
                opt["bt"] *= inputs["bt_ic"]

    print(f"\nRun Npos={Npos} selesai ✅  "
          f"({time.strftime('%H:%M:%S', time.gmtime(time.time()-run_start))})")

    if force_profile_final is None:
        force_profile_final = f_pos.copy()

    scale_factor = opt.get("scale_factor", 1.0)

    # ── Save force profile individually ───────────────────────────
    forces_scaled = np.array(force_profile_final) * scale_factor
    positions_mm  = np.arange(len(forces_scaled))   # index posisi (0,1,2,...)

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

# ================================================================
# PLOT COMPARISON — all Npos in 1 graphic
# ================================================================

def plot_comparison(all_results, out_dir):
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
    # ax.set_title("Force Profile Comparison", fontsize=13)
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

    print(f"\n✅  Plot perbandingan disimpan ke: {save_path}")


# ================================================================
# ENTRY POINT
# ================================================================

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

    # Plot Comparison for all run
    plot_comparison(all_results, out_dir=RESULTS_DIR)

    total_elapsed = time.time() - total_start
    print("\n" + "="*60)
    print(f"  SEMUA RUN SELESAI ✅")
    print(f"  Total waktu: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
    print(f"  Hasil ada di folder: {RESULTS_DIR}/")
    print("="*60)