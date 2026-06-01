"""
F9_Post_Process_Plot.py
=======================

Post-processing and visualization for the nonlinear-material
actuator of Section 5.2. Renders the optimized topology, the
magnetic field, and the convergence histories (Fig. 5(a-b)).

The general role of this module is documented in detail in the
Example 3 version (``3. pyTOM Numerical ex 3/F9_Post_Process_Plot.py``).

Module is infrastructure: no equation reference.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from collections import defaultdict


def F9_Post_Process_Plot(fem, opt, fields_pos=None, mst_pos=None,
                         out_dir="Figures", dpi=200, density_thr=0.9):

    os.makedirs(out_dir, exist_ok=True)
    it = int(opt["iter"])

    # ============================================================
    # 1) Objective History
    # ============================================================
    if "f" in opt and len(opt["f"]) > 0:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        f_hist = np.array(opt["f"], dtype=float)

        ax.plot(np.arange(1, len(f_hist) + 1), f_hist, "-k", linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_title("Objective History")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Obj_History.png"), dpi=dpi)
        plt.close(fig)

    # ============================================================
    # 2) Volume History
    # ============================================================
    if "g" in opt and len(opt["g"]) > 0:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))

        g_hist = np.array(opt["g"], dtype=float)
        volfrac = float(opt["volfrac"])
        volume_curve = g_hist + volfrac

        ax.plot(np.arange(1, len(volume_curve) + 1),
                volume_curve, "-b", linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Volume")
        ax.set_ylim([0, 1])
        ax.set_title("Volume Constraint History")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Vol_History.png"), dpi=dpi)
        plt.close(fig)

    # ============================================================
    # 3) Density (Topology Result)
    # ============================================================
    X     = np.asarray(fem["X"], dtype=float)
    Xv    = X[:, 0]
    Yv    = X[:, 1]

    IX    = np.asarray(fem["IX"], dtype=int)
    faces = IX[:, 0:3] - 1
    domID = IX[:, 3]

    erho  = np.asarray(opt["erho"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    # --- Air + Iron ---
    for dom, color_val, cmap in [
        (1,  1.0,  "gray"),
        ((5, 6), 0.6, "gray"),
    ]:
        mask = np.isin(domID, list(dom)) if isinstance(dom, tuple) else (domID == dom)

        if np.any(mask):
            ax.tripcolor(Xv, Yv,
                         triangles=faces[mask],
                         facecolors=np.full(np.sum(mask), color_val),
                         cmap=cmap,
                         vmin=0,
                         vmax=1,
                         edgecolors="none",
                         shading="flat")

    # --- Coil + PM ---
    for dom_val, hex_color in [(3, "#E6B800"), (4, "#E6B800"), (7, "#1F4ED8")]:
        mask = (domID == dom_val)
        if np.any(mask):
            ax.tripcolor(Xv, Yv,
                         triangles=faces[mask],
                         facecolors=np.ones(np.sum(mask)),
                         cmap=ListedColormap([hex_color]),
                         edgecolors="none",
                         shading="flat")

    # --- Design domain ---
    dd_mask = (domID == 2)
    rho_dd = erho[dd_mask]

    if "B" not in fem:
        raise ValueError("fem['B'] not found. Ensure B has been computed and stored in FEM.")

    B_all = np.asarray(fem["B"], dtype=float)
    B_dd = B_all[dd_mask]

    B_thr = 0.05 * np.max(B_all)

    solid_idx = (rho_dd >= density_thr) & (B_dd >= B_thr)

    if np.any(solid_idx):
        ax.tripcolor(
            Xv, Yv,
            triangles=faces[dd_mask][solid_idx],
            facecolors=rho_dd[solid_idx],
            cmap="gray_r",
            edgecolors="none",
            shading="flat"
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Topology Optimization Result")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Density_Iter_{it:03d}.png"), dpi=dpi)
    plt.close(fig)

    # ============================================================
    # 4) Vector Potential & Flux Density
    # ============================================================
    if fields_pos is not None:

        for pos, field in fields_pos.items():

            A_f  = field["A"]
            B_f  = field["B"]
            IXp  = field["IX"]
            faces_p = IXp[:, 0:3] - 1

            A_values = np.asarray(A_f, dtype=float).reshape(-1)

            Xq, Yq = np.meshgrid(
                np.linspace(np.min(Xv), np.max(Xv), 391),
                np.linspace(np.min(Yv), np.max(Yv), 251)
            )

            Aq = griddata((Xv, Yv), A_values, (Xq, Yq), method="linear")

            nan_mask = np.isnan(Aq)
            if np.any(nan_mask):
                Aq[nan_mask] = griddata(
                    (Xv, Yv),
                    A_values,
                    (Xq[nan_mask], Yq[nan_mask]),
                    method="nearest"
                )

            # --- A contour ---
            fig, ax = plt.subplots(figsize=(6.0, 3.5))

            levels = np.linspace(np.nanmin(Aq), np.nanmax(Aq), 42)

            ax.contour(Xq, Yq, Aq,
                       levels=levels,
                       colors="k",
                       linewidths=0.4)

            Func0_Draw_Domain(ax, fem["X"], IXp)

            ax.set_aspect("equal")
            ax.axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f"A_Pos{pos}_Iter_{it:03d}.png"),
                        dpi=dpi)
            plt.close(fig)

            # --- B field ---
            fig, ax = plt.subplots(figsize=(6.0, 3.5))

            tpc = ax.tripcolor(Xv, Yv,
                               triangles=faces_p,
                               facecolors=B_f,
                               edgecolors="none",
                               shading="flat")

            fig.colorbar(tpc, ax=ax)

            ax.set_aspect("equal")
            ax.set_title(f"|B| (Pos {pos})")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f"B_Pos{pos}_Iter_{it:03d}.png"),
                        dpi=dpi)
            plt.close(fig)

    # ============================================================
    # 5) Force Profile
    # ============================================================
    if "force_profile" in opt and opt["force_profile"] is not None:

        forces = np.array(opt["force_profile"], dtype=float)
        scale  = opt.get("scale_factor", 1.0)
        forces_scaled = forces * scale

        positions = np.arange(len(forces_scaled))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(positions, forces_scaled, '-o', linewidth=2)

        ax.set_xlabel("Plunger Position (index)")
        ax.set_ylabel("Force")
        ax.set_title("Force Profile Along Stroke")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Force_Profile.png"), dpi=dpi)
        plt.close(fig)

    return fem, opt


# ============================================================
# HELPER FUNCTION
# ============================================================
def Func0_Draw_Domain(ax, X, IX, lw=0.6):

    Xv, Yv = X[:, 0], X[:, 1]
    faces = IX[:, 0:3] - 1
    domID = IX[:, 3]

    edge_map = defaultdict(list)

    for i, tri in enumerate(faces):
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_map[tuple(sorted(e))].append(domID[i])

    for (n1, n2), domains in edge_map.items():
        if len(domains) == 1 or (len(domains) == 2 and domains[0] != domains[1]):
            ax.plot(
                [Xv[n1], Xv[n2]],
                [Yv[n1], Yv[n2]],
                color="k",
                linewidth=lw
            )
