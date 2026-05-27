"""
F9_Post_Process_Plot.py — Numerical Example 1 (IPM motor benchmark)
==================================================================

Post-processing and visualization for the IPM motor of Section 5.1.
Renders the magnetic vector potential contours for the validation comparison against COMSOL (Fig. 3).

The general role of this module is documented in detail in the Example 3 version (``3. pyTOM Numerical ex 3/F9_Post_Process_Plot.py``).

Module is infrastructure: no equation reference.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
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
    # 3) Vector Potential
    # ============================================================
    X     = np.asarray(fem["X"], dtype=float)
    Xv    = X[:, 0]
    Yv    = X[:, 1]

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

            levels = np.linspace(np.nanmin(Aq), np.nanmax(Aq), 48)

            ax.contour(Xq, Yq, Aq,
                       levels=levels,
                       colors="k",
                       linewidths=0.4,
                       linestyles="solid")

            Func0_Draw_Domain(ax, fem["X"], IXp)

            ax.set_aspect("equal")
            ax.axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f"A_Pos{pos}_Iter_{it:03d}.png"),
                        dpi=dpi)
            plt.close(fig)

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
