import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap


def F9_Post_Process_Plot(fem, opt, out_dir="Figures", dpi=200, density_thr=0.9):

    os.makedirs(out_dir, exist_ok=True)
    it = int(opt["iter"])

    # ============================================================
    # 1) Objective History  (OVERWRITE EVERY ITERATION)
    # ============================================================
    if "f" in opt and len(opt["f"]) > 0:
        fig = plt.figure(figsize=(4.5, 3.5))
        ax = fig.add_subplot(111)

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
    # 2) Volume History  (OVERWRITE EVERY ITERATION)
    # ============================================================
    if "g" in opt and len(opt["g"]) > 0:
        fig = plt.figure(figsize=(4.5, 3.5))
        ax = fig.add_subplot(111)

        g_hist = np.array(opt["g"], dtype=float)
        volfrac = float(opt["volfrac"])

        volume_curve = g_hist + volfrac

        ax.plot(np.arange(1, len(volume_curve) + 1),
                volume_curve, "-b", linewidth=2)

        ax.plot([1, len(volume_curve)],
                [volfrac, volfrac], ":b", linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Volume")
        ax.set_ylim([0, 1])
        ax.set_title("Volume Constraint History")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Vol_History.png"), dpi=dpi)
        plt.close(fig)

    # ============================================================
    # 3) Density (FULL GEOMETRY + TOPOLOGY RESULT)
    # ============================================================

    X = np.asarray(fem["X"], dtype=float)
    Xv = X[:, 0]
    Yv = X[:, 1]

    IX = np.asarray(fem["IX"], dtype=int)

    faces_all = IX[:, 0:3] - 1
    domID = IX[:, 3]

    erho = np.asarray(opt["erho"], dtype=float)

    fig = plt.figure(figsize=(6.0, 3.5))
    ax = fig.add_subplot(111)

    # ---------- AIR (WHITE) ----------
    air_faces = faces_all[domID == 1]
    if len(air_faces) > 0:
        ax.tripcolor(
            Xv, Yv,
            triangles=air_faces,
            facecolors=np.ones(len(air_faces)),
            cmap="gray",
            vmin=0,
            vmax=1,
            edgecolors="none",
            shading="flat"
        )

    # ---------- FIXED / NON-DESIGN IRON (GRAY) ----------
    iron_faces = faces_all[(domID == 5) | (domID == 6)]
    if len(iron_faces) > 0:
        ax.tripcolor(
            Xv, Yv,
            triangles=iron_faces,
            facecolors=np.ones(len(iron_faces)) * 0.6,
            cmap="gray",
            edgecolors="none",
            shading="flat"
        )

    # ---------- COILS (YELLOW) ----------
    coil_cmap = ListedColormap(["#E6B800"])  # dark yellow
    coil_faces = faces_all[(domID == 3) | (domID == 4)]
    if len(coil_faces) > 0:
        ax.tripcolor(
            Xv, Yv,
            triangles=coil_faces,
            facecolors=np.ones(len(coil_faces)),  # 1D scalar
            cmap=coil_cmap,
            vmin=0, vmax=1,
            edgecolors="none",
            shading="flat"
        )

    # ---------- PERMANENT MAGNET (BLUE) ----------
    pm_cmap = ListedColormap(["#1F4ED8"])  # dark blue
    pm_faces = faces_all[domID == 7]
    if len(pm_faces) > 0:
        ax.tripcolor(
            Xv, Yv,
            triangles=pm_faces,
            facecolors=np.ones(len(pm_faces)),  # 1D scalar
            cmap=pm_cmap,
            vmin=0, vmax=1,
            edgecolors="none",
            shading="flat"
        )

    # ---------- TOPOLOGY RESULT ----------
    design_faces = faces_all[domID == 2]
    rho_design = erho[domID == 2]

    B_all = np.asarray(fem["B"], dtype=float)
    B_design = B_all[domID == 2]

    B_thr = 0.05 * np.max(B_all)

    mask = (rho_design >= density_thr) & (B_design >= B_thr)

    design_faces = design_faces[mask]
    rho_design = rho_design[mask]

    if len(design_faces) > 0:
        ax.tripcolor(
            Xv, Yv,
            triangles=design_faces,
            facecolors=rho_design,
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
    # DOMAIN OUTLINE PLOT
    # ============================================================

    X = np.asarray(fem["X"], dtype=float)
    Xv = X[:, 0]
    Yv = X[:, 1]

    IX = np.asarray(fem["IX"], dtype=int)

    faces_all = IX[:, 0:3] - 1
    domID = IX[:, 3]

    fig = plt.figure(figsize=(6.0, 3.5))
    ax = fig.add_subplot(111)

    # ============================================================
    # DOMAIN BACKGROUND (WHITE)
    # ============================================================

    ax.tripcolor(
        Xv,
        Yv,
        triangles=faces_all,
        facecolors=np.ones(len(faces_all)),
        cmap="gray",
        vmin=0,
        vmax=1,
        edgecolors="none",
        shading="flat"
    )

    # ============================================================
    # DOMAIN OUTLINE (BOUNDARY BETWEEN DOMAINS)
    # ============================================================

    from collections import defaultdict

    edge_map = defaultdict(list)

    for i, tri in enumerate(faces_all):

        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]

        for e in edges:
            edge_map[e].append(domID[i])

    for e, domains in edge_map.items():

        draw_edge = False

        if len(domains) == 1:
            draw_edge = True

        elif len(domains) == 2 and domains[0] != domains[1]:
            draw_edge = True

        if draw_edge:
            n1, n2 = e

            ax.plot(
                [Xv[n1], Xv[n2]],
                [Yv[n1], Yv[n2]],
                color="k",
                linewidth=0.8
            )

    # ============================================================
    # VECTOR POTENTIAL INTERPOLATION
    # ============================================================

    A_values = np.asarray(fem["A"], dtype=float).reshape(-1)

    Xq, Yq = np.meshgrid(
        np.linspace(np.min(Xv), np.max(Xv), 391),
        np.linspace(np.min(Yv), np.max(Yv), 251)
    )

    Aq = griddata(
        (Xv, Yv),
        A_values,
        (Xq, Yq),
        method="linear"
    )

    nan_mask = np.isnan(Aq)
    if np.any(nan_mask):
        Aq_near = griddata(
            (Xv, Yv),
            A_values,
            (Xq, Yq),
            method="nearest"
        )
        Aq[nan_mask] = Aq_near[nan_mask]

    # ============================================================
    # VECTOR POTENTIAL CONTOUR
    # ============================================================

    n_lines = 50

    Amin = np.nanmin(Aq)
    Amax = np.nanmax(Aq)

    levels = np.linspace(Amin, Amax, n_lines)

    ax.contour(
        Xq,
        Yq,
        Aq,
        levels=levels,
        colors="k",
        linewidths=0.4,
        linestyles="solid"
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, f"A_Iter_{it:03d}.png"),
        dpi=dpi
    )

    plt.close(fig)

    # ============================================================
    # 5) Magnetic Flux Density (SAVE EVERY ITERATION)
    # ============================================================
    B = np.asarray(fem["B"], dtype=float).reshape(-1)

    fig = plt.figure(figsize=(6.0, 3.5))
    ax = fig.add_subplot(111)

    tpc = ax.tripcolor(
        Xv, Yv,
        triangles=faces_all,
        facecolors=B,
        edgecolors="none",
        shading="flat",
    )

    fig.colorbar(tpc, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Magnetic Flux Density |B|")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"B_Iter_{it:03d}.png"), dpi=dpi)
    plt.close(fig)

    return fem, opt