import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

def F9_Post_Process_Plot(fem, opt, fields_pos=None, mst_pos=None, out_dir="Figures", dpi=200, density_thr=0.9):

    # ============================================================
    # INTERNAL: DRAW DOMAIN (INLINE VERSION)
    # ============================================================
    from collections import defaultdict

    def draw_domain(ax, X, IX, lw=0.6):

        Xv, Yv = X[:, 0], X[:, 1]
        faces = IX[:, 0:3] - 1
        domID = IX[:, 3]

        edge_map = defaultdict(list)

        # build edge map
        for i, tri in enumerate(faces):
            for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_map[tuple(sorted(e))].append(domID[i])

        # draw boundary edges only
        for (n1, n2), domains in edge_map.items():
            if len(domains) == 1 or (len(domains) == 2 and domains[0] != domains[1]):
                ax.plot(
                    [Xv[n1], Xv[n2]],
                    [Yv[n1], Yv[n2]],
                    color="k",
                    linewidth=lw
                )

    os.makedirs(out_dir, exist_ok=True)
    it = int(opt["iter"])

    # ============================================================
    # 4) Vector Potential & Flux Density for Selected Positions
    # ============================================================

    if fields_pos is not None:

        for pos, field in fields_pos.items():

            X = np.asarray(fem["X"], dtype=float)
            Xv = X[:, 0]
            Yv = X[:, 1]

            A = field["A"]
            B = field["B"]
            IXp = field["IX"]
            faces = IXp[:, 0:3] - 1
            domID = IXp[:, 3]

            # -------- Vector Potential --------

            A_values = np.asarray(A, dtype=float).reshape(-1)

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

            # ===== fill NaN using nearest =====
            nan_mask = np.isnan(Aq)
            if np.any(nan_mask):
                Aq_near = griddata(
                    (Xv, Yv),
                    A_values,
                    (Xq, Yq),
                    method="nearest"
                )
                Aq[nan_mask] = Aq_near[nan_mask]

            fig = plt.figure(figsize=(6.0, 3.5))
            ax = fig.add_subplot(111)

            # ============================================================
            # VECTOR POTENTIAL CONTOUR
            # ============================================================

            n_lines = 48

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

            # ============================================================
            # DOMAIN OUTLINE OVERLAY
            # ============================================================

            draw_domain(ax, fem["X"], IXp)

            # ============================================================
            # FIGURE STYLE
            # ============================================================

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            fig.tight_layout()

            fig.savefig(
                os.path.join(out_dir,
                             f"A_Pos{pos}_Iter_{it:03d}.png"),
                dpi=dpi
            )

            plt.close(fig)

            # -------- Flux Density --------
            fig = plt.figure(figsize=(6.0, 3.5))
            ax = fig.add_subplot(111)

            tpc = ax.tripcolor(
                Xv, Yv,
                triangles=faces,
                facecolors=B,
                edgecolors="none",
                shading="flat",
            )

            fig.colorbar(tpc, ax=ax)

            ax.set_aspect("equal")
            ax.set_title(f"Flux Density |B| (Pos {pos})")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f"B_Pos{pos}_Iter_{it:03d}.png"), dpi=dpi)
            plt.close(fig)

    return fem, opt
