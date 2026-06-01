"""
F1_Pre_Mesh_Import.py
=====================

Gmsh mesh parser specialized for the one-quarter IPM motor of Section 5.1 (Fig. 3).

The physical groups parsed for this example are:
    Air, Stator, Rotor, Coil, PM (with the rotor PMs and the stator coils corresponding to distinct sub-domains in the mesh file).

The general algorithmic role of this module is documented in detail at the top of the Numerical Example 3 version (``3. pyTOM Numerical ex 3/F1_Pre_Mesh_Import.py``).
The example-3 docstring describes the Gmsh format parsing, the ``IX`` matrix layout, and the multi-position domain-mapping strategy.

For this example (Example 1) Npos is always 1 (no plunger motion).

Module is infrastructure: no equation reference.
"""

import numpy as np

def F1_Pre_Mesh_Import(modelname: str, Npos: int = 1):

    fname = f"{modelname}.msh"
    with open(fname, "r") as f:
        lines = f.readlines()

    # =====================
    # INITIAL
    # =====================
    phys_tags = {
        "Air": [], "Design": [], "Coil1": [], "Coil2": [],
        "NonDesign": [], "Coil3": [], "PM1": [], "PM2": []
    }
    domain_entities = {k: [] for k in phys_tags.keys()}

    t_tri = []
    p     = None

    i      = 0
    nlines = len(lines)

    # =====================
    # PARSING
    # =====================
    while i < nlines:
        s = lines[i].strip()

        # ---------- PhysicalNames ----------
        # Map each physical-group NAME to its integer tag
        if s == "$PhysicalNames":
            i += 1
            nphys = int(lines[i])
            for _ in range(nphys):
                i += 1
                line = lines[i]
                if '"' in line:
                    name = line.split('"')[1]        # quoted physical name
                    tag  = int(line.split()[1])      # its integer tag
                    if name in phys_tags:
                        phys_tags[name].append(tag)
            i += 1

        # ---------- Entities ----------
        elif s == "$Entities":
            i += 1
            header       = list(map(int, lines[i].split()))
            num_points   = header[0]
            num_curves   = header[1]
            num_surfaces = header[2]
            num_volumes  = header[3]

            # skip points and curves
            for _ in range(num_points + num_curves):
                i += 1

            # surfaces: tag is parts[0], physical tags after parts[7]
            for _ in range(num_surfaces):
                i += 1
                parts    = lines[i].split()
                etag     = int(parts[0])
                numPhys  = int(parts[7])
                physIDs  = list(map(int, parts[8:8 + numPhys]))

                for name, tags in phys_tags.items():
                    if any(pid in tags for pid in physIDs):
                        domain_entities[name].append(etag)

            # skip volumes
            for _ in range(num_volumes):
                i += 1
            i += 1

        # ---------- Nodes ----------
        elif s == "$Nodes":
            i += 1
            header      = list(map(int, lines[i].split()))
            total_nodes = header[1]
            p           = np.zeros((total_nodes, 2))

            for _ in range(header[0]):
                i += 1
                b      = list(map(int, lines[i].split()))
                nblock = b[3]

                node_tags = []
                while len(node_tags) < nblock:
                    i += 1
                    node_tags += list(map(int, lines[i].split()))

                coords = []
                while len(coords) < 3 * nblock:
                    i += 1
                    coords += list(map(float, lines[i].split()))

                coords = np.array(coords).reshape(-1, 3)
                for k, nid in enumerate(node_tags):
                    p[nid - 1, :] = coords[k, 0:2]

            i += 1

        # ---------- Elements ----------
        elif s == "$Elements":
            i += 1
            header = list(map(int, lines[i].split()))

            for _ in range(header[0]):
                i += 1
                b = list(map(int, lines[i].split()))
                dim, tag, etype, ne = b

                dom = _get_domain_id(tag, domain_entities)

                if dim != 2:
                    i += ne
                    continue

                for _e in range(ne):
                    i += 1
                    parts = list(map(int, lines[i].split()))

                    if etype == 2:                   # 3-node triangle
                        t_tri.append([parts[1], parts[2], parts[3], dom])   # nodes + domain id
                    # IPM motor mesh is all-tri, so quads are ignored

        else:
            i += 1

    # =====================
    # PROCESS
    # =====================
    if not t_tri:
        raise ValueError("No 2D triangular elements found in msh file.")

    IX = np.array(t_tri, dtype=int)
    IX[:, 0:3] = _fix_ccw(IX[:, 0:3], p)

    mesh = {"X": p, "IX": IX}

    # No plunger stroke for IPM motor: just replicate IX for each "position"
    IX_all = [IX.copy() for _ in range(max(Npos, 1))]

    return mesh, IX_all


# =====================
# HELPERS
# =====================

def _get_domain_id(entityTag, domain_entities):
    if entityTag in domain_entities["Air"]:        return 1
    if entityTag in domain_entities["Design"]:     return 2
    if entityTag in domain_entities["Coil1"]:      return 3
    if entityTag in domain_entities["Coil2"]:      return 4
    if entityTag in domain_entities["NonDesign"]:  return 5
    if entityTag in domain_entities["Coil3"]:      return 6
    if entityTag in domain_entities["PM1"]:        return 7
    if entityTag in domain_entities["PM2"]:        return 8
    return 1  # default to air if unknown


def _fix_ccw(F, X):
    """Reorder each triangle to counter-clockwise (positive signed area)."""
    F  = F.copy()
    n1 = F[:, 0] - 1
    n2 = F[:, 1] - 1
    n3 = F[:, 2] - 1

    x1, y1 = X[n1, 0], X[n1, 1]
    x2, y2 = X[n2, 0], X[n2, 1]
    x3, y3 = X[n3, 0], X[n3, 1]

    A    = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)     # 2 * signed area
    flip = A < 0                                             # clockwise triangles
    F[flip, 1], F[flip, 2] = F[flip, 2], F[flip, 1].copy()   # swap to CCW
    return F
