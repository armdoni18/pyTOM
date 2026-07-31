"""
F1_Pre_Mesh_Import.py
=====================

Gmsh mesh parser specialized for Numerical Example 1: the
one-quarter IPM motor field-validation case of Section 5.1
(Fig. 3 and Table 3).

The physical groups parsed for this example are:
    1 = Air
    2 = Design
    3 = Coil1
    4 = Coil2
    5 = NonDesign
    6 = Coil3
    7 = PM1
    8 = PM2

The parser reads a Gmsh ``.msh`` file, extracts the node
coordinates and triangular element connectivity, and returns
the pyTOM connectivity matrix ``IX`` with shape (ne, 4). Columns
0-2 contain one-based node indices of each triangle, and column
3 contains the integer domain identifier.

For this example, ``Npos`` is always 1 because the IPM motor is
treated as a static single-position validation problem. The
returned ``IX_all`` list therefore contains replicated copies of
the same ``IX`` matrix.

Module is infrastructure: no equation reference.
"""

import numpy as np

def F1_Pre_Mesh_Import(modelname: str, Npos: int = 1):
    """Read the Example-1 Gmsh mesh and return pyTOM mesh structures.

    Parameters
    ----------
    modelname : str
        Gmsh model name without the ``.msh`` extension.
    Npos : int, optional
        Number of positions requested by the driver. For Example 1,
        this is normally 1 and no domain remapping is performed.

    Returns
    -------
    mesh : dict
        Mesh dictionary containing ``X`` and ``IX``.
    IX_all : list of ndarray
        Position-wise connectivity tables. For Example 1, each entry
        is a copy of the same static connectivity matrix.
    """

    # Open the Gmsh .msh file and read all lines into memory.
    fname = f"{modelname}.msh"
    with open(fname, "r") as f:
        lines = f.readlines()

    # =====================
    # INITIALIZE STORAGE
    # =====================

    # Physical-group name -> list of Gmsh physical tags.
    phys_tags = {
        "Air": [], "Design": [], "Coil1": [], "Coil2": [],
        "NonDesign": [], "Coil3": [], "PM1": [], "PM2": []
    }

    # Physical-group name -> list of geometric surface entity tags.
    domain_entities = {k: [] for k in phys_tags.keys()}

    t_tri = []          # collected triangles [n1, n2, n3, domain]
    p     = None        # nodal coordinate array filled from $Nodes

    i      = 0          # running line cursor
    nlines = len(lines)

    # =====================
    # PARSE GMSH SECTIONS
    # =====================

    while i < nlines:
        s = lines[i].strip()

        # ---------- PhysicalNames ----------
        # Map each physical-group name to its integer physical tag.
        if s == "$PhysicalNames":
            i += 1
            nphys = int(lines[i])
            for _ in range(nphys):
                i += 1
                line = lines[i]
                if '"' in line:
                    name = line.split('"')[1]        # quoted physical name
                    tag  = int(line.split()[1])      # integer physical tag
                    if name in phys_tags:
                        phys_tags[name].append(tag)
            i += 1

        # ---------- Entities ----------
        # Map each 2D surface entity to the physical group it belongs to.
        elif s == "$Entities":
            i += 1
            header       = list(map(int, lines[i].split()))
            num_points   = header[0]
            num_curves   = header[1]
            num_surfaces = header[2]
            num_volumes  = header[3]

            # Skip point and curve entities.
            for _ in range(num_points + num_curves):
                i += 1

            # Read surface entities and their attached physical tags.
            for _ in range(num_surfaces):
                i += 1
                parts    = lines[i].split()
                etag     = int(parts[0])                # surface entity tag
                numPhys  = int(parts[7])                # number of physical tags
                physIDs  = list(map(int, parts[8:8 + numPhys]))

                for name, tags in phys_tags.items():
                    if any(pid in tags for pid in physIDs):
                        domain_entities[name].append(etag)

            # Skip volume entities.
            for _ in range(num_volumes):
                i += 1
            i += 1

        # ---------- Nodes ----------
        # Read node coordinates; only the in-plane (x, y) part is kept.
        elif s == "$Nodes":
            i += 1
            header      = list(map(int, lines[i].split()))
            total_nodes = header[1]
            p           = np.zeros((total_nodes, 2))

            for _ in range(header[0]):
                i += 1
                b      = list(map(int, lines[i].split()))
                nblock = b[3]

                # First read node tags.
                node_tags = []
                while len(node_tags) < nblock:
                    i += 1
                    node_tags += list(map(int, lines[i].split()))

                # Then read the corresponding (x, y, z) coordinates.
                coords = []
                while len(coords) < 3 * nblock:
                    i += 1
                    coords += list(map(float, lines[i].split()))

                coords = np.array(coords).reshape(-1, 3)
                for k, nid in enumerate(node_tags):
                    p[nid - 1, :] = coords[k, 0:2]      # store x,y at 0-based node id

            i += 1

        # ---------- Elements ----------
        # Read 2D triangular elements and tag each with its domain id.
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
                    # The IPM motor mesh is triangular; quads are ignored.

        else:
            i += 1

    # =====================
    # BUILD CONNECTIVITY
    # =====================
    if not t_tri:
        raise ValueError("No 2D triangular elements found in msh file.")

    IX = np.array(t_tri, dtype=int)
    IX[:, 0:3] = _fix_ccw(IX[:, 0:3], p)

    mesh = {"X": p, "IX": IX}

    # Static validation case: all requested positions share one mesh.
    IX_all = [IX.copy() for _ in range(max(Npos, 1))]

    return mesh, IX_all

# =====================
# HELPER FUNCTIONS
# =====================

def _get_domain_id(entityTag, domain_entities):
    """Map a Gmsh surface entity tag to the integer domain id used by pyTOM."""
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
    """Reorder each triangle's nodes to counter-clockwise ordering."""
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
