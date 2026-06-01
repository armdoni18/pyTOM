"""
F1_Pre_Mesh_Import.py
=====================

Gmsh mesh parser for the magnetic-actuator examples (Examples 2
and 3). The same parser serves the single-position actuator
(Npos = 1) and the multi-position actuator (Npos > 1); the
plunger-stroke handling below is exercised only when Npos > 1.

Reads a Gmsh ``.msh`` file (format 4.1) and extracts the node
coordinates, the triangular element connectivity, and the
physical-group domain identifiers. The output ``IX`` matrix has
shape (ne, 4) where columns 0-2 are one-based node indices of
each triangle and column 3 is the integer domain identifier.

Domain identifiers used in this example:
    1 = Air
    2 = Design (yoke design domain)
    3 = Coil1
    4 = Coil2
    5 = Plunger
    6 = FixIron (non-design iron of the yoke)
    7 = PM1 (permanent magnet)

For the multi-position case (Npos > 1), this function additionally
returns ``IX_all``: a list of ``Npos`` alternative column-3 arrays,
one per plunger position. The driver swaps column 3 of ``IX``
across positions instead of physically translating the mesh; this
"domain-mapping" strategy preserves mesh quality and avoids costly
remeshing operations during the multi-position evaluation.

Module is infrastructure: no equation reference.
"""

import numpy as np


def F1_Pre_Mesh_Import(modelname: str, Npos: int = 1):

# Open the Gmsh .msh file (format 4.1) and read all lines into memory
    fname = f"{modelname}.msh"
    with open(fname, "r") as f:
        lines = f.readlines()

    # =====================
    # INITIAL
    # =====================
# Physical-group name -> list of Gmsh physical tags (filled from $PhysicalNames)
    phys_tags = {
        "Air": [], "Design": [], "Coil1": [], "Coil2": [],
        "NonDesign": [], "FixIron": [], "PM1": []
    }
# Physical-group name -> list of geometric surface entity tags (filled from $Entities)
    domain_entities = {k: [] for k in phys_tags.keys()}

    t_tri  = []                          # collected triangles  [n1,n2,n3,domain]
    q_quad = []                          # collected quads      [n1,n2,n3,n4,domain]
    p      = None                        # nodal coordinate array (filled in $Nodes)

    i      = 0                           # running line cursor
    nlines = len(lines)

    # =====================
    # PARSING
    # =====================
# Single forward pass over the file; each $Section is handled in its own branch
    while i < nlines:
        s = lines[i].strip()

        if s == "$PhysicalNames":
        # Map each physical-group NAME to its integer tag
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

        elif s == "$Entities":
        # Map each 2D surface ENTITY to the physical group it belongs to
            i += 1
            header      = list(map(int, lines[i].split()))
            num_surfaces = header[2]                 # header = [nPts,nCurves,nSurf,nVol]

            for _ in range(header[0] + header[1]):   # skip point + curve entities
                i += 1

            for _ in range(num_surfaces):
                i += 1
                parts    = lines[i].split()
                etag     = int(parts[0])             # surface entity tag
                numPhys  = int(parts[7])             # number of physical tags attached
                physIDs  = list(map(int, parts[8:8+numPhys]))

                # Attach this surface entity to every matching physical group
                for name, tags in phys_tags.items():
                    if any(pid in tags for pid in physIDs):
                        domain_entities[name].append(etag)

            for _ in range(header[3]):               # skip volume entities
                i += 1
            i += 1

        elif s == "$Nodes":
        # Read node coordinates; only the in-plane (x, y) part is kept
            i += 1
            header      = list(map(int, lines[i].split()))
            total_nodes = header[1]
            p           = np.zeros((total_nodes, 2))

            for _ in range(header[0]):               # loop over node blocks
                i += 1
                b      = list(map(int, lines[i].split()))
                nblock = b[3]                        # number of nodes in this block

                # First the block's node tags ...
                node_tags = []
                while len(node_tags) < nblock:
                    i += 1
                    node_tags += list(map(int, lines[i].split()))

                # ... then their (x, y, z) coordinates
                coords = []
                while len(coords) < 3 * nblock:
                    i += 1
                    coords += list(map(float, lines[i].split()))

                coords = np.array(coords).reshape(-1, 3)
                for k, nid in enumerate(node_tags):
                    p[nid - 1, :] = coords[k, 0:2]   # store x,y at 0-based node id

            i += 1

        elif s == "$Elements":
        # Read 2D elements (triangles / quads) and tag each with its domain id
            i += 1
            header = list(map(int, lines[i].split()))

            for _ in range(header[0]):               # loop over element blocks
                i += 1
                b = list(map(int, lines[i].split()))
                dim, tag, etype, ne = b              # block dim, entity tag, type, count

                dom = _get_domain_id(tag, domain_entities)   # resolve domain id

                if dim != 2:                         # skip 0D/1D entities
                    i += ne
                    continue

                for _e in range(ne):
                    i += 1
                    parts = list(map(int, lines[i].split()))
                    if etype == 2:                   # 3-node triangle
                        t_tri.append([parts[1], parts[2], parts[3], dom])
                    elif etype == 3:                 # 4-node quad
                        q_quad.append([parts[1], parts[2], parts[3], parts[4], dom])

        else:
            i += 1

    # =====================
    # PROCESS
    # =====================
# Convert collected lists to arrays; split quads into triangles
    t_tri  = np.array(t_tri,  dtype=int) if t_tri  else None
    q_quad = np.array(q_quad, dtype=int) if q_quad else None

    IX_quad = None
    IX_tri  = None

    if q_quad is not None:
        IX_quad = _simple_quad_to_tri(q_quad[:, 0:4], q_quad[:, 4])  # 1 quad -> 2 tris
        IX_quad[:, 0:3] = _fix_ccw(IX_quad[:, 0:3], p)               # enforce CCW ordering

    if t_tri is not None:
        t_tri[:, 0:3] = _fix_ccw(t_tri[:, 0:3], p)                   # enforce CCW ordering
        IX_tri = t_tri

# Stack quad-derived and native triangles into a single connectivity table
    IX_parts = []
    if IX_quad is not None:
        IX_parts.append(IX_quad)
    if IX_tri is not None:
        IX_parts.append(IX_tri)

    IX   = np.vstack(IX_parts) if IX_parts else np.zeros((0, 4), int)
    mesh = {"X": p, "IX": IX}
    mesh = _sort_elements_structured_like_quad(mesh)    # deterministic element ordering

    # =====================
    # MULTI POSITION
    # =====================
# Build one connectivity table per plunger position (domain-mapping strategy)
    IX_all = []

    if Npos > 1 and IX_quad is not None:
        # Generate the plunger/air domain-id swaps directly on the quad grid
        mesh_quad   = {"X": p, "IX": IX_quad}
        IX_quad_all = _make_plunger_sets_blockshift(mesh_quad, Npos)

        for k in range(Npos):
            parts = [IX_quad_all[k]]
            if IX_tri is not None:
                parts.append(IX_tri)
            IXk      = np.vstack(parts)
            mesh_tmp = {"X": p, "IX": IXk}
            mesh_tmp = _sort_elements_structured_like_quad(mesh_tmp)
            IX_all.append(mesh_tmp["IX"])
    else:
        # Single position (or no quad grid): all positions share one mesh
        IX_all = [mesh["IX"].copy() for _ in range(Npos)]

    return mesh, IX_all


# =====================
# HELPERS
# =====================

def _get_domain_id(entityTag, domain_entities):
    """Map a Gmsh surface entity tag to the integer domain id used by pyTOM."""
    if entityTag in domain_entities["Air"]:       return 1
    if entityTag in domain_entities["Design"]:    return 2
    if entityTag in domain_entities["Coil1"]:     return 3
    if entityTag in domain_entities["Coil2"]:     return 4
    if entityTag in domain_entities["NonDesign"]: return 5
    if entityTag in domain_entities["FixIron"]:   return 6
    if entityTag in domain_entities["PM1"]:       return 7
    return 0   # unassigned / outside the modelled domains


def _simple_quad_to_tri(IXq, domq):
    """Split each quad (n1,n2,n3,n4) into two triangles (n1,n2,n3) and (n1,n3,n4)."""
    n    = IXq.shape[0]
    out  = np.zeros((2 * n, 4), dtype=int)
    out[0::2, 0] = IXq[:, 0]; out[0::2, 1] = IXq[:, 1]   # first triangle of each quad
    out[0::2, 2] = IXq[:, 2]; out[0::2, 3] = domq
    out[1::2, 0] = IXq[:, 0]; out[1::2, 1] = IXq[:, 2]   # second triangle of each quad
    out[1::2, 2] = IXq[:, 3]; out[1::2, 3] = domq
    return out


def _fix_ccw(F, X):
    """Reorder each triangle's nodes to counter-clockwise (positive signed area)."""
    F  = F.copy()
    n1 = F[:, 0] - 1
    n2 = F[:, 1] - 1
    n3 = F[:, 2] - 1

    x1, y1 = X[n1, 0], X[n1, 1]
    x2, y2 = X[n2, 0], X[n2, 1]
    x3, y3 = X[n3, 0], X[n3, 1]

    A    = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)  # 2 * signed area
    flip = A < 0                                          # clockwise triangles
    F[flip, 1], F[flip, 2] = F[flip, 2], F[flip, 1].copy()  # swap two nodes -> CCW
    return F


def _sort_elements_structured_like_quad(mesh):
    """Sort triangles into a deterministic, quad-block-aligned order.

    Triangles come in pairs (two per original quad). The pairs are ordered
    by their shared quad centroid via a lexicographic (y, then x) sort so
    that downstream indexing is reproducible across runs and positions.
    """
    X  = mesh["X"]
    IX = mesh["IX"]
    ne = IX.shape[0]

    has_leftover = (ne % 2 != 0)         # odd triangle count -> one unpaired tri
    if has_leftover:
        IX_main     = IX[:-1, :]
        IX_leftover = IX[-1:, :]
        ne_pair     = ne - 1
    else:
        IX_main     = IX
        IX_leftover = np.zeros((0, IX.shape[1]), dtype=IX.dtype)
        ne_pair     = ne

    if ne_pair == 0:
        return {"X": X, "IX": IX.copy()}

    tri1 = np.arange(0, ne_pair, 2)      # first triangle of each pair
    tri2 = np.arange(1, ne_pair, 2)      # second triangle of each pair
    Nq   = ne_pair // 2

    # Quad centroid = mean of the two triangle centroids
    nodes1 = IX_main[tri1, 0:3] - 1
    nodes2 = IX_main[tri2, 0:3] - 1
    Cx = (np.mean(X[nodes1, 0], axis=1) + np.mean(X[nodes2, 0], axis=1)) / 2.0
    Cy = (np.mean(X[nodes1, 1], axis=1) + np.mean(X[nodes2, 1], axis=1)) / 2.0

    order_q = np.lexsort((Cx, Cy))       # sort quads by (y, then x)

    # Re-interleave the two triangles of each quad in the sorted order
    idx_sorted        = np.empty(ne_pair, dtype=int)
    idx_sorted[0::2]  = tri1[order_q]
    idx_sorted[1::2]  = tri2[order_q]

    IX_sorted = IX_main[idx_sorted, :]
    if has_leftover:
        IX_sorted = np.vstack([IX_sorted, IX_leftover])
    return {"X": X, "IX": IX_sorted}


def _make_plunger_sets_blockshift(mesh: dict, Npos: int):
    """Generate the per-position connectivity by shifting the plunger one
    quad-column at a time.

    Starting from the reference layout, at each position the left-most
    plunger column becomes air and the next air column to the right becomes
    plunger. This realizes the rigid-translation of the plunger purely by
    re-labelling domain ids (column 3 of IX), with no remeshing.
    """
    IX0 = np.asarray(mesh["IX"], dtype=int)
    X   = np.asarray(mesh["X"],  dtype=float)
    ne  = IX0.shape[0]

    if ne % 2 != 0:
        raise ValueError("Expected TRI count even (2 per quad).")

    tri1 = np.arange(0, ne, 2, dtype=int)
    tri2 = np.arange(1, ne, 2, dtype=int)
    Nq   = ne // 2

    # Quad centroids (same construction as the sorter)
    nodes1 = IX0[tri1, 0:3] - 1
    nodes2 = IX0[tri2, 0:3] - 1
    Cx = (np.mean(X[nodes1, 0], axis=1) + np.mean(X[nodes2, 0], axis=1)) / 2.0
    Cy = (np.mean(X[nodes1, 1], axis=1) + np.mean(X[nodes2, 1], axis=1)) / 2.0

    cols = np.unique(np.round(Cx, 8))    # distinct quad-column x-positions
    dx   = np.min(np.diff(cols))         # column spacing (one shift step)
    tol  = max(1e-10, dx * 1e-4)         # geometric comparison tolerance

    isPl_q0 = (IX0[tri1, 3] == 5) | (IX0[tri2, 3] == 5)   # quads tagged plunger at pos 0

    if not np.any(isPl_q0):
        return [IX0.copy() for _ in range(Npos)]

    yMin = np.min(Cy[isPl_q0])           # vertical band occupied by the plunger
    yMax = np.max(Cy[isPl_q0])

    IX_all = [IX0.copy()]                # position 0 = reference layout

    for k in range(1, Npos):
        IX_prev = IX_all[k - 1].copy()

        isPl_q = (IX_prev[tri1, 3] == 5) | (IX_prev[tri2, 3] == 5)
        inY    = (Cy >= yMin - tol) & (Cy <= yMax + tol)   # restrict to plunger band
        pl_set = np.where(isPl_q & inY)[0]

        if pl_set.size == 0:
            IX_all.append(IX_prev)
            continue

        xL = np.min(Cx[pl_set])          # current left edge of the plunger
        xR = np.max(Cx[pl_set])          # current right edge of the plunger

        left_col  = np.abs(Cx - xL) < tol            # left-most plunger column
        right_col = np.abs(Cx - (xR + dx)) < tol     # next air column to the right

        # Left plunger column -> air; right air column -> plunger
        q_left  = np.where(left_col  & isPl_q &
                           (IX_prev[tri1, 3] == 5) & inY)[0]
        q_right = np.where(right_col &
                           (IX_prev[tri1, 3] == 1) &
                           (IX_prev[tri2, 3] == 1) & inY)[0]

        tL = np.concatenate([tri1[q_left],  tri2[q_left]])
        tR = np.concatenate([tri1[q_right], tri2[q_right]])

        IX_new        = IX_prev.copy()
        IX_new[tL, 3] = 1   # LEFT  -> AIR
        IX_new[tR, 3] = 5   # RIGHT -> PLUNGER
        IX_all.append(IX_new)

    return IX_all
