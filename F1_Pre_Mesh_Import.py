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
        "NonDesign": [], "FixIron": [], "PM1": []
    }
    domain_entities = {k: [] for k in phys_tags.keys()}

    t_tri  = []
    q_quad = []
    p      = None

    i      = 0
    nlines = len(lines)

    # =====================
    # PARSING
    # =====================
    while i < nlines:
        s = lines[i].strip()

        if s == "$PhysicalNames":
            i += 1
            nphys = int(lines[i])
            for _ in range(nphys):
                i += 1
                line = lines[i]
                if '"' in line:
                    name = line.split('"')[1]
                    tag  = int(line.split()[1])
                    if name in phys_tags:
                        phys_tags[name].append(tag)
            i += 1

        elif s == "$Entities":
            i += 1
            header      = list(map(int, lines[i].split()))
            num_surfaces = header[2]

            for _ in range(header[0] + header[1]):
                i += 1

            for _ in range(num_surfaces):
                i += 1
                parts    = lines[i].split()
                etag     = int(parts[0])
                numPhys  = int(parts[7])
                physIDs  = list(map(int, parts[8:8+numPhys]))

                for name, tags in phys_tags.items():
                    if any(pid in tags for pid in physIDs):
                        domain_entities[name].append(etag)

            for _ in range(header[3]):
                i += 1
            i += 1

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

                    if etype == 2:
                        t_tri.append([parts[1], parts[2], parts[3], dom])
                    elif etype == 3:
                        q_quad.append([parts[1], parts[2], parts[3], parts[4], dom])

        else:
            i += 1

    # =====================
    # PROCESS
    # =====================
    t_tri  = np.array(t_tri,  dtype=int) if t_tri  else None
    q_quad = np.array(q_quad, dtype=int) if q_quad else None

    IX_quad = None
    IX_tri  = None

    if q_quad is not None:
        IX_quad = _simple_quad_to_tri(q_quad[:, 0:4], q_quad[:, 4])
        IX_quad[:, 0:3] = _fix_ccw(IX_quad[:, 0:3], p)

    if t_tri is not None:
        t_tri[:, 0:3] = _fix_ccw(t_tri[:, 0:3], p)
        IX_tri = t_tri

    IX_parts = []
    if IX_quad is not None:
        IX_parts.append(IX_quad)
    if IX_tri is not None:
        IX_parts.append(IX_tri)

    IX   = np.vstack(IX_parts) if IX_parts else np.zeros((0, 4), int)
    mesh = {"X": p, "IX": IX}
    mesh = _sort_elements_structured_like_quad(mesh)

    # =====================
    # MULTI POSITION
    # =====================
    IX_all = []

    if Npos > 1 and IX_quad is not None:
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
        IX_all = [mesh["IX"].copy() for _ in range(Npos)]

    return mesh, IX_all


# =====================
# HELPERS
# =====================

def _get_domain_id(entityTag, domain_entities):
    if entityTag in domain_entities["Air"]:       return 1
    if entityTag in domain_entities["Design"]:    return 2
    if entityTag in domain_entities["Coil1"]:     return 3
    if entityTag in domain_entities["Coil2"]:     return 4
    if entityTag in domain_entities["NonDesign"]: return 5
    if entityTag in domain_entities["FixIron"]:   return 6
    if entityTag in domain_entities["PM1"]:       return 7
    return 0


def _simple_quad_to_tri(IXq, domq):
    n    = IXq.shape[0]
    out  = np.zeros((2 * n, 4), dtype=int)
    out[0::2, 0] = IXq[:, 0]; out[0::2, 1] = IXq[:, 1]
    out[0::2, 2] = IXq[:, 2]; out[0::2, 3] = domq
    out[1::2, 0] = IXq[:, 0]; out[1::2, 1] = IXq[:, 2]
    out[1::2, 2] = IXq[:, 3]; out[1::2, 3] = domq
    return out


def _fix_ccw(F, X):
    F  = F.copy()
    n1 = F[:, 0] - 1
    n2 = F[:, 1] - 1
    n3 = F[:, 2] - 1

    x1, y1 = X[n1, 0], X[n1, 1]
    x2, y2 = X[n2, 0], X[n2, 1]
    x3, y3 = X[n3, 0], X[n3, 1]

    A    = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    flip = A < 0
    F[flip, 1], F[flip, 2] = F[flip, 2], F[flip, 1].copy()
    return F


def _sort_elements_structured_like_quad(mesh):
    X  = mesh["X"]
    IX = mesh["IX"]
    ne = IX.shape[0]

    has_leftover = (ne % 2 != 0)
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

    tri1 = np.arange(0, ne_pair, 2)
    tri2 = np.arange(1, ne_pair, 2)
    Nq   = ne_pair // 2

    # Vectorized centroid per quad
    nodes1 = IX_main[tri1, 0:3] - 1   # (Nq,3)
    nodes2 = IX_main[tri2, 0:3] - 1   # (Nq,3)

    Cx = (np.mean(X[nodes1, 0], axis=1) + np.mean(X[nodes2, 0], axis=1)) / 2.0
    Cy = (np.mean(X[nodes1, 1], axis=1) + np.mean(X[nodes2, 1], axis=1)) / 2.0

    order_q = np.lexsort((Cx, Cy))

    idx_sorted        = np.empty(ne_pair, dtype=int)
    idx_sorted[0::2]  = tri1[order_q]
    idx_sorted[1::2]  = tri2[order_q]

    IX_sorted = IX_main[idx_sorted, :]

    if has_leftover:
        IX_sorted = np.vstack([IX_sorted, IX_leftover])

    return {"X": X, "IX": IX_sorted}


def _make_plunger_sets_blockshift(mesh: dict, Npos: int):
    IX0 = np.asarray(mesh["IX"], dtype=int)
    X   = np.asarray(mesh["X"],  dtype=float)
    ne  = IX0.shape[0]

    if ne % 2 != 0:
        raise ValueError("Expected TRI count even (2 per quad).")

    tri1 = np.arange(0, ne, 2, dtype=int)
    tri2 = np.arange(1, ne, 2, dtype=int)
    Nq   = ne // 2

    # Vectorized centroid
    nodes1 = IX0[tri1, 0:3] - 1
    nodes2 = IX0[tri2, 0:3] - 1
    Cx = (np.mean(X[nodes1, 0], axis=1) + np.mean(X[nodes2, 0], axis=1)) / 2.0
    Cy = (np.mean(X[nodes1, 1], axis=1) + np.mean(X[nodes2, 1], axis=1)) / 2.0

    cols = np.unique(np.round(Cx, 8))
    dx   = np.min(np.diff(cols))
    tol  = max(1e-10, dx * 1e-4)

    isPl_q0 = (IX0[tri1, 3] == 5) | (IX0[tri2, 3] == 5)

    if not np.any(isPl_q0):
        return [IX0.copy() for _ in range(Npos)]

    yMin = np.min(Cy[isPl_q0])
    yMax = np.max(Cy[isPl_q0])

    IX_all = [IX0.copy()]

    for k in range(1, Npos):
        IX_prev = IX_all[k - 1].copy()

        isPl_q = (IX_prev[tri1, 3] == 5) | (IX_prev[tri2, 3] == 5)
        inY    = (Cy >= yMin - tol) & (Cy <= yMax + tol)
        pl_set = np.where(isPl_q & inY)[0]

        if pl_set.size == 0:
            IX_all.append(IX_prev)
            continue

        xL = np.min(Cx[pl_set])
        xR = np.max(Cx[pl_set])

        left_col  = np.abs(Cx - xL) < tol
        right_col = np.abs(Cx - (xR + dx)) < tol

        q_left  = np.where(left_col  & isPl_q &
                           (IX_prev[tri1, 3] == 5) & inY)[0]
        q_right = np.where(right_col &
                           (IX_prev[tri1, 3] == 1) &
                           (IX_prev[tri2, 3] == 1) & inY)[0]

        tL = np.concatenate([tri1[q_left],  tri2[q_left]])
        tR = np.concatenate([tri1[q_right], tri2[q_right]])

        IX_new       = IX_prev.copy()
        IX_new[tL, 3] = 1   # LEFT → AIR
        IX_new[tR, 3] = 5   # RIGHT → PLUNGER

        IX_all.append(IX_new)

    return IX_all
