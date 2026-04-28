import numpy as np

def F0_Main_PM_Source(fem, inputs):

    ndof = int(fem["ndof"])
    T_pm = np.zeros(ndof, dtype=float)

    PM = inputs.get("PM", None)
    if PM is None:
        return T_pm

    domIDs    = np.asarray(PM.get("domIDs", []), dtype=int).reshape(-1)
    Br_vals   = np.asarray(PM.get("Br",     []), dtype=float).reshape(-1)
    theta_deg = np.asarray(PM.get("theta",  []), dtype=float).reshape(-1)

    if domIDs.size == 0:
        return T_pm

    if not (len(domIDs) == len(Br_vals) == len(theta_deg)):
        raise ValueError("PM.domIDs, PM.Br, PM.theta must have same length")

    mu0     = 4.0 * np.pi * 1e-7
    Hc_vals = Br_vals / mu0

    # Build Hc map per domain
    Hc_map = {}
    for k in range(len(domIDs)):
        th  = np.deg2rad(theta_deg[k])
        Hcx = Hc_vals[k] * np.cos(th)
        Hcy = Hc_vals[k] * np.sin(th)
        Hc_map[int(domIDs[k])] = (Hcx, Hcy)

    IX = fem["IX"][:, :4]
    X  = fem["X"]
    eps = np.finfo(float).eps

    # Process each PM domain
    for dom_id, (Hcx, Hcy) in Hc_map.items():
        mask = (IX[:, 3] == dom_id)
        if not np.any(mask):
            continue

        elems = IX[mask, :3]

        na_all = np.concatenate([elems[:, 0], elems[:, 1], elems[:, 2]])
        nb_all = np.concatenate([elems[:, 1], elems[:, 2], elems[:, 0]])

        xa = X[na_all - 1, 0];  ya = X[na_all - 1, 1]
        xb = X[nb_all - 1, 0];  yb = X[nb_all - 1, 1]

        vx = xb - xa
        vy = yb - ya
        l  = np.hypot(vx, vy)

        valid = l > eps

        # orientation per element (signed area)
        n1 = elems[:, 0]; n2 = elems[:, 1]; n3 = elems[:, 2]
        x1 = X[n1-1,0]; y1 = X[n1-1,1]
        x2 = X[n2-1,0]; y2 = X[n2-1,1]
        x3 = X[n3-1,0]; y3 = X[n3-1,1]
        area2 = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        # orientation repeated 3 times (one per edge)
        orient = np.where(np.concatenate([area2, area2, area2]) > 0, 1.0, -1.0)

        # tangent
        tx = np.where(valid, vx / (l + eps), 0.0)
        ty = np.where(valid, vy / (l + eps), 0.0)

        # outward normal
        nx_out =  orient * ty
        ny_out = -orient * tx

        # PM contribution: Kbz = Hcx*ny - Hcy*nx
        Kbz   = Hcx * ny_out - Hcy * nx_out
        I_edge = np.where(valid, Kbz * l, 0.0)
        I2     = 0.5 * I_edge

        # Accumulate to nodes (lump to both endpoints)
        np.add.at(T_pm, na_all - 1, I2)
        np.add.at(T_pm, nb_all - 1, I2)

    return T_pm
