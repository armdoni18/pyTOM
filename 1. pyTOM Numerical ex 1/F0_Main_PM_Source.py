import numpy as np

def F0_Main_PM_Source(fem, inputs):
    ndof = int(fem["ndof"])
    T_pm = np.zeros(ndof, dtype=float)

    PM = inputs.get("PM", None)
    if PM is None:
        return T_pm

    # ================================
    # READ INPUT PM (Using Br)
    # ================================
    domIDs   = np.asarray(PM.get("domIDs", []), dtype=int).reshape(-1)
    Br_vals  = np.asarray(PM.get("Br", []), dtype=float).reshape(-1)
    theta_deg = np.asarray(PM.get("theta", []), dtype=float).reshape(-1)

    if domIDs.size == 0:
        return T_pm

    if not (len(domIDs) == len(Br_vals) == len(theta_deg)):
        raise ValueError("PM.domIDs, PM.Br, PM.theta must have same length")

    # ================================
    # CONVERT Br → Hc
    # ================================
    nu0 = 1.0 / (4.0 * np.pi * 1e-7)
    Hc_vals = Br_vals * nu0

    # ================================
    # BUILD H VECTOR MAP
    # ================================
    Hc_map = {}

    # existing domains in mesh
    mesh_domains = np.unique(fem["IX"][:, 3]).astype(int)

    for k in range(len(domIDs)):

        dom = int(domIDs[k])

        # skip if domain not in mesh
        if dom not in mesh_domains:
            continue

        th = np.deg2rad(theta_deg[k])
        Hcx = Hc_vals[k] * np.cos(th)
        Hcy = Hc_vals[k] * np.sin(th)

        Hc_map[dom] = (Hcx, Hcy)

    # optional debug
    print("Active PM domains:", list(Hc_map.keys()))

    # ================================
    # FEM DATA
    # ================================
    IX = fem["IX"][:, :4]   # [n1 n2 n3 dom]
    X  = fem["X"]

    ne = IX.shape[0]
    eps = np.finfo(float).eps

    # ================================
    # MAIN LOOP (EDGE INTEGRATION)
    # ================================
    for e in range(ne):

        dom = int(IX[e, 3])
        if dom not in Hc_map:
            continue

        Hcx, Hcy = Hc_map[dom]

        n1 = int(IX[e, 0])
        n2 = int(IX[e, 1])
        n3 = int(IX[e, 2])

        # Coordinates
        x1, y1 = X[n1-1]
        x2, y2 = X[n2-1]
        x3, y3 = X[n3-1]

        # ================================
        # ORIENTATION CHECK --- Detect orientation (signed area)
        # ================================
        area2 = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

        # If area2 > 0 → CCW
        # If area2 < 0 → CW

        orientation = 1.0 if area2 > 0 else -1.0

        edges = [(n1, n2), (n2, n3), (n3, n1)]

        for (na, nb) in edges:

            xa, ya = X[na-1]
            xb, yb = X[nb-1]

            vx = xb - xa
            vy = yb - ya
            l = np.hypot(vx, vy)

            if l <= eps:
                continue

            # Tangent
            tx = vx / l
            ty = vy / l

            # ================================
            # OUTWARD NORMAL

            # ================================
            #  For CCW:
            #  n_out = [ ty, -tx ]
            #  For CW:
            #  n_out must be flipped

            nx = orientation * ty
            ny = -orientation * tx

            # ================================
            # PM CONTRIBUTION
            # ================================
            # Kb = (Hc × n)
            Kbz = Hcx * ny - Hcy * nx

            # Edge integral
            I_edge = Kbz * l

            # Lump to nodes
            I2 = 0.5 * I_edge
            T_pm[na-1] += I2
            T_pm[nb-1] += I2

    return T_pm
