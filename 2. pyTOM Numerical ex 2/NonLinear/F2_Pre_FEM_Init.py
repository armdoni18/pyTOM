"""
F2_Pre_FEM_Init.py
==================

Finite-element pre-processing for the magnetostatic problem.

This module is called once before the topology-optimization loop
begins. It produces three things that are reused on every iteration
of the outer TO loop and the inner Newton-Raphson loop:

1. The element-level stiffness building block ``K0_e`` (without
   the reluctivity factor), stored in flat form as ``fem["S_S"]``
   together with COO triplet indices ``fem["is"]`` and
   ``fem["js"]``. This implements the element kernel of Eq. (21)
   in the form  K0_e = (b o b + c o c) / (4 * A_e).

2. The Dirichlet boundary-condition information ``bcdof``,
   ``bcval``. The constrained nodes are detected geometrically
   from the bounding box of the actuator domain
   detected from the mesh extents (rectangular actuator domain).

3. The coil current source vector data ``Tdof``, ``Tval``,
   contributing to f in Eq. (2). Each coil element distributes
   J * A_e / 3 to each of its three nodes, with opposite signs
   for the two coil regions modelling the in-and-out return path.
"""

import numpy as np

def F2_Pre_FEM_Init(inputs, mesh):
    """
    Build the FEM data structures used by F4, F5, F6, F7, F8.

    Parameters
    ----------
    inputs : dict
        Problem inputs. Used here for ``inputs["J_am2"]`` (coil
        current density).
    mesh : dict
        Mesh data from ``F1_Pre_Mesh_Import``. Must contain
        ``IX`` (element connectivity), ``X`` (node coordinates).

    Returns
    -------
    fem : dict
        Finite-element data structure with the following entries:

            IX, X, nn, ndof, ne   : mesh and DOF counts
            edof                  : (ne, 3) element-to-DOF map
            is, js                : COO triplet row/col indices
            D                     : 2D constitutive identity
            nx, ny                : (ne, 3) nodal coords per elem
            Ae                    : (ne,)   element areas
            S_S                   : (9*ne,) flat K0_e kernel
            bcdof, bcval          : Dirichlet info (1-based dofs)
            ncoil1, ncoil2        : (nc, 3) coil node sets
            c_A_all_coil1/2       : (nc,) coil element areas
            Tdof, Tval            : coil source triplets (1-based)
    """

# Basic mesh / DOF bookkeeping (scalar A_z problem: 1 DOF per node)
    fem = {}
    fem["IX"]   = mesh["IX"]                 # element connectivity (ne, >=4)
    fem["X"]    = mesh["X"]                  # nodal coordinates (nn, 2)
    fem["nn"]   = fem["X"].shape[0]          # number of nodes
    fem["ndof"] = fem["nn"]                  # one DOF (A_z) per node
    fem["ne"]   = fem["IX"].shape[0]         # number of elements
    fem["edof"] = np.array([fem["IX"][:, 0],
                            fem["IX"][:, 1],
                            fem["IX"][:, 2]], dtype=int).T   # element-to-DOF map (ne, 3)

# COO triplet index arrays for the 3x3 element blocks (reused by F4 and F6)
    # "is" repeats each node row-wise, "js" tiles it column-wise, so that
    # (is, js) enumerate the 9 (row, col) positions of every element block.
    fem["is"] = np.reshape(np.kron(fem["edof"], np.ones((1, 3), dtype=int)), 9 * fem["ne"])
    fem["js"] = np.reshape(np.kron(fem["edof"], np.ones((3, 1), dtype=int)), 9 * fem["ne"])
    fem["D"]  = np.eye(2, dtype=float)       # 2D constitutive identity (isotropic)

# Per-element nodal coordinates
    IX  = fem["IX"]
    X   = fem["X"]
    ne  = fem["ne"]
    n0 = IX[:, 0] - 1                        # node indices (0-based)
    n1 = IX[:, 1] - 1
    n2 = IX[:, 2] - 1
    x0 = X[n0, 0]; y0 = X[n0, 1]             # coords of node 0
    x1 = X[n1, 0]; y1 = X[n1, 1]             # coords of node 1
    x2 = X[n2, 0]; y2 = X[n2, 1]             # coords of node 2
    fem["nx"] = np.column_stack([x0, x1, x2])   # (ne, 3) x-coords
    fem["ny"] = np.column_stack([y0, y1, y2])   # (ne, 3) y-coords

# Element area:  A_e = 0.5 * |det([1 x0 y0; 1 x1 y1; 1 x2 y2])|
    Ae = 0.5 * np.abs(
        (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    )
    fem["Ae"] = Ae                           # (ne,)

# Shape-function gradient coefficients (b_i, c_i)
    #   b_0 = y_1 - y_2,  b_1 = y_2 - y_0,  b_2 = y_0 - y_1
    #   c_0 = x_2 - x_1,  c_1 = x_0 - x_2,  c_2 = x_1 - x_0
    b = np.column_stack([y1 - y2, y2 - y0, y0 - y1])   # (ne, 3)
    c = np.column_stack([x2 - x1, x0 - x2, x1 - x0])   # (ne, 3)
    inv2A = 1.0 / (2.0 * Ae)                 # 1/(2 A_e) per element

# Reluctivity-free element kernel  K0_e = (b o b + c o c)/(4 A_e)  -- Eq. (21)
    # Vectorized outer products over all elements; the 0.25 factor turns the
    # 1/(2 A_e) into the required 1/(4 A_e). nu_e is applied later (F4/F6).
    Se_all = (
        np.einsum('ei,ej->eij', c, c) + np.einsum('ei,ej->eij', b, b)
    ) * (inv2A * 0.25)[:, None, None]
    fem["S_S"] = Se_all.reshape(-1)          # flat (9*ne,) for COO assembly

# Dirichlet boundary conditions: A_z = 0 on the rectangular domain
    # bounding box. The edges are detected from the mesh extents so that
    # the SAME routine serves every actuator mesh without hard-coded
    # coordinates (Example 2: 390 x 250 mm; Example 3: 120 x 140 mm).
    tol = 1e-9
    X_   = np.asarray(mesh["X"], dtype=float)
    if X_.ndim == 1:
        X_ = X_.reshape(-1, 2)
    x_min, x_max = X_[:, 0].min(), X_[:, 0].max()
    y_min, y_max = X_[:, 1].min(), X_[:, 1].max()
    left0   = np.where(np.abs(X_[:, 0] - x_min) < tol)[0]   # left edge nodes
    right0  = np.where(np.abs(X_[:, 0] - x_max) < tol)[0]   # right edge nodes
    top0    = np.where(np.abs(X_[:, 1] - y_min) < tol)[0]   # bottom edge nodes
    bottom0 = np.where(np.abs(X_[:, 1] - y_max) < tol)[0]   # top edge nodes
    bcdof0      = np.unique(np.concatenate([left0, right0, top0, bottom0])).astype(int)
    fem["bcdof"] = bcdof0 + 1                # store as 1-based DOF indices
    fem["bcval"] = np.zeros(bcdof0.size, dtype=float)       # homogeneous (A_z = 0)

# Coil current source -> contributes to f in Eq. (2)
    # Coil 1 (domain 3): + sign ; coil 2 (domain 4): - sign (return path).
    J_val = float(inputs["J_am2"])           # coil current density
    Tdof_list = []
    Tval_list = []

    for coil_dom, sign in [(3, +1.0), (4, -1.0)]:
        mask   = (IX[:, 3] == coil_dom)      # elements in this coil domain
        coil_e = IX[mask, :]
        key    = "ncoil1" if coil_dom == 3 else "ncoil2"

        if coil_e.shape[0] == 0:             # coil domain empty -> store empties
            fem[key] = np.zeros((0, 3), dtype=int)
            mesh[key] = fem[key]
            c_A_key = "c_A_all_coil1" if coil_dom == 3 else "c_A_all_coil2"
            fem[c_A_key] = np.zeros(0, dtype=float)
            continue

        nodes_c = coil_e[:, 0:3].astype(int) # coil element node table (nc, 3), 1-based
        fem[key]  = nodes_c
        mesh[key] = nodes_c

        # Coil element areas (same triangle-area formula as above)
        na = nodes_c[:, 0] - 1
        nb = nodes_c[:, 1] - 1
        nc_ = nodes_c[:, 2] - 1
        v1 = X[na, :]
        v2 = X[nb, :]
        v3 = X[nc_, :]
        c_A = 0.5 * np.abs(
            (v1[:, 0] - v3[:, 0]) * (v2[:, 1] - v3[:, 1]) -
            (v1[:, 1] - v3[:, 1]) * (v2[:, 0] - v3[:, 0])
        )
        c_A_key = "c_A_all_coil1" if coil_dom == 3 else "c_A_all_coil2"
        fem[c_A_key] = c_A

        # Per-node load: sign * (A_e / 3) * J  -> lumped equally to 3 nodes
        val = sign * (c_A / 3.0) * J_val     # (nc,)
        Tdof_list.append(nodes_c[:, 0])      # node 1 of each coil element
        Tdof_list.append(nodes_c[:, 1])      # node 2
        Tdof_list.append(nodes_c[:, 2])      # node 3
        Tval_list.append(val)
        Tval_list.append(val)
        Tval_list.append(val)

    fem["Tdof"] = np.concatenate(Tdof_list).astype(int)     # coil source node indices
    fem["Tval"] = np.concatenate(Tval_list).astype(float)   # coil source values

    print("FEM Initialization Done ✅")
    return fem
