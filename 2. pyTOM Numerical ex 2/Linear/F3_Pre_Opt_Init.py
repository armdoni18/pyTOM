"""
F3_Pre_Opt_Init.py
==================

Optimization-side pre-processing. This module is called once
before the topology-optimization loop and produces:

1. The design-domain / non-design-domain DOF partition.

2. The initial design variables. The design-variable bounds are
   intentionally set to [-1, +1] rather than [0, 1] as a
   numerical safeguard against transient overshoots that can
   occur in early iterations when the projection sharpness beta
   is small. The Heaviside projection of Eq. (19), followed by
   hard clipping in the main driver, maps these bounds onto a
   strictly bounded physical-density range that approaches [0,1]
   as beta increases. The same bound extension is also adopted
   in other recent educational implementations (e.g. the
   node-based MATLAB code of Kim et al., 2023). The optimization
   formulation of Eqs. (10)-(14) is unchanged.

3. The Helmholtz filter operator of Eq. (18). The filter is
   assembled as a 2D linear PDE (-R^2 * Laplacian + I) * phi_tilde
   = M * phi, with R = rmin / (2 * sqrt(3)) following Lazarov &
   Sigmund (2011). The LU factorization of the filter matrix is
   cached so that each subsequent TO iteration only needs a
   back-substitution.

4. The element-to-nodal averaging matrix ``Ten`` corresponding to
   the linear interpolation of Eq. (17): Ten @ nodal_field gives
   the element-wise average of the three nodal values.

5. The MMA optimizer parameters (a0, a, c, d, bounds), in the
   format expected by ``mma.mmasub``.
"""

import numpy as np
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from scipy.sparse import coo_matrix, lil_matrix

def F3_Pre_Opt_Init(inputs, fem):
    """
    Initialize the topology-optimization side of pyTOM.

    Parameters
    ----------
    inputs : dict
        Problem inputs (volume fraction, penalization, beta
        schedule, filter radius rmin, MMA tuning constant, etc.).
    fem : dict
        Finite-element data built by ``F2_Pre_FEM_Init``.

    Returns
    -------
    opt : dict
        Optimization state and operators (design variables,
        filter LU factors, Ten matrix, continuation state, ...).
    MMA : dict
        MMA tuning parameters (a0, a, c, d, bounds).
    """

# Optimization-state container and convergence-history lists
    opt = {}
    opt["f"] = []                            # objective history
    opt["g"] = []                            # constraint history

    IX = fem["IX"]
    X  = fem["X"]
    ne = fem["ne"]
    nn = fem["nn"]

# Partition DOFs into non-design and design sets
    # Non-design domains: air(1), coil1(3), coil2(4), plunger(5),
    # fix-iron(6), PM(7). The design domain (yoke) carries id 2.
    values     = np.array([1, 3, 4, 5, 6, 7])
    nd_ele     = np.where(np.isin(IX[:, 3], values))[0]          # non-design elements
    opt["dof_nd"]  = np.unique(IX[nd_ele, 0:3].flatten())        # non-design nodes (1-based)
    opt["dof_dd"]  = np.setdiff1d(np.arange(1, nn + 1), opt["dof_nd"])  # design nodes
    opt["nn_dd"]   = opt["dof_dd"].shape[0]                      # number of design nodes

    dd_ele         = 2                       # design-domain element id
    opt["erho_dd"] = np.where(IX[:, 3] == dd_ele)[0] + 1         # design elements (1-based)
    opt["ne_dd"]   = opt["erho_dd"].shape[0]                     # number of design elements

# Problem parameters carried into the optimization state
    opt["VT"]      = inputs["VT"]            # total volume
    opt["VND"]     = inputs["VND"]           # non-design volume
    opt["VDD"]     = inputs["VDD"]           # design-domain volume
    opt["volfrac"] = inputs["volfrac"]       # target volume fraction
    opt["penal"]   = inputs["penal"]         # SIMP penalization power p

# Initial design variables (bounds extended to [-1, +1]; see docstring)
    # The uniform start at initdv (default -0.5) gives a near-empty
    # topology once the Heaviside projection is applied.
    opt["dv"]      = np.ones((len(opt["dof_dd"]), 1)) * inputs["initdv"]
    opt["nv"]      = np.zeros((nn, 1))       # full nodal design field
    opt["nv"][opt["dof_nd"] - 1] = 1.0       # non-design nodes pinned to solid
    opt["nv"][opt["dof_dd"] - 1] = opt["dv"] # design nodes take the design variable

    opt["dvold"]   = opt["dv"].copy()        # MMA history: previous iterate
    opt["dvolder"] = opt["dv"].copy()        # MMA history: iterate before that
    opt["dvmin"]   = opt["dv"] * 0 - 1       # lower bound = -1
    opt["dvmax"]   = opt["dv"] * 0 + 1       # upper bound = +1

    opt["iter"]    = 1                       # TO iteration counter
    opt["deltaf"]  = 1.0                     # last relative objective change
    opt["bt"]      = inputs["bt_init"]       # current projection sharpness beta
    opt["cont_sw"] = 0                       # continuation switch (off until triggered)
    opt["cont_iter"] = 0                     # iterations since continuation began

# MMA tuning parameters (single constraint: volume only)
    MMA = {}
    MMA["a0"]  = 1.0
    MMA["a"]   = np.zeros((1, 1))
    MMA["c"]   = np.array([[1.0]])
    MMA["d"]   = np.ones((1, 1))
    MMA["low"] = opt["dvmin"]                # initial MMA lower asymptote
    MMA["upp"] = opt["dvmax"]                # initial MMA upper asymptote
    opt["MMA"] = MMA

# Helmholtz filter assembly  -- Eq. (18):  (K_d + K_m) phi_tilde = K_m phi
    n0 = IX[:, 0] - 1
    n1 = IX[:, 1] - 1
    n2 = IX[:, 2] - 1
    x0 = X[n0, 0]; y0 = X[n0, 1]
    x1 = X[n1, 0]; y1 = X[n1, 1]
    x2 = X[n2, 0]; y2 = X[n2, 1]
    Ae = fem["Ae"]

    # Same gradient coefficients (b_i, c_i) as in F2
    b_n = np.column_stack([y1 - y2, y2 - y0, y0 - y1])   # (ne, 3)
    c_n = np.column_stack([x2 - x1, x0 - x2, x1 - x0])   # (ne, 3)
    inv2A = 1.0 / (2.0 * Ae)

    # Filter radius R = rmin/(2 sqrt(3)) (Lazarov & Sigmund 2011); store R^2
    Kd_scale = (inputs["rmin"] / (2.0 * np.sqrt(3.0))) ** 2

    # Diffusion (Laplacian) part:  K_d_e = R^2 (b o b + c o c)/(4 A_e)
    Se_diff = Kd_scale * (
        np.einsum('ei,ej->eij', c_n, c_n) + np.einsum('ei,ej->eij', b_n, b_n)
    ) * (inv2A * 0.25)[:, None, None]        # same kernel form as F2's S_S

    # Consistent mass part:  K_m_e = (A_e/12) [[2,1,1],[1,2,1],[1,1,2]]
    NN_base = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float) / 12.0
    Se_mass = Ae[:, None, None] * NN_base[None, :, :]    # (ne, 3, 3)

    Se_filt = Se_diff + Se_mass              # filter LHS operator (K_d + K_m)
    Se_tft  = Se_mass                        # filter RHS operator (K_m)

    # COO triplet indices for the nn x nn filter system
    isf = np.reshape(np.kron(IX[:, 0:3], np.ones((1, 3), dtype=int)), (9 * ne,))
    jsf = np.reshape(np.kron(IX[:, 0:3], np.ones((3, 1), dtype=int)), (9 * ne,))
    Kft_vals = Se_filt.reshape(-1)
    Tft_vals = Se_tft.reshape(-1)

    Kft_sparse     = coo_matrix((Kft_vals, (isf - 1, jsf - 1)), shape=(nn, nn)).tocsc()
    opt["Kft_sparse"] = Kft_sparse

    # Cache the LU factors: each later filter solve is just a back-substitution
    LU = splu(Kft_sparse, permc_spec="NATURAL")
    opt["lu_L_Kft"] = LU.L
    opt["lu_U_Kft"] = LU.U

    opt["Tft"] = coo_matrix((Tft_vals, (isf - 1, jsf - 1)), shape=(nn, nn)).tocsc()

# Element-to-nodal averaging matrix Ten  -- discrete form of Eq. (17)
    # Row e holds three 1/3 entries at the columns of element e's nodes,
    # so Ten @ nodal_field returns the per-element average of its 3 nodes.
    rows_ten = np.repeat(np.arange(ne), 3)   # element (row) index, repeated 3x
    cols_ten = IX[:, 0:3].reshape(-1) - 1    # the three node (column) indices
    data_ten = np.full(3 * ne, 1.0 / 3.0)    # equal 1/3 weights
    opt["Ten"] = sp.coo_matrix((data_ten, (rows_ten, cols_ten)),
                               shape=(ne, nn)).tocsr()

    print("Optimization Initialization Done ✅")
    return (opt, MMA)
