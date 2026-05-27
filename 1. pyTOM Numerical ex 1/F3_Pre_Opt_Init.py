"""
F3_Pre_Opt_Init.py — Numerical Example 1 (IPM motor benchmark)
==================================================================

Optimization-side pre-processing for Example 1.

NOTE: Example 1 is a magnetic-field validation benchmark and does not actually perform topology optimization.
This module is present for structural consistency with the other examples but is invoked in field-only mode.

Refer to the Example 3 version (``3. pyTOM Numerical ex 3/F3_Pre_Opt_Init.py``) for the full documentation of the optimization initialization
(design-domain DOF separation, design-variable bounds, Helmholtz filter assembly with cached LU, element-to-nodal averaging matrix, MMA parameters).
"""

import numpy as np
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def F3_Pre_Opt_Init(inputs, fem):

    opt = {}
    opt["f"] = []
    opt["g"] = []

    IX = fem["IX"]
    X  = fem["X"]
    ne = fem["ne"]
    nn = fem["nn"]

    # -------------------------------------------------------
    # Design / non-design separation
    # -------------------------------------------------------
    nd_values   = np.array([1, 3, 4, 5, 6, 7, 8])
    nd_ele      = np.where(np.isin(IX[:, 3], nd_values))[0]
    opt["dof_nd"] = np.unique(IX[nd_ele, 0:3].flatten())
    opt["dof_dd"] = np.setdiff1d(np.arange(1, nn + 1), opt["dof_nd"])
    opt["nn_dd"]  = opt["dof_dd"].shape[0]

    dd_ele         = 2
    opt["erho_dd"] = np.where(IX[:, 3] == dd_ele)[0] + 1
    opt["ne_dd"]   = opt["erho_dd"].shape[0]

    # -------------------------------------------------------
    # Problem parameters
    # -------------------------------------------------------
    opt["VT"]      = inputs["VT"]
    opt["VND"]     = inputs["VND"]
    opt["VDD"]     = inputs["VDD"]
    opt["volfrac"] = inputs["volfrac"]
    opt["penal"]   = inputs["penal"]

    # -------------------------------------------------------
    # Optimization variables
    # -------------------------------------------------------
    opt["dv"] = np.ones((len(opt["dof_dd"]), 1)) * inputs["initdv"]
    opt["nv"] = np.zeros((nn, 1))
    opt["nv"][opt["dof_nd"] - 1] = 1.0
    opt["nv"][opt["dof_dd"] - 1] = opt["dv"]

    opt["dvold"]    = opt["dv"].copy()
    opt["dvolder"]  = opt["dv"].copy()
    opt["dvmin"]    = opt["dv"] * 0 - 1
    opt["dvmax"]    = opt["dv"] * 0 + 1

    opt["iter"]      = 1
    opt["deltaf"]    = 1.0
    opt["bt"]        = inputs["bt_init"]
    opt["cont_sw"]   = 0
    opt["cont_iter"] = 0

    # -------------------------------------------------------
    # MMA parameters
    # -------------------------------------------------------
    MMA = {}
    MMA["a0"]  = 1.0
    MMA["a"]   = np.zeros((1, 1))
    MMA["c"]   = np.array([[1.0]])
    MMA["d"]   = np.ones((1, 1))
    MMA["low"] = opt["dvmin"]
    MMA["upp"] = opt["dvmax"]
    opt["MMA"] = MMA

    # -------------------------------------------------------
    # Helmholtz filter assembly
    # -------------------------------------------------------
    n0 = IX[:, 0] - 1
    n1 = IX[:, 1] - 1
    n2 = IX[:, 2] - 1

    x0 = X[n0, 0]; y0 = X[n0, 1]
    x1 = X[n1, 0]; y1 = X[n1, 1]
    x2 = X[n2, 0]; y2 = X[n2, 1]

    Ae = fem["Ae"]

    b_n = np.column_stack([y1 - y2, y2 - y0, y0 - y1])
    c_n = np.column_stack([x2 - x1, x0 - x2, x1 - x0])
    inv4A = 1.0 / (4.0 * Ae)

    Kd_scale = (inputs["rmin"] / (2.0 * np.sqrt(3.0))) ** 2

    # Diffusion part of Helmholtz filter:
    Se_diff = Kd_scale * (
        np.einsum('ei,ej->eij', c_n, c_n) + np.einsum('ei,ej->eij', b_n, b_n)
    ) * inv4A[:, None, None]

    NN_base = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float) / 12.0
    Se_mass = Ae[:, None, None] * NN_base[None, :, :]

    Se_filt = Se_diff + Se_mass
    Se_tft  = Se_mass

    isf = np.reshape(np.kron(IX[:, 0:3], np.ones((1, 3), dtype=int)), (9 * ne,))
    jsf = np.reshape(np.kron(IX[:, 0:3], np.ones((3, 1), dtype=int)), (9 * ne,))

    Kft_vals = Se_filt.reshape(-1)
    Tft_vals = Se_tft.reshape(-1)

    Kft_sparse = coo_matrix((Kft_vals, (isf - 1, jsf - 1)), shape=(nn, nn)).tocsc()
    opt["Kft_sparse"] = Kft_sparse

    LU = splu(Kft_sparse, permc_spec="NATURAL")
    opt["lu_L_Kft"] = LU.L
    opt["lu_U_Kft"] = LU.U

    opt["Tft"] = coo_matrix((Tft_vals, (isf - 1, jsf - 1)), shape=(nn, nn)).tocsc()

    # -------------------------------------------------------
    # Element->nodal averaging matrix
    # -------------------------------------------------------
    rows_ten = np.repeat(np.arange(ne), 3)
    cols_ten = IX[:, 0:3].reshape(-1) - 1
    data_ten = np.full(3 * ne, 1.0 / 3.0)
    opt["Ten"] = sp.coo_matrix((data_ten, (rows_ten, cols_ten)),
                               shape=(ne, nn)).tocsr()

    print("Optimization Initialization Done ✅")
    return (opt, MMA)
