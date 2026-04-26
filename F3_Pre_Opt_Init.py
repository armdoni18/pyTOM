import numpy as np
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix

def F3_Pre_Opt_Init(inputs, fem):
    opt = {}
    opt["f"] = []
    opt["g"] = []
    # Separation of other than design domain
    values = np.array([1, 3, 4, 5, 6, 7])
    nd_ele          = np.where(np.isin(fem["IX"][:, 3], values))[0]
    opt["dof_nd"]   = np.unique(fem["IX"][nd_ele, 0:3].flatten())
    opt["dof_dd"]   = np.setdiff1d(np.arange(1, fem["nn"] + 1), opt["dof_nd"])
    opt["nn_dd"]    = opt["dof_dd"].shape[0]

    # Separation of design domain
    dd_ele = 2
    opt["erho_dd"] = np.where(fem["IX"][:, 3] == dd_ele)[0]+1
    opt["ne_dd"]   = opt["erho_dd"].shape[0]

    # Setting problem parameters
    opt["VT"]      = inputs["VT"]
    opt["VND"]     = inputs["VND"]
    opt["VDD"]     = inputs["VDD"]
    opt["volfrac"] = inputs["volfrac"]
    opt["penal"]   = inputs["penal"]

    # Setting of optimization variables
    opt["dv"] = np.ones((len(opt["dof_dd"]), 1)) * inputs["initdv"]
    opt["nv"] = np.zeros((fem["nn"], 1))
    opt["nv"][opt["dof_nd"] - 1] = 1.0
    opt["nv"][opt["dof_dd"] - 1] = opt["dv"]

    opt["dvold"] = opt["dv"].copy()
    opt["dvolder"] = opt["dv"].copy()
    opt["dvmin"] = opt["dv"] * 0 - 1
    opt["dvmax"] = opt["dv"] * 0 + 1

    opt["iter"] = 1
    opt["deltaf"] = 1.0
    opt["deltaf2"] = 1.0
    opt["deltaf3"] = 1.0
    opt["bt"] = inputs["bt_init"]

    opt["cont_sw"] = 0
    opt["cont_iter"] = 0

    # Setting of MMA variable
    MMA = {}
    # MMA["a0"]  = 1
    # MMA["a"]   = 0
    # MMA["c"]   = np.array([inputs["MMA_c"]])
    # MMA["d"]   = 1

    MMA["a0"] = 1.0
    MMA["a"] = np.zeros((1, 1))  # <-- FIX
    MMA["c"] = np.array([[inputs["MMA_c"]]])  # ensure column
    MMA["d"] = np.ones((1, 1))  # <-- FIX

    MMA["low"] = opt["dvmin"]
    MMA["upp"] = opt["dvmax"]
    opt["MMA"] = MMA

    # # Build filter element stiffness (Kft) and Transformation matrix (Tft)
    nx = np.transpose(np.vstack([fem["X"][fem["IX"][:, 0] - 1, [0]],
                                 fem["X"][fem["IX"][:, 1] - 1, [0]],
                                 fem["X"][fem["IX"][:, 2] - 1, [0]]]))
    ny = np.transpose(np.vstack([fem["X"][fem["IX"][:, 0] - 1, [1]],
                                 fem["X"][fem["IX"][:, 1] - 1, [1]],
                                 fem["X"][fem["IX"][:, 2] - 1, [1]]]))

    Kd = (inputs["rmin"] / (2 * np.sqrt(3))) ** 2 * np.array([[1, 0], [0, 1]], dtype=float)
    NN = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float) / 12.0
    isf = np.reshape(np.kron(fem["IX"][:, 0:3], np.ones((1, 3), dtype=int)), (9 * fem["ne"],))
    jsf = np.reshape(np.kron(fem["IX"][:, 0:3], np.ones((3, 1), dtype=int)), (9 * fem["ne"],))

    # Initialization
    Kft = np.zeros((9 * fem["ne"],), dtype=float)
    Tft = np.zeros((9 * fem["ne"],), dtype=float)

    for e in range(fem["ne"]):
        px = np.array([nx[e, 0], nx[e, 1], nx[e, 2]], dtype=float)
        py = np.array([ny[e, 0], ny[e, 1], ny[e, 2]], dtype=float)

        Ae = fem["Ae"][e]
        B = np.zeros((2, 3), dtype=float)

        for i in range(3):
            ind = np.arange(3)
            ind = np.delete(ind, i)
            pm = np.array([1, -1, 1], dtype=float)

            b_i = -pm[i] * np.linalg.det(np.vstack((np.ones(2), py[ind])))
            c_i = pm[i] * np.linalg.det(np.vstack((np.ones(2), px[ind])))

            B[:, i] = (1.0 / (2.0 * Ae)) * np.array([b_i, c_i], dtype=float)

        Se = (np.matmul(np.matmul(B.T, Kd), B) + NN) * Ae
        Kft[e * 9:(e + 1) * 9] = np.reshape(Se, (9,))
        Tft[e * 9:(e + 1) * 9] = np.reshape(NN * Ae, (9,))

    # Sparse + LU
    Kft_sparse = coo_matrix((Kft, (isf-1, jsf-1))).tocsc()
    opt["Kft_sparse"] = Kft_sparse
    LU = splu(Kft_sparse, permc_spec="NATURAL")
    opt["lu_L_Kft"] = LU.L
    opt["lu_U_Kft"] = LU.U
    opt["Tft"] = coo_matrix((Tft, (isf-1, jsf-1))).tocsc()

    # Build matrix for transformation from nodal to element density (Ten)
    opt["Ten"] = lil_matrix((fem["ne"], fem["nn"]))

    for e in range(fem["ne"]):
        opt["Ten"][e, fem["IX"][e, 0:3] - 1] = 1 / 3

    print("Optimization Initialization Done ✅")
    return (opt, MMA)