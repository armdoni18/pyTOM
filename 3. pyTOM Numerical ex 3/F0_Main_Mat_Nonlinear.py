import numpy as np

def F0_Main_Mat_Nonlinear(B):

    B = np.asarray(B, dtype=float)
    mu0 = 4.0 * np.pi * 1e-7

    a, b, c = 49.4, 1.46, 520.6

    z = np.clip(b * (B ** 2), 0.0, 700.0)
    expterm = np.exp(z)

    mu_raw = 1.0 / (a * expterm + c)
    mu = np.maximum(mu_raw, mu0)
    return mu
