
import numpy as np

def F0_Main_Mat_Derivative(B):
    B = np.asarray(B, dtype=float)
    mu0 = 4.0 * np.pi * 1e-7

    a, b, c = 49.4, 1.46, 520.6

    z = np.clip(b * (B ** 2), 0.0, 700.0) #c --> b
    expterm = np.exp(z)

    denom = (a * expterm + c) ** 2 #b --> c
    dmu_raw = -(2.0 * a * b * B * expterm) / denom #c --> b

    mu_raw = 1.0 / (a * expterm + c) #b --> c
    # kalau mu kena clamp -> derivative = 0
    dmu = np.where(mu_raw > mu0, dmu_raw, 0.0)
    return dmu
