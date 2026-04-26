import numpy as np

def F0_Main_Mat_Derivative(B):

    B = np.asarray(B, dtype=float)

    a = 49.4
    b = 1.46
    c = 520.6

    # avoid overflow
    z = np.clip(b * (B ** 2), 0.0, 700.0)
    expterm = np.exp(z)

    # derivative:
    dnu_dB = a * expterm * (2.0 * b * B)

    return dnu_dB
