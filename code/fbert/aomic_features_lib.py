"""Cleaning helpers shared by aomic_features.py / adhd_features.py (identical code)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend


def clean(x, tr, gsr):
    x = np.nan_to_num(x.astype(np.float64))
    x = detrend(x, axis=1, type="linear")
    if gsr:
        g = x.mean(0)
        g = g - g.mean()
        beta = (x @ g) / (g @ g + 1e-12)
        x = x - beta[:, None] * g[None]
    sos = butter(5, [0.01, 0.08], btype="bandpass", fs=1.0 / tr, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    sd = x.std(1, keepdims=True)
    return ((x - x.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def pearson(z):
    z = z - z.mean(1, keepdims=True)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    C = z @ z.T
    np.fill_diagonal(C, 1.0)
    return C
