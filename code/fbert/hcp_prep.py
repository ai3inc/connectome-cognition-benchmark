"""Shared preprocessing for the F-BERT comparison (model-card input recipe).

Each run: linear detrend, Butterworth band-pass 0.01-0.08 Hz (order 5, zero phase,
TR 0.72 s), z-score per region. Full scan = the runs concatenated (4 x 1200 = 4800).
Short scan = the first 200 timepoints (2.4 min) of REST1_LR, processed on its own.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

TR = 0.72
SOS = butter(5, [0.01, 0.08], btype="bandpass", fs=1.0 / TR, output="sos")
SHORT_T = 200


def clean_run(x):
    """x: (regions, T) raw parcel means, no NaN."""
    x = detrend(x.astype(np.float64), axis=1, type="linear")
    x = sosfiltfilt(SOS, x, axis=1)
    sd = x.std(1, keepdims=True)
    return ((x - x.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def load_subject(path):
    d = np.load(path)
    ts, runs, lens = d["ts"], list(d["runs"]), list(d["lengths"])
    cleaned = [clean_run(ts[i, :, :lens[i]]) for i in range(len(runs)) if lens[i] >= 100]
    full = np.concatenate(cleaned, axis=1)
    # short scan: first 2.4 min of REST1_LR, or of the first available run if
    # REST1_LR is missing (the subject is then listed by features_pod.py)
    short = None
    order = (["REST1_LR"] if "REST1_LR" in runs else []) + [r for r in runs if r != "REST1_LR"]
    for r in order:
        i = runs.index(r)
        if lens[i] >= SHORT_T:
            short = clean_run(ts[i, :, :SHORT_T])
            break
    return full, short, runs, lens


def pearson(z):
    """z: (regions, T) z-scored rows -> correlation matrix."""
    z = z - z.mean(1, keepdims=True)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    C = z @ z.T
    np.fill_diagonal(C, 1.0)
    return C


def power_fc(C, alpha=0.35):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.clip(w, 0.0, None)
    return (V * w ** alpha) @ V.T


def teacher_vec(C, alpha=0.35):
    """The corpus teacher: strict upper triangle of FC^alpha (101,025 entries at
    450 regions, as teachers.npy), sqrt(2)-weighted, centred, unit norm.
    (order_check.py's run on 11-17 used a diagonal-included variant; the
    between-subject Gram it feeds is nearly unchanged and the check is relative.)"""
    P = power_fc(C, alpha)
    iu = np.triu_indices(P.shape[0], 1)
    v = P[iu] * np.sqrt(2.0)
    v = v - v.mean()
    return v / np.linalg.norm(v)
