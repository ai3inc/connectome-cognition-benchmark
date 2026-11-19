"""Addendum M per-pod temporal features for ADHD-200 subjects stored on this pod:
DVARS motion proxy (raw parcel means, percent-signal, RMS of frame-to-frame change),
lag-1 autocorrelation per region, lag-1 cross-correlation matrix (full 450 x 450),
sliding-window FC variability (window 30, step 5; SD per edge, upper triangle).
Runs are cleaned on their own (noGSR, aomic_features_lib.clean) and features averaged
over runs (weighted by length).

usage: python3 adhd_temporal_pod.py TSDIR OUT.npz
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aomic_features_lib import clean

tsdir, outp = sys.argv[1:3]
iu = np.triu_indices(450, 1)
out = {k: [] for k in ["sids", "dvars", "ac1", "xc1", "dfc_sd"]}
for f in sorted(glob.glob(os.path.join(tsdir, "*.npz"))):
    d = np.load(f)
    n = int(d["nruns"]); trs = d["trs"]
    acc = {"dvars": 0.0, "ac1": 0.0, "xc1": 0.0, "dfc_sd": 0.0}; wsum = 0.0
    for i in range(n):
        x = d[f"ts{i}"]
        if x.shape[1] < 60 or np.isnan(x).any():
            continue
        T = x.shape[1]
        ps = x / (np.abs(x.mean(1, keepdims=True)) + 1e-6) * 100.0
        dv = float(np.sqrt((np.diff(ps, axis=1) ** 2).mean(0)).mean())
        z = clean(x, float(trs[i]), False).astype(np.float64)
        ac1 = (z[:, :-1] * z[:, 1:]).mean(1)
        xc1 = (z[:, :-1] @ z[:, 1:].T) / (T - 1)
        wins = []
        for s in range(0, T - 30 + 1, 5):
            w = z[:, s:s + 30]
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-12)
            wins.append(((w @ w.T) / 30)[iu])
        dsd = np.std(np.stack(wins), 0) if len(wins) > 1 else np.zeros(len(iu[0]))
        acc["dvars"] += T * dv; acc["ac1"] = acc["ac1"] + T * ac1
        acc["xc1"] = acc["xc1"] + T * xc1; acc["dfc_sd"] = acc["dfc_sd"] + T * dsd; wsum += T
    if wsum == 0:
        continue
    out["sids"].append(os.path.basename(f)[:-4])
    out["dvars"].append(acc["dvars"] / wsum)
    out["ac1"].append((acc["ac1"] / wsum).astype(np.float32))
    out["xc1"].append((acc["xc1"] / wsum).astype(np.float32).ravel())
    out["dfc_sd"].append((acc["dfc_sd"] / wsum).astype(np.float32))
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]))
