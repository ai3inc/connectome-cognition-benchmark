"""AOMIC per-pod features (Addendum H): for each participant's raw 450-region
timeseries, two cleaning variants at the file's TR — noGSR (detrend, band-pass
0.01-0.08 Hz order-5 zero-phase, z-score) and GSR (detrend, regress the mean of the
450 regions, then the same) — and for each: Pearson FC (strict upper triangle) and
F-BERT-8M / 1M frozen embeddings in the HCP-fixed region order.

usage: python3 aomic_features.py TSDIR ORDER_JSON FBERT_DIR OUT.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm

tsdir, ordp, fdir, outp = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("NT", "6")))
order = np.array(json.load(open(ordp))["chosen_order"])
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
iu = np.triu_indices(450, 1)


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


out = {k: [] for k in ["sids", "tr", "T", "n_empty"] +
       [f"{a}_{v}" for a in ["fc", "e8", "e1"] for v in ["nogsr", "gsr"]]}
t0 = time.time()
for j, f in enumerate(sorted(glob.glob(os.path.join(tsdir, "*.npz")))):
    d = np.load(f)
    ts, tr = d["ts"], float(d["tr"])
    out["sids"].append(os.path.basename(f)[:-4]); out["tr"].append(tr)
    out["T"].append(ts.shape[1]); out["n_empty"].append(int(np.isnan(ts).all(1).sum()))
    for v, gsr in [("nogsr", False), ("gsr", True)]:
        z = clean(ts, tr, gsr)
        out[f"fc_{v}"].append(pearson(z)[iu].astype(np.float32))
        with torch.no_grad():
            for t, m in models.items():
                out[f"e{t[0]}_{v}"].append(m(torch.from_numpy(z[order][None]))[0].numpy())
    if (j + 1) % 10 == 0:
        print(j + 1, f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]), flush=True)
