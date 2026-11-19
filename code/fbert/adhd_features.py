"""ADHD-200 per-pod features (Addendum L): every run cleaned on its own (noGSR / GSR
as aomic_features.clean) and concatenated; Pearson FC and F-BERT-8M / 1M embeddings.
Output keys match aomic_features.py so abide_eval.py-style evaluation can read them.

usage: python3 adhd_features.py TSDIR ORDER_JSON FBERT_DIR OUT.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm
from aomic_features_lib import clean, pearson   # same functions as aomic_features.py

tsdir, ordp, fdir, outp = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("NT", "6")))
order = np.array(json.load(open(ordp))["chosen_order"])
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
iu = np.triu_indices(450, 1)
out = {k: [] for k in ["sids", "T", "n_empty"] + [f"{a}_{v}" for a in ["fc", "e8", "e1"] for v in ["nogsr", "gsr"]]}
t0 = time.time()
for j, f in enumerate(sorted(glob.glob(os.path.join(tsdir, "*.npz")))):
    d = np.load(f)
    n = int(d["nruns"]); trs = d["trs"]
    runs = [d[f"ts{i}"] for i in range(n) if d[f"ts{i}"].shape[1] >= 60]
    if not runs:
        continue
    out["sids"].append(os.path.basename(f)[:-4]); out["T"].append(sum(r.shape[1] for r in runs))
    out["n_empty"].append(int(np.isnan(runs[0]).all(1).sum()))
    for v, gsr in [("nogsr", False), ("gsr", True)]:
        z = np.concatenate([clean(r, float(trs[i]), gsr) for i, r in enumerate(runs)], 1)
        out[f"fc_{v}"].append(pearson(z)[iu].astype(np.float32))
        with torch.no_grad():
            for t, m in models.items():
                out[f"e{t[0]}_{v}"].append(m(torch.from_numpy(z[order][None]))[0].numpy())
    if (j + 1) % 10 == 0:
        print(j + 1, f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]), flush=True)
