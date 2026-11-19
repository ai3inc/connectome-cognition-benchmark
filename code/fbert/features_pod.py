"""Per-pod features for the F-BERT comparison: everything every arm needs, from
one set of cleaned timeseries, so only small arrays leave the pod.

Per subject: Pearson FC (strict upper triangle, float32) for the full scan and
the 2.4-min short scan, and F-BERT-8M / 1M frozen embeddings for both, with the
region order fixed by order_check.py.

usage: python3 features_pod.py TSDIR ORDER_JSON FBERT_DIR OUT.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm
import hcp_prep as hp

tsdir, ordp, fdir, outp = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("NT", "6")))
order = np.array(json.load(open(ordp))["chosen_order"])
assert sorted(order.tolist()) == list(range(450))
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
iu = np.triu_indices(450, 1)

files = sorted(glob.glob(os.path.join(tsdir, "*.npz")))
out = {k: [] for k in ["sids", "nruns", "tfull", "fc_full", "fc_short",
                       "e8_full", "e1_full", "e8_short", "e1_short"]}
t0 = time.time()
for j, f in enumerate(files):
    sid = os.path.basename(f)[:-4]
    full, short, runs, lens = hp.load_subject(f)
    if "REST1_LR" not in runs:
        print("SHORT_FROM_OTHER_RUN", sid, runs, flush=True)
    out["sids"].append(sid); out["nruns"].append(len(runs)); out["tfull"].append(full.shape[1])
    out["fc_full"].append(hp.pearson(full)[iu].astype(np.float32))
    out["fc_short"].append(hp.pearson(short)[iu].astype(np.float32))
    with torch.no_grad():
        for t, m in models.items():
            out[f"e{t[0]}_full"].append(m(torch.from_numpy(full[order][None]))[0].numpy())
            out[f"e{t[0]}_short"].append(m(torch.from_numpy(short[order][None]))[0].numpy())
    if (j + 1) % 10 == 0:
        print(j + 1, "/", len(files), f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]), flush=True)
