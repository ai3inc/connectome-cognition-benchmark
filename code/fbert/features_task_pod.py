"""Per-pod task features (Addendum A): per task, LR and RL each cleaned with the
model-card recipe (hcp_prep.clean_run) and concatenated; Pearson FC (strict upper
triangle, float32) and F-BERT-8M / 1M embeddings in the order fixed by order_check.

usage: python3 features_task_pod.py TASK_TSDIR ORDER_JSON FBERT_DIR OUT.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm
import hcp_prep as hp

tsdir, ordp, fdir, outp = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("NT", "6")))
ALLT = {"LANGUAGE": "lang", "SOCIAL": "social", "GAMBLING": "gamb", "MOTOR": "motor", "WM": "wm", "RELATIONAL": "rel", "EMOTION": "emo"}
TASKS = {t: ALLT[t] for t in os.environ.get("TASKS", "LANGUAGE,SOCIAL,GAMBLING,MOTOR,WM").split(",")}
order = np.array(json.load(open(ordp))["chosen_order"])
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
iu = np.triu_indices(450, 1)

files = sorted(glob.glob(os.path.join(tsdir, "*.npz")))
out = {"sids": [], "has": []}
for m in TASKS.values():
    for k in ["fc", "e8", "e1"]:
        out[f"{k}_{m}"] = []
t0 = time.time()
for j, f in enumerate(files):
    d = np.load(f)
    out["sids"].append(os.path.basename(f)[:-4])
    has = []
    for T, m in TASKS.items():
        runs = [hp.clean_run(d[f"ts_{T}_{pe}"]) for pe in ["LR", "RL"]
                if f"ts_{T}_{pe}" in d.files and d[f"ts_{T}_{pe}"].shape[1] >= 100]
        if runs:
            x = np.concatenate(runs, 1)
            has.append(len(runs))
            out[f"fc_{m}"].append(hp.pearson(x)[iu].astype(np.float32))
            with torch.no_grad():
                for t, mod in models.items():
                    out[f"e{t[0]}_{m}"].append(mod(torch.from_numpy(x[order][None]))[0].numpy())
        else:
            has.append(0)
            out[f"fc_{m}"].append(np.full(len(iu[0]), np.nan, np.float32))
            out[f"e8_{m}"].append(np.full(384, np.nan, np.float32))
            out[f"e1_{m}"].append(np.full(128, np.nan, np.float32))
    out["has"].append(has)
    if (j + 1) % 10 == 0:
        print(j + 1, "/", len(files), f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]), flush=True)
