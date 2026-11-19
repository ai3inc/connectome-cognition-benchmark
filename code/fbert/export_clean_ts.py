"""Export cleaned 450-region timeseries (float16, encoder input order = atlas file order;
the HCP-fixed F-BERT order is applied at embedding time) so any new encoder can be
evaluated from one place. Same cleaning as every feature script so far.

usage: python3 export_clean_ts.py KIND SRC_DIR OUT_DIR
  KIND: hcp_rest | hcp_task | single (AOMIC/ABIDE: ts + tr) | adhd (ts{i} + trs)
writes OUT_DIR/<sid>[_<task|variant>].npy  (450, T) float16
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hcp_prep as hp
from aomic_features_lib import clean

kind, src, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
TASKS = {"LANGUAGE": "lang", "SOCIAL": "social", "GAMBLING": "gamb", "MOTOR": "motor", "WM": "wm"}
n = 0
for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
    sid = os.path.basename(f)[:-4]
    d = np.load(f)
    if kind == "hcp_rest":
        full, short, runs, lens = hp.load_subject(f)
        np.save(os.path.join(out, f"{sid}.npy"), full.astype(np.float16))
    elif kind == "hcp_task":
        for T, m in TASKS.items():
            segs = [hp.clean_run(d[f"ts_{T}_{pe}"]) for pe in ["LR", "RL"]
                    if f"ts_{T}_{pe}" in d.files and d[f"ts_{T}_{pe}"].shape[1] >= 100]
            if segs:
                np.save(os.path.join(out, f"{sid}_{m}.npy"), np.concatenate(segs, 1).astype(np.float16))
    elif kind == "single":
        for v, g in [("nogsr", False), ("gsr", True)]:
            np.save(os.path.join(out, f"{sid}_{v}.npy"), clean(d["ts"], float(d["tr"]), g).astype(np.float16))
    elif kind == "adhd":
        nr = int(d["nruns"]); trs = d["trs"]
        runs = [d[f"ts{i}"] for i in range(nr) if d[f"ts{i}"].shape[1] >= 60]
        if not runs:
            continue
        for v, g in [("nogsr", False), ("gsr", True)]:
            z = np.concatenate([clean(r, float(trs[i]), g) for i, r in enumerate(runs)], 1)
            np.save(os.path.join(out, f"{sid}_{v}.npy"), z.astype(np.float16))
    n += 1
print("EXPORTED", kind, n)
