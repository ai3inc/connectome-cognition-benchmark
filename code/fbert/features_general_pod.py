"""Fairness gate of Addendum B: F-BERT reading the same 'general' input our GFC uses —
every run of the subject (rest 1-4, then LANGUAGE, SOCIAL, GAMBLING, MOTOR, WM, LR
then RL), each cleaned on its own (hcp_prep.clean_run), concatenated into one
sequence (~7,864 timepoints), encoded once.

usage: python3 features_general_pod.py REST_TSDIR TASK_TSDIR ORDER_JSON FBERT_DIR OUT.npz
(subjects = those present in TASK_TSDIR; their rest npz must be in REST_TSDIR)
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm
import hcp_prep as hp

rdir, tdir, ordp, fdir, outp = sys.argv[1:6]
torch.set_num_threads(int(os.environ.get("NT", "7")))
order = np.array(json.load(open(ordp))["chosen_order"])
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
TASKS = ["LANGUAGE", "SOCIAL", "GAMBLING", "MOTOR", "WM"]
iu = np.triu_indices(450, 1)
out = {"sids": [], "T": [], "e8_all": [], "e1_all": [], "fc_all": []}
t0 = time.time()
files = sorted(glob.glob(os.path.join(tdir, "*.npz")))
for j, f in enumerate(files):
    sid = os.path.basename(f)[:-4]
    r = np.load(os.path.join(rdir, f"{sid}.npz"))
    segs = [hp.clean_run(r["ts"][i, :, :r["lengths"][i]]) for i in range(len(r["runs"])) if r["lengths"][i] >= 100]
    d = np.load(f)
    for T in TASKS:
        for pe in ["LR", "RL"]:
            k = f"ts_{T}_{pe}"
            if k in d.files and d[k].shape[1] >= 100:
                segs.append(hp.clean_run(d[k]))
    x = np.concatenate(segs, 1)
    out["sids"].append(sid); out["T"].append(x.shape[1])
    out["fc_all"].append(hp.pearson(x)[iu].astype(np.float32))
    with torch.no_grad():
        for t, m in models.items():
            out[f"e{t[0]}_all"].append(m(torch.from_numpy(x[order][None]))[0].numpy())
    if (j + 1) % 10 == 0:
        print(j + 1, "/", len(files), f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, **{k: np.array(v) for k, v in out.items()})
print("SAVED", outp, len(out["sids"]), flush=True)
