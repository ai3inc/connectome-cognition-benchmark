"""Embed exported cleaned timeseries (export_clean_ts.py tars) with one or more encoder
checkpoints. Region order: the HCP-fixed F-BERT order (cortex 7-network first, then Tian),
which is also the order of the pretraining corpus our encoders were trained on.

usage: python embed_clean.py ORDER_JSON CLEAN_DIR OUT.npz CKPT [CKPT ...]
  CLEAN_DIR holds <sid>[_<suffix>].npy files (450, T) float16
  OUT.npz: keys  ids (file stems)  and  e<k>  (n, D) for checkpoint k
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm

ordp, cdir, outp = sys.argv[1:4]
ckpts = sys.argv[4:]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("NT", "8")))
order = torch.as_tensor(json.load(open(ordp))["chosen_order"], device=dev)
best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = []
for c in ckpts:
    m, ck = fm.load(c, **best)
    models.append(m.to(dev).eval())
files = sorted(glob.glob(os.path.join(cdir, "*.npy")))
ids, embs = [], [[] for _ in models]
t0 = time.time()
with torch.no_grad():
    for j, f in enumerate(files):
        x = torch.from_numpy(np.load(f).astype(np.float32)).to(dev)[order][None]
        ids.append(os.path.basename(f)[:-4])
        for k, m in enumerate(models):
            embs[k].append(m(x)[0].float().cpu().numpy())
        if (j + 1) % 500 == 0:
            print(j + 1, "/", len(files), f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, ids=np.array(ids), ckpts=np.array(ckpts), **{f"e{k}": np.stack(e) for k, e in enumerate(embs)})
print("SAVED", outp, len(ids), flush=True)
