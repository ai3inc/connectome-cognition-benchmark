"""Brain-JEPA embeddings for a share of the 753 (Addendum I). Reads every exported
window file, keeps subjects i::N for this pod, writes their 768-d embeddings.

usage: python3 bjepa_embed_pod.py WIN_GLOB SHARE_INDEX N_SHARES OUT.npz
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bjepa_model as bm

pat, k, N, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(int(os.environ.get("NT", "8")))
S, W, NW = [], [], []
for f in sorted(glob.glob(pat)):
    d = np.load(f)
    S += [str(s) for s in d["sids"]]; W.append(d["windows"]); NW += list(d["nwin"])
W = np.concatenate(W)
order = np.argsort(S)
mine = order[k::N]
enc, mi, un = bm.load()
assert not mi and not un
out_s, out_e, t0 = [], [], time.time()
for j, i in enumerate(mine):
    wins = [bm.prep_window(W[i, w]) for w in range(NW[i])]
    out_s.append(S[i]); out_e.append(bm.embed(enc, wins))
    if (j + 1) % 5 == 0:
        print(j + 1, "/", len(mine), f"{time.time() - t0:.0f}s", flush=True)
np.savez(outp, sids=np.array(out_s), emb=np.stack(out_e), nwin=np.array([NW[i] for i in mine]))
print("SAVED", outp, len(out_s), flush=True)
