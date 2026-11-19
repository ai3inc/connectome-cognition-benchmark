"""Build the canonical six-modality kernel (code/canonical.py, unchanged settings) on
the benchmark's published features and save it with its subject order, so it can be
combined with our 450-region D1 kernels. Read-only use of the main session's method.

usage: python3 build_canon_kernel.py WORKSPACE OUT.npz
"""
import csv, sys
import numpy as np
sys.path.insert(0, "/workspace/fbert")
import canonical as C

ws, outp = sys.argv[1:3]
oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]
pos = {s: i for i, s in enumerate(off)}
idx = np.array([pos[s] for s in use])
total, per = None, {}
for m in C.MODS:
    edges = np.load(ws + "/mod_%s.npz" % m)["bp"][idx].astype(np.float64)
    K = C.modality_kernel(edges, C.G, C.W)
    per[m] = K
    total = K * C.MOD_WEIGHTS[m] if total is None else total + K * C.MOD_WEIGHTS[m]
    print(m, flush=True)
K = total / sum(C.MOD_WEIGHTS.values())
np.savez(outp, K=K, use=np.array(use), **{f"K_{m}": v for m, v in per.items()})
print("SAVED", K.shape, flush=True)
