"""Canonical-chain kernels over the main session's nested grid (results/
NESTED_SELECTION_V6_1116.json): g in {0.90, 0.92, 0.94}, w = 0.05, alpha in
{0.10, 0.25}, keep in {150, 200}, language weight in {1, 2} -> 24 ensemble kernels,
on the benchmark's published features with code/canonical.py functions.

usage: python3 build_canon_grid.py WORKSPACE OUT.npz
"""
import csv, sys
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
sys.path.insert(0, "/workspace/fbert")
import canonical as C

GS, ALPHAS, KEEPS, LANGW = [0.90, 0.92, 0.94], [0.10, 0.25], [150, 200], [1, 2]
W = 0.05


def subj(v):
    """one subject: full edges and, per (g, alpha), alpha-reweighted partial edges"""
    with threadpool_limits(1):
        M = C.to_matrix(v[None])
        M = C.normalise(M)
        full = M[0][C.IU]
        out = {}
        for g in GS:
            P = C.partial_correlation(C.shrink(M.copy(), g))[0]
            for a in ALPHAS:
                out[(g, a)] = C.alpha_filter(P[None], a)[0].astype(np.float32)
        return full.astype(np.float32), out


ws, outp = sys.argv[1:3]
oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]
pos = {s: i for i, s in enumerate(off)}
idx = np.array([pos[s] for s in use])
per = {}   # (mod, g, alpha, keep) -> modality kernel
for m in C.MODS:
    edges = np.load(ws + "/mod_%s.npz" % m)["bp"][idx].astype(np.float64)
    with Pool(8) as pool:
        res = pool.map(subj, list(edges), chunksize=8)
    Kf = C.correlation_kernel(np.stack([r[0] for r in res]))
    for g in GS:
        for a in ALPHAS:
            Kp = C.correlation_kernel(np.stack([r[1][(g, a)] for r in res]))
            for k in KEEPS:
                per[(m, g, a, k)] = (1 - W) * C.spectral_filter(Kp, C.TAU, k) + W * Kf
    print(m, flush=True)
keys, Ks = [], []
for g in GS:
    for a in ALPHAS:
        for k in KEEPS:
            for lw in LANGW:
                wts = {mm: (lw if mm == "lang" else 1) for mm in C.MODS}
                Ks.append(sum(wts[mm] * per[(mm, g, a, k)] for mm in C.MODS) / sum(wts.values()))
                keys.append(f"g{g}_a{a}_k{k}_lw{lw}")
np.savez(outp, Ks=np.stack(Ks).astype(np.float64), keys=np.array(keys), use=np.array(use))
print("SAVED", len(Ks), flush=True)
