"""Nested selection under the benchmark's own metric and covariate step (2026-11-18).

The headline A = 0.7215 is the canonical chain at FIXED settings (g = 0.92, keep = 150),
scored like Ooi et al. (2022): per-fold r averaged, age band + sex regressed from the target
inside every training fold. This script asks what selection costs under those same
conditions. Grid g in {0.90, 0.92} x keep in {150, 200} (language x2, alpha 0.10, w 0.05,
tau 0.5 fixed), chosen together with lambda by mean inner 4-fold r on the training fold's
residualised target. Lambda grid 1e-6 ... 10 (unfiltered arms reach their ridgeless limit
at 1e-6). CBIG six (Ooi's kernel) is scored the same way for the paired test.

usage: python3 nested_matched.py WS COVARIATES.csv OUT.json [N_SEEDS]
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

WS = sys.argv[1]
sys.path.insert(0, WS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical as cn
from metric_check import cbig_corr_kernel, resid, nb_exact

MODS = ["rs", "lang", "social", "gamb", "motor", "wm"]
W = {m: (2 if m == "lang" else 1) for m in MODS}
GS, KEEPS = [0.90, 0.92], [150, 200]
LAMS = np.logspace(-6, 1, 8)
G = {}


def run_seed(nm):
    with threadpool_limits(1):
        y, C, Ks, names = G["y"], G["C"], G["Ks"], G["names"]
        n = len(y)
        pf = {a: [] for a in names + ["NESTED"]}
        picks = []
        for te in G["folds"][nm]:
            tr = np.setdiff1d(np.arange(n), te)
            ytr, yte = resid(y[tr], y[te], C[tr], C[te])
            S = np.zeros((len(Ks), len(LAMS)))
            Ktrs = [K[np.ix_(tr, tr)] for K in Ks]
            for a, b in cn._inner_splits(len(tr)):
                yc = ytr[a] - ytr[a].mean()
                for ki, Kt in enumerate(Ktrs):
                    w, V = np.linalg.eigh(Kt[np.ix_(a, a)])
                    Vt, VtK = V.T @ yc, Kt[np.ix_(b, a)] @ V
                    for li, lam in enumerate(LAMS):
                        S[ki, li] += np.corrcoef(VtK @ (Vt / (w + lam)), ytr[b])[0, 1] / 4
            S = np.nan_to_num(S, nan=-1)
            preds = []
            for ki, (K, Kt) in enumerate(zip(Ks, Ktrs)):
                lam = LAMS[int(np.argmax(S[ki]))]
                w, V = np.linalg.eigh(Kt)
                p = (K[np.ix_(te, tr)] @ V) @ (V.T @ (ytr - ytr.mean()) / (w + lam)) + ytr.mean()
                preds.append(p)
                pf[names[ki]].append(float(np.corrcoef(p, yte)[0, 1]))
            cand = [i for i, a in enumerate(names) if a.startswith("CANON")]
            best = max(cand, key=lambda i: S[i].max())
            picks.append(names[best])
            pf["NESTED"].append(float(np.corrcoef(preds[best], yte)[0, 1]))
        return nm, pf, picks


def main(cov_path, outp, n_seeds=60):
    t0 = time.time()
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(WS + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(WS + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    beh = {r["Subject"].strip(): r for r in csv.DictReader(open(cov_path, encoding="utf-8-sig"))}
    assert all(s in beh for s in use)
    bands = sorted({beh[s]["Age"] for s in use})
    C = np.array([[1.0 if beh[s]["Gender"] == "M" else 0.0] + [1.0 if beh[s]["Age"] == b else 0.0 for b in bands[1:]]
                  for s in use])
    pos = {s: i for i, s in enumerate(off)}
    idx = np.array([pos[s] for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(WS + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    seeds = list(folds._fieldnames)[:n_seeds]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in f.subject_list if str(s) in sidx])
                             for f in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in seeds}
    acc = {(g, k): None for g in GS for k in KEEPS}
    KC = None
    for m in MODS:
        e = np.load(WS + f"/mod_{m}.npz")["bp"][idx].astype(np.float64)
        kc = cbig_corr_kernel(e)
        KC = kc if KC is None else KC + kc
        for g in GS:
            full_e, part_e = cn._per_subject(e, g)
            Kf = cn.correlation_kernel(full_e)
            Ku = cn.correlation_kernel(cn.alpha_filter(part_e))
            del full_e, part_e
            for k in KEEPS:
                Km = W[m] * ((1 - cn.W_FULL) * cn.spectral_filter(Ku, keep=k) + cn.W_FULL * Kf)
                acc[(g, k)] = Km if acc[(g, k)] is None else acc[(g, k)] + Km
        del e
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    names = [f"CANON_g{g}_k{k}" for g in GS for k in KEEPS] + ["CBIG6_eq"]
    Ks = [acc[(g, k)] / sum(W.values()) for g in GS for k in KEEPS] + [KC / 6]
    G.update(y=y, C=C, Ks=Ks, names=names, folds=fl)
    with Pool(8) as pool:
        res = pool.map(run_seed, seeds)
    pf = {a: np.array([r[1][a] for r in res]) for a in names + ["NESTED"]}
    picks = [p for r in res for p in r[2]]
    ratio = float(np.mean([len(te) / (len(y) - len(te)) for nm in seeds for te in fl[nm]]))
    R = {"date": "2026-11-18", "purpose": __doc__.strip().splitlines()[0], "n": len(y), "seeds": len(seeds),
         "grid": {"g": GS, "keep": KEEPS, "lambda": LAMS.tolist()}, "covariates": "sex + age band (public), in-fold OLS",
         "mean_perfold_r": {a: round(float(v.mean()), 4) for a, v in pf.items()},
         "seed_sd": {a: round(float(v.mean(1).std()), 4) for a, v in pf.items()},
         "picks": {a: picks.count(a) for a in names if picks.count(a)},
         "NB": {"NESTED_vs_fixed_g0.92_k150": nb_exact(pf["NESTED"], pf["CANON_g0.92_k150"], ratio),
                "NESTED_vs_CBIG6": nb_exact(pf["NESTED"], pf["CBIG6_eq"], ratio),
                "fixed_vs_CBIG6": nb_exact(pf["CANON_g0.92_k150"], pf["CBIG6_eq"], ratio)},
         "per_fold_r": {a: v.round(5).tolist() for a, v in pf.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "per_fold_r"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 60)
