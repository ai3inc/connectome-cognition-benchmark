"""Confirmation on subjects OUTSIDE the 753 (prereg Addendum D).

Train on all 753 (every hyperparameter chosen by inner CV inside the 753), predict
the held-out subjects once. Strictly inductive: feature centring and z-scoring use
training statistics only; no test subject enters any kernel spectrum.

Arms
  inductive:  A8cat (F-BERT-8M concat, train z-stats), D0 (ours, shared g), H0 = D0/A8cat half-half
  batch-transductive (kernels over train + held-out, canonical filter, no labels):
              A8F (F-BERT kernels filtered), D1 (ours + filter), H1 = D1/A8F half-half
  FBERT_best: inner CV on the 753 picks A8cat or A8F (primary comparator)

Output: held-out r per arm, paired bootstrap of r_ours - r_fbert over held-out
subjects (10,000 draws), and the same on the similarity-filtered (leak-controlled)
subset.

usage: python3 eval_holdout.py WORKSPACE OUT.json TRAIN_REST TRAIN_TASK HOLD_REST HOLD_TASK
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

GS = [0.9, 0.95, 0.97, 0.98]
INNER_K = 10


def partial_edges(v):
    """D0 chain for one subject at every g in GS (no FC^0.35 here)."""
    with threadpool_limits(1):
        C = ev.to_mat(v.astype(np.float64))
        out = []
        for g in GS:
            P = np.linalg.inv((1 - g) * C + g * np.eye(ev.N_REG))
            d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            P = -P / np.outer(d, d)
            np.fill_diagonal(P, 1.0)
            w, V = np.linalg.eigh(P)
            out.append(((V * np.clip(w, 0, None) ** ev.ALPHA) @ V.T)[ev.IU].astype(np.float32))
        return out


def inductive_corr_kernel(X, tr):
    """Pearson-style kernel with centring from the training rows only."""
    Xc = X - X[tr].mean(0)
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    K = Xc @ Xc.T
    return K / np.mean(np.diag(K)[tr]) / len(X)


def inner_select(Ks, tr, y, k=INNER_K, seed=0):
    """Choose (kernel, lambda) by k-fold CV inside the training rows (mean r)."""
    idx = np.array(tr)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(idx))
    parts = np.array_split(perm, k)
    S = np.zeros((len(Ks), len(ev.LAMBDAS)))
    for part in parts:
        a = idx[np.setdiff1d(perm, part)]
        b = idx[part]
        ya = y[a] - y[a].mean()
        for ki, K in enumerate(Ks):
            w, V = np.linalg.eigh(K[np.ix_(a, a)])
            VtK = K[np.ix_(b, a)] @ V
            Vt = V.T @ ya
            for li, lam in enumerate(ev.LAMBDAS):
                S[ki, li] += np.corrcoef(VtK @ (Vt / (w + lam)), y[b])[0, 1] / k
    ki, li = np.unravel_index(np.nanargmax(S), S.shape)
    return int(ki), float(ev.LAMBDAS[li]), S


def fit_predict(K, tr, te, y, lam):
    w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
    yc = y[tr] - y[tr].mean()
    return (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()


def boot_diff(pa, pb, yt, n=10000, seed=1):
    rs = np.random.RandomState(seed)
    m = len(yt)
    d = np.empty(n)
    for i in range(n):
        j = rs.randint(0, m, m)
        d[i] = np.corrcoef(pa[j], yt[j])[0, 1] - np.corrcoef(pb[j], yt[j])[0, 1]
    return {"diff": float(np.corrcoef(pa, yt)[0, 1] - np.corrcoef(pb, yt)[0, 1]),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0": float((d > 0).mean())}


def load_all(ws, pats):
    rest = {}
    task = {}
    for rp, tp in pats:
        rest.update(sx.load_feats(rp, {"fc_full": "fc_rs", "e8_full": "e8_rs"}))
        task.update(sx.load_feats(tp, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]}))
    return rest, task


def main():
    ws, outp, trr, trt, hor, hot = sys.argv[1:7]
    t0 = time.time()
    rest, task = load_all(ws, [(trr, trt), (hor, hot)])
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    s753 = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]

    def ok(s):
        if s not in rest or s not in task or s not in oi:
            return False
        if not np.isfinite(float(oi[s].get("varimax_cog") or "nan")):
            return False
        r = {**rest[s], **task[s]}
        return all(f"fc_{m}" in r and np.isfinite(r[f"fc_{m}"]).all() for m in sx.MODS)
    train = [s for s in s753 if ok(s)]
    hold = sorted(s for s in task if s not in set(s753) and ok(s))
    use = train + hold
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    tr = np.arange(len(train)); te = np.arange(len(train), len(use))
    print(f"train {len(train)} held-out {len(hold)}", flush=True)

    # leak control (as the main session's leak961): rest FC minus the group mean edge
    # profile; a held-out subject is kept if its max correlation with any training
    # subject is below the 90th percentile of within-training max correlations.
    F = np.stack([rest[s]["fc_rs"] for s in use]).astype(np.float64)
    F = F - F[tr].mean(0)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    Ctt = F[tr] @ F[tr].T
    np.fill_diagonal(Ctt, -1)
    thr = float(np.percentile(Ctt.max(1), 90))
    hmax = (F[te] @ F[tr].T).max(1)
    clean = hmax < thr
    print(f"leak control: threshold {thr:.3f}, kept {clean.sum()}/{len(hold)}", flush=True)

    # kernels: inductive (train-only centring, no filter) and batch-transductive
    # (canonical correlation kernel over train + held-out, canonical filter; no labels)
    KI = {g: 0 for g in GS}; KT = {g: 0 for g in GS}
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(partial_edges, list(fc), chunksize=8)
        Kf_i = inductive_corr_kernel(fc.astype(np.float64), tr)
        Kf_t = ev.correlation_kernel(fc)
        for gi, g in enumerate(GS):
            Xp = np.stack([r[gi] for r in res])
            KI[g] = KI[g] + (1 - ev.W_FULL) * inductive_corr_kernel(Xp.astype(np.float64), tr) + ev.W_FULL * Kf_i
            KT[g] = KT[g] + (1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(Xp)) + ev.W_FULL * Kf_t
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    KI = {g: K / 6 for g, K in KI.items()}; KT = {g: K / 6 for g, K in KT.items()}
    Es = [np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64) for m in sx.MODS]
    K_A8 = ev.zlin_kernel(np.concatenate(Es, 1), tr)
    K_A8F = sum(ev.spectral_filter(ev.correlation_kernel(E)) for E in Es) / 6
    arms = {"A8cat": [K_A8], "A8F": [K_A8F], "FBERT_best": [K_A8, K_A8F],
            "D0": [KI[g] for g in GS], "H0": [0.5 * KI[g] + 0.5 * K_A8 for g in GS],
            "D1": [KT[g] for g in GS], "H1": [0.5 * KT[g] + 0.5 * K_A8F for g in GS]}
    preds, chosen = {}, {}
    for a, Ks in arms.items():
        ki, lam, _ = inner_select(Ks, tr, y)
        preds[a] = fit_predict(Ks[ki], tr, te, y, lam)
        chosen[a] = {"kernel": ki, "lambda": lam}
    yt = y[te]
    R = {"addendum": "D (confirmation on subjects outside the 753)",
         "n_train": len(train), "n_holdout": len(hold), "n_holdout_leak_controlled": int(clean.sum()),
         "leak_threshold": thr, "chosen": chosen,
         "holdout_r": {a: float(np.corrcoef(p, yt)[0, 1]) for a, p in preds.items()},
         "holdout_r_leak_controlled": {a: float(np.corrcoef(p[clean], yt[clean])[0, 1]) for a, p in preds.items()},
         "primary_D1_vs_FBERT_best": boot_diff(preds["D1"], preds["FBERT_best"], yt),
         "primary_D1_vs_FBERT_best_leak_controlled": boot_diff(preds["D1"][clean], preds["FBERT_best"][clean], yt[clean]),
         "paired_vs_FBERT_best": {a: boot_diff(preds[a], preds["FBERT_best"], yt) for a in ["D0", "D1", "H0", "H1"]},
         "paired_vs_A8cat": {a: boot_diff(preds[a], preds["A8cat"], yt) for a in ["D0", "D1", "H0", "H1", "A8F"]},
         "paired_vs_A8cat_leak_controlled": {a: boot_diff(preds[a][clean], preds["A8cat"][clean], yt[clean])
                                             for a in ["D0", "D1", "H0", "H1", "A8F"]},
         "holdout_ids": hold, "predictions": {a: p.round(6).tolist() for a, p in preds.items()},
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: R[k] for k in ["n_train", "n_holdout", "n_holdout_leak_controlled", "holdout_r",
                                        "holdout_r_leak_controlled", "primary_D1_vs_FBERT_best",
                                        "primary_D1_vs_FBERT_best_leak_controlled", "paired_vs_A8cat",
                                        "paired_vs_A8cat_leak_controlled", "chosen"]}, indent=1))


if __name__ == "__main__":
    main()
