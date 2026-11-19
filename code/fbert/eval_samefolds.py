"""F-BERT vs connectome kernels on identical subjects, timeseries and folds.
Arms and decision rule: PREREG_FBERT_SAMEFOLDS_1117.md (committed before this ran).

usage: python3 eval_samefolds.py WORKSPACE OUT.json feat_*.npz
  WORKSPACE holds ooi_comp.csv, off753.txt, splits.mat (as canonical.py)
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

N_REG = 450
IU = np.triu_indices(N_REG, 1)
ALPHA, TAU, KEEP, W_FULL = 0.10, 0.5, 150, 0.05      # canonical.py values
G_GRID = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
LAMBDAS = np.logspace(-6, 2, 17)
N_SEEDS = int(os.environ.get("N_SEEDS", "60"))


def correlation_kernel(X):  # canonical.py, unchanged
    X = np.asarray(X, dtype=np.float32)
    Xc = X - X.mean(0)
    norms = np.linalg.norm(Xc, axis=1) + 1e-12
    K = (Xc @ Xc.T) / np.outer(norms, norms)
    H = np.eye(len(X)) - np.ones((len(X), len(X))) / len(X)
    K = H @ K @ H
    return K / np.trace(K)


def spectral_filter(K, tau=TAU, keep=KEEP):  # canonical.py, unchanged
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0.0, None)
    f = np.clip(w - tau * np.std(w), 0.0, None)
    if keep is not None and keep < len(f):
        thr = np.sort(f)[::-1][keep - 1]
        f = np.where(f >= thr, f, 0.0)
    Kt = (V * f) @ V.T
    tr = np.trace(Kt)
    return Kt / tr if tr > 0 else Kt


def to_mat(v):
    M = np.zeros((N_REG, N_REG))
    M[IU] = v
    M = M + M.T
    np.fill_diagonal(M, 1.0)
    return M


def per_subject(v):
    """One subject's FC edges -> FC^0.35 edges and, per g, conditioned partial edges."""
    C = to_mat(v.astype(np.float64))
    w, V = np.linalg.eigh(C)
    p035 = ((V * np.clip(w, 0, None) ** 0.35) @ V.T)[IU].astype(np.float32)
    parts = []
    for g in G_GRID:
        Cg = (1 - g) * C + g * np.eye(N_REG)
        P = np.linalg.inv(Cg)
        d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
        P = -P / np.outer(d, d)
        np.fill_diagonal(P, 1.0)
        wa, Va = np.linalg.eigh(P)
        parts.append(((Va * np.clip(wa, 0, None) ** ALPHA) @ Va.T)[IU].astype(np.float32))
    return p035, parts


def inner_splits(n, seed=0):
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    for part in np.array_split(idx, 4):
        yield np.setdiff1d(idx, part), part


def krr_select(Ks, tr, te, y):
    """Ks: list of candidate n x n kernels. Choose (kernel, lambda) by mean inner
    4-fold r; also return each kernel's own outer prediction at its best lambda."""
    ytr = y[tr]
    S = np.zeros((len(Ks), len(LAMBDAS)))
    for a, b in inner_splits(len(tr)):
        ya = ytr[a] - ytr[a].mean()
        for ki, K in enumerate(Ks):
            Ka = K[np.ix_(tr[a], tr[a])]
            w, V = np.linalg.eigh(Ka)
            VtK = K[np.ix_(tr[b], tr[a])] @ V
            Vt = V.T @ ya
            for li, lam in enumerate(LAMBDAS):
                p = VtK @ (Vt / (w + lam))
                S[ki, li] += np.corrcoef(p, ytr[b])[0, 1] / 4
    per_kernel = []
    for ki, K in enumerate(Ks):
        lam = LAMBDAS[int(np.nanargmax(S[ki]))]
        w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
        yc = ytr - ytr.mean()
        per_kernel.append((K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + ytr.mean())
    kbest = int(np.unravel_index(np.nanargmax(S), S.shape)[0])
    return per_kernel[kbest], kbest, per_kernel


def zlin_kernel(E, tr):
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-12
    Z = (E - mu) / sd
    K = Z @ Z.T
    return K / (np.mean(np.diag(K)[tr]) * len(E))


G = {}


def run_seed(nm):
    y, folds_of, arms = G["y"], G["folds"][nm], G["arms"]
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in arms}
    pred_g = {a: {gi: np.full(n, np.nan) for gi in range(len(G_GRID))} for a in arms if a.startswith("D")}
    gsel = {a: [] for a in arms if a.startswith("D")}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        for a, spec in arms.items():
            if spec[0] == "zlin":
                Ks = [zlin_kernel(spec[1], tr)]
            else:
                Ks = spec[1]
            p, kb, perk = krr_select(Ks, tr, te, y)
            pred[a][te] = p
            if a in gsel:
                gsel[a].append(G_GRID[kb])
                for gi, pk in enumerate(perk):
                    pred_g[a][gi][te] = pk
    r = {a: float(np.corrcoef(pred[a], y)[0, 1]) for a in arms}
    rg = {a: [float(np.corrcoef(pred_g[a][gi], y)[0, 1]) for gi in range(len(G_GRID))] for a in pred_g}
    return nm, r, rg, gsel


def main():
    ws, outp, feats = sys.argv[1], sys.argv[2], sys.argv[3:]
    t0 = time.time()
    D = {}
    for f in feats:
        # NpzFile re-reads the whole array on every d[k]; and d[k][i] is a view that
        # keeps that whole copy alive. Load each array once, copy rows out.
        with np.load(f) as d:
            arrs = {k: d[k] for k in d.files}
        for i, s in enumerate(arrs["sids"]):
            D[str(s)] = {k: np.array(v[i]) for k, v in arrs.items() if k != "sids"}
        del arrs
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use0 = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]
    use = [s for s in use0 if s in D]
    missing = [s for s in use0 if s not in D]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:N_SEEDS]
    fl = {}
    for nm in names:
        fl[nm] = []
        for fold in getattr(folds, nm).sub_fold:
            te = np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
            if len(te) >= 20:
                fl[nm].append(te)
    print(f"subjects {len(use)} (missing {len(missing)}), seeds {len(names)}", flush=True)

    arms = {}
    for scan in ["full", "short"]:
        fc = np.stack([D[s][f"fc_{scan}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(per_subject, list(fc), chunksize=8)
        p035 = np.stack([r[0] for r in res])
        K_full = correlation_kernel(fc)
        arms[f"B_{scan}"] = ("k", [K_full])
        Kc035 = correlation_kernel(p035)
        arms[f"C_{scan}"] = ("k", [Kc035])
        d0, d1 = [], []
        for gi in range(len(G_GRID)):
            Kp = correlation_kernel(np.stack([r[1][gi] for r in res]))
            d0.append((1 - W_FULL) * Kp + W_FULL * K_full)
            d1.append((1 - W_FULL) * spectral_filter(Kp) + W_FULL * K_full)
        arms[f"D0_{scan}"] = ("k", d0)
        arms[f"U_{scan}"] = ("k", d0 + [Kc035])   # addendum K: D0 and FC^0.35 chosen in-fold
        arms[f"D1_{scan}"] = ("k", d1)
        for t in ["8", "1"]:
            E = np.stack([D[s][f"e{t}_{scan}"] for s in use]).astype(np.float64)
            arms[f"A{t}P1_{scan}"] = ("zlin", E)
            arms[f"A{t}P2_{scan}"] = ("k", [correlation_kernel(E)])
        del fc, res, p035
        print(f"kernels for {scan} built {time.time() - t0:.0f}s", flush=True)

    G.update(y=y, folds=fl, arms=arms)
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    per_seed = {a: np.array([o[1][a] for o in out]) for a in arms}
    per_seed_g = {a: np.array([o[2][a] for o in out]) for a in out[0][2]}
    gsel = {a: sum((o[3][a] for o in out), []) for a in out[0][3]}

    def summ(v):
        return {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    def paired(a, b):
        d = a - b
        return {"mean_diff": float(d.mean()), "positive": int((d > 0).sum()), "of": len(d),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else None,
                "min_diff": float(d.min()), "max_diff": float(d.max())}

    R = {"n_subjects": len(use), "missing_subjects": missing, "n_seeds": len(names),
         "g_grid": G_GRID, "lambdas": LAMBDAS.tolist(),
         "arms": {a: summ(v) for a, v in per_seed.items()},
         "g_curve": {a: dict(zip([str(g) for g in G_GRID], per_seed_g[a].mean(0).round(5).tolist())) for a in per_seed_g},
         "g_selected_counts": {a: {str(g): gsel[a].count(g) for g in G_GRID} for a in gsel},
         "per_seed": {a: v.round(6).tolist() for a, v in per_seed.items()}}
    contrasts = {}
    for scan in ["full", "short"]:
        A = {}
        for t in ["8", "1"]:
            p1, p2 = per_seed[f"A{t}P1_{scan}"], per_seed[f"A{t}P2_{scan}"]
            pick = "P1" if p1.mean() >= p2.mean() else "P2"
            A[t] = per_seed[f"A{t}{pick}_{scan}"]
            R.setdefault("fbert_probe_used", {})[f"A{t}_{scan}"] = pick
        D0, D1, B, C = (per_seed[f"{k}_{scan}"] for k in ["D0", "D1", "B", "C"])
        contrasts[scan] = {"D0-A8": paired(D0, A["8"]), "D0-A1": paired(D0, A["1"]),
                           "D1-A8": paired(D1, A["8"]), "B-A8": paired(B, A["8"]),
                           "C-A8": paired(C, A["8"]), "D0-B": paired(D0, B), "D0-C": paired(D0, C)}
    R["contrasts"] = contrasts
    R["decision_rule"] = "claim only if positive in >= 57/60 seeds"
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in sorted(R["arms"].items()):
        print(f"{a:12s} {v['mean']:.4f} +- {v['sd']:.4f}")
    print(json.dumps(contrasts, indent=1))


if __name__ == "__main__":
    main()
