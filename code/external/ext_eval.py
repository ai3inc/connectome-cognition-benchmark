"""Pre-registered external-cohort test (PREREG_EXTERNAL_COHORT.md): AABC Release 2, first visits, resting-state FC.

Arms (one ridge grid, inner 4-fold selection on the residualised training target, shrinkage g chosen in the inner loop):
  CBIG     the benchmark's correlation kernel: Pearson correlation between two subjects' edge vectors (pairwise)
  PLAIN_I  correlation kernel on the raw FC edges, centred and scaled with training-fold statistics only
  D0_I     conditioned connectome (shrinkage g -> partial correlation -> negative eigenvalues to zero -> eigenvalue
           power 0.10), training-only correlation kernel, mixed 95/5 with PLAIN_I            <- primary vs CBIG
  D1_I     D0_I with the kernel spectral filter fitted on the training block, extended to test subjects (Nystrom)
  D1_T     the filter computed over all subjects of the cohort, as in the benchmark analysis (transductive)
Every inductive kernel is computed exactly from the between-subject Gram matrix: centring a feature vector by the
training mean only changes inner products by row and column means of the Gram matrix over training subjects.

usage: python3 ext_eval.py FC.npz PHENO.csv OUT.json        (the one unblinded run)
       python3 ext_eval.py FC.npz --synthetic OUT.json      (dry run: synthetic target, covariates and families;
                                                              no phenotype file is read)
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 exported in the shell.
"""
import csv, hashlib, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats

G_GRID = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99]
ALPHA, TAU, KEEP, W_FULL = 0.10, 0.5, 150, 0.05
LAMBDAS = np.logspace(-12, 1, 14)
N_REP, N_FOLD, INNER = 20, 10, 4
TARGET = "tlbx_ctc_uncorrected_standard_score"
COVS = ("age_open", "sex", "site")
ARMS = ("CBIG", "PLAIN_I", "D0_I", "D1_I", "D1_T")
PRIMARY = ("D0_I", "CBIG")
SECONDARY = [("D0_I", "PLAIN_I"), ("D1_I", "D0_I"), ("D1_T", "D0_I"), ("D1_T", "CBIG")]
W = {}                                                                   # globals shared with workers (fork)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def to_mat(v, P, iu):
    M = np.zeros((P, P))
    M[iu] = v
    M = M + M.T
    np.fill_diagonal(M, 1.0)
    return M


def conditioned(args):
    """one subject's edges -> conditioned edges for every g in G_GRID"""
    v, P = args
    iu = np.triu_indices(P, 1)
    C = to_mat(v.astype(np.float64), P, iu)
    out = []
    for g in G_GRID:
        Pm = np.linalg.inv((1 - g) * C + g * np.eye(P))
        d = np.sqrt(np.clip(np.diag(Pm), 1e-12, None))
        R = -Pm / np.outer(d, d)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)
        out.append(((V * np.clip(w, 0, None) ** ALPHA) @ V.T)[iu].astype(np.float32))
    return out


def ind_kernel(G, tr):
    """correlation kernel with centring and scaling from training rows only (as eval_holdout.inductive_corr_kernel)"""
    r = G[:, tr].mean(1)
    m = G[np.ix_(tr, tr)].mean()
    Kc = G - r[:, None] - r[None, :] + m
    d = np.sqrt(np.clip(np.diag(Kc), 1e-12, None))
    return Kc / np.outer(d, d) / len(G)


def trans_kernel(G):
    """the benchmark analysis' correlation kernel: columns centred over all subjects, rows unit norm, HKH, trace 1"""
    K = ind_kernel(G, np.arange(len(G))) * len(G)
    n = len(G)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ K @ H
    return K / np.trace(K)


def spectral_filter(K):
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0.0, None)
    f = np.clip(w - TAU * np.std(w), 0.0, None)
    if KEEP < len(f):
        thr = np.sort(f)[::-1][KEEP - 1]
        f = np.where(f >= thr, f, 0.0)
    Kt = (V * f) @ V.T
    return Kt / np.trace(Kt)


def inductive_filter(Ku, KF, tr, te):
    """filter fitted on the training block of Ku, extended to test subjects; returns an n x n kernel whose
    tr x tr and te x tr blocks are defined (the only blocks KRR uses)"""
    n, ntr = len(Ku), len(tr)
    w, V = np.linalg.eigh(Ku[np.ix_(tr, tr)])
    w = np.clip(w, 0.0, None)
    f = np.clip(w - TAU * np.std(w), 0.0, None)
    if KEEP < len(f):
        thr = np.sort(f)[::-1][KEEP - 1]
        f = np.where(f >= thr, f, 0.0)
    s = f.sum() / (ntr / n)
    g = np.where(w > 1e-12, f / np.where(w > 1e-12, w, 1.0), 0.0) / s
    K = np.zeros((n, n))
    K[np.ix_(tr, tr)] = (1 - W_FULL) * ((V * (f / s)) @ V.T) + W_FULL * KF[np.ix_(tr, tr)]
    K[np.ix_(te, tr)] = (1 - W_FULL) * ((Ku[np.ix_(te, tr)] @ V * g) @ V.T) + W_FULL * KF[np.ix_(te, tr)]
    return K


def krr_select(Ks, tr, te, y, seed):
    """choose (kernel, lambda) by inner INNER-fold mean r on the training rows; predict the test rows"""
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(tr))
    S = np.zeros((len(Ks), len(LAMBDAS)))
    for part in np.array_split(perm, INNER):
        a, b = tr[np.setdiff1d(perm, part)], tr[part]
        ya = y[a] - y[a].mean()
        for ki, K in enumerate(Ks):
            w, V = np.linalg.eigh(K[np.ix_(a, a)])
            VtK = K[np.ix_(b, a)] @ V
            Vt = V.T @ ya
            for li, lam in enumerate(LAMBDAS):
                p = VtK @ (Vt / (w + lam))
                S[ki, li] += (np.corrcoef(p, y[b])[0, 1] if p.std() > 0 else 0.0) / INNER
    ki, li = np.unravel_index(np.nanargmax(S), S.shape)
    K, lam = Ks[ki], LAMBDAS[li]
    w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
    yc = y[tr] - y[tr].mean()
    return (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean(), int(ki), float(lam)


def make_folds(fam, site, seed):
    """N_FOLD folds; whole families per fold, balanced within acquisition site"""
    rs = np.random.RandomState(seed)
    fams = list(dict.fromkeys(fam))
    rs.shuffle(fams)
    members = {f: np.where(fam == f)[0] for f in fams}
    fams.sort(key=lambda f: -len(members[f]))                         # stable: big families first, ties shuffled
    sites = sorted(set(site))
    load = {s: np.zeros(N_FOLD) for s in sites}
    total = np.zeros(N_FOLD)
    fold = np.empty(len(fam), dtype=int)
    for f in fams:
        idx = members[f]
        s = max(set(site[idx]), key=list(site[idx]).count)
        k = int(np.lexsort((rs.rand(N_FOLD), total, load[s]))[0])
        fold[idx] = k
        load[s][k] += len(idx)
        total[k] += len(idx)
    return [np.where(fold == k)[0] for k in range(N_FOLD)]


def residualise(y, Z, tr):
    beta, *_ = np.linalg.lstsq(Z[tr], y[tr], rcond=None)
    return y - Z @ beta


def run_null(args):
    sd, rep = args
    W["y"] = np.random.RandomState(sd).randn(len(W["Z"]))
    return sd, run_rep(rep)


def run_rep(rep):
    y, Z, fam, site = W["y"], W["Z"], W["fam"], W["site"]
    n = len(y)
    folds = make_folds(fam, site, 1000 + rep)
    out = {a: [] for a in ARMS}
    picks = {a: [] for a in ARMS}
    sizes = []
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        sizes.append([int(len(te)), int(len(tr))])
        yr = residualise(y, Z, tr)
        KF = ind_kernel(W["G_full"], tr)
        Ku = [ind_kernel(Gp, tr) for Gp in W["G_part"]]
        cands = {"CBIG": [W["K_cbig"]], "PLAIN_I": [KF],
                 "D0_I": [(1 - W_FULL) * K + W_FULL * KF for K in Ku],
                 "D1_I": [inductive_filter(K, KF, tr, te) for K in Ku],
                 "D1_T": W["K_d1t"]}
        for a in ARMS:
            p, ki, lam = krr_select(cands[a], tr, te, yr, seed=rep * 100 + k)
            out[a].append(float(np.corrcoef(p, yr[te])[0, 1]))
            picks[a].append([G_GRID[ki] if len(cands[a]) > 1 else None, lam])
    return rep, out, picks, sizes


def nb(d, ratio):
    d = np.asarray(d, dtype=float).ravel()
    t = d.mean() / math.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))
    return {"diff": round(float(d.mean()), 4), "folds": int(len(d)), "corrected_t": round(float(t), 3),
            "p_two_sided": float(f"{2 * stats.t.sf(abs(t), len(d) - 1):.3g}"),
            "splits_positive": int((np.asarray(d).reshape(N_REP, N_FOLD).mean(1) > 0).sum())}


def main():
    fc_p, ph, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    F = np.load(fc_p, allow_pickle=False)
    ids, edges, complete = F["ids"].astype(str), F["edges"], F["complete"]
    P = len(F["parcels"])
    info = {"fc_file": os.path.basename(fc_p), "fc_sha16": sha16(fc_p), "visits_V1": int(len(ids)), "fc_complete": int(complete.sum())}
    if ph == "--synthetic":
        rs = np.random.RandomState(int(os.environ.get("DRY_SEED", "0")))
        keep = np.where(complete)[0]
        n = len(keep)
        y = rs.randn(n)
        age = rs.uniform(36, 90, n)
        sex = rs.randint(0, 2, n).astype(float)
        site = np.array(["S%d" % s for s in rs.randint(0, 4, n)])
        fam = np.array(["F%d" % (i // 2 if rs.rand() < 0.1 else 10000 + i) for i in range(n)])
        info.update({"mode": "synthetic (dry run, no phenotype read)", "n": n})
    else:
        rows = {r["id_event"]: r for r in csv.DictReader(open(ph, encoding="utf-8-sig"))}
        pos = {s: i for i, s in enumerate(ids)}
        keep, y, age, sex, site, fam = [], [], [], [], [], []
        drop = {"no_phenotype_row": 0, "fc_incomplete": 0, "missing_target_or_covariate": 0}
        for s in ids:
            r = rows.get(s)
            if r is None:
                drop["no_phenotype_row"] += 1; continue
            if not complete[pos[s]]:
                drop["fc_incomplete"] += 1; continue
            vals = [r.get(TARGET, ""), r.get("age_open", ""), r.get("sex", ""), r.get("site", ""), r.get("pedid", "")]
            try:
                yt, ag = float(vals[0]), float(vals[1])
                assert math.isfinite(yt) and math.isfinite(ag) and vals[2] in ("M", "F") and vals[3].strip()
            except (ValueError, AssertionError):
                drop["missing_target_or_covariate"] += 1; continue
            keep.append(pos[s]); y.append(yt); age.append(ag); sex.append(1.0 if vals[2] == "M" else 0.0)
            site.append(vals[3].strip()); fam.append(vals[4].strip() or ("solo_" + s))
        keep, y, age, sex = np.array(keep), np.array(y), np.array(age), np.array(sex)
        site, fam = np.array(site), np.array(fam)
        n = len(keep)
        info.update({"mode": "unblinded", "pheno_file": os.path.basename(ph), "pheno_sha16": sha16(ph), "n": n, "excluded": drop,
                     "families": int(len(set(fam))), "sites": {s: int((site == s).sum()) for s in sorted(set(site))}})
    sites = sorted(set(site))
    Z = np.column_stack([np.ones(n), age, age ** 2, sex] + [(site == s).astype(float) for s in sites[1:]])
    X = edges[keep].astype(np.float32)
    # full-correlation Gram and the benchmark's kernel
    G_full = (X.astype(np.float64) @ X.T.astype(np.float64))
    Xr = X - X.mean(1, keepdims=True)
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    K_cbig = (Xr @ Xr.T).astype(np.float64) / n
    del Xr
    # conditioned Gram per g
    with Pool(8) as pool:
        parts = pool.map(conditioned, [(v, P) for v in X], chunksize=8)
    G_part = []
    for gi in range(len(G_GRID)):
        Xg = np.stack([p[gi] for p in parts])
        G_part.append(Xg.astype(np.float64) @ Xg.T.astype(np.float64))
        del Xg
    del parts
    KFt = trans_kernel(G_full)
    K_d1t = [(1 - W_FULL) * spectral_filter(trans_kernel(Gp)) + W_FULL * KFt for Gp in G_part]
    info["kernels_seconds"] = round(time.time() - t0)
    print(json.dumps(info), flush=True)
    W.update(y=y, Z=Z, fam=fam, site=site, G_full=G_full, G_part=G_part, K_cbig=K_cbig, K_d1t=K_d1t)
    if ph == "--synthetic" and os.environ.get("DRY_NULL_SEEDS"):          # null check: many synthetic targets, same kernels
        seeds = [int(x) for x in os.environ["DRY_NULL_SEEDS"].split(",")]
        with Pool(8) as pool:
            rr = pool.map(run_null, [(sd, 0) for sd in seeds])
        nulls = {sd: {a: round(float(np.mean(r[1][a])), 4) for a in ARMS} for sd, r in rr}
        for sd in seeds:
            print(sd, nulls[sd], flush=True)
        nulls["mean_over_seeds"] = {a: round(float(np.mean([nulls[sd][a] for sd in seeds])), 4) for a in ARMS}
        print("mean", nulls["mean_over_seeds"], flush=True)
        json.dump({"null_check": nulls, "info": info}, open(outp, "w"), indent=1)
        return
    with Pool(8) as pool:
        res = sorted(pool.map(run_rep, range(N_REP)))
    S = {a: np.array([r[1][a] for r in res]) for a in ARMS}                 # N_REP x N_FOLD
    sizes = [s for r in res for s in r[3]]
    ratio = float(np.mean([te / tr for te, tr in sizes]))
    out = {"prereg": "PREREG_EXTERNAL_COHORT.md", "info": info, "settings": {"G_GRID": G_GRID, "ALPHA": ALPHA, "TAU": TAU, "KEEP": KEEP,
           "W_FULL": W_FULL, "LAMBDAS": [float(x) for x in LAMBDAS], "N_REP": N_REP, "N_FOLD": N_FOLD, "INNER": INNER, "TARGET": TARGET,
           "covariates": ["intercept", "age_open", "age_open^2", "sex", "site dummies"]},
           "test_train_ratio": round(ratio, 5),
           "mean_perfold_r": {a: round(float(S[a].mean()), 4) for a in ARMS},
           "split_sd": {a: round(float(S[a].mean(1).std()), 4) for a in ARMS},
           "primary": {"%s_vs_%s" % PRIMARY: nb(S[PRIMARY[0]] - S[PRIMARY[1]], ratio)},
           "secondary": {"%s_vs_%s" % (a, b): nb(S[a] - S[b], ratio) for a, b in SECONDARY},
           "g_chosen": {a: {str(g): int(sum(1 for r in res for p in r[2][a] if p[0] == g)) for g in G_GRID} for a in ("D0_I", "D1_I", "D1_T")},
           "per_fold": {a: S[a].tolist() for a in ARMS}, "fold_sizes": sizes, "seconds": round(time.time() - t0)}
    pr = out["primary"]["%s_vs_%s" % PRIMARY]
    out["verdict"] = "CONFIRMED" if (pr["diff"] > 0 and pr["corrected_t"] >= stats.t.ppf(0.975, pr["folds"] - 1)) else "NOT CONFIRMED"
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("mean_perfold_r", "primary", "secondary", "verdict", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
