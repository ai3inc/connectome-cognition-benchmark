"""Metric check for the cross-paper comparisons (SOTA review, 2026-11-18).

Ooi et al. (2022) report, per split, the MEAN of the ten per-fold Pearson r
(results/README.md of ThomasYeoLab/Ooi2022_MMP_HCP: "Results from the 10 fold-cross
validation from each split were averaged"), after regressing age and sex from the target.
code/canonical.py reports ONE r per split over the 753 pooled predictions, on the raw
target. This script scores the same predictions both ways, on the benchmark's own
features and folds, and compares with the numbers in Ooi's own result files.

Part A (753, raw target): arms scored as pooled r and as mean per-fold r; per-fold
         values saved, so the Nadeau-Bengio corrected resampled t-test is exact.
Part B (the 279 of the 753 in the public HCP file with age band and sex): raw target vs
         age+sex residualised in-fold (CBIG convention: coefficients from the training
         fold), same kernels and folds restricted to the subset.

usage: python3 metric_check.py WS OOI_KRR.mat OOI_COMBINED.mat BEH.csv OUT.json
WS holds mod_{rs,lang,social,gamb,motor,wm}.npz (key bp, 753 x 79800), ooi_comp.csv,
off753.txt, splits.mat, canonical.py.
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

REST_ONLY = os.environ.get("REST_ONLY") == "1"      # resting arm alone, against Ooi's KRR rs 0.5276
VARIANTS = os.environ.get("VARIANTS") == "1"        # which parts of the chain are inductive
LAMS = tuple(float(x) for x in os.environ["LAMS"].split(",")) if os.environ.get("LAMS") else None
MODS = ["rs"] if REST_ONLY else ["rs", "lang", "social", "gamb", "motor", "wm"]
OOI_FMRI_ORDER = ["rs", "social", "gamb", "lang", "wm", "motor"]
OOI_COMBINED_ORDER = ["best", "multiKRR_six", "stacking_LRR_six", "stacking_LRR_all"]
G = {}


def cbig_corr_kernel(X):
    """CBIG 'corr' kernel: Pearson r between two subjects' feature vectors. Inductive."""
    X = np.asarray(X, dtype=np.float64)
    Xr = X - X.mean(1, keepdims=True)
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    return Xr @ Xr.T


def resid(ytr, yte, Ctr, Cte):
    A = np.column_stack([np.ones(len(ytr)), Ctr])
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return ytr - A @ b, yte - np.column_stack([np.ones(len(yte)), Cte]) @ b


def fit_predict_lam(Ktr, Kte, ytr, lams):
    """canonical._fit_predict with a configurable grid; also returns the chosen lambda."""
    R = np.zeros((len(lams), 4))
    for fi, (a, b) in enumerate(cn._inner_splits(len(ytr))):
        w, V = np.linalg.eigh(Ktr[np.ix_(a, a)])
        Vt = V.T @ (ytr[a] - ytr[a].mean())
        VtK = Ktr[np.ix_(b, a)] @ V
        for li, lam in enumerate(lams):
            R[li, fi] = np.corrcoef(VtK @ (Vt / (w + lam)), ytr[b])[0, 1]
    li = int(np.argmax(np.nan_to_num(R, nan=-1).mean(1)))
    w, V = np.linalg.eigh(Ktr)
    return (Kte @ V) @ (V.T @ (ytr - ytr.mean()) / (w + lams[li])) + ytr.mean(), lams[li]


def run_seed(nm):
    with threadpool_limits(1):
        out = {}
        for part, (y, fl, arms, C) in G["parts"].items():
            n = len(y)
            for target in (["raw", "resid"] if C is not None else ["raw"]):
                for a, K in arms.items():
                    pred = np.full(n, np.nan)
                    yt = np.full(n, np.nan)
                    pf = []
                    for te in fl[nm]:
                        tr = np.setdiff1d(np.arange(n), te)
                        ytr, yte = y[tr], y[te]
                        if target == "resid":
                            ytr, yte = resid(ytr, yte, C[tr], C[te])
                        if LAMS:
                            pred[te], lam = fit_predict_lam(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], ytr, LAMS)
                            out.setdefault(("lam", part, target, a), []).append(lam)
                        else:
                            pred[te] = cn._fit_predict(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], ytr)
                        yt[te] = yte
                        pf.append(float(np.corrcoef(pred[te], yte)[0, 1]))
                    out[(part, target, a)] = (float(np.corrcoef(pred, yt)[0, 1]), pf)
        return nm, out


def nb_exact(pf_a, pf_b, ratio):
    d = (np.asarray(pf_a) - np.asarray(pf_b)).ravel()
    J = len(d)
    tn = d.mean() / (d.std(ddof=1) / np.sqrt(J))
    tc = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
    from scipy import stats
    return {"diff_mean_perfold_r": round(float(d.mean()), 4), "folds": J, "naive_t": round(float(tn), 2),
            "corrected_t": round(float(tc), 2), "p_two_sided": float(f"{2 * stats.t.sf(abs(tc), J - 1):.3g}")}


def ooi_tables(krr_path, comb_path):
    res = {}
    for tag, p, order in [("KRR_single", krr_path, None), ("combined", comb_path, OOI_COMBINED_ORDER)]:
        m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        res[tag + "_vars"] = {k: str(getattr(v, "shape", type(v).__name__)) for k, v in m.items() if not k.startswith("__")}

        def walk(x, path):
            if isinstance(x, np.ndarray) and x.dtype != object and x.ndim == 3 and x.shape[1] == 60:
                yield path, x
            elif hasattr(x, "_fieldnames"):
                for f in x._fieldnames:
                    yield from walk(getattr(x, f), path + "." + f)
        for k, v in m.items():
            if k.startswith("__"):
                continue
            for path, arr in walk(v, k):
                cog = arr[0]                                     # Cognition row: 60 splits x models
                names = order if (order and cog.shape[1] == len(order)) else (
                    OOI_FMRI_ORDER if cog.shape[1] == 6 else [str(i) for i in range(cog.shape[1])])
                res[f"{tag}:{path}"] = {nme: {"mean": round(float(cog[:, i].mean()), 4),
                                               "sd": round(float(cog[:, i].std()), 4),
                                               "per_split": cog[:, i].round(5).tolist()}
                                        for i, nme in enumerate(names)}
    return res


def main(krr_path, comb_path, beh_path, outp):
    t0 = time.time()
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(WS + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(WS + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    pos = {s: i for i, s in enumerate(off)}
    idx = np.array([pos[s] for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(WS + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:60]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in f.subject_list if str(s) in sidx])
                             for f in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    KP_f, KP_u, KF, KC, KV = {}, {}, {}, {}, {}
    for m in MODS:
        e = np.load(WS + f"/mod_{m}.npz")["bp"][idx].astype(np.float64)
        KC[m] = cbig_corr_kernel(e)
        full_e, part_e = cn._per_subject(e, cn.G_SHRINK)
        del e
        KF[m] = cn.correlation_kernel(full_e)
        Ap = cn.alpha_filter(part_e)
        Ku = cn.correlation_kernel(Ap)
        KP_u[m], KP_f[m] = Ku, cn.spectral_filter(Ku)
        if VARIANTS:
            KV.setdefault("tau", {})[m] = cn.spectral_filter(Ku, keep=None)        # soft threshold only
            KV.setdefault("noalpha", {})[m] = cn.correlation_kernel(part_e)       # partial, no alpha, no filter
            KV.setdefault("cbigpart", {})[m] = cbig_corr_kernel(Ap)               # partial+alpha, CBIG kernel
            KV.setdefault("cbigpart_noalpha", {})[m] = cbig_corr_kernel(part_e)
        del full_e, part_e, Ap
        print(m, f"{time.time() - t0:.0f}s", flush=True)

    def ens(KP, lang2):
        w = {m: (2 if (lang2 and m == "lang") else 1) for m in MODS}
        return sum(w[m] * ((1 - cn.W_FULL) * KP[m] + cn.W_FULL * KF[m]) for m in MODS) / sum(w.values())
    if VARIANTS:
        def ens2(KP, KFULL):
            w = {m: (2 if m == "lang" else 1) for m in MODS}
            return sum(w[m] * ((1 - cn.W_FULL) * KP[m] + cn.W_FULL * KFULL[m]) for m in MODS) / sum(w.values())
        arms = {"CANON_lang2": ens(KP_f, True), "NOFILT_lang2": ens(KP_u, True),
                "TAUONLY_lang2": ens2(KV["tau"], KF), "NOALPHA_NOFILT_lang2": ens2(KV["noalpha"], KF),
                "CBIGPART_lang2": ens2(KV["cbigpart"], KC), "CBIGPART_NOALPHA_lang2": ens2(KV["cbigpart_noalpha"], KC),
                "CBIG6_eq": sum(KC.values()) / 6}
    elif REST_ONLY:
        arms = {"CANON_rs": ens(KP_f, False), "NOFILT_rs": ens(KP_u, False),
                "PLAIN_hkh_rs": KF["rs"], "CBIG_rs": KC["rs"]}
    else:
        arms = {"CANON_lang2": ens(KP_f, True), "CANON_eq": ens(KP_f, False),
                "NOFILT_lang2": ens(KP_u, True), "NOFILT_eq": ens(KP_u, False),
                "PLAIN6_hkh_eq": sum(KF.values()) / 6, "CBIG6_eq": sum(KC.values()) / 6}
        for m in MODS:
            arms[f"CBIG_{m}"] = KC[m]
    parts = {"A753": (y, fl, arms, None)}
    # Part B: the subset with public age band and sex
    beh = {r["Subject"].strip(): r for r in csv.DictReader(open(beh_path))}
    sub = [i for i, s in enumerate(use) if s in beh and beh[s]["Gender"] in ("M", "F") and beh[s]["Age"]]
    bands = sorted({beh[use[i]]["Age"] for i in sub})
    C = np.array([[1.0 if beh[use[i]]["Gender"] == "M" else 0.0] +
                  [1.0 if beh[use[i]]["Age"] == b else 0.0 for b in bands[1:]] for i in sub])
    pos_sub = {i: j for j, i in enumerate(sub)}
    flB = {nm: [np.array([pos_sub[i] for i in te if i in pos_sub]) for te in fl[nm]] for nm in names}
    flB = {nm: [te for te in v if len(te) >= 8] for nm, v in flB.items()}
    sb = np.array(sub)
    armsB = {a: arms[a][np.ix_(sb, sb)] for a in (list(arms) if (REST_ONLY or VARIANTS) else
                                                    ["CANON_lang2", "NOFILT_lang2", "CBIG6_eq", "CBIG_rs"])}
    parts["B_sub"] = (y[sb], flB, armsB, C)
    G["parts"] = parts
    print(f"753 arms {len(arms)}; subset n={len(sub)} bands={bands}", flush=True)
    with Pool(8) as pool:
        res = dict(pool.map(run_seed, names))
    lamhist = {}
    for nm in names:
        for k, v in list(res[nm].items()):
            if k[0] == "lam":
                for lam in v:
                    h = lamhist.setdefault("|".join(k[1:]), {})
                    h[str(lam)] = h.get(str(lam), 0) + 1
                del res[nm][k]
    keys = list(res[names[0]].keys())
    summ, perfold = {}, {}
    for k in keys:
        pooled = np.array([res[nm][k][0] for nm in names])
        pf = np.array([res[nm][k][1] for nm in names])           # seeds x folds
        lab = "|".join(k)
        summ[lab] = {"pooled_r": round(float(pooled.mean()), 4), "pooled_sd": round(float(pooled.std()), 4),
                     "mean_perfold_r": round(float(pf.mean()), 4), "perfold_seed_sd": round(float(pf.mean(1).std()), 4),
                     "perfold_fold_sd": round(float(pf.std()), 4),
                     "fisherz_perfold_r": round(float(np.tanh(np.arctanh(np.clip(pf, -0.999, 0.999)).mean())), 4),
                     "pooled_minus_perfold": round(float(pooled.mean() - pf.mean()), 4)}
        perfold[lab] = pf.round(5).tolist()
    ratio = float(np.mean([len(te) / (len(y) - len(te)) for nm in names for te in fl[nm]]))
    P = lambda a, b, part="A753", t="raw": nb_exact(perfold[f"{part}|{t}|{a}"], perfold[f"{part}|{t}|{b}"], ratio)
    pairs = ({f"{a}_vs_CBIG6": (a, "CBIG6_eq") for a in arms if a != "CBIG6_eq"} if VARIANTS else
             {"CANON_rs_vs_CBIG_rs": ("CANON_rs", "CBIG_rs"), "NOFILT_rs_vs_CBIG_rs": ("NOFILT_rs", "CBIG_rs")}
             if REST_ONLY else
             {"CANON_vs_CBIG6": ("CANON_lang2", "CBIG6_eq"), "CANON_vs_NOFILT": ("CANON_lang2", "NOFILT_lang2"),
              "NOFILT_vs_CBIG6": ("NOFILT_lang2", "CBIG6_eq"), "CANON_vs_PLAIN6_hkh": ("CANON_lang2", "PLAIN6_hkh_eq")})
    R = {"date": "2026-11-18", "purpose": __doc__.strip().splitlines()[0], "rest_only": REST_ONLY, "variants": VARIANTS,
         "lambda_grid": list(LAMS) if LAMS else list(cn.LAMBDAS), "lambda_chosen": lamhist,
         "n": len(y), "n_subset": len(sub), "test_train_ratio": round(ratio, 4), "summary": summ,
         "nadeau_bengio_exact_A753": {k: P(*v) for k, v in pairs.items()},
         "nadeau_bengio_exact_subset_resid": {k: P(*v, part="B_sub", t="resid") for k, v in pairs.items()
                                              if all(f"B_sub|resid|{a}" in perfold for a in v)},
         "ooi_own_results": ooi_tables(krr_path, comb_path),
         "per_fold_r": perfold, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("per_fold_r", "ooi_own_results")}, indent=1))
    print(json.dumps({k: ({a: {"mean": b["mean"], "sd": b["sd"]} for a, b in v.items()} if isinstance(v, dict)
                          and all(isinstance(b, dict) for b in v.values()) else v)
                      for k, v in R["ooi_own_results"].items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[2:6])
