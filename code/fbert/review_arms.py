"""Arms requested by the manuscript review (2026-11-19), scored the benchmark's way. POST HOC.

Same features, subjects, splits, covariates (public age band + sex, in-fold OLS), metric
(mean per-fold r) and ridge grid as results/METRIC_CHECK_COV753_1118.json, so CANON, NOFILT
and CBIG6 must reproduce 0.7215 / 0.6948 / 0.6798 exactly (sanity check, asserted).

Arms (language x2 unless marked eq):
  CANON        conditioned chain + kernel filter (the headline)
  NOFILT       conditioned chain, no filter
  FULLW1       our kernel on full correlations only (no partial, no filter)        } in-repo
  PARTW0       partial correlation + eigenvalue power, no full mix, no filter      } re-implementation
  PARTNA_W0    partial correlation without the eigenvalue power, no mix, no filter } of the decomposition
  FULL_FILT    the kernel filter applied to the full-correlation kernel (no conditioning)
  CBIG6        the benchmark's kernel (pairwise Pearson between subjects), eq
  CBIG6_HKH    the benchmark's kernel double-centred and trace-normalised like ours, eq
  CBIG6_FILT   CBIG6_HKH + the kernel filter, eq
  CANON_EQ     the headline chain with equal modality weights
  CANON_IND    the headline chain with an INDUCTIVE filter: eigenvectors and the threshold
               from the training block only, test rows by Nystrom extension
               K~(te,tr) = K(te,tr) V diag(f(w)/w) V^T, trace normalised on the training block

usage: python3 review_arms.py WS COVARIATES.csv OUT.json [N_SEEDS]
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
from metric_check import cbig_corr_kernel, resid, nb_exact, fit_predict_lam

MODS = ["rs", "lang", "social", "gamb", "motor", "wm"]
WL = {m: (2 if m == "lang" else 1) for m in MODS}
WE = {m: 1 for m in MODS}
LAMS = (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # = METRIC_CHECK_COV753_1118 grid
G = {}


def hkh_trace(K):
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ K @ H
    return K / np.trace(K)


def ens(blocks, w):
    return sum(w[m] * blocks[m] for m in MODS) / sum(w.values())


def inductive_blocks(tr, te):
    """Training and test-by-training blocks of CANON with the filter computed on the training block only."""
    Ktr = 0.0
    Kte = 0.0
    ntr, n = len(tr), len(G["y"])
    for m in MODS:
        Ku, KF = G["Ku"][m], G["KF"][m]
        A = Ku[np.ix_(tr, tr)]
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        f = np.clip(w - cn.TAU * np.std(w), 0.0, None)
        if cn.KEEP < len(f):
            thr = np.sort(f)[::-1][cn.KEEP - 1]
            f = np.where(f >= thr, f, 0.0)
        s = f.sum() / (ntr / n)            # training trace -> same share as the training block of a trace-1 kernel
        Ftr = (V * (f / s)) @ V.T
        g = np.where(w > 1e-12, f / np.where(w > 1e-12, w, 1.0), 0.0) / s
        Fte = (Ku[np.ix_(te, tr)] @ V * g) @ V.T
        Ktr = Ktr + WL[m] * ((1 - cn.W_FULL) * Ftr + cn.W_FULL * KF[np.ix_(tr, tr)])
        Kte = Kte + WL[m] * ((1 - cn.W_FULL) * Fte + cn.W_FULL * KF[np.ix_(te, tr)])
    z = sum(WL.values())
    return Ktr / z, Kte / z


def run_seed(nm):
    with threadpool_limits(1):
        y, C, arms = G["y"], G["C"], G["arms"]
        n = len(y)
        out = {}
        for target in ("raw", "resid"):
            pf = {a: [] for a in list(arms) + ["CANON_IND"]}
            lam = {a: [] for a in pf}
            for te in G["folds"][nm]:
                tr = np.setdiff1d(np.arange(n), te)
                ytr, yte = y[tr], y[te]
                if target == "resid":
                    ytr, yte = resid(ytr, yte, C[tr], C[te])
                for a, K in arms.items():
                    p, l = fit_predict_lam(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], ytr, LAMS)
                    pf[a].append(float(np.corrcoef(p, yte)[0, 1]))
                    lam[a].append(l)
                Ktr, Kte = inductive_blocks(tr, te)
                p, l = fit_predict_lam(Ktr, Kte, ytr, LAMS)
                pf["CANON_IND"].append(float(np.corrcoef(p, yte)[0, 1]))
                lam["CANON_IND"].append(l)
            out[target] = (pf, lam)
        return nm, out


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
    B = {k: {} for k in ["Kf", "Ku", "KF", "Kna", "KFf", "KC", "KCn", "KCf"]}
    for m in MODS:
        e = np.load(WS + f"/mod_{m}.npz")["bp"][idx].astype(np.float64)
        B["KC"][m] = cbig_corr_kernel(e)
        B["KCn"][m] = hkh_trace(B["KC"][m])
        B["KCf"][m] = cn.spectral_filter(B["KCn"][m])
        full_e, part_e = cn._per_subject(e, cn.G_SHRINK)
        del e
        B["KF"][m] = cn.correlation_kernel(full_e)
        B["KFf"][m] = cn.spectral_filter(B["KF"][m])
        B["Kna"][m] = cn.correlation_kernel(part_e)
        Ap = cn.alpha_filter(part_e)
        B["Ku"][m] = cn.correlation_kernel(Ap)
        B["Kf"][m] = cn.spectral_filter(B["Ku"][m])
        del full_e, part_e, Ap
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    mix = lambda P, w: ens({m: (1 - cn.W_FULL) * B[P][m] + cn.W_FULL * B["KF"][m] for m in MODS}, w)
    arms = {"CANON": mix("Kf", WL), "NOFILT": mix("Ku", WL), "FULLW1": ens(B["KF"], WL), "PARTW0": ens(B["Ku"], WL),
            "PARTNA_W0": ens(B["Kna"], WL), "FULL_FILT": ens(B["KFf"], WL), "CBIG6": ens(B["KC"], WE),
            "CBIG6_HKH": ens(B["KCn"], WE), "CBIG6_FILT": ens(B["KCf"], WE), "CANON_EQ": mix("Kf", WE)}
    G.update(y=y, C=C, arms=arms, folds=fl, Ku=B["Ku"], KF=B["KF"])
    with Pool(8) as pool:
        res = pool.map(run_seed, seeds)
    names = list(arms) + ["CANON_IND"]
    R = {"date": "2026-11-19", "purpose": __doc__.strip().splitlines()[0], "n": len(y), "seeds": len(seeds),
         "lambda_grid": list(LAMS), "covariates": "sex + age band (public), in-fold OLS", "post_hoc": True}
    ratio = float(np.mean([len(te) / (len(y) - len(te)) for nm in seeds for te in fl[nm]]))
    R["test_train_ratio"] = round(ratio, 4)
    for target in ("raw", "resid"):
        pf = {a: np.array([r[1][target][0][a] for r in res]) for a in names}
        lam = {a: [l for r in res for l in r[1][target][1][a]] for a in names}
        R[target] = {"mean_perfold_r": {a: round(float(v.mean()), 4) for a, v in pf.items()},
                     "seed_sd": {a: round(float(v.mean(1).std()), 4) for a, v in pf.items()},
                     "lambda_chosen": {a: {str(x): lam[a].count(x) for x in sorted(set(lam[a]))} for a in names},
                     "per_fold_r": {a: v.round(5).tolist() for a, v in pf.items()}}
        pairs = {"CANON_vs_CBIG6": ("CANON", "CBIG6"), "NOFILT_vs_CBIG6": ("NOFILT", "CBIG6"),
                 "filter_on_conditioned": ("CANON", "NOFILT"), "filter_on_full": ("FULL_FILT", "FULLW1"),
                 "conditioning_without_filter": ("NOFILT", "FULLW1"), "conditioning_with_filter": ("CANON", "FULL_FILT"),
                 "filter_on_benchmark_kernel": ("CBIG6_FILT", "CBIG6_HKH"), "benchmark_kernel_normalisation": ("CBIG6_HKH", "CBIG6"),
                 "CBIG6_FILT_vs_CBIG6": ("CBIG6_FILT", "CBIG6"), "FULL_FILT_vs_CBIG6": ("FULL_FILT", "CBIG6"),
                 "FULLW1_vs_CBIG6": ("FULLW1", "CBIG6"), "partial_step": ("PARTW0", "FULLW1"),
                 "partial_without_power": ("PARTNA_W0", "FULLW1"), "power_step": ("PARTW0", "PARTNA_W0"),
                 "mix_step": ("NOFILT", "PARTW0"), "language_weight": ("CANON", "CANON_EQ"),
                 "inductive_filter_vs_none": ("CANON_IND", "NOFILT"), "transductive_vs_inductive_filter": ("CANON", "CANON_IND"),
                 "CANON_IND_vs_CBIG6": ("CANON_IND", "CBIG6"), "CANON_EQ_vs_CBIG6": ("CANON_EQ", "CBIG6")}
        R[target]["NB"] = {k: nb_exact(pf[a], pf[b], ratio) for k, (a, b) in pairs.items()}
    rr = R["resid"]["mean_perfold_r"]
    R["sanity"] = {"CANON": rr["CANON"], "NOFILT": rr["NOFILT"], "CBIG6": rr["CBIG6"],
                   "reproduces_METRIC_CHECK_COV753": abs(rr["CANON"] - 0.7215) < 1e-4 and abs(rr["NOFILT"] - 0.6948) < 1e-4
                   and abs(rr["CBIG6"] - 0.6798) < 1e-4}
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({"sanity": R["sanity"], "resid": R["resid"]["mean_perfold_r"],
                      "NB_resid": {k: (v["diff_mean_perfold_r"], v["corrected_t"]) for k, v in R["resid"]["NB"].items()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 60)
