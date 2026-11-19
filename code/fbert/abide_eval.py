"""Addendum J evaluation: ABIDE-I, site-grouped folds (20 repeats; whole sites
assigned to 10 folds by RandomState(9000 + s)), autism diagnosis (AUC of the KRR
score on +/-1 labels) and FIQ (r). Arms as AOMIC; each arm's cleaning variant
{noGSR, GSR} and knobs chosen in-fold (inner 4-fold mean r).

usage: python3 abide_eval.py PHENO.csv FEAT_GLOB OUT.json
"""
import csv, glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import aomic_eval as ae          # per_subject_g and the same arm construction

REPS = 20
G = {}


def auc(score, lab):
    pos, neg = score[lab > 0], score[lab < 0]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def site_folds(sites, s):
    u = sorted(set(sites))
    perm = np.random.RandomState(9000 + s).permutation(len(u))
    fold_of = {u[j]: k % 10 for k, j in enumerate(perm)}
    f = np.array([fold_of[x] for x in sites])
    return [np.nonzero(f == k)[0] for k in range(10)]


def run_rep(args):
    s, task = args
    with threadpool_limits(1):
        y, idx, arms, sites = G[task]["y"], G[task]["idx"], G["arms"], G[task]["sites"]
        n = len(y)
        pred = {a: np.full(n, np.nan) for a in arms}
        for te in site_folds(sites, s):
            tr = np.setdiff1d(np.arange(n), te)
            for a, spec in arms.items():
                Ks = []
                for kind, E in spec:
                    Esub = E[np.ix_(idx, idx)] if kind == "k" else E[idx]
                    Ks.append(ev.zlin_kernel(Esub, tr) if kind == "zlin" else Esub)
                pred[a][te] = ev.krr_select(Ks, tr, te, y)[0]
        if task == "dx":
            return s, task, {a: auc(v, y) for a, v in pred.items()}
        return s, task, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}


def main():
    php, fglob, outp = sys.argv[1:4]
    t0 = time.time()
    ph = {r["FILE_ID"]: r for r in csv.DictReader(open(php))}
    F = {}
    for f in sorted(glob.glob(fglob)):
        with np.load(f) as d:
            arrs = {k: d[k] for k in d.files}
        for i, s in enumerate(arrs["sids"]):
            F[str(s)] = {k: np.array(v[i]) for k, v in arrs.items() if k != "sids"}
    use = sorted(s for s in F if s in ph and ph[s]["DX_GROUP"] in ("1", "2") and F[s]["n_empty"] == 0)
    print(f"subjects {len(use)}", flush=True)
    arms = {k: [] for k in ["A8", "A1", "B", "C", "D0", "D1"]}
    for v in ["nogsr", "gsr"]:
        fc = np.stack([F[s][f"fc_{v}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(ae.per_subject_g, list(fc), chunksize=8)
        Kf = ev.correlation_kernel(fc)
        arms["B"].append(("k", Kf))
        arms["C"].append(("k", ev.correlation_kernel(np.stack([r[0] for r in res]))))
        for gi in range(len(ae.GS)):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            arms["D0"].append(("k", (1 - ev.W_FULL) * Kp + ev.W_FULL * Kf))
            arms["D1"].append(("k", (1 - ev.W_FULL) * ev.spectral_filter(Kp) + ev.W_FULL * Kf))
        for t in ["8", "1"]:
            E = np.stack([F[s][f"e{t}_{v}"] for s in use]).astype(np.float64)
            arms[f"A{t}"] += [("k", ev.correlation_kernel(E)), ("zlin", E)]
        print(f"  kernels {v} {time.time() - t0:.0f}s", flush=True)
    arms["U"] = arms["D0"] + arms["C"]   # addendum K
    sites = [ph[s]["SITE_ID"] for s in use]
    dx = np.array([1.0 if ph[s]["DX_GROUP"] == "1" else -1.0 for s in use])
    fiq = np.array([float(ph[s]["FIQ"]) if ph[s]["FIQ"] not in ("", "-9999", "-9999.0") else np.nan for s in use])
    fi = np.nonzero(np.isfinite(fiq) & (fiq > 0))[0]
    G["arms"] = arms
    G["dx"] = {"y": dx, "idx": np.arange(len(use)), "sites": sites}
    G["fiq"] = {"y": fiq[fi], "idx": fi, "sites": [sites[i] for i in fi]}
    jobs = [(s, t) for t in ["dx", "fiq"] for s in range(REPS)]
    with Pool(8) as pool:
        out = pool.map(run_rep, jobs)
    R = {"addendum": "J (ABIDE-I, site-grouped)", "n_dx": len(use), "n_fiq": int(len(fi)),
         "n_sites": len(set(sites)), "seconds": None, "tasks": {}}
    for t in ["dx", "fiq"]:
        ps = {a: np.array([o[2][a] for o in out if o[1] == t]) for a in arms}

        def paired(a, b):
            x = a - b
            return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                    "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std() > 0 else None}
        R["tasks"][t] = {"metric": "AUC" if t == "dx" else "r",
                         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
                         "D0_vs_A8": paired(ps["D0"], ps["A8"]), "D1_vs_A8": paired(ps["D1"], ps["A8"]),
                         "D0_vs_C": paired(ps["D0"], ps["C"]), "D0_vs_A1": paired(ps["D0"], ps["A1"]),
                         "per_rep": {a: v.round(6).tolist() for a, v in ps.items()}}
    R["PRIMARY_dx_D0_vs_A8"] = R["tasks"]["dx"]["D0_vs_A8"]
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)
    for t in ["dx", "fiq"]:
        print(t, {a: round(v["mean"], 4) for a, v in R["tasks"][t]["arms"].items()})
        print("  D0-A8", R["tasks"][t]["D0_vs_A8"], " D1-A8", R["tasks"][t]["D1_vs_A8"])


if __name__ == "__main__":
    main()
