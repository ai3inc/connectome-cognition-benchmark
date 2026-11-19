"""Addendum L (amended) evaluation: ADHD-200, leave-one-site-out over 6 sites;
ADHD (any) vs typically developing — AUC of the KRR score on +/-1; full4_iq — r.
Arms as ABIDE (variant noGSR/GSR and knobs chosen in-fold). Decision: paired
bootstrap over subjects of the out-of-fold metric difference.

usage: python3 adhd_eval.py PHENO.json FEAT_GLOB OUT.json
"""
import glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import aomic_eval as ae
from abide_eval import auc

G = {}


def run_arm(args):
    task, a = args
    with threadpool_limits(1):
        y, idx, sites = G[task]["y"], G[task]["idx"], G[task]["sites"]
        spec = G["arms"][a]
        n = len(y)
        pred = np.full(n, np.nan)
        for s in sorted(set(sites)):
            te = np.array([i for i in range(n) if sites[i] == s]); tr = np.setdiff1d(np.arange(n), te)
            Ks = []
            for kind, E in spec:
                Esub = E[np.ix_(idx, idx)] if kind == "k" else E[idx]
                Ks.append(ev.zlin_kernel(Esub, tr) if kind == "zlin" else Esub)
            pred[te] = ev.krr_select(Ks, tr, te, y)[0]
        return task, a, pred


def boot(pa, pb, y, metric, B=10000, seed=1):
    rs = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(B):
        j = rs.randint(0, n, n)
        if metric == "auc" and (len(set(y[j])) < 2):
            continue
        f = (lambda p: auc(p[j], y[j])) if metric == "auc" else (lambda p: np.corrcoef(p[j], y[j])[0, 1])
        d.append(f(pa) - f(pb))
    d = np.array(d)
    full = (auc(pa, y) - auc(pb, y)) if metric == "auc" else (np.corrcoef(pa, y)[0, 1] - np.corrcoef(pb, y)[0, 1])
    return {"diff": float(full), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0": float((d > 0).mean())}


def main():
    php, fglob, outp = sys.argv[1:4]
    t0 = time.time()
    ph = json.load(open(php))
    F = {}
    for f in sorted(glob.glob(fglob)):
        with np.load(f) as d:
            arrs = {k: d[k] for k in d.files}
        for i, s in enumerate(arrs["sids"]):
            F[str(s)] = {k: np.array(v[i]) for k, v in arrs.items() if k != "sids"}
    use = sorted(s for s in F if s in ph and F[s]["n_empty"] == 0)
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
    sites = [ph[s]["site"] for s in use]
    dx = np.array([-1.0 if "Typically" in ph[s]["dx"] else 1.0 for s in use])

    def iqv(x):
        try:
            v = float(x); return v if 40 < v < 200 else np.nan
        except Exception:
            return np.nan
    iq = np.array([iqv(ph[s]["iq"]) for s in use])
    fi = np.nonzero(np.isfinite(iq))[0]
    G["arms"] = arms
    G["dx"] = {"y": dx, "idx": np.arange(len(use)), "sites": sites}
    G["iq"] = {"y": iq[fi], "idx": fi, "sites": [sites[i] for i in fi]}
    jobs = [(t, a) for t in ["dx", "iq"] for a in arms]
    with Pool(8) as pool:
        out = pool.map(run_arm, jobs)
    P = {(t, a): p for t, a, p in out}
    R = {"addendum": "L (ADHD-200, leave-one-site-out)", "n_dx": len(use), "n_iq": int(len(fi)),
         "sites": sorted(set(sites)), "n_adhd": int((dx > 0).sum()), "tasks": {}}
    for t, metric in [("dx", "auc"), ("iq", "r")]:
        y = G[t]["y"]
        sc = {a: (auc(P[(t, a)], y) if metric == "auc" else float(np.corrcoef(P[(t, a)], y)[0, 1])) for a in arms}
        R["tasks"][t] = {"metric": metric, "arms": sc,
                         "D0_vs_A8": boot(P[(t, "D0")], P[(t, "A8")], y, metric),
                         "D1_vs_A8": boot(P[(t, "D1")], P[(t, "A8")], y, metric),
                         "D0_vs_C": boot(P[(t, "D0")], P[(t, "C")], y, metric)}
    R["PRIMARY_dx_D0_vs_A8"] = R["tasks"]["dx"]["D0_vs_A8"]
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps(R["tasks"], indent=1))


if __name__ == "__main__":
    main()
