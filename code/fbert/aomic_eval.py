"""Addendum H evaluation: F-BERT vs our chain on AOMIC-ID1000, the project's AOMIC
splits (20 x 10-fold, RandomState(7000+s) over participants sorted by ID), target
IST_intelligence_total, lambda by inner 4-fold mean-r. Each arm's cleaning variant
{noGSR, GSR} is chosen in-fold together with its other knobs.

usage: python3 aomic_eval.py PARTICIPANTS.tsv FEAT_GLOB OUT.json
"""
import csv, glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

GS = [0.80, 0.85, 0.90, 0.95]
REPS = 20
G = {}


def per_subject_g(v):
    with threadpool_limits(1):
        C = ev.to_mat(v.astype(np.float64))
        w, V = np.linalg.eigh(C)
        p035 = ((V * np.clip(w, 0, None) ** 0.35) @ V.T)[ev.IU].astype(np.float32)
        parts = []
        for g in GS:
            P = np.linalg.inv((1 - g) * C + g * np.eye(ev.N_REG))
            d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            P = -P / np.outer(d, d); np.fill_diagonal(P, 1.0)
            wa, Va = np.linalg.eigh(P)
            parts.append(((Va * np.clip(wa, 0, None) ** ev.ALPHA) @ Va.T)[ev.IU].astype(np.float32))
        return p035, parts


def run_rep(s):
    with threadpool_limits(1):
        y, arms = G["y"], G["arms"]
        n = len(y)
        perm = np.random.RandomState(7000 + s).permutation(n)
        pred = {a: np.full(n, np.nan) for a in arms}
        for te in np.array_split(perm, 10):
            tr = np.setdiff1d(np.arange(n), te)
            for a, spec in arms.items():
                Ks = [ev.zlin_kernel(E, tr) if kind == "zlin" else E for kind, E in spec]
                p, _, _ = ev.krr_select(Ks, tr, te, y)
                pred[a][te] = p
        return s, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}


def main():
    pp, fglob, outp = sys.argv[1:4]
    t0 = time.time()
    parts = {r["participant_id"]: r for r in csv.DictReader(open(pp), delimiter="\t")}
    F = {}
    for f in sorted(glob.glob(fglob)):
        with np.load(f) as d:
            arrs = {k: d[k] for k in d.files}
        for i, s in enumerate(arrs["sids"]):
            F[str(s)] = {k: np.array(v[i]) for k, v in arrs.items() if k != "sids"}

    def num(v):
        try:
            x = float(v); return x if x == x else None
        except Exception:
            return None
    use = sorted(s for s in F if s in parts and num(parts[s].get("IST_intelligence_total")) is not None
                 and F[s]["n_empty"] == 0)
    y = np.array([num(parts[s]["IST_intelligence_total"]) for s in use])
    print(f"participants {len(use)} (dropped {len(F) - len(use)})", flush=True)
    arms = {k: [] for k in ["A8", "A1", "B", "C", "D0", "D1"]}
    for v in ["nogsr", "gsr"]:
        fc = np.stack([F[s][f"fc_{v}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(per_subject_g, list(fc), chunksize=8)
        Kf = ev.correlation_kernel(fc)
        arms["B"].append(("k", Kf))
        arms["C"].append(("k", ev.correlation_kernel(np.stack([r[0] for r in res]))))
        for gi in range(len(GS)):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            arms["D0"].append(("k", (1 - ev.W_FULL) * Kp + ev.W_FULL * Kf))
            arms["D1"].append(("k", (1 - ev.W_FULL) * ev.spectral_filter(Kp) + ev.W_FULL * Kf))
        for t in ["8", "1"]:
            E = np.stack([F[s][f"e{t}_{v}"] for s in use]).astype(np.float64)
            arms[f"A{t}"] += [("k", ev.correlation_kernel(E)), ("zlin", E)]
        print(f"  kernels {v} {time.time() - t0:.0f}s", flush=True)
    arms["U"] = arms["D0"] + arms["C"]   # addendum K
    G.update(y=y, arms=arms)
    with Pool(8) as pool:
        out = pool.map(run_rep, range(REPS))
    ps = {a: np.array([o[1][a] for o in out]) for a in arms}

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std() > 0 else None}
    R = {"addendum": "H (AOMIC-ID1000, same folds)", "n": len(use), "target": "IST_intelligence_total",
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "PRIMARY_D0_vs_A8": paired(ps["D0"], ps["A8"]),
         "D1_vs_A8": paired(ps["D1"], ps["A8"]), "D0_vs_A1": paired(ps["D0"], ps["A1"]),
         "D0_vs_C": paired(ps["D0"], ps["C"]), "D0_vs_B": paired(ps["D0"], ps["B"]),
         "per_rep": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in R["arms"].items():
        print(f"{a:3s} {v['mean']:.4f} +- {v['sd']:.4f}")
    for k in ["PRIMARY_D0_vs_A8", "D1_vs_A8", "D0_vs_A1", "D0_vs_C", "D0_vs_B"]:
        print(k, R[k])


if __name__ == "__main__":
    main()
