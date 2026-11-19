"""Addendum M evaluation, within NYU (random 10-fold x 20), ADHD vs TD AUC:
M1 DVARS alone (linear kernel on z-scored DVARS); M2 F-BERT-8M and D0 with DVARS
regressed out of the out-of-fold score inside each training fold; M3 closed-form
temporal kernels (ac1, xc1, dfc_sd) alone and half-half with D0.

usage: python3 adhd_motion_eval.py PHENO.json FEAT_GLOB TEMP_GLOB OUT.json
"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import aomic_eval as ae
from abide_eval import auc

G = {}


def load(pat):
    D = {}
    for f in sorted(glob.glob(pat)):
        with np.load(f) as d:
            a = {k: d[k] for k in d.files}
        for i, s in enumerate(a["sids"]):
            D[str(s)] = {k: np.array(v[i]) for k, v in a.items() if k != "sids"}
    return D


def rep(s):
    with threadpool_limits(1):
        y, arms, dv = G["y"], G["arms"], G["dv"]
        n = len(y)
        perm = np.random.RandomState(500 + s).permutation(n)
        pred = {a: np.full(n, np.nan) for a in arms}
        resid = {a: np.full(n, np.nan) for a in ["A8", "D0"]}
        for te in np.array_split(perm, 10):
            tr = np.setdiff1d(np.arange(n), te)
            for a, spec in arms.items():
                Ks = [ev.zlin_kernel(E, tr) if k == "zlin" else E for k, E in spec]
                p, _, _ = ev.krr_select(Ks, tr, te, y)
                pred[a][te] = p
                if a in resid:   # in-fold: regress DVARS out of the training-fold scores, apply to test
                    ptr, _, _ = ev.krr_select(Ks, tr, tr, y)
                    X = np.c_[np.ones(len(tr)), dv[tr]]
                    b = np.linalg.lstsq(X, ptr, rcond=None)[0]
                    resid[a][te] = p - (b[0] + b[1] * dv[te])
        out = {a: auc(v, y) for a, v in pred.items()}
        out.update({f"{a}_dvars_resid": auc(v, y) for a, v in resid.items()})
        out["corr_A8score_dvars"] = float(np.corrcoef(pred["A8"], dv)[0, 1])
        out["corr_D0score_dvars"] = float(np.corrcoef(pred["D0"], dv)[0, 1])
        return s, out


def main():
    php, fglob, tglob, outp = sys.argv[1:5]
    ph = json.load(open(php))
    F, T = load(fglob), load(tglob)
    use = sorted(s for s in F if s in T and s in ph and ph[s]["site"] == "NYU")
    y = np.array([-1.0 if "Typically" in ph[s]["dx"] else 1.0 for s in use])
    dv = np.array([float(T[s]["dvars"]) for s in use])
    arms = {"A8": [], "D0": [], "DV": [("zlin", ((dv - dv.mean()) / dv.std())[:, None])]}
    D0K = []
    for v in ["nogsr", "gsr"]:
        fc = np.stack([F[s][f"fc_{v}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(ae.per_subject_g, list(fc), chunksize=8)
        Kf = ev.correlation_kernel(fc)
        for gi in range(len(ae.GS)):
            K = 0.95 * ev.correlation_kernel(np.stack([r[1][gi] for r in res])) + 0.05 * Kf
            arms["D0"].append(("k", K)); D0K.append(K)
        E = np.stack([F[s][f"e8_{v}"] for s in use]).astype(np.float64)
        arms["A8"] += [("k", ev.correlation_kernel(E)), ("zlin", E)]
    for name in ["ac1", "xc1", "dfc_sd"]:
        X = np.stack([T[s][name] for s in use]).astype(np.float64)
        K = ev.correlation_kernel(X)
        arms[name.upper()] = [("k", K)]
        arms[name.upper() + "+D0"] = [("k", 0.5 * K + 0.5 * Kd) for Kd in D0K]
    G.update(y=y, arms=arms, dv=dv)
    with Pool(8) as pool:
        out = pool.map(rep, range(20))
    keys = out[0][1].keys()
    R = {"addendum": "M (diagnostic, NYU only)", "n": len(use), "n_adhd": int((y > 0).sum()),
         "dvars_adhd_minus_td": float(dv[y > 0].mean() - dv[y < 0].mean()),
         "mean": {k: float(np.mean([o[1][k] for o in out])) for k in keys},
         "per_rep": {k: [o[1][k] for o in out] for k in keys}}
    m = R["mean"]
    margin = m["A8"] - m["D0"]
    R["reading"] = {"dvars_alone_ge_0.60": m["DV"] >= 0.60,
                    "fbert_margin_raw": margin,
                    "fbert_margin_after_dvars": m["A8_dvars_resid"] - m["D0_dvars_resid"],
                    "half_of_margin_removed": (m["A8_dvars_resid"] - m["D0_dvars_resid"]) <= margin / 2,
                    "best_temporal": max(["AC1", "XC1", "DFC_SD", "AC1+D0", "XC1+D0", "DFC_SD+D0"], key=lambda k: m[k])}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: R[k] for k in ["n", "n_adhd", "dvars_adhd_minus_td", "mean", "reading"]}, indent=1))


if __name__ == "__main__":
    main()
