"""DEVELOPMENT on the 753 (free exploration; the confirmation is on subjects outside
the 753, preregistered separately). Hybrid of our per-subject conditioned ensemble
D0-6(g) with F-BERT-8M six-modality kernels.

  H1(rho, g) = (1 - rho) * D0-6(g) + rho * mean of F-BERT correlation kernels
  H2(rho, g) = (1 - rho) * D0-6(g) + rho * F-BERT concatenated z-scored linear kernel (in-fold z)
each with its own nested lambda; H1sel / H2sel / HALL = in-fold selection over the set.

usage: python3 dev_hybrid.py WORKSPACE DECOR_KERNELS.npz OUT.json
"""
import json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

RHOS = [0.25, 0.5, 0.75]
G = {}


def run_seed(nm):
    y, folds_of, D0, A86, E = G["y"], G["folds"][nm], G["D0"], G["A86"], G["E"]
    n = len(y)
    cand_names = ([f"H1_r{r}_g{g}" for r in RHOS for g in D0] + [f"H2_r{r}_g{g}" for r in RHOS for g in D0])
    pred = {k: np.full(n, np.nan) for k in cand_names + ["H1sel", "H2sel", "HALL", "D0sel", "A8cat"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        Z = ev.zlin_kernel(E, tr)
        Z = Z / np.trace(Z[np.ix_(tr, tr)]) * len(tr) / n * 1.0   # scale like a trace-1 kernel
        K1 = [(1 - r) * D0[g] + r * A86 for r in RHOS for g in D0]
        K2 = [(1 - r) * D0[g] + r * Z for r in RHOS for g in D0]
        allK = K1 + K2
        p, kb, perk = ev.krr_select(allK, tr, te, y)
        pred["HALL"][te] = p
        for i, k in enumerate(cand_names):
            pred[k][te] = perk[i]
        p, _, _ = ev.krr_select(K1, tr, te, y); pred["H1sel"][te] = p
        p, _, _ = ev.krr_select(K2, tr, te, y); pred["H2sel"][te] = p
        p, _, _ = ev.krr_select(list(D0.values()), tr, te, y); pred["D0sel"][te] = p
        p, _, _ = ev.krr_select([Z], tr, te, y); pred["A8cat"][te] = p
    return nm, {k: float(np.corrcoef(v, y)[0, 1]) for k, v in pred.items()}


def main():
    ws, kp, outp = sys.argv[1:4]
    t0 = time.time()
    d = np.load(kp)
    keys = list(d["keys"])
    D0 = {k.split("_")[0][1:]: d["Ks"][i] for i, k in enumerate(keys) if k.endswith("_b0.0")}
    E = d["E"]                       # (6, n, 384)
    y, use = d["y"], list(d["use"])
    A86 = sum(ev.correlation_kernel(E[m]) for m in range(6)) / 6
    Ecat = np.concatenate(list(E), 1)
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, D0=D0, A86=A86, E=Ecat)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    R = {"stage": "development on the 753 (not a confirmation)",
         "arms": {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in ps.items()},
         "vs_A8cat": {k: paired(ps[k], ps["A8cat"]) for k in ps if k != "A8cat"},
         "per_seed": {k: v.round(6).tolist() for k, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for k in sorted(R["arms"], key=lambda k: -R["arms"][k]["mean"]):
        v = R["vs_A8cat"].get(k)
        print(f"{k:16s} {R['arms'][k]['mean']:.4f}", "" if v is None else f"{v['mean_diff']:+.4f} {v['positive']}/60")


if __name__ == "__main__":
    main()
