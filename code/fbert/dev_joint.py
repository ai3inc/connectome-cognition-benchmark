"""DEVELOPMENT on the 753 (benchmark protocol): canonical chain (benchmark's 400-region
features, code/canonical.py) combined with our D1 (450-region re-extraction).

  CANON      canonical six-modality kernel, nested lambda
  D1         ours, (g, weighting) in-fold over 6
  JOINT50    0.5 CANON + 0.5 D1(g, w), (g, w) in-fold      <- no new tuning knob
  JOINTsel   mu in {0.25, 0.5, 0.75} x D1(g, w), in-fold
  PRED50     mean of the CANON and D1 predictions           <- no tuning at all

usage: python3 dev_joint.py WS CANON_K.npz D1H_K.npz OUT.json
"""
import json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

GS = [0.85, 0.9, 0.95]
WTS = {"eq": [1, 1, 1, 1, 1, 1], "lang2": [1, 2, 1, 1, 1, 1]}
MUS = [0.25, 0.5, 0.75]
G = {}


def run_seed(nm):
    y, folds_of, KC, D1 = G["y"], G["folds"][nm], G["KC"], G["D1"]
    n = len(y)
    keys = list(D1)
    pred = {a: np.full(n, np.nan) for a in ["CANON", "D1", "JOINT50", "JOINTsel", "PRED50"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        pc, _, _ = ev.krr_select([KC], tr, te, y)
        pd, _, _ = ev.krr_select([D1[k] for k in keys], tr, te, y)
        pj, _, _ = ev.krr_select([0.5 * KC + 0.5 * D1[k] for k in keys], tr, te, y)
        ps, _, _ = ev.krr_select([(1 - mu) * KC + mu * D1[k] for mu in MUS for k in keys], tr, te, y)
        pred["CANON"][te] = pc; pred["D1"][te] = pd; pred["JOINT50"][te] = pj
        pred["JOINTsel"][te] = ps; pred["PRED50"][te] = 0.5 * (pc + pd)
    return nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}


def main():
    ws, ckp, dkp, outp = sys.argv[1:5]
    t0 = time.time()
    c = np.load(ckp); d = np.load(dkp)
    cu = [str(s) for s in c["use"]]; du = [str(s) for s in d["use"]]
    common = [s for s in cu if s in set(du)]
    ci = np.array([cu.index(s) for s in common]); di = np.array([du.index(s) for s in common])
    KC = c["K"][np.ix_(ci, ci)]
    KM = d["KM"][:, :, di][:, :, :, di]
    y = d["y"][di]
    D1 = {}
    for wn, w in WTS.items():
        for gi, g in enumerate(GS):
            D1[f"{wn}_g{g}"] = sum(w[m] * KM[m][gi] for m in range(6)) / sum(w)
    sidx = {s: i for i, s in enumerate(common)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, KC=KC, D1=D1)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {a: np.array([o[1][a] for o in out]) for a in out[0][1]}

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    R = {"stage": "development on the 753, benchmark protocol", "n": len(common),
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "vs_CANON": {a: paired(ps[a], ps["CANON"]) for a in ps if a != "CANON"},
         "per_seed": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in R["arms"].items():
        cmp = R["vs_CANON"].get(a)
        print(f"{a:9s} {v['mean']:.4f} +- {v['sd']:.4f}",
              "" if cmp is None else f"vs CANON {cmp['mean_diff']:+.4f} {cmp['positive']}/60 t={cmp['t']:.1f}")


if __name__ == "__main__":
    main()
