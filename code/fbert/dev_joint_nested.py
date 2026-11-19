"""Fully nested joint on the 753 (benchmark protocol): the canonical chain with ALL of
its hyperparameters chosen in-fold over the main session's 24-cell nested grid, our
D1 with (g, weighting) chosen in-fold, and their prediction mean (no tuning).
This removes the fixed-setting optimism from the 0.7483 joint number.

usage: python3 dev_joint_nested.py WS CANON_GRID_K.npz D1H_K.npz OUT.json
"""
import json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

GS = [0.85, 0.9, 0.95]
WTS = {"eq": [1, 1, 1, 1, 1, 1], "lang2": [1, 2, 1, 1, 1, 1]}
G = {}


def run_seed(nm):
    y, folds_of, KC, D1 = G["y"], G["folds"][nm], G["KC"], G["D1"]
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in ["CANONnest", "D1", "PRED50nest"]}
    picks = []
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        pc, kb, _ = ev.krr_select(list(KC), tr, te, y)
        pd, _, _ = ev.krr_select(list(D1), tr, te, y)
        pred["CANONnest"][te] = pc; pred["D1"][te] = pd; pred["PRED50nest"][te] = 0.5 * (pc + pd)
        picks.append(int(kb))
    return nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}, picks


def main():
    ws, ckp, dkp, outp = sys.argv[1:5]
    t0 = time.time()
    c = np.load(ckp); d = np.load(dkp)
    cu = [str(s) for s in c["use"]]; du = [str(s) for s in d["use"]]
    common = [s for s in cu if s in set(du)]
    ci = np.array([cu.index(s) for s in common]); di = np.array([du.index(s) for s in common])
    KC = c["Ks"][:, ci][:, :, ci]
    keys = list(c["keys"])
    KM = d["KM"][:, :, di][:, :, :, di]
    y = d["y"][di]
    D1 = [sum(w[m] * KM[m][gi] for m in range(6)) / sum(w) for w in WTS.values() for gi in range(len(GS))]
    sidx = {s: i for i, s in enumerate(common)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, KC=KC, D1=D1)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {a: np.array([o[1][a] for o in out]) for a in out[0][1]}
    picks = sum((o[2] for o in out), [])

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    R = {"stage": "fully nested joint, 753, benchmark protocol", "n": len(common),
         "canon_grid": keys, "canon_picks": {keys[i]: picks.count(i) for i in sorted(set(picks))},
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "PRED50nest_vs_CANONnest": paired(ps["PRED50nest"], ps["CANONnest"]),
         "PRED50nest_vs_D1": paired(ps["PRED50nest"], ps["D1"]),
         "per_seed": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("per_seed", "canon_grid")}, indent=1))


if __name__ == "__main__":
    main()
