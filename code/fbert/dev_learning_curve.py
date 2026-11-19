"""DIAGNOSTIC on the 753: does the strictly per-subject D0 gain more from training
size than F-BERT? Each training fold is subsampled to a fraction f (same subset for
both arms, fixed seed); the test fold is unchanged. D0 = six-modality mean at
g = 0.97 (the value the in-fold selector prefers); F-BERT = concatenated z-scored
linear probe (in-fold z on the subsample).

usage: python3 dev_learning_curve.py WS PERMOD.npz FEAT_GLOB TASKFEAT_GLOB OUT.json
"""
import json, os, sys
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

FRACS = [0.5, 0.75, 1.0]
G = {}


def run_seed(nm):
    y, folds_of, D0, E = G["y"], G["folds"][nm], G["D0"], G["E"]
    n = len(y)
    rs = np.random.RandomState(abs(hash(nm)) % 2**31)
    out = {}
    for f in FRACS:
        pd, pf = np.full(n, np.nan), np.full(n, np.nan)
        for te in folds_of:
            tr_all = np.setdiff1d(np.arange(n), te)
            tr = np.sort(rs.choice(tr_all, int(round(f * len(tr_all))), replace=False)) if f < 1 else tr_all
            p, _, _ = ev.krr_select([D0], tr, te, y); pd[te] = p
            p, _, _ = ev.krr_select([ev.zlin_kernel(E, tr)], tr, te, y); pf[te] = p
        out[f] = (float(np.corrcoef(pd, y)[0, 1]), float(np.corrcoef(pf, y)[0, 1]))
    return nm, out


def main():
    ws, kp, fp, tp, outp = sys.argv[1:6]
    d = np.load(kp)
    use = [str(s) for s in d["use"]]
    D0 = d["KD"][:, 1].mean(0)          # g = 0.97 (gs = [0.95, 0.97])
    rest = sx.load_feats(fp, {"e8_full": "e8_rs"})
    task = sx.load_feats(tp, {f"e8_{m}": f"e8_{m}" for m in sx.MODS[1:]})
    E = np.concatenate([np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64)
                        for m in sx.MODS], 1)
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "20"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=d["y"], folds=fl, D0=D0, E=E)
    with Pool(8) as pool:
        res = pool.map(run_seed, names)
    R = {"stage": "diagnostic on the 753", "fractions": FRACS, "seeds": len(names), "curve": {}}
    for f in FRACS:
        dd = np.array([o[1][f][0] for o in res]); ff = np.array([o[1][f][1] for o in res])
        x = dd - ff
        R["curve"][str(f)] = {"D0": float(dd.mean()), "FBERT": float(ff.mean()), "diff": float(x.mean()),
                              "positive": int((x > 0).sum()), "approx_n_train": int(round(f * 677))}
        print(f, R["curve"][str(f)], flush=True)
    json.dump(R, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
