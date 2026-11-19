"""DIAGNOSTIC (753, 60 x 10): is the strictly per-subject D0's 0.007 loss to F-BERT a
lambda-grid-edge artefact? Both arms re-run with the default grid logspace(-6, 2, 17)
and an extended grid logspace(-10, 2, 25); the lambda each fold picks is recorded.

usage: python3 dev_lambda_edge.py WS DECOR_KERNELS.npz OUT.json
"""
import json, os, sys
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

GRIDS = {"default": np.logspace(-6, 2, 17), "extended": np.logspace(-10, 2, 25)}
G = {}


def run_seed(nm):
    y, folds_of, D0, E = G["y"], G["folds"][nm], G["D0"], G["E"]
    n = len(y)
    out, picks = {}, {}
    for gname, grid in GRIDS.items():
        ev.LAMBDAS = grid
        pd, pf = np.full(n, np.nan), np.full(n, np.nan)
        lam_d, lam_f = [], []
        for te in folds_of:
            tr = np.setdiff1d(np.arange(n), te)
            p, kb, perk = ev.krr_select(D0, tr, te, y); pd[te] = p
            # recover the chosen lambda for D0 by re-running the inner scoring
            S = np.zeros((len(D0), len(grid)))
            for a, b in ev.inner_splits(len(tr)):
                ya = y[tr][a] - y[tr][a].mean()
                for ki, K in enumerate(D0):
                    w, V = np.linalg.eigh(K[np.ix_(tr[a], tr[a])])
                    VtK = K[np.ix_(tr[b], tr[a])] @ V; Vt = V.T @ ya
                    for li, lam in enumerate(grid):
                        S[ki, li] += np.corrcoef(VtK @ (Vt / (w + lam)), y[tr][b])[0, 1]
            lam_d.append(float(grid[np.unravel_index(np.nanargmax(S), S.shape)[1]]))
            Z = ev.zlin_kernel(E, tr)
            p, _, _ = ev.krr_select([Z], tr, te, y); pf[te] = p
        out[gname] = (float(np.corrcoef(pd, y)[0, 1]), float(np.corrcoef(pf, y)[0, 1]))
        picks[gname] = lam_d
    return nm, out, picks


def main():
    ws, kp, outp = sys.argv[1:4]
    d = np.load(kp)
    keys = list(d["keys"])
    D0 = [d["Ks"][i] for i, k in enumerate(keys) if k.endswith("_b0.0")]
    E = np.concatenate(list(d["E"]), 1)
    use = [str(s) for s in d["use"]]
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=d["y"], folds=fl, D0=D0, E=E)
    with Pool(8) as pool:
        res = pool.map(run_seed, names)
    R = {"stage": "diagnostic on the 753", "grids": {k: v.tolist() for k, v in GRIDS.items()}, "results": {}}
    for gname, grid in GRIDS.items():
        dd = np.array([o[1][gname][0] for o in res]); ff = np.array([o[1][gname][1] for o in res])
        x = dd - ff
        picks = sum((o[2][gname] for o in res), [])
        R["results"][gname] = {"D0": float(dd.mean()), "FBERT": float(ff.mean()), "diff": float(x.mean()),
                               "positive": int((x > 0).sum()),
                               "D0_lambda_at_lower_edge": int(sum(p == grid[0] for p in picks)), "folds": len(picks),
                               "D0_lambda_picks": {f"{l:.0e}": picks.count(l) for l in sorted(set(picks))}}
        print(gname, R["results"][gname], flush=True)
    json.dump(R, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
