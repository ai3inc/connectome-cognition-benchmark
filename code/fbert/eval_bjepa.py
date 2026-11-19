"""Addendum I evaluation: Brain-JEPA frozen embeddings (rest) on the 753, 60 x 10,
probes P1 (in-fold z-scored linear) and P2 (correlation kernel), best kept; paired
with the main comparison's saved per-seed D0 (rest) and F-BERT-8M (rest).

usage: python3 eval_bjepa.py WS EMB_GLOB SAMEFOLDS.json OUT.json
"""
import csv, glob, json, os, sys
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev

G = {}


def run_seed(nm):
    with threadpool_limits(1):
        y, folds_of, E, Kc = G["y"], G["folds"][nm], G["E"], G["Kc"]
        n = len(y)
        p1, p2 = np.full(n, np.nan), np.full(n, np.nan)
        for te in folds_of:
            tr = np.setdiff1d(np.arange(n), te)
            p1[te] = ev.krr_select([ev.zlin_kernel(E, tr)], tr, te, y)[0]
            p2[te] = ev.krr_select([Kc], tr, te, y)[0]
        return nm, float(np.corrcoef(p1, y)[0, 1]), float(np.corrcoef(p2, y)[0, 1])


def main():
    ws, eglob, sfp, outp = sys.argv[1:5]
    emb = {}
    for f in sorted(glob.glob(eglob)):
        d = np.load(f)
        for s, e in zip(d["sids"], d["emb"]):
            emb[str(s)] = e
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan")) and s in emb]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    E = np.stack([emb[s] for s in use]).astype(np.float64)
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:60]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, E=E, Kc=ev.correlation_kernel(E))
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    r1 = np.array([o[1] for o in out]); r2 = np.array([o[2] for o in out])
    bj, probe = (r1, "P1") if r1.mean() >= r2.mean() else (r2, "P2")
    sf = json.load(open(sfp))
    assert sf["n_subjects"] == len(use) == 753
    d0 = np.array(sf["per_seed"]["D0_full"]); a8 = np.array(sf["per_seed"]["A8P1_full"])

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    R = {"addendum": "I (Brain-JEPA)", "n": len(use), "probe_used": probe,
         "BrainJEPA_P1": float(r1.mean()), "BrainJEPA_P2": float(r2.mean()), "BrainJEPA": float(bj.mean()),
         "D0_rest": float(d0.mean()), "FBERT8M_rest": float(a8.mean()),
         "PRIMARY_D0_vs_BrainJEPA": paired(d0, bj), "FBERT_vs_BrainJEPA": paired(a8, bj),
         "per_seed_BrainJEPA": bj.round(6).tolist()}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "per_seed_BrainJEPA"}, indent=1))


if __name__ == "__main__":
    main()
