"""Addendum R: six-modality ensembles with and without the motor task, our 450-region chain,
753, 60 x 10 family-aware folds, g (and weighting) chosen in-fold.

usage: python3 motor_drop.py WS REST_GLOB TASK_GLOB OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx
from sota_hcp import per_subject_grid

GS = [0.85, 0.9, 0.95]
G = {}


def run(nm):
    with threadpool_limits(1):
        y, fl, arms = G["y"], G["folds"][nm], G["arms"]
        n = len(y)
        pred = {a: np.full(n, np.nan) for a in arms}
        for te in fl:
            tr = np.setdiff1d(np.arange(n), te)
            for a, Ks in arms.items():
                pred[a][te] = ev.krr_select(Ks, tr, te, y)[0]
        return nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}


def main():
    ws, rglob, tglob, outp = sys.argv[1:5]
    t0 = time.time()
    rest = sx.load_feats(rglob, {"fc_full": "fc_rs"})
    task = sx.load_feats(tglob, {f"fc_{m}": f"fc_{m}" for m in sx.MODS[1:]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan")) and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:60]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    D0, D1 = {}, {}
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(per_subject_grid, [(v, GS) for v in fc], chunksize=8)
        Kf = ev.correlation_kernel(fc)
        for gi, g in enumerate(GS):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            D0[(m, g)] = 0.95 * Kp + 0.05 * Kf
            D1[(m, g)] = 0.95 * ev.spectral_filter(Kp) + 0.05 * Kf
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    six = sx.MODS; five = [m for m in sx.MODS if m != "motor"]

    def ens(KD, mods, lang2):
        w = {m: (2 if (lang2 and m == "lang") else 1) for m in mods}
        return [sum(w[m] * KD[(m, g)] for m in mods) / sum(w.values()) for g in GS]
    arms = {}
    for tag, KD in [("D1", D1), ("D0", D0)]:
        arms[f"{tag}_six"] = ens(KD, six, False) + ens(KD, six, True)
        arms[f"{tag}_nomotor"] = ens(KD, five, False) + ens(KD, five, True)
        if os.environ.get("DIAG"):   # post-hoc diagnostic: weighting fixed (g in-fold), and g=0.9 equal fixed
            for nm_, mods in [("six", six), ("nomotor", five)]:
                arms[f"{tag}_{nm_}_eq"] = ens(KD, mods, False)
                arms[f"{tag}_{nm_}_l2"] = ens(KD, mods, True)
                arms[f"{tag}_{nm_}_eq_g090"] = [ens(KD, mods, False)[1]]
    G.update(y=y, folds=fl, arms=arms)
    with Pool(8) as pool:
        out = pool.map(run, names)
    ps = {a: np.array([o[1][a] for o in out]) for a in arms}

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    R = {"addendum": "R (drop motor)", "n": len(use),
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "PRIMARY_D1_nomotor_vs_six": paired(ps["D1_nomotor"], ps["D1_six"]),
         "D0_nomotor_vs_six": paired(ps["D0_nomotor"], ps["D0_six"]),
         "diag": ({f"{t}_{w}_nomotor_vs_six": paired(ps[f"{t}_nomotor_{w}"], ps[f"{t}_six_{w}"])
                   for t in ["D1", "D0"] for w in ["eq", "l2", "eq_g090"]} if os.environ.get("DIAG") else None),
         "per_seed": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
