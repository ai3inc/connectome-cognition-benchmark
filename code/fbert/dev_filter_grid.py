"""DEVELOPMENT on the 753 (benchmark protocol): the kernel filter's settings were
tuned on the benchmark's 400-region features; re-choose them for our 450-region
extraction inside the folds. g fixed at 0.90; tau x keep grid; six-modality mean
(equal weights). Each candidate nested lambda, plus in-fold selection.

  build: OMP_NUM_THREADS=8 python3 dev_filter_grid.py build WS K.npz REST_GLOB TASK_GLOB
  eval:  OMP_NUM_THREADS=1 python3 dev_filter_grid.py eval  WS K.npz OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

G_FIX = 0.90
TAUS = [0.25, 0.5, 1.0]
KEEPS = [100, 150, 250, 400]


def build(ws, kp, rest_pat, task_pat):
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs"})
    task = sx.load_feats(task_pat, {f"fc_{m}": f"fc_{m}" for m in sx.MODS[1:]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))
           and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    ev.G_GRID = [G_FIX]
    KP, KF = [], []
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        res = [ev.per_subject(v) for v in fc]
        KP.append(ev.correlation_kernel(np.stack([r[1][0] for r in res])))
        KF.append(ev.correlation_kernel(fc))
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(kp, KP=np.array(KP), KF=np.array(KF), y=y, use=np.array(use))
    print("SAVED", flush=True)


G = {}


def run_seed(nm):
    y, folds_of, C = G["y"], G["folds"][nm], G["C"]
    n = len(y)
    keys = list(C)
    pred = {k: np.full(n, np.nan) for k in keys + ["SEL"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        p, kb, perk = ev.krr_select([C[k] for k in keys], tr, te, y)
        pred["SEL"][te] = p
        for i, k in enumerate(keys):
            pred[k][te] = perk[i]
    return nm, {k: float(np.corrcoef(v, y)[0, 1]) for k, v in pred.items()}


def evaluate(ws, kp, outp):
    t0 = time.time()
    d = np.load(kp)
    KP, KF, y, use = d["KP"], d["KF"], d["y"], list(d["use"])
    C = {}
    for tau in TAUS:
        for keep in KEEPS:
            C[f"tau{tau}_keep{keep}"] = sum((1 - ev.W_FULL) * ev.spectral_filter(KP[m], tau, keep)
                                            + ev.W_FULL * KF[m] for m in range(6)) / 6
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, C=C)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}
    R = {"stage": "development on the 753, benchmark protocol", "g": G_FIX,
         "arms": {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in ps.items()},
         "per_seed": {k: v.round(6).tolist() for k, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for k in sorted(R["arms"], key=lambda k: -R["arms"][k]["mean"]):
        print(f"{k:18s} {R['arms'][k]['mean']:.4f} +- {R['arms'][k]['sd']:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    else:
        evaluate(*sys.argv[2:5])
