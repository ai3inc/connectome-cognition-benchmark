"""DEVELOPMENT on the 753 (benchmark protocol): D1 (ours + kernel filter) mixed with
F-BERT-8M's concatenated z-scored linear kernel (in-fold z), rho in {0, .1, .2, .3},
equal or language x2 weighting, g in {0.85, 0.9, 0.95}. Each candidate with nested
lambda; plus in-fold selection over all.

  build: OMP_NUM_THREADS=8 python3 dev_d1_hybrid.py build WS K.npz REST_GLOB TASK_GLOB
  eval:  OMP_NUM_THREADS=1 python3 dev_d1_hybrid.py eval  WS K.npz OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

GS = [0.85, 0.9, 0.95]
RHOS = [0.0, 0.1, 0.2, 0.3]
WTS = {"eq": [1, 1, 1, 1, 1, 1], "lang2": [1, 2, 1, 1, 1, 1]}


def build(ws, kp, rest_pat, task_pat):
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs"})
    task = sx.load_feats(task_pat, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))
           and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    ev.G_GRID = GS
    KM = []   # (6, len(GS), n, n): filtered partial kernel mixed with 5% full, per modality
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        res = [ev.per_subject(v) for v in fc]
        Kf = ev.correlation_kernel(fc)
        KM.append([(1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(np.stack([r[1][gi] for r in res])))
                   + ev.W_FULL * Kf for gi in range(len(GS))])
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    E = np.concatenate([np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64)
                        for m in sx.MODS], 1)
    np.savez(kp, KM=np.array(KM), E=E, y=y, use=np.array(use))
    print("SAVED", flush=True)


G = {}


def run_seed(nm):
    y, folds_of, D1, E = G["y"], G["folds"][nm], G["D1"], G["E"]
    n = len(y)
    keys = list(D1)
    names = [f"{k}_r{r}" for k in keys for r in RHOS]
    pred = {c: np.full(n, np.nan) for c in names + ["SEL_eq", "SEL_all", "A8cat"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        Z = ev.zlin_kernel(E, tr)
        Ks = [(1 - r) * D1[k] + r * Z for k in keys for r in RHOS]
        p, kb, perk = ev.krr_select(Ks, tr, te, y)
        pred["SEL_all"][te] = p
        for i, c in enumerate(names):
            pred[c][te] = perk[i]
        eq_idx = [i for i, c in enumerate(names) if c.startswith("eq_")]
        p, _, _ = ev.krr_select([Ks[i] for i in eq_idx], tr, te, y); pred["SEL_eq"][te] = p
        p, _, _ = ev.krr_select([Z], tr, te, y); pred["A8cat"][te] = p
    return nm, {c: float(np.corrcoef(v, y)[0, 1]) for c, v in pred.items()}


def evaluate(ws, kp, outp):
    t0 = time.time()
    d = np.load(kp)
    KM, E, y, use = d["KM"], d["E"], d["y"], list(d["use"])
    D1 = {}
    for wn, w in WTS.items():
        for gi, g in enumerate(GS):
            D1[f"{wn}_g{g}"] = sum(w[m] * KM[m][gi] for m in range(6)) / sum(w)
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, D1=D1, E=E)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {c: np.array([o[1][c] for o in out]) for c in out[0][1]}
    R = {"stage": "development on the 753, benchmark protocol (not a confirmation)",
         "arms": {c: {"mean": float(v.mean()), "sd": float(v.std())} for c, v in ps.items()},
         "per_seed": {c: v.round(6).tolist() for c, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for c in sorted(R["arms"], key=lambda c: -R["arms"][c]["mean"]):
        dd = ps[c] - ps["A8cat"]
        print(f"{c:18s} {R['arms'][c]['mean']:.4f}  vs F-BERT {dd.mean():+.4f} {int((dd > 0).sum())}/60")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    else:
        evaluate(*sys.argv[2:5])
