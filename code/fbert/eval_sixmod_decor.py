"""Addendum C: remove the shared cross-modality component, beta chosen in-fold,
for ours (DECOR) and for F-BERT-8M (A8DECOR). Two phases so kernel building can use
all BLAS threads and the fold loop can run 8 single-threaded workers:

  OMP_NUM_THREADS=8 python3 eval_sixmod_decor.py build WORKSPACE KERNELS.npz REST_GLOB TASK_GLOB
  OMP_NUM_THREADS=1 python3 eval_sixmod_decor.py eval  WORKSPACE KERNELS.npz OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

GS = [0.95, 0.97, 0.98]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def subjects(ws, rest, task):
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))
           and s in rest and s in task]
    return use, np.array([float(oi[s]["varimax_cog"]) for s in use])


def build(ws, kp, rest_pat, task_pat):
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs"})
    task = sx.load_feats(task_pat, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]})
    use, y = subjects(ws, rest, task)
    Kfull, keys, Ks = [], [], []
    for gi, g in enumerate(GS):
        ev.G_GRID = [g]
        X = []
        for m in sx.MODS:
            src = rest if m == "rs" else task
            fc = np.stack([src[s][f"fc_{m}"] for s in use])
            if gi == 0:
                Kfull.append(ev.correlation_kernel(fc))
            res = [ev.per_subject(v) for v in fc]   # sequential: BLAS uses all threads here
            X.append(np.stack([r[1][0] for r in res]))
            del fc, res
        S = sum(X) / len(X)
        for b in BETAS:
            K = sum((1 - ev.W_FULL) * ev.correlation_kernel(X[j] - b * S) + ev.W_FULL * Kfull[j]
                    for j in range(6)) / 6
            keys.append(f"g{g}_b{b}"); Ks.append(K)
            print(f"  g={g} beta={b} {time.time() - t0:.0f}s", flush=True)
        if gi == 1:   # mechanism at g=0.97: cross-modality alignment of the per-modality kernels per beta
            mech = {}
            for b in BETAS:
                Km = [ev.correlation_kernel(X[j] - b * S) for j in range(6)]
                n = len(y); H = np.eye(n) - 1.0 / n
                C = [H @ K @ H for K in Km]
                vals = [float((C[i] * C[j]).sum() / np.linalg.norm(C[i]) / np.linalg.norm(C[j]))
                        for i in range(6) for j in range(i + 1, 6)]
                mech[str(b)] = float(np.mean(vals))
            print("  mechanism", mech, flush=True)
        del X, S
    E = [np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64) for m in sx.MODS]
    np.savez(kp, Ks=np.stack(Ks), keys=np.array(keys), E=np.stack(E), y=y, use=np.array(use),
             mech=json.dumps(mech))
    print("SAVED", kp, f"{time.time() - t0:.0f}s", flush=True)


G = {}


def run_seed(nm):
    y, folds_of, Ks, E = G["y"], G["folds"][nm], G["Ks"], G["E"]
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in ["DECOR", "A8cat", "A8DECOR"]}
    pick_o, pick_f = [], []
    Ebar = E.mean(0)
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        p, kb, _ = ev.krr_select(list(Ks), tr, te, y)
        pred["DECOR"][te] = p; pick_o.append(int(kb))
        cands = [ev.zlin_kernel(np.concatenate(list(E - b * Ebar[None]), 1), tr) for b in BETAS]
        p, kb, perk = ev.krr_select(cands, tr, te, y)
        pred["A8DECOR"][te] = p; pick_f.append(BETAS[kb])
        pred["A8cat"][te] = perk[0]   # beta=0 candidate at its own inner-best lambda = A8cat
    return nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}, pick_o, pick_f


def evaluate(ws, kp, outp):
    t0 = time.time()
    d = np.load(kp)
    Ks, keys, E, y, use = d["Ks"], list(d["keys"]), d["E"], d["y"], list(d["use"])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:sx.N_SEEDS]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, Ks=Ks, E=E)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {a: np.array([o[1][a] for o in out]) for a in out[0][1]}
    po = sum((o[2] for o in out), []); pf = sum((o[3] for o in out), [])

    def paired(a, b):
        x = a - b
        return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
    prev = json.load(open(os.path.join(os.path.dirname(outp) or ".", "FBERT_SIXMOD_1117.json")))
    R = {"addendum": "C (development round 3, declared before running)", "n_subjects": len(y),
         "n_seeds": len(names), "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "DECOR_picks": {keys[i]: po.count(i) for i in sorted(set(po))},
         "A8DECOR_beta_picks": {str(b): pf.count(b) for b in BETAS},
         "mechanism_cross_modality_cka_g0.97": json.loads(str(d["mech"])),
         "contrasts": {"DECOR-A8cat": paired(ps["DECOR"], ps["A8cat"]),
                       "DECOR-A8DECOR": paired(ps["DECOR"], ps["A8DECOR"]),
                       "DECOR-D06_eq(addendum A)": paired(ps["DECOR"], np.array(prev["per_seed"]["D06_eq"]))},
         "A8cat_max_absdiff_vs_addendumA": float(np.abs(ps["A8cat"] - np.array(prev["per_seed"]["A8cat"])).max()),
         "per_seed": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in R["arms"].items():
        print(f"{a:8s} {v['mean']:.4f} +- {v['sd']:.4f}")
    print("picks", R["DECOR_picks"], R["A8DECOR_beta_picks"], "mech", R["mechanism_cross_modality_cka_g0.97"])
    for k, v in R["contrasts"].items():
        print(k, round(v["mean_diff"], 4), f"{v['positive']}/{v['of']}", "t=", round(v["t"], 1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    else:
        evaluate(*sys.argv[2:5])
