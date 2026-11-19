"""EXPLORATORY (not preregistered) follow-ups to Addendum A, run after its result.

1. Grid edge. In the six-modality ensemble the in-fold g landed on the grid edge
   0.98 in 577/600 folds (every single modality chose an interior value). Re-run
   D0-6 (equal weights) on a grid extended toward 1, and record the fixed-g curve.
2. Is the D1 margin the filter? Apply the same kernel spectral filter to F-BERT's
   kernels: per-modality correlation kernels filtered then averaged (A8F6), and the
   filtered correlation kernel of the six embeddings concatenated (A8catF).
Same subjects, folds, inner splits and lambda grid as eval_sixmod.py, so per-seed
values pair with results/FBERT_SIXMOD_1117.json.

usage: python3 eval_sixmod_explore.py WORKSPACE OUT.json REST_GLOB TASK_GLOB
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

G2 = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
ev.G_GRID = G2          # per_subject reads it at call time; set before the Pool forks
G = {}


def run_seed(nm):
    y, folds_of, arms = G["y"], G["folds"][nm], G["arms"]
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in arms}
    pred_g = {gi: np.full(n, np.nan) for gi in range(len(G2))}
    gsel = []
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        for a, Ks in arms.items():
            p, kb, perk = ev.krr_select(Ks, tr, te, y)
            pred[a][te] = p
            if a == "D06x_eq":
                gsel.append(G2[kb])
                for gi, pk in enumerate(perk):
                    pred_g[gi][te] = pk
    r = {a: float(np.corrcoef(pred[a], y)[0, 1]) for a in arms}
    rg = [float(np.corrcoef(pred_g[gi], y)[0, 1]) for gi in range(len(G2))]
    return nm, r, rg, gsel


def main():
    ws, outp, rest_pat, task_pat = sys.argv[1:5]
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs"})
    task = sx.load_feats(task_pat, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))
           and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:sx.N_SEEDS]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    print(f"subjects {len(use)} seeds {len(names)}", flush=True)

    d0 = [None] * len(G2)
    a8f, Ecat = [], []
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(ev.per_subject, list(fc), chunksize=8)
        K_full = ev.correlation_kernel(fc)
        for gi in range(len(G2)):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            K = (1 - ev.W_FULL) * Kp + ev.W_FULL * K_full
            d0[gi] = K if d0[gi] is None else d0[gi] + K
        e8 = np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64)
        a8f.append(ev.spectral_filter(ev.correlation_kernel(e8)))
        Ecat.append((e8 - e8.mean(0)) / (e8.std(0) + 1e-12))
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    arms = {"D06x_eq": [K / len(sx.MODS) for K in d0],
            "A8F6_eq": [sum(a8f) / len(sx.MODS)],
            "A8catF": [ev.spectral_filter(ev.correlation_kernel(np.concatenate(Ecat, 1)))],
            "A8F_rs": [a8f[0]]}
    G.update(y=y, folds=fl, arms=arms)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    per_seed = {a: np.array([o[1][a] for o in out]) for a in arms}
    curve = np.array([o[2] for o in out]).mean(0)
    gsel = sum((o[3] for o in out), [])
    prev = json.load(open(os.path.join(os.path.dirname(outp) or ".", "FBERT_SIXMOD_1117.json")))
    P = {k: np.array(v) for k, v in prev["per_seed"].items()}

    def paired(a, b):
        d = a - b
        return {"mean_diff": float(d.mean()), "positive": int((d > 0).sum()), "of": len(d),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}

    R = {"exploratory": True, "g_grid": G2, "n_subjects": len(use), "n_seeds": len(names),
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in per_seed.items()},
         "D06x_fixed_g_curve": dict(zip(map(str, G2), curve.round(5).tolist())),
         "D06x_g_selected": {str(g): gsel.count(g) for g in G2},
         "contrasts": {
             "D06x_eq - A8cat": paired(per_seed["D06x_eq"], P["A8cat"]),
             "D06x_eq - D06_eq(prereg grid)": paired(per_seed["D06x_eq"], P["D06_eq"]),
             "D16_eq - A8F6_eq": paired(P["D16_eq"], per_seed["A8F6_eq"]),
             "D16_eq - A8catF": paired(P["D16_eq"], per_seed["A8catF"]),
             "A8F6_eq - A8cat": paired(per_seed["A8F6_eq"], P["A8cat"]),
             "A8F_rs - A8_rs": paired(per_seed["A8F_rs"], P["A8_rs"])},
         "per_seed": {a: v.round(6).tolist() for a, v in per_seed.items()},
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in R["arms"].items():
        print(f"{a:10s} {v['mean']:.4f} +- {v['sd']:.4f}")
    print("curve", R["D06x_fixed_g_curve"], "sel", R["D06x_g_selected"])
    for k, v in R["contrasts"].items():
        print(k, round(v["mean_diff"], 4), f"{v['positive']}/{v['of']}", "t=", round(v["t"], 1))


if __name__ == "__main__":
    main()
