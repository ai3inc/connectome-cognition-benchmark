"""Addendum A: F-BERT vs connectome kernels with all six modalities for every arm,
identical subjects, timeseries and the 60 x 10 family-aware splits.

usage: python3 eval_sixmod.py WORKSPACE OUT.json REST_FEATS_GLOB TASK_FEATS_GLOB
  rest features: feat_cbd*.npz (features_pod.py), task: taskfeat_cbd*.npz
"""
import csv, glob, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev   # correlation_kernel, spectral_filter, per_subject, krr_select, zlin_kernel, G_GRID

MODS = ["rs", "lang", "social", "gamb", "motor", "wm"]
WEIGHTS = {"eq": {m: 1 for m in MODS}, "lang2": {m: (2 if m == "lang" else 1) for m in MODS}}
N_SEEDS = int(os.environ.get("N_SEEDS", "60"))
G = {}


def load_feats(pattern, keymap):
    D = {}
    for f in sorted(glob.glob(pattern)):
        with np.load(f) as d:
            arrs = {k: d[k] for k in d.files}
        for i, s in enumerate(arrs["sids"]):
            row = D.setdefault(str(s), {})
            for src, dst in keymap.items():
                if src in arrs:
                    row[dst] = np.array(arrs[src][i])
        del arrs
    return D


def run_seed(nm):
    y, folds_of, arms = G["y"], G["folds"][nm], G["arms"]
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in arms}
    gsel = {a: [] for a in arms if a.startswith("D")}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        for a, spec in arms.items():
            Ks = [ev.zlin_kernel(spec[1], tr)] if spec[0] == "zlin" else spec[1]
            p, kb, _ = ev.krr_select(Ks, tr, te, y)
            pred[a][te] = p
            if a in gsel:
                gsel[a].append(ev.G_GRID[kb])
    return nm, {a: float(np.corrcoef(pred[a], y)[0, 1]) for a in arms}, gsel


def main():
    ws, outp, rest_pat, task_pat = sys.argv[1:5]
    t0 = time.time()
    rest = load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs", "e1_full": "e1_rs"})
    km = {f"{k}_{m}": f"{k}_{m}" for m in MODS[1:] for k in ["fc", "e8", "e1"]}
    task = load_feats(task_pat, km)
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use0 = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))]

    def complete(s):
        if s not in rest or s not in task:
            return False
        r = {**rest[s], **task[s]}
        return all(f"fc_{m}" in r and np.isfinite(r[f"fc_{m}"]).all() for m in MODS)
    use = [s for s in use0 if complete(s)]
    dropped = [s for s in use0 if s not in use]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:N_SEEDS]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    print(f"subjects {len(use)} (dropped {len(dropped)} without all six), seeds {len(names)}", flush=True)

    per_mod = {}
    E8, E1 = [], []
    for m in MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(ev.per_subject, list(fc), chunksize=8)
        K_full = ev.correlation_kernel(fc)
        Kd0, Kd1 = [], []
        for gi in range(len(ev.G_GRID)):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            Kd0.append((1 - ev.W_FULL) * Kp + ev.W_FULL * K_full)
            Kd1.append((1 - ev.W_FULL) * ev.spectral_filter(Kp) + ev.W_FULL * K_full)
        e8 = np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64)
        e1 = np.stack([src[s][f"e1_{m}"] for s in use]).astype(np.float64)
        E8.append(e8); E1.append(e1)
        per_mod[m] = {"B": K_full, "C": ev.correlation_kernel(np.stack([r[0] for r in res])),
                      "D0": Kd0, "D1": Kd1, "A8": ev.correlation_kernel(e8), "A1": ev.correlation_kernel(e1)}
        del fc, res
        print(f"  {m} kernels {time.time() - t0:.0f}s", flush=True)

    def ens(key, wname, gi=None):
        w = WEIGHTS[wname]
        tot = sum(w[m] * (per_mod[m][key][gi] if gi is not None else per_mod[m][key]) for m in MODS)
        return tot / sum(w.values())

    arms = {}
    for wn in WEIGHTS:
        for k in ["B", "C", "A8", "A1"]:
            arms[f"{k}6_{wn}"] = ("k", [ens(k, wn)])
        for k in ["D0", "D1"]:
            arms[f"{k}6_{wn}"] = ("k", [ens(k, wn, gi) for gi in range(len(ev.G_GRID))])
    arms["A8cat"] = ("zlin", np.concatenate(E8, 1))
    arms["A1cat"] = ("zlin", np.concatenate(E1, 1))
    # single-modality reference for each modality (ours D0 and F-BERT-8M), equal footing
    for m in MODS:
        arms[f"D0_{m}"] = ("k", per_mod[m]["D0"])
        arms[f"A8_{m}"] = ("k", [per_mod[m]["A8"]])
    del per_mod
    print(f"arms {len(arms)} built {time.time() - t0:.0f}s", flush=True)

    G.update(y=y, folds=fl, arms=arms)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    per_seed = {a: np.array([o[1][a] for o in out]) for a in arms}
    gsel = {a: sum((o[2][a] for o in out), []) for a in out[0][2]}

    def summ(v):
        return {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    def paired(a, b):
        d = a - b
        return {"mean_diff": float(d.mean()), "positive": int((d > 0).sum()), "of": len(d),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std() > 0 else None}

    R = {"n_subjects": len(use), "dropped_subjects": dropped, "n_seeds": len(names),
         "arms": {a: summ(v) for a, v in per_seed.items()},
         "g_selected_counts": {a: {str(g): gsel[a].count(g) for g in ev.G_GRID} for a in gsel},
         "per_seed": {a: v.round(6).tolist() for a, v in per_seed.items()}}
    contrasts, used = {}, {}
    for wn in WEIGHTS:
        A = {}
        for t in ["8", "1"]:
            km_, cat = per_seed[f"A{t}6_{wn}"], per_seed[f"A{t}cat"]
            pick = "kernel_mean" if km_.mean() >= cat.mean() else "concat_probe"
            A[t] = km_ if pick == "kernel_mean" else cat
            used[f"A{t}6_{wn}"] = pick
        D0, D1, C, B = (per_seed[f"{k}6_{wn}"] for k in ["D0", "D1", "C", "B"])
        contrasts[wn] = {"D0-A8": paired(D0, A["8"]), "D0-A1": paired(D0, A["1"]),
                         "D1-A8": paired(D1, A["8"]), "D0-C": paired(D0, C), "D1-C": paired(D1, C),
                         "C-A8": paired(C, A["8"]), "B-A8": paired(B, A["8"]), "D0-B": paired(D0, B)}
    R["fbert_ensemble_used"] = used
    R["contrasts"] = contrasts
    R["decision_rule"] = "claim only if positive in >= 57/60 seeds; primary = eq, D0-A8"
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)
    for a, v in sorted(R["arms"].items()):
        print(f"{a:14s} {v['mean']:.4f} +- {v['sd']:.4f}")
    print(json.dumps(used))
    for wn, c in contrasts.items():
        for k, v in c.items():
            print(wn, k, round(v["mean_diff"], 4), f"{v['positive']}/{v['of']}", "t=", v["t"] and round(v["t"], 1))


if __name__ == "__main__":
    main()
