"""Addendum T: add the unused HCP tasks (RELATIONAL, EMOTION) to our 450-region chain.

  infold : 753 (those with all eight runs), 60 x 10 family-aware folds; D1 / D0 ensembles with g and
           language weighting chosen in-fold; F-BERT-8M ensembles (best of concat / filtered corr).
  holdout: train on the 753, predict the held-out subjects once (Addendum D pipeline).

usage: python3 rel_eval.py infold  WS REST_GLOB TASK_GLOB REL_GLOB OUT.json
       python3 rel_eval.py holdout WS TRAIN_REST TRAIN_TASK HOLD_REST HOLD_TASK REL_GLOB OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

BASE = sx.MODS                                    # rs lang social gamb motor wm
SETS = {"six": BASE, "rel": BASE + ["rel"], "emo": BASE + ["emo"], "relemo": BASE + ["rel", "emo"]}
ALL = BASE + ["rel", "emo"]
G = {}


def paired_seeds(a, b):
    x = a - b
    return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
            "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}


def load(rest_globs, task_globs, rel_glob):
    rest, task = {}, {}
    for g in rest_globs:
        rest.update(sx.load_feats(g, {"fc_full": "fc_rs", "e8_full": "e8_rs"}))
    for g in task_globs:
        task.update(sx.load_feats(g, {f"{k}_{m}": f"{k}_{m}" for m in BASE[1:] for k in ["fc", "e8"]}))
    rel = sx.load_feats(rel_glob, {f"{k}_{m}": f"{k}_{m}" for m in ["rel", "emo"] for k in ["fc", "e8"]})
    for s, r in rel.items():
        task.setdefault(s, {}).update(r)
    return rest, task


def complete(s, rest, task, oi):
    if s not in rest or s not in task or s not in oi or not np.isfinite(float(oi[s].get("varimax_cog") or "nan")):
        return False
    r = {**rest[s], **task[s]}
    return all(f"fc_{m}" in r and np.isfinite(r[f"fc_{m}"]).all() and f"e8_{m}" in r
               and np.isfinite(r[f"e8_{m}"]).all() for m in ALL)


def run_infold(nm):
    with threadpool_limits(1):
        y, fl, arms = G["y"], G["folds"][nm], G["arms"]
        n = len(y)
        pred = {a: np.full(n, np.nan) for a in arms}
        for te in fl:
            tr = np.setdiff1d(np.arange(n), te)
            for a, Ks in arms.items():
                Kl = [K(tr) if callable(K) else K for K in Ks]
                pred[a][te] = ev.krr_select(Kl, tr, te, y)[0]
        return nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}


def infold(ws, rglob, tglob, relglob, outp):
    from sota_hcp import per_subject_grid
    t0 = time.time()
    GS = [0.85, 0.9, 0.95]
    rest, task = load([rglob], [tglob], relglob)
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if complete(s, rest, task, oi)]
    print("subjects with all eight runs", len(use), "of", len(off), flush=True)
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:60]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    D0, D1, FBc, FBe = {}, {}, {}, {}
    for m in ALL:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(per_subject_grid, [(v, GS) for v in fc], chunksize=8)
        Kf = ev.correlation_kernel(fc)
        for gi, g in enumerate(GS):
            Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
            D0[(m, g)] = 0.95 * Kp + 0.05 * Kf
            D1[(m, g)] = 0.95 * ev.spectral_filter(Kp) + 0.05 * Kf
        E = np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64)
        Kc = ev.correlation_kernel(E)
        FBc[m] = (Kc, ev.spectral_filter(Kc)); FBe[m] = E
        del fc, res
        print(m, f"{time.time() - t0:.0f}s", flush=True)

    def ens(KD, mods, lang2):
        w = {m: (2 if (lang2 and m == "lang") else 1) for m in mods}
        return [sum(w[m] * KD[(m, g)] for m in mods) / sum(w.values()) for g in GS]
    arms = {}
    for st, mods in SETS.items():
        arms[f"D1_{st}"] = ens(D1, mods, False) + ens(D1, mods, True)
        arms[f"D0_{st}"] = ens(D0, mods, False) + ens(D0, mods, True)
        Ecat = np.concatenate([FBe[m] for m in mods], 1)
        arms[f"FB_{st}"] = [sum(FBc[m][0] for m in mods) / len(mods), sum(FBc[m][1] for m in mods) / len(mods),
                            (lambda tr, E=Ecat: ev.zlin_kernel(E, tr))]
    G.update(y=y, folds=fl, arms=arms)
    with Pool(8) as pool:
        out = pool.map(run_infold, names)
    ps = {a: np.array([o[1][a] for o in out]) for a in arms}
    R = {"addendum": "T in-fold (RELATIONAL/EMOTION)", "n": len(use),
         "arms": {a: {"mean": float(v.mean()), "sd": float(v.std())} for a, v in ps.items()},
         "PRIMARY_D1_rel_vs_six": paired_seeds(ps["D1_rel"], ps["D1_six"]),
         "D0_rel_vs_six": paired_seeds(ps["D0_rel"], ps["D0_six"]),
         "D1_emo_vs_six": paired_seeds(ps["D1_emo"], ps["D1_six"]),
         "D1_relemo_vs_six": paired_seeds(ps["D1_relemo"], ps["D1_six"]),
         "FB_rel_vs_six": paired_seeds(ps["FB_rel"], ps["FB_six"]),
         "D1_rel_vs_FB_rel": paired_seeds(ps["D1_rel"], ps["FB_rel"]),
         "D0_rel_vs_FB_rel": paired_seeds(ps["D0_rel"], ps["FB_rel"]),
         "per_seed": {a: v.round(6).tolist() for a, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "per_seed"}, indent=1))


def holdout(ws, trr, trt, hor, hot, relglob, outp):
    import eval_holdout as eh
    t0 = time.time()
    rest, task = load([trr, hor], [trt, hot], relglob)
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    s753 = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    train = [s for s in s753 if complete(s, rest, task, oi)]
    hold = sorted(s for s in task if s not in set(s753) and complete(s, rest, task, oi))
    use = train + hold
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    tr = np.arange(len(train)); te = np.arange(len(train), len(use))
    print(f"train {len(train)} held-out {len(hold)}", flush=True)
    KI, KT, EB = {}, {}, {}
    for m in ALL:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(eh.partial_edges, list(fc), chunksize=8)
        Kf_i = eh.inductive_corr_kernel(fc.astype(np.float64), tr)
        Kf_t = ev.correlation_kernel(fc)
        for gi, g in enumerate(eh.GS):
            Xp = np.stack([r[gi] for r in res])
            KI[(m, g)] = (1 - ev.W_FULL) * eh.inductive_corr_kernel(Xp.astype(np.float64), tr) + ev.W_FULL * Kf_i
            KT[(m, g)] = (1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(Xp)) + ev.W_FULL * Kf_t
        EB[m] = np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64)
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    arms = {}
    for st, mods in SETS.items():
        arms[f"D0_{st}"] = [sum(KI[(m, g)] for m in mods) / len(mods) for g in eh.GS]
        arms[f"D1_{st}"] = [sum(KT[(m, g)] for m in mods) / len(mods) for g in eh.GS]
        arms[f"FB_{st}"] = [ev.zlin_kernel(np.concatenate([EB[m] for m in mods], 1), tr),
                            sum(ev.spectral_filter(ev.correlation_kernel(EB[m])) for m in mods) / len(mods),
                            sum(ev.correlation_kernel(EB[m]) for m in mods) / len(mods)]
    preds, chosen = {}, {}
    for a, Ks in arms.items():
        ki, lam, _ = eh.inner_select(Ks, tr, y)
        preds[a] = eh.fit_predict(Ks[ki], tr, te, y, lam)
        chosen[a] = {"kernel": ki, "lambda": lam}
    yt = y[te]
    B = lambda a, b: eh.boot_diff(preds[a], preds[b], yt)
    R = {"addendum": "T held-out (RELATIONAL/EMOTION)", "n_train": len(train), "n_holdout": len(hold), "chosen": chosen,
         "holdout_r": {a: float(np.corrcoef(p, yt)[0, 1]) for a, p in preds.items()},
         "COPRIMARY_D1_rel_vs_six": B("D1_rel", "D1_six"), "D0_rel_vs_six": B("D0_rel", "D0_six"),
         "D1_emo_vs_six": B("D1_emo", "D1_six"), "D1_relemo_vs_six": B("D1_relemo", "D1_six"),
         "FB_rel_vs_six": B("FB_rel", "FB_six"),
         "D1_rel_vs_FB_rel": B("D1_rel", "FB_rel"), "D0_rel_vs_FB_rel": B("D0_rel", "FB_rel"),
         "holdout_ids": hold, "predictions": {a: p.round(6).tolist() for a, p in preds.items()},
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("holdout_ids", "predictions")}, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "infold":
        infold(*sys.argv[2:7])
    else:
        holdout(*sys.argv[2:9])
