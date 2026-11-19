"""Addendum B (post-hoc development, candidates fixed in PREREG before running):
in-fold selection over 18 per-subject kernels vs F-BERT-8M six-modality concat probe.

usage: python3 eval_sixmod_plus.py WORKSPACE OUT.json REST_GLOB TASK_GLOB
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

GRID = [0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
ev.G_GRID = GRID                      # set before any Pool forks
GI = {g: i for i, g in enumerate(GRID)}
TASK_LEN = {"lang": 316, "social": 274, "gamb": 253, "motor": 284, "wm": 405}  # per run, HCP fixed
TASK_ORDER = ["lang", "social", "gamb", "motor", "wm"]                           # = 'has' columns
G = {}


def run_seed(nm):
    y, folds_of, fam, a8 = G["y"], G["folds"][nm], G["fam"], G["a8cat"]
    n = len(y)
    names = [k for k in fam]
    allK = [K for k in names for K in fam[k]]
    owner = [k for k in names for _ in fam[k]]
    pred = {k: np.full(n, np.nan) for k in names + ["PLUS", "A8cat"]}
    picks = []
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        # PLUS: one selection over every candidate; per-family values from the same pass
        S = np.zeros((len(allK), len(ev.LAMBDAS)))
        for a, b in ev.inner_splits(len(tr)):
            ya = y[tr][a] - y[tr][a].mean()
            for ki, K in enumerate(allK):
                w, V = np.linalg.eigh(K[np.ix_(tr[a], tr[a])])
                VtK = K[np.ix_(tr[b], tr[a])] @ V
                Vt = V.T @ ya
                for li, lam in enumerate(ev.LAMBDAS):
                    S[ki, li] += np.corrcoef(VtK @ (Vt / (w + lam)), y[tr][b])[0, 1] / 4

        def fit(ki):
            lam = ev.LAMBDAS[int(np.nanargmax(S[ki]))]
            K = allK[ki]
            w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
            yc = y[tr] - y[tr].mean()
            return (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()
        best_all = int(np.unravel_index(np.nanargmax(S), S.shape)[0])
        pred["PLUS"][te] = fit(best_all)
        picks.append(f"{owner[best_all]}#{best_all}")
        for k in names:
            idx = [i for i, o in enumerate(owner) if o == k]
            bi = idx[int(np.argmax([np.nanmax(S[i]) for i in idx]))]
            pred[k][te] = fit(bi)
        p, _, _ = ev.krr_select([ev.zlin_kernel(a8, tr)], tr, te, y)
        pred["A8cat"][te] = p
    return nm, {k: float(np.corrcoef(v, y)[0, 1]) for k, v in pred.items()}, picks


def main():
    ws, outp, rest_pat, task_pat = sys.argv[1:5]
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs", "tfull": "t_rs"})
    km = {f"{k}_{m}": f"{k}_{m}" for m in TASK_ORDER for k in ["fc", "e8"]}
    km["has"] = "has"
    task = sx.load_feats(task_pat, km)
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

    # time weights per subject and modality
    W = np.zeros((len(use), 6))
    for i, s in enumerate(use):
        W[i, 0] = float(rest[s]["t_rs"])
        for j, m in enumerate(TASK_ORDER):
            W[i, j + 1] = float(task[s]["has"][j]) * TASK_LEN[m]
    gfc = np.zeros((len(use), len(ev.IU[0])), np.float64)
    d0 = {g: None for g in GRID}
    c6 = None
    for j, m in enumerate(sx.MODS):
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        gfc += W[:, j:j + 1] * fc
        with Pool(8) as pool:
            res = pool.map(ev.per_subject, list(fc), chunksize=8)
        K_full = ev.correlation_kernel(fc)
        for g in GRID:
            Kp = ev.correlation_kernel(np.stack([r[1][GI[g]] for r in res]))
            K = (1 - ev.W_FULL) * Kp + ev.W_FULL * K_full
            d0[g] = K if d0[g] is None else d0[g] + K
        Kc = ev.correlation_kernel(np.stack([r[0] for r in res]))
        c6 = Kc if c6 is None else c6 + Kc
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    d0 = {g: K / 6 for g, K in d0.items()}
    c6 = c6 / 6
    gfc = (gfc / W.sum(1, keepdims=True)).astype(np.float32)
    with Pool(8) as pool:
        res = pool.map(ev.per_subject, list(gfc), chunksize=8)
    Kgf = ev.correlation_kernel(gfc)
    k2 = {g: (1 - ev.W_FULL) * ev.correlation_kernel(np.stack([r[1][GI[g]] for r in res])) + ev.W_FULL * Kgf
          for g in GRID}
    del res
    print(f"  GFC {time.time() - t0:.0f}s", flush=True)

    fam = {"K1": [d0[g] for g in [0.9, 0.95, 0.97, 0.98, 0.99]],
           "K2": [k2[g] for g in [0.85, 0.9, 0.95, 0.97, 0.98]],
           "K3": [0.5 * d0[g] + 0.5 * k2[g] for g in [0.9, 0.95, 0.97, 0.98]],
           "K4": [sum(d0[g] for g in [0.85, 0.9, 0.95, 0.98]) / 4],
           "K5": [0.5 * d0[g] + 0.5 * c6 for g in [0.95, 0.97, 0.98]]}
    assert sum(len(v) for v in fam.values()) == 18
    E = []
    for m in sx.MODS:
        src = rest if m == "rs" else task
        E.append(np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64))
    G.update(y=y, folds=fl, fam=fam, a8cat=np.concatenate(E, 1))
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    per_seed = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}
    picks = sum((o[2] for o in out), [])
    fam_picks = {}
    for p in picks:
        f = p.split("#")[0]
        fam_picks[f] = fam_picks.get(f, 0) + 1

    def paired(a, b):
        d = a - b
        return {"mean_diff": float(d.mean()), "positive": int((d > 0).sum()), "of": len(d),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}
    prev = json.load(open(os.path.join(os.path.dirname(outp) or ".", "FBERT_SIXMOD_1117.json")))
    R = {"addendum": "B (post-hoc development, candidates fixed before running)",
         "n_subjects": len(use), "n_seeds": len(names),
         "arms": {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in per_seed.items()},
         "PLUS_family_picks": fam_picks,
         "PLUS_kernel_picks": {p: picks.count(p) for p in sorted(set(picks))},
         "contrasts": {f"{k}-A8cat": paired(per_seed[k], per_seed["A8cat"]) for k in per_seed if k != "A8cat"},
         "A8cat_matches_addendumA": float(np.abs(per_seed["A8cat"] - np.array(prev["per_seed"]["A8cat"])).max()),
         "per_seed": {k: v.round(6).tolist() for k, v in per_seed.items()},
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for k, v in R["arms"].items():
        print(f"{k:6s} {v['mean']:.4f} +- {v['sd']:.4f}")
    print("picks", fam_picks, "A8cat max |diff| vs addendum A", R["A8cat_matches_addendumA"])
    for k, v in R["contrasts"].items():
        print(k, round(v["mean_diff"], 4), f"{v['positive']}/{v['of']}", "t=", round(v["t"], 1))


if __name__ == "__main__":
    main()
