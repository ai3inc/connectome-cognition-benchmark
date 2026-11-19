"""Addendum G: multi-atlas ensemble of our own extractions.

Part 1 (development, 753, 60 x 10 folds): D1-450 (from D1H_K.npz) vs D1-232 vs MA
(mean of the two predictions).
Part 2 (held-out, fourth look): D1-232 trained on the 753 once (g, weighting by
10-fold inner CV), predicted on the 263; MA = mean with the Addendum D D1-450
held-out prediction (saved). Primary: MA - D1-450, paired bootstrap.

usage: python3 eval_multiatlas.py WS FC232_GLOB D1H_K.npz HOLDOUT_D.json OUT.json
"""
import csv, glob, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_holdout as eh

MODS = ["rs", "lang", "social", "gamb", "motor", "wm"]
GS = [0.85, 0.9, 0.95]
WTS = {"eq": [1, 1, 1, 1, 1, 1], "lang2": [1, 2, 1, 1, 1, 1]}
G = {}


def partial_generic(v):
    with threadpool_limits(1):
        n_e = len(v)
        R = int(round((1 + np.sqrt(1 + 8 * n_e)) / 2))
        iu = np.triu_indices(R, 1)
        C = np.zeros((R, R)); C[iu] = v; C = C + C.T; np.fill_diagonal(C, 1.0)
        out = []
        for g in GS:
            P = np.linalg.inv((1 - g) * C + g * np.eye(R))
            d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            P = -P / np.outer(d, d); np.fill_diagonal(P, 1.0)
            w, V = np.linalg.eigh(P)
            out.append(((V * np.clip(w, 0, None) ** ev.ALPHA) @ V.T)[iu].astype(np.float32))
        return out


def d1_kernels(F):
    """F: dict mod -> (n, edges). Returns {f'{wn}_g{g}': kernel} over all rows (transductive)."""
    KM = []
    for m in MODS:
        with Pool(8) as pool:
            res = pool.map(partial_generic, list(F[m]), chunksize=8)
        Kf = ev.correlation_kernel(F[m])
        KM.append([(1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(np.stack([r[gi] for r in res])))
                   + ev.W_FULL * Kf for gi in range(len(GS))])
        print("   ", m, flush=True)
    return {f"{wn}_g{g}": sum(w[mi] * KM[mi][gi] for mi in range(6)) / sum(w)
            for wn, w in WTS.items() for gi, g in enumerate(GS)}


def run_seed(nm):
    with threadpool_limits(1):
        return _run_seed(nm)


def _run_seed(nm):
    y, folds_of, A, B = G["y"], G["folds"][nm], G["A"], G["B"]
    n = len(y)
    pred = {k: np.full(n, np.nan) for k in ["D1_450", "D1_232", "MA"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        pa, _, _ = ev.krr_select(A, tr, te, y)
        pb, _, _ = ev.krr_select(B, tr, te, y)
        pred["D1_450"][te] = pa; pred["D1_232"][te] = pb; pred["MA"][te] = 0.5 * (pa + pb)
    return nm, {k: float(np.corrcoef(v, y)[0, 1]) for k, v in pred.items()}


def main():
    ws, fcglob, d1hp, hdp, outp = sys.argv[1:6]
    t0 = time.time()
    F232 = {}
    for f in sorted(glob.glob(fcglob)):
        with np.load(f) as d:
            for i, s in enumerate(d["sids"]):
                F232[str(s)] = {m: np.array(d[f"fc_{m}"][i]) for m in MODS}
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    d1h = np.load(d1hp)
    use = [str(s) for s in d1h["use"] if str(s) in F232]
    ui = np.array([[str(s) for s in d1h["use"]].index(s) for s in use])
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    KM = d1h["KM"][:, :, ui][:, :, :, ui]
    A = [sum(w[m] * KM[m][gi] for m in range(6)) / sum(w) for w in WTS.values() for gi in range(len(GS))]
    print(f"part 1: {len(use)} subjects with both atlases", flush=True)
    B = list(d1_kernels({m: np.stack([F232[s][m] for s in use]) for m in MODS}).values())
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=y, folds=fl, A=A, B=B)
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}
    x = ps["MA"] - ps["D1_450"]
    part1 = {"arms": {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in ps.items()},
             "MA_vs_D1_450": {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()),
                              "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))},
             "per_seed": {k: v.round(6).tolist() for k, v in ps.items()}}
    print("part 1", json.dumps({k: v for k, v in part1.items() if k != "per_seed"}), flush=True)

    # part 2: held-out
    H = json.load(open(hdp))
    hold = [s for s in H["holdout_ids"] if s in F232]
    s753 = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    train = [s for s in s753 if s in F232]
    allids = train + hold
    tr = np.arange(len(train)); te = np.arange(len(train), len(allids))
    yy = np.array([float(oi[s]["varimax_cog"]) for s in allids])
    KB = d1_kernels({m: np.stack([F232[s][m] for s in allids]) for m in MODS})
    labels = list(KB)
    ki, lam, _ = eh.inner_select([KB[k] for k in labels], tr, yy)
    p232 = eh.fit_predict(KB[labels[ki]], tr, te, yy, lam)
    hpos = [H["holdout_ids"].index(s) for s in hold]
    p450 = np.array(H["predictions"]["D1"])[hpos]
    pfb = np.array(H["predictions"]["FBERT_best"])[hpos]
    yt = yy[te]
    pma = 0.5 * (p450 + p232)
    part2 = {"n_train": len(train), "n_holdout": len(hold), "chosen_232": {"kernel": labels[ki], "lambda": lam},
             "r": {"D1_450": float(np.corrcoef(p450, yt)[0, 1]), "D1_232": float(np.corrcoef(p232, yt)[0, 1]),
                   "MA": float(np.corrcoef(pma, yt)[0, 1]), "FBERT": float(np.corrcoef(pfb, yt)[0, 1])},
             "PRIMARY_MA_vs_D1_450": eh.boot_diff(pma, p450, yt),
             "MA_vs_FBERT": eh.boot_diff(pma, pfb, yt),
             "D1_232_vs_D1_450": eh.boot_diff(p232, p450, yt),
             "corr_450_232_predictions": float(np.corrcoef(p450, p232)[0, 1]),
             "holdout_ids": hold, "prediction_232": p232.round(6).tolist()}
    cp = os.path.join(os.path.dirname(outp) or ".", "CANON_HOLDOUT_233_1117.json")
    if os.path.exists(cp):
        Cn = json.load(open(cp))
        cid = [str(s) for s in Cn["holdout_ids"]]
        common = [s for s in hold if s in cid]
        hi = np.array([hold.index(s) for s in common]); ci = np.array([cid.index(s) for s in common])
        p3 = (p450[hi] + p232[hi] + np.array(Cn["predictions"]["D1"])[ci]) / 3
        part2["three_way_on_intersection"] = {"n": len(common), "r": float(np.corrcoef(p3, yt[hi])[0, 1]),
                                              "vs_D1_450": eh.boot_diff(p3, p450[hi], yt[hi])}
    R = {"addendum": "G (multi-atlas, fourth held-out look)", "part1_dev_753": part1, "part2_holdout": part2,
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print("part 2", json.dumps({k: v for k, v in part2.items() if k not in ("holdout_ids", "prediction_232")}, indent=1))


if __name__ == "__main__":
    main()
