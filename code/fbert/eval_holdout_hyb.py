"""Addendum E: held-out second look for HYB = (1-rho) D1(g, weighting) + rho F-BERT.
Train on the 753 once (choices by 10-fold CV inside the 753), predict the 263 once.
Paired against the Addendum D predictions (same held-out order).

usage: python3 eval_holdout_hyb.py WS OUT.json PREV_HOLDOUT.json TRAIN_REST TRAIN_TASK HOLD_REST HOLD_TASK
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx
import eval_holdout as eh

eh.GS = [0.85, 0.9, 0.95]
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
WTS = {"eq": [1, 1, 1, 1, 1, 1], "lang2": [1, 2, 1, 1, 1, 1]}


def main():
    ws, outp, prevp, trr, trt, hor, hot = sys.argv[1:8]
    t0 = time.time()
    prev = json.load(open(prevp))
    rest, task = eh.load_all(ws, [(trr, trt), (hor, hot)])
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    s753 = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]

    def ok(s):
        if s not in rest or s not in task or s not in oi:
            return False
        if not np.isfinite(float(oi[s].get("varimax_cog") or "nan")):
            return False
        r = {**rest[s], **task[s]}
        return all(f"fc_{m}" in r and np.isfinite(r[f"fc_{m}"]).all() for m in sx.MODS)
    train = [s for s in s753 if ok(s)]
    hold = sorted(s for s in task if s not in set(s753) and ok(s))
    assert hold == prev["holdout_ids"], "held-out set differs from addendum D"
    use = train + hold
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    tr = np.arange(len(train)); te = np.arange(len(train), len(use))

    KM = []   # per modality, per g: filtered partial kernel + 5% full, over train + held-out
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(eh.partial_edges, list(fc), chunksize=8)
        Kf = ev.correlation_kernel(fc)
        KM.append([(1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(np.stack([r[gi] for r in res])))
                   + ev.W_FULL * Kf for gi in range(len(eh.GS))])
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    E = np.concatenate([np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64)
                        for m in sx.MODS], 1)
    Z = ev.zlin_kernel(E, tr)
    cands, labels = [], []
    for wn, w in WTS.items():
        for gi, g in enumerate(eh.GS):
            D1 = sum(w[m] * KM[m][gi] for m in range(6)) / sum(w)
            for r in RHOS:
                cands.append((1 - r) * D1 + r * Z); labels.append(f"{wn}_g{g}_r{r}")
    ki, lam, S = eh.inner_select(cands, tr, y)
    p = eh.fit_predict(cands[ki], tr, te, y, lam)
    yt = y[te]
    P = {a: np.array(v) for a, v in prev["predictions"].items()}
    R = {"addendum": "E (second look at the held-out set)", "n_train": len(train), "n_holdout": len(hold),
         "chosen": {"candidate": labels[ki], "lambda": lam},
         "inner_cv_top5": sorted(((float(np.nanmax(S[i])), labels[i]) for i in range(len(labels))), reverse=True)[:5],
         "holdout_r_HYB": float(np.corrcoef(p, yt)[0, 1]),
         "holdout_r_prev": prev["holdout_r"],
         "primary_HYB_vs_FBERT_best": eh.boot_diff(p, P["FBERT_best"], yt),
         "HYB_vs_D1": eh.boot_diff(p, P["D1"], yt),
         "HYB_vs_A8cat": eh.boot_diff(p, P["A8cat"], yt),
         "holdout_ids": hold, "prediction_HYB": p.round(6).tolist(), "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("holdout_ids", "prediction_HYB")}, indent=1))


if __name__ == "__main__":
    main()
