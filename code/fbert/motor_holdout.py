"""Addendum S: drop-motor confirmation on subjects outside the 753 (Addendum D pipeline).

usage: python3 motor_holdout.py WORKSPACE OUT.json TRAIN_REST TRAIN_TASK HOLD_REST HOLD_TASK
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx
import eval_holdout as eh


def main():
    ws, outp, trr, trt, hor, hot = sys.argv[1:7]
    t0 = time.time()
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
    use = train + hold
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    tr = np.arange(len(train)); te = np.arange(len(train), len(use))
    print(f"train {len(train)} held-out {len(hold)}", flush=True)
    F = np.stack([rest[s]["fc_rs"] for s in use]).astype(np.float64)
    F = F - F[tr].mean(0); F = F / np.linalg.norm(F, axis=1, keepdims=True)
    Ctt = F[tr] @ F[tr].T; np.fill_diagonal(Ctt, -1)
    thr = float(np.percentile(Ctt.max(1), 90)); clean = (F[te] @ F[tr].T).max(1) < thr
    KI = {}; KT = {}
    for m in sx.MODS:
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
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    six = sx.MODS; five = [m for m in sx.MODS if m != "motor"]
    E = {m: np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64) for m in sx.MODS}

    def fb(mods):
        return [ev.zlin_kernel(np.concatenate([E[m] for m in mods], 1), tr),
                sum(ev.spectral_filter(ev.correlation_kernel(E[m])) for m in mods) / len(mods)]
    arms = {}
    for nm, mods in [("six", six), ("nomotor", five)]:
        arms[f"D0_{nm}"] = [sum(KI[(m, g)] for m in mods) / len(mods) for g in eh.GS]
        arms[f"D1_{nm}"] = [sum(KT[(m, g)] for m in mods) / len(mods) for g in eh.GS]
        arms[f"FBERT_{nm}"] = fb(mods)
    preds, chosen = {}, {}
    for a, Ks in arms.items():
        ki, lam, _ = eh.inner_select(Ks, tr, y)
        preds[a] = eh.fit_predict(Ks[ki], tr, te, y, lam)
        chosen[a] = {"kernel": ki, "lambda": lam}
    yt = y[te]
    B = lambda a, b, msk=slice(None): eh.boot_diff(preds[a][msk], preds[b][msk], yt[msk])
    R = {"addendum": "S (drop motor, subjects outside the 753)", "n_train": len(train), "n_holdout": len(hold),
         "n_holdout_leak_controlled": int(clean.sum()), "chosen": chosen,
         "holdout_r": {a: float(np.corrcoef(p, yt)[0, 1]) for a, p in preds.items()},
         "holdout_r_leak_controlled": {a: float(np.corrcoef(p[clean], yt[clean])[0, 1]) for a, p in preds.items()},
         "PRIMARY_D0_nomotor_vs_six": B("D0_nomotor", "D0_six"),
         "D1_nomotor_vs_six": B("D1_nomotor", "D1_six"),
         "FBERT_nomotor_vs_six": B("FBERT_nomotor", "FBERT_six"),
         "D0_nomotor_vs_FBERT_nomotor": B("D0_nomotor", "FBERT_nomotor"),
         "D1_nomotor_vs_FBERT_nomotor": B("D1_nomotor", "FBERT_nomotor"),
         "leak_controlled": {"D0": B("D0_nomotor", "D0_six", clean), "D1": B("D1_nomotor", "D1_six", clean),
                             "FBERT": B("FBERT_nomotor", "FBERT_six", clean)},
         "holdout_ids": hold, "predictions": {a: p.round(6).tolist() for a, p in preds.items()},
         "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("holdout_ids", "predictions")}, indent=1))


if __name__ == "__main__":
    main()
