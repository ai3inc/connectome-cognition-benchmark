"""The chain against the benchmark's own kernel on the 263 new subjects
(PREREG_CHAIN_VS_OOI_KERNEL_263.md, committed before this run).

Reuses eval_holdout.py (Addendum D) unchanged for loading, the chain kernels, inner
selection, fitting and the paired bootstrap; adds three baseline arms. D1 and D0 must
reproduce Addendum D (0.7863 / 0.7696, +-0.0005) before any contrast is read.

usage: python3 eval_holdout_chain.py WORKSPACE OUT.json TRAIN_REST TRAIN_TASK HOLD_REST HOLD_TASK
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx
import eval_holdout as eh

GATE = {"D1": 0.786310, "D0": 0.769641}
TOL = 5e-4


def cbig_corr_kernel(X):
    """CBIG 'corr' kernel: Pearson correlation between two subjects' edge vectors."""
    X = np.asarray(X, dtype=np.float64)
    Xr = X - X.mean(1, keepdims=True)
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    return Xr @ Xr.T


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
    F = F - F[tr].mean(0)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    Ctt = F[tr] @ F[tr].T
    np.fill_diagonal(Ctt, -1)
    thr = float(np.percentile(Ctt.max(1), 90))
    clean = (F[te] @ F[tr].T).max(1) < thr
    del F, Ctt

    KI = {g: 0 for g in eh.GS}; KT = {g: 0 for g in eh.GS}
    KC = 0; KPT = 0; KPI = 0
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(eh.partial_edges, list(fc), chunksize=8)
        Kf_i = eh.inductive_corr_kernel(fc.astype(np.float64), tr)
        Kf_t = ev.correlation_kernel(fc)
        for gi, g in enumerate(eh.GS):
            Xp = np.stack([r[gi] for r in res])
            KI[g] = KI[g] + (1 - ev.W_FULL) * eh.inductive_corr_kernel(Xp.astype(np.float64), tr) + ev.W_FULL * Kf_i
            KT[g] = KT[g] + (1 - ev.W_FULL) * ev.spectral_filter(ev.correlation_kernel(Xp)) + ev.W_FULL * Kf_t
        KC = KC + cbig_corr_kernel(fc)
        KPT = KPT + Kf_t
        KPI = KPI + Kf_i
        del fc, res
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    KI = {g: K / 6 for g, K in KI.items()}; KT = {g: K / 6 for g, K in KT.items()}
    arms = {"D1": [KT[g] for g in eh.GS], "D0": [KI[g] for g in eh.GS],
            "CBIG6": [KC / 6], "PLAIN_T": [KPT / 6], "PLAIN_I": [KPI / 6]}
    yt = y[te]
    preds, chosen = {}, {}

    def run(a):
        ki, lam, _ = eh.inner_select(arms[a], tr, y)
        preds[a] = eh.fit_predict(arms[a][ki], tr, te, y, lam)
        chosen[a] = {"kernel": ki, "lambda": lam}

    for a in ("D1", "D0"):
        run(a)
    gate = {a: {"r": float(np.corrcoef(preds[a], yt)[0, 1]), "addendum_D": GATE[a]} for a in GATE}
    for a in gate:
        gate[a]["ok"] = abs(gate[a]["r"] - GATE[a]) <= TOL
    R = {"prereg": "PREREG_CHAIN_VS_OOI_KERNEL_263.md", "n_train": len(train), "n_holdout": len(hold),
         "n_holdout_leak_controlled": int(clean.sum()), "sanity_gate": gate,
         "gate_passed": all(v["ok"] for v in gate.values())}
    if not R["gate_passed"]:
        R["verdict"] = "VOID: D1/D0 did not reproduce Addendum D; no contrast read"
        json.dump(R, open(outp, "w"), indent=1)
        print(json.dumps(R, indent=1))
        return
    for a in ("CBIG6", "PLAIN_T", "PLAIN_I"):
        run(a)
    prim = eh.boot_diff(preds["D1"], preds["CBIG6"], yt)
    pairs = {"D0_vs_CBIG6": ("D0", "CBIG6"), "D1_vs_PLAIN_T": ("D1", "PLAIN_T"),
             "D0_vs_PLAIN_I": ("D0", "PLAIN_I"), "D1_vs_D0": ("D1", "D0"), "D1_vs_CBIG6": ("D1", "CBIG6")}
    R.update({
        "chosen": chosen,
        "holdout_r": {a: float(np.corrcoef(p, yt)[0, 1]) for a, p in preds.items()},
        "holdout_r_leak_controlled": {a: float(np.corrcoef(p[clean], yt[clean])[0, 1]) for a, p in preds.items()},
        "primary_D1_vs_CBIG6": prim,
        "verdict": "CONFIRMED" if prim["ci95"][0] > 0 else "NOT CONFIRMED",
        "secondary": {k: eh.boot_diff(preds[a], preds[b], yt) for k, (a, b) in pairs.items() if k != "D1_vs_CBIG6"},
        "secondary_leak_controlled": {k: eh.boot_diff(preds[a][clean], preds[b][clean], yt[clean])
                                      for k, (a, b) in pairs.items()},
        "holdout_ids": hold, "predictions": {a: p.round(6).tolist() for a, p in preds.items()},
        "seconds": time.time() - t0})
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("holdout_ids", "predictions")}, indent=1))


if __name__ == "__main__":
    main()
