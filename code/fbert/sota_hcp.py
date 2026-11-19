"""Addendum N on HCP (753, the benchmark's 60 x 10 family-aware folds).
cell rest: OURS = D0 (7 g) + FC^0.35 + D1 (7 g); A8 = F-BERT corr / in-fold z-linear;
           SOTA = A8 + OURS + (1-rho) OURS_k + rho F-BERT-corr.
cell six : OURS = D0-6 (g 0.95/0.97/0.98, equal) + D1-6 (g 0.85/0.90/0.95 x equal/lang2);
           A8 = F-BERT six corr-kernel mean / six concatenated in-fold z-linear;
           SOTA = A8 + OURS + (1-rho) OURS_k + rho F-BERT (corr-mean and z-linear).

usage: python3 sota_hcp.py WS REST_GLOB TASK_GLOB OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

RHOS = [0.25, 0.5, 0.75]
G = {}


def per_subject_grid(args):
    v, grid = args
    with threadpool_limits(1):
        C = ev.to_mat(v.astype(np.float64))
        w, V = np.linalg.eigh(C)
        p035 = ((V * np.clip(w, 0, None) ** 0.35) @ V.T)[ev.IU].astype(np.float32)
        parts = []
        for g in grid:
            P = np.linalg.inv((1 - g) * C + g * np.eye(ev.N_REG))
            d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            P = -P / np.outer(d, d); np.fill_diagonal(P, 1.0)
            wa, Va = np.linalg.eigh(P)
            parts.append(((Va * np.clip(wa, 0, None) ** ev.ALPHA) @ Va.T)[ev.IU].astype(np.float32))
        return p035, parts


def run(args):
    cell, nm = args
    with threadpool_limits(1):
        y, fl = G["y"], G["folds"][nm]
        c = G[cell]; n = len(y)
        arms_out = ["A8", "OURS", "SOTA"] + (["CB"] if "cb_corr" in c else [])
        pred = {a: np.full(n, np.nan) for a in arms_out}
        pick = []
        for te in fl:
            tr = np.setdiff1d(np.arange(n), te)
            Z = ev.zlin_kernel(c["emb"], tr)
            Zs = Z / np.mean(np.diag(Z)[tr]) / n * np.mean(np.diag(c["fb_corr"])[tr]) * n   # same scale as corr kernel
            A8 = [c["fb_corr"], Z]
            fbs = [c["fb_corr"]] + ([Zs] if cell == "six" else [])
            HYB = [(1 - r) * K + r * F for K in c["ours"] for F in fbs for r in RHOS]
            pred["A8"][te] = ev.krr_select(A8, tr, te, y)[0]
            pred["OURS"][te] = ev.krr_select(c["ours"], tr, te, y)[0]
            if "cb_corr" in c:
                Zc = ev.zlin_kernel(c["cb_emb"], tr)
                CB = [c["cb_corr"], Zc]
                pred["CB"][te] = ev.krr_select(CB, tr, te, y)[0]
                Zcs = Zc * np.mean(np.diag(c["cb_corr"])[tr]) / np.mean(np.diag(Zc)[tr])
                HYB = HYB + [(1 - r) * K + r * F for K in c["ours"] for F in ([c["cb_corr"]] + ([Zcs] if cell == "six" else [])) for r in RHOS]
                A8 = A8 + CB
            p, kb, _ = ev.krr_select(A8 + c["ours"] + HYB, tr, te, y)
            pred["SOTA"][te] = p
            pick.append("A8" if kb < 2 else ("OURS" if kb < 2 + len(c["ours"]) else "HYB"))
        return cell, nm, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}, pick


def main():
    ws, rglob, tglob, outp = sys.argv[1:5]
    t0 = time.time()
    rest = sx.load_feats(rglob, {"fc_full": "fc_rs", "e8_full": "e8_rs"})
    task = sx.load_feats(tglob, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan")) and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:60]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    CELL = os.environ.get("CELL", "both")
    # --- rest
    grid_r = [0.85, 0.9, 0.95]      # the range the in-fold selector used (main comparison)
    fc = np.stack([rest[s]["fc_rs"] for s in use])
    with Pool(8) as pool:
        res = pool.map(per_subject_grid, [(v, grid_r) for v in fc], chunksize=8)
    Kf = ev.correlation_kernel(fc)
    ours_r = [ev.correlation_kernel(np.stack([r[0] for r in res]))]
    for gi in range(len(grid_r)):
        Kp = ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
        ours_r += [0.95 * Kp + 0.05 * Kf, 0.95 * ev.spectral_filter(Kp) + 0.05 * Kf]
    Er = np.stack([rest[s]["e8_rs"] for s in use]).astype(np.float64)
    G["rest"] = {"ours": ours_r, "fb_corr": ev.correlation_kernel(Er), "emb": Er}
    print("rest kernels", f"{time.time() - t0:.0f}s", flush=True)
    XR = XT = None
    if os.environ.get("EXTRA_REST"):
        pr, kr = os.environ["EXTRA_REST"].split(":"); XR = np.load(pr); idr = {str(i): j for j, i in enumerate(XR["ids"])}
        Ecb = np.stack([XR[f"e{kr}"][idr[s_]] for s_ in use]).astype(np.float64)
        G["rest"]["cb_corr"] = ev.correlation_kernel(Ecb); G["rest"]["cb_emb"] = Ecb
    cells = ["rest", "six"] if CELL == "both" else [CELL]
    # --- six modalities
    if "six" not in cells:
        G.update(y=y, folds=fl)
        return finish(cells, names, outp, t0)
    g_d0, g_d1 = [0.97], [0.85, 0.9, 0.95]          # in-fold picks in addenda A-C
    grid6 = sorted(set(g_d0 + g_d1))
    D0s = {g: 0 for g in g_d0}; KM = []
    Es = []
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fcm = np.stack([src[s][f"fc_{m}"] for s in use])
        with Pool(8) as pool:
            rm = pool.map(per_subject_grid, [(v, grid6) for v in fcm], chunksize=8)
        Kfm = ev.correlation_kernel(fcm)
        kp = {g: ev.correlation_kernel(np.stack([r[1][grid6.index(g)] for r in rm])) for g in grid6}
        for g in g_d0:
            D0s[g] = D0s[g] + (0.95 * kp[g] + 0.05 * Kfm) / 6
        KM.append({g: 0.95 * ev.spectral_filter(kp[g]) + 0.05 * Kfm for g in g_d1})
        Es.append(np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64))
        del fcm, rm
        print("  six", m, f"{time.time() - t0:.0f}s", flush=True)
    ours6 = [D0s[g] for g in g_d0]
    for w in ([1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1]):
        for g in g_d1:
            ours6.append(sum(w[i] * KM[i][g] for i in range(6)) / sum(w))
    G["six"] = {"ours": ours6, "fb_corr": sum(ev.correlation_kernel(E) for E in Es) / 6, "emb": np.concatenate(Es, 1)}
    if os.environ.get("EXTRA_REST") and os.environ.get("EXTRA_TASK"):
        pt, kt = os.environ["EXTRA_TASK"].split(":"); XT = np.load(pt); idt = {str(i): j for j, i in enumerate(XT["ids"])}
        pr, kr = os.environ["EXTRA_REST"].split(":")
        Ec = [np.stack([XR[f"e{kr}"][idr[s_]] for s_ in use]).astype(np.float64)]
        for m in sx.MODS[1:]:
            Ec.append(np.stack([XT[f"e{kt}"][idt[f"{s_}_{m}"]] for s_ in use]).astype(np.float64))
        G["six"]["cb_corr"] = sum(ev.correlation_kernel(E) for E in Ec) / 6
        G["six"]["cb_emb"] = np.concatenate(Ec, 1)
    G.update(y=y, folds=fl)
    return finish(cells, names, outp, t0)


def finish(cells, names, outp, t0):
    jobs = [(c, nm) for c in cells for nm in names]
    with Pool(8) as pool:
        out = pool.map(run, jobs)
    R = {"addendum": "N (SOTA system), HCP", "n": len(G["y"]), "cells": {}}
    for c in cells:
        rows = [o for o in out if o[0] == c]
        an = list(rows[0][2].keys())
        sc = {a: np.array([o[2][a] for o in rows]) for a in an}
        picks = sum((o[3] for o in rows), [])
        cell = {"mean": {a: float(v.mean()) for a, v in sc.items()},
                "picks": {k: picks.count(k) for k in ["A8", "OURS", "HYB"]}}
        for b in [a for a in an if a != "SOTA"]:
            x = sc["SOTA"] - sc[b]
            cell[f"SOTA_vs_{b}"] = {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x),
                                    "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
        cell["per_seed"] = {a: v.round(6).tolist() for a, v in sc.items()}
        R["cells"][c] = cell
        print(c, json.dumps({k: v for k, v in cell.items() if k != "per_seed"}), flush=True)
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
