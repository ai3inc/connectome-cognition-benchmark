"""Addendum N on the single-scan cohorts. Three arms, each chosen in-fold:
  A8   = F-BERT-8M: correlation kernel / in-fold z-linear, x {noGSR, GSR}
  OURS = closed-form: D0 (g grid) and FC^0.35, x {noGSR, GSR}
  SOTA = A8 + OURS + hybrids (1-rho) OURS_k + rho F-BERT-corr (same variant), rho in .25/.5/.75

usage: python3 sota_single.py COHORT PHENO FEAT_GLOB OUT.json
  COHORT: aomic | abide | adhd | adhd_nyu
"""
import csv, glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import aomic_eval as ae
from abide_eval import auc, site_folds

RHOS = [0.25, 0.5, 0.75]
G = {}


def load(pat):
    D = {}
    for f in sorted(glob.glob(pat)):
        with np.load(f) as d:
            a = {k: d[k] for k in d.files}
        for i, s in enumerate(a["sids"]):
            D[str(s)] = {k: np.array(v[i]) for k, v in a.items() if k != "sids"}
    return D


def folds_for(task, rep):
    c = G["cohort"]; n = len(G[task]["y"])
    if c in ("aomic", "adhd_nyu"):
        base = 7000 if c == "aomic" else 500
        perm = np.random.RandomState(base + rep).permutation(n)
        return np.array_split(perm, 10)
    if c == "abide":
        return site_folds(G[task]["sites"], rep)
    sites = G[task]["sites"]                      # adhd: leave one site out
    return [np.array([i for i in range(n) if sites[i] == s]) for s in sorted(set(sites))]


def run(args):
    task, rep = args
    with threadpool_limits(1):
        y, idx = G[task]["y"], G[task]["idx"]
        n = len(y)
        arms_out = ["A8", "A1", "OURS", "SOTA"] if G.get("inc1m") else ["A8", "OURS", "SOTA"]
        if G.get("cb_corr"):
            arms_out = arms_out + ["CB", "ENC"]
        pred = {a: np.full(n, np.nan) for a in arms_out}
        pick = []
        for te in folds_for(task, rep):
            tr = np.setdiff1d(np.arange(n), te)
            sub = lambda K: K[np.ix_(idx, idx)]
            A8 = [sub(K) for K in G["fb_corr"]] + [ev.zlin_kernel(E[idx], tr) for E in G["fb_emb"]]
            OURS = [sub(K) for K in G["ours"]]
            HYB = [sub((1 - r) * K + r * G["fb_corr"][G["ours_var"][k]]) for k, K in enumerate(G["ours"]) for r in RHOS]
            if G.get("inc1m"):
                A1 = [sub(K) for K in G["fb1_corr"]] + [ev.zlin_kernel(E[idx], tr) for E in G["fb1_emb"]]
                HYB += [sub((1 - r) * K + r * G["fb1_corr"][G["ours_var"][k]]) for k, K in enumerate(G["ours"]) for r in RHOS]
                HYB += [sub((1 - r) * G["fb_corr"][v] + r * G["fb1_corr"][v]) for v in range(2) for r in RHOS]
                pred["A1"][te] = ev.krr_select(A1, tr, te, y)[0]
                A8 = A8 + A1          # SOTA candidate pool gets 1M; the A8 arm below stays 8M-only
                pred["A8"][te] = ev.krr_select(A8[:4], tr, te, y)[0]
            else:
                pred["A8"][te] = ev.krr_select(A8, tr, te, y)[0]
            pred["OURS"][te] = ev.krr_select(OURS, tr, te, y)[0]
            if G.get("cb_corr"):
                CB = [sub(K) for K in G["cb_corr"]] + [ev.zlin_kernel(E[idx], tr) for E in G["cb_emb"]]
                pred["CB"][te] = ev.krr_select(CB, tr, te, y)[0]
                for j, Kc in (enumerate(G["cb_corr"]) if os.environ.get("CB_HYB", "1") == "1" else []):
                    HYB = HYB + [sub((1 - r) * K + r * Kc) for k, K in enumerate(G["ours"]) if G["ours_var"][k] == G["cb_var"][j] for r in RHOS]
                ENCP = [sub((1 - r) * A + r * B) for i, (va, A) in enumerate(G["enc_corr"]) for (vb, B) in G["enc_corr"][i + 1:] if va == vb for r in RHOS]
                ENC = A8 + CB + ENCP
                pred["ENC"][te] = ev.krr_select(ENC, tr, te, y)[0]
                HYB = HYB + ENCP
                A8 = A8 + CB
            p, kb, _ = ev.krr_select(A8 + OURS + HYB, tr, te, y)
            pred["SOTA"][te] = p
            pick.append("A8" if kb < len(A8) else ("OURS" if kb < len(A8) + len(OURS) else "HYB"))
        metric = G[task]["metric"]
        sc = {a: (auc(v, y) if metric == "auc" else float(np.corrcoef(v, y)[0, 1])) for a, v in pred.items()}
        return task, rep, sc, pick, {a: v.tolist() for a, v in pred.items()}


def boot(pa, pb, y, metric, B=10000):
    rs = np.random.RandomState(1); n = len(y); d = []
    for _ in range(B):
        j = rs.randint(0, n, n)
        if metric == "auc" and len(set(y[j])) < 2:
            continue
        f = (lambda p: auc(p[j], y[j])) if metric == "auc" else (lambda p: np.corrcoef(p[j], y[j])[0, 1])
        d.append(f(pa) - f(pb))
    d = np.array(d)
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_gt_0": float((d > 0).mean())}


def main():
    cohort, php, fglob, outp = sys.argv[1:5]
    t0 = time.time()
    F = load(fglob)
    G["cohort"] = cohort
    if cohort == "aomic":
        parts = {r["participant_id"]: r for r in csv.DictReader(open(php), delimiter="\t")}
        def num(v):
            try:
                x = float(v); return x if x == x else None
            except Exception:
                return None
        use = sorted(s for s in F if s in parts and num(parts[s].get("IST_intelligence_total")) is not None and F[s]["n_empty"] == 0)
        tasks = {"ist": (np.array([num(parts[s]["IST_intelligence_total"]) for s in use]), np.arange(len(use)), None, "r")}
    elif cohort == "abide":
        ph = {r["FILE_ID"]: r for r in csv.DictReader(open(php))}
        use = sorted(s for s in F if s in ph and ph[s]["DX_GROUP"] in ("1", "2") and F[s]["n_empty"] == 0)
        sites = [ph[s]["SITE_ID"] for s in use]
        dx = np.array([1.0 if ph[s]["DX_GROUP"] == "1" else -1.0 for s in use])
        fiq = np.array([float(ph[s]["FIQ"]) if ph[s]["FIQ"] not in ("", "-9999", "-9999.0") else np.nan for s in use])
        fi = np.nonzero(np.isfinite(fiq) & (fiq > 0))[0]
        tasks = {"dx": (dx, np.arange(len(use)), sites, "auc"), "fiq": (fiq[fi], fi, [sites[i] for i in fi], "r")}
    else:
        ph = json.load(open(php))
        use = sorted(s for s in F if s in ph and F[s]["n_empty"] == 0 and (cohort == "adhd" or ph[s]["site"] == "NYU"))
        sites = [ph[s]["site"] for s in use]
        dx = np.array([-1.0 if "Typically" in ph[s]["dx"] else 1.0 for s in use])
        def iqv(x):
            try:
                v = float(x); return v if 40 < v < 200 else np.nan
            except Exception:
                return np.nan
        iq = np.array([iqv(ph[s]["iq"]) for s in use]); fi = np.nonzero(np.isfinite(iq))[0]
        tasks = {"dx": (dx, np.arange(len(use)), sites, "auc"), "iq": (iq[fi], fi, [sites[i] for i in fi], "r")}
    print(cohort, "subjects", len(use), flush=True)
    ours, ours_var, fb_corr, fb_emb = [], [], [], []
    for vi, v in enumerate(["nogsr", "gsr"]):
        fc = np.stack([F[s][f"fc_{v}"] for s in use])
        with Pool(8) as pool:
            res = pool.map(ae.per_subject_g, list(fc), chunksize=8)
        Kf = ev.correlation_kernel(fc)
        for gi in range(len(ae.GS)):
            ours.append(0.95 * ev.correlation_kernel(np.stack([r[1][gi] for r in res])) + 0.05 * Kf); ours_var.append(vi)
        ours.append(ev.correlation_kernel(np.stack([r[0] for r in res]))); ours_var.append(vi)
        E = np.stack([F[s][f"e8_{v}"] for s in use]).astype(np.float64)
        fb_corr.append(ev.correlation_kernel(E)); fb_emb.append(E)
        if os.environ.get("INCLUDE_1M"):
            E1 = np.stack([F[s][f"e1_{v}"] for s in use]).astype(np.float64)
            G.setdefault("fb1_corr", []).append(ev.correlation_kernel(E1)); G.setdefault("fb1_emb", []).append(E1)
        print("  kernels", v, f"{time.time() - t0:.0f}s", flush=True)
    G.update(ours=ours, ours_var=ours_var, fb_corr=fb_corr, fb_emb=fb_emb, inc1m=bool(os.environ.get("INCLUDE_1M")))
    if os.environ.get("EXTRA"):
        G["cb_corr"], G["cb_emb"], G["cb_var"] = [], [], []
        G["enc_corr"] = [(vi, K) for vi, K in enumerate(fb_corr)]     # (variant, kernel) for encoder pairs
        for spec in os.environ["EXTRA"].split(","):
            path, k = spec.split(":")
            X = np.load(path); ids = {str(i): j for j, i in enumerate(X["ids"])}; Ek = X[f"e{k}"]
            for vi, v in enumerate(["nogsr", "gsr"]):
                E = np.stack([Ek[ids[f"{s_}_{v}"]] for s_ in use]).astype(np.float64)
                K = ev.correlation_kernel(E)
                G["cb_corr"].append(K); G["cb_emb"].append(E); G["cb_var"].append(vi)
                G["enc_corr"].append((vi, K))
            print("extra encoder", path, k, flush=True)
    for t, (y, idx, st, metric) in tasks.items():
        G[t] = {"y": y, "idx": idx, "sites": st, "metric": metric}
    reps = 1 if cohort == "adhd" else 20
    jobs = [(t, r) for t in tasks for r in range(reps)]
    with Pool(8) as pool:
        out = pool.map(run, jobs)
    R = {"addendum": "N (SOTA system)", "cohort": cohort, "n": len(use), "cells": {}}
    for t in tasks:
        rows = [o for o in out if o[0] == t]
        arm_names = list(rows[0][2].keys())
        sc = {a: np.array([o[2][a] for o in rows]) for a in arm_names}
        picks = sum((o[3] for o in rows), [])
        cell = {"metric": tasks[t][3], "mean": {a: float(v.mean()) for a, v in sc.items()},
                "picks": {k: picks.count(k) for k in ["A8", "OURS", "HYB"]}}
        if reps > 1:
            for b in [a for a in arm_names if a != "SOTA"]:
                x = sc["SOTA"] - sc[b]
                cell[f"SOTA_vs_{b}"] = {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x)}
        else:
            P = {a: np.array(rows[0][4][a]) for a in arm_names}
            for b in [a for a in arm_names if a != "SOTA"]:
                cell[f"SOTA_vs_{b}"] = {"mean_diff": float(cell["mean"]["SOTA"] - cell["mean"][b]),
                                        **boot(P["SOTA"], P[b], tasks[t][0], tasks[t][3])}
            if "CB" in P:   # encoder-vs-encoder and encoder-vs-ours, same predictions (leave-one-site-out)
                for b in ["A8", "OURS"]:
                    cell[f"CB_vs_{b}"] = {"mean_diff": float(cell["mean"]["CB"] - cell["mean"][b]),
                                          **boot(P["CB"], P[b], tasks[t][0], tasks[t][3])}
            if os.environ.get("SAVE_PRED"):
                cell["predictions"] = {a: v.round(6).tolist() for a, v in P.items()}
        R["cells"][t] = cell
        print(t, json.dumps(cell), flush=True)
    R["seconds"] = time.time() - t0
    json.dump(R, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
