"""Exact Nadeau-Bengio corrected t from the per-fold scores written by fold_scores.py (review, 2026-11-19).

Per arm: mean of per-fold scores (the benchmark's metric), split-level SD, and for each
pair the corrected resampled t over all paired fold differences,
    t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)),
with n_test/n_train the mean over folds (from the recorded fold sizes, or --ratio).

usage: python3 fold_nb.py FOLD_DIR TAG OUT.json [--ratio R]
"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats

PAIRS = {
    "samefolds": [("D0_full", "A8P1_full"), ("D0_full", "C_full"), ("D1_full", "A8P1_full"), ("D0_full", "A1P1_full"),
                  ("C_full", "A8P1_full"), ("D1_full", "D0_full"), ("B_full", "A8P1_full"), ("D0_full", "B_full")],
    "sixmod": [("D06_eq", "A8cat"), ("D16_eq", "A8cat"), ("C6_eq", "A8cat"), ("D16_eq", "D06_eq"), ("D06_eq", "C6_eq"),
               ("D06_lang2", "A8cat"), ("D16_lang2", "A8cat"), ("D06_eq", "B6_eq"), ("D16_eq", "B6_eq")],
    "aomic": [("D0", "A8"), ("D1", "A8"), ("C", "A8"), ("D0", "C"), ("D1", "D0")],
    "abide_dx": [("D0", "A8"), ("D1", "A8"), ("C", "A8"), ("D0", "C"), ("D1", "D0")],
    "abide_fiq": [("D0", "A8"), ("D1", "A8"), ("C", "A8"), ("D0", "C"), ("D1", "D0")],
}


def main():
    fold_dir, tag, outp = sys.argv[1:4]
    ratio_arg = float(sys.argv[sys.argv.index("--ratio") + 1]) if "--ratio" in sys.argv else None
    files = sorted(glob.glob(os.path.join(fold_dir, f"{tag}_*.json")),
                   key=lambda f: (len(f), f))
    files = [f for f in files if os.path.basename(f)[len(tag) + 1:-5].replace("_", "").isalnum()
             and not os.path.basename(f)[len(tag) + 1:].startswith(("dx_", "fiq_"))]
    recs = [json.load(open(f)) for f in files]
    arms = list(recs[0]["fold_scores"])
    S = {a: np.array([[np.nan if x is None else x for x in r["fold_scores"][a]] for r in recs], dtype=float) for a in arms}
    sizes = [s for r in recs if r.get("test_sizes") for s in r["test_sizes"]]
    ratio = ratio_arg if ratio_arg is not None else float(np.mean([te / tr for te, tr in sizes]))
    R = {"tag": tag, "splits": len(recs), "folds_per_split": int(S[arms[0]].shape[1]), "test_train_ratio": round(ratio, 4),
         "ratio_source": "--ratio" if ratio_arg is not None else "recorded fold sizes",
         "mean_perfold": {a: round(float(np.nanmean(v)), 4) for a, v in S.items()},
         "split_sd": {a: round(float(np.nanmean(v, 1).std()), 4) for a, v in S.items()}, "NB": {}}
    for a, b in PAIRS.get(tag, []):
        if a not in S or b not in S:
            continue
        d = (S[a] - S[b]).ravel()
        d = d[np.isfinite(d)]
        J = len(d)
        tc = d.mean() / math.sqrt((1.0 / J + ratio) * d.var(ddof=1))
        tn = d.mean() / (d.std(ddof=1) / math.sqrt(J))
        dm = np.nanmean(S[a] - S[b], 1)
        R["NB"][f"{a}_vs_{b}"] = {"diff": round(float(d.mean()), 4), "folds": J, "naive_t": round(float(tn), 2),
                                  "corrected_t": round(float(tc), 2), "p_two_sided": float(f"{2 * stats.t.sf(abs(tc), J - 1):.3g}"),
                                  "splits_positive": int((dm > 0).sum()), "splits": int(len(dm))}
    R["per_fold"] = {a: np.where(np.isfinite(v), v, None).tolist() for a, v in S.items()}
    json.dump(R, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "per_fold"}, indent=1))


if __name__ == "__main__":
    main()
