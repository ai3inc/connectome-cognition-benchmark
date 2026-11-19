"""SOTA-claim review (2026-11-18): seed-count tests vs the corrected resampled t-test.

The project's decision rules count seeds (57/60) and report t over the 60 seed means.
Seeds re-partition the same 753 subjects, so those t values measure robustness to the
partition, not uncertainty over subjects (the same flaw as the retracted P = 1.000).
Ooi et al. (2022) and the 2026 competitor papers use the Nadeau-Bengio corrected
resampled t-test. Per-fold scores were not saved, so the correction is approximated from
per-seed scores, treating the k folds inside a seed as independent:
    var_fold ~= k * var_seed;  corrected var(mean) = (1/(k*J) + n_test/n_train) * var_fold
which divides the naive t by sqrt((1/(k*J) + 1/(k-1)) * k * J)  (= 8.2 for k=10, J=60).

usage: python3 code/fbert/corrected_t_review.py   -> results/REVIEW_CORRECTED_T_1118.json
"""
import json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
K = 10


def ld(f):
    return json.load(open(os.path.join(R, f)))


def factor(J, k=K):
    return float(np.sqrt((1 / (k * J) + 1 / (k - 1)) * k * J))


def from_seeds(a, b):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    tn = d.mean() / (d.std(ddof=1) / np.sqrt(J))
    tc = tn / factor(J)
    return {"diff": round(float(d.mean()), 4), "wins": f"{int((d > 0).sum())}/{J}", "naive_t": round(float(tn), 2),
            "corrected_t": round(float(tc), 2), "p_two_sided": float(f"{2 * stats.t.sf(abs(tc), J - 1):.3g}")}


def from_summary(diff, tn, J=60):
    tc = tn / factor(J)
    return {"diff": diff, "naive_t": tn, "corrected_t": round(tc, 2),
            "p_two_sided": float(f"{2 * stats.t.sf(abs(tc), J - 1):.3g}"), "note": "from the recorded t"}


out = {"date": "2026-11-18", "method": __doc__.strip().splitlines()[0], "factor_k10_J60": round(factor(60), 2)}
s = ld("FBERT_SAMEFOLDS_1117.json")["per_seed"]
x = ld("FBERT_SIXMOD_1117.json")["per_seed"]
six = ld("FBERT_SOTA_HCP_six_1117.json")["cells"]["six"]["per_seed"]
rest = ld("FBERT_SOTA_HCP_rest_1117.json")["cells"]["rest"]["per_seed"]
j = ld("FBERT_JOINT_NESTED_1117.json")["per_seed"]
pr = ld("PRIORI_SHARE_SERIES_1117.json")["results_60seeds"]
rp = ld("REPLIT_RANDOM_SPLITS_1117.json")["results"]
em = ld("EMOTION_DEFICIT_CONFIRMED_1117.json")
out["from_per_seed"] = {
    "rest_D0_vs_FBERT8M": from_seeds(s["D0_full"], s["A8P1_full"]),
    "rest_D0_vs_FC035_teacher": from_seeds(s["D0_full"], s["C_full"]),
    "six_D1_lang2_vs_FBERT8M_lang2": from_seeds(x["D16_lang2"], x["A86_lang2"]),
    "six_D1_eq_vs_FBERT8M_eq": from_seeds(x["D16_eq"], x["A86_eq"]),
    "six_D0_eq_vs_FBERT8M_eq_inductive": from_seeds(x["D06_eq"], x["A86_eq"]),
    "six_D0_lang2_vs_FBERT8M_lang2_inductive": from_seeds(x["D06_lang2"], x["A86_lang2"]),
    "six_OURS_system_vs_FBERT8M": from_seeds(six["OURS"], six["A8"]),
    "six_SOTA_system_incl_FBERT_vs_FBERT8M": from_seeds(six["SOTA"], six["A8"]),
    "rest_OURS_system_vs_FBERT8M": from_seeds(rest["OURS"], rest["A8"]),
    "joint_0.7481_vs_nested_canonical": from_seeds(j["PRED50nest"], j["CANONnest"]),
}
out["from_recorded_t"] = {
    "REL_1of9_0.7382_vs_0.7357": from_summary(pr["w_1of9"]["diff"], pr["w_1of9"]["t"]),
    "REL_1of8": from_summary(pr["w_1of8"]["diff"], pr["w_1of8"]["t"]),
    "REL_1of9_random_splits": from_summary(rp["cognition_on_INDEPENDENT_random_splits"]["diff"],
                                           rp["cognition_on_INDEPENDENT_random_splits"]["t"]),
    "REL_satisfaction_random_splits": from_summary(rp["satisfaction_on_random_splits"]["diff"],
                                                   rp["satisfaction_on_random_splits"]["t"], 30),
    "REL_emotion_random_splits": from_summary(rp["emotion_recognition_on_random_splits"]["diff"],
                                              rp["emotion_recognition_on_random_splits"]["t"], 30),
}
for k, v in em["results_60seeds_family_aware_folds"].items():
    out["from_recorded_t"][f"REL_{k}_family_folds"] = from_summary(v["diff"], v["t"])
out["reading"] = ("Survive the correction: D0 vs F-BERT-8M on rest, D1 vs F-BERT-8M on six modalities, "
                  "the joint over the nested canonical. Do not survive: every RELATIONAL share effect "
                  "(0.7382 included), the target-specificity contrasts, and the inductive six-modality "
                  "D0 vs F-BERT gap (a tie, not an F-BERT win). Approximate: per-fold scores were not saved.")
json.dump(out, open(os.path.join(R, "REVIEW_CORRECTED_T_1118.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
