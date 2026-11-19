"""Family-calibrated leak check for the held-out test (sensitivity analysis, run
after the Addendum D result; it re-scores the SAME frozen predictions, it does not
refit or choose anything).

The benchmark's splits are family-aware inside the 753: across 60 seeds a family
always sits in one fold. So a pair together in all 60 seeds is (almost surely) a
family pair; a pair separated in any seed is unrelated. That gives known relative
and known unrelated pairs, from which a relative detector on connectome similarity
is calibrated and applied to held-out-vs-training pairs.

usage: python3 holdout_family_check.py WORKSPACE HOLDOUT.json OUT.json TRAIN_REST HOLD_REST
"""
import csv, json, sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, "/workspace/fbx")
import eval_sixmod as sx

ws, hp, outp, trr, hor = sys.argv[1:6]
H = json.load(open(hp))
rest = sx.load_feats(trr, {"fc_full": "fc_rs"})
rest.update(sx.load_feats(hor, {"fc_full": "fc_rs"}))
s753 = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
train = [s for s in s753 if s in rest]
hold = H["holdout_ids"]
oi = {r["Subject"].strip(): float(r["varimax_cog"]) for r in csv.DictReader(open(ws + "/ooi_comp.csv"))
      if r["varimax_cog"] not in ("", "NaN")}
idx = {s: i for i, s in enumerate(train)}

# 1. families inside the 753 from the splits
folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
n = len(train)
together = np.zeros((n, n), np.int16)
names = list(folds._fieldnames)
for nm in names:
    lab = np.full(n, -1)
    for k, fold in enumerate(getattr(folds, nm).sub_fold):
        for s in fold.subject_list:
            if str(s) in idx:
                lab[idx[str(s)]] = k
    together += (lab[:, None] == lab[None, :])
related = (together == len(names))
np.fill_diagonal(related, False)
iu = np.triu_indices(n, 1)
rel_pairs = related[iu]

# 2. similarity: rest FC minus the training mean edge profile, cosine
F = np.stack([rest[s]["fc_rs"] for s in train + hold]).astype(np.float64)
F -= F[:n].mean(0)
F /= np.linalg.norm(F, axis=1, keepdims=True)
Stt = F[:n] @ F[:n].T
sim_rel = Stt[iu][rel_pairs]
sim_unrel = Stt[iu][~rel_pairs]
thr_999 = float(np.percentile(sim_unrel, 99.9))
thr_rel5 = float(np.percentile(sim_rel, 5)) if len(sim_rel) else None
# detection quality of the 99.9th-unrelated threshold on known pairs
sens = float((sim_rel > thr_999).mean()) if len(sim_rel) else None

# 3. held-out subjects with a likely relative in training
Sht = F[n:] @ F[:n].T
hmax = Sht.max(1)
likely_rel = hmax > thr_999
yt = np.array([oi[s] for s in hold])
P = {a: np.array(v) for a, v in H["predictions"].items()}


def r(p, m):
    return float(np.corrcoef(p[m], yt[m])[0, 1])


def boot(pa, pb, m, B=10000, seed=1):
    rs = np.random.RandomState(seed)
    ii = np.nonzero(m)[0]
    d = []
    for _ in range(B):
        j = rs.choice(ii, len(ii))
        d.append(np.corrcoef(pa[j], yt[j])[0, 1] - np.corrcoef(pb[j], yt[j])[0, 1])
    d = np.array(d)
    return {"diff": r(pa, m) - r(pb, m), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0": float((d > 0).mean()), "n": int(m.sum())}


subsets = {"all": np.ones(len(hold), bool), "no_likely_relative": ~likely_rel, "likely_relative": likely_rel}
# stricter: below the median of unrelated-pair maxima would be too strict to keep anyone; use quantiles of hmax
for q in [50, 25]:
    subsets[f"hmax_below_q{q}"] = hmax < np.percentile(hmax, q)
out = {"n_train": n, "n_holdout": len(hold),
       "family_pairs_in_753": int(rel_pairs.sum()), "unrelated_pairs": int((~rel_pairs).sum()),
       "sim_related_median": float(np.median(sim_rel)) if len(sim_rel) else None,
       "sim_unrelated_median": float(np.median(sim_unrel)), "sim_unrelated_p99.9": thr_999,
       "sim_related_p5": thr_rel5, "detector_sensitivity_on_known_family_pairs": sens,
       "held_out_with_likely_relative": int(likely_rel.sum()),
       "subsets": {}}
for name, m in subsets.items():
    if m.sum() < 30:
        out["subsets"][name] = {"n": int(m.sum()), "skipped": "too few"}
        continue
    out["subsets"][name] = {"n": int(m.sum()), "r": {a: r(p, m) for a, p in P.items()},
                            "D1_vs_A8cat": boot(P["D1"], P["A8cat"], m),
                            "D0_vs_A8cat": boot(P["D0"], P["A8cat"], m),
                            "H0_vs_A8cat": boot(P["H0"], P["A8cat"], m)}
# does our advantage grow with similarity to the training set? (per-subject squared-error gain)
gain = (P["A8cat"] - yt) ** 2 - (P["D1"] - yt) ** 2
out["corr_D1_gain_vs_hmax"] = float(np.corrcoef(gain, hmax)[0, 1])
gain0 = (P["A8cat"] - yt) ** 2 - (P["D0"] - yt) ** 2
out["corr_D0_gain_vs_hmax"] = float(np.corrcoef(gain0, hmax)[0, 1])
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(out, indent=1))
