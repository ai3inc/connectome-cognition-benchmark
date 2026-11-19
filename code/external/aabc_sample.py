"""Aggregate description of the AABC sample analysed by ext_eval.py (no subject-level output).

Applies ext_eval.py's inclusion rule unchanged (V1 visit with complete FC, finite target and age,
sex M/F, site present) and writes the parcel and edge counts, subject counts, the age range, mean and SD, and the share of women.

usage: python3 -I aabc_sample.py FC.npz PHENO.csv OUT.json
"""
import csv, hashlib, json, math, os, sys
import numpy as np

TARGET = "tlbx_ctc_uncorrected_standard_score"


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    fc_p, ph, outp = sys.argv[1:4]
    F = np.load(fc_p, allow_pickle=False)
    ids, complete = F["ids"].astype(str), F["complete"]
    rows = {r["id_event"]: r for r in csv.DictReader(open(ph, encoding="utf-8-sig"))}
    age, sex, site, fam = [], [], [], []
    for i, s in enumerate(ids):
        r = rows.get(s)
        if r is None or not complete[i]:
            continue
        try:
            yt, ag = float(r.get(TARGET, "")), float(r.get("age_open", ""))
            assert math.isfinite(yt) and math.isfinite(ag) and r.get("sex") in ("M", "F") and r.get("site", "").strip()
        except (ValueError, AssertionError):
            continue
        age.append(ag); sex.append(r["sex"]); site.append(r["site"].strip()); fam.append(r.get("pedid", "").strip() or "solo_" + s)
    age = np.array(age)
    out = {"fc_file": os.path.basename(fc_p), "fc_sha16": sha16(fc_p), "pheno_file": os.path.basename(ph), "pheno_sha16": sha16(ph),
           "rule": "ext_eval.py inclusion rule", "parcels": int(len(F["parcels"])), "edges": int(F["edges"].shape[1]), "n": int(len(age)), "families": int(len(set(fam))),
           "sites": {s: int(site.count(s)) for s in sorted(set(site))},
           "age_min": float(age.min()), "age_max": float(age.max()), "age_mean": round(float(age.mean()), 1),
           "age_sd": round(float(age.std(ddof=1)), 1), "female_share": round(sex.count("F") / len(sex), 3)}
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
