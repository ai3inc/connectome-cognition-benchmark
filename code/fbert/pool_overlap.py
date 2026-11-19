"""Count how many of the pre-registered new subjects were in the early development cohorts (review, 2026-11-19).

The early development runs (results/*_1115.json, n = 974) used pool.npz: subjects with resting-state FC and the
benchmark's cognition score, i.e. 728 of the 753 plus subjects outside the benchmark. This script intersects the
pool's subjects outside the 753 with the 263 new subjects of the pre-registered test. Only counts are written.

usage: python3 pool_overlap.py POOL.npz OFF753.txt OOI_COMP.csv HOLDOUT_IDS.json OUT.json
"""
import csv, json, sys
import numpy as np

pool_p, off_p, comp_p, hold_p, out_p = sys.argv[1:6]
z = np.load(pool_p, allow_pickle=True)
key = [k for k in z.files if k in ("subs", "sids", "subjects", "ids", "subject_ids")][0]
pool = {str(s).strip() for s in z[key]}
off = {x.strip() for x in open(off_p) if x.strip()}
comp = {r["Subject"].strip() for r in csv.DictReader(open(comp_p)) if (r.get("varimax_cog") or "").strip() not in ("", "nan", "NaN")}
hold = {str(s) for s in json.load(open(hold_p))}
res = {"pool_n": len(pool), "pool_in_753": len(pool & off), "pool_outside_753": len(pool - off),
       "outside_753_with_target": len(comp - off), "new_subjects": len(hold),
       "new_subjects_in_pool": len(hold & pool), "new_subjects_in_753": len(hold & off), "pool_key": key}
json.dump(res, open(out_p, "w"), indent=1)
print(res)
