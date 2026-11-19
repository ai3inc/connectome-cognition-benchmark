"""Fairness check for addendum R: F-BERT-8M six vs five modalities (no motor), same folds,
best of its two ensemble probes in-fold; paired with D0 no-motor per seed."""
import csv, json, sys
import numpy as np, scipy.io as sio
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
sys.path.insert(0, '/workspace/sota')
import eval_samefolds as ev, eval_sixmod as sx
rest = sx.load_feats('/workspace/sota/feats/feat_cbd*.npz', {"e8_full": "e8_rs"})
task = sx.load_feats('/workspace/sota/feats/taskfeat_cbd*.npz', {f"e8_{m}": f"e8_{m}" for m in sx.MODS[1:]})
oi = {r["Subject"].strip(): r for r in csv.DictReader(open('/workspace/ooi_comp.csv'))}
off = [x.strip() for x in open('/workspace/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan")) and s in rest and s in task]
y = np.array([float(oi[s]["varimax_cog"]) for s in use]); sidx = {s: i for i, s in enumerate(use)}
folds = sio.loadmat('/workspace/splits.mat', squeeze_me=True, struct_as_record=False)["folds"]
names = list(folds._fieldnames)[:60]
fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx]) for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
E = {m: np.stack([(rest if m == "rs" else task)[s][f"e8_{m}"] for s in use]).astype(np.float64) for m in sx.MODS}
def setup(mods):
    return {"corr": sum(ev.correlation_kernel(E[m]) for m in mods) / len(mods), "cat": np.concatenate([E[m] for m in mods], 1)}
SETS = {"six": setup(sx.MODS), "five": setup([m for m in sx.MODS if m != "motor"])}
def run(nm):
    with threadpool_limits(1):
        n = len(y); out = {}
        for k, c in SETS.items():
            p = np.full(n, np.nan)
            for te in fl[nm]:
                tr = np.setdiff1d(np.arange(n), te)
                p[te] = ev.krr_select([c["corr"], ev.zlin_kernel(c["cat"], tr)], tr, te, y)[0]
            out[k] = float(np.corrcoef(p, y)[0, 1])
        return nm, out
with Pool(8) as pool: res = pool.map(run, names)
fb6 = np.array([o[1]["six"] for o in res]); fb5 = np.array([o[1]["five"] for o in res])
R = json.load(open('/workspace/sota/R_MOTOR_1117.json')); d0n = np.array(R["per_seed"]["D0_nomotor"])
def paired(a, b):
    x = a - b; return {"mean_diff": float(x.mean()), "positive": int((x > 0).sum()), "of": len(x), "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))}
out = {"note": "post-hoc fairness check for addendum R", "FBERT_six": float(fb6.mean()), "FBERT_five": float(fb5.mean()),
       "FBERT_five_minus_six": paired(fb5, fb6), "D0_nomotor": float(d0n.mean()),
       "D0_nomotor_minus_FBERT_best": paired(d0n, np.maximum(fb5, fb6) if fb5.mean() > fb6.mean() else fb6),
       "D0_nomotor_minus_FBERT_five": paired(d0n, fb5), "per_seed": {"FBERT_six": fb6.round(6).tolist(), "FBERT_five": fb5.round(6).tolist()}}
json.dump(out, open('/workspace/sota/R_FBERT_NOMOTOR_1117.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))
