"""POST-HOC DIAGNOSTIC (not a claim): ADHD-200 within NYU only (no site shift),
random 10-fold x 20 repeats, IQ r and ADHD AUC, D0 vs F-BERT-8M vs FC^0.35."""
import glob, json, sys
import numpy as np
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
sys.path.insert(0, '/workspace/fbert')
import eval_samefolds as ev, aomic_eval as ae
from abide_eval import auc
ph = json.load(open('/workspace/fbert/adhd_pheno.json'))
F = {}
for f in sorted(glob.glob('/workspace/fbert/adhd/adhdfeat_*.npz')):
    with np.load(f) as d:
        a = {k: d[k] for k in d.files}
    for i, s in enumerate(a['sids']):
        F[str(s)] = {k: np.array(v[i]) for k, v in a.items() if k != 'sids'}
use = sorted(s for s in F if s in ph and ph[s]['site'] == 'NYU')
arms = {k: [] for k in ['A8', 'C', 'D0']}
for v in ['nogsr', 'gsr']:
    fc = np.stack([F[s][f'fc_{v}'] for s in use])
    with Pool(8) as p: res = p.map(ae.per_subject_g, list(fc), chunksize=8)
    Kf = ev.correlation_kernel(fc)
    arms['C'].append(('k', ev.correlation_kernel(np.stack([r[0] for r in res]))))
    for gi in range(len(ae.GS)):
        arms['D0'].append(('k', 0.95 * ev.correlation_kernel(np.stack([r[1][gi] for r in res])) + 0.05 * Kf))
    E = np.stack([F[s][f'e8_{v}'] for s in use]).astype(np.float64)
    arms['A8'] += [('k', ev.correlation_kernel(E)), ('zlin', E)]
def iqv(x):
    try:
        v = float(x); return v if 40 < v < 200 else np.nan
    except Exception: return np.nan
iq = np.array([iqv(ph[s]['iq']) for s in use]); fi = np.nonzero(np.isfinite(iq))[0]
dx = np.array([-1.0 if 'Typically' in ph[s]['dx'] else 1.0 for s in use])
def rep(args):
    s, task = args
    with threadpool_limits(1):
        idx = fi if task == 'iq' else np.arange(len(use)); y = iq[fi] if task == 'iq' else dx
        n = len(y); perm = np.random.RandomState(500 + s).permutation(n); out = {}
        for a, spec in arms.items():
            pr = np.full(n, np.nan)
            for te in np.array_split(perm, 10):
                tr = np.setdiff1d(np.arange(n), te)
                Ks = [ev.zlin_kernel(E[idx], tr) if k == 'zlin' else E[np.ix_(idx, idx)] for k, E in spec]
                pr[te] = ev.krr_select(Ks, tr, te, y)[0]
            out[a] = float(np.corrcoef(pr, y)[0, 1]) if task == 'iq' else auc(pr, y)
        return task, out
with Pool(8) as p: res = p.map(rep, [(s, t) for t in ['iq', 'dx'] for s in range(20)])
R = {'note': 'post-hoc diagnostic, NYU only, random 10-fold x 20', 'n': len(use), 'n_iq': int(len(fi))}
for t in ['iq', 'dx']:
    v = {a: np.array([o[1][a] for o in res if o[0] == t]) for a in arms}
    x = v['D0'] - v['A8']
    R[t] = {a: float(z.mean()) for a, z in v.items()}
    R[t]['D0_minus_A8'] = float(x.mean()); R[t]['D0_ahead_repeats'] = int((x > 0).sum())
json.dump(R, open('/workspace/fbert/FBERT_ADHD_NYU_WITHIN_1117.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
