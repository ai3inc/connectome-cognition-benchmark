"""Axis 3 -- TARGET TRANSFORM. Untried in the ledger (grep for rank/quantile
returns nothing).

varimax_cog is a composite score; its distribution is mildly skewed and its
tails are heavy. Two transforms, both label-free and both applied identically
to train and test (so no leakage):

  RANK : replace y by its normal-score rank (van der Waerden). Prediction is
         scored by correlating the prediction with the same transform, which is
         what r is anyway for a monotone transform -- so the real question is
         whether KRR *fits* better when the target is Gaussianised.
  WINSOR: clip the target at +-k sd, keeping the metric on the raw scale.

Both are compared fold-to-fold against the untransformed arm with an NB t.

Also tests a THIRD, different thing that is in the same family and cheap:
  y = varimax_cog residualised on varimax_satisf (partial-out the
  satisfaction composite, which shares variance with cognition).
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
sat = np.array([float(oi[s].get('varimax_satisf') or 'nan') for s in use])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']

CACHE = BASE + '/KTA_STRUCT_K.npz'
z = np.load(CACHE)
KP = {m: z['p_' + m] for m in MODS}; KF = {m: z['f_' + m] for m in MODS}
K = 0.0
for m in MODS:
    K = K + WT[m] * ((1 - W) * KP[m] + W * KF[m])
K = K / sum(WT.values())
print('kernel ready', flush=True)


def pick(Ktr, ytr):
    rs = np.random.RandomState(0); p = rs.permutation(len(ytr))
    f = np.zeros(len(ytr), int); f[p] = np.arange(len(ytr)) % 4
    R = np.zeros((len(LAM), 4))
    for k in range(4):
        te = np.where(f == k)[0]; tr = np.where(f != k)[0]
        w, V = np.linalg.eigh(Ktr[np.ix_(tr, tr)]); yc = ytr[tr] - ytr[tr].mean()
        Vt = V.T @ yc; VtK = Ktr[np.ix_(te, tr)] @ V
        for i, l in enumerate(LAM):
            R[i, k] = np.corrcoef(VtK @ (Vt / (w + l)), ytr[te])[0, 1]
    return LAM[int(np.argmax(R.mean(1)))]


def foldvec(yy, seeds=60):
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            lam = pick(K[np.ix_(tr, tr)], yy[tr])
            w, V = np.linalg.eigh(K[np.ix_(tr, tr)]); yc = yy[tr] - yy[tr].mean()
            p = (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + yy[tr].mean()
            if np.std(p) > 1e-9:
                fr.append(float(np.corrcoef(p, yy[te])[0, 1]))
        if fr:
            per.append(float(np.mean(fr))); flat.append(fr)
    return np.array(per), flat


def nb_t(a, b):
    d = np.array([x - yv for fa, fb in zip(a, b) for x, yv in zip(fa, fb)])
    k = len(a[0]); n_test = n / k; n_train = n - n_test
    ratio = n_test / n_train
    # NB exact (Nadeau & Bengio 2003): var(d) * ((1/m) + n_te/n_tr).
    # The earlier version multiplied by (1/k + ratio) AND divided by m, which
    # under-corrected by a factor of ~18 (t was inflated 17.9x).
    t = d.mean() / np.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))
    return float(t), float(erfc(abs(t) / sqrt(2)))


def ranknorm(v):
    """Normal-score (van der Waerden) transform: the i-th smallest value
    becomes the i-th normal quantile. The rank assignment must be written at
    the sorted POSITIONS; writing it at the original indices inverts the map
    and turns the correlation negative -- that bug cost one run."""
    from scipy.stats import norm
    order = np.argsort(v)                     # order[i] = index of i-th smallest
    ranks = np.empty(len(v))
    ranks[order] = (np.arange(1, len(v) + 1) - 0.5) / len(v)
    return norm.ppf(ranks)


res = {'n': n, 'seeds': 60}
b_per, b_flat = foldvec(y)
res['baseline_raw_target'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

# 1. rank-normal target
yr = ranknorm(y)
assert np.corrcoef(yr, y)[0, 1] > 0.99, 'rank transform inverted'
print('rank-vs-raw corr %.4f' % np.corrcoef(yr, y)[0, 1], flush=True)
a, af = foldvec(yr)
d = a - b_per
res['rank_normal_target'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                             'wins': '%d/%d' % (int((d > 0).sum()), len(d))}
t, p = nb_t(af, b_flat)   # new minus baseline, so t and diff share a sign
res['rank_normal_target'].update({'NB_t': round(t, 2), 'p': round(p, 4)})
print('ranknorm %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

# 2. winsorise
for ksd in (2.5, 2.0):
    yw = np.clip(y, y.mean() - ksd * y.std(), y.mean() + ksd * y.std())
    a, af = foldvec(yw)
    d = a - b_per
    t, p = nb_t(af, b_flat)   # new minus baseline
    res['winsor_%.1f' % ksd] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                                'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                                'NB_t': round(t, 2), 'p': round(p, 4)}
    print('winsor %.1f  %.4f  %+.4f  %d/%d  t=%.2f' % (ksd, a.mean(), d.mean(),
          (d > 0).sum(), len(d), t), flush=True)

# 3. partial out satisfaction (in-fold)
per = []; flat = []
for nm in list(folds._fieldnames)[:60]:
    fr = []
    for fd in getattr(folds, nm).sub_fold:
        te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
        if len(te) < 20:
            continue
        tr = np.setdiff1d(np.arange(n), te)
        ok = np.isfinite(sat[tr])
        tr2 = tr[ok]
        if len(tr2) < 100:
            continue
        A = np.column_stack([np.ones(len(tr2)), sat[tr2]])
        beta = np.linalg.lstsq(A, y[tr2], rcond=None)[0]
        yy = np.empty(n); yy[:] = np.nan
        yy[tr2] = y[tr2] - A @ beta
        Ate = np.column_stack([np.ones(len(te)), sat[te]])
        okte = np.isfinite(sat[te])
        yy[te[okte]] = y[te[okte]] - Ate[okte] @ beta
        lam = pick(K[np.ix_(tr2, tr2)], yy[tr2])
        w, V = np.linalg.eigh(K[np.ix_(tr2, tr2)]); yc = yy[tr2] - yy[tr2].mean()
        p = (K[np.ix_(te, tr2)] @ V) @ (V.T @ yc / (w + lam)) + yy[tr2].mean()
        m = okte & np.isfinite(yy[te])
        if m.sum() > 25 and np.std(p[m]) > 1e-9:
            fr.append(float(np.corrcoef(p[m], yy[te][m])[0, 1]))
    if fr:
        per.append(float(np.mean(fr))); flat.append(fr)
a = np.array(per)
res['satisfaction_partialled_out'] = {'r': round(float(a.mean()), 4),
                                      'vs_baseline': round(float(a.mean() - b_per.mean()), 4),
                                      'note': 'different target, so not a paired gain'}
print('satisf-partial %.4f' % a.mean(), flush=True)

json.dump(res, open(BASE + '/TARGET_TRANSFORM.json', 'w'), indent=1)
print('DONE', flush=True)
