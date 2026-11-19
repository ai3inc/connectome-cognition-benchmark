"""The covariate-matched, fold-selected, exact-t version of the clip result.

session's three conditions, all addressed here:
 (a) clip is a POST-HOC choice found on the same 753, so its +0.0062 cannot be
     added to 0.7215. It must be chosen inside the fold.
 (b) the t must be the Nadeau-Bengio exact form on the OUTER fold scores,
     not a naive over seeds.
 (c) the target must be residualised (age band + sex, in-fold), because that
     is the condition 0.7215 is quoted under.

Design: {off, clip1.5} is a 2-way choice made by inner 4-fold CV on the
training block. Outer folds are the benchmark's 10 per seed x 60 seeds.
Everything else is the certified chain.
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
G, W, ALPHA, TAU, KEEP = 0.92, 0.05, 0.10, 0.5, 150
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
INNER = (1e-4, 1e-3, 1e-2, 0.1, 1.0)

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
cov = {}
for r in csv.DictReader(open(BASE + '/hcp_cov.csv')):
    g = str(r['Gender']).strip().upper(); a = str(r['Age']).strip()
    band = {'22-25': 0, '26-30': 1, '31-35': 2, '36+': 3}.get(a)
    if band is None:
        continue
    cov[r['Subject'].strip()] = (1.0 if g in ('M', '1', 'MALE') else 0.0, float(band))
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and s in cov
       and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y_raw = np.array([float(oi[s]['varimax_cog']) for s in use])
X = np.column_stack([[cov[s][0] for s in use]] +
                    [[float(cov[s][1] == k) for s in use] for k in (1, 2, 3)])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']
print('n=%d' % n, flush=True)

z = np.load(BASE + '/RANKVAR_K.npz')
KCERT = 0.0; KCLIP = 0.0
for m in MODS:
    KCERT = KCERT + WT[m] * z['none_' + m]
    KCLIP = KCLIP + WT[m] * z['clip1.5_' + m]
KCERT = KCERT / sum(WT.values()); KCLIP = KCLIP / sum(WT.values())
print('kernels ready', flush=True)


def pick_lam(Ktr, ytr):
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


def inner_score(Km, tr, ytr):
    rs = np.random.RandomState(0); p = rs.permutation(len(tr))
    f = np.zeros(len(tr), int); f[p] = np.arange(len(tr)) % 4
    Ktr = Km[np.ix_(tr, tr)]
    R = []
    for k in range(4):
        a = np.where(f != k)[0]; b = np.where(f == k)[0]
        w, V = np.linalg.eigh(Ktr[np.ix_(a, a)])
        ya = ytr[a]; yc = ya - ya.mean()
        Vt = V.T @ yc; VtK = Ktr[np.ix_(b, a)] @ V
        R.append(max(float(np.corrcoef(VtK @ (Vt / (w + l)), ytr[b])[0, 1]) for l in INNER))
    return float(np.mean(R))


def run(mode, seeds=60):
    """mode: 'cert' | 'clip' | 'nested'"""
    per = []; flat = []; picks = {0: 0, 1: 0}
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
            beta = np.linalg.lstsq(Xtr, y_raw[tr], rcond=None)[0]
            ytr = y_raw[tr] - Xtr @ beta
            Xte = np.column_stack([np.ones(len(te)), X[te]])
            yte = y_raw[te] - Xte @ beta
            if mode == 'cert':
                Km = KCERT
            elif mode == 'clip':
                Km = KCLIP
            else:
                s0 = inner_score(KCERT, tr, ytr)
                s1 = inner_score(KCLIP, tr, ytr)
                arm = 0 if s0 >= s1 else 1
                picks[arm] += 1
                Km = KCERT if arm == 0 else KCLIP
            lam = pick_lam(Km[np.ix_(tr, tr)], ytr)
            w, V = np.linalg.eigh(Km[np.ix_(tr, tr)]); yc = ytr - ytr.mean()
            p = (Km[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + ytr.mean()
            if np.std(p) > 1e-9 and np.std(yte) > 1e-9:
                fr.append(float(np.corrcoef(p, yte)[0, 1]))
        if fr:
            per.append(float(np.mean(fr))); flat.append(fr)
    return np.array(per), flat, picks


def nb_exact(a, b):
    """Nadeau-Bengio on outer fold scores: (1/m + n_test/n_train) inflation.
    a, b: list of per-seed lists of outer-fold r."""
    d = np.array([x - yv for fa, fb in zip(a, b) for x, yv in zip(fa, fb)])
    m = len(d)
    k = len(a[0])
    n_test = n / k; n_train = n - n_test
    ratio = n_test / n_train
    # NB exact (Nadeau & Bengio 2003): var(d) * (1/m + n_te/n_tr).
    # The version that produced the first run's t divided by m as well.
    t = d.mean() / np.sqrt((1.0 / m + ratio) * d.var(ddof=1))
    return float(t), float(erfc(abs(t) / sqrt(2))), float(d.mean()), m


res = {'n': n, 'seeds': 60, 'target': 'varimax_cog residualised in-fold on age band(4)+sex',
       'metric': 'mean of per-fold r', 'note': 'this is the condition 0.7215 is quoted under'}
c_per, c_flat, _ = run('cert')
res['cert_raw_value_kernel'] = round(float(c_per.mean()), 4)
print('cert %.4f' % c_per.mean(), flush=True)
json.dump(res, open(BASE + '/CLIP_COV_GATE.json', 'w'), indent=1)

k_per, k_flat, _ = run('clip')
d = k_per - c_per
t, p, dm, m = nb_exact(k_flat, c_flat)
res['clip_fixed'] = {'r': round(float(k_per.mean()), 4), 'diff': round(float(d.mean()), 4),
                     'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                     'NB_t': round(t, 2), 'p': round(p, 4), 'n_fold_scores': m}
print('clip %.4f  %+.4f  %d/%d  NB t %.2f  p %.4f'
      % (k_per.mean(), d.mean(), (d > 0).sum(), len(d), t, p), flush=True)
json.dump(res, open(BASE + '/CLIP_COV_GATE.json', 'w'), indent=1)

n_per, n_flat, picks = run('nested')
d = n_per - c_per
t, p, dm, m = nb_exact(n_flat, c_flat)
res['clip_fold_internal_2way'] = {'r': round(float(n_per.mean()), 4), 'diff': round(float(d.mean()), 4),
                                  'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                                  'NB_t': round(t, 2), 'p': round(p, 4), 'n_fold_scores': m,
                                  'picks_clip': picks[1], 'picks_raw': picks[0]}
print('nested %.4f  %+.4f  %d/%d  NB t %.2f  p %.4f  picks clip %d raw %d'
      % (n_per.mean(), d.mean(), (d > 0).sum(), len(d), t, p, picks[1], picks[0]), flush=True)

json.dump(res, open(BASE + '/CLIP_COV_GATE.json', 'w'), indent=1)
print('DONE', flush=True)
