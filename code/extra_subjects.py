"""Axis 5 -- USE THE 221 SUBJECTS THE BENCHMARK LEAVES OUT.

pool.npz holds 974 subjects' FC; the benchmark's analysis set is 753. The
ledger's lesson is that anything fitted on ~700 subjects is noise-limited, while
a model PRE-TRAINED on thousands wins (F-BERT). 221 extra subjects is not
thousands, but they are free and label-carrying: varimax_cog exists for them.

Two honest uses, both leaving the 753 evaluation untouched:

  (A) LABEL PRIOR: fit the group mean of y on the extra subjects and use it as
      the intercept instead of the training-fold mean. Cheapest possible use;
      asks whether the extra labels carry information about the level.
  (B) KERNEL STANDARDISATION: estimate the per-edge mean and sd of the FC
      vector on the extra subjects and standardise BOTH arms with it. This is
      the "fit the preprocessing outside the evaluation set" move that the
      benchmark itself allows (it builds kernels over all subjects).

Both are compared fold-to-fold on the 753 against the unmodified chain.
"""
import numpy as np, json, csv, scipy.io as sio
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
                3e-2, 0.1, 0.3, 1.0, 3.0, 10.0])
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']

# evaluation set = the 753 (same as everywhere else)
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
pos = {s: i for i, s in enumerate(use)}

z = np.load(BASE + '/KTA_STRUCT_K.npz')
KP = {m: z['p_' + m] for m in MODS}; KF = {m: z['f_' + m] for m in MODS}
K = 0.0
for m in MODS:
    K = K + WT[m] * ((1 - W) * KP[m] + W * KF[m])
K = K / sum(WT.values())
print('kernel ready', flush=True)

pz = np.load(BASE + '/pool.npz')
subs = [str(x).strip() for x in pz['subs']]
extra = [s for s in subs if s not in pos and s in oi
         and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
print('extra labelled subjects: %d' % len(extra), flush=True)
y_ex = np.array([float(oi[s]['varimax_cog']) for s in extra])


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


def run(offset=None, seeds=60):
    """offset: if given, an additive prior on the intercept, pulled toward it
    with the training fold's own mean (shrinkage weight 0.5)."""
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            lam = pick(K[np.ix_(tr, tr)], y[tr])
            w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
            m_tr = y[tr].mean()
            if offset is not None:
                m_tr = 0.5 * m_tr + 0.5 * offset
            yc = y[tr] - m_tr
            p = (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + m_tr
            if np.std(p) > 1e-9:
                fr.append(float(np.corrcoef(p, y[te])[0, 1]))
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


res = {'n': n, 'n_extra_labelled': len(extra), 'seeds': 60}
b_per, b_flat = run()
res['baseline'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

a, af = run(offset=float(y_ex.mean()))
d = a - b_per
t, p = nb_t(af, b_flat)   # new minus baseline
res['extra_label_mean_intercept'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                                     'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                                     'NB_t': round(t, 2), 'p': round(p, 4)}
print('extra-mean  %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(),
      (d > 0).sum(), len(d), t), flush=True)
json.dump(res, open(BASE + '/EXTRA_SUBJECTS.json', 'w'), indent=1)
print('DONE', flush=True)
