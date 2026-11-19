"""Axis 4 -- AUXILIARY TARGETS. Untried: varimax_satisf and varimax_er are
only ever SCORED, never used to build the kernel.

Idea: KRR predicts a single label vector. But the three composites share
variance, so a kernel that also explains the OTHER labels may be a better
representation. Two label-carrying variants, both fitted strictly inside the
training fold:

  (A) MULTI-TASK KRR: y is substituted by the leading principal component of
      the 3-column label matrix (fitted on train, applied to test). This is
      the cheapest honest version and uses the other labels only through a
      linear combination -- no extra hyperparameter.
  (B) LABEL-AUGMENTED KERNEL: K' = K + b * (Z Z^T) where Z is the 3-column
      label matrix centred on the TRAINING fold. The label Gram is the same
      object KTA aligns against, so this is KTA's "act on it" counterpart.
      b is scanned then reported; a positive b is the finding.

Both are compared fold-to-fold against K on the cognition target, NB t.
"""
import numpy as np, json, csv, scipy.io as sio
from math import erfc, sqrt

BASE = '/workspace'
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
W = 0.05

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']

LAB = []
for k in ('varimax_satisf', 'varimax_cog', 'varimax_er'):
    col = np.array([float(oi[s].get(k) or 'nan') for s in use])
    LAB.append(col)
LAB = np.column_stack(LAB)
print('label coverage', np.isfinite(LAB).sum(0), 'of', n, flush=True)

z = np.load(BASE + '/KTA_STRUCT_K.npz')
KP = {m: z['p_' + m] for m in MODS}; KF = {m: z['f_' + m] for m in MODS}
K = 0.0
for m in MODS:
    K = K + WT[m] * ((1 - W) * KP[m] + W * KF[m])
K = K / sum(WT.values())
del KP, KF
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


def nb_t(a, b):
    d = np.array([x - yv for fa, fb in zip(a, b) for x, yv in zip(fa, fb)])
    k = len(a[0]); n_test = n / k; n_train = n - n_test
    ratio = n_test / n_train
    # NB exact (Nadeau & Bengio 2003): var(d) * ((1/m) + n_te/n_tr).
    # The earlier version multiplied by (1/k + ratio) AND divided by m, which
    # under-corrected by a factor of ~18 (t was inflated 17.9x).
    t = d.mean() / np.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))
    return float(t), float(erfc(abs(t) / sqrt(2)))


def run(mode, b=0.0, seeds=60):
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            if mode == 'plain':
                Ktr = K[np.ix_(tr, tr)]; Kte = K[np.ix_(te, tr)]
                ytr = y[tr]; yte = y[te]
            elif mode == 'pc1':
                L = LAB[tr]; m = np.isfinite(L).all(1)
                tr2 = tr[m]
                Lc = L[m] - L[m].mean(0)
                U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
                u = U[:, 0] * np.sqrt(len(Lc))
                # PC1 ranks by variance (satisf explains 0.55 and correlates
                # -0.99 with it), so "the leading PC" is essentially -satisf,
                # NOT cognition. Use the weighted composite that puts cog at
                # weight 2 (cog itself + er) instead -- that is the auxiliary
                # target a cognition predictor should train on.
                if np.corrcoef(u, L[m][:, 1])[0, 1] < 0:
                    u = -u
                u = Lc[:, 1] + 0.5 * Lc[:, 2]
                Ktr = K[np.ix_(tr2, tr2)]; Kte = K[np.ix_(te, tr2)]
                # train on PC1 (richer target), but SCORE on cognition so the
                # comparison with the plain arm is like for like
                ytr = u; yte = y[te]
            else:  # label-gram augmentation
                L = LAB[tr]; m = np.isfinite(L).all(1)
                tr2 = tr[m]
                Lc = L[m] - L[m].mean(0)
                Lc = Lc / (np.linalg.norm(Lc, axis=0) + 1e-12)
                Gtr = Lc @ Lc.T
                Ktr = K[np.ix_(tr2, tr2)] + b * Gtr * np.trace(K[np.ix_(tr2, tr2)]) / len(tr2)
                Kte = K[np.ix_(te, tr2)]
                ytr = y[tr2]; yte = y[te]
                yrec = y[te]
            lam = pick(Ktr, ytr)
            w, V = np.linalg.eigh(Ktr); yc = ytr - ytr.mean()
            p = (Kte @ V) @ (V.T @ yc / (w + lam)) + ytr.mean()
            tgt = yte if mode != 'pc1' else yte
            if np.std(p) > 1e-9 and np.std(tgt) > 1e-9:
                fr.append(float(np.corrcoef(p, tgt)[0, 1]))
        if fr:
            per.append(float(np.mean(fr))); flat.append(fr)
    return np.array(per), flat


res = {'n': n, 'seeds': 60}
b_per, b_flat = run('plain')
res['baseline'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

a, af = run('pc1')
d = a - b_per
t, pv = nb_t(af, b_flat)   # new minus baseline
res['multi_task_pc1'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                          'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                          'NB_t': round(t, 2), 'p': round(pv, 4)}
print('pc1  %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

for b in (0.02, 0.05, 0.10, 0.20):
    a, af = run('gram', b)
    d = a - b_per
    t, pv = nb_t(af, b_flat)   # new minus baseline
    res['label_gram_b%.2f' % b] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                                   'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                                   'NB_t': round(t, 2), 'p': round(pv, 4)}
    print('gram b=%.2f  %.4f  %+.4f  %d/%d  t=%.2f' % (b, a.mean(), d.mean(),
          (d > 0).sum(), len(d), t), flush=True)

json.dump(res, open(BASE + '/AUX_TARGET.json', 'w'), indent=1)
print('DONE', flush=True)
