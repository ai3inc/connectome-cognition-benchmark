"""Two follow-ups on the rank-kernel gain (+0.0054, 58/60).

A) IS IT THE RANKING, OR JUST OUTLIER SUPPRESSION?
   A rank transform is the limit of a monotone squash. If the gain is really
   "the extreme edges were dominating", then a much cheaper transform --
   clipping each subject's edges at +-k robust sd -- should recover most of it.
   If clipping does nothing but ranking helps, the gain is in the ORDERING,
   which is a different (and more interesting) statement.

   Variants, all on the same cached kernels' construction path:
     - clip at 2.5 / 1.5 robust sd, then build the kernel exactly as certified
     - signed sqrt (a smooth squash that compresses the tails less brutally)
     - rank (the arm that won)

B) SECOND CONSTRUCTION: rank the FULL-correlation edges instead of the partial
   ones, and rank before vs after the alpha re-weight. If the gain survives
   every ordering, it is a property of the data; if it only survives one
   ordering, it is a tuning accident.
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']
iu = np.triu_indices(400, 1); dia = np.arange(400)
print('n=%d' % n, flush=True)


def cks(M):
    h = len(M); Mc = M - M.mean(0); N = np.linalg.norm(Mc, axis=1) + 1e-12
    K = (Mc @ Mc.T) / np.outer(N, N)
    H = np.eye(h) - np.ones((h, h)) / h
    K = H @ K @ H
    return K / np.trace(K)


def sfilt(K):
    w, V = np.linalg.eigh(K); w = np.clip(w, 0, None)
    f = np.clip(w - TAU * np.std(w), 0.0, None)
    kk = min(KEEP, len(f)); thr = np.sort(f)[::-1][kk - 1]
    f = np.where(f >= thr, f, 0.0)
    Kt = (V * f) @ V.T; tr = np.trace(Kt)
    return Kt / tr if tr > 0 else Kt


def transform(M, kind):
    if kind == 'rank':
        o = np.argsort(np.argsort(np.asarray(M, np.float32), axis=1), axis=1)
        return (o.astype(np.float32) + 0.5) / M.shape[1]
    if kind == 'clip2.5':
        med = np.median(M, axis=1, keepdims=True)
        rsd = 1.4826 * np.median(np.abs(M - med), axis=1, keepdims=True) + 1e-9
        return np.clip(M, med - 2.5 * rsd, med + 2.5 * rsd)
    if kind == 'clip1.5':
        med = np.median(M, axis=1, keepdims=True)
        rsd = 1.4826 * np.median(np.abs(M - med), axis=1, keepdims=True) + 1e-9
        return np.clip(M, med - 1.5 * rsd, med + 1.5 * rsd)
    if kind == 'sqrt':
        return np.sign(M) * np.sqrt(np.abs(M))
    if kind == 'none':
        return M
    raise ValueError(kind)


ARMS = ['none', 'clip2.5', 'clip1.5', 'sqrt', 'rank']
CACHE = BASE + '/RANKVAR_K.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    KK = {a: {m: z[a + '_' + m] for m in MODS} for a in ARMS}
    print('cached', flush=True)
else:
    KK = {a: {} for a in ARMS}
    for m in MODS:
        E = np.load(BASE + '/mod_%s.npz' % m)['bp'].astype(np.float32)
        C = np.zeros((n, 400, 400)); C[:, iu[0], iu[1]] = E
        C = C + np.transpose(C, (0, 2, 1)); C[:, dia, dia] = 1.0
        for i in range(n):
            d = np.sqrt(np.clip(np.diag(C[i]), 1e-12, None)); C[i] = C[i] / np.outer(d, d)
        C[:, dia, dia] = 1.0
        F = C[:, iu[0], iu[1]].copy()
        Cg = C.copy() * (1.0 - G); Cg[:, dia, dia] += G
        P = np.empty((n, len(iu[0])))
        for i in range(n):
            Pi = np.linalg.inv(Cg[i]); dd = np.sqrt(np.clip(np.diag(Pi), 1e-12, None))
            P[i] = (-Pi / np.outer(dd, dd))[iu]
        del Cg; gc.collect()
        Pa = np.empty_like(P)
        for i in range(n):
            M = np.zeros((400, 400)); M[iu] = P[i]; M = M + M.T; M[dia, dia] = 1.0
            ev, V = np.linalg.eigh(M); ev = np.clip(ev, 0, None)
            Pa[i] = ((V * ev ** ALPHA) @ V.T)[iu]
        for a in ARMS:
            Pf = transform(Pa, a); Ff = transform(F, a)
            KK[a][m] = (1 - W) * sfilt(cks(Pf)) + W * cks(Ff)
            del Pf, Ff
        del E, C, F, Pa, P; gc.collect()
        print('built', m, flush=True)
    np.savez_compressed(CACHE, **{(a + '_' + m): KK[a][m] for a in ARMS for m in MODS})
    print('cached', flush=True)


def ens(parts):
    tot = 0.0
    for m in MODS:
        tot = tot + WT[m] * parts[m]
    return tot / sum(WT.values())


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


def run(Km, seeds=60):
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            lam = pick(Km[np.ix_(tr, tr)], y[tr])
            w, V = np.linalg.eigh(Km[np.ix_(tr, tr)]); yc = y[tr] - y[tr].mean()
            p = (Km[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()
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


res = {'n': n, 'seeds': 60,
       'question': 'is the rank-kernel gain the ORDERING, or just outlier suppression?'}
b_per, b_flat = None, None
for a in ARMS:
    per, flat = run(ens(KK[a]))
    if b_per is None:
        b_per, b_flat = per, flat
        res[a] = {'r': round(float(per.mean()), 4)}
    else:
        d = per - b_per
        t, p = nb_t(flat, b_flat)
        res[a] = {'r': round(float(per.mean()), 4), 'diff_vs_none': round(float(d.mean()), 4),
                  'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                  'NB_t': round(t, 2), 'p': round(p, 4)}
    print('%-9s %.4f  %s' % (a, per.mean(),
          '' if a == 'none' else '%+.4f  %d/%d  t=%.2f' % (res[a]['diff_vs_none'], (d > 0).sum(), len(d), t)),
          flush=True)
    json.dump(res, open(BASE + '/RANK_VARIANTS.json', 'w'), indent=1)

json.dump(res, open(BASE + '/RANK_VARIANTS.json', 'w'), indent=1)
print('DONE', flush=True)
