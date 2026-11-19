"""Hypothesis C, finally run: the alpha transform ON TOP of the current chain.

The instruction's C was "F-BERT alpha filter on top at g=0.98 / 2 % mix",
which died on a broadcasting bug on <pod> (`V * clip(ev)**0.35` with shapes
(753,400,400) x (753,400)) and was never retried. It also targeted the old
0.6927 chain.

This runs it against the CURRENT certified chain (0.7326 raw target), as a
mixture: K' = (1-w) K_certified + w K_alpha, where K_alpha is built by taking
each subject's correlation matrix, raising its eigenvalues to 0.35
(F-BERT's exponent), and rebuilding a kernel from that. The exponent is fixed
at 0.35 as in the original hypothesis; the mixture weight w is what varies.

ANOTHER, SEPARATE THING THAT WAS NEVER TRIED: applying the alpha exponent to
the PARTIAL matrix at a second, different exponent from the chain's own 0.10.
The chain already uses alpha=0.10 there; this asks whether a second alpha pass
at 0.35 adds anything.

Both are evaluated on the 753 / 60x10 folds, fold-mean r, NB-corrected t.
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
EXP = 0.35

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']
iu = np.triu_indices(400, 1); dia = np.arange(400)
print('n=%d' % n, flush=True)

z = np.load(BASE + '/chain_K.npz')
KCERT = z['filtered'] if z['filtered'].trace() > 0 else z['filtered']
# per-modality certified kernels are what the filter needs; rebuild them here
# (chain_K.npz stores only the ensemble, and the alpha arm must be filtered
#  per modality exactly as the certified one is)


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


def alpha_pow(M_edges, p):
    out = np.empty_like(M_edges)
    for i in range(len(M_edges)):
        M = np.zeros((400, 400)); M[iu] = M_edges[i]; M = M + M.T; M[dia, dia] = 1.0
        ev, V = np.linalg.eigh(M); ev = np.clip(ev, 0, None)
        out[i] = ((V * ev ** p) @ V.T)[iu]
    return out


CACHE = BASE + '/ALPHAONTOP_K.npz'
if os.path.exists(CACHE):
    zz = np.load(CACHE)
    KA = {m: zz['a_' + m] for m in MODS}
    KC = {m: zz['c_' + m] for m in MODS}
    print('cached', flush=True)
else:
    KA = {}; KC = {}
    for m in MODS:
        E = np.load(BASE + '/mod_%s.npz' % m)['bp'].astype(np.float32)
        C = np.zeros((n, 400, 400)); C[:, iu[0], iu[1]] = E
        C = C + np.transpose(C, (0, 2, 1)); C[:, dia, dia] = 1.0
        for i in range(n):
            d = np.sqrt(np.clip(np.diag(C[i]), 1e-12, None)); C[i] = C[i] / np.outer(d, d)
        C[:, dia, dia] = 1.0
        F = C[:, iu[0], iu[1]].copy()
        Cg = C.copy() * (1.0 - G); Cg[:, dia, dia] += G
        del C; gc.collect()
        P = np.empty((n, len(iu[0])))
        for i in range(n):
            Pi = np.linalg.inv(Cg[i]); dd = np.sqrt(np.clip(np.diag(Pi), 1e-12, None))
            P[i] = (-Pi / np.outer(dd, dd))[iu]
        del Cg; gc.collect()
        Pa = alpha_pow(P, ALPHA)
        KC[m] = (1 - W) * sfilt(cks(Pa)) + W * cks(F)
        # hypothesis C: alpha 0.35 on the CORRELATION matrix, filtered the
        # same way, mixed at 5 % full as in the certified arm
        Cf = alpha_pow(F, EXP)
        KA[m] = (1 - W) * sfilt(cks(Cf)) + W * cks(F)
        del E, F, P, Pa, Cf; gc.collect()
        print('built', m, flush=True)
    np.savez_compressed(CACHE, **{('a_' + m): KA[m] for m in MODS},
                        **{('c_' + m): KC[m] for m in MODS})
    print('cached', flush=True)


def ens(parts):
    tot = 0.0
    for m in MODS:
        tot = tot + WT[m] * parts[m]
    return tot / sum(WT.values())


KCERT = ens(KC); KALPHA = ens(KA)


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
    m = len(d); k = len(a[0])
    n_te = n / k; ratio = n_te / (n - n_te)
    t = d.mean() / np.sqrt((1.0 / m + ratio) * d.var(ddof=1))
    return float(t), float(erfc(abs(t) / sqrt(2)))


res = {'n': n, 'seeds': 60, 'exponent': EXP,
       'hypothesis': 'C: F-BERT alpha (ev^0.35) on the correlation matrix, on top of the current chain'}
b_per, b_flat = run(KCERT)
res['certified_chain'] = round(float(b_per.mean()), 4)
print('certified %.4f' % b_per.mean(), flush=True)
json.dump(res, open(BASE + '/ALPHA_ON_TOP.json', 'w'), indent=1)

for wv in (1.0, 0.5, 0.25, 0.10):
    M = (1 - wv) * (KCERT / np.trace(KCERT)) + wv * (KALPHA / np.trace(KALPHA))
    M = M / np.trace(M)
    a, af = run(M)
    d = a - b_per
    t, p = nb_t(af, b_flat)
    key = 'alpha_only' if wv == 1.0 else 'mix_%.2f' % wv
    res[key] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                'NB_t': round(t, 2), 'p': round(p, 4)}
    print('%-12s %.4f  %+.4f  %d/%d  NB t %.2f  p %.3f'
          % (key, a.mean(), d.mean(), (d > 0).sum(), len(d), t, p), flush=True)
    json.dump(res, open(BASE + '/ALPHA_ON_TOP.json', 'w'), indent=1)

json.dump(res, open(BASE + '/ALPHA_ON_TOP.json', 'w'), indent=1)
print('DONE', flush=True)
