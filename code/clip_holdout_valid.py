"""The one open claim: does clip help on subjects the fold never saw?

Every in-fold gain in this project has died here. The earlier attempt was
invalid because it trained on the published features and tested on pool.npz,
which is a DIFFERENT extraction pipeline (feature correlation 0.12-0.24).

This one is valid because it uses ONE pipeline throughout: the 999-subject
block in HOLDOUT_RANK_K.npz was built from a single source, so train and test
are the same representation.

Design:
  train on a random subset of the 753 benchmark subjects
  test on the subjects in the block that are NOT in the 753
  Same chain (g, alpha, w, filter, lambda in-fold) for both arms; the only
  difference is the edge clip.
  Repeated over 30 random training subsets. The test set is fixed, so the
  interval comes from a SUBJECT bootstrap (training-set repetition only
  measures training variance -- the pseudoreplication lesson).
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
iu = np.triu_indices(400, 1); dia = np.arange(400)

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
inside = set(off)

pz = np.load(BASE + '/pool.npz')
psubs = [str(x).strip() for x in pz['subs']]
bpp = pz['bpp']
print('pool %d' % len(psubs), flush=True)

in_block = [s for s in psubs if s in oi]
ysub = {s: float(oi[s]['varimax_cog']) for s in in_block
        if np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))}
tr_all = [s for s in psubs if s in inside and s in ysub]
te_all = [s for s in psubs if s not in inside and s in ysub]
print('train pool %d  test %d' % (len(tr_all), len(te_all)), flush=True)
idx = {s: i for i, s in enumerate(psubs)}


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


def clip_rows(M, k=1.5):
    med = np.median(M, axis=1, keepdims=True)
    rsd = 1.4826 * np.median(np.abs(M - med), axis=1, keepdims=True) + 1e-9
    return np.clip(M, med - k * rsd, med + k * rsd)


CACHE = BASE + '/HOLDOUT_CLIP_K.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    for nm in ('none', 'clip1.5'):
        pass
    KN = {m: z['n_' + m] for m in MODS}
    KC = {m: z['k_' + m] for m in MODS}
    print('clip holdout kernels cached', flush=True)
else:
    KN = {}; KC = {}
    nsub = len(psubs)
    for m in MODS:
        E = bpp.astype(np.float32)          # all 999 raw edges for this modality
        # build the same chain twice: raw edges, and clipped edges
        for tag, transform, store in (('none', None, KN), ('clip', clip_rows, KC)):
            C = np.zeros((nsub, 400, 400), dtype=np.float32)
            C[:, iu[0], iu[1]] = E
            C = C + np.transpose(C, (0, 2, 1)); C[:, dia, dia] = 1.0
            for i in range(nsub):
                d = np.sqrt(np.clip(np.diag(C[i]), 1e-12, None))
                C[i] = C[i] / np.outer(d, d)
            C[:, dia, dia] = 1.0
            F = C[:, iu[0], iu[1]].copy()
            Cg = C.copy() * (1.0 - G); Cg[:, dia, dia] += G
            del C; gc.collect()
            P = np.empty((nsub, len(iu[0])), dtype=np.float32)
            for i in range(nsub):
                Pi = np.linalg.inv(Cg[i]); dd = np.sqrt(np.clip(np.diag(Pi), 1e-12, None))
                P[i] = (-Pi / np.outer(dd, dd))[iu]
            del Cg; gc.collect()
            if transform is not None:
                P = transform(P); F = transform(F)
            Pa = np.empty_like(P)
            for i in range(nsub):
                M = np.zeros((400, 400)); M[iu] = P[i]; M = M + M.T; M[dia, dia] = 1.0
                ev, V = np.linalg.eigh(M); ev = np.clip(ev, 0, None)
                Pa[i] = ((V * ev ** ALPHA) @ V.T)[iu]
            store[m] = (1 - W) * sfilt(cks(Pa)) + W * cks(F)
            del F, P, Pa; gc.collect()
        print('built', m, flush=True)
    np.savez_compressed(CACHE, **{('n_' + m): KN[m] for m in MODS},
                        **{('k_' + m): KC[m] for m in MODS})
    print('clip holdout kernels cached', flush=True)


def ens(parts):
    tot = 0.0
    for m in MODS:
        tot = tot + WT[m] * parts[m]
    return tot / sum(WT.values())


K_NONE = ens(KN); K_CLIP = ens(KC)


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


def one(K, sd, frac=0.6):
    rs = np.random.RandomState(sd)
    order = rs.permutation(len(tr_all))
    k = int(len(tr_all) * frac)
    a = [idx[tr_all[i]] for i in order[:k]]
    b = [idx[s] for s in te_all]
    ya = np.array([ysub[tr_all[i]] for i in order[:k]])
    yb = np.array([ysub[s] for s in te_all])
    lam = pick(K[np.ix_(a, a)], ya)
    w, V = np.linalg.eigh(K[np.ix_(a, a)]); yc = ya - ya.mean()
    p = (K[np.ix_(b, a)] @ V) @ (V.T @ yc / (w + lam)) + ya.mean()
    return p, yb


res = {'n_train_pool': len(tr_all), 'n_test': len(te_all), 'seeds': 30,
       'note': 'one extraction pipeline throughout (the 999-subject block), so train and test share a representation -- unlike the earlier invalid attempt'}
P = {'none': [], 'clip': []}
for sd in range(30):
    for nm, K in (('none', K_NONE), ('clip', K_CLIP)):
        p, yb = one(K, sd)
        P[nm].append(p)
    if sd % 5 == 0:
        print('seed', sd, flush=True)

yb = np.array([ysub[s] for s in te_all])
for nm in P:
    rr = [float(np.corrcoef(p, yb)[0, 1]) for p in P[nm]]
    res[nm] = {'r_mean': round(float(np.mean(rr)), 4),
               'r_per_seed': [round(x, 4) for x in rr],
               'sd': round(float(np.std(rr, ddof=1)), 4)}

# paired per-subject absolute error (subject bootstrap, the valid interval)
d = np.array([np.abs(pn - yb) - np.abs(pc - yb) for pn, pc in zip(P['none'], P['clip'])])
m = d.mean(0)
rs = np.random.RandomState(7); bs = []
for _ in range(3000):
    i = rs.randint(0, len(m), len(m)); bs.append(m[i].mean())
bs = np.array(bs)
res['clip_abs_err_gain'] = {'mean': round(float(m.mean()), 4),
                             'ci95': [round(float(np.percentile(bs, 2.5)), 4),
                                      round(float(np.percentile(bs, 97.5)), 4)],
                             'p_two_sided': round(float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), 4)}
print('none %.4f  clip %.4f  abs-err gain %+.4f CI [%+.4f,%+.4f] p %.4f'
      % (np.mean([np.corrcoef(p, yb)[0, 1] for p in P['none']]),
         np.mean([np.corrcoef(p, yb)[0, 1] for p in P['clip']]),
         m.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
         2 * min((bs <= 0).mean(), (bs >= 0).mean())), flush=True)

# and a paired r difference over seeds
rn = np.array([np.corrcoef(p, yb)[0, 1] for p in P['none']])
rc = np.array([np.corrcoef(p, yb)[0, 1] for p in P['clip']])
res['clip_minus_none_r'] = {'mean': round(float((rc - rn).mean()), 4),
                             'wins': '%d/%d' % (int((rc > rn).sum()), len(rn))}
json.dump(res, open(BASE + '/CLIP_HOLDOUT_VALID.json', 'w'), indent=1)
print('DONE', flush=True)
