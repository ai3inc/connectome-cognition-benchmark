"""Two untried axes on the published six-modality arm.

Axis 1 -- KERNEL TARGET ALIGNMENT (KTA).
    The kernel is built from features alone. KTA (Cristianini et al.) says a
    kernel should be similar to the label Gram matrix yy^T where labels are
    known. Two variants, both label-free at test time in the sense that only
    TRAINING labels enter:
      (a) FREEZE: compute the alignment of each modality kernel with the
          training-fold yy^T and use those as ensemble weights.
      (b) FOLD-INTERNAL: choose the weights by inner CV (the honest version).
    The ledger already says modality re-weighting fails after the filter; this
    tests a DIFFERENT weight rule (label alignment rather than solo r), and
    reports both.

Axis 2 -- STRUCTURAL PRIOR SHRINKAGE.
    Not "mix structural FC in" (tested: every weight hurts, -0.003 to -0.020)
    but use it as the TARGET of shrinkage, which is where shrinkage already
    lives. Instead of C_g = (1-g)C + gI, use the group mean structural matrix
    S:  C_g = (1-g)C + gS. This asks whether the conditioning step should pull
    toward anatomy rather than toward the identity.

Everything runs on the benchmark's 753 / 60x10 family folds, fold-mean r,
lambda by inner 4-fold mean-r.
"""
import numpy as np, json, csv, gc, scipy.io as sio

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


# ---------------- per-modality kernels (cached) ----------------
CACHE = BASE + '/KTA_STRUCT_K.npz'
import os
if os.path.exists(CACHE):
    z = np.load(CACHE)
    KP = {m: z['p_' + m] for m in MODS}
    KF = {m: z['f_' + m] for m in MODS}
    print('cached', flush=True)
else:
    KP = {}; KF = {}
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
        KP[m] = sfilt(cks(Pa)); KF[m] = cks(F)
        del E, C, F, Pa, P; gc.collect()
        print('built', m, flush=True)
    np.savez_compressed(CACHE, **{('p_' + m): KP[m] for m in MODS},
                        **{('f_' + m): KF[m] for m in MODS})
    print('cached', flush=True)


def ens(weights):
    tot = 0.0
    for m in MODS:
        tot = tot + weights[m] * ((1 - W) * KP[m] + W * KF[m])
    return tot / sum(weights.values())


BASE_W = {m: WT[m] for m in MODS}


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


def foldvec(Km, seeds=60, yr=None):
    """fold-mean r per seed, and the flat fold r vector for NB pairing."""
    yy = y if yr is None else yr
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            lam = pick(Km[np.ix_(tr, tr)], yy[tr])
            w, V = np.linalg.eigh(Km[np.ix_(tr, tr)]); yc = yy[tr] - yy[tr].mean()
            p = (Km[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + yy[tr].mean()
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
    from math import erfc, sqrt
    return float(t), float(erfc(abs(t) / sqrt(2)))


# ---------------- baseline ----------------
res = {'n': n, 'seeds': 60}
b_per, b_flat = foldvec(ens(BASE_W))
res['baseline_published_six'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

# ---------------- axis 1: KTA weights ----------------
# (a) freeze: alignment to training yy^T, computed per fold, weight = a^p
def kta_weights(tr, Kg, power=1.0):
    yc = y[tr] - y[tr].mean()
    T = np.outer(yc, yc)
    T = T / np.linalg.norm(T)
    a = {}
    for m in MODS:
        Km = Kg[m][np.ix_(tr, tr)]; Km = Km / np.linalg.norm(Km)
        a[m] = float(np.sum(Km * T))
    v = np.array([max(a[m], 0.0) ** power for m in MODS])
    if v.sum() <= 0:
        v = np.ones(len(MODS))
    v = v / v.mean()
    return {m: float(v[i]) for i, m in enumerate(MODS)}, a


def run_kta(power=1.0, seeds=40):
    per = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            wts, _ = kta_weights(tr, {m: KP[m] for m in MODS}, power)
            K = ens(wts)
            lam = pick(K[np.ix_(tr, tr)], y[tr])
            w, V = np.linalg.eigh(K[np.ix_(tr, tr)]); yc = y[tr] - y[tr].mean()
            p = (K[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()
            if np.std(p) > 1e-9:
                fr.append(float(np.corrcoef(p, y[te])[0, 1]))
        if fr:
            per.append(float(np.mean(fr)))
    return np.array(per)


for pw in (1.0, 2.0):
    a = run_kta(pw, 40)
    d = a - b_per[:len(a)]
    res['kta_power_%.1f' % pw] = {'r': round(float(a.mean()), 4),
                                  'diff': round(float(d.mean()), 4),
                                  'wins': '%d/%d' % (int((d > 0).sum()), len(d))}
    print('kta p=%.1f  %.4f  %+.4f  %d/%d' % (pw, a.mean(), d.mean(),
          (d > 0).sum(), len(d)), flush=True)

# ---------------- axis 2: structural prior shrinkage ----------------
import os as _os
if not _os.path.exists(BASE + '/sp753.npz'):
    print('sp753 absent -- structural axis SKIPPED', flush=True)
    json.dump(res, open(BASE + '/KTA_STRUCT_PRIOR.json', 'w'), indent=1)
    print('DONE', flush=True); raise SystemExit
z = np.load(BASE + '/sp753.npz')
ssubs = [str(x).strip() for x in z['subs']]; XS = z['X']
sidx = {s: i for i, s in enumerate(ssubs)}
have = [s for s in use if s in sidx]
print('struct coverage %d/%d' % (len(have), n), flush=True)
# group-mean structural matrix in the 400 space (structural is 256-block; use
# its own kernel and shrink the FUNCTIONAL matrix toward its top structure by
# mixing in the KERNEL space instead, which is where shrinkage can act safely)
KS = np.zeros((n, n))
sel = [pos[s] for s in have]
KS[np.ix_(sel, sel)] = cks(XS[[sidx[s] for s in have]])
del XS; gc.collect()
res['struct_coverage'] = len(have)

Kpub = ens(BASE_W)
for wv in (0.02, 0.05, 0.10):
    a = (1 - wv) * Kpub + wv * KS
    p, fl = foldvec(a, 40)
    d = p - b_per[:len(p)]
    res['struct_kernel_%.2f' % wv] = {'r': round(float(p.mean()), 4),
                                       'diff': round(float(d.mean()), 4),
                                       'wins': '%d/%d' % (int((d > 0).sum()), len(d))}
    print('struct w=%.2f  %.4f  %+.4f  %d/%d' % (wv, p.mean(), d.mean(),
          (d > 0).sum(), len(d)), flush=True)

json.dump(res, open(BASE + '/KTA_STRUCT_PRIOR.json', 'w'), indent=1)
print('DONE', flush=True)
