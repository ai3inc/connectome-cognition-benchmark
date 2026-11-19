"""Two axes the ledger does not have.

AXIS A -- RANK (Spearman) KERNEL.
    Every kernel in this project is built from raw edge values, which are
    heavy-tailed and subject to outlier edges. Build the subject x subject
    kernel from the RANK of each edge within each subject instead: the
    question becomes "do these two subjects rank their edges the same way",
    which is invariant to monotone per-edge rescaling and to a few extreme
    edges. Kernel construction is otherwise identical (HKH centring on the
    ranks, trace normalisation), so the comparison is like for like.

AXIS B -- LAMBDA-BAG KRR.
    The current rule picks one lambda by inner CV. At n=753 the ledger shows
    that fine hyperparameter choices cannot be recovered (keep, tau, alpha, w,
    per-modality g all fail in-fold), which is what a flat lambda axis looks
    like. Instead of choosing, AVERAGE the predictions of several lambdas that
    span the plateau. This removes a selection step rather than adding one, so
    it cannot be blamed on selection noise.

Both are evaluated on the benchmark's 753 / 60x10 folds, fold-mean r, and are
paired against the certified kernel fold by fold with a Nadeau-Bengio t.
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


def rank_rows(M):
    """Rank each ROW (a subject's edges) so two subjects are compared by the
    SHAPE of their edge profile, not its scale. Vectorised: argsort along
    axis 1 twice for the rank, which is the same as a per-row argsort."""
    o = np.argsort(np.argsort(np.asarray(M, dtype=np.float32), axis=1), axis=1)
    return (o.astype(np.float32) + 0.5) / M.shape[1]


# ---------------- build both kernel families ----------------
CACHE = BASE + '/RANKLAM_K.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    KR = {m: z['r_' + m] for m in MODS}     # rank/Spearman kernel
    KC = {m: z['c_' + m] for m in MODS}     # certified kernel (cached copy)
    print('cached', flush=True)
else:
    KR = {}; KC = {}
    for m in MODS:
        E = np.load(BASE + '/mod_%s.npz' % m)['bp'].astype(np.float32)
        C = np.zeros((n, 400, 400)); C[:, iu[0], iu[1]] = E
        C = C + np.transpose(C, (0, 2, 1)); C[:, dia, dia] = 1.0
        for i in range(n):
            d = np.sqrt(np.clip(np.diag(C[i]), 1e-12, None)); C[i] = C[i] / np.outer(d, d)
        C[:, dia, dia] = 1.0
        F = C[:, iu[0], iu[1]].copy()
        # certified
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
        KC[m] = (1 - W) * sfilt(cks(Pa)) + W * cks(F)
        # rank version of the SAME pipeline: rank the partial edges first
        Pr = rank_rows(Pa)
        KR[m] = (1 - W) * sfilt(cks(Pr)) + W * cks(rank_rows(F))
        del E, C, F, Pa, P, Pr; gc.collect()
        print('built', m, flush=True)
    np.savez_compressed(CACHE, **{('r_' + m): KR[m] for m in MODS},
                        **{('c_' + m): KC[m] for m in MODS})
    print('cached', flush=True)


def ens(parts):
    tot = 0.0
    for m in MODS:
        tot = tot + WT[m] * parts[m]
    return tot / sum(WT.values())


KCERT = ens(KC); KRANK = ens(KR)


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


def run(Km, mode='pick', seeds=60):
    """mode='pick': one lambda by inner CV (certified).
       mode='bag':  average the standardised predictions of the plateau lambdas."""
    per = []; flat = []
    bag_lams = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            Ktr = Km[np.ix_(tr, tr)]; Kte = Km[np.ix_(te, tr)]
            w, V = np.linalg.eigh(Ktr); yc = y[tr] - y[tr].mean()
            VT = V.T @ yc; KT = Kte @ V
            if mode == 'pick':
                lam = pick(Ktr, y[tr])
                p = KT @ (VT / (w + lam)) + y[tr].mean()
            else:
                ps = [KT @ (VT / (w + l)) for l in bag_lams]
                ps = np.array(ps); ps = (ps - ps.mean(1, keepdims=True)) / (ps.std(1, keepdims=True) + 1e-12)
                p = ps.mean(0)
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


res = {'n': n, 'seeds': 60}
b_per, b_flat = run(KCERT, 'pick')
res['baseline_certified'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

a, af = run(KCERT, 'bag')
d = a - b_per
t, p = nb_t(af, b_flat)
res['lambda_bag'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                     'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                     'NB_t': round(t, 2), 'p': round(p, 4)}
print('lambda-bag %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

a, af = run(KRANK, 'pick')
d = a - b_per
t, p = nb_t(af, b_flat)
res['rank_kernel'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                      'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                      'NB_t': round(t, 2), 'p': round(p, 4)}
print('rank-kernel %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

# and the two together
MIX = 0.5 * (KCERT / np.trace(KCERT) + KRANK / np.trace(KRANK))
MIX = MIX / np.trace(MIX)
a, af = run(MIX, 'pick')
d = a - b_per
t, p = nb_t(af, b_flat)
res['certified_plus_rank_mix'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                                  'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                                  'NB_t': round(t, 2), 'p': round(p, 4)}
print('mix %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

json.dump(res, open(BASE + '/RANKLAM.json', 'w'), indent=1)
print('DONE', flush=True)
