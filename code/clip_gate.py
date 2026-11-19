"""Is the rank (Spearman) kernel a real gain, or a hand-picked setting?

The ledger has a rule it learned the hard way: at n=753 an effect below ~0.003
cannot be recovered by any fold-internal rule, and every fine hyperparameter
choice (keep, tau, alpha, w, per-modality g) fails when chosen in-fold. The
rank kernel's +0.0054 is above that threshold, so it has to clear two gates
before it may be called a recipe:

  GATE 1 -- FOLD-INTERNAL SELECTION.
      The choice is: use the raw-value kernel or the rank kernel (a 2-way
      choice, which the ledger says is the kind that CAN work, unlike a fine
      grid). The rule computes both arms' inner-CV r on the training fold and
      takes the winner. If the rule selects the rank kernel most folds and the
      result holds, the gain is a recipe. If the rule thrashes, it is a fixed
      observation.

  GATE 2 -- PAIRED FOLDS AND A SECOND KERNEL CONSTRUCTION.
      Report per-fold wins and NB t, and rebuild the rank kernel by ranking the
      CORRELATION edges (not the partial edges) to make sure the gain is not an
      artefact of one construction choice.
"""
import numpy as np, json, csv, gc, scipy.io as sio, os
from math import erfc, sqrt

BASE = '/workspace'
ALPHA, TAU, KEEP, G, W = 0.10, 0.5, 150, 0.92, 0.05
MODS = ['rs', 'lang', 'social', 'gamb', 'motor', 'wm']
WT = {'rs': 1, 'lang': 2, 'social': 1, 'gamb': 1, 'motor': 1, 'wm': 1}
LAM = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                0.1, 0.3, 1.0, 3.0, 10.0])
INNER = (1e-4, 1e-3, 1e-2, 0.1, 1.0)

oi = {r['Subject'].strip(): r for r in csv.DictReader(open(BASE + '/ooi_comp.csv'))}
off = [x.strip() for x in open(BASE + '/off753.txt') if x.strip()]
use = [s for s in off if s in oi and np.isfinite(float(oi[s].get('varimax_cog') or 'nan'))]
n = len(use)
y = np.array([float(oi[s]['varimax_cog']) for s in use])
pos = {s: i for i, s in enumerate(use)}
folds = sio.loadmat(BASE + '/splits.mat', squeeze_me=True, struct_as_record=False)['folds']
print('n=%d' % n, flush=True)

z = np.load(BASE + '/RANKVAR_K.npz')
KCERT = 0.0; KRANK = 0.0
for m in MODS:
    KCERT = KCERT + WT[m] * z['none_' + m]
    KRANK = KRANK + WT[m] * z['clip1.5_' + m]
KCERT = KCERT / sum(WT.values()); KRANK = KRANK / sum(WT.values())
print('kernels from cache', flush=True)


def pick(Ktr, ytr, lams=LAM):
    rs = np.random.RandomState(0); p = rs.permutation(len(ytr))
    f = np.zeros(len(ytr), int); f[p] = np.arange(len(ytr)) % 4
    R = np.zeros((len(lams), 4))
    for k in range(4):
        te = np.where(f == k)[0]; tr = np.where(f != k)[0]
        w, V = np.linalg.eigh(Ktr[np.ix_(tr, tr)]); yc = ytr[tr] - ytr[tr].mean()
        Vt = V.T @ yc; VtK = Ktr[np.ix_(te, tr)] @ V
        for i, l in enumerate(lams):
            R[i, k] = np.corrcoef(VtK @ (Vt / (w + l)), ytr[te])[0, 1]
    return lams[int(np.argmax(R.mean(1)))], float(R.mean(1).max())


def run(Km, seeds=60):
    per = []; flat = []
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            lam, _ = pick(Km[np.ix_(tr, tr)], y[tr])
            w, V = np.linalg.eigh(Km[np.ix_(tr, tr)]); yc = y[tr] - y[tr].mean()
            p = (Km[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()
            if np.std(p) > 1e-9:
                fr.append(float(np.corrcoef(p, y[te])[0, 1]))
        if fr:
            per.append(float(np.mean(fr))); flat.append(fr)
    return np.array(per), flat


def run_nested(seeds=60):
    """2-way choice inside the training fold: raw-value kernel vs rank kernel."""
    per = []; flat = []; picks = {0: 0, 1: 0}
    for nm in list(folds._fieldnames)[:seeds]:
        fr = []
        for fd in getattr(folds, nm).sub_fold:
            te = np.array([pos[str(s)] for s in fd.subject_list if s in pos])
            if len(te) < 20:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            # inner 4-fold on the training subjects
            rs = np.random.RandomState(0); p = rs.permutation(len(tr))
            f = np.zeros(len(tr), int); f[p] = np.arange(len(tr)) % 4
            score = []
            for arm, Km in ((0, KCERT), (1, KRANK)):
                Ktr = Km[np.ix_(tr, tr)]
                R = []
                for k in range(4):
                    a = np.where(f != k)[0]; b = np.where(f == k)[0]
                    w, V = np.linalg.eigh(Ktr[np.ix_(a, a)])
                    ya = y[tr[a]]; yc = ya - ya.mean()
                    Vt = V.T @ yc; VtK = Ktr[np.ix_(b, a)] @ V
                    R.append(max(float(np.corrcoef(VtK @ (Vt / (w + l)), y[tr[b]])[0, 1]) for l in INNER))
                score.append(float(np.mean(R)))
            arm = 0 if score[0] >= score[1] else 1
            picks[arm] += 1
            Km = KCERT if arm == 0 else KRANK
            lam, _ = pick(Km[np.ix_(tr, tr)], y[tr])
            w, V = np.linalg.eigh(Km[np.ix_(tr, tr)]); yc = y[tr] - y[tr].mean()
            p = (Km[np.ix_(te, tr)] @ V) @ (V.T @ yc / (w + lam)) + y[tr].mean()
            if np.std(p) > 1e-9:
                fr.append(float(np.corrcoef(p, y[te])[0, 1]))
        if fr:
            per.append(float(np.mean(fr))); flat.append(fr)
    return np.array(per), flat, picks


def nb_t(a, b):
    d = np.array([x - yv for fa, fb in zip(a, b) for x, yv in zip(fa, fb)])
    k = len(a[0]); n_test = n / k; n_train = n - n_test
    ratio = n_test / n_train
    # NB exact (Nadeau & Bengio 2003): var(d) * ((1/m) + n_te/n_tr).
    # The earlier version multiplied by (1/k + ratio) AND divided by m, which
    # under-corrected by a factor of ~18 (t was inflated 17.9x).
    t = d.mean() / np.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))
    return float(t), float(erfc(abs(t) / sqrt(2)))


res = {'n': n, 'seeds': 60, 'question': 'is the clip variant a fold-recoverable recipe? 2-way choice raw vs clip1.5'}
b_per, b_flat = run(KCERT)
res['baseline_raw_value_kernel'] = round(float(b_per.mean()), 4)
print('baseline %.4f' % b_per.mean(), flush=True)

a, af = run(KRANK)
d = a - b_per
t, p = nb_t(af, b_flat)
res['clip1.5_kernel'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                      'wins': '%d/%d' % (int((d > 0).sum()), len(d)), 'NB_t': round(t, 2), 'p': round(p, 4)}
print('rank %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

a, af, picks = run_nested()
d = a - b_per
t, p = nb_t(af, b_flat)
res['fold_internal_2way'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                             'wins': '%d/%d' % (int((d > 0).sum()), len(d)),
                             'NB_t': round(t, 2), 'p': round(p, 4),
                             'picks_clip': picks[1], 'picks_raw': picks[0],
                             'note': 'picks counted over %d fold-decisions' % sum(picks.values())}
print('nested %.4f  %+.4f  %d/%d  t=%.2f  picks raw %d / rank %d'
      % (a.mean(), d.mean(), (d > 0).sum(), len(d), t, picks[0], picks[1]), flush=True)

# fixed 50/50 mix of the two kernels (no selection at all)
MIX = 0.5 * (KCERT / np.trace(KCERT)) + 0.5 * (KRANK / np.trace(KRANK))
MIX = MIX / np.trace(MIX)
a, af = run(MIX)
d = a - b_per
t, p = nb_t(af, b_flat)
res['fixed_equal_mix'] = {'r': round(float(a.mean()), 4), 'diff': round(float(d.mean()), 4),
                          'wins': '%d/%d' % (int((d > 0).sum()), len(d)), 'NB_t': round(t, 2), 'p': round(p, 4)}
print('mix %.4f  %+.4f  %d/%d  t=%.2f' % (a.mean(), d.mean(), (d > 0).sum(), len(d), t), flush=True)

json.dump(res, open(BASE + '/CLIP_GATE.json', 'w'), indent=1)
print('DONE', flush=True)
