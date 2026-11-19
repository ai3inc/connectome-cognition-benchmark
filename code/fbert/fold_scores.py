"""Re-run an existing F-BERT comparison and also record per-fold scores (review, 2026-11-19).

The original scripts stored one pooled score per split, so the Nadeau-Bengio corrected t
could only be approximated from split means, and that approximation was found to be about
twice too large on data where both are available. This wrapper runs the ORIGINAL script
unchanged except for its per-split function, which is replaced by a copy that additionally
writes the per-fold score of every arm to FOLD_DIR/<tag>_<split>.json. The pooled scores the
script writes are unchanged, so its own output must reproduce the stored result file.

usage: FOLD_DIR=/workspace/folds python3 fold_scores.py samefolds|sixmod|aomic|abide <original argv...>
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from threadpoolctl import threadpool_limits

FOLD_DIR = os.environ.get("FOLD_DIR", "/workspace/folds")
os.makedirs(FOLD_DIR, exist_ok=True)
which = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]


def dump(tag, key, fold, sizes=None):
    json.dump({"split": key, "fold_scores": fold, "test_sizes": sizes},
              open(os.path.join(FOLD_DIR, f"{tag}_{key}.json"), "w"))


def r(p, y):
    return float(np.corrcoef(p, y)[0, 1])


if which == "samefolds":
    import eval_samefolds as M
    ev = M

    def run_seed(nm):
        y, folds_of, arms = M.G["y"], M.G["folds"][nm], M.G["arms"]
        n = len(y)
        pred = {a: np.full(n, np.nan) for a in arms}
        pred_g = {a: {gi: np.full(n, np.nan) for gi in range(len(M.G_GRID))} for a in arms if a.startswith("D")}
        gsel = {a: [] for a in arms if a.startswith("D")}
        fold = {a: [] for a in arms}
        for te in folds_of:
            tr = np.setdiff1d(np.arange(n), te)
            for a, spec in arms.items():
                Ks = [M.zlin_kernel(spec[1], tr)] if spec[0] == "zlin" else spec[1]
                p, kb, perk = M.krr_select(Ks, tr, te, y)
                pred[a][te] = p
                fold[a].append(r(p, y[te]))
                if a in gsel:
                    gsel[a].append(M.G_GRID[kb])
                    for gi, pk in enumerate(perk):
                        pred_g[a][gi][te] = pk
        dump("samefolds", nm, fold)
        rr = {a: float(np.corrcoef(pred[a], y)[0, 1]) for a in arms}
        rg = {a: [float(np.corrcoef(pred_g[a][gi], y)[0, 1]) for gi in range(len(M.G_GRID))] for a in pred_g}
        return nm, rr, rg, gsel
    M.run_seed = run_seed

elif which == "sixmod":
    import eval_sixmod as M
    ev = M.ev

    def run_seed(nm):
        y, folds_of, arms = M.G["y"], M.G["folds"][nm], M.G["arms"]
        n = len(y)
        pred = {a: np.full(n, np.nan) for a in arms}
        gsel = {a: [] for a in arms if a.startswith("D")}
        fold = {a: [] for a in arms}
        for te in folds_of:
            tr = np.setdiff1d(np.arange(n), te)
            for a, spec in arms.items():
                Ks = [ev.zlin_kernel(spec[1], tr)] if spec[0] == "zlin" else spec[1]
                p, kb, _ = ev.krr_select(Ks, tr, te, y)
                pred[a][te] = p
                fold[a].append(r(p, y[te]))
                if a in gsel:
                    gsel[a].append(ev.G_GRID[kb])
        dump("sixmod", nm, fold)
        return nm, {a: float(np.corrcoef(pred[a], y)[0, 1]) for a in arms}, gsel
    M.run_seed = run_seed

elif which == "aomic":
    import aomic_eval as M
    ev = M.ev

    def run_rep(s):
        with threadpool_limits(1):
            y, arms = M.G["y"], M.G["arms"]
            n = len(y)
            perm = np.random.RandomState(7000 + s).permutation(n)
            pred = {a: np.full(n, np.nan) for a in arms}
            fold = {a: [] for a in arms}
            sizes = []
            for te in np.array_split(perm, 10):
                sizes.append([int(len(te)), int(n - len(te))])
                tr = np.setdiff1d(np.arange(n), te)
                for a, spec in arms.items():
                    Ks = [ev.zlin_kernel(E, tr) if kind == "zlin" else E for kind, E in spec]
                    p, _, _ = ev.krr_select(Ks, tr, te, y)
                    pred[a][te] = p
                    fold[a].append(r(p, y[te]))
            dump("aomic", s, fold, sizes)
            return s, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}
    M.run_rep = run_rep

elif which == "abide":
    import abide_eval as M
    ev = M.ev

    def run_rep(args):
        s, task = args
        with threadpool_limits(1):
            y, idx, arms, sites = M.G[task]["y"], M.G[task]["idx"], M.G["arms"], M.G[task]["sites"]
            n = len(y)
            pred = {a: np.full(n, np.nan) for a in arms}
            fold = {a: [] for a in arms}
            sizes = []
            for te in M.site_folds(sites, s):
                sizes.append([int(len(te)), int(n - len(te))])
                tr = np.setdiff1d(np.arange(n), te)
                for a, spec in arms.items():
                    Ks = []
                    for kind, E in spec:
                        Esub = E[np.ix_(idx, idx)] if kind == "k" else E[idx]
                        Ks.append(ev.zlin_kernel(Esub, tr) if kind == "zlin" else Esub)
                    p = ev.krr_select(Ks, tr, te, y)[0]
                    pred[a][te] = p
                    yt = y[te]
                    ok = len(set(np.round(yt, 6))) > 1
                    fold[a].append((M.auc(p, yt) if task == "dx" else r(p, yt)) if ok else None)
            dump(f"abide_{task}", s, fold, sizes)
            if task == "dx":
                return s, task, {a: M.auc(v, y) for a, v in pred.items()}
            return s, task, {a: float(np.corrcoef(v, y)[0, 1]) for a, v in pred.items()}
    M.run_rep = run_rep
else:
    sys.exit("unknown target " + which)

M.main()
