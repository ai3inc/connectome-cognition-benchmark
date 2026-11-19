"""DEVELOPMENT on the 753 (confirmation is on subjects outside the 753, separately).
Inductive spectral flattening of each modality kernel inside the training fold:
  K_tr = V diag(l) V^T  ->  V diag(l^p) V^T,  K_te,tr -> K_te,tr V diag(l^(p-1)) V^T
(the Nystrom-consistent extension), each modality rescaled to unit mean train
diagonal, then the six-modality mean. p = 1 is the plain ensemble. Also the same
operation on F-BERT's six correlation kernels.

  build: python3 dev_flatten.py build WORKSPACE PERMOD.npz REST_GLOB TASK_GLOB
  eval:  python3 dev_flatten.py eval  WORKSPACE PERMOD.npz OUT.json
"""
import csv, json, os, sys, time
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_samefolds as ev
import eval_sixmod as sx

PS = [1.0, 0.75, 0.5, 0.25]
GS = [0.95, 0.97]


def build(ws, outp, rest_pat, task_pat):
    t0 = time.time()
    rest = sx.load_feats(rest_pat, {"fc_full": "fc_rs", "e8_full": "e8_rs"})
    task = sx.load_feats(task_pat, {f"{k}_{m}": f"{k}_{m}" for m in sx.MODS[1:] for k in ["fc", "e8"]})
    oi = {r["Subject"].strip(): r for r in csv.DictReader(open(ws + "/ooi_comp.csv"))}
    off = [x.strip() for x in open(ws + "/off753.txt") if x.strip()]
    use = [s for s in off if s in oi and np.isfinite(float(oi[s].get("varimax_cog") or "nan"))
           and s in rest and s in task]
    y = np.array([float(oi[s]["varimax_cog"]) for s in use])
    ev.G_GRID = GS
    KD, KF = [], []
    for m in sx.MODS:
        src = rest if m == "rs" else task
        fc = np.stack([src[s][f"fc_{m}"] for s in use])
        res = [ev.per_subject(v) for v in fc]
        Kfull = ev.correlation_kernel(fc)
        KD.append([(1 - ev.W_FULL) * ev.correlation_kernel(np.stack([r[1][gi] for r in res])) + ev.W_FULL * Kfull
                   for gi in range(len(GS))])
        KF.append(ev.correlation_kernel(np.stack([src[s][f"e8_{m}"] for s in use]).astype(np.float64)))
        print(m, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(outp, KD=np.array(KD), KF=np.array(KF), y=y, use=np.array(use), gs=np.array(GS))
    print("SAVED", flush=True)


def flat(K, tr, te, p):
    Ktr = K[np.ix_(tr, tr)]
    w, V = np.linalg.eigh(Ktr)
    keep = w > 1e-10 * w.max()
    w, V = w[keep], V[:, keep]
    Kp_tr = (V * w ** p) @ V.T
    Kp_te = (K[np.ix_(te, tr)] @ V) @ (V * w ** (p - 1)).T
    s = np.mean(np.diag(Kp_tr))
    return Kp_tr / s, Kp_te / s


G = {}


def krr_blocks(cands, ytr, n_tr):
    """cands: list of (Ktr, Kte). Inner 4-fold on Ktr picks (cand, lambda)."""
    S = np.zeros((len(cands), len(ev.LAMBDAS)))
    for a, b in ev.inner_splits(n_tr):
        ya = ytr[a] - ytr[a].mean()
        for ci, (Ktr, _) in enumerate(cands):
            w, V = np.linalg.eigh(Ktr[np.ix_(a, a)])
            VtK = Ktr[np.ix_(b, a)] @ V
            Vt = V.T @ ya
            for li, lam in enumerate(ev.LAMBDAS):
                S[ci, li] += np.corrcoef(VtK @ (Vt / (w + lam)), ytr[b])[0, 1] / 4
    outs = []
    for ci, (Ktr, Kte) in enumerate(cands):
        lam = ev.LAMBDAS[int(np.nanargmax(S[ci]))]
        w, V = np.linalg.eigh(Ktr)
        outs.append((Kte @ V) @ (V.T @ (ytr - ytr.mean()) / (w + lam)) + ytr.mean())
    best = int(np.unravel_index(np.nanargmax(S), S.shape)[0])
    return outs, best


def run_seed(nm):
    y, folds_of, KD, KF = G["y"], G["folds"][nm], G["KD"], G["KF"]
    n = len(y)
    names = [f"D_g{g}_p{p}" for g in GS for p in PS] + [f"F_p{p}" for p in PS]
    pred = {k: np.full(n, np.nan) for k in names + ["Dsel", "Fsel"]}
    for te in folds_of:
        tr = np.setdiff1d(np.arange(n), te)
        cd, cf = [], []
        for gi, g in enumerate(GS):
            for p in PS:
                parts = [flat(KD[m][gi], tr, te, p) for m in range(6)]
                cd.append((sum(a for a, _ in parts) / 6, sum(b for _, b in parts) / 6))
        for p in PS:
            parts = [flat(KF[m], tr, te, p) for m in range(6)]
            cf.append((sum(a for a, _ in parts) / 6, sum(b for _, b in parts) / 6))
        od, bd = krr_blocks(cd, y[tr], len(tr))
        of, bf = krr_blocks(cf, y[tr], len(tr))
        for i, k in enumerate(names[:len(cd)]):
            pred[k][te] = od[i]
        for i, k in enumerate(names[len(cd):]):
            pred[k][te] = of[i]
        pred["Dsel"][te] = od[bd]; pred["Fsel"][te] = of[bf]
    return nm, {k: float(np.corrcoef(v, y)[0, 1]) for k, v in pred.items()}


def evaluate(ws, kp, outp):
    t0 = time.time()
    d = np.load(kp)
    use = list(d["use"])
    sidx = {s: i for i, s in enumerate(use)}
    folds = sio.loadmat(ws + "/splits.mat", squeeze_me=True, struct_as_record=False)["folds"]
    names = list(folds._fieldnames)[:int(os.environ.get("N_SEEDS", "60"))]
    fl = {nm: [te for te in (np.array([sidx[str(s)] for s in fold.subject_list if str(s) in sidx])
                             for fold in getattr(folds, nm).sub_fold) if len(te) >= 20] for nm in names}
    G.update(y=d["y"], folds=fl, KD=d["KD"], KF=d["KF"])
    with Pool(8) as pool:
        out = pool.map(run_seed, names)
    ps = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}
    R = {"stage": "development on the 753 (not a confirmation)",
         "arms": {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in ps.items()},
         "per_seed": {k: v.round(6).tolist() for k, v in ps.items()}, "seconds": time.time() - t0}
    json.dump(R, open(outp, "w"), indent=1)
    for k in sorted(R["arms"], key=lambda k: -R["arms"][k]["mean"]):
        print(f"{k:14s} {R['arms'][k]['mean']:.4f} +- {R['arms'][k]['sd']:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    else:
        evaluate(*sys.argv[2:5])
