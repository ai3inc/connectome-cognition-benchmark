"""Which region order does F-BERT expect? Decided without any behavioural label.

Candidates: cortex first or subcortex first; Schaefer 7- or 17-network ordering
(same 400 parcels, different order); Tian-S3 right-first (atlas file) or left-first.
Two independent checks on HCP subjects:
  1. correlation of the HCP group-mean FC with the corpus group-mean FC
     (corpus_meanfc.npy, from 1,024 pretraining windows);
  2. linear CKA between F-BERT embeddings (full scan) and each subject's own
     FC^0.35 teacher; the teacher Gram matrix does not depend on region order,
     so only the encoder's input order moves this number.

usage: python3 order_check.py TSDIR ATLAS7 ATLAS17 CORPUS_MEANFC FBERT_DIR OUT.json [NSUB]
"""
import glob, json, os, sys
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm
import hcp_prep as hp

tsdir, a7, a17, cfc, fdir, outp = sys.argv[1:7]
NSUB = int(sys.argv[7]) if len(sys.argv) > 7 else 40
torch.set_num_threads(int(os.environ.get("NT", "8")))

l7 = np.asarray(nib.load(a7).dataobj).ravel().astype(int)
l17 = np.asarray(nib.load(a17).dataobj).ravel().astype(int)
names7 = nib.load(a7).header.get_axis(0).label[0]
# 7-network label k (51..450) -> 17-network label with the largest overlap
perm17 = []
for k17 in range(51, 451):
    m = l7[l17 == k17]
    vals, cnt = np.unique(m[m > 0], return_counts=True)
    perm17.append(int(vals[cnt.argmax()]))
assert len(set(perm17)) == 400, "17-net mapping not a permutation"
c7 = list(range(50, 450))                 # file indices of cortex, 7-net order
c17 = [k - 1 for k in perm17]             # file indices of cortex, 17-net order
sub_r = list(range(0, 25)); sub_l = list(range(25, 50))
print("tian 1:", names7[1][0], " 26:", names7[26][0], " 51:", names7[51][0], flush=True)
cands = {}
for cn, c in [("7net", c7), ("17net", c17)]:
    for sn, s in [("tianRL", sub_r + sub_l), ("tianLR", sub_l + sub_r)]:
        cands[f"cortex_first|{cn}|{sn}"] = np.array(c + s)
        cands[f"subcortex_first|{cn}|{sn}"] = np.array(s + c)

files = sorted(glob.glob(os.path.join(tsdir, "*.npz")))[:NSUB]
fulls, Cs = [], []
for f in files:
    full, short, runs, lens = hp.load_subject(f)
    fulls.append(full)
    Cs.append(hp.pearson(full))
meanC = np.mean(Cs, 0)
T = np.stack([hp.teacher_vec(C) for C in Cs])
corpus = np.load(cfc)
iu = np.triu_indices(450, 1)


def cka(E, T):
    E = E.astype(np.float64)
    K, L = E @ E.T, T @ T.T
    n = len(K)
    H = np.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {t: fm.load(os.path.join(fdir, f"fbert_{t}.pth"), **best)[0] for t in ["8m", "1m"]}
res = {"n_subjects": len(files), "candidates": {}}
for name, o in cands.items():
    r_fc = float(np.corrcoef(meanC[np.ix_(o, o)][iu], corpus[iu])[0, 1])
    row = {"meanfc_r_vs_corpus": r_fc}
    for t, m in models.items():
        with torch.no_grad():
            E = np.stack([m(torch.from_numpy(x[o][None]))[0].numpy() for x in fulls])
        row[f"cka_teacher_{t}"] = cka(E, T)
    res["candidates"][name] = row
    print(name, json.dumps(row), flush=True)
# reference: teacher-teacher sanity and a random order
rs = np.random.default_rng(0)
o = rs.permutation(450)
row = {"meanfc_r_vs_corpus": float(np.corrcoef(meanC[np.ix_(o, o)][iu], corpus[iu])[0, 1])}
for t, m in models.items():
    with torch.no_grad():
        E = np.stack([m(torch.from_numpy(x[o][None]))[0].numpy() for x in fulls])
    row[f"cka_teacher_{t}"] = cka(E, T)
res["random_order_reference"] = row
print("random", json.dumps(row))
res["chosen"] = max(res["candidates"], key=lambda k: res["candidates"][k]["cka_teacher_8m"])
res["chosen_by_meanfc"] = max(res["candidates"], key=lambda k: res["candidates"][k]["meanfc_r_vs_corpus"])
res["chosen_order"] = cands[res["chosen"]].tolist()
json.dump(res, open(outp, "w"), indent=1)
print("chosen by CKA:", res["chosen"], " by mean FC:", res["chosen_by_meanfc"])
