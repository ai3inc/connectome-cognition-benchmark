"""Second-atlas extraction for the multi-atlas ensemble: any CIFTI dlabel atlas on the
91,282 grayordinates (here Schaefer-200 + Tian S2, 232 regions). For each subject,
all 14 runs (rest 1-4, five tasks LR/RL), parcel means, the same cleaning as the
450-region line (hcp_prep.clean_run), and Pearson FC per modality: rest = the runs
concatenated, task = LR+RL concatenated. Only FC (strict upper triangle, float32)
is kept, so nothing large leaves the pod.

usage: python3 extract_atlas_fc.py SUBJECT_LIST ATLAS.dlabel.nii OUT.npz [NPAR]
"""
import os, sys, subprocess, time
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hcp_prep as hp

subs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
atlas, outp = sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 5
tmp = outp + "_dl"
os.makedirs(tmp, exist_ok=True)
lab = np.asarray(nib.load(atlas).dataobj).ravel().astype(int)
R = int(lab.max())
assert lab.shape[0] == 91282
counts = np.bincount(lab, minlength=R + 1)[1:]
assert (counts > 0).all()
M = np.zeros((R, 91282), np.float32)
M[lab[lab > 0] - 1, np.nonzero(lab > 0)[0]] = 1.0
M /= counts[:, None]
iu = np.triu_indices(R, 1)
REST = [f"rfMRI_{r}" for r in ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]]
TASKS = {"lang": "LANGUAGE", "social": "SOCIAL", "gamb": "GAMBLING", "motor": "MOTOR", "wm": "WM"}


def fetch(sid, run, fname):
    key = f"s3://hcp-openaccess/HCP_1200/{sid}/MNINonLinear/Results/{run}/{fname}"
    dst = os.path.join(tmp, f"{sid}_{run}.nii")
    for attempt in range(3):
        r = subprocess.run(["aws", "s3", "cp", key, dst, "--no-progress", "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            return dst
        if "404" in r.stderr or "does not exist" in r.stderr or "Not Found" in r.stderr:
            return None
        time.sleep(5 * (attempt + 1))
    return None


def parcel(path):
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if a.shape[1] != 91282:
        a = a.T
    return (M @ a.T).astype(np.float32)


def run_ts(sid, run, fname):
    p = fetch(sid, run, fname)
    if p is None:
        return None
    try:
        x = parcel(p)
    finally:
        os.remove(p)
    return hp.clean_run(x) if x.shape[1] >= 100 else None


def one(sid):
    out = {}
    segs = [run_ts(sid, r, f"{r}_Atlas_MSMAll_hp2000_clean.dtseries.nii") for r in REST]
    segs = [s for s in segs if s is not None]
    if segs:
        out["fc_rs"] = hp.pearson(np.concatenate(segs, 1))[iu].astype(np.float32)
    for m, T in TASKS.items():
        segs = [run_ts(sid, f"tfMRI_{T}_{pe}", f"tfMRI_{T}_{pe}_Atlas_MSMAll.dtseries.nii") for pe in ["LR", "RL"]]
        segs = [s for s in segs if s is not None]
        if segs:
            out[f"fc_{m}"] = hp.pearson(np.concatenate(segs, 1))[iu].astype(np.float32)
    return sid, out


res = {}
t0 = time.time()
with ThreadPoolExecutor(max_workers=NP) as ex:
    for k, (sid, o) in enumerate(ex.map(one, subs)):
        res[sid] = o
        print(k + 1, sid, sorted(o), f"{time.time() - t0:.0f}s", flush=True)
mods = ["rs"] + list(TASKS)
sids = [s for s in subs if all(f"fc_{m}" in res[s] for m in mods)]
np.savez(outp, sids=np.array(sids), regions=R,
         **{f"fc_{m}": np.stack([res[s][f"fc_{m}"] for s in sids]) for m in mods})
print("SAVED", outp, len(sids), "of", len(subs), flush=True)
