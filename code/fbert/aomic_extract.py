"""AOMIC-ID1000 volumetric 450-region timeseries (prereg Addendum H).

For each participant: download the fMRIPrep MNI152NLin2009cAsym preproc BOLD and
brain mask (OpenNeuro ds003097, anonymous S3), map every voxel of Tian's combined
Schaefer-400 + Tian-S3 2 mm atlas to the BOLD voxel that contains it (weights =
number of atlas voxels; BOLD voxels inside the brain mask preferred, outside ones
kept only for a label with no in-mask voxel), parcel means = weighted average.
Raw parcel means stored (no filtering), plus the TR.

usage: python3 aomic_extract.py SUBJECT_LIST ATLAS_2mm.nii.gz OUTDIR [NPAR]
"""
import os, sys, subprocess, time, json
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

subs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
atlas_p, out = sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, "_dl"); os.makedirs(tmp, exist_ok=True)
A = nib.load(atlas_p)
lab = np.asarray(A.dataobj).astype(int)
R = int(lab.max())
vox = np.argwhere(lab > 0)
vlab = lab[lab > 0]
world = (A.affine @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]
BASE = "s3://openneuro.org/ds003097/derivatives/fmriprep"


def get(sid, suffix):
    key = f"{BASE}/{sid}/func/{sid}_task-moviewatching_space-MNI152NLin2009cAsym_{suffix}"
    dst = os.path.join(tmp, f"{sid}_{suffix}")
    for k in range(3):
        r = subprocess.run(["aws", "s3", "cp", "--no-sign-request", key, dst, "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            return dst
        time.sleep(3 * (k + 1))
    return None


def weights(bold_img, mask):
    inv = np.linalg.inv(bold_img.affine)
    ijk = np.rint((inv @ np.c_[world, np.ones(len(world))].T).T[:, :3]).astype(int)
    shp = bold_img.shape[:3]
    ok = np.all((ijk >= 0) & (ijk < np.array(shp)), 1)
    flat = np.ravel_multi_index(ijk[ok].T, shp)
    labs = vlab[ok]
    inmask = mask.ravel()[flat] > 0
    W = {}
    for L in range(1, R + 1):
        sel = labs == L
        use = sel & inmask
        if not use.any():
            use = sel
        if not use.any():
            continue
        u, c = np.unique(flat[use], return_counts=True)
        W[L] = (u, c / c.sum())
    return W


def one(sid):
    o = os.path.join(out, f"{sid}.npz")
    if os.path.exists(o):
        return sid, "skip"
    b = get(sid, "desc-preproc_bold.nii.gz")
    m = get(sid, "desc-brain_mask.nii.gz")
    if b is None or m is None:
        return sid, "missing"
    try:
        img = nib.load(b)
        data = np.asarray(img.dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(m).dataobj)
        W = weights(img, mask)
        T = data.shape[3]
        D = data.reshape(-1, T)
        ts = np.full((R, T), np.nan, np.float32)
        for L, (u, w) in W.items():
            ts[L - 1] = w @ D[u]
        tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else float("nan")
        np.savez(o, ts=ts, tr=tr, empty=np.array([L for L in range(1, R + 1) if L not in W]))
    finally:
        for p in (b, m):
            if p and os.path.exists(p):
                os.remove(p)
    return sid, f"ok T={T} tr={tr} empty={R - len(W)}"


with ThreadPoolExecutor(NP) as ex:
    for k, (sid, msg) in enumerate(ex.map(one, subs)):
        print(k + 1, sid, msg, flush=True)
print("FINISHED", flush=True)
