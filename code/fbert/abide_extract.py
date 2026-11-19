"""ABIDE-I 450-region timeseries (prereg Addendum J): CPAC filt_noglobal func_preproc
(MNI152 FSL space) + func_mask from the public fcp-indi bucket, Tian's
Schaefer-400 + Tian-S3 MNI152NLin6Asym 2 mm atlas mapped to the BOLD grid exactly as
aomic_extract.py (atlas voxel -> containing BOLD voxel, mask voxels preferred).
Raw parcel means stored with the header TR.

usage: python3 abide_extract.py FILE_ID_LIST ATLAS_2mm.nii.gz OUTDIR [NPAR]
"""
import os, sys, subprocess, time
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
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
BASE = "s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal"


def get(fid, kind):
    key = f"{BASE}/{kind}/{fid}_{kind}.nii.gz"
    dst = os.path.join(tmp, f"{fid}_{kind}.nii.gz")
    for k in range(3):
        r = subprocess.run(["aws", "s3", "cp", "--no-sign-request", key, dst, "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            return dst
        time.sleep(3 * (k + 1))
    return None


def weights(img, mask):
    inv = np.linalg.inv(img.affine)
    ijk = np.rint((inv @ np.c_[world, np.ones(len(world))].T).T[:, :3]).astype(int)
    shp = img.shape[:3]
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


def one(fid):
    o = os.path.join(out, f"{fid}.npz")
    if os.path.exists(o):
        return fid, "skip"
    b = get(fid, "func_preproc"); m = get(fid, "func_mask")
    if b is None or m is None:
        return fid, "missing"
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
        z = img.header.get_zooms()
        tr = float(z[3]) if len(z) > 3 else float("nan")
        np.savez(o, ts=ts, tr=tr)
    finally:
        for p in (b, m):
            if p and os.path.exists(p):
                os.remove(p)
    return fid, f"ok T={T} tr={tr} empty={R - len(W)}"


with ThreadPoolExecutor(NP) as ex:
    for k, (fid, msg) in enumerate(ex.map(one, ids)):
        print(k + 1, fid, msg, flush=True)
print("FINISHED", flush=True)
