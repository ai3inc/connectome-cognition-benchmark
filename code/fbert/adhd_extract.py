"""ADHD-200 450-region timeseries (prereg Addendum L): fMRIPrep session-1 resting
runs from the public fcp-indi bucket, each run parcellated like aomic_extract.py
(Tian combined MNI152NLin2009cAsym 2 mm atlas -> BOLD grid, brain mask preferred).
Raw parcel means per run stored with the header TR.

usage: python3 adhd_extract.py RUN_LIST ATLAS_2mm.nii.gz OUTDIR [NPAR]
  RUN_LIST lines: <subject>\t<s3 key of preproc_bold> (several lines per subject)
"""
import os, sys, subprocess, time, collections
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

lines = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
runs = collections.OrderedDict()
for sid, key in lines:
    runs.setdefault(sid, []).append(key)
atlas_p, out = sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, "_dl"); os.makedirs(tmp, exist_ok=True)
A = nib.load(atlas_p)
lab = np.asarray(A.dataobj).astype(int)
R = int(lab.max())
vox = np.argwhere(lab > 0); vlab = lab[lab > 0]
world = (A.affine @ np.c_[vox, np.ones(len(vox))].T).T[:, :3]


def cp(key, dst):
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


def one(sid):
    o = os.path.join(out, f"{sid}.npz")
    if os.path.exists(o):
        return sid, "skip"
    tss, trs = [], []
    for key in runs[sid]:
        b = cp("s3://fcp-indi/" + key, os.path.join(tmp, os.path.basename(key)))
        mkey = key.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
        m = cp("s3://fcp-indi/" + mkey, os.path.join(tmp, os.path.basename(mkey)))
        if b is None or m is None:
            continue
        try:
            img = nib.load(b)
            data = np.asarray(img.dataobj, dtype=np.float32)
            W = weights(img, np.asarray(nib.load(m).dataobj))
            T = data.shape[3]; D = data.reshape(-1, T)
            ts = np.full((R, T), np.nan, np.float32)
            for L, (u, w) in W.items():
                ts[L - 1] = w @ D[u]
            tss.append(ts); z = img.header.get_zooms(); trs.append(float(z[3]) if len(z) > 3 else float("nan"))
        finally:
            for p in (b, m):
                if p and os.path.exists(p):
                    os.remove(p)
    if not tss:
        return sid, "missing"
    np.savez(o, **{f"ts{i}": t for i, t in enumerate(tss)}, trs=np.array(trs), nruns=len(tss))
    return sid, f"ok runs={len(tss)} T={[t.shape[1] for t in tss]} tr={trs}"


with ThreadPoolExecutor(NP) as ex:
    for k, (sid, msg) in enumerate(ex.map(one, list(runs))):
        print(k + 1, sid, msg, flush=True)
print("FINISHED", flush=True)
