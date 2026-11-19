"""HCP-YA task-fMRI parcel timeseries at 450 regions (Addendum A of the prereg).

For each subject and each task in LANGUAGE, SOCIAL, GAMBLING, MOTOR, WM: runs
tfMRI_<T>_{LR,RL}_Atlas_MSMAll.dtseries.nii averaged within the Tian combined
atlas labels (file order). Raw parcel means, no filtering.

usage: python3 extract_hcp_task_ts.py SUBJECT_LIST ATLAS.dlabel.nii OUTDIR [NPAR]
writes OUTDIR/<sid>.npz with, per task T: ts_<T>_LR / ts_<T>_RL (450, T) float32
(missing runs simply absent)
"""
import os, sys, subprocess, time
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

subs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
atlas, out = sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
TASKS = os.environ.get("TASKS", "LANGUAGE,SOCIAL,GAMBLING,MOTOR,WM").split(",")
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, "_dl")
os.makedirs(tmp, exist_ok=True)

lab = np.asarray(nib.load(atlas).dataobj).ravel().astype(int)
assert lab.shape[0] == 91282 and lab.max() == 450
counts = np.bincount(lab, minlength=451)[1:]
M = np.zeros((450, 91282), np.float32)
M[lab[lab > 0] - 1, np.nonzero(lab > 0)[0]] = 1.0
M /= counts[:, None]


def fetch(sid, task, pe):
    key = (f"s3://hcp-openaccess/HCP_1200/{sid}/MNINonLinear/Results/tfMRI_{task}_{pe}/"
           f"tfMRI_{task}_{pe}_Atlas_MSMAll.dtseries.nii")
    dst = os.path.join(tmp, f"{sid}_{task}_{pe}.nii")
    for attempt in range(3):
        r = subprocess.run(["aws", "s3", "cp", key, dst, "--no-progress", "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            return dst
        if "404" in r.stderr or "Not Found" in r.stderr or "does not exist" in r.stderr:
            return None
        time.sleep(5 * (attempt + 1))
    return None


def parcel(path):
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if a.shape[1] != 91282:
        a = a.T
    return (M @ a.T).astype(np.float32)


def one(sid):
    o = os.path.join(out, f"{sid}.npz")
    if os.path.exists(o):
        return sid, "skip"
    res, t0 = {}, time.time()
    for task in TASKS:
        for pe in ["LR", "RL"]:
            p = fetch(sid, task, pe)
            if p is None:
                continue
            try:
                res[f"ts_{task}_{pe}"] = parcel(p)
            finally:
                os.remove(p)
    if not res:
        return sid, "none"
    np.savez(o + ".tmp.npz", **res)
    os.replace(o + ".tmp.npz", o)
    return sid, f"ok runs={len(res)} {time.time() - t0:.0f}s"


done = 0
with ThreadPoolExecutor(max_workers=NP) as ex:
    for sid, msg in ex.map(one, subs):
        done += 1
        print(done, sid, msg, flush=True)
print("FINISHED", flush=True)
