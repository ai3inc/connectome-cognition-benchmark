"""HCP-YA resting-state parcel timeseries at Schaefer-400 + Tian-S3 (450 regions).

For each subject: the four rfMRI runs (REST1_LR, REST1_RL, REST2_LR, REST2_RL),
file rfMRI_<run>_Atlas_MSMAll_hp2000_clean.dtseries.nii (ICA-FIX cleaned, MSMAll),
averaged within each of the 450 labels of the Tian combined CIFTI atlas
(label order of the atlas file: 1-50 subcortex, 51-450 Schaefer cortex).

The output is the RAW parcel mean (no band-pass, no z-score) so every later
preprocessing choice can be changed without downloading 1.3 TB again.

usage: python3 extract_hcp_ts.py SUBJECT_LIST ATLAS.dlabel.nii OUTDIR [NPAR]
writes OUTDIR/<sid>.npz with ts (nruns, 450, T) float32, runs, lengths
"""
import os, sys, subprocess, time, json
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor

subs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
atlas, out = sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, "_dl")
os.makedirs(tmp, exist_ok=True)
RUNS = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]

lab = np.asarray(nib.load(atlas).dataobj).ravel().astype(int)
assert lab.shape[0] == 91282 and lab.max() == 450
# indicator matrix, rows = labels 1..450, each row averages its grayordinates
counts = np.bincount(lab, minlength=451)[1:]
assert (counts > 0).all(), "empty label"
M = np.zeros((450, 91282), np.float32)
M[lab[lab > 0] - 1, np.nonzero(lab > 0)[0]] = 1.0
M /= counts[:, None]


def fetch(sid, run):
    key = f"s3://hcp-openaccess/HCP_1200/{sid}/MNINonLinear/Results/rfMRI_{run}/rfMRI_{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    dst = os.path.join(tmp, f"{sid}_{run}.nii")
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
    assert a.shape[1] == 91282, a.shape
    return (M @ a.T).astype(np.float32)  # (450, T)


def one(sid):
    o = os.path.join(out, f"{sid}.npz")
    if os.path.exists(o):
        return sid, "skip"
    tss, runs, lens = [], [], []
    t0 = time.time()
    for run in RUNS:
        p = fetch(sid, run)
        if p is None:
            continue
        try:
            ts = parcel(p)
        finally:
            os.remove(p)
        tss.append(ts); runs.append(run); lens.append(ts.shape[1])
    if not tss:
        return sid, "none"
    T = max(lens)
    arr = np.full((len(tss), 450, T), np.nan, np.float32)
    for i, ts in enumerate(tss):
        arr[i, :, :ts.shape[1]] = ts
    np.savez(o + ".tmp.npz", ts=arr, runs=np.array(runs), lengths=np.array(lens))
    os.replace(o + ".tmp.npz", o)
    return sid, f"ok runs={len(runs)} lens={lens} {time.time() - t0:.0f}s"


print(json.dumps({"subjects": len(subs), "parallel": NP, "labels": 450,
                  "min_count": int(counts.min())}), flush=True)
done = 0
with ThreadPoolExecutor(max_workers=NP) as ex:
    for sid, msg in ex.map(one, subs):
        done += 1
        print(done, sid, msg, flush=True)
print("FINISHED", flush=True)
