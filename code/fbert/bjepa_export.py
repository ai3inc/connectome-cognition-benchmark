"""Export Brain-JEPA input windows (first 160 timepoints of each resting run, raw
parcel means in atlas order) from a pod's stored timeseries, so the heavy encoder
can run elsewhere. Light: reads npz files, writes one small npz.

usage: python3 bjepa_export.py TSDIR OUT.npz
"""
import glob, os, sys
import numpy as np

tsdir, outp = sys.argv[1:3]
sids, wins, nwin = [], [], []
for f in sorted(glob.glob(os.path.join(tsdir, "*.npz"))):
    d = np.load(f)
    ts, lens = d["ts"], list(d["lengths"])
    W = np.full((4, 450, 160), np.nan, np.float32)
    k = 0
    for i in range(len(lens)):
        if lens[i] >= 160 and k < 4:
            W[k] = ts[i, :, :160]; k += 1
    sids.append(os.path.basename(f)[:-4]); wins.append(W); nwin.append(k)
np.savez(outp, sids=np.array(sids), windows=np.stack(wins), nwin=np.array(nwin))
print("SAVED", outp, len(sids))
