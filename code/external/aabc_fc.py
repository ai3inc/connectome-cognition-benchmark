"""Assemble per-visit resting-state FC matrices from the AABC Release 2 REST FullCorrelation IDP archive.

The archive holds one CSV per parcel (HCP_MMP v1.0 cortical areas + FreeSurfer subcortical structures): rows are
subject visits (id_event, e.g. <visit id>), columns the correlation of that parcel with every parcel. This script
keeps first visits (V1) only, stacks the parcel files into one symmetric matrix per visit and writes the upper triangle.
It reads no phenotype file. Network-level files (one per functional network) are ignored.

usage: python3 aabc_fc.py ARCHIVE.zip OUT.npz
"""
import csv, io, sys, zipfile
import numpy as np

zp, outp = sys.argv[1:3]
z = zipfile.ZipFile(zp)
names = [n for n in z.namelist() if n.endswith(".csv")]
first = io.TextIOWrapper(z.open(names[0]), encoding="utf-8")
cols = next(csv.reader(first))[1:]                       # parcel names in matrix order
P = len(cols)
pref = "rfMRI_REST_FullCorrelationConnectivity_"
byname = {n[len(pref):-4].replace("-", "_"): n for n in names if n.startswith(pref)}   # file names use "-" where columns use "_"
missing = [c for c in cols if c not in byname]
assert not missing, f"no file for parcels {missing[:5]}"
ids, mat = None, None
for j, c in enumerate(cols):
    rows = list(csv.reader(io.TextIOWrapper(z.open(byname[c]), encoding="utf-8")))
    assert rows[0][1:] == cols, c
    body = [r for r in rows[1:] if r and r[0].endswith("_V1")]
    if ids is None:
        ids = [r[0] for r in body]
        mat = np.full((len(ids), P, P), np.nan, dtype=np.float32)
    assert [r[0] for r in body] == ids, c
    mat[:, j, :] = np.array([[float(x) if x not in ("", "NaN", "nan") else np.nan for x in r[1:]] for r in body], dtype=np.float32)
asym = float(np.nanmax(np.abs(mat - mat.transpose(0, 2, 1))))
diag = np.einsum("kii->ki", mat)
iu = np.triu_indices(P, 1)
edges = 0.5 * (mat + mat.transpose(0, 2, 1))[:, iu[0], iu[1]]
ok = np.isfinite(edges).all(1)
np.savez_compressed(outp, ids=np.array(ids), parcels=np.array(cols), edges=edges.astype(np.float32), complete=ok)
print({"visits_V1": len(ids), "parcels": P, "edges": int(edges.shape[1]), "complete": int(ok.sum()),
       "max_asymmetry": round(asym, 6), "diag_mean": float(np.nanmean(diag)), "edge_mean": float(np.nanmean(edges)),
       "edge_sd": float(np.nanstd(edges))})
