"""Second check of the rebuilt forward pass: random training-like batches.

validate_variants.py used the first 1,024 recordings of the corpus (a few datasets).
Training batches are 1,024 windows from 1,024 random distinct recordings, so the
recorded loss is one draw from a spread. Here: NB random batches, one random window
per recording, range-read from the public corpus, loss of the chosen variant and of
two small residual switches (tanh-approximate GELU, LayerNorm eps 1e-6).

usage: python3 validate_random.py WORKDIR OUT.json [NBATCH]
"""
import io, json, os, sys, urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm

BASE = "https://huggingface.co/datasets/Marraffini-Giovanni/fbert-corpus/resolve/main/"
work, outp = sys.argv[1], sys.argv[2]
NBATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_num_threads(8)


def rng(url, a, b):
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except Exception:
            pass
    raise RuntimeError("range read failed")


def header(name):
    f = io.BytesIO(rng(BASE + name, 0, 4096))
    ver = np.lib.format.read_magic(f)
    shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                       else np.lib.format.read_array_header_2_0(f))
    return shape, dtype, f.tell()


wsh, wdt, woff = header("windows.npy")
tsh, tdt, toff = header("teachers.npy")
wper = int(np.prod(wsh[1:])) * wdt.itemsize
tper = int(np.prod(tsh[1:])) * tdt.itemsize
w2r = np.load(os.path.join(work, "window_to_recording.npy"))
nrec = int(w2r.max()) + 1
starts = np.searchsorted(w2r, np.arange(nrec))
ends = np.searchsorted(w2r, np.arange(nrec), side="right")


def get_pair(args):
    r, w = args
    x = np.frombuffer(rng(BASE + "windows.npy", woff + w * wper, woff + (w + 1) * wper), wdt).reshape(wsh[1:])
    t = np.frombuffer(rng(BASE + "teachers.npy", toff + r * tper, toff + (r + 1) * tper), tdt)
    return x, t


def cka(E, T):
    E = E.double()
    K, L = E @ E.T, T @ T.T
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm()))


best = dict(norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True)
models = {}
for tag in ["8m", "1m"]:
    p = os.path.join(work, f"fbert_{tag}.pth")
    m, ck = fm.load(p, **best)
    models[(tag, "chosen")] = (m, ck["loss"])
    m2, _ = fm.load(p, **dict(best, act=lambda x: F.gelu(x, approximate="tanh")))
    models[(tag, "gelu_tanh")] = (m2, ck["loss"])
    m3, _ = fm.load(p, **best)
    for mod in m3.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            mod.eps = 1e-6
    models[(tag, "eps1e-6")] = (m3, ck["loss"])

rs = np.random.default_rng(20261117)
out = {"variant": best, "batches": []}
for b in range(NBATCH):
    recs = rs.choice(nrec, 1024, replace=False)
    wins = [int(rs.integers(starts[r], ends[r])) for r in recs]
    with ThreadPoolExecutor(32) as ex:
        pairs = list(ex.map(get_pair, zip(recs.tolist(), wins)))
    X = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float32))
    T = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.float64))
    row = {"batch": b}
    for (tag, name), (m, rec) in models.items():
        with torch.no_grad():
            E = torch.cat([m(X[i:i + 128]) for i in range(0, 1024, 128)])
        row[f"{tag}_{name}"] = 1 - cka(E, T)
    out["batches"].append(row)
    print(json.dumps(row), flush=True)
summ = {}
for k in out["batches"][0]:
    if k == "batch":
        continue
    v = np.array([r[k] for r in out["batches"]])
    summ[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
out["summary"] = summ
out["recorded"] = {"8m": models[("8m", "chosen")][1], "1m": models[("1m", "chosen")][1]}
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(summ, indent=1), out["recorded"])
