"""Pick the F-BERT forward-pass variant that reproduces the checkpoint's recorded loss.

Each checkpoint stores the training loss at its step (8M: step 12,000, 0.2115;
1M: step 56,000, 0.3051). The loss is 1 - linear CKA between the batch Gram matrix
of window embeddings and the batch Gram matrix of their recordings' teachers,
over 1,024 windows from 1,024 distinct recordings. We fetch a contiguous slice of
the public pretraining corpus (HTTP range reads, no full download), take one window
per recording, and compute that loss under every unfixed architectural switch.
The right variant must land near the recorded loss for BOTH checkpoints.

usage: python3 validate_variants.py WORKDIR OUT.json
"""
import io, json, os, sys, itertools, urllib.request
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm

BASE = "https://huggingface.co/datasets/Marraffini-Giovanni/fbert-corpus/resolve/main/"
work, outp = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
torch.set_num_threads(int(os.environ.get("NT", "8")))


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    assert len(data) == b - a, (url, a, b, len(data))
    return data


def header(name):
    raw = rng(BASE + name, 0, 4096)
    f = io.BytesIO(raw)
    ver = np.lib.format.read_magic(f)
    shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                             else np.lib.format.read_array_header_2_0(f))
    assert not fortran
    return shape, dtype, f.tell()


def rows(name, i0, i1, cache):
    p = os.path.join(work, cache)
    if os.path.exists(p):
        return np.load(p)
    shape, dtype, off = header(name)
    per = int(np.prod(shape[1:])) * dtype.itemsize
    buf = rng(BASE + name, off + i0 * per, off + i1 * per)
    a = np.frombuffer(buf, dtype=dtype).reshape((i1 - i0,) + tuple(shape[1:]))
    np.save(p, a)
    return a


w2r_p = os.path.join(work, "window_to_recording.npy")
if not os.path.exists(w2r_p):
    urllib.request.urlretrieve(BASE + "window_to_recording.npy", w2r_p)
w2r = np.load(w2r_p)
sorted_ok = bool(np.all(np.diff(w2r) >= 0))
print("windows", len(w2r), "recordings", int(w2r.max()) + 1, "sorted", sorted_ok, flush=True)

NB = 1024
# contiguous window slice large enough to hold NB distinct recordings
first = {}
for i, r in enumerate(w2r):
    if r not in first:
        first[int(r)] = i
    if len(first) >= NB:
        break
W = max(first.values()) + 1
recs = sorted(first)
R0, R1 = min(recs), max(recs) + 1
print("window slice 0..%d, recordings %d..%d" % (W, R0, R1), flush=True)
win = rows("windows.npy", 0, W, "win_slice.npy")
tea = rows("teachers.npy", R0, R1, "tea_slice.npy")
X = torch.from_numpy(np.stack([win[first[r]] for r in recs]).astype(np.float32))
T = torch.from_numpy(np.stack([tea[r - R0] for r in recs]).astype(np.float64))
print("batch", tuple(X.shape), "teacher", tuple(T.shape),
      "window z check mean %.3f sd %.3f" % (X.mean(-1).abs().mean(), X.std(-1).mean()), flush=True)


def cka(E, T):
    E = E.double()
    K, L = E @ E.T, T @ T.T
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm()))


grid = list(itertools.product([True, False], ["gelu", "relu"], ["interleave", "concat"], [False, True]))
res = {"recorded_loss": {}, "variants": [], "batch": NB, "window_slice": W,
       "recordings": [R0, R1], "sorted_w2r": sorted_ok}
for tag in ["8m", "1m"]:
    path = os.path.join(work, f"fbert_{tag}.pth")
    _, ck = fm.load(path)
    res["recorded_loss"][tag] = ck["loss"]
    for nf, act, lay, pes in grid:
        m, _ = fm.load(path, norm_first=nf, act=act, pe_layout=lay, pe_summary=pes)
        with torch.no_grad():
            E = torch.cat([m(X[i:i + 128]) for i in range(0, NB, 128)])
        c = cka(E, T)
        cn = cka(torch.nn.functional.normalize(E, dim=1), T)
        row = dict(model=tag, norm_first=nf, act=act, pe_layout=lay, pe_summary=pes,
                   loss=1 - c, loss_l2norm=1 - cn, recorded=ck["loss"])
        res["variants"].append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(outp, "w"), indent=1)
best = sorted(res["variants"], key=lambda r: abs(r["loss"] - r["recorded"]))[:6]
print("closest:")
for b in best:
    print(b)
