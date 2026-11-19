"""Addendum P: pretrain C-BERT (our teacher) next to an F-BERT reproduction (FC^0.35 teacher)
and a mixed-teacher model, in lockstep on the same batches, on F-BERT's public corpus with
every AOMIC dataset removed.

Architecture = F-BERT-8M as rebuilt in fbert_model.py (pre-LN, GELU, interleaved sinusoid,
summary token at position 0). Loss = 1 - linear CKA between the batch Gram of embeddings and
the batch Gram of teacher vectors (1,024 windows from 1,024 distinct recordings).

usage: python cbert_train.py CORPUS_DIR OUT_DIR [STEPS]
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbert_model as fm

corpus, out = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
EXCLUDE = ("ds003097", "ds002785", "ds002790")      # AOMIC-ID1000, PIOP1, PIOP2
G_SHRINK, ALPHA, W_FULL = 0.9, 0.10, 0.05
BATCH, LR, WD, WARM = 1024, float(os.environ.get("LR", "8.5e-4")), 1e-4, 100
DIM, LAYERS, HEADS, FF = (int(os.environ.get(k, v)) for k, v in [("DIM", "384"), ("LAYERS", "8"), ("HEADS", "8"), ("FF", "512")])
TAG = os.environ.get("TAG", "8m")
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")
if os.environ.get("NO_CUDNN_SDP"):
    # torch 2.11 + B300: cuDNN SDPA builds no execution plan; use flash / memory-efficient kernels
    torch.backends.cuda.enable_cudnn_sdp(False)
torch.manual_seed(0); rs = np.random.RandomState(0)
torch.set_num_threads(int(os.environ.get("NT", "4")))
log = open(os.path.join(out, f"train_{os.environ.get('TAG', '8m')}.log"), "a")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


# ---------------- data ----------------
rec = json.load(open(os.path.join(corpus, "recordings.json")))
w2r = np.load(os.path.join(corpus, "window_to_recording.npy"))
W = np.load(os.path.join(corpus, "windows.npy"), mmap_mode="r")
keep_rec = np.array([not r.split("/")[0] in EXCLUDE for r in rec])
nrec = len(rec)
starts = np.searchsorted(w2r, np.arange(nrec)); ends = np.searchsorted(w2r, np.arange(nrec), side="right")
usable = np.nonzero(keep_rec & (ends > starts))[0]
perm = rs.permutation(usable)
nval = int(0.05 * len(perm))
val_rec, tr_rec = np.sort(perm[:nval]), np.sort(perm[nval:])
say(f"recordings {nrec}, excluded AOMIC {int((~keep_rec).sum())}, train {len(tr_rec)}, val {len(val_rec)}")
t0 = time.time()
# chunked transfer without memory mapping: read the .npy body directly, ~150 MB at a time,
# drop the page cache after each chunk (shared-pod condition: host RSS small)
import threading
RSS_LIMIT_GB = float(os.environ.get("RSS_LIMIT_GB", "0"))


def _rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1e6
    return 0.0


def _watchdog():
    while True:
        if RSS_LIMIT_GB and _rss_gb() > RSS_LIMIT_GB:
            say(f"RSS {_rss_gb():.2f} GB > limit {RSS_LIMIT_GB} GB - exiting")
            os._exit(3)
        time.sleep(5)


if RSS_LIMIT_GB:
    threading.Thread(target=_watchdog, daemon=True).start()
shape = W.shape; per = 450 * 80 * 4
hdr_off = W.offset
del W
Wg = torch.empty((shape[0], 450, 80), dtype=torch.float16, device=dev)
CH = 1000
with open(os.path.join(corpus, "windows.npy"), "rb", buffering=0) as fh:
    fd = fh.fileno()
    for i in range(0, shape[0], CH):
        n = min(CH, shape[0] - i)
        fh.seek(hdr_off + i * per)
        buf = np.frombuffer(fh.read(n * per), dtype=np.float32).reshape(n, 450, 80)
        Wg[i:i + n] = torch.from_numpy(buf.copy()).to(dev, torch.float16)
        del buf
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
say(f"host RSS after load {_rss_gb():.2f} GB")
say(f"windows on GPU {tuple(Wg.shape)} {time.time() - t0:.0f}s")

# ---------------- teachers ----------------
IU = torch.triu_indices(450, 450, 1, device=dev)
ncol = IU.shape[1]
T_D0 = torch.zeros((nrec, 2 * ncol), dtype=torch.float16, device=dev)
T_C = torch.zeros((nrec, ncol), dtype=torch.float16, device=dev)
eye = torch.eye(450, device=dev, dtype=torch.float64)


def unit(v):
    v = v - v.mean(-1, keepdim=True)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


nwin = ends - starts
for k in sorted(set(nwin[usable].tolist())):
    rs_k = usable[nwin[usable] == k]
    for b in range(0, len(rs_k), 256):
        ids = rs_k[b:b + 256]
        X = torch.stack([Wg[starts[r]:ends[r]].permute(1, 0, 2).reshape(450, -1) for r in ids]).double()
        X = X - X.mean(-1, keepdim=True)
        X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        C = X @ X.transpose(1, 2)
        C.diagonal(dim1=1, dim2=2).fill_(1.0)
        w, V = torch.linalg.eigh(C)
        Cp = (V * (w.clamp_min(0) ** 0.35).unsqueeze(1)) @ V.transpose(1, 2)
        Cg = (1 - G_SHRINK) * C + G_SHRINK * eye
        P = torch.linalg.inv(Cg)
        d = P.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
        P = -P / (d.unsqueeze(2) * d.unsqueeze(1))
        P.diagonal(dim1=1, dim2=2).fill_(1.0)
        wa, Va = torch.linalg.eigh(P)
        Pa = (Va * (wa.clamp_min(0) ** ALPHA).unsqueeze(1)) @ Va.transpose(1, 2)
        fc_v = unit(C[:, IU[0], IU[1]]); d0_v = unit(Pa[:, IU[0], IU[1]]); c_v = unit(Cp[:, IU[0], IU[1]])
        idx = torch.as_tensor(ids, device=dev)
        T_D0[idx] = torch.cat([math.sqrt(1 - W_FULL) * d0_v, math.sqrt(W_FULL) * fc_v], 1).half()
        T_C[idx] = c_v.half()
say(f"teachers built {time.time() - t0:.0f}s")


def teacher_gram(ids, kind):
    if kind == "C":
        T = T_C[ids].float(); return T @ T.T
    if kind == "D0":
        T = T_D0[ids].float(); return T @ T.T
    a = T_D0[ids].float(); c = T_C[ids].float()
    return 0.5 * (a @ a.T) + 0.5 * (c @ c.T)


def cka(K, L):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return (Kc * Lc).sum() / (Kc.norm() * Lc.norm())


# ---------------- models ----------------
kinds = os.environ.get("KINDS", "D0,C,MIX").split(",")
models, opts = {}, {}
cfg = {"embed_dim": DIM, "layers": LAYERS, "heads": HEADS, "ff": FF, "corpus": "fbert-corpus minus AOMIC", "g": G_SHRINK, "alpha": ALPHA, "tag": TAG}
for kd in kinds:
    m = fm.FBERT(DIM, LAYERS, HEADS, ff=FF, norm_first=True, act="gelu", pe_layout="interleave", pe_summary=True).to(dev)
    torch.nn.init.normal_(m.summary_token, std=0.02)
    models[kd] = m
    opts[kd] = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    return LR * 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, STEPS - WARM)))


def sample(rec_pool, n, rng):
    r = rng.choice(rec_pool, n, replace=False)
    w = np.array([rng.randint(starts[i], ends[i]) for i in r])
    return torch.as_tensor(r, device=dev), torch.as_tensor(w, device=dev)


vr = np.random.RandomState(123)
val_r, val_w = sample(val_rec, min(2048, len(val_rec)), vr)
L_val = {kd: teacher_gram(val_r, kd) for kd in kinds}


@torch.no_grad()
def validate(m, kd):
    m.eval()
    E = torch.cat([m(Wg[val_w[i:i + 256]].float()) for i in range(0, len(val_w), 256)]).float()
    m.train()
    return float(cka(E @ E.T, L_val[kd]))


best = {kd: -1.0 for kd in kinds}
for s in range(STEPS + 1):
    r, w = sample(tr_rec, BATCH, rs)
    X = Wg[w].float()
    Ls = {"D0": teacher_gram(r, "D0"), "C": teacher_gram(r, "C")}
    Ls["MIX"] = 0.5 * Ls["D0"] + 0.5 * Ls["C"]
    for kd in kinds:
        for g in opts[kd].param_groups:
            g["lr"] = lr_at(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E = models[kd](X)
        E = E.float()
        loss = 1 - cka(E @ E.T, Ls[kd])
        opts[kd].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models[kd].parameters(), 1.0)
        opts[kd].step()
        if s % 200 == 0:
            say(f"step {s} {kd} loss {float(loss):.4f} lr {lr_at(s):.2e} t {time.time() - t0:.0f}s")
    if s % 2000 == 0 and s > 0:
        for kd in kinds:
            v = validate(models[kd], kd)
            ck = {"state_dict": models[kd].state_dict(), "config": dict(cfg, teacher=kd), "step": s,
                  "loss": float(loss), "val_cka": v}
            torch.save(ck, os.path.join(out, f"cbert_{TAG}_{kd}_step{s}.pth"))
            if v > best[kd]:
                best[kd] = v
                torch.save(ck, os.path.join(out, f"cbert_{TAG}_{kd}_best.pth"))
            say(f"VAL step {s} {kd} val_cka {v:.4f} best {best[kd]:.4f}")
say("FINISHED", json.dumps(best))
