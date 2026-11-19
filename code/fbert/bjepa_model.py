"""Brain-JEPA frozen encoder (Dong et al., NeurIPS 2024; github.com/Eric-LRL/Brain-JEPA,
pretrained checkpoint jepa-ep300.pth.tar from the authors' Google Drive), built with
the authors' own src/helper.init_model, normal attention on CPU.

Input (B, 1, 450, 160): Tian-S3 subcortex (50) then Schaefer-400 cortex — the atlas
file order we already store. Output: mean over the encoder's 4,500 tokens (768-d).
"""
import os, sys, types
import numpy as np
import pandas as pd
import torch

SRC = os.environ.get("BJEPA_SRC", "/workspace/bjepa_src/Brain-JEPA-HEAD")
CKPT = os.environ.get("BJEPA_CKPT", "/workspace/logs/BrainJEPA-Checkpoints/Pretraining/jepa-ep300.pth.tar")
# the authors' module imports flash_attn at top level (GPU only); stub it
if "flash_attn" not in sys.modules:
    fa = types.ModuleType("flash_attn")
    def _no(*a, **k):
        raise RuntimeError("flash_attn not available; use attn_mode='normal'")
    fa.flash_attn_qkvpacked_func = _no
    sys.modules["flash_attn"] = fa
sys.path.insert(0, SRC)
from src.helper import init_model  # noqa: E402


def load():
    grad = torch.tensor(pd.read_csv(os.path.join(SRC, "data/gradient_mapping_450.csv"), header=None).values,
                        dtype=torch.float32).unsqueeze(0)
    enc, _ = init_model(device=torch.device("cpu"), patch_size=16, crop_size=(450, 160), pred_depth=6,
                        pred_emb_dim=384, model_name="vit_base", gradient_pos_embed=grad,
                        attn_mode="normal", add_w="mapping", gradient_checkpointing=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["target_encoder"].items()}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    enc.eval()
    return enc, missing, unexpected


def prep_window(x):
    """x: (450, 160) raw parcel means -> linear detrend and z per ROI, then the
    authors' per-sample global z (their `use_normalization`)."""
    from scipy.signal import detrend
    x = detrend(x.astype(np.float64), axis=1, type="linear")
    sd = x.std(1, keepdims=True)
    x = (x - x.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x.astype(np.float32)


@torch.no_grad()
def embed(enc, windows):
    """windows: (W, 450, 160) prepared -> (768,) mean over tokens, then over windows"""
    X = torch.from_numpy(np.stack(windows))[:, None]
    out = enc(X)
    return out.mean(1).mean(0).numpy()
