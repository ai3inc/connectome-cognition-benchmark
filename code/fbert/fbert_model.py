"""F-BERT encoder rebuilt from the released checkpoint (the authors' code repository
github.com/GioMarraffini/fc-power returned 404 on 2026-11-17, so `fcpower.encoder`
is not available).

What the checkpoint fixes: summary_token (1,1,D), token_proj Linear(450->D), an
nn.TransformerEncoder of `layers` standard nn.TransformerEncoderLayer blocks
(in_proj/out_proj, linear1 D->512, linear2 512->D, norm1, norm2), final_norm.

What it does not fix, and is therefore a switch here, chosen by reproducing the
checkpoint's own recorded training loss on corpus windows (validate_variants.py):
  norm_first   pre-LN or post-LN blocks
  act          relu or gelu in the feed-forward
  pe_layout    sinusoidal table interleaved (sin,cos,sin,...) or concatenated [sin|cos]
  pe_summary   positional code added to the summary token (position 0, timepoints
               at 1..T) or only to the timepoint tokens (0..T-1)
"""
import math
import torch
import torch.nn as nn


def sinusoid(T, D, layout, device=None):
    pos = torch.arange(T, dtype=torch.float32, device=device)[:, None]
    i = torch.arange(D // 2, dtype=torch.float32, device=device)[None, :]
    ang = pos / torch.pow(10000.0, 2 * i / D)
    if layout == "interleave":
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang)
    else:
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], 1)
    return pe


class FBERT(nn.Module):
    def __init__(self, D, layers, heads, ff=512, regions=450, norm_first=True,
                 act="gelu", pe_layout="interleave", pe_summary=False):
        super().__init__()
        self.D, self.pe_layout, self.pe_summary = D, pe_layout, pe_summary
        self.summary_token = nn.Parameter(torch.zeros(1, 1, D))
        self.token_proj = nn.Linear(regions, D)
        layer = nn.TransformerEncoderLayer(D, heads, ff, dropout=0.0, activation=act,
                                           batch_first=True, norm_first=norm_first)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(D)

    def forward(self, x):  # x: (B, regions, T), z-scored per region
        B, _, T = x.shape
        tok = self.token_proj(x.transpose(1, 2))  # (B,T,D)
        s = self.summary_token.expand(B, 1, self.D)
        if self.pe_summary:
            h = torch.cat([s, tok], 1) + sinusoid(T + 1, self.D, self.pe_layout, x.device)[None]
        else:
            h = torch.cat([s, tok + sinusoid(T, self.D, self.pe_layout, x.device)[None]], 1)
        h = self.blocks(h)
        return self.final_norm(h[:, 0])


def load(path, **variant):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = FBERT(c["embed_dim"], c["layers"], c["heads"], ff=c.get("ff", 512), **variant)
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=True), None
    m.eval()
    return m, ck
