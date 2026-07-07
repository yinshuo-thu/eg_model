"""Model definitions for the Engineering-Gates OOS package.

Copied (unmodified in spirit) from the original trainers so the submit package
is self-contained and does not import from the source trees:
  * MTMLP      -- multi-task DCN-style MLP           (ML_single/scripts/run_mlp.py)
  * EGTransformer -- v3-lineage daily temporal Transformer
                    (Transformer/v1/scripts/run_transformer.py)

Both carry a main (y_xs), sign and magnitude head; only the main head is used at
inference.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================== multi-task MLP (DCN)
class CrossNet(nn.Module):
    def __init__(self, d, n=2):
        super().__init__()
        self.w = nn.ModuleList([nn.Linear(d, 1, bias=True) for _ in range(n)])

    def forward(self, x0):
        x = x0
        for lin in self.w:
            x = x0 * lin(x) + x
        return x


class MTMLP(nn.Module):
    def __init__(self, d, h=384, p=0.15):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
        self.cross = CrossNet(d, 2)
        self.tower = nn.Sequential(
            nn.Linear(d * 2, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.SiLU(), nn.Dropout(p))
        self.main = nn.Linear(h // 2, 1)
        self.sign = nn.Linear(h // 2, 1)
        self.mag = nn.Linear(h // 2, 1)

    def forward(self, x):
        x = self.bn(x)
        z = torch.cat([self.cross(x), x], -1)
        z = self.tower(z)
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)


# =========================================================== temporal Transformer
class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h)
        self.o = nn.Linear(h, d)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.o(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, nh, p, K):
        super().__init__()
        self.K = K
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d))
        self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(1))          # time-decay bias scale
        self.register_buffer("dist", torch.arange(K).float())

    def forward(self, x):
        K = self.K
        h = self.n1(x)
        amask = -F.softplus(self.bias) * (self.dist[-1] - self.dist).abs().view(1, 1, K)
        amask = amask.expand(x.size(0) * self.attn.num_heads, K, K)
        a, _ = self.attn(h, h, h, attn_mask=amask, need_weights=False)
        x = x + self.ls1 * a
        x = x + self.ls2 * self.ff(self.n2(x))
        return x


class EGTransformer(nn.Module):
    def __init__(self, Fn, K=32, d=128, nl=3, nh=4, p=0.15):
        super().__init__()
        self.K = K
        self.inp = nn.Sequential(nn.Linear(Fn, d), nn.LayerNorm(d))
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, K, d))
        self.blocks = nn.ModuleList([Block(d, nh, p, K) for _ in range(nl)])
        self.attn_pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(p))
        self.main = nn.Linear(d, 1)
        self.sign = nn.Linear(d, 1)
        self.mag = nn.Linear(d, 1)

    def forward(self, x):                                  # x: [B, K, Fn]
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos
        for b in self.blocks:
            h = b(h)
        last = h[:, -1, :]
        w = torch.softmax(self.attn_pool(h).squeeze(-1), -1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        z = self.head(self.drop(torch.cat([last, pooled], -1)))
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)


# ================================= v2 temporal + cross-sectional Transformer (EGCS)
# v3-lineage architecture + scenario-specific optimizations (Transformer/v2):
#   scale-up (d 128->176, depth 3->4) + stochastic depth (Huang+2016) for IC/IR,
#   trained with R-Drop consistency (Liang+2021, in train_submit.py). MSCALE ruled
#   out -> plain depthwise conv stem. This is the shipped Transformer.
class TBlock(nn.Module):
    """Temporal block: ALiBi-style time-decay-biased self-attention over the K-day
    window + SwiGLU FFN, LayerScale residuals, optional stochastic depth."""
    def __init__(self, d, nh, p, K, sd=0.0):
        super().__init__()
        self.K = K
        self.sd = sd
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d))
        self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("dist", torch.arange(K).float())

    def _dp(self, x):                                     # stochastic depth (train only)
        if not self.training or self.sd <= 0:
            return x
        keep = 1 - self.sd
        return x * torch.empty(x.size(0), 1, 1, device=x.device).bernoulli_(keep) / keep

    def forward(self, x):
        K = self.K
        h = self.n1(x)
        amask = -F.softplus(self.bias) * (self.dist[-1] - self.dist).abs().view(1, 1, K)
        amask = amask.expand(x.size(0) * self.attn.num_heads, K, K)
        a, _ = self.attn(h, h, h, attn_mask=amask, need_weights=False)
        x = x + self._dp(self.ls1 * a)
        x = x + self._dp(self.ls2 * self.ff(self.n2(x)))
        return x


class CSBlock(nn.Module):
    """Cross-sectional block: attention ACROSS instruments within a day (masking
    absent instruments via key_padding_mask) + SwiGLU FFN, LayerScale residuals."""
    def __init__(self, d, nh, p):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d))
        self.ls2 = nn.Parameter(1e-3 * torch.ones(d))

    def forward(self, z, kpm):
        h = self.n1(z)
        a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        z = z + self.ls1 * a
        z = z + self.ls2 * self.ff(self.n2(z))
        return z


class EGCSTransformer(nn.Module):
    """Per-(instrument, day) temporal encoder (TBlocks over the K-day window)
    followed by cross-sectional CSBlocks attending across instruments in a day.

        forward(x, kpm):
          x   : (B_days, N_inst, K, Fn)
          kpm : (B_days, N_inst) bool — True where an instrument is ABSENT that day
        returns (main, sign, mag), each (B_days, N_inst).
    """
    def __init__(self, Fn, K=32, d=176, nl=4, ncs=2, nh=4, p=0.15, sd=0.15):
        super().__init__()
        self.K = K
        self.inp = nn.Sequential(nn.Linear(Fn, d), nn.LayerNorm(d))
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, K, d))
        self.tblocks = nn.ModuleList([TBlock(d, nh, p, K, sd) for _ in range(nl)])
        self.attn_pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(p))
        self.cs = nn.ModuleList([CSBlock(d, nh, p) for _ in range(ncs)])
        self.main = nn.Linear(d, 1)
        self.sign = nn.Linear(d, 1)
        self.mag = nn.Linear(d, 1)

    def encode(self, x):                                  # x: (B*N, K, Fn)
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos
        for b in self.tblocks:
            h = b(h)
        last = h[:, -1, :]
        w = torch.softmax(self.attn_pool(h).squeeze(-1), -1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        return self.head(self.drop(torch.cat([last, pooled], -1)))

    def forward(self, x, kpm):                            # x: (B, N, K, Fn)
        B, N = x.shape[0], x.shape[1]
        z = self.encode(x.reshape(B * N, self.K, x.shape[-1])).reshape(B, N, -1)
        for cb in self.cs:
            z = cb(z, kpm)
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)
