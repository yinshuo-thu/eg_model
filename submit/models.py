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
