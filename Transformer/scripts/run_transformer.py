"""EG temporal-panel Transformer (v3-lineage, adapted to a daily cross-sectional panel).

The jump end2end-v3 model read per-symbol intraday sequences; here each instrument
forms a per-day sequence. For each (instrument, day t) we feed the last K days of the
213 leak-free features and predict y_xs at day t. Causal by construction (only days
<= t). v3 ideas kept: input projection + causal conv stem, multi-scale-free but
pre-LN time-biased Transformer blocks (SwiGLU + LayerScale), last-token + attention
dual readout, and a multitask head (main y_xs + sign + magnitude). Seed-ensembled.

Balanced panel (1333 instruments x 1259 days) lets us hold a dense
(n_inst, n_days, F) tensor on GPU and gather K-windows per batch.
"""
from __future__ import annotations
import sys, time, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path("/root/autodl-tmp/eg_model/ML_single/scripts")))
from common import load, feature_cols, evaluate, save_pred, print_row

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/root/autodl-tmp/eg_model")
K = 32                      # day lookback
TRAIN_END, VALID_END = 760, 880


def build_panel():
    fcols = feature_cols()
    df = load(fcols)
    df = df.sort_values(["instrument_id", "day"]).reset_index(drop=True)
    insts = np.sort(df["instrument_id"].unique()); days = np.sort(df["day"].unique())
    iidx = {v: k for k, v in enumerate(insts)}; didx = {v: k for k, v in enumerate(days)}
    NI, ND, Fn = len(insts), len(days), len(fcols)
    panel = np.zeros((NI, ND, Fn), dtype="float32")
    yxs = np.full((NI, ND), np.nan, dtype="float32")
    yraw = np.full((NI, ND), np.nan, dtype="float32")
    ii = df["instrument_id"].map(iidx).to_numpy(); dd = df["day"].map(didx).to_numpy()
    panel[ii, dd] = df[fcols].to_numpy("float32")
    yxs[ii, dd] = df["y_xs"].to_numpy("float32")
    yraw[ii, dd] = df["y"].to_numpy("float32")
    return panel, yxs, yraw, insts, days, fcols


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.w12 = nn.Linear(d, 2 * h); self.o = nn.Linear(h, d)
    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1); return self.o(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, nh, p):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d)); self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(1))  # time-decay bias scale
        self.register_buffer("dist", torch.arange(K).float())
    def forward(self, x):
        h = self.n1(x)
        # ALiBi-style decay: closer days weigh more (causal, regular spacing)
        amask = -F.softplus(self.bias) * (self.dist[-1] - self.dist).abs().view(1, 1, K)
        amask = amask.expand(x.size(0) * self.attn.num_heads, K, K)
        a, _ = self.attn(h, h, h, attn_mask=amask, need_weights=False)
        x = x + self.ls1 * a
        x = x + self.ls2 * self.ff(self.n2(x))
        return x


class EGTransformer(nn.Module):
    def __init__(self, Fn, d=128, nl=3, nh=4, p=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(Fn, d), nn.LayerNorm(d))
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)  # causal-ish local stem
        self.pos = nn.Parameter(0.02 * torch.randn(1, K, d))
        self.blocks = nn.ModuleList([Block(d, nh, p) for _ in range(nl)])
        self.attn_pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(p))
        self.main = nn.Linear(d, 1); self.sign = nn.Linear(d, 1); self.mag = nn.Linear(d, 1)
    def forward(self, x):                       # x: [B,K,Fn]
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


def daily_ic_np(pred, y, day):
    d = pd.DataFrame({"p": pred, "t": y, "day": day}).dropna()
    return float(d.groupby("day").apply(lambda s: s.p.corr(s.t)).mean())


def main():
    t0 = time.time()
    print(f"[xfmr] building dense panel, dev={DEV}", flush=True)
    panel, yxs, yraw, insts, days, fcols = build_panel()
    NI, ND, Fn = panel.shape
    print(f"[xfmr] panel {panel.shape} ({panel.nbytes/1e9:.2f} GB)", flush=True)
    P = torch.from_numpy(panel).to(DEV)                  # [NI, ND, Fn]
    Yx = torch.from_numpy(np.nan_to_num(yxs)).to(DEV)
    sign = torch.from_numpy((np.nan_to_num(yraw) > 0).astype("float32")).to(DEV)
    mag = torch.from_numpy(np.abs(np.nan_to_num(yxs))).to(DEV)

    # sample index = (inst, day_idx) with day_idx>=K-1; split by actual day value
    di = np.arange(ND)
    day_val = days  # day value for each day_idx
    valid_day = di >= (K - 1)
    sp = np.where(day_val <= TRAIN_END, 0, np.where(day_val <= VALID_END, 1, 2))
    samp = [(ii, dd) for dd in di[valid_day] for ii in range(NI)]
    samp = np.array(samp, dtype=np.int64)
    samp_sp = sp[samp[:, 1]]
    tr_idx = np.where(samp_sp == 0)[0]; va_idx = np.where(samp_sp == 1)[0]
    print(f"[xfmr] samples: train {len(tr_idx):,} valid {len(va_idx):,} all {len(samp):,}", flush=True)
    samp_t = torch.from_numpy(samp).to(DEV)
    offs = torch.arange(-(K - 1), 1, device=DEV)  # window offsets

    def gather(idx_t):
        ii = samp_t[idx_t, 0]; dd = samp_t[idx_t, 1]
        win_d = dd.unsqueeze(1) + offs.unsqueeze(0)          # [b,K]
        x = P[ii.unsqueeze(1), win_d]                        # [b,K,Fn]
        return x, Yx[ii, dd], sign[ii, dd], mag[ii, dd]

    # day/y arrays aligned to sample order (for IC eval)
    samp_day = day_val[samp[:, 1]]
    samp_yraw = yraw[samp[:, 0], samp[:, 1]]

    def train_one(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        net = EGTransformer(Fn).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=1e-4, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
        bs = 4096; best, best_state, bad = -9, None, 0
        tr_t = torch.from_numpy(tr_idx).to(DEV)
        for ep in range(28):
            net.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                x, ym, ys, yg = gather(b)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=DEV == "cuda"):
                    m, s, g = net(x)
                    loss = F.smooth_l1_loss(m, ym) + 0.3 * F.binary_cross_entropy_with_logits(s, ys) + 0.3 * F.mse_loss(g, yg)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update()
            # valid IC
            net.eval(); pv = np.empty(len(va_idx), "float32")
            with torch.no_grad():
                for i in range(0, len(va_idx), 16384):
                    b = torch.from_numpy(va_idx[i:i + 16384]).to(DEV)
                    pv[i:i + 16384] = net(gather(b)[0])[0].float().cpu().numpy()
            ic = daily_ic_np(pv, samp_yraw[va_idx], samp_day[va_idx])
            print(f"   seed{seed} ep{ep+1} valid IC {ic:.5f}", flush=True)
            if ic > best + 1e-5:
                best, best_state, bad = ic, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 5: break
        net.load_state_dict(best_state); net.eval()
        pred = np.empty(len(samp), "float32")
        with torch.no_grad():
            for i in range(0, len(samp), 16384):
                b = torch.arange(i, min(i + 16384, len(samp)), device=DEV)
                pred[i:i + 16384] = net(gather(b)[0])[0].float().cpu().numpy()
        print(f"   seed{seed}: best valid IC {best:.5f}", flush=True)
        return pred

    seeds = tuple(range(int(__import__("os").environ.get("NSEED", "8"))))
    preds = np.mean([train_one(s) for s in seeds], axis=0)
    out = pd.DataFrame({"day": samp_day, "instrument_id": insts[samp[:, 0]],
                        "y": samp_yraw, "pred": preds})
    out["split"] = np.where(out.day <= TRAIN_END, "train", np.where(out.day <= VALID_END, "valid", "test"))
    r = evaluate(out, "transformer"); print_row(r)
    out[out.split.isin(["valid", "test"])][["day", "instrument_id", "split", "y", "pred"]].to_parquet(
        ROOT / "artifacts" / "preds" / "transformer.parquet", index=False)
    (ROOT / "Transformer" / "metrics" / "leaderboard.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r]).to_csv(ROOT / "Transformer" / "metrics" / "leaderboard.csv", index=False)
    print(f"[xfmr] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
