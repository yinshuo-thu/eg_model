"""EG temporal+cross-sectional Transformer (v3-lineage architecture, unchanged)
on whatever feature set common_generic points at. Identical architecture/loop
to Transformer/v1/scripts/run_transformer_v3factors.py -- only made generic
over the RETRAIN_* env vars instead of a hand-copied common_vN import.

Env: NSEED, EPOCHS, K, NCS, DMODEL, NL, NH, DROP, SD, LR, WD, WARMUP, PATIENCE,
EMA, SAVESEEDS, SEED0, DPS, RETRAIN_TAG, RETRAIN_FEATURES, RETRAIN_FEATURE_LIST
"""
from __future__ import annotations
import sys, os, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_generic import load, feature_cols, evaluate, print_row, TAG, ROOT, PREDS, METR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_END, VALID_END = 760, 880
def env(k, d): return os.environ.get(k, d)
K        = int(env("K", "32"))
EPOCHS   = int(env("EPOCHS", "28"))
NSEED    = int(env("NSEED", "3"))
NCS      = int(env("NCS", "2"))
DMODEL   = int(env("DMODEL", "128"))
NL       = int(env("NL", "3"))
NH       = int(env("NH", "4"))
DROP     = float(env("DROP", "0.15"))
SD       = float(env("SD", "0.0"))
LR       = float(env("LR", "7e-4"))
WD       = float(env("WD", "1e-4"))
WARMUP   = float(env("WARMUP", "0.05"))
PATIENCE = int(env("PATIENCE", "6"))
EMA      = float(env("EMA", "0.0"))
OUTPRED  = env("OUTPRED", f"transformer_{TAG}")
SAVESEEDS= int(env("SAVESEEDS", "0"))
SEED0    = int(env("SEED0", "0"))
DPS      = int(env("DPS", "4"))
# --- optimization toggles (cumulative) ---
MSCALE   = int(env("MSCALE", "0"))          # opt1: multi-scale dilated temporal stem (TCN/WaveNet)
RDROP    = float(env("RDROP", "0.0"))       # opt2: R-Drop twin-dropout consistency weight (Liang+2021)


def build_panel():
    fcols = feature_cols()
    df = load(fcols).sort_values(["instrument_id", "day"]).reset_index(drop=True)
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


class MultiScaleConv(nn.Module):
    """Opt-1: parallel causal-padded depthwise convs at dilations {1,2,4} fused by
    a gated projection — a TCN/WaveNet-style multi-resolution temporal stem that
    widens the receptive field (1→9 days) without the signal dilution a single
    long window (K=48) caused. Refs: Bai/Kolter/Koltun 2018 (TCN); van den Oord+ 2016 (WaveNet)."""
    def __init__(self, d, dilations=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(d, d, 3, padding=dl, dilation=dl, groups=d) for dl in dilations])
        self.proj = nn.Linear(d * len(dilations), d)
        self.gate = nn.Linear(d * len(dilations), d)
    def forward(self, h):  # h: (B, K, d)
        x = h.transpose(1, 2)  # (B, d, K)
        outs = torch.cat([c(x).transpose(1, 2) for c in self.convs], -1)  # (B, K, d*n)
        return self.proj(outs) * torch.sigmoid(self.gate(outs))  # gated fusion


class TBlock(nn.Module):
    def __init__(self, d, nh, p, sd=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d)); self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(1)); self.sd = sd
        self.register_buffer("dist", torch.arange(K).float())
    def _dp(self, x):
        if not self.training or self.sd <= 0: return x
        keep = 1 - self.sd
        return x * torch.empty(x.size(0), 1, 1, device=x.device).bernoulli_(keep) / keep
    def forward(self, x):
        h = self.n1(x)
        amask = (-F.softplus(self.bias) * (self.dist[-1] - self.dist).abs().view(1, 1, K)).expand(
            x.size(0) * self.attn.num_heads, K, K)
        a, _ = self.attn(h, h, h, attn_mask=amask, need_weights=False)
        x = x + self._dp(self.ls1 * a)
        x = x + self._dp(self.ls2 * self.ff(self.n2(x)))
        return x


class CSBlock(nn.Module):
    def __init__(self, d, nh, p):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d)); self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
    def forward(self, z, kpm):
        h = self.n1(z)
        a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        z = z + self.ls1 * a
        z = z + self.ls2 * self.ff(self.n2(z))
        return z


class EGCSTransformer(nn.Module):
    def __init__(self, Fn, d=128, nl=3, ncs=2, nh=4, p=0.15, sd=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(Fn, d), nn.LayerNorm(d))
        self.conv = MultiScaleConv(d) if MSCALE else nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, K, d))
        self.tblocks = nn.ModuleList([TBlock(d, nh, p, sd) for _ in range(nl)])
        self.attn_pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(p))
        self.cs = nn.ModuleList([CSBlock(d, nh, p) for _ in range(ncs)])
        self.main = nn.Linear(d, 1); self.sign = nn.Linear(d, 1); self.mag = nn.Linear(d, 1)
    def encode(self, x):
        h = self.inp(x)
        h = h + (self.conv(h) if MSCALE else self.conv(h.transpose(1, 2)).transpose(1, 2))
        h = h + self.pos
        for b in self.tblocks:
            h = b(h)
        last = h[:, -1, :]
        w = torch.softmax(self.attn_pool(h).squeeze(-1), -1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        return self.head(self.drop(torch.cat([last, pooled], -1)))
    def forward(self, x, kpm):
        B, N = x.shape[0], x.shape[1]
        z = self.encode(x.reshape(B * N, K, x.shape[-1])).reshape(B, N, -1)
        for cb in self.cs:
            z = cb(z, kpm)
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)


class EMAHelper:
    def __init__(self, model, decay):
        self.decay = decay; self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point: s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else: s.copy_(v)
    def state_dict(self): return {k: v.clone() for k, v in self.shadow.items()}


def daily_ic_days(pred_by_day, yraw_by_day):
    ics = []
    for p, y in zip(pred_by_day, yraw_by_day):
        m = ~np.isnan(y)
        if m.sum() < 5: continue
        pc, yc = p[m], y[m]
        if pc.std() < 1e-9 or yc.std() < 1e-9: continue
        ics.append(np.corrcoef(pc, yc)[0, 1])
    return float(np.mean(ics)) if ics else float("nan")


def main():
    t0 = time.time()
    print(f"[{TAG}] K={K} EPOCHS={EPOCHS} NSEED={NSEED} NCS={NCS} D={DMODEL} NL={NL} NH={NH} "
          f"DROP={DROP} SD={SD} LR={LR} EMA={EMA} MSCALE={MSCALE} RDROP={RDROP} OUT={OUTPRED}", flush=True)
    panel, yxs, yraw, insts, days, fcols = build_panel()
    NI, ND, Fn = panel.shape
    print(f"[{TAG}] panel {panel.shape} ({panel.nbytes/1e9:.2f} GB)", flush=True)
    P = torch.from_numpy(panel).to(DEV)
    Yx = torch.from_numpy(np.nan_to_num(yxs)).to(DEV)
    sign = torch.from_numpy((np.nan_to_num(yraw) > 0).astype("float32")).to(DEV)
    mag = torch.from_numpy(np.abs(np.nan_to_num(yxs))).to(DEV)
    present = torch.from_numpy((~np.isnan(yxs)).astype(bool)).to(DEV)

    day_idx = np.arange(K - 1, ND)
    sp = np.where(days[day_idx] <= TRAIN_END, 0, np.where(days[day_idx] <= VALID_END, 1, 2))
    tr_days = day_idx[sp == 0]; va_days = day_idx[sp == 1]
    print(f"[{TAG}] days: train {len(tr_days)} valid {len(va_days)} total {len(day_idx)}", flush=True)
    offs = torch.arange(-(K - 1), 1, device=DEV)

    def day_batch(dds):
        win = (dds.unsqueeze(1) + offs.unsqueeze(0)).reshape(-1)
        x = P[:, win, :].reshape(NI, len(dds), K, Fn).permute(1, 0, 2, 3)
        kpm = (~present[:, dds]).transpose(0, 1)
        return x, kpm

    def predict_days(net, dlist):
        net.eval(); preds = [None] * len(dlist)
        with torch.no_grad():
            for i in range(0, len(dlist), DPS):
                dds = torch.as_tensor(dlist[i:i + DPS], device=DEV)
                x, kpm = day_batch(dds)
                m = net(x, kpm)[0].float().cpu().numpy()
                for j in range(len(dds)):
                    preds[i + j] = m[j]
        return preds

    def train_one(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        net = EGCSTransformer(Fn, d=DMODEL, nl=NL, ncs=NCS, nh=NH, p=DROP, sd=SD).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
        ema = EMAHelper(net, EMA) if EMA > 0 else None
        steps_per_ep = math.ceil(len(tr_days) / DPS)
        total_steps = steps_per_ep * EPOCHS; warm = max(1, int(WARMUP * total_steps))
        def lr_at(s):
            if s < warm: return LR * s / warm
            prog = (s - warm) / max(1, total_steps - warm)
            return 0.5 * LR * (1 + math.cos(math.pi * min(1.0, prog)))
        va_yraw = [yraw[:, dd] for dd in va_days]
        gstep = 0; best, best_state, bad = -9, None, 0
        for ep in range(EPOCHS):
            net.train(); order = np.random.permutation(tr_days)
            for i in range(0, len(order), DPS):
                for pg in opt.param_groups: pg["lr"] = lr_at(gstep)
                dds = torch.as_tensor(order[i:i + DPS], device=DEV)
                x, kpm = day_batch(dds)
                pmask = present[:, dds].transpose(0, 1)
                ym = Yx[:, dds].transpose(0, 1); ys = sign[:, dds].transpose(0, 1); yg = mag[:, dds].transpose(0, 1)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                    def _sup(mm, ss, gg):
                        return (F.smooth_l1_loss(mm[pmask], ym[pmask])
                                + 0.3 * F.binary_cross_entropy_with_logits(ss[pmask], ys[pmask])
                                + 0.3 * F.mse_loss(gg[pmask], yg[pmask]))
                    m, s, mgv = net(x, kpm)
                    loss = _sup(m, s, mgv)
                    if RDROP > 0:  # opt2: R-Drop — second dropout pass + consistency penalty
                        m2, s2, mg2 = net(x, kpm)
                        loss = 0.5 * (loss + _sup(m2, s2, mg2)) + RDROP * F.mse_loss(m[pmask], m2[pmask])
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update()
                if ema is not None: ema.update(net)
                gstep += 1
            eval_state = ema.state_dict() if ema is not None else net.state_dict()
            backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(eval_state)
            pv = predict_days(net, va_days); ic = daily_ic_days(pv, va_yraw)
            net.load_state_dict(backup)
            print(f"   seed{seed} ep{ep+1} valid IC {ic:.5f} (lr {lr_at(gstep):.2e})", flush=True)
            if ic > best + 1e-5:
                best, best_state, bad = ic, {k: v.detach().clone() for k, v in eval_state.items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE: break
        net.load_state_dict(best_state)
        allp = predict_days(net, day_idx)
        print(f"   seed{seed}: best valid IC {best:.5f}", flush=True)
        return allp, best

    acc = None
    for s in range(SEED0, SEED0 + NSEED):
        allp, bv = train_one(s)
        arr = np.stack(allp, 1)
        acc = arr if acc is None else acc + arr
    pred_mat = acc / NSEED

    rows_day = np.repeat(days[day_idx], NI)
    rows_inst = np.tile(insts, len(day_idx))
    rows_y = yraw[:, day_idx].reshape(-1, order="F")
    rows_pred = pred_mat.reshape(-1, order="F")
    out = pd.DataFrame({"day": rows_day, "instrument_id": rows_inst, "y": rows_y, "pred": rows_pred})
    out = out.dropna(subset=["y"])
    out["split"] = np.where(out.day <= TRAIN_END, "train", np.where(out.day <= VALID_END, "valid", "test"))
    r = evaluate(out, TAG if OUTPRED == f"transformer_{TAG}" else OUTPRED); print_row(r)
    out[out.split.isin(["valid", "test"])][["day", "instrument_id", "split", "y", "pred"]].to_parquet(
        PREDS / f"{OUTPRED}.parquet", index=False)
    lb = METR / f"leaderboard_{TAG}.csv"
    row = pd.DataFrame([r])
    if lb.exists(): row = pd.concat([pd.read_csv(lb), row], ignore_index=True)
    row.to_csv(lb, index=False)
    if SAVESEEDS:
        np.save(PREDS / f"{OUTPRED}_predmat.npy", pred_mat)
    print(f"[{TAG}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
