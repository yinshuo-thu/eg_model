"""EG temporal-panel Transformer v2 (IR-focused improvements over run_transformer.py).

Same v3-lineage architecture (input proj + causal conv stem, time-biased attention,
SwiGLU + LayerScale, last-token + attention dual readout, multitask head) but adds
levers aimed at raising daily-IC stability (IR) without hurting IC:

  * Weight EMA (Polyak averaging) of model params -> eval/predict with EMA weights.
  * Cosine LR schedule with linear warmup.
  * Optional cross-sectional IC loss term (day-batched).
  * More seeds (NSEED env).

Everything is configured by env vars so the same file can screen many ideas:
  NSEED      number of seeds                       (default 3 for screening)
  EPOCHS     max epochs                            (default 28)
  K          day lookback                          (default 32)
  EMA        EMA decay (0 disables)                (default 0.999)
  ICW        cross-sectional IC loss weight        (default 0.0)
  DAYBATCH   1 => batch by whole days (needed for ICW>0) else random pairs
  LR         peak lr                               (default 7e-4)
  WD         weight decay                          (default 1e-4)
  WARMUP     warmup fraction of total steps        (default 0.05)
  DROP       dropout p                             (default 0.15)
  SD         stochastic depth rate (0 disables)    (default 0.0)
  PATIENCE   early-stop patience                   (default 6)
  TAG        leaderboard model name                (default transformer_v2)
  OUTPRED    output parquet filename stem          (default transformer_v2)
  SAVESEEDS  1 => dump per-seed full pred matrix to npy for offline blending
"""
from __future__ import annotations
import sys, os, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path("/root/autodl-tmp/eg_model/ML_single/scripts")))
from common import load, feature_cols, evaluate, print_row

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/root/autodl-tmp/eg_model")
TRAIN_END, VALID_END = 760, 880

def env(k, d):
    return os.environ.get(k, d)

K        = int(env("K", "32"))
EPOCHS   = int(env("EPOCHS", "28"))
NSEED    = int(env("NSEED", "3"))
EMA      = float(env("EMA", "0.999"))
ICW      = float(env("ICW", "0.0"))
DAYBATCH = int(env("DAYBATCH", "0"))
LR       = float(env("LR", "7e-4"))
WD       = float(env("WD", "1e-4"))
WARMUP   = float(env("WARMUP", "0.05"))
DROP     = float(env("DROP", "0.15"))
SD       = float(env("SD", "0.0"))
PATIENCE = int(env("PATIENCE", "6"))
TAG      = env("TAG", "transformer_v2")
OUTPRED  = env("OUTPRED", "transformer_v2")
SAVESEEDS= int(env("SAVESEEDS", "0"))
GEMB     = int(env("GEMB", "0"))      # 1 => condition on current-day group g via embedding
NGROUP   = 76                          # g in {-1..74} -> shift +1 -> {0..75}
SEED0    = int(env("SEED0", "0"))      # seed offset (avoid duplicating prior runs)
REFIT    = int(env("REFIT", "0"))      # 1 => train on days<=880 (train+valid); 5% random pair holdout for early stop


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
    gpanel = np.zeros((NI, ND), dtype="int64")  # group id +1 (so -1 -> 0)
    ii = df["instrument_id"].map(iidx).to_numpy(); dd = df["day"].map(didx).to_numpy()
    panel[ii, dd] = df[fcols].to_numpy("float32")
    yxs[ii, dd] = df["y_xs"].to_numpy("float32")
    yraw[ii, dd] = df["y"].to_numpy("float32")
    gpanel[ii, dd] = (df["g"].to_numpy("int64") + 1)
    return panel, yxs, yraw, gpanel, insts, days, fcols


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.w12 = nn.Linear(d, 2 * h); self.o = nn.Linear(h, d)
    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1); return self.o(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, nh, p, sd=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nh, dropout=p, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ff = SwiGLU(d, 2 * d)
        self.ls1 = nn.Parameter(1e-3 * torch.ones(d)); self.ls2 = nn.Parameter(1e-3 * torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(1))
        self.sd = sd
        self.register_buffer("dist", torch.arange(K).float())
    def _drop_path(self, x):
        if not self.training or self.sd <= 0:
            return x
        keep = 1 - self.sd
        mask = torch.empty(x.size(0), 1, 1, device=x.device).bernoulli_(keep) / keep
        return x * mask
    def forward(self, x):
        h = self.n1(x)
        amask = -F.softplus(self.bias) * (self.dist[-1] - self.dist).abs().view(1, 1, K)
        amask = amask.expand(x.size(0) * self.attn.num_heads, K, K)
        a, _ = self.attn(h, h, h, attn_mask=amask, need_weights=False)
        x = x + self._drop_path(self.ls1 * a)
        x = x + self._drop_path(self.ls2 * self.ff(self.n2(x)))
        return x


class EGTransformer(nn.Module):
    def __init__(self, Fn, d=128, nl=3, nh=4, p=0.15, sd=0.0, gemb=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(Fn, d), nn.LayerNorm(d))
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, K, d))
        self.gemb = nn.Embedding(NGROUP, d) if gemb else None
        if self.gemb is not None:
            nn.init.normal_(self.gemb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, nh, p, sd) for _ in range(nl)])
        self.attn_pool = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(p))
        self.main = nn.Linear(d, 1); self.sign = nn.Linear(d, 1); self.mag = nn.Linear(d, 1)
    def forward(self, x, g=None):
        h = self.inp(x)
        if self.gemb is not None and g is not None:
            h = h + self.gemb(g).unsqueeze(1)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.pos
        for b in self.blocks:
            h = b(h)
        last = h[:, -1, :]
        w = torch.softmax(self.attn_pool(h).squeeze(-1), -1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        z = self.head(self.drop(torch.cat([last, pooled], -1)))
        return self.main(z).squeeze(-1), self.sign(z).squeeze(-1), self.mag(z).squeeze(-1)


class EMAHelper:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def daily_ic_np(pred, y, day):
    d = pd.DataFrame({"p": pred, "t": y, "day": day}).dropna()
    return float(d.groupby("day").apply(lambda s: s.p.corr(s.t)).mean())


def corr_loss(pred, target):
    # negative pearson correlation within the batch (or per-day group handled by caller)
    pred = pred.float(); target = target.float()
    p = pred - pred.mean(); t = target - target.mean()
    denom = (p.std() * t.std() + 1e-8)
    return -(p * t).mean() / denom


def main():
    t0 = time.time()
    print(f"[xfmr-v2] K={K} EPOCHS={EPOCHS} NSEED={NSEED} EMA={EMA} ICW={ICW} "
          f"DAYBATCH={DAYBATCH} LR={LR} WD={WD} WARMUP={WARMUP} DROP={DROP} SD={SD} TAG={TAG}", flush=True)
    panel, yxs, yraw, gpanel, insts, days, fcols = build_panel()
    NI, ND, Fn = panel.shape
    print(f"[xfmr-v2] panel {panel.shape} ({panel.nbytes/1e9:.2f} GB)", flush=True)
    P = torch.from_numpy(panel).to(DEV)
    Yx = torch.from_numpy(np.nan_to_num(yxs)).to(DEV)
    sign = torch.from_numpy((np.nan_to_num(yraw) > 0).astype("float32")).to(DEV)
    mag = torch.from_numpy(np.abs(np.nan_to_num(yxs))).to(DEV)
    Gp = torch.from_numpy(gpanel).to(DEV)

    di = np.arange(ND)
    day_val = days
    valid_day = di >= (K - 1)
    sp = np.where(day_val <= TRAIN_END, 0, np.where(day_val <= VALID_END, 1, 2))
    samp = np.array([(ii, dd) for dd in di[valid_day] for ii in range(NI)], dtype=np.int64)
    samp_sp = sp[samp[:, 1]]
    if REFIT:
        pool = np.where(samp_sp <= 1)[0]              # all days <= 880
        rng = np.random.default_rng(12345)
        hold = rng.choice(len(pool), size=int(0.05 * len(pool)), replace=False)
        va_idx = np.sort(pool[hold])                  # 5% OOS pairs for early-stop signal
        tr_idx = np.setdiff1d(pool, va_idx)
        print(f"[xfmr-v2] REFIT<=880: train {len(tr_idx):,} holdout {len(va_idx):,} all {len(samp):,}", flush=True)
    else:
        tr_idx = np.where(samp_sp == 0)[0]; va_idx = np.where(samp_sp == 1)[0]
        print(f"[xfmr-v2] samples: train {len(tr_idx):,} valid {len(va_idx):,} all {len(samp):,}", flush=True)
    samp_t = torch.from_numpy(samp).to(DEV)
    offs = torch.arange(-(K - 1), 1, device=DEV)

    def gather(idx_t):
        ii = samp_t[idx_t, 0]; dd = samp_t[idx_t, 1]
        win_d = dd.unsqueeze(1) + offs.unsqueeze(0)
        x = P[ii.unsqueeze(1), win_d]
        return x, Yx[ii, dd], sign[ii, dd], mag[ii, dd], Gp[ii, dd]

    samp_day = day_val[samp[:, 1]]
    samp_yraw = yraw[samp[:, 0], samp[:, 1]]

    # for day-batched training: group train sample indices by day_idx
    tr_days = np.unique(samp[tr_idx, 1])
    tr_by_day = {d: tr_idx[samp[tr_idx, 1] == d] for d in tr_days}

    def train_one(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        net = EGTransformer(Fn, p=DROP, sd=SD, gemb=bool(GEMB)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
        ema = EMAHelper(net, EMA) if EMA > 0 else None
        bs = 4096
        tr_t = torch.from_numpy(tr_idx).to(DEV)

        if DAYBATCH:
            days_per_batch = max(1, bs // NI + 1)  # ~3 days
            steps_per_epoch = math.ceil(len(tr_days) / days_per_batch)
        else:
            steps_per_epoch = math.ceil(len(tr_t) / bs)
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(1, int(WARMUP * total_steps))
        def lr_at(step):
            if step < warmup_steps:
                return LR * step / warmup_steps
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * LR * (1 + math.cos(math.pi * min(1.0, prog)))

        gstep = 0
        best, best_state, bad = -9, None, 0
        for ep in range(EPOCHS):
            net.train()
            if DAYBATCH:
                dperm = np.random.permutation(tr_days)
                batches = [np.concatenate([tr_by_day[d] for d in dperm[i:i + days_per_batch]])
                           for i in range(0, len(dperm), days_per_batch)]
            else:
                perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
                batches = [perm[i:i + bs] for i in range(0, len(perm), bs)]
            for b in batches:
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(gstep)
                if DAYBATCH:
                    b_days = samp[b, 1]
                    b = torch.from_numpy(b).to(DEV)
                x, ym, ys, yg, gid = gather(b)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=DEV == "cuda"):
                    m, s, mg = net(x, gid if GEMB else None)
                    loss = F.smooth_l1_loss(m, ym) + 0.3 * F.binary_cross_entropy_with_logits(s, ys) + 0.3 * F.mse_loss(mg, yg)
                    if ICW > 0:
                        if DAYBATCH:
                            # per-day cross-sectional corr loss
                            bd = torch.from_numpy(b_days).to(DEV)
                            cl = 0.0; uds = torch.unique(bd)
                            for ud in uds:
                                msk = bd == ud
                                cl = cl + corr_loss(m[msk], ym[msk])
                            cl = cl / len(uds)
                        else:
                            cl = corr_loss(m, ym)
                        loss = loss + ICW * cl
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update()
                if ema is not None:
                    ema.update(net)
                gstep += 1
            # valid IC using EMA weights if enabled
            eval_state = ema.state_dict() if ema is not None else net.state_dict()
            backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(eval_state)
            net.eval(); pv = np.empty(len(va_idx), "float32")
            with torch.no_grad():
                for i in range(0, len(va_idx), 16384):
                    bb = torch.from_numpy(va_idx[i:i + 16384]).to(DEV)
                    gx = gather(bb)
                    pv[i:i + 16384] = net(gx[0], gx[4] if GEMB else None)[0].float().cpu().numpy()
            ic = daily_ic_np(pv, samp_yraw[va_idx], samp_day[va_idx])
            net.load_state_dict(backup)
            print(f"   seed{seed} ep{ep+1} valid IC {ic:.5f} (lr {lr_at(gstep):.2e})", flush=True)
            if ic > best + 1e-5:
                best, best_state, bad = ic, {k: v.detach().clone() for k, v in eval_state.items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        net.load_state_dict(best_state); net.eval()
        pred = np.empty(len(samp), "float32")
        with torch.no_grad():
            for i in range(0, len(samp), 16384):
                bb = torch.arange(i, min(i + 16384, len(samp)), device=DEV)
                gx = gather(bb)
                pred[i:i + 16384] = net(gx[0], gx[4] if GEMB else None)[0].float().cpu().numpy()
        print(f"   seed{seed}: best valid IC {best:.5f}", flush=True)
        return pred, best

    all_preds = []
    for s in range(SEED0, SEED0 + NSEED):
        p, bv = train_one(s)
        all_preds.append(p)
    seed_mat = np.stack(all_preds, 0)
    if SAVESEEDS:
        np.save(ROOT / "artifacts" / "preds" / f"{OUTPRED}_seedmat.npy", seed_mat)
        np.save(ROOT / "artifacts" / "preds" / f"{OUTPRED}_meta.npy",
                np.stack([samp_day, insts[samp[:, 0]], samp_yraw, samp_sp], 1))
    preds = seed_mat.mean(0)

    out = pd.DataFrame({"day": samp_day, "instrument_id": insts[samp[:, 0]],
                        "y": samp_yraw, "pred": preds})
    out["split"] = np.where(out.day <= TRAIN_END, "train", np.where(out.day <= VALID_END, "valid", "test"))
    r = evaluate(out, TAG); print_row(r)
    out[out.split.isin(["valid", "test"])][["day", "instrument_id", "split", "y", "pred"]].to_parquet(
        ROOT / "artifacts" / "preds" / f"{OUTPRED}.parquet", index=False)
    lb = ROOT / "Transformer" / "v1" / "metrics" / "leaderboard_v2.csv"
    lb.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([r])
    if lb.exists():
        prev = pd.read_csv(lb)
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(lb, index=False)
    print(f"[xfmr-v2] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
