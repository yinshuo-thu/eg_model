"""Retrain the Engineering-Gates OOS ensemble on ALL labelled days (1..1259).

Unlike the research trainers (which held out valid 761-880 / test 881-1259 for
*evaluation*), this script trains on every labelled row for the real
out-of-sample deployment.  A tiny random slice of days is held out ONLY as an
early-stopping monitor for the neural nets; LightGBM uses a fixed round budget.

Produces, in submit/weights/ :
  feature_artifacts.json  -- top_x list + PCA loadings + 213-feature order
  lgb_dart_seed{0..4}.txt -- tuned DART boosters (seed bag)
  mlp_seed{0..5}.pt       -- multi-task DCN-MLP state_dicts (seed bag)
  xfmr_seed{0..7}.pt      -- v3-lineage temporal Transformer state_dicts (seed bag)
  ensemble_config.json    -- diversity weights + group-neutralisation alpha,
                             fitted on the in-sample base predictions.

Run:  python submit/train_submit.py
"""
from __future__ import annotations
import json, time, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from features_core import compute_features
from models import MTMLP, EGTransformer

ROOT = Path("/root/autodl-tmp/eg_model")
RAW = ROOT / "artifacts" / "panel_raw.parquet"
WDIR = HERE / "weights"
WDIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _seeds(env, default):
    v = os.environ.get(env)
    return tuple(int(s) for s in v.split(",")) if v else default


K = 32                                    # transformer day-lookback
LGB_SEEDS = _seeds("EG_LGB_SEEDS", (0, 1, 2, 3, 4))
MLP_SEEDS = _seeds("EG_MLP_SEEDS", (0, 1, 2, 3, 4, 5))
XFMR_SEEDS = tuple(range(int(os.environ.get("NSEED", "8"))))
MLP_EPOCHS = int(os.environ.get("EG_MLP_EPOCHS", "30"))
XFMR_EPOCHS = int(os.environ.get("EG_XFMR_EPOCHS", "14"))
MIN_DAY = int(os.environ.get("EG_MIN_DAY", "0"))   # >0 -> smoke-test on a day slice
MON_FRAC = 0.035                          # random days held out only for early-stop
MON_SEED = 12345
FAMILIES = ["lightgbm", "mlp", "transformer"]
ALPHA_GRID = np.round(np.arange(0.0, 0.86, 0.05), 2)
PLATEAU = 0.997
RESEARCH_ALPHA = 0.35                      # OOS-validated group-neutralisation strength

# tuned, seed-bagged DART robust config (== lgb_dart_tuned_v2, the strongest LGB)
LGB_BEST = dict(objective="regression_l1", num_leaves=63, learning_rate=0.02,
                feature_fraction=0.4, bagging_fraction=0.7, bagging_freq=1,
                min_data_in_leaf=600, lambda_l1=2.0, lambda_l2=5.0, max_depth=8,
                num_threads=8, verbosity=-1)
LGB_DART = dict(LGB_BEST, boosting="dart", drop_rate=0.1, skip_drop=0.5)
LGB_ROUNDS = int(os.environ.get("EG_LGB_ROUNDS", "500"))


def daily_ic_np(pred, y, day) -> float:
    d = pd.DataFrame({"p": np.asarray(pred, "float64"),
                      "t": np.asarray(y, "float64"),
                      "day": np.asarray(day)}).dropna()
    if d.empty:
        return 0.0
    def _c(g):
        if g["p"].std() > 0 and g["t"].std() > 0:
            return g["p"].corr(g["t"])
        return np.nan
    ic = d.groupby("day").apply(_c, include_groups=False)
    return float(np.nanmean(ic.to_numpy()))


def per_day_z_np(vals, day):
    s = pd.DataFrame({"v": vals, "day": day})
    g = s.groupby("day")["v"]
    return ((s["v"] - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()


# ============================================================ LightGBM (DART bag)
def train_lightgbm(X, yxs, labelled):
    print(f"[lgb] training tuned DART bag ({len(LGB_SEEDS)} seeds x {LGB_ROUNDS} rounds) "
          f"on {int(labelled.sum()):,} rows", flush=True)
    dtr = lgb.Dataset(X[labelled], label=yxs[labelled])
    preds = np.zeros(len(X), dtype="float64")
    for sd in LGB_SEEDS:
        t0 = time.time()
        # NB: data_random_seed is a Dataset-construction param and cannot vary
        # across train() calls that share one Dataset; seed diversity comes from
        # seed / bagging_seed / feature_fraction_seed (as in run_classic.py).
        p = dict(LGB_DART, seed=sd, bagging_seed=sd + 7, feature_fraction_seed=sd + 17)
        m = lgb.train(p, dtr, num_boost_round=LGB_ROUNDS)
        m.save_model(str(WDIR / f"lgb_dart_seed{sd}.txt"))
        preds += m.predict(X)
        print(f"   seed {sd} done in {time.time()-t0:.0f}s", flush=True)
    return preds / len(LGB_SEEDS)


# ============================================================ multi-task MLP (bag)
def _gather_pred(net, Xg, idx, chunk=200000):
    out = np.empty(len(idx), "float32")
    for j in range(0, len(idx), chunk):
        b = torch.from_numpy(idx[j:j + chunk]).to(DEV)
        out[j:j + chunk] = net(Xg[b])[0].cpu().numpy()
    return out


# ============================================================ Transformer (bag)
def build_panel(feat, fcols):
    df = feat.sort_values(["instrument_id", "day"]).reset_index(drop=True)
    insts = np.sort(df["instrument_id"].unique())
    days = np.sort(df["day"].unique())
    iidx = {v: k for k, v in enumerate(insts)}
    didx = {v: k for k, v in enumerate(days)}
    NI, ND, Fn = len(insts), len(days), len(fcols)
    panel = np.zeros((NI, ND, Fn), dtype="float32")
    yxs = np.full((NI, ND), np.nan, dtype="float32")
    yraw = np.full((NI, ND), np.nan, dtype="float32")
    ii = df["instrument_id"].map(iidx).to_numpy()
    dd = df["day"].map(didx).to_numpy()
    panel[ii, dd] = df[fcols].to_numpy("float32")
    yxs[ii, dd] = df["y_xs"].to_numpy("float32")
    yraw[ii, dd] = df["y"].to_numpy("float32")
    return panel, yxs, yraw, insts, days


def train_transformer(feat, fcols, mon_days):
    print(f"[xfmr] training temporal Transformer ({len(XFMR_SEEDS)} seeds, K={K})", flush=True)
    panel, yxs, yraw, insts, days = build_panel(feat, fcols)
    NI, ND, Fn = panel.shape
    print(f"[xfmr] panel {panel.shape} ({panel.nbytes/1e9:.2f} GB)", flush=True)
    P = torch.from_numpy(panel).to(DEV)
    Yx = torch.from_numpy(np.nan_to_num(yxs)).to(DEV)
    sign = torch.from_numpy((np.nan_to_num(yraw) > 0).astype("float32")).to(DEV)
    mag = torch.from_numpy(np.abs(np.nan_to_num(yxs))).to(DEV)

    di = np.arange(ND)
    valid_day = di >= (K - 1)
    is_mon = np.isin(days, list(mon_days))
    samp = np.array([(ii, dd) for dd in di[valid_day] for ii in range(NI)], dtype=np.int64)
    samp_t = torch.from_numpy(samp).to(DEV)
    offs = torch.arange(-(K - 1), 1, device=DEV)
    samp_day = days[samp[:, 1]]
    samp_yraw = yraw[samp[:, 0], samp[:, 1]]
    samp_yxs = yxs[samp[:, 0], samp[:, 1]]
    samp_ismon = is_mon[samp[:, 1]]

    labelled = ~np.isnan(samp_yxs)
    tr_idx = np.where(labelled & ~samp_ismon)[0]
    mo_idx = np.where(labelled & samp_ismon)[0]
    print(f"[xfmr] samples: train {len(tr_idx):,} monitor {len(mo_idx):,} total {len(samp):,}", flush=True)

    def gather(idx_t):
        ii = samp_t[idx_t, 0]; dd = samp_t[idx_t, 1]
        win_d = dd.unsqueeze(1) + offs.unsqueeze(0)
        x = P[ii.unsqueeze(1), win_d]
        return x, Yx[ii, dd], sign[ii, dd], mag[ii, dd]

    # accumulate per-seed predictions over ALL samples, average at the end
    pred_sum = np.zeros(len(samp), dtype="float64")
    tr_t = torch.from_numpy(tr_idx).to(DEV)
    for sd in XFMR_SEEDS:
        t0 = time.time()
        torch.manual_seed(sd); np.random.seed(sd)
        net = EGTransformer(Fn, K=K).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=1e-4, betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
        bs = 4096; best, best_state, bad = -9.0, None, 0
        for ep in range(XFMR_EPOCHS):
            net.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                x, ym, ys, yg = gather(b)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                    m, s, g = net(x)
                    loss = (F.smooth_l1_loss(m, ym) + 0.3 * F.binary_cross_entropy_with_logits(s, ys)
                            + 0.3 * F.mse_loss(g, yg))
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update()
            net.eval(); pv = np.empty(len(mo_idx), "float32")
            with torch.no_grad():
                for i in range(0, len(mo_idx), 16384):
                    b = torch.from_numpy(mo_idx[i:i + 16384]).to(DEV)
                    pv[i:i + 16384] = net(gather(b)[0])[0].float().cpu().numpy()
            ic = daily_ic_np(pv, samp_yraw[mo_idx], samp_day[mo_idx])
            if ic > best + 1e-5:
                best, bad = ic, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 4:
                    break
        net.load_state_dict(best_state); net.eval()
        torch.save({k: v.cpu() for k, v in best_state.items()}, WDIR / f"xfmr_seed{sd}.pt")
        pred = np.empty(len(samp), "float32")
        with torch.no_grad():
            for i in range(0, len(samp), 16384):
                b = torch.arange(i, min(i + 16384, len(samp)), device=DEV)
                pred[i:i + 16384] = net(gather(b)[0])[0].float().cpu().numpy()
        pred_sum += pred
        print(f"   seed {sd}: monitor IC {best:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    pred_avg = pred_sum / len(XFMR_SEEDS)
    # scatter back to (instrument_id, day) so it aligns with feat rows
    out = pd.DataFrame({"instrument_id": insts[samp[:, 0]], "day": samp_day, "xfmr": pred_avg})
    return out


# ============================================================ ensemble config fit
def fit_ensemble_config(cfg_df):
    """cfg_df has columns day, g, y and per-family base preds; fit diversity
    weights + group-neutralisation alpha on the in-sample predictions."""
    for fam in FAMILIES:
        cfg_df[fam + "_z"] = per_day_z_np(cfg_df[fam].to_numpy(), cfg_df["day"].to_numpy())
    zcols = [f + "_z" for f in FAMILIES]
    sub = cfg_df.dropna(subset=zcols + ["y"]).copy()

    C = sub[zcols].corr().to_numpy()
    w = np.clip(np.linalg.pinv(C).sum(1), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(FAMILIES)) / len(FAMILIES)
    print(f"[ens] diversity weights {dict(zip(FAMILIES, np.round(w, 4)))}", flush=True)

    blend = (sub[zcols].to_numpy() * w).sum(1)
    sub = sub.assign(_blend=blend)

    def neutralize(a):
        gm = sub.groupby(["day", "g"])["_blend"].transform("mean")
        r = sub["_blend"] - a * gm
        gg = pd.DataFrame({"day": sub["day"], "r": r}).groupby("day")["r"]
        return ((r - gg.transform("mean")) / (gg.transform("std") + 1e-9)).to_numpy()

    ic_by_alpha = {}
    for a in ALPHA_GRID:
        ic_by_alpha[float(a)] = daily_ic_np(neutralize(a), sub["y"].to_numpy(), sub["day"].to_numpy())
    vmax = max(ic_by_alpha.values())
    alpha_insample = float(min(a for a in ALPHA_GRID if ic_by_alpha[float(a)] >= PLATEAU * vmax))

    # ship a robust alpha: the winning recipe's group-neutralisation strength was
    # selected OUT-OF-SAMPLE on the research valid split (alpha ~= 0.35). Fitting
    # alpha to maximise IN-SAMPLE IC is biased toward 0 (the bagged models partly
    # memorise the group tilt in-sample), so we only adopt the in-sample plateau
    # value when it lands in a sane positive band; otherwise we keep 0.35.
    alpha = alpha_insample if 0.2 <= alpha_insample <= 0.6 else RESEARCH_ALPHA
    final_ic = ic_by_alpha[alpha]
    print(f"[ens] in-sample IC max {vmax:.5f} @plateau alpha_insample={alpha_insample} "
          f"-> shipped alpha {alpha}  (in-sample blend IC {final_ic:.5f})", flush=True)

    return {
        "families": FAMILIES,
        "weights": [float(x) for x in w],
        "neut_alpha": alpha,
        "alpha_insample_plateau": alpha_insample,
        "K": K,
        "clip": 6.0,
        "in_sample_blend_ic": round(final_ic, 5),
        "ic_by_alpha": {str(k): round(v, 6) for k, v in ic_by_alpha.items()},
        "note": ("per-day z-score each base pred; blend with diversity weights; "
                 "then pred' = perday_z(blend - alpha * groupmean_g(blend)). "
                 "alpha carried from the OOS-validated research recipe (~0.35)."),
    }


# ============================================================ main
def main():
    t0 = time.time()
    print(f"[train] device={DEV}", flush=True)
    raw = pd.read_parquet(RAW)
    if MIN_DAY > 0:
        raw = raw[raw["day"] >= MIN_DAY].copy()
        print(f"[train] SMOKE slice: days >= {MIN_DAY}", flush=True)
    print(f"[train] raw {raw.shape}  days {raw['day'].min()}-{raw['day'].max()}", flush=True)

    print("[train] building/loading 263 features (213 base + 50 genalpha factors, fit mode)", flush=True)
    tf = time.time()
    import genalpha
    CACHE = ROOT / "artifacts" / "genalpha_263.parquet"
    ACACHE = ROOT / "artifacts" / "genalpha_artifacts.json"
    if MIN_DAY == 0 and CACHE.exists() and ACACHE.exists():
        feat = pd.read_parquet(CACHE)
        artifacts = json.load(open(ACACHE))
        fcols = artifacts["feature_list_263"]
        print(f"[train] loaded cached 263 features {feat.shape} ({len(fcols)} cols)", flush=True)
    else:
        feat, artifacts, fcols = genalpha.compute_263(raw, artifacts=None)
    json.dump(artifacts, open(WDIR / "feature_artifacts.json", "w"))
    print(f"[train] features {feat.shape}  ({len(fcols)} cols) in {time.time()-tf:.0f}s", flush=True)

    day = feat["day"].to_numpy()
    yraw = feat["y"].to_numpy("float32")
    yxs = feat["y_xs"].to_numpy("float32")
    X = feat[fcols].to_numpy("float32")
    labelled = ~np.isnan(yraw)

    udays = np.unique(day)
    rng = np.random.default_rng(MON_SEED)
    mon_days = set(rng.choice(udays, size=int(len(udays) * MON_FRAC), replace=False))
    mon_mask = np.isin(day, list(mon_days)) & labelled
    train_mask = labelled & ~mon_mask
    print(f"[train] monitor days {len(mon_days)}  train rows {int(train_mask.sum()):,} "
          f"monitor rows {int(mon_mask.sum()):,}", flush=True)

    p_lgb = train_lightgbm(X, yxs, labelled)
    p_mlp = train_mlp(X, yxs, yraw, day, train_mask, mon_mask)
    xfmr_df = train_transformer(feat, fcols, mon_days)

    cfg_df = pd.DataFrame({"day": day, "g": feat["g"].to_numpy(), "y": yraw,
                           "instrument_id": feat["instrument_id"].to_numpy(),
                           "lightgbm": p_lgb, "mlp": p_mlp})
    cfg_df = cfg_df.merge(xfmr_df, on=["instrument_id", "day"], how="left").rename(
        columns={"xfmr": "transformer"})
    cfg = fit_ensemble_config(cfg_df)
    json.dump(cfg, open(WDIR / "ensemble_config.json", "w"), indent=2)

    print(f"\n[train] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[train] weights -> {WDIR}", flush=True)
    for f in sorted(WDIR.glob("*")):
        print(f"   {f.name:26s} {f.stat().st_size/1e6:7.2f} MB", flush=True)


# ---- MLP with correct monitor gathering (uses _gather_pred) ----
def train_mlp(X, yxs, yraw, day, train_mask, mon_mask):
    print(f"[mlp] training multi-task DCN-MLP ({len(MLP_SEEDS)} seeds)", flush=True)
    Xg = torch.from_numpy(X).to(DEV)
    tr = np.where(train_mask)[0]
    mo = np.where(mon_mask)[0]
    ytr = torch.from_numpy(yxs[tr]).to(DEV)
    sgn = torch.from_numpy((yraw[tr] > 0).astype("float32")).to(DEV)
    mag = torch.from_numpy(np.abs(yxs[tr]).astype("float32")).to(DEV)
    tr_t = torch.from_numpy(tr).to(DEV)
    yraw_mo, day_mo = yraw[mo], day[mo]
    d = X.shape[1]
    all_preds = np.zeros(len(X), dtype="float64")

    for sd in MLP_SEEDS:
        t0 = time.time()
        torch.manual_seed(sd); np.random.seed(sd)
        net = MTMLP(d).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
        n = len(tr); bs = 8192
        best_ic, best_state, bad = -9.0, None, 0
        for ep in range(MLP_EPOCHS):
            net.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = tr_t[perm[i:i + bs]]
                opt.zero_grad()
                m, s, gg = net(Xg[idx])
                loss = (F.smooth_l1_loss(m, ytr[perm[i:i + bs]])
                        + 0.3 * F.binary_cross_entropy_with_logits(s, sgn[perm[i:i + bs]])
                        + 0.3 * F.mse_loss(gg, mag[perm[i:i + bs]]))
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            net.eval()
            with torch.no_grad():
                pv = _gather_pred(net, Xg, mo)
            ic = daily_ic_np(pv, yraw_mo, day_mo)
            if ic > best_ic + 1e-5:
                best_ic, bad = ic, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(best_state); net.eval()
        torch.save({k: v.cpu() for k, v in best_state.items()}, WDIR / f"mlp_seed{sd}.pt")
        with torch.no_grad():
            pred = np.concatenate([net(Xg[j:j + 200000])[0].cpu().numpy()
                                   for j in range(0, len(X), 200000)])
        all_preds += pred
        print(f"   seed {sd}: monitor IC {best_ic:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return all_preds / len(MLP_SEEDS)


if __name__ == "__main__":
    main()
