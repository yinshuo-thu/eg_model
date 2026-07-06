"""Engineering-Gates OOS prediction API.

Loads the full-in-sample-trained LightGBM + MLP + Transformer ensemble from
``submit/weights/`` and predicts the daily cross-sectional signal ``y_hat`` on
new raw-panel rows.

Public API
----------
predict(panel_df, predict_days=None) -> DataFrame[day, instrument_id, y_hat]

    panel_df : a RAW panel with the training schema
        day, instrument_id, x_0..x_85, prc1..prc5, vol0, g   (+ optional y)
      It MUST contain enough per-instrument HISTORY before the day(s) you want
      predicted so the causal temporal features and the Transformer's K-day
      window can be formed.  Recommended: >= ~60 prior days of history (the
      Transformer looks back K=32 days and the rolling features warm up over
      ~20 days).  Passing the entire available history is best (it makes the
      group target-encoding feature exact).

    predict_days : optional iterable of day values to predict.  If omitted:
        * if `y` is present with both observed & missing values -> predict the
          days that contain missing y (the OOS rows);
        * otherwise -> predict every day that has a full K-day lookback in the
          input.

    Returns one row per (day, instrument_id) for the predicted days, with the
    final ensemble signal ``y_hat`` (per-day z-scored, diversity-blended and
    group-neutralised).  Higher y_hat == more positive expected forward return;
    only the cross-sectional ranking per day is meaningful (the metric is the
    per-day cross-sectional Pearson IC).

CLI
---
    python submit/predict.py --input panel.parquet --output preds.parquet
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import genalpha
from models import MTMLP, EGTransformer

WDIR = HERE / "weights"
# device: env EG_DEVICE ('cpu'/'cuda') overrides; else auto. Lets an evaluator
# force CPU when the GPU is busy/small (the neural nets fit easily on CPU).
DEV = os.environ.get("EG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _per_day_z(vals, day):
    s = pd.DataFrame({"v": np.asarray(vals, "float64"), "day": np.asarray(day)})
    g = s.groupby("day")["v"]
    return ((s["v"] - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()


class Predictor:
    def __init__(self, weights_dir: str | Path = WDIR, device: str = DEV):
        self.wdir = Path(weights_dir)
        self.device = device
        self.artifacts = json.load(open(self.wdir / "feature_artifacts.json"))
        self.cfg = json.load(open(self.wdir / "ensemble_config.json"))
        self.fcols = self.artifacts.get("feature_list_263", self.artifacts["feature_list"])
        self.Fn = len(self.fcols)
        self.K = int(self.cfg.get("K", 32))
        self.families = self.cfg["families"]
        self.weights = np.asarray(self.cfg["weights"], dtype="float64")
        self.alpha = float(self.cfg["neut_alpha"])

        # LightGBM boosters
        self.lgb = [lgb.Booster(model_file=str(p))
                    for p in sorted(self.wdir.glob("lgb_dart_seed*.txt"))]
        # MLP seed bag
        self.mlp = []
        for p in sorted(self.wdir.glob("mlp_seed*.pt")):
            net = MTMLP(self.Fn).to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            net.eval()
            self.mlp.append(net)
        # Transformer seed bag
        self.xfmr = []
        for p in sorted(self.wdir.glob("xfmr_seed*.pt")):
            net = EGTransformer(self.Fn, K=self.K).to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            net.eval()
            self.xfmr.append(net)
        if not (self.lgb and self.mlp and self.xfmr):
            raise FileNotFoundError(
                f"missing model weights in {self.wdir} "
                f"(lgb={len(self.lgb)} mlp={len(self.mlp)} xfmr={len(self.xfmr)})")

    # ------------------------------------------------------------------ base models
    def _pred_lgb(self, X):
        return np.mean([m.predict(X) for m in self.lgb], axis=0)

    def _pred_mlp(self, X):
        Xg = torch.from_numpy(X.astype("float32")).to(self.device)
        acc = np.zeros(len(X), dtype="float64")
        for net in self.mlp:
            with torch.no_grad():
                acc += np.concatenate([net(Xg[i:i + 200000])[0].cpu().numpy()
                                       for i in range(0, len(X), 200000)])
        return acc / len(self.mlp)

    def _pred_xfmr(self, feat, target_days):
        """Transformer preds for (instrument, day) with day in target_days and a
        full K-day lookback.  Returns dict {(instrument_id, day): pred}."""
        df = feat.sort_values(["instrument_id", "day"]).reset_index(drop=True)
        insts = np.sort(df["instrument_id"].unique())
        days = np.sort(df["day"].unique())
        iidx = {v: k for k, v in enumerate(insts)}
        didx = {v: k for k, v in enumerate(days)}
        NI, ND = len(insts), len(days)
        panel = np.zeros((NI, ND, self.Fn), dtype="float32")
        ii = df["instrument_id"].map(iidx).to_numpy()
        dd = df["day"].map(didx).to_numpy()
        panel[ii, dd] = df[self.fcols].to_numpy("float32")
        P = torch.from_numpy(panel).to(self.device)
        offs = torch.arange(-(self.K - 1), 1, device=self.device)

        tgt_didx = [didx[d] for d in target_days if d in didx and didx[d] >= self.K - 1]
        if not tgt_didx:
            return {}
        samp = np.array([(i, di) for di in tgt_didx for i in range(NI)], dtype=np.int64)
        samp_t = torch.from_numpy(samp).to(self.device)

        def gather(idx_t):
            iis = samp_t[idx_t, 0]; dds = samp_t[idx_t, 1]
            win_d = dds.unsqueeze(1) + offs.unsqueeze(0)
            return P[iis.unsqueeze(1), win_d]

        acc = np.zeros(len(samp), dtype="float64")
        for net in self.xfmr:
            pred = np.empty(len(samp), "float32")
            with torch.no_grad():
                for i in range(0, len(samp), 16384):
                    b = torch.arange(i, min(i + 16384, len(samp)), device=self.device)
                    pred[i:i + 16384] = net(gather(b))[0].float().cpu().numpy()
            acc += pred
        acc /= len(self.xfmr)
        inst_of = insts[samp[:, 0]]
        day_of = days[samp[:, 1]]
        return {(int(iv), int(dv)): float(pv) for iv, dv, pv in zip(inst_of, day_of, acc)}

    # ------------------------------------------------------------------ public
    def predict(self, panel_df: pd.DataFrame, predict_days=None,
                model: str = "ensemble") -> pd.DataFrame:
        """model: 'ensemble' (default, the diversity-weighted group-neutralised
        blend of all families) or a single sub-model 'lightgbm' / 'mlp' /
        'transformer' (per-day z-scored). Any other value raises."""
        allowed = {"ensemble"} | set(self.families)
        if model not in allowed:
            raise ValueError(f"model={model!r}; choose one of {sorted(allowed)}")
        feat, _, _ = genalpha.compute_263(panel_df, artifacts=self.artifacts)

        # ---- decide which days to score ----
        udays = np.sort(feat["day"].unique())
        if predict_days is not None:
            target_days = sorted(int(d) for d in predict_days)
        elif "y" in panel_df.columns and panel_df["y"].isna().any() and panel_df["y"].notna().any():
            miss = panel_df.loc[panel_df["y"].isna(), "day"].unique()
            target_days = sorted(int(d) for d in miss)
        else:
            target_days = [int(d) for d in udays[self.K - 1:]]
        target_set = set(target_days)

        tmask = feat["day"].isin(target_set).to_numpy()
        sub = feat.loc[tmask, ["day", "instrument_id", "g"]].reset_index(drop=True).copy()
        Xt = feat.loc[tmask, self.fcols].to_numpy("float32")

        # ---- base predictions on the target rows ----
        base = {}
        base["lightgbm"] = self._pred_lgb(Xt)
        base["mlp"] = self._pred_mlp(Xt)
        xf = self._pred_xfmr(feat, target_days)
        sub["_key"] = list(zip(sub["instrument_id"].astype(int), sub["day"].astype(int)))
        base["transformer"] = np.array([xf.get(k, np.nan) for k in sub["_key"]], dtype="float64")

        for fam in self.families:
            sub[fam] = base[fam]

        # rows must have all 3 base preds finite (Transformer needs full history)
        ok = np.isfinite(sub[self.families].to_numpy()).all(1)
        n_drop = int((~ok).sum())
        if n_drop:
            print(f"[predict] warning: {n_drop} target rows dropped for insufficient "
                  f"history (Transformer needs >= K={self.K} prior days)", flush=True)
        sub = sub.loc[ok].reset_index(drop=True)

        if model != "ensemble":
            # single sub-model: per-day z-score of that family (only ranking matters)
            sub["y_hat"] = _per_day_z(sub[model].to_numpy(), sub["day"].to_numpy())
        else:
            # ---- per-day z-score each base pred, diversity blend ----
            Z = np.column_stack([_per_day_z(sub[f].to_numpy(), sub["day"].to_numpy())
                                 for f in self.families])
            blend = (Z * self.weights).sum(1)
            sub["_blend"] = blend
            # ---- group-g neutralisation, then per-day z ----
            gm = sub.groupby(["day", "g"])["_blend"].transform("mean")
            r = sub["_blend"] - self.alpha * gm
            sub["y_hat"] = _per_day_z(r.to_numpy(), sub["day"].to_numpy())

        out = sub[["day", "instrument_id", "y_hat"]].copy()
        out = out.sort_values(["day", "instrument_id"]).reset_index(drop=True)
        return out


_PREDICTOR: Predictor | None = None


def predict(panel_df: pd.DataFrame, predict_days=None,
            weights_dir: str | Path = WDIR, model: str = "ensemble",
            device: str | None = None) -> pd.DataFrame:
    """Module-level convenience wrapper (caches the loaded models).
    model: 'ensemble' (default) | 'lightgbm' | 'mlp' | 'transformer'.
    device: None -> auto (env EG_DEVICE or cuda-if-available); or 'cpu' / 'cuda'."""
    global _PREDICTOR
    dev = device or DEV
    if (_PREDICTOR is None or Path(_PREDICTOR.wdir) != Path(weights_dir)
            or _PREDICTOR.device != dev):
        _PREDICTOR = Predictor(weights_dir, device=dev)
    return _PREDICTOR.predict(panel_df, predict_days=predict_days, model=model)


def _main():
    ap = argparse.ArgumentParser(description="Engineering-Gates OOS prediction")
    ap.add_argument("--input", required=True, help="raw panel parquet/csv (schema: day, instrument_id, x_0..x_85, prc1..prc5, vol0, g[, y])")
    ap.add_argument("--output", required=True, help="output path (.parquet or .csv)")
    ap.add_argument("--days", default=None, help="comma-separated day values to predict (optional)")
    ap.add_argument("--weights", default=str(WDIR), help="weights directory")
    ap.add_argument("--model", default="ensemble",
                    choices=["ensemble", "lightgbm", "mlp", "transformer"],
                    help="which model's signal to output (default: ensemble)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="force compute device (default: auto / env EG_DEVICE)")
    args = ap.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp) if inp.suffix == ".csv" else pd.read_parquet(inp)
    days = [int(d) for d in args.days.split(",")] if args.days else None
    out = predict(df, predict_days=days, weights_dir=args.weights,
                  model=args.model, device=args.device)

    outp = Path(args.output)
    if outp.suffix == ".csv":
        out.to_csv(outp, index=False)
    else:
        out.to_parquet(outp, index=False)
    print(f"[predict] wrote {len(out):,} rows -> {outp}  "
          f"(days {out['day'].min()}-{out['day'].max()}, "
          f"y_hat finite={np.isfinite(out['y_hat']).mean():.3f})", flush=True)


if __name__ == "__main__":
    _main()
