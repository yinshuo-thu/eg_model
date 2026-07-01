"""
EG model — leak-free feature engineering.

Metric is the daily cross-sectional Pearson IC, so the workhorse transform is a
per-day cross-sectional z-score of every feature. On top of the 86 raw x's we add
per-instrument temporal features (lags / momentum / rolling stats of y and the
strongest x's), price/volume features built from prc1..prc5 & vol0, group-aware
features from g, and cross-sectional PCA factors that denoise the common
structure. Everything is causal: temporal ops use only shift()>=0 within an
instrument; cross-sectional ops use only same-day rows.

Output: artifacts/features.parquet  (+ artifacts/feature_list.json)
Splits: train day<=760, valid 761-880, test 881-1259 (test = the y_hat0 eval period).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/eg_model")
RAW = ROOT / "artifacts" / "panel_raw.parquet"
OUT = ROOT / "artifacts" / "features.parquet"
TRAIN_END, VALID_END = 760, 880
XCOLS = [f"x_{i}" for i in range(86)]
CLIP = 6.0


def cs_zscore(df: pd.DataFrame, cols: list[str], by: str = "day") -> pd.DataFrame:
    g = df.groupby(by, sort=False)
    mean = g[cols].transform("mean")
    std = g[cols].transform("std")
    z = (df[cols] - mean) / (std + 1e-9)
    return z.clip(-CLIP, CLIP)


def daily_ic(pred: pd.Series, y: pd.Series, day: pd.Series) -> float:
    d = pd.DataFrame({"p": pred.values, "t": y.values, "day": day.values}).dropna()
    return d.groupby("day").apply(lambda s: s["p"].corr(s["t"])).mean()


def main() -> None:
    t0 = time.time()
    print("[feat] loading raw panel", flush=True)
    df = pd.read_parquet(RAW)
    df = df.sort_values(["instrument_id", "day"]).reset_index(drop=True)
    n = len(df)
    print(f"[feat] {n:,} rows, {df['day'].nunique()} days, {df['instrument_id'].nunique()} instruments", flush=True)

    feat = pd.DataFrame({
        "day": df["day"].astype("int16"),
        "instrument_id": df["instrument_id"].astype("int32"),
        "g": df["g"].astype("int16"),
        "y": df["y"].astype("float32"),
    })
    feat["split"] = np.where(df["day"] <= TRAIN_END, "train",
                      np.where(df["day"] <= VALID_END, "valid", "test"))

    # ---- 1) Cross-sectional z-score of all 86 raw x (impute via CS, NaN->0) ----
    print("[feat] (1) cross-sectional z-score of raw x", flush=True)
    xz = cs_zscore(df, XCOLS).fillna(0.0)
    xz.columns = [f"{c}z" for c in XCOLS]
    feat = pd.concat([feat, xz.astype("float32")], axis=1)

    # ---- pick strongest x by |train daily IC| for temporal expansion ----
    print("[feat] ranking x by train daily IC", flush=True)
    tr = df["day"] <= TRAIN_END
    ics = {}
    day_tr = df.loc[tr, "day"]
    y_tr = df.loc[tr, "y"]
    for c in XCOLS:
        ics[c] = abs(daily_ic(xz.loc[tr, f"{c}z"], y_tr, day_tr))
    top_x = [c for c, _ in sorted(ics.items(), key=lambda kv: -kv[1])[:30]]
    print("[feat] top x:", top_x[:12], "...", flush=True)

    gi = df.groupby("instrument_id", sort=False)

    # ---- 2) Temporal features on y (autocorrelation / momentum / vol) ----
    print("[feat] (2) temporal y features", flush=True)
    yv = df["y"]
    tmp = {}
    for L in (1, 2, 3, 5):
        tmp[f"y_lag{L}"] = gi["y"].shift(L)
    ylag1 = gi["y"].shift(1)
    for W in (5, 10, 20):
        tmp[f"y_roll{W}"] = ylag1.groupby(df["instrument_id"], sort=False).transform(lambda s: s.rolling(W, min_periods=2).mean())
        tmp[f"y_vol{W}"] = ylag1.groupby(df["instrument_id"], sort=False).transform(lambda s: s.rolling(W, min_periods=2).std())
    tmp["y_ewm"] = ylag1.groupby(df["instrument_id"], sort=False).transform(lambda s: s.ewm(span=10, min_periods=2).mean())
    ytemp = pd.DataFrame(tmp, index=df.index)
    ytemp_z = cs_zscore(pd.concat([df["day"], ytemp], axis=1), list(tmp.keys())).fillna(0.0)
    ytemp_z.columns = [f"{c}_z" for c in ytemp_z.columns]
    feat = pd.concat([feat, ytemp_z.astype("float32")], axis=1)

    # ---- 3) Temporal features on strongest x's: lag1, momentum(5), rolling mean(10) ----
    print("[feat] (3) temporal x features (top-30)", flush=True)
    xtmp = {}
    for c in top_x:
        s = df[c]
        lag1 = gi[c].shift(1)
        xtmp[f"{c}_lag1"] = lag1
        xtmp[f"{c}_mom5"] = s - gi[c].shift(5)
        xtmp[f"{c}_r10"] = lag1.groupby(df["instrument_id"], sort=False).transform(lambda v: v.rolling(10, min_periods=3).mean())
    xtmp = pd.DataFrame(xtmp, index=df.index)
    xtmp_z = cs_zscore(pd.concat([df["day"], xtmp], axis=1), list(xtmp.columns)).fillna(0.0)
    xtmp_z.columns = [f"{c}_z" for c in xtmp_z.columns]
    feat = pd.concat([feat, xtmp_z.astype("float32")], axis=1)

    # ---- 4) Price / volume features from prc1..prc5, vol0 ----
    print("[feat] (4) price/volume features", flush=True)
    prc = df[[f"prc{i}" for i in range(1, 6)]]
    pv = pd.DataFrame(index=df.index)
    pv["prc_mean"] = prc.mean(axis=1)
    pv["prc_std"] = prc.std(axis=1)
    pv["prc_rng"] = prc.max(axis=1) - prc.min(axis=1)
    pv["prc_1m5"] = df["prc1"] - df["prc5"]
    pv["prc_1m3"] = df["prc1"] - df["prc3"]
    pv["prc_skew"] = df["prc1"] - 2 * df["prc3"] + df["prc5"]
    pv["vol0"] = df["vol0"]
    pmean = pv["prc_mean"]
    pv["prc_mom1"] = pmean - pmean.groupby(df["instrument_id"], sort=False).shift(1)
    pv["prc_mom5"] = pmean - pmean.groupby(df["instrument_id"], sort=False).shift(5)
    pv["vol0_lag1"] = df["vol0"].groupby(df["instrument_id"], sort=False).shift(1)
    pv["vol0_chg"] = df["vol0"] - pv["vol0_lag1"]
    pv_z = cs_zscore(pd.concat([df["day"], pv], axis=1), list(pv.columns)).fillna(0.0)
    pv_z.columns = [f"{c}_z" for c in pv_z.columns]
    feat = pd.concat([feat, pv_z.astype("float32")], axis=1)

    # ---- 5) Group-aware features (g): group factor = per-(day,g) mean of CS signals ----
    print("[feat] (5) group features", flush=True)
    # group momentum: average lagged-y within (day, group) — a leak-free group return
    glag = pd.DataFrame({"day": df["day"], "g": df["g"], "ylag1": ylag1.fillna(0.0)})
    gmean = glag.groupby(["day", "g"], sort=False)["ylag1"].transform("mean")
    feat["grp_ylag_mean"] = cs_zscore(pd.DataFrame({"day": df["day"], "v": gmean.values}), ["v"]).fillna(0.0)["v"].astype("float32")
    # group mean of the single strongest contemporaneous signal (cross-sectional group tilt)
    best = top_x[0]
    gbest = pd.DataFrame({"day": df["day"], "g": df["g"], "v": xz[f"{best}z"].values})
    gbm = gbest.groupby(["day", "g"], sort=False)["v"].transform("mean")
    feat["grp_xbest_mean"] = gbm.astype("float32").fillna(0.0)
    # group target-encoding: expanding mean of y per group up to the PREVIOUS day (no leak)
    enc = df[["day", "g", "y"]].copy()
    daygrp = enc.groupby(["g", "day"], sort=False)["y"].mean().reset_index()  # mean y per group-day
    daygrp = daygrp.sort_values(["g", "day"])
    daygrp["cum"] = daygrp.groupby("g")["y"].cumsum() - daygrp["y"]
    daygrp["cnt"] = daygrp.groupby("g").cumcount()
    daygrp["g_te"] = (daygrp["cum"] / daygrp["cnt"].clip(lower=1)).where(daygrp["cnt"] > 0, 0.0)
    te_map = {(r.g, r.day): r.g_te for r in daygrp.itertuples()}
    feat["grp_te"] = [te_map.get((g_, d_), 0.0) for g_, d_ in zip(df["g"].values, df["day"].values)]
    feat["grp_te"] = feat["grp_te"].astype("float32")

    # ---- 6) Cross-sectional PCA factors (denoise common structure of raw x) ----
    print("[feat] (6) cross-sectional PCA factors", flush=True)
    from numpy.linalg import svd
    # fit PCA on TRAIN cross-section (z-scored x), project all days
    Xtr = xz.loc[tr, [f"{c}z" for c in XCOLS]].to_numpy(dtype=np.float32)
    Xtr = Xtr - Xtr.mean(0, keepdims=True)
    # SVD on a random subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(Xtr.shape[0], size=min(200000, Xtr.shape[0]), replace=False)
    U, S, Vt = svd(Xtr[idx], full_matrices=False)
    comps = Vt[:12]  # (12, 86)
    Xall = xz[[f"{c}z" for c in XCOLS]].to_numpy(dtype=np.float32)
    pcs = Xall @ comps.T  # (n, 12)
    pcs_df = pd.DataFrame(pcs, columns=[f"pca{i}" for i in range(12)], index=df.index)
    pcs_z = cs_zscore(pd.concat([df["day"], pcs_df], axis=1), list(pcs_df.columns)).fillna(0.0)
    pcs_z.columns = [f"{c}_z" for c in pcs_z.columns]
    feat = pd.concat([feat, pcs_z.astype("float32")], axis=1)

    # ---- target: per-day cross-sectional z-score of y (training target) ----
    feat["y_xs"] = cs_zscore(df[["day", "y"]], ["y"]).fillna(0.0)["y"].astype("float32")

    feat_cols = [c for c in feat.columns if c not in ("day", "instrument_id", "g", "split", "y", "y_xs")]
    print(f"[feat] total features: {len(feat_cols)}", flush=True)
    feat = feat.sort_values(["day", "instrument_id"]).reset_index(drop=True)
    feat.to_parquet(OUT, index=False)
    (ROOT / "artifacts" / "feature_list.json").write_text(json.dumps(feat_cols, indent=0))
    print(f"[feat] wrote {OUT} ({len(feat):,} rows, {len(feat.columns)} cols) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
