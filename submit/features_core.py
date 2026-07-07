"""Leak-free causal feature engineering for the Engineering-Gates OOS package.

This is a self-contained, refactored version of ``tools/build_features.py`` that
can be *fit* on labelled history and then *applied* to new rows.  The only
fit-time state is (a) the list of strongest ``x`` columns (ranked by |daily IC|)
that receive a temporal expansion and (b) the 12 cross-sectional PCA loadings.
Both are saved to ``weights/feature_artifacts.json`` so that predictions on new
out-of-sample rows recompute *exactly* the same 200 base features the models were
trained on.

Every transform is causal, and NO feature reads an instrument's own y:
  * temporal ops use only ``shift() >= 1`` within an instrument,
  * cross-sectional ops (z-score / PCA projection / group means) use only the
    same day's rows,
  * the autoregressive y-history and past-y group encodings have been removed, so
    the base features are pure functions of x / prc / vol / g and are fully
    computable on an OOS block whose label y is withheld.

Public API
----------
compute_features(raw_df, artifacts=None) -> (feat_df, artifacts)
    artifacts is None  -> FIT   mode (rank x, fit PCA), returns fitted artifacts.
    artifacts is dict  -> APPLY mode, reuses the saved top_x / PCA loadings.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

XCOLS = [f"x_{i}" for i in range(86)]
PRCCOLS = [f"prc{i}" for i in range(1, 6)]
CLIP = 6.0
TOPN = 30          # number of strongest x's that get a temporal expansion
NPCA = 12          # cross-sectional PCA factors
PCA_SUBSAMPLE = 200_000


# ----------------------------------------------------------------------------- helpers
def cs_zscore(df: pd.DataFrame, cols: list[str], by: str = "day") -> pd.DataFrame:
    """Per-day cross-sectional z-score, clipped to +-CLIP."""
    g = df.groupby(by, sort=False)
    mean = g[cols].transform("mean")
    std = g[cols].transform("std")
    z = (df[cols] - mean) / (std + 1e-9)
    return z.clip(-CLIP, CLIP)


def _daily_ic(pred: pd.Series, y: pd.Series, day: pd.Series) -> float:
    d = pd.DataFrame({"p": np.asarray(pred), "t": np.asarray(y), "day": np.asarray(day)}).dropna()
    if d.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic = d.groupby("day").apply(lambda s: s["p"].corr(s["t"]), include_groups=False)
    return float(ic.mean())


# ----------------------------------------------------------------------------- main
def compute_features(raw: pd.DataFrame, artifacts: dict | None = None):
    """Build the 200 leak-free base features (no autoregressive y-history).

    Parameters
    ----------
    raw : DataFrame with columns day, instrument_id, g, x_0..x_85, prc1..prc5,
          vol0 and (optionally) y.  ``y`` may be missing/NaN for the days to
          predict; the causal features never use the current day's y.
    artifacts : if None, fit top_x + PCA on the labelled rows (y observed) and
          return them; otherwise reuse the supplied ``top_x`` / ``pca_comps``.

    Returns
    -------
    (feat_df, artifacts)
        feat_df sorted by (day, instrument_id) with columns
        [day, instrument_id, g, y, y_xs, <200 features>].
    """
    fit = artifacts is None
    df = raw.copy()
    if "y" not in df.columns:
        df["y"] = np.nan
    # types + causal ordering
    df["day"] = df["day"].astype("int64")
    df["instrument_id"] = df["instrument_id"].astype("int64")
    df["g"] = df["g"].astype("int64")
    df = df.sort_values(["instrument_id", "day"]).reset_index(drop=True)

    feat = pd.DataFrame({
        "day": df["day"].astype("int32"),
        "instrument_id": df["instrument_id"].astype("int32"),
        "g": df["g"].astype("int32"),
        "y": df["y"].astype("float32"),
    })

    # ---- 1) cross-sectional z-score of all 86 raw x (CS impute, NaN->0) ----
    xz = cs_zscore(df, XCOLS).fillna(0.0)
    xz.columns = [f"{c}z" for c in XCOLS]
    feat = pd.concat([feat, xz.astype("float32")], axis=1)

    # ---- pick strongest x by |daily IC| on labelled rows (fit) ----
    if fit:
        obs = df["y"].notna().to_numpy()
        ics = {}
        day_o = df.loc[obs, "day"]
        y_o = df.loc[obs, "y"]
        for c in XCOLS:
            ics[c] = abs(_daily_ic(xz.loc[obs, f"{c}z"], y_o, day_o))
        top_x = [c for c, _ in sorted(ics.items(), key=lambda kv: -kv[1])[:TOPN]]
    else:
        top_x = list(artifacts["top_x"])

    gi = df.groupby("instrument_id", sort=False)
    iid = df["instrument_id"]

    # ---- 2) [REMOVED] autoregressive y-history features (y_lag{1,2,3,5} / y_roll /
    #         y_vol / y_ewm).  These read an instrument's OWN recently-realised y as a
    #         direct model input, which is unavailable when the OOS label is withheld
    #         (and would leak if the forward horizon were >1 day).  Dropped so every
    #         feature is computable on a withheld-label OOS block.

    # ---- 3) temporal features on strongest x's: lag1, momentum(5), rolling mean(10) ----
    xtmp = {}
    for c in top_x:
        s = df[c]
        lag1 = gi[c].shift(1)
        xtmp[f"{c}_lag1"] = lag1
        xtmp[f"{c}_mom5"] = s - gi[c].shift(5)
        xtmp[f"{c}_r10"] = lag1.groupby(iid, sort=False).transform(
            lambda v: v.rolling(10, min_periods=3).mean())
    xtmp = pd.DataFrame(xtmp, index=df.index)
    xtmp_z = cs_zscore(pd.concat([df["day"], xtmp], axis=1), list(xtmp.columns)).fillna(0.0)
    xtmp_z.columns = [f"{c}_z" for c in xtmp_z.columns]
    feat = pd.concat([feat, xtmp_z.astype("float32")], axis=1)

    # ---- 4) price / volume features from prc1..prc5, vol0 ----
    prc = df[PRCCOLS]
    pv = pd.DataFrame(index=df.index)
    pv["prc_mean"] = prc.mean(axis=1)
    pv["prc_std"] = prc.std(axis=1)
    pv["prc_rng"] = prc.max(axis=1) - prc.min(axis=1)
    pv["prc_1m5"] = df["prc1"] - df["prc5"]
    pv["prc_1m3"] = df["prc1"] - df["prc3"]
    pv["prc_skew"] = df["prc1"] - 2 * df["prc3"] + df["prc5"]
    pv["vol0"] = df["vol0"]
    pmean = pv["prc_mean"]
    pv["prc_mom1"] = pmean - pmean.groupby(iid, sort=False).shift(1)
    pv["prc_mom5"] = pmean - pmean.groupby(iid, sort=False).shift(5)
    pv["vol0_lag1"] = df["vol0"].groupby(iid, sort=False).shift(1)
    pv["vol0_chg"] = df["vol0"] - pv["vol0_lag1"]
    pv_z = cs_zscore(pd.concat([df["day"], pv], axis=1), list(pv.columns)).fillna(0.0)
    pv_z.columns = [f"{c}_z" for c in pv_z.columns]
    feat = pd.concat([feat, pv_z.astype("float32")], axis=1)

    # ---- 5) group-aware features (g) ----
    # [REMOVED] grp_ylag_mean (per-(day,g) mean of an instrument's lagged y) — an
    #           autoregressive past-y group feature; dropped for a leak-safe OOS.
    # group mean of the strongest contemporaneous x signal (cross-sectional group
    # tilt) — pure-x, kept.
    best = top_x[0]
    gbest = pd.DataFrame({"day": df["day"], "g": df["g"], "v": xz[f"{best}z"].values})
    gbm = gbest.groupby(["day", "g"], sort=False)["v"].transform("mean")
    feat["grp_xbest_mean"] = gbm.astype("float32").fillna(0.0)
    # [REMOVED] grp_te (expanding-window group target-encoding of y) — a past-y group
    #           feature; dropped for a leak-safe OOS.

    # ---- 6) cross-sectional PCA factors (denoise common structure of raw x) ----
    xzcols = [f"{c}z" for c in XCOLS]
    if fit:
        obs = df["y"].notna().to_numpy()
        Xf = xz.loc[obs, xzcols].to_numpy(dtype=np.float32)
        Xf = Xf - Xf.mean(0, keepdims=True)
        rng = np.random.default_rng(0)
        idx = rng.choice(Xf.shape[0], size=min(PCA_SUBSAMPLE, Xf.shape[0]), replace=False)
        _, _, Vt = np.linalg.svd(Xf[idx], full_matrices=False)
        comps = Vt[:NPCA]                                  # (NPCA, 86)
    else:
        comps = np.asarray(artifacts["pca_comps"], dtype=np.float32)
    Xall = xz[xzcols].to_numpy(dtype=np.float32)
    pcs = Xall @ comps.T                                   # (n, NPCA)
    pcs_df = pd.DataFrame(pcs, columns=[f"pca{i}" for i in range(NPCA)], index=df.index)
    pcs_z = cs_zscore(pd.concat([df["day"], pcs_df], axis=1), list(pcs_df.columns)).fillna(0.0)
    pcs_z.columns = [f"{c}_z" for c in pcs_z.columns]
    feat = pd.concat([feat, pcs_z.astype("float32")], axis=1)

    # ---- training target: per-day cross-sectional z-score of y ----
    feat["y_xs"] = cs_zscore(df[["day", "y"]], ["y"]).fillna(0.0)["y"].astype("float32")

    feat_cols = [c for c in feat.columns if c not in ("day", "instrument_id", "g", "y", "y_xs")]
    feat = feat.sort_values(["day", "instrument_id"]).reset_index(drop=True)

    if fit:
        artifacts = {
            "top_x": top_x,
            "pca_comps": comps.astype("float32").tolist(),
            "feature_list": feat_cols,
            "clip": CLIP,
            "topn": TOPN,
            "npca": NPCA,
        }
    return feat, artifacts


def feature_list(artifacts: dict) -> list[str]:
    return list(artifacts["feature_list"])
