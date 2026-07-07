"""genalpha — regenerate the full 250-feature matrix (200 causal base features +
50 curated alpha factors) from a RAW panel, causally, so the same code that made
the training features also makes them on a brand-new OOS block.

Public API:
    compute(panel_df, artifacts=None) -> (feat_df, artifacts, feature_list)

`panel_df` schema: day, instrument_id, g, x_0..x_85, prc1..prc5, vol0[, y].
Include enough per-instrument history before the day(s) you want scored (the
temporal features and the 50 factors' EWM/rolling states warm up; supervised
factors freeze once y runs out, so pass the labelled history too).

The 200 base come from submit/features_core.py (proven causal, frozen top-x/PCA
artifacts; NO autoregressive y-history — every base feature is a pure function of
x / prc / vol / g). The 50 curated factors come from the Ctx-based factor modules;
each is a SUPERVISED factor that learns cross-sectional weights/signs/payoffs from
*past* y (>=1-day delay) and applies them to the current day's x — its supervised
state FREEZES on unlabelled (OOS) days, so no row ever reads its own or any future
y and the whole matrix is computable on a withheld-label OOS block.
"""
from __future__ import annotations
import json, importlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # for features_core
from core import Ctx
import features_core

# the 50 curated factors, in canonical confpos50 order (matches the report's 263)
CONFPOS50 = json.load(open(HERE / "confpos50.json")) if (HERE / "confpos50.json").exists() else None

# factor modules -> each exposes gen(ctx) -> {name: np.ndarray}
FACTOR_MODULES = ["factors_charpay", "factors_pw", "factors_icw_sup",
                  "factors_sc_ff", "factors_gx", "factors_miscA", "factors_miscB"]


def _collect_factors(ctx) -> dict:
    out = {}
    for mod in FACTOR_MODULES:
        m = importlib.import_module(mod)
        out.update(m.gen(ctx))
    return out


def compute(panel_df: pd.DataFrame, artifacts: dict | None = None):
    # 200 base features (sorted day,instrument_id) + frozen artifacts
    feat, artifacts = features_core.compute_features(panel_df, artifacts=artifacts)

    # attach the SAME live PCA controls the base uses onto the panel, so the 50
    # factors' PCA-orthogonalisation is self-consistent and OOS-portable (no
    # dependency on any precomputed features.parquet)
    pca_cols = [f"pca{i}_z" for i in range(12)]
    panel_aug = panel_df.merge(feat[["day", "instrument_id"] + pca_cols],
                               on=["day", "instrument_id"], how="left")

    # 50 curated factors from the same raw panel (+ live PCA)
    ctx = Ctx(panel_aug)
    fac = _collect_factors(ctx)
    fac_df = pd.DataFrame({"day": ctx.day_vals, "instrument_id": ctx.inst.to_numpy()})
    order = CONFPOS50 if CONFPOS50 is not None else list(fac.keys())
    for nm in order:
        if nm not in fac:
            raise KeyError(f"factor {nm} not produced by any genalpha module")
        fac_df[nm] = np.nan_to_num(np.asarray(fac[nm], dtype="float32"))

    # align factors onto the base rows by (day, instrument_id)
    feat = feat.merge(fac_df, on=["day", "instrument_id"], how="left")
    for nm in order:
        feat[nm] = feat[nm].fillna(0.0).astype("float32")

    base_list = list(artifacts["feature_list"])
    feature_list = base_list + list(order)
    # feature_list_full is the canonical 250-col training/inference order; the
    # legacy feature_list_263 alias is kept so older callers still resolve.
    artifacts = dict(artifacts, feature_list_full=feature_list, feature_list_263=feature_list,
                     base_feature_list=base_list, factor_list=list(order))
    return feat, artifacts, feature_list


# backward-compatible alias (older scripts import compute_263)
compute_263 = compute
