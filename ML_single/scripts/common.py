"""Shared utilities for EG models: data loading, daily-IC metric, prediction IO."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/eg_model")
FEAT = ROOT / "artifacts" / "features.parquet"
PREDS = ROOT / "artifacts" / "preds"
PREDS.mkdir(parents=True, exist_ok=True)
METR = ROOT / "artifacts" / "metrics"
METR.mkdir(parents=True, exist_ok=True)


def feature_cols() -> list[str]:
    return json.loads((ROOT / "artifacts" / "feature_list.json").read_text())


def load(cols: list[str] | None = None) -> pd.DataFrame:
    base = ["day", "instrument_id", "g", "split", "y", "y_xs"]
    fcols = cols if cols is not None else feature_cols()
    return pd.read_parquet(FEAT, columns=base + fcols)


def daily_ic(df: pd.DataFrame, pred="pred", true="y", method="pearson") -> tuple[float, float, float]:
    ic = df.dropna(subset=[pred, true]).groupby("day").apply(
        lambda s: s[pred].corr(s[true], method=method))
    ic = ic.dropna()
    m, s = float(ic.mean()), float(ic.std())
    return m, s, (m / s if s > 0 else float("nan"))


def evaluate(df: pd.DataFrame, name: str, pred="pred") -> dict:
    out = {"model": name}
    for split in ("valid", "test"):
        sub = df[df["split"] == split]
        m, s, ir = daily_ic(sub, pred=pred)
        sm, _, _ = daily_ic(sub, pred=pred, method="spearman")
        pos = float((sub.dropna(subset=[pred, "y"]).groupby("day")
                     .apply(lambda g: g[pred].corr(g["y"])) > 0).mean())
        out[f"{split}_IC"] = round(m, 5)
        out[f"{split}_IR"] = round(ir, 3)
        out[f"{split}_spear"] = round(sm, 5)
        out[f"{split}_pos"] = round(pos, 3)
    return out


def save_pred(df: pd.DataFrame, name: str, pred="pred") -> None:
    keep = df[df["split"].isin(["valid", "test"])][["day", "instrument_id", "split", "y", pred]].copy()
    keep = keep.rename(columns={pred: "pred"})
    keep.to_parquet(PREDS / f"{name}.parquet", index=False)


def print_row(d: dict) -> None:
    print(f"  {d['model']:22s} valid IC {d['valid_IC']:.5f} IR {d['valid_IR']:.2f} | "
          f"test IC {d['test_IC']:.5f} IR {d['test_IR']:.2f} spear {d['test_spear']:.5f} pos {d['test_pos']:.2f}",
          flush=True)
