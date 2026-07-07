"""Transformer PROCESSED: single-model group-g neutralization applied to
transformer_{TAG}'s own prediction (alpha scanned on valid). Generic version
of Transformer/v1/scripts/neutralize_v3factors.py."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_generic import daily_ic, PREDS, ROOT, TAG, METR

ALPHA_GRID = np.round(np.arange(0.0, 0.86, 0.01), 2)
PLATEAU = 0.997


def per_day_z(df, col):
    g = df.groupby("day")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def neutralize_g(df, srccol, alpha, gcol="g"):
    gm = df.groupby(["day", gcol])[srccol].transform("mean")
    tmp = pd.DataFrame({"day": df["day"], "r": df[srccol] - alpha * gm})
    g = tmp.groupby("day")["r"]
    return (tmp["r"] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main():
    d = pd.read_parquet(PREDS / f"transformer_{TAG}.parquet")
    g = pd.read_parquet(ROOT / "artifacts" / "features.parquet", columns=["day", "instrument_id", "g"])
    d = d.merge(g, on=["day", "instrument_id"], how="left")
    d["pred"] = per_day_z(d, "pred")

    raw_te = d[d.split == "test"]
    raw_ic, _, raw_ir = daily_ic(raw_te)
    print(f"[neut-{TAG}] transformer_{TAG} RAW  test IC {raw_ic:.5f} IR {raw_ir:.3f}", flush=True)

    valid_ic_by_alpha = {}
    for a in ALPHA_GRID:
        d[f"_na_{a}"] = neutralize_g(d, "pred", a)
        sub = d[d.split == "valid"]
        valid_ic_by_alpha[a] = daily_ic(sub.assign(pred=sub[f"_na_{a}"]))[0]
    vmax = max(valid_ic_by_alpha.values())
    alpha = float(min(a for a in ALPHA_GRID if valid_ic_by_alpha[a] >= PLATEAU * vmax))
    print(f"[neut-{TAG}] valid IC max {vmax:.5f} @plateau -> selected alpha = {alpha}", flush=True)

    d["pred_neut"] = neutralize_g(d, "pred", alpha)
    te = d[d.split == "test"]
    ic, _, ir = daily_ic(te.assign(pred=te["pred_neut"]))
    va = d[d.split == "valid"]
    vic, _, vir = daily_ic(va.assign(pred=va["pred_neut"]))
    print(f"[neut-{TAG}] transformer_{TAG} PROCESSED (alpha={alpha})  "
          f"valid IC {vic:.5f} IR {vir:.3f} | test IC {ic:.5f} IR {ir:.3f}", flush=True)

    out = d[["day", "instrument_id", "split", "y"]].copy()
    out["pred"] = d["pred_neut"]
    out.to_parquet(PREDS / f"transformer_{TAG}_neut.parquet", index=False)
    pd.DataFrame([{"model": f"transformer_{TAG}_neut", "alpha": alpha,
                   "valid_IC": round(vic, 5), "valid_IR": round(vir, 3),
                   "test_IC": round(ic, 5), "test_IR": round(ir, 3)}]).to_csv(
        METR / f"neutralized_{TAG}.csv", index=False)
    print(f"[neut-{TAG}] wrote {PREDS/f'transformer_{TAG}_neut.parquet'}", flush=True)

    print(f"\n[neut-{TAG}] === comparison vs. existing baselines ===", flush=True)
    print(f"  transformer (raw, 213 feat baseline)        test IC 0.05906 IR 0.846")
    print(f"  v2_diverseblend_neutA0.62 (processed, 213)   test IC 0.05901 IR 1.101")
    print(f"  transformer_{TAG} (raw)         test IC {raw_ic:.5f} IR {raw_ir:.3f}")
    print(f"  transformer_{TAG} (processed)   test IC {ic:.5f} IR {ir:.3f}")


if __name__ == "__main__":
    main()
