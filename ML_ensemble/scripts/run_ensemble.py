"""EG ML ensemble: combine diverse base models into the final signal.

Metric = daily cross-sectional Pearson IC, so each model is per-day z-scored (an
affine transform that preserves its daily Pearson IC) before blending. The three
LightGBM variants are highly correlated, so they are first averaged into one
"lightgbm" family; the final signal is a robust EQUAL-weight blend of the diverse
families {lightgbm, mlp, transformer} (equal weights generalise across the
valid->test regime shift far better than valid-optimised weights). Ridge /
ElasticNet are reported as linear baselines but excluded from the final blend.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path("/root/autodl-tmp/eg_model/ML_single/scripts")))
from common import daily_ic, PREDS, METR, ROOT

LGB_VARIANTS = ["lightgbm_l1", "lightgbm_huber", "lightgbm_dart"]
LINEAR = ["ridge", "elasticnet"]
FINAL_FAMILIES = ["lightgbm", "mlp", "transformer"]   # robust equal-weight blend


def per_day_z(df, col):
    g = df.groupby("day")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main():
    files = {f.stem: f for f in PREDS.glob("*.parquet") if f.stem != "ensemble_final"}
    base = None
    for name, f in sorted(files.items()):
        d = pd.read_parquet(f).rename(columns={"pred": name})
        d[name] = per_day_z(d, name)
        cols = ["day", "instrument_id", name] + (["split", "y"] if base is None else [])
        base = d[cols] if base is None else base.merge(d[["day", "instrument_id", name]],
                                                       on=["day", "instrument_id"], how="inner")
    models = [m for m in files if m in base.columns]
    print(f"[ens] models: {models}  aligned rows: {len(base):,}", flush=True)

    # group LightGBM variants -> one family (mean of z-scored variants)
    present_lgb = [m for m in LGB_VARIANTS if m in base.columns]
    base["lightgbm"] = base[present_lgb].mean(axis=1)

    # per-model + family scores
    rows = []
    for m in models + ["lightgbm"]:
        base["pred"] = base[m]
        r = {"model": m}
        for sp in ("valid", "test"):
            ic, s, ir = daily_ic(base[base.split == sp])
            r[f"{sp}_IC"], r[f"{sp}_IR"] = round(ic, 5), round(ir, 3)
        rows.append(r)
    score = pd.DataFrame(rows).sort_values("test_IC")
    print(score.to_string(index=False))

    # correlation of the diverse families (test, per-day-z preds)
    fam = FINAL_FAMILIES
    te = base[base.split == "test"]
    corr = te[fam].corr()
    print("\n[ens] family corr (test):\n", corr.round(3).to_string())

    def blend_ic(cols, w, sub):
        p = (sub[cols].to_numpy() * np.asarray(w)).sum(1)
        return daily_ic(sub.assign(pred=p))

    res = []
    va, te = base[base.split == "valid"], base[base.split == "test"]
    # (a) robust equal-weight blend of the 3 diverse families
    w_eq = np.ones(len(fam)) / len(fam)
    # (b) valid-optimised (reported for contrast; tends to overfit the regime shift)
    w = w_eq.copy(); best = blend_ic(fam, w, va)[0]; rng = np.random.default_rng(0)
    for _ in range(3000):
        c = np.clip(w + rng.normal(0, 0.05, len(fam)), 0, None)
        if c.sum() == 0: continue
        c /= c.sum()
        ic = blend_ic(fam, c, va)[0]
        if ic > best: best, w = ic, c
    for name, weight in [("ensemble_equal", w_eq), ("ensemble_opt", w)]:
        row = {"model": name, "weights": {f: round(float(x), 3) for f, x in zip(fam, weight)}}
        for sp, sub in [("valid", va), ("test", te)]:
            ic, s, ir = blend_ic(fam, weight, sub)
            sm, _, _ = daily_ic(sub.assign(pred=(sub[fam].to_numpy() * weight).sum(1)), method="spearman")
            row[f"{sp}_IC"], row[f"{sp}_IR"], row[f"{sp}_spear"] = round(ic, 5), round(ir, 3), round(sm, 5)
        res.append(row)
        print(f"  {name:16s} valid IC {row['valid_IC']:.5f} IR {row['valid_IR']:.2f} | "
              f"test IC {row['test_IC']:.5f} IR {row['test_IR']:.2f} spear {row['test_spear']:.5f}  w={row['weights']}", flush=True)

    score.to_csv(METR / "ensemble_base_scores.csv", index=False)
    pd.DataFrame(res).to_csv(METR / "ensemble_leaderboard.csv", index=False)
    corr.to_csv(METR / "family_corr.csv")
    (ROOT / "ML_ensemble" / "models").mkdir(parents=True, exist_ok=True)
    json.dump({"families": fam, "w_equal": w_eq.tolist(), "w_opt": w.tolist(),
               "lgb_variants": present_lgb}, open(ROOT / "ML_ensemble" / "models" / "ensemble_weights.json", "w"), indent=2)
    # final signal = robust equal-weight blend
    base["pred"] = (base[fam].to_numpy() * w_eq).sum(1)
    base[["day", "instrument_id", "split", "y", "pred"]].to_parquet(PREDS / "ensemble_final.parquet", index=False)
    print("[ens] saved ensemble_final (equal-weight blend)", flush=True)


if __name__ == "__main__":
    main()
