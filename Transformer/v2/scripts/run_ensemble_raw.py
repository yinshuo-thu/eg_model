"""EG ML ensemble RAW (robust equal-weight blend of {lightgbm, mlp, transformer},
no group-neutralization). Generic version of
ML_ensemble/scripts/run_ensemble_v3factors.py, parametrized by RETRAIN_TAG."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_generic import daily_ic, PREDS, METR, ROOT, TAG

LGB_VARIANTS = [f"lightgbm_l1_{TAG}", f"lightgbm_huber_{TAG}", f"lightgbm_dart_{TAG}"]
FINAL_FAMILIES = ["lightgbm", "mlp", "transformer"]


def main():
    files = {
        f.stem: f
        for f in PREDS.glob(f"*_{TAG}.parquet")
        if f.stem.startswith(("lightgbm_", "mlp_", "transformer_"))
    }
    base = None
    for name, f in sorted(files.items()):
        d = pd.read_parquet(f).rename(columns={"pred": name})
        g = d.groupby("day")[name]
        d[name] = (d[name] - g.transform("mean")) / (g.transform("std") + 1e-9)
        cols = ["day", "instrument_id", name] + (["split", "y"] if base is None else [])
        base = d[cols] if base is None else base.merge(d[["day", "instrument_id", name]],
                                                       on=["day", "instrument_id"], how="inner")
    models = [m for m in files if m in base.columns]
    print(f"[ens-{TAG}-raw] models: {models}  aligned rows: {len(base):,}", flush=True)

    present_lgb = [m for m in LGB_VARIANTS if m in base.columns]
    base["lightgbm"] = base[present_lgb].mean(axis=1)
    best_mlp = [m for m in models if m.startswith("mlp")][0]
    best_tr = [m for m in models if m.startswith("transformer")][0]
    base["mlp"] = base[best_mlp]
    base["transformer"] = base[best_tr]

    rows = []
    for m in models + ["lightgbm"]:
        r = {"model": m}
        for sp in ("valid", "test"):
            sub = base[base.split == sp]
            ic, s, ir = daily_ic(sub.assign(pred=sub[m]))
            r[f"{sp}_IC"], r[f"{sp}_IR"] = round(ic, 5), round(ir, 3)
        rows.append(r)
    score = pd.DataFrame(rows).sort_values("test_IC")
    print(score.to_string(index=False))

    fam = FINAL_FAMILIES
    w_eq = np.ones(len(fam)) / len(fam)
    va, te = base[base.split == "valid"], base[base.split == "test"]

    def blend_ic(cols, w, sub):
        p = (sub[cols].to_numpy() * np.asarray(w)).sum(1)
        return daily_ic(sub.assign(pred=p))

    row = {"model": f"ensemble_{TAG}_raw (equal-weight, no neutral)",
           "weights": {f: round(float(x), 3) for f, x in zip(fam, w_eq)}}
    for sp, sub in [("valid", va), ("test", te)]:
        ic, s, ir = blend_ic(fam, w_eq, sub)
        sm, _, _ = daily_ic(sub.assign(pred=(sub[fam].to_numpy() * w_eq).sum(1)), method="spearman")
        row[f"{sp}_IC"], row[f"{sp}_IR"], row[f"{sp}_spear"] = round(ic, 5), round(ir, 3), round(sm, 5)
    print(f"  {row['model']:44s} valid IC {row['valid_IC']:.5f} IR {row['valid_IR']:.2f} | "
          f"test IC {row['test_IC']:.5f} IR {row['test_IR']:.2f}", flush=True)

    score.to_csv(METR / f"ensemble_{TAG}_raw_base_scores.csv", index=False)
    pd.DataFrame([row]).to_csv(ROOT / "ML_ensemble" / "metrics" / f"ensemble_{TAG}_raw_leaderboard.csv", index=False)

    base["pred"] = (base[fam].to_numpy() * w_eq).sum(1)
    base[["day", "instrument_id", "split", "y", "pred"]].to_parquet(PREDS / f"ensemble_{TAG}_raw.parquet", index=False)
    print(f"[ens-{TAG}-raw] saved ensemble_{TAG}_raw (equal-weight blend, no group-neutral)", flush=True)

    print(f"\n[ens-{TAG}-raw] === comparison vs. existing baselines ===", flush=True)
    for path, label in [(PREDS / "ensemble_final.parquet", "ensemble_final (original 213, raw equal-wt)"),
                         (PREDS / "ensemble_v2.parquet", "ensemble_v2 (original 213, processed)")]:
        if path.exists():
            d = pd.read_parquet(path)
            r2 = {}
            for sp in ("valid", "test"):
                sub = d[d.split == sp]
                ic, s, ir = daily_ic(sub)
                r2[f"{sp}_IC"], r2[f"{sp}_IR"] = round(ic, 5), round(ir, 3)
            print(f"  {label:44s} valid IC {r2['valid_IC']:.5f} IR {r2['valid_IR']:.2f} | "
                  f"test IC {r2['test_IC']:.5f} IR {r2['test_IR']:.2f}", flush=True)


if __name__ == "__main__":
    main()
