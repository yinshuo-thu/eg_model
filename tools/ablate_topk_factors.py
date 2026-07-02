"""Quick ablation: does capping the new-factor block to its top-K by |train IC|
(instead of dumping all 111) reduce noise dilution for a fixed-hyperparameter
LightGBM? One model (Huber, bagged) for a fast directional read.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("/root/autodl-tmp/eg_model/ML_single/scripts")))
from common_nf import load, daily_ic
import lightgbm as lgb

ROOT = Path("/root/autodl-tmp/eg_model")
base_cols = json.loads((ROOT / "artifacts" / "feature_list.json").read_text())
topk = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else []
K = int(sys.argv[2]) if len(sys.argv) > 2 else len(topk)
fcols = base_cols + topk[:K]
print(f"[ablate] {len(base_cols)} base + {len(topk[:K])} new = {len(fcols)} total", flush=True)

t0 = time.time()
df = load(fcols)
tr = df["split"] == "train"
Xtr = df.loc[tr, fcols].to_numpy("float32")
ytr = df.loc[tr, "y_xs"].to_numpy("float32")
Xall = df[fcols].to_numpy("float32")

base = dict(objective="huber", alpha=0.9, num_leaves=31, learning_rate=0.02,
            feature_fraction=0.55, bagging_fraction=0.7, bagging_freq=1,
            min_data_in_leaf=400, lambda_l1=2.0, lambda_l2=5.0, max_depth=6,
            num_threads=8, verbosity=-1)
preds = np.zeros(len(df), dtype="float64")
for sd in (0, 1, 2, 3, 4):
    p = dict(base); p["seed"] = sd; p["bagging_seed"] = sd + 7; p["feature_fraction_seed"] = sd + 17
    dtr = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train(p, dtr, num_boost_round=600)
    preds += m.predict(Xall)
df["pred"] = preds / 5
for sp in ("valid", "test"):
    sub = df[df["split"] == sp]
    ic, s, ir = daily_ic(sub)
    print(f"  {sp}: IC {ic:.5f} IR {ir:.3f}", flush=True)
print(f"[ablate] done in {time.time()-t0:.0f}s", flush=True)
