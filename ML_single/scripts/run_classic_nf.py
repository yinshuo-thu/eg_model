"""Classic ML baselines on the NEW-FEATURES (nf) extended feature set (213 original +
surviving new prc/vol factors). Mirrors run_classic.py exactly, only swapping
common -> common_nf and output names -> *_nf, to isolate the lift from the new
factors alone (same models, same hyperparameters as the original baseline).
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common_nf import load, feature_cols, evaluate, save_pred, print_row, METR

import lightgbm as lgb
from sklearn.linear_model import Ridge, ElasticNet


def main():
    t0 = time.time()
    fcols = feature_cols()
    print(f"[classic-nf] loading features ({len(fcols)} cols)", flush=True)
    df = load(fcols)
    tr = df["split"] == "train"
    Xtr = df.loc[tr, fcols].to_numpy("float32")
    ytr = df.loc[tr, "y_xs"].to_numpy("float32")
    Xall = df[fcols].to_numpy("float32")
    rows = []

    print("[classic-nf] Ridge", flush=True)
    rg = Ridge(alpha=200.0)
    rg.fit(Xtr, ytr)
    df["pred"] = rg.predict(Xall)
    r = evaluate(df, "ridge_nf"); rows.append(r); print_row(r); save_pred(df, "ridge_nf")

    print("[classic-nf] ElasticNet", flush=True)
    en = ElasticNet(alpha=2e-3, l1_ratio=0.2, max_iter=3000)
    en.fit(Xtr, ytr)
    df["pred"] = en.predict(Xall)
    r = evaluate(df, "elasticnet_nf"); rows.append(r); print_row(r); save_pred(df, "elasticnet_nf")

    def lgb_train(params, nrounds, seeds=(0, 1, 2, 3, 4)):
        preds = np.zeros(len(df), dtype="float64")
        dtr = lgb.Dataset(Xtr, label=ytr)
        for sd in seeds:
            p = dict(params); p["seed"] = sd; p["bagging_seed"] = sd + 7; p["feature_fraction_seed"] = sd + 17
            m = lgb.train(p, dtr, num_boost_round=nrounds)
            preds += m.predict(Xall)
        return preds / len(seeds)

    base = dict(objective="regression_l1", num_leaves=31, learning_rate=0.02,
                feature_fraction=0.55, bagging_fraction=0.7, bagging_freq=1,
                min_data_in_leaf=400, lambda_l1=2.0, lambda_l2=5.0, max_depth=6,
                num_threads=8, verbosity=-1)
    print("[classic-nf] LightGBM L1 (bagged)", flush=True)
    df["pred"] = lgb_train(base, 600)
    r = evaluate(df, "lightgbm_l1_nf"); rows.append(r); print_row(r); save_pred(df, "lightgbm_l1_nf")

    print("[classic-nf] LightGBM Huber (bagged)", flush=True)
    hub = dict(base); hub["objective"] = "huber"; hub["alpha"] = 0.9
    df["pred"] = lgb_train(hub, 600)
    r = evaluate(df, "lightgbm_huber_nf"); rows.append(r); print_row(r); save_pred(df, "lightgbm_huber_nf")

    print("[classic-nf] LightGBM DART (bagged)", flush=True)
    dart = dict(base); dart["boosting"] = "dart"; dart["drop_rate"] = 0.1; dart["skip_drop"] = 0.5
    df["pred"] = lgb_train(dart, 500)
    r = evaluate(df, "lightgbm_dart_nf"); rows.append(r); print_row(r); save_pred(df, "lightgbm_dart_nf")

    lb = pd.DataFrame(rows)
    lb.to_csv(METR / "classic_nf_leaderboard.csv", index=False)
    print(f"[classic-nf] done in {time.time()-t0:.0f}s", flush=True)
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
