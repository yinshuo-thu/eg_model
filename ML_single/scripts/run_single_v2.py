"""Optimized SINGLE LightGBM models for EG (v2). Self-contained; imports only common.py.

Two routes to beat the gate (test IC > 0.0565 OR test IR > 1.10):
  * IC route  : a tuned, seed-bagged LightGBM (feature_fraction=0.4, num_leaves=63,
                large min_data, max_depth=8) with daily-IC early-stopping and
                per-day z-scored seed averaging -> higher cross-sectional IC.
  * IR route  : partial group-`g` NEUTRALISATION of a stable single model's daily
                predictions, pred' = zscore_day( pred - alpha * groupmean_g(pred) ),
                alpha selected ONLY on VALID daily IC -> removes the unstable group
                tilt, lifting IR well past 1.10 with IC roughly flat.

Target = y_xs (per-day z-score of y); eval = daily cross-sectional Pearson IC on raw y.
Train day<=760, valid 761-880, test 881-1259. VALID is used for all selection; TEST
is reported only. Writes NEW preds (preds/*_v2.parquet) and a NEW leaderboard
(ML_single/metrics/single_v2_leaderboard.csv). Does not touch existing files.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, feature_cols, evaluate, save_pred, print_row, ROOT
import lightgbm as lgb

V2_METR = Path(__file__).resolve().parent.parent / "metrics"
V2_METR.mkdir(parents=True, exist_ok=True)
NT = 14

# Best combo found by single-seed valid-daily-IC grid (see artifacts/notes/single_ineffective.md
# for the knobs that did NOT help). feature_fraction=0.4 was the single biggest lever.
BEST = dict(objective="regression_l1", num_leaves=63, learning_rate=0.02,
            feature_fraction=0.4, bagging_fraction=0.7, bagging_freq=1,
            min_data_in_leaf=600, lambda_l1=2.0, lambda_l2=5.0, max_depth=8,
            num_threads=NT, verbosity=-1)


def make_arrays():
    fcols = feature_cols()
    df = load(fcols)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "valid").to_numpy()
    X = df[fcols].to_numpy("float32")
    yxs = df["y_xs"].to_numpy("float32")
    day = df["day"].to_numpy()
    g = df["g"].to_numpy()
    Xtr, ytr = X[tr], yxs[tr]
    Xva, yva = X[va], yxs[va]
    day_va = day[va]
    order = np.argsort(day_va, kind="stable")
    _, starts = np.unique(day_va[order], return_index=True)
    va_groups = np.split(order, starts[1:])
    order_a = np.argsort(day, kind="stable")
    _, starts_a = np.unique(day[order_a], return_index=True)
    all_groups = np.split(order_a, starts_a[1:])
    return dict(df=df, fcols=fcols, X=X, yxs=yxs, day=day, g=g, tr=tr, va=va,
                Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, day_va=day_va,
                va_groups=va_groups, all_groups=all_groups)


def feval_ic(D):
    yva, groups = D["yva"], D["va_groups"]
    def f(preds, dataset):
        ics = [np.corrcoef(preds[g], yva[g])[0, 1] for g in groups
               if preds[g].std() > 1e-12 and yva[g].std() > 1e-12]
        return "dic", float(np.mean(ics)), True
    return f


def zscore_perday(pred, groups):
    out = np.empty_like(pred, dtype="float64")
    for g in groups:
        p = pred[g]; s = p.std()
        out[g] = (p - p.mean()) / s if s > 1e-12 else 0.0
    return out


def bag(D, params, seeds, fixed_rounds=None, ranker=False, label=None, group=None):
    Xtr, ytr, Xva, yva, X = D["Xtr"], D["ytr"], D["Xva"], D["yva"], D["X"]
    agg = np.zeros(len(X), dtype="float64")
    for sd in seeds:
        p = dict(params); p.update(seed=sd, bagging_seed=sd + 7,
                                   feature_fraction_seed=sd + 17, data_random_seed=sd + 23)
        if ranker:
            dtr = lgb.Dataset(Xtr, label=label, group=group)
            m = lgb.train(p, dtr, num_boost_round=fixed_rounds)
        elif fixed_rounds is not None:
            dtr = lgb.Dataset(Xtr, label=ytr)
            m = lgb.train(p, dtr, num_boost_round=fixed_rounds)
        else:
            dtr = lgb.Dataset(Xtr, label=ytr)
            dva = lgb.Dataset(Xva, label=yva, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                          feval=feval_ic(D), callbacks=[lgb.early_stopping(150, verbose=False)])
        agg += zscore_perday(m.predict(X), D["all_groups"])
    return agg / len(seeds)


def scan_rounds(D, params, max_rounds, label=None, group=None):
    """Single-seed round scan picking the round with max VALID daily IC.
    If label/group given, trains a ranker (needs group)."""
    rec = {}
    if group is not None:
        dtr = lgb.Dataset(D["Xtr"], label=label, group=group)
    else:
        dtr = lgb.Dataset(D["Xtr"], label=D["ytr"])
    dva = lgb.Dataset(D["Xva"], label=D["yva"], reference=dtr)
    p = dict(params); p["seed"] = 0
    lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
              feval=feval_ic(D), callbacks=[lgb.record_evaluation(rec)])
    dic = np.array(rec["valid_0"]["dic"])
    return int(np.argmax(dic) + 1), float(dic.max())


def neutralize(D, pred, select="ic"):
    """pred' = zscore_day(pred - alpha*groupmean_g(pred)); alpha by VALID daily IC."""
    df = D["df"]
    tmp = pd.DataFrame({"day": D["day"], "g": D["g"], "p": pred, "split": df["split"].to_numpy(),
                        "y": df["y"].to_numpy()})
    gm = tmp.groupby(["day", "g"])["p"].transform("mean").to_numpy()
    best_a, best_key, best_z = 0.0, -9, None
    for a in np.round(np.arange(0, 1.001, 0.05), 2):
        resid = pred - a * gm
        z = zscore_perday(resid, D["all_groups"])
        tmp["_z"] = z
        e = evaluate(tmp, "tmp", pred="_z")
        key = e["valid_IC"] if select == "ic" else e["valid_IR"]
        if key > best_key:
            best_key, best_a, best_z = key, float(a), z
    return best_z, best_a


def main():
    t0 = time.time()
    D = make_arrays()
    print(f"[v2] data ready, {len(D['fcols'])} feats", flush=True)
    df = D["df"]
    rows = []
    seeds = tuple(range(12))

    def emit(pred, name, save=True):
        df["pred"] = pred
        r = evaluate(df, name); rows.append(r); print_row(r)
        if save:
            save_pred(df, name)
        return r

    # ---- IC route: tuned bagged regression_l1 ----
    print("[v2] tuned bagged LightGBM (lgb_tuned_v2)", flush=True)
    emit(bag(D, BEST, seeds), "lgb_tuned_v2")

    # ---- tuned DART (stability) ----
    print("[v2] tuned DART (lgb_dart_tuned_v2)", flush=True)
    dart = dict(BEST); dart.update(boosting="dart", drop_rate=0.1, skip_drop=0.5)
    dr, _ = scan_rounds(D, dart, 1000)
    print(f"    dart rounds={dr}", flush=True)
    emit(bag(D, dart, seeds, fixed_rounds=dr), "lgb_dart_tuned_v2")

    # ---- rank_xendcg ----
    print("[v2] rank_xendcg (lgb_rank_v2)", flush=True)
    day_tr = D["day"][D["tr"]]
    _, counts = np.unique(day_tr, return_counts=True)
    group = counts.tolist()
    NB = 32; lab = np.zeros(len(D["ytr"]), dtype="int32"); idx = 0
    for c in counts:
        seg = D["ytr"][idx:idx + c]; ranks = seg.argsort().argsort()
        lab[idx:idx + c] = (ranks * NB // c).astype("int32"); idx += c
    rp = dict(BEST); rp.pop("objective")
    rp = dict(objective="rank_xendcg", label_gain=list(range(NB)),
              num_leaves=BEST["num_leaves"], learning_rate=BEST["learning_rate"],
              feature_fraction=BEST["feature_fraction"], bagging_fraction=BEST["bagging_fraction"],
              bagging_freq=1, min_data_in_leaf=BEST["min_data_in_leaf"], lambda_l1=BEST["lambda_l1"],
              lambda_l2=BEST["lambda_l2"], max_depth=BEST["max_depth"], num_threads=NT, verbosity=-1)
    rb, _ = scan_rounds(D, rp, 800, label=lab, group=group)
    print(f"    rank rounds={rb}", flush=True)
    emit(bag(D, rp, seeds, fixed_rounds=rb, ranker=True, label=lab, group=group), "lgb_rank_v2")

    # ---- IR route: neutralise the best/most-stable models ----
    for src in ("lgb_tuned_v2", "lgb_dart_tuned_v2"):
        sub = next(r for r in rows if r["model"] == src)
        base_pred = pd.read_parquet(ROOT / "artifacts" / "preds" / f"{src}.parquet")
        # rebuild full-length pred aligned to df for neutralisation
        full = df.merge(base_pred[["day", "instrument_id", "pred"]], on=["day", "instrument_id"], how="left")["pred"].to_numpy()
        # train preds are NaN in saved file; neutralise only needs valid+test which are present
        full = np.where(np.isnan(full), 0.0, full)
        z, a = neutralize(D, full, select="ic")
        print(f"    {src} neutral alpha*={a}", flush=True)
        emit(z, f"{src}_neutral")

    lb = pd.DataFrame(rows)
    lb.to_csv(V2_METR / "single_v2_leaderboard.csv", index=False)
    print(lb.to_string(index=False))
    print(f"[v2] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
