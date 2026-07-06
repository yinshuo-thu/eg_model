#!/usr/bin/env python3
"""Generate visualization assets for the EG (Engineering Gates) take-home report.

Reads the saved model predictions in artifacts/preds/ and produces PNG figures in
summary_assets/. English labels (report HTML is bilingual; avoids CJK font deps).
Style matches the cn-future-alpha report.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#e6e8eb", "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
import os
ROOT = Path("/root/autodl-tmp/eg_model")
PRE = Path(os.environ.get("PREDS_DIR", ROOT / "artifacts" / "preds"))
OUT = ROOT / "summary_assets"; OUT.mkdir(exist_ok=True)
BASE_IC, BASE_IR = 0.056, 1.1
C = {"ridge": "#94a3b8", "elasticnet": "#cbd5e1", "lightgbm_l1": "#38bdf8",
     "lightgbm_huber": "#0ea5e9", "lightgbm_dart": "#0369a1", "mlp": "#34d399",
     "transformer": "#a78bfa", "ensemble_opt": "#f59e0b", "ensemble_equal": "#fbbf24",
     "y_hat0": "#64748b", "thresh": "#475569", "ink": "#1f2937"}
NAMES = {"ridge": "Ridge", "elasticnet": "ElasticNet", "lightgbm_l1": "LightGBM-L1",
         "lightgbm_huber": "LightGBM-Huber", "lightgbm_dart": "LightGBM-DART",
         "mlp": "MLP", "transformer": "Transformer", "ensemble_opt": "Ensemble",
         "ensemble_equal": "Ensemble (eq)", "y_hat0": "y_hat0 (baseline)"}
ORDER = ["ridge", "elasticnet", "lightgbm_dart", "lightgbm_huber", "lightgbm_l1",
         "mlp", "ensemble_opt", "transformer"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name, flush=True)


def daily_ic_series(df, sp="test", method="pearson"):
    d = df[df.split == sp].dropna(subset=["pred", "y"])
    return d.groupby("day").apply(lambda g: g.pred.corr(g.y, method=method))


def load_all():
    out = {}
    for f in sorted(PRE.glob("*.parquet")):
        if f.stem == "ensemble_final": continue
        out[f.stem] = pd.read_parquet(f)
    if (PRE / "ensemble_final.parquet").exists():
        out["ensemble_opt"] = pd.read_parquet(PRE / "ensemble_final.parquet")
    return out


def main():
    M = load_all()
    models = [m for m in ORDER if m in M]
    stats = {}
    for m in models:
        ic = daily_ic_series(M[m], "test")
        stats[m] = (ic.mean(), ic.mean() / ic.std() if ic.std() > 0 else 0, ic)
    # also equal blend if present in metrics
    best_single = max([m for m in models if not m.startswith("ensemble")], key=lambda m: stats[m][0])

    # ---- fig1: headline IC + IR bars. The Ensemble and Transformer bars show the
    #      FURTHER-PROCESSED (v2) models (diversity weighting + group-g neutralization),
    #      so the IR panel reflects the post-processing; the ML bars stay raw baselines. ----
    F1 = ["ridge", "elasticnet", "lightgbm_dart", "lightgbm_huber", "lightgbm_l1",
          "mlp", "ensemble_v2", "transformer_v2"]
    F1 = [m for m in F1 if m in M]
    f1nm = {**NAMES, "ensemble_v2": "Ensemble*", "transformer_v2": "Transformer*"}
    f1col = {**C, "ensemble_v2": C["ensemble_opt"], "transformer_v2": C["transformer"]}
    f1st = {}
    for m in F1:
        ic = daily_ic_series(M[m], "test")
        f1st[m] = (ic.mean(), ic.mean() / ic.std() if ic.std() > 0 else 0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = range(len(F1)); cols = [f1col.get(m, "#888") for m in F1]
    ax[0].bar(xs, [f1st[m][0] for m in F1], color=cols)
    ax[0].axhline(BASE_IC, ls="--", color=C["thresh"], lw=1.4, label=f"baseline IC ({BASE_IC})")
    ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([f1nm[m] for m in F1], rotation=35, ha="right", fontsize=9)
    ax[0].set_title("Test mean daily IC (days 881–1259)"); ax[0].legend(fontsize=9)
    for i, m in enumerate(F1): ax[0].text(i, f1st[m][0] + 0.0005, f"{f1st[m][0]:.4f}", ha="center", fontsize=7.5)
    ax[1].bar(xs, [f1st[m][1] for m in F1], color=cols)
    ax[1].axhline(BASE_IR, ls="--", color=C["thresh"], lw=1.4, label=f"baseline IR ({BASE_IR})")
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([f1nm[m] for m in F1], rotation=35, ha="right", fontsize=9)
    ax[1].set_title("Test IC information ratio (mean/std)"); ax[1].legend(fontsize=9)
    for i, m in enumerate(F1): ax[1].text(i, f1st[m][1] + 0.012, f"{f1st[m][1]:.2f}", ha="center", fontsize=7.5)
    fig.text(0.5, -0.02, "* Ensemble / Transformer = further-processed final models (diversity weighting + group-g neutralization); the ML bars are raw baselines.",
             ha="center", fontsize=7.4, color="#6b7280")
    save(fig, "fig1_headline.png")

    # ---- fig2: daily IC time series for ensemble ----
    ens = "ensemble_opt" if "ensemble_opt" in stats else best_single
    have_yh = "y_hat0" in M

    def test_df(name):
        return M[name][M[name].split == "test"].dropna(subset=["pred", "y"]).copy()

    def dic_series(name):
        d = test_df(name)
        return d.groupby("day").apply(lambda g: g.pred.corr(g.y))

    # ---- fig2: stacked daily IC over the test period — Ensemble / y_hat0 / Transformer ----
    stack = [(ens, C["ensemble_opt"]), ("transformer", C["transformer"])]
    stack = [(m, c) for m, c in stack if m in M]
    fig, axes = plt.subplots(len(stack), 1, figsize=(11, 2.5 * len(stack)), sharex=True)
    if len(stack) == 1: axes = [axes]
    ymax = 0
    series = {m: dic_series(m) for m, _ in stack}
    ymax = max(abs(s).quantile(0.98) for s in series.values())
    for axi, (m, col) in zip(axes, stack):
        s = series[m]
        axi.plot(s.index, s.values, color="#cbd5e1", lw=0.6)
        axi.plot(s.index, s.rolling(20).mean(), color=col, lw=2.0, label="20-day rolling")
        axi.axhline(s.mean(), color=C["ink"], ls=":", lw=1.1)
        axi.axhline(0, color="#9aa0a6", lw=0.7); axi.axhline(BASE_IC, ls="--", color=C["thresh"], lw=1.0)
        axi.set_ylim(-ymax, ymax); axi.set_ylabel("daily IC")
        axi.set_title(f"{NAMES[m]} — mean daily IC {s.mean():.4f}, IR {s.mean()/s.std():.2f}", fontsize=11, loc="left")
        axi.legend(fontsize=8.5, loc="upper right")
    axes[-1].set_xlabel("day")
    save(fig, "fig2_daily_ic.png")

    # ---- fig3: quintile (5-bucket) returns + long-short PnL — y_hat0 / Transformer / Ensemble ----
    pnl_models = [(m, C.get(m, "#888")) for m in ["transformer", ens] if m in M]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / len(pnl_models)
    for i, (m, col) in enumerate(pnl_models):
        d = test_df(m)
        d["q"] = d.groupby("day")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        qm = d.groupby("q")["y"].mean()
        ax[0].bar(np.arange(5) + (i - (len(pnl_models) - 1) / 2) * width, qm.values, width, color=col, label=NAMES[m])
        # long-short daily return (Q5 - Q1), cumulated
        daily = d.groupby(["day", "q"])["y"].mean().unstack("q")
        ls = (daily[4] - daily[0]).sort_index()
        ax[1].plot(ls.index, ls.cumsum().values, lw=2.0, color=col, label=f"{NAMES[m]} (Σ={ls.sum():.2f})")
    ax[0].axhline(0, color="#9aa0a6", lw=0.7); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax[0].set_title("Quintile mean realized y (test)"); ax[0].set_xlabel("prediction quintile (low → high)"); ax[0].set_ylabel("mean y"); ax[0].legend(fontsize=9)
    ax[1].axhline(0, color="#9aa0a6", lw=0.7); ax[1].set_title("Long–short PnL (top quintile − bottom quintile, cumulated)")
    ax[1].set_xlabel("day"); ax[1].set_ylabel("cumulative Q5−Q1 return"); ax[1].legend(fontsize=9)
    save(fig, "fig3_pnl.png")

    # ---- fig5: model prediction correlation heatmap (decorrelation) ----
    base_models = [m for m in models if not m.startswith("ensemble")]
    piv = None
    for m in base_models:
        t = M[m][M[m].split == "test"][["day", "instrument_id", "pred"]].copy()
        t["r"] = t.groupby("day")["pred"].rank(pct=True)
        t = t.set_index(["day", "instrument_id"])["r"].rename(m)
        piv = t.to_frame() if piv is None else piv.join(t, how="inner")
    corr = piv.corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr.values, vmin=0.2, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(base_models))); ax.set_xticklabels([NAMES[m] for m in base_models], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(base_models))); ax.set_yticklabels([NAMES[m] for m in base_models], fontsize=8)
    for i in range(len(base_models)):
        for j in range(len(base_models)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    color="white" if corr.values[i, j] < 0.8 else "black", fontsize=7)
    ax.set_title("Base-model prediction correlation (test, per-day rank)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig5_modelcorr.png")

    # ---- fig6: ensemble lift over best single ----
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = [NAMES[best_single], "Ensemble (eq)", "Ensemble (opt)"]
    vals = [stats[best_single][0],
            stats.get("ensemble_equal", stats[ens])[0] if "ensemble_equal" in stats else stats[ens][0],
            stats[ens][0]]
    cols2 = [C.get(best_single, "#888"), C["ensemble_equal"], C["ensemble_opt"]]
    ax.bar(labels, vals, color=cols2)
    ax.axhline(BASE_IC, ls="--", color=C["thresh"], lw=1.4, label=f"baseline {BASE_IC}")
    for i, v in enumerate(vals): ax.text(i, v + 0.0004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_title("Ensemble lift over best single model (test IC)"); ax.legend(fontsize=9)
    save(fig, "fig6_lift.png")

    # ---- comparison figures (multi-model blocked IC + 20-bucket) ----
    def get_df(name):
        if name == "lightgbm":
            parts, base = [], None
            for v in ["lightgbm_l1", "lightgbm_huber", "lightgbm_dart"]:
                d = M[v][M[v].split == "test"][["day", "instrument_id", "y", "pred"]].copy()
                g = d.groupby("day")["pred"]
                d["z"] = (d["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)
                parts.append(d.set_index(["day", "instrument_id"])["z"].rename(v))
                if base is None:
                    base = d.set_index(["day", "instrument_id"])[["y"]]
            base["pred"] = pd.concat(parts, axis=1).mean(axis=1)
            return base.reset_index()
        return M[name][M[name].split == "test"][["day", "instrument_id", "y", "pred"]].copy()

    def blocked_ic(d, nb=13):
        days = np.sort(d.day.unique()); xs, ys = [], []
        for blk in np.array_split(days, nb):
            sub = d[d.day.isin(blk)].dropna(subset=["pred", "y"])
            ys.append(sub.groupby("day").apply(lambda g: g.pred.corr(g.y)).mean()); xs.append(int(np.mean(blk)))
        return xs, ys

    def bucket_curve(d, nb=20):
        dd = d.dropna(subset=["pred", "y"]).copy()
        dd["bk"] = dd.groupby("day")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), nb, labels=False))
        return dd.groupby("bk")["y"].mean()

    def compare(specs, fname, tic, tbk):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for label, color, name in specs:
            d = get_df(name); x, y = blocked_ic(d)
            ax[0].plot(x, y, marker="o", ms=3, lw=1.8, color=color, label=f"{label} (mean {np.nanmean(y):.4f})")
            bc = bucket_curve(d); ax[1].plot(bc.index, bc.values, marker="o", ms=3, lw=1.8, color=color, label=label)
        ax[0].axhline(BASE_IC, ls="--", color=C["thresh"], lw=1.2, label=f"baseline {BASE_IC}"); ax[0].axhline(0, color="#9aa0a6", lw=0.7)
        ax[0].set_title(tic); ax[0].set_xlabel("day (block mid)"); ax[0].set_ylabel("mean daily IC"); ax[0].legend(fontsize=8.5)
        ax[1].axhline(0, color="#9aa0a6", lw=0.7); ax[1].set_title(tbk); ax[1].set_xlabel("prediction bucket (low → high)"); ax[1].set_ylabel("mean y"); ax[1].legend(fontsize=8.5)
        save(fig, fname)

    compare([("Ridge", C["ridge"], "ridge"), ("LightGBM", C["lightgbm_l1"], "lightgbm"), ("MLP", C["mlp"], "mlp")],
            "fig7_ml_compare.png", "Blocked daily IC (test)", "20-bucket monotonicity (test)")
    if "ensemble_opt" in M:
        final_specs = [("LightGBM", C["lightgbm_l1"], "lightgbm"), ("Transformer", C["transformer"], "transformer"), ("Ensemble", C["ensemble_opt"], "ensemble_opt")]
        compare(final_specs,
                "fig8_final_compare.png", "Blocked daily IC (test)", "20-bucket monotonicity (test)")

    json.dump({m: {"test_IC": round(stats[m][0], 5), "test_IR": round(stats[m][1], 3)} for m in models},
              open(OUT / "fig_stats.json", "w"), indent=2)
    print("baseline", BASE_IC, "| best single", best_single, round(stats[best_single][0], 5),
          "| ensemble", round(stats[ens][0], 5))


if __name__ == "__main__":
    main()
