#!/usr/bin/env python3
"""Figures for the "New Features" report section: (a) headline IC/IR comparison
across original vs full-111-new-factor vs curated-top-20-new-factor ensembles,
(b) the per-factor 3-round optimization ladder for the strongest new ideas.
Numbers are read live from artifacts/preds/*.parquet and
artifacts/notes/new_factors_ic.json -- nothing here is hand-typed.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/eg_model")
PRE = ROOT / "artifacts" / "preds"
OUT = ROOT / "summary_assets"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#e6e8eb", "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C_ORIG, C_FULL, C_TOP20 = "#94a3b8", "#f87171", "#0ea5e9"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name, flush=True)


def daily_ic(df, sp):
    d = df.loc[df.split == sp, ["day", "pred", "y"]].dropna()
    if d.empty:
        return float("nan"), float("nan")
    ic = d.groupby("day").apply(lambda g: np.corrcoef(g["pred"].to_numpy(), g["y"].to_numpy())[0, 1])
    m, s = float(ic.mean()), float(ic.std())
    return m, (m / s if s > 0 else float("nan"))


def score(name):
    f = PRE / f"{name}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    return {"test": daily_ic(d, "test"), "valid": daily_ic(d, "valid")}


# ------------------------------------------------------------------ fig (a)
BASE_IC, BASE_IR = 0.056, 1.10   # canonical y_hat0 stats from the brief (no row-level preds given)


def fig_compare():
    rows = [
        ("ensemble_final", "original\n213 feat"),
        ("ensemble_nf", "+111 new\nfactors (naive)"), ("ensemble_nf20", "+20 curated\nnew factors"),
    ]
    labels, ic_test, ir_test = [], [], []
    for stem, lab in rows:
        sc = score(stem)
        if sc is None:
            continue
        labels.append(lab); ic_test.append(sc["test"][0]); ir_test.append(sc["test"][1])

    x = np.arange(len(labels))
    colors = [C_ORIG, C_FULL, C_TOP20][:len(labels)]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))

    b1 = ax[0].bar(x, ic_test, color=colors, width=0.55)
    ax[0].axhline(BASE_IC, ls="--", color="#475569", lw=1.2, label=f"y_hat0 baseline {BASE_IC:g}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=10)
    ax[0].set_title("Test mean daily Pearson IC"); ax[0].set_ylim(0, max(ic_test) * 1.25)
    ax[0].legend(fontsize=8.5, loc="lower right")
    for r, v in zip(b1, ic_test):
        ax[0].text(r.get_x() + r.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    b2 = ax[1].bar(x, ir_test, color=colors, width=0.55)
    ax[1].axhline(BASE_IR, ls="--", color="#475569", lw=1.2, label=f"y_hat0 baseline {BASE_IR:g}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=10)
    ax[1].set_title("Test IR (mean/std)"); ax[1].set_ylim(0, max(ir_test) * 1.25)
    ax[1].legend(fontsize=8.5, loc="lower right")
    for r, v in zip(b2, ir_test):
        ax[1].text(r.get_x() + r.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("New prc/vol factors: naive dump dilutes, IC-curated top-20 recovers it",
                 fontsize=13.5, fontweight="bold", y=1.03)
    save(fig, "fig_nf_compare.png")


# ------------------------------------------------------------------ fig (b)
def fig_ladder():
    d = json.loads((ROOT / "artifacts" / "notes" / "new_factors_ic.json").read_text())
    kept = set(d["kept_ideas"])
    ledger = d["ledger"]
    rows = [r for r in ledger if r["idea"] in kept]
    rows.sort(key=lambda r: -abs(r["versions"][-1]["train_ic"]))
    top = rows[:18]
    names = [f"{r['idea']}" + (f" ({r['variant']})" if r["variant"] else "") for r in top][::-1]
    v1 = [r["versions"][0]["train_ic"] for r in top][::-1]
    v3 = [r["versions"][-1]["train_ic"] for r in top][::-1]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.barh(y - 0.19, v1, height=0.36, color="#cbd5e1", label="v1 raw (cs z-score)")
    ax.barh(y + 0.19, v3, height=0.36, color="#0369a1", label="v3 optimized (final)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#475569", lw=0.9)
    ax.set_xlabel("train mean daily Pearson IC")
    ax.set_title("Top-18 new factors: v1 -> v3 optimization ladder (train IC)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig_nf_ladder.png")


if __name__ == "__main__":
    fig_compare()
    fig_ladder()
    print("done", flush=True)
