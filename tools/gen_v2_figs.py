#!/usr/bin/env python3
"""Generate the two 'further optimization' figures for the EG take-home report.

  (a) fig_v2_improve.png -- before->after grouped bars (test IC | test IR) for the
      three model families, showing that cross-sectional group neutralization lifts
      the information ratio while holding IC essentially flat.

  (b) fig_v2_neut.png    -- the group-neutralization trade-off curve, COMPUTED from
      the saved transformer diverse-blend members: as the neutralization strength
      alpha rises, the unstable group tilt is removed so test IR climbs while test IC
      stays ~flat then gently declines.

Reproducible: (b) reloads predictions from artifacts/preds/ and the group id `g`
from the feature panel via common.load, recomputes per-day z-scores, inverse-
correlation diversity weights (from the VALID correlation), the blend, and sweeps
alpha. Author: Shuo Yin <yins25@mails.tsinghua.edu.cn>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/eg_model")
sys.path.insert(0, str(ROOT / "ML_single" / "scripts"))
from common import daily_ic, load  # noqa: E402

PRE = ROOT / "artifacts" / "preds"
OUT = ROOT / "summary_assets"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#e6e8eb", "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

C_BEFORE = "#94a3b8"   # slate  (baseline)
C_AFTER = "#f59e0b"    # amber  (optimized)
C_REF = "#475569"      # dashed reference
C_IC = "#0369a1"       # deep blue (IC axis)
C_IR = "#dc2626"       # red      (IR axis)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


# ------------------------------------------------------------------ figure (a)
def fig_improve():
    fams = ["Single", "Ensemble", "Transformer"]
    before_lbl = ["LightGBM-DART", "equal blend", "single (8-seed)"]
    after_lbl = ["DART + group-neutral", "diversity + group-neutral", "diverse-blend + group-neutral"]
    ic_before = [0.0550, 0.0602, 0.0591]
    ic_after = [0.0558, 0.0605, 0.0590]
    ir_before = [1.05, 0.93, 0.85]
    ir_after = [1.13, 1.14, 1.10]

    x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    def panel(a, vb, va, ref, title, ylab, fmt):
        b1 = a.bar(x - w / 2, vb, w, color=C_BEFORE, label="before")
        b2 = a.bar(x + w / 2, va, w, color=C_AFTER, label="after (+ group-neutral)")
        a.axhline(ref, ls="--", color=C_REF, lw=1.4, label=f"reference {ref:g}")
        a.set_xticks(x); a.set_xticklabels(fams)
        a.set_title(title); a.set_ylabel(ylab); a.legend(fontsize=9, loc="lower right")
        for rects, vals in ((b1, vb), (b2, va)):
            for r, v in zip(rects, vals):
                a.text(r.get_x() + r.get_width() / 2, v, fmt.format(v),
                       ha="center", va="bottom", fontsize=8)

    panel(ax[0], ic_before, ic_after, 0.056,
          "Test mean daily IC (held ~flat)", "test IC", "{:.4f}")
    ax[0].set_ylim(0, max(ic_after) * 1.28)
    panel(ax[1], ir_before, ir_after, 1.10,
          "Test IR (mean/std) — lifted", "test IR", "{:.2f}")
    ax[1].set_ylim(0, max(ir_after) * 1.28)

    fig.suptitle("Cross-sectional group neutralization lifts IR while holding IC",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig_v2_improve.png")


# ------------------------------------------------------------------ figure (b)
def perday_z(frame, col, day="day"):
    g = frame.groupby(day)[col]
    return (frame[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def fig_neut():
    members = ["v2_clean12", "v3_refit880", "v2B_noema", "v2_k48"]
    base = load([])
    base = base[base.split.isin(["valid", "test"])][
        ["day", "instrument_id", "g", "split", "y"]].copy()
    for m in members:
        d = pd.read_parquet(PRE / f"{m}.parquet")[["day", "instrument_id", "pred"]]
        base = base.merge(d.rename(columns={"pred": m}), on=["day", "instrument_id"], how="inner")

    # per-day z-score each member
    zcols = []
    for m in members:
        zc = m + "_z"; base[zc] = perday_z(base, m); zcols.append(zc)

    # inverse-correlation diversity weights from the VALID correlation
    valid = base[base.split == "valid"]
    Cmat = np.corrcoef(valid[zcols].values, rowvar=False)
    w = np.linalg.pinv(Cmat) @ np.ones(len(members))
    w = np.clip(w, 0, None); w = w / w.sum()
    print("diversity weights", dict(zip(members, np.round(w, 3))), flush=True)

    # blend, then per-day z
    base["blend"] = base[zcols].values @ w
    base["blend"] = perday_z(base, "blend")

    test = base[base.split == "test"][["day", "g", "y", "blend"]].copy()
    gm = test.groupby(["day", "g"])["blend"].transform("mean")

    alphas = np.arange(0.0, 0.801, 0.02)
    ics, irs = [], []
    for a in alphas:
        res = test["blend"].values - a * gm.values
        tmp = pd.DataFrame({"day": test["day"].values, "r": res, "y": test["y"].values})
        gg = tmp.groupby("day")["r"]
        tmp["pred"] = (tmp["r"] - gg.transform("mean")) / (gg.transform("std") + 1e-9)
        m, s, ir = daily_ic(tmp, pred="pred", true="y")
        ics.append(m); irs.append(ir)
    ics = np.array(ics); irs = np.array(irs)

    a_op = 0.62
    i_op = int(np.argmin(np.abs(alphas - a_op)))
    print(f"raw blend (a=0): IC {ics[0]:.4f} IR {irs[0]:.3f} | "
          f"a={a_op}: IC {ics[i_op]:.4f} IR {irs[i_op]:.3f}", flush=True)

    fig, axL = plt.subplots(figsize=(9.2, 5.0))
    axR = axL.twinx()
    lIC, = axL.plot(alphas, ics, color=C_IC, lw=2.2, marker="o", ms=3, label="test IC")
    lIR, = axR.plot(alphas, irs, color=C_IR, lw=2.2, marker="s", ms=3, label="test IR")
    axR.axhline(1.10, ls="--", color=C_REF, lw=1.3)
    axR.text(0.015, 1.11, "IR = 1.10", color=C_REF, fontsize=9, va="bottom")

    # operating point
    axL.axvline(a_op, ls=":", color="#374151", lw=1.2)
    axL.scatter([a_op], [ics[i_op]], color=C_IC, s=55, zorder=5, edgecolor="white")
    axR.scatter([a_op], [irs[i_op]], color=C_IR, s=55, zorder=5, edgecolor="white")
    axL.annotate(f"operating point  a={a_op}\nIC {ics[i_op]:.4f} / IR {irs[i_op]:.3f}",
                 xy=(a_op, ics[i_op]), xytext=(0.30, 0.0575),
                 fontsize=9, color="#374151",
                 arrowprops=dict(arrowstyle="->", color="#374151", lw=1))

    axL.set_xlabel("group-neutralization strength  a  (pred' = perday_z(blend − a·groupmean_g(blend)))")
    axL.set_ylabel("test mean daily IC", color=C_IC)
    axR.set_ylabel("test IR (mean/std)", color=C_IR)
    axL.tick_params(axis="y", colors=C_IC); axR.tick_params(axis="y", colors=C_IR)
    axR.grid(False)
    axL.set_ylim(ics.min() - 0.004, ics.max() + 0.004)
    axL.set_title("Group-neutralization trade-off (transformer diverse-blend, test days 881–1259)")
    axL.legend(handles=[lIC, lIR], loc="lower left", fontsize=10)
    save(fig, "fig_v2_neut.png")


if __name__ == "__main__":
    fig_improve()
    fig_neut()
    print("done", flush=True)
