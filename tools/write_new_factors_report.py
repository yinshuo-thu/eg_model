"""Render common_docs/new_factors_report.md from artifacts/notes/new_factors_ic.json
(the IC ledger written by tools/build_new_factors.py). Auto-generates the per-idea
formula/IC tables; the narrative framing around them is hand-written in this script.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/eg_model")
LEDGER = json.loads((ROOT / "artifacts" / "notes" / "new_factors_ic.json").read_text())
OUT = ROOT / "common_docs" / "new_factors_report.md"

CAT_LABEL = {
    "reversal": "反转/动量 Reversal & Momentum",
    "vol_liquidity": "波动率/非流动性 Volatility & Illiquidity",
    "volume": "成交量/换手 Volume & Turnover",
    "technical": "技术/贝塔/趋势 Technical, Beta & Trend",
    "interaction": "交互/条件化 Interaction & Conditioning",
}

ledger = LEDGER["ledger"]
kept = set(LEDGER["kept_ideas"])
dropped = {d["idea"]: d for d in LEDGER["dropped"]}

# group ledger rows by idea
by_idea: dict[str, list[dict]] = {}
for row in ledger:
    by_idea.setdefault(row["idea"], []).append(row)

# best v1->v3 IC gain per idea (for the "3x+ showcase" section). Require the same
# sign v1->v3 and a non-trivial v1 floor, so a near-zero-noise v1 flipping sign
# under processing doesn't masquerade as a "30x improvement".
MIN_V1 = 5e-4
gains = []
for idea, rows in by_idea.items():
    if idea not in kept:
        continue
    for r in rows:
        v1, v3 = r["versions"][0]["train_ic"], r["versions"][-1]["train_ic"]
        if abs(v1) >= MIN_V1 and v1 * v3 > 0:
            gains.append((idea, r["variant"], v1, v3, v3 / v1))
gains.sort(key=lambda t: -abs(t[4]) if t[4] == t[4] else 0)

lines = []
lines.append("# 新因子库：100+ 思路的构造、三轮优化与相关性剪枝")
lines.append("# New Factor Library: 100+ Ideas, a 3-Round Optimization Ladder, and Correlation Pruning")
lines.append("")
lines.append(f"> 由 `tools/build_new_factors.py` 自动生成本文档的表格部分（`tools/write_new_factors_report.py` 渲染）。")
lines.append(f"> 输入：`artifacts/panel_raw.parquet` 的 `prc1..prc5`、`vol0`（+ `g`、`x_0..x_7`、`x_60` 作为条件化/正交化辅助）。")
lines.append(f"> 全部变换均为因果的（时序仅用 `shift(L>=0)`，截面操作仅用同日行），详见 §0 方法论。")
lines.append("")
lines.append("## 0. 方法论：统一的三轮优化梯度 Methodology: a uniform 3-round optimization ladder")
lines.append("")
lines.append("每个思路（idea）先给出一个**因果 raw 信号**，再统一走三轮：")
lines.append("")
lines.append("| 轮次 | 处理 | 说明 |")
lines.append("|---|---|---|")
lines.append("| **v1 raw** | 逐日截面 z-score | 最小可行处理，作为对照基线 |")
lines.append("| **v2 processing (因子处理)** | 按类别选择：稳健(median/MAD) z-score / EWMA平滑 / 截面rank变换 / 十分位分桶平滑(分类优化) | 针对该类信号的统计特性（重尾、噪声、非线性）做针对性处理 |")
lines.append("| **v3 optimized (分类优化 + 残差回归)** | 按类别选择：行业(g)中性化 + 波动率残差回归 / log1p+稳健z+行业中性 / EWMA+行业相对中性 / rank+对x_0..x_7反转簇残差回归 / 分桶+行业中性 | 结构性优化：剥离已知因子暴露、行业共同因子、或做非线性分桶 |")
lines.append("")
lines.append("**决策纪律**：正负号、版本选择均只用 **train IC (day<=760)** 判定；**valid IC (761-880)** 仅作稳定性复核；"
             "**test (881-1259)** 全程不参与因子构造决策，留给下游模型做最终一次性评估——与仓库原有的 train/valid/test 纪律完全一致。")
lines.append("")
lines.append("**跨思路相关性剪枝**：每个思路取其（多参数变体中）train IC 绝对值最大的 v3 作为代表信号，"
             f"按 |train IC| 降序贪心保留；若某思路的代表信号与已保留思路的代表信号相关性 > **{LEDGER['corr_thresh']}**，则整组（含其所有参数变体）舍弃。")
lines.append("")
lines.append(f"**结果**：{LEDGER['n_ideas']} 个不同思路（{LEDGER['n_variants']} 个参数变体）→ 相关性剪枝后保留 "
             f"**{LEDGER['n_kept_ideas']} 个思路**、**{LEDGER['n_final_cols']} 个最终因子列**（每个思路的最终 v3 版本，"
             f"含全部参数变体）。")
lines.append("")

# ---- correlation pruning audit ----
lines.append("## 1. 相关性剪枝审计 Correlation-pruning audit")
lines.append("")
lines.append("| 被舍弃思路 dropped idea | 与其碰撞的已保留思路 collides with | \\|corr\\| |")
lines.append("|---|---|---:|")
for d in LEDGER["dropped"]:
    lines.append(f"| `{d['idea']}` | `{d['collides_with']}` | {d['abs_corr']:.3f} |")
lines.append("")

# ---- 3x+ IC showcase ----
lines.append("## 2. ≥3× 有效 IC 提升案例 Showcase: ideas with a ≥3x train-IC lift through the optimization ladder")
lines.append("")
lines.append("| 思路 idea | 变体 variant | v1 raw IC | v3 optimized IC | 倍数 multiple |")
lines.append("|---|---|---:|---:|---:|")
shown = 0
for idea, variant, v1, v3, mult in gains:
    if mult == mult and abs(mult) >= 3.0 and shown < 25:
        lines.append(f"| `{idea}` | {variant or '—'} | {v1:+.5f} | {v3:+.5f} | {mult:+.2f}x |")
        shown += 1
if shown == 0:
    lines.append("| _(none exceeded 3x on this run; see full ledger below for the largest gains)_ | | | | |")
lines.append("")
lines.append(f"（共 {shown} 例展示，完整每思路 IC 见 §3 全量表。）")
lines.append("")

# ---- full per-idea table, grouped by category ----
lines.append("## 3. 全量因子表：思路 x 变体 x 三轮 IC Full ledger: every idea x variant x 3-round IC")
lines.append("")
by_cat: dict[str, list[str]] = {}
for idea, rows in by_idea.items():
    cat = rows[0]["category"]
    by_cat.setdefault(cat, []).append(idea)

for cat in ("reversal", "vol_liquidity", "volume", "technical", "interaction"):
    ideas_in_cat = sorted(set(by_cat.get(cat, [])))
    if not ideas_in_cat:
        continue
    lines.append(f"### {CAT_LABEL[cat]}")
    lines.append("")
    for idea in ideas_in_cat:
        rows = by_idea[idea]
        status = "**[保留 kept]**" if idea in kept else f"**[舍弃 dropped, collides `{dropped[idea]['collides_with']}` |corr|={dropped[idea]['abs_corr']:.3f}]**"
        zh, en = rows[0]["zh"], rows[0]["en"]
        lines.append(f"#### `{idea}` {status}")
        lines.append(f"- **中文**：{zh}")
        lines.append(f"- **EN**: {en}")
        lines.append("")
        lines.append("| 变体 | 版本 | 处理方法 | train IC | valid IC |")
        lines.append("|---|---|---|---:|---:|")
        for r in rows:
            vname = r["variant"] or "—"
            for v in r["versions"]:
                lines.append(f"| {vname} | {v['name']} | {v['formula']} | {v['train_ic']:+.5f} | {v['valid_ic']:+.5f} |")
        lines.append("")

OUT.write_text("\n".join(lines))
print(f"[report] wrote {OUT} ({len(lines)} lines)", flush=True)
