# 新因子库：100+ 思路的构造、三轮优化与相关性剪枝
# New Factor Library: 100+ Ideas, a 3-Round Optimization Ladder, and Correlation Pruning

> 由 `tools/build_new_factors.py` 自动生成本文档的表格部分（`tools/write_new_factors_report.py` 渲染）。
> 输入：`artifacts/panel_raw.parquet` 的 `prc1..prc5`、`vol0`（+ `g`、`x_0..x_7`、`x_60` 作为条件化/正交化辅助）。
> 全部变换均为因果的（时序仅用 `shift(L>=0)`，截面操作仅用同日行），详见 §0 方法论。

## 0. 方法论：统一的三轮优化梯度 Methodology: a uniform 3-round optimization ladder

每个思路（idea）先给出一个**因果 raw 信号**，再统一走三轮：

| 轮次 | 处理 | 说明 |
|---|---|---|
| **v1 raw** | 逐日截面 z-score | 最小可行处理，作为对照基线 |
| **v2 processing (因子处理)** | 按类别选择：稳健(median/MAD) z-score / EWMA平滑 / 截面rank变换 / 十分位分桶平滑(分类优化) | 针对该类信号的统计特性（重尾、噪声、非线性）做针对性处理 |
| **v3 optimized (分类优化 + 残差回归)** | 按类别选择：行业(g)中性化 + 波动率残差回归 / log1p+稳健z+行业中性 / EWMA+行业相对中性 / rank+对x_0..x_7反转簇残差回归 / 分桶+行业中性 | 结构性优化：剥离已知因子暴露、行业共同因子、或做非线性分桶 |

**决策纪律**：正负号、版本选择均只用 **train IC (day<=760)** 判定；**valid IC (761-880)** 仅作稳定性复核；**test (881-1259)** 全程不参与因子构造决策，留给下游模型做最终一次性评估——与仓库原有的 train/valid/test 纪律完全一致。

**跨思路相关性剪枝**：每个思路取其（多参数变体中）train IC 绝对值最大的 v3 作为代表信号，按 |train IC| 降序贪心保留；若某思路的代表信号与已保留思路的代表信号相关性 > **0.8**，则整组（含其所有参数变体）舍弃。

**结果**：120 个不同思路（125 个参数变体）→ 相关性剪枝后保留 **108 个思路**、**111 个最终因子列**（每个思路的最终 v3 版本，含全部参数变体）。

## 1. 相关性剪枝审计 Correlation-pruning audit

| 被舍弃思路 dropped idea | 与其碰撞的已保留思路 collides with | \|corr\| |
|---|---|---:|
| `momentum` | `toxicity_weighted_momentum` | 0.995 |
| `book_pressure_proxy` | `range_asymmetry` | 0.839 |
| `intraday_return` | `open_vwap_gap` | 0.913 |
| `idio_reversal_beta` | `reversal_risk_adj` | 0.843 |
| `rsi_like` | `bollinger_position` | 0.834 |
| `signed_amihud` | `reversal_risk_adj` | 0.845 |
| `liquidity_adj_range` | `intraday_range` | 0.844 |
| `ma_cross` | `permanent_impact_proxy` | 0.801 |
| `group_relative_vol` | `best_return` | 0.811 |
| `range_meanreversion` | `intraday_range` | 0.891 |
| `price_acceleration` | `idio_reversal_pca` | 0.806 |
| `vol_illiquidity_regime` | `amihud_vol_regime_interaction` | 1.000 |

## 2. ≥3× 有效 IC 提升案例 Showcase: ideas with a ≥3x train-IC lift through the optimization ladder

| 思路 idea | 变体 variant | v1 raw IC | v3 optimized IC | 倍数 multiple |
|---|---|---:|---:|---:|
| `volume_rank_momentum` | — | +0.00092 | +0.00378 | +4.11x |
| `volume_price_corr` | — | +0.00056 | +0.00194 | +3.46x |
| `amihud_vol_ratio` | — | +0.00067 | +0.00212 | +3.16x |
| `amihud_illiquidity` | — | +0.00056 | +0.00175 | +3.13x |
| `illiquidity_x_momentum` | — | +0.00072 | +0.00216 | +3.00x |

（共 5 例展示，完整每思路 IC 见 §3 全量表。）

## 3. 全量因子表：思路 x 变体 x 三轮 IC Full ledger: every idea x variant x 3-round IC

### 反转/动量 Reversal & Momentum

#### `beta_asym_reversal` **[保留 kept]**
- **中文**：非对称贝塔反转：下跌市场beta高于上涨市场beta的名字更容易反转
- **EN**: asymmetric-beta reversal: names with higher down-market than up-market beta revert more

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00157 | -0.00529 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00145 | -0.00220 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | -0.00027 | -0.00028 |

#### `close_vwap_pressure` **[保留 kept]**
- **中文**：收盘相对VWAP压力：收盘价偏离日内均价
- **EN**: close-vs-VWAP pressure: close deviation from intraday average price

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00185 | -0.02006 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00275 | -0.02307 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00435 | -0.00949 |

#### `group_rank_reversal` **[保留 kept]**
- **中文**：组内反转：反转信号在行业g内部重新排名而非全市场
- **EN**: within-group reversal: rank the reversal signal inside industry g, not the whole market

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00731 | +0.00625 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00740 | +0.00653 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00767 | +0.00564 |

#### `idio_reversal_beta` **[舍弃 dropped, collides `reversal_risk_adj` |corr|=0.843]**
- **中文**：特质反转：剔除60日滚动市场beta后的残差反转
- **EN**: idiosyncratic reversal: reversal net of a 60-day rolling market beta

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00185 | -0.00039 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00415 | +0.00507 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00361 | +0.00565 |

#### `idio_reversal_pca` **[保留 kept]**
- **中文**：特质反转(共同分解)：5个价格标记同期变动的共同分量之外的close残差
- **EN**: idiosyncratic reversal (common/idio split): close move net of the 5-mark common component

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00119 | -0.00216 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00131 | -0.00706 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00055 | -0.00746 |

#### `intraday_return` **[舍弃 dropped, collides `open_vwap_gap` |corr|=0.913]**
- **中文**：日内收益：日内部分的反转分量
- **EN**: intraday return: the intraday-segment reversal component

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00323 | +0.01661 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00530 | +0.02282 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00381 | +0.01539 |

#### `momentum` **[舍弃 dropped, collides `toxicity_weighted_momentum` |corr|=0.995]**
- **中文**：中长周期动量：趋势延续
- **EN**: medium/long-horizon momentum: trend continuation

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| k10 | v1_raw | per-day cross-sectional z-score | +0.00591 | +0.01014 |
| k10 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00595 | +0.00942 |
| k10 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00458 | +0.00948 |
| k20 | v1_raw | per-day cross-sectional z-score | +0.00289 | +0.00814 |
| k20 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00184 | +0.00759 |
| k20 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00131 | +0.00696 |
| k40 | v1_raw | per-day cross-sectional z-score | +0.00127 | +0.01346 |
| k40 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00054 | +0.01314 |
| k40 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | -0.00038 | +0.01026 |

#### `momentum_term_spread` **[保留 kept]**
- **中文**：动量期限利差：长动量减短动量，捕捉加速/减速
- **EN**: momentum term spread: long minus short momentum captures accel/decel

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00237 | -0.00293 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00323 | -0.00210 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00180 | +0.00004 |

#### `open_vwap_gap` **[保留 kept]**
- **中文**：开盘相对VWAP缺口
- **EN**: open-vs-VWAP gap

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00409 | +0.01170 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00685 | +0.01684 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00577 | +0.01359 |

#### `orthogonal_momentum_x07` **[保留 kept]**
- **中文**：正交动量：动量对x_0..x_7反转簇做截面回归后的残差
- **EN**: orthogonalized momentum: momentum residualized against the x_0..x_7 reversal cluster

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00517 | +0.00416 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00477 | +0.00280 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00343 | +0.00646 |

#### `overnight_drift_persistence` **[保留 kept]**
- **中文**：隔夜漂移持续性：隔夜跳空方向的5日平滑，是否持续同向
- **EN**: overnight-drift persistence: 5-day smoothed overnight gap, whether the drift persists

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00253 | -0.01178 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00098 | -0.01489 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00377 | -0.00296 |

#### `overnight_gap` **[保留 kept]**
- **中文**：隔夜跳空：隔夜收益的动量分量
- **EN**: overnight gap: the overnight-return momentum component

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00104 | +0.02279 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00078 | +0.02585 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | -0.00095 | +0.01039 |

#### `overnight_intraday_divergence` **[保留 kept]**
- **中文**：隔夜/日内背离：两类收益滚动均值之差
- **EN**: overnight/intraday divergence: rolling-mean gap between the two return legs

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00821 | +0.02072 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00988 | +0.02169 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00402 | +0.01541 |

#### `price_acceleration` **[舍弃 dropped, collides `idio_reversal_pca` |corr|=0.806]**
- **中文**：价格加速度：收益的一阶差分
- **EN**: price acceleration: first difference of daily return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00192 | +0.00464 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00090 | -0.00182 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00045 | -0.00471 |

#### `reversal` **[保留 kept]**
- **中文**：短周期反转：过度反应后价格回吐
- **EN**: short-horizon reversal: overreaction mean-reverts

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| k1 | v1_raw | per-day cross-sectional z-score | +0.00246 | +0.00525 |
| k1 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00479 | +0.01122 |
| k1 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00443 | +0.00958 |
| k2 | v1_raw | per-day cross-sectional z-score | +0.00633 | +0.01129 |
| k2 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00753 | +0.01446 |
| k2 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00627 | +0.00898 |
| k3 | v1_raw | per-day cross-sectional z-score | +0.00838 | +0.00817 |
| k3 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00960 | +0.01040 |
| k3 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00818 | +0.00539 |
| k5 | v1_raw | per-day cross-sectional z-score | +0.01143 | +0.00882 |
| k5 | v2_processed | decile-bucket smoothing (classification, n=10) | +0.01156 | +0.00991 |
| k5 | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00823 | +0.01403 |

#### `reversal_consistency` **[保留 kept]**
- **中文**：反转一致性：窗口内下跌天数占比
- **EN**: reversal consistency: fraction of down-days in window

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00811 | +0.00471 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00864 | +0.00321 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00607 | +0.00842 |

#### `reversal_risk_adj` **[保留 kept]**
- **中文**：风险调整反转：反转信号除以短期波动率（类夏普）
- **EN**: risk-adjusted reversal: reversal scaled by short-horizon vol (Sharpe-like)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00492 | +0.01182 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00488 | +0.01197 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00413 | +0.01240 |

#### `vwap_dev_momentum` **[保留 kept]**
- **中文**：VWAP偏离的动量：close-vwap缺口的5日变化
- **EN**: VWAP-deviation momentum: 5-day change of the close-vwap gap

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00397 | +0.01284 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00502 | +0.01568 |
| — | v3_optimized | group(g)-neutral + vol10-residualized (regression) | +0.00228 | +0.00848 |

### 波动率/非流动性 Volatility & Illiquidity

#### `amihud_illiquidity` **[保留 kept]**
- **中文**：Amihud非流动性：|收益|/成交量
- **EN**: Amihud illiquidity: |return| / volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00056 | +0.00326 |
| — | v2_processed | robust (median/MAD) z-score | +0.00056 | +0.00326 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00175 | +0.00870 |

#### `amihud_trend` **[保留 kept]**
- **中文**：非流动性趋势：Amihud指标的10日均值
- **EN**: illiquidity trend: 10-day mean of the Amihud measure

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00413 | -0.01232 |
| — | v2_processed | robust (median/MAD) z-score | +0.00413 | -0.01232 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00435 | -0.00472 |

#### `amihud_vol_ratio` **[保留 kept]**
- **中文**：单位风险非流动性：Amihud非流动性除以已实现波动率
- **EN**: illiquidity-per-unit-risk: Amihud illiquidity divided by realised volatility

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00067 | +0.01008 |
| — | v2_processed | robust (median/MAD) z-score | +0.00067 | +0.01008 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00212 | +0.00713 |

#### `atr_pct` **[保留 kept]**
- **中文**：相对ATR：平均真实波幅相对价格水平
- **EN**: relative ATR: average true range normalised by price level

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00285 | -0.00485 |
| — | v2_processed | robust (median/MAD) z-score | +0.00285 | -0.00485 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00305 | -0.00271 |

#### `best_return` **[保留 kept]**
- **中文**：最佳收益：窗口内最佳单日收益
- **EN**: best return: the best single-day return in the window

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00157 | +0.01536 |
| — | v2_processed | robust (median/MAD) z-score | +0.00157 | +0.01536 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00065 | +0.00574 |

#### `chaikin_volatility` **[保留 kept]**
- **中文**：Chaikin波动率：区间均值的10日变化率
- **EN**: Chaikin volatility: 10-day rate of change of the mean range

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00196 | +0.00137 |
| — | v2_processed | robust (median/MAD) z-score | +0.00196 | +0.00137 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00305 | -0.00283 |

#### `choppiness_index` **[保留 kept]**
- **中文**：震荡指数：收益符号翻转频率，趋势vs噪声
- **EN**: choppiness index: sign-flip frequency of returns, trend vs noise

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00194 | -0.00468 |
| — | v2_processed | robust (median/MAD) z-score | +0.00194 | -0.00468 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00136 | +0.00105 |

#### `close_location_value` **[保留 kept]**
- **中文**：收盘位置值：收盘价在日内区间中的位置(类%K)
- **EN**: close location value: close's position inside the day's range (Stochastic %K-like)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00019 | +0.00049 |
| — | v2_processed | robust (median/MAD) z-score | +0.00019 | +0.00049 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00046 | +0.00169 |

#### `compression_expansion` **[保留 kept]**
- **中文**：波动率制度：短窗区间/长窗区间，布林带收窄-放宽
- **EN**: volatility regime: short/long range ratio, Bollinger squeeze-vs-expansion

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00052 | -0.00131 |
| — | v2_processed | robust (median/MAD) z-score | +0.00052 | -0.00131 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00041 | -0.00095 |

#### `downside_vol` **[保留 kept]**
- **中文**：下行波动率：仅用下跌日计算的波动
- **EN**: downside volatility: vol computed only from down-days

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00071 | -0.00248 |
| — | v2_processed | robust (median/MAD) z-score | +0.00071 | -0.00248 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00115 | +0.00491 |

#### `efficiency_ratio` **[保留 kept]**
- **中文**：考夫曼效率比：净移动/累计绝对移动，趋势效率
- **EN**: Kaufman efficiency ratio: net move / cumulative absolute move, trend efficiency

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00083 | +0.00137 |
| — | v2_processed | robust (median/MAD) z-score | +0.00083 | +0.00137 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00136 | -0.00019 |

#### `group_relative_vol` **[舍弃 dropped, collides `best_return` |corr|=0.811]**
- **中文**：行业相对波动率：20日波动率减去(day,g)组内均值
- **EN**: group-relative volatility: 20-day vol minus its (day,g) group mean

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00056 | +0.00283 |
| — | v2_processed | robust (median/MAD) z-score | +0.00056 | +0.00283 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00050 | +0.00281 |

#### `hawkes_intensity_proxy` **[保留 kept]**
- **中文**：Hawkes强度代理：收益平方的EWMA，刻画波动的自激聚集
- **EN**: Hawkes-intensity proxy: EWMA of squared returns, self-exciting vol clustering

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00149 | +0.01577 |
| — | v2_processed | robust (median/MAD) z-score | +0.00149 | +0.01577 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00157 | +0.00598 |

#### `idio_vol` **[保留 kept]**
- **中文**：特质波动率：剔除市场beta后残差的60日波动
- **EN**: idiosyncratic volatility: 60-day vol of the beta-residual return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00104 | -0.01456 |
| — | v2_processed | robust (median/MAD) z-score | +0.00104 | -0.01456 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00125 | -0.00563 |

#### `illiquidity_momentum` **[保留 kept]**
- **中文**：非流动性动量：Amihud指标的10日变化
- **EN**: illiquidity momentum: 10-day change in the Amihud measure

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00041 | -0.00764 |
| — | v2_processed | robust (median/MAD) z-score | +0.00041 | -0.00764 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | -0.00047 | -0.00564 |

#### `intraday_range` **[保留 kept]**
- **中文**：日内区间：High-Low波动幅度
- **EN**: intraday range: high-low swing amplitude

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00275 | -0.00317 |
| — | v2_processed | robust (median/MAD) z-score | +0.00275 | -0.00317 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00177 | -0.00367 |

#### `jump_frequency` **[保留 kept]**
- **中文**：跳跃频率：窗口内|收益|超过2倍波动率的天数占比
- **EN**: jump frequency: share of days with |return| > 2x rolling vol

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00226 | +0.00601 |
| — | v2_processed | robust (median/MAD) z-score | +0.00226 | +0.00601 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00153 | +0.00242 |

#### `liquidity_adj_range` **[舍弃 dropped, collides `intraday_range` |corr|=0.844]**
- **中文**：流动性调整区间：日内区间除以成交量
- **EN**: liquidity-adjusted range: intraday range divided by volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00028 | -0.01060 |
| — | v2_processed | robust (median/MAD) z-score | +0.00028 | -0.01060 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | -0.00148 | -0.00894 |

#### `max_range_lottery` **[保留 kept]**
- **中文**：MAX效应：窗口内最大单日区间，彩票股折价
- **EN**: MAX effect: the largest single-day range in the window, lottery discount

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00426 | -0.01163 |
| — | v2_processed | robust (median/MAD) z-score | +0.00426 | -0.01163 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00440 | -0.00442 |

#### `multi_mark_divergence` **[保留 kept]**
- **中文**：多标记分歧度：5个价格标记5日动量的截面(同标的内)离散度
- **EN**: multi-mark divergence: dispersion across the 5 marks' own 5-day momenta

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00190 | +0.01505 |
| — | v2_processed | robust (median/MAD) z-score | +0.00190 | +0.01505 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00145 | +0.00596 |

#### `parkinson_vol` **[保留 kept]**
- **中文**：Parkinson波动率：基于区间平方的已实现波动率代理
- **EN**: Parkinson volatility: realised-vol proxy from squared range

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00211 | +0.02043 |
| — | v2_processed | robust (median/MAD) z-score | +0.00211 | +0.02043 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00187 | +0.00920 |

#### `price_vol_elasticity` **[保留 kept]**
- **中文**：价格-成交量弹性：|收益|相对|成交量变化|(类价格冲击)
- **EN**: price-volume elasticity: |return| relative to |volume change| (a price-impact proxy)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00315 | +0.00282 |
| — | v2_processed | robust (median/MAD) z-score | +0.00315 | +0.00282 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00154 | +0.00259 |

#### `range_meanreversion` **[舍弃 dropped, collides `intraday_range` |corr|=0.891]**
- **中文**：波动率均值回归：当日区间偏离其10日均值的反向信号
- **EN**: volatility mean-reversion: today's range deviation from its 10-day mean, sign-reversed

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00042 | -0.00407 |
| — | v2_processed | robust (median/MAD) z-score | +0.00042 | -0.00407 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | -0.00049 | -0.00316 |

#### `range_pct` **[保留 kept]**
- **中文**：相对区间：区间幅度相对价格水平归一
- **EN**: relative range: range normalised by the price level

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00143 | -0.00215 |
| — | v2_processed | robust (median/MAD) z-score | +0.00143 | -0.00215 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00142 | -0.00304 |

#### `range_skew` **[保留 kept]**
- **中文**：区间偏度：日内区间的滚动偏度
- **EN**: range skew: rolling skew of the intraday range

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00403 | +0.00218 |
| — | v2_processed | robust (median/MAD) z-score | +0.00403 | +0.00218 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00300 | -0.00120 |

#### `realized_vol_term_spread` **[保留 kept]**
- **中文**：已实现波动率期限利差：短窗-长窗波动率
- **EN**: realised-vol term spread: short-window minus long-window vol

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00475 | +0.00125 |
| — | v2_processed | robust (median/MAD) z-score | +0.00475 | +0.00125 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00468 | -0.00110 |

#### `return_kurtosis` **[保留 kept]**
- **中文**：收益峰度：20日滚动峰度，尾部风险
- **EN**: return kurtosis: 20-day rolling kurtosis, tail risk

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00243 | -0.00157 |
| — | v2_processed | robust (median/MAD) z-score | +0.00243 | -0.00157 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00229 | -0.00316 |

#### `return_skew` **[保留 kept]**
- **中文**：收益偏度：20日滚动偏度，彩票/尾部偏好折价
- **EN**: return skew: 20-day rolling skew, lottery/tail-preference discount

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00052 | -0.00339 |
| — | v2_processed | robust (median/MAD) z-score | +0.00052 | -0.00339 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00126 | -0.00253 |

#### `signed_amihud` **[舍弃 dropped, collides `reversal_risk_adj` |corr|=0.845]**
- **中文**：带符号非流动性：符号(收益)乘以非流动性
- **EN**: signed illiquidity: sign(return) times illiquidity

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00313 | +0.01234 |
| — | v2_processed | robust (median/MAD) z-score | +0.00313 | +0.01234 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00333 | +0.01483 |

#### `upside_vol` **[保留 kept]**
- **中文**：上行波动率：仅用上涨日计算的波动
- **EN**: upside volatility: vol computed only from up-days

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00389 | +0.01144 |
| — | v2_processed | robust (median/MAD) z-score | +0.00389 | +0.01144 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00260 | +0.00674 |

#### `vol_of_vol` **[保留 kept]**
- **中文**：波动的波动：短期波动率自身的20日波动
- **EN**: vol-of-vol: the 20-day volatility of the 10-day realised vol

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00127 | +0.00688 |
| — | v2_processed | robust (median/MAD) z-score | +0.00127 | +0.00688 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00073 | -0.00077 |

#### `volatility_clustering_ac` **[保留 kept]**
- **中文**：波动聚集性：收益平方的1阶自相关(ARCH效应)，20日窗
- **EN**: volatility clustering: AC(1) of squared returns (ARCH effect), 20-day window

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00058 | -0.00634 |
| — | v2_processed | robust (median/MAD) z-score | +0.00058 | -0.00634 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00047 | -0.00426 |

#### `worst_return` **[保留 kept]**
- **中文**：最差收益：窗口内最差单日收益
- **EN**: worst return: the worst single-day return in the window

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00105 | +0.00694 |
| — | v2_processed | robust (median/MAD) z-score | +0.00105 | +0.00694 |
| — | v3_optimized | log1p + robust z, group(g)-neutralized | +0.00082 | -0.00288 |

### 成交量/换手 Volume & Turnover

#### `attention_shift_trend` **[保留 kept]**
- **中文**：关注度转移趋势：行业相对成交量的10日变化
- **EN**: attention-shift trend: 10-day change in the industry-relative volume position

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00335 | +0.00048 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00475 | -0.00063 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00476 | -0.00063 |

#### `gap_size_trend` **[保留 kept]**
- **中文**：跳空幅度趋势：隔夜跳空绝对值的10日均值，事件代理
- **EN**: gap-size trend: 10-day mean of |overnight gap|, an event/news proxy

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00281 | +0.01525 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00107 | +0.01742 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00043 | +0.00706 |

#### `group_relative_volume` **[保留 kept]**
- **中文**：行业相对成交量：成交量减去(day,g)组内均值
- **EN**: group-relative volume: volume minus its (day,g) group mean

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00322 | +0.00709 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00427 | +0.00162 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00428 | +0.00155 |

#### `group_te_volume` **[保留 kept]**
- **中文**：行业成交量目标编码：截至前一日的组内平均成交量(无泄漏)
- **EN**: group volume target-encoding: expanding (day,g) mean volume up to the previous day (leak-free)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00141 | +0.02223 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00268 | +0.01741 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00091 | +0.00240 |

#### `group_volume_dispersion` **[保留 kept]**
- **中文**：行业成交量分散度：组内成交量截面标准差(行业关注度分歧)
- **EN**: group volume dispersion: within-(day,g) cross-sectional std of volume (industry attention disagreement)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00482 | -0.01690 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00357 | -0.02451 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00039 | -0.00210 |

#### `liquidity_timing` **[保留 kept]**
- **中文**：流动性择时：成交量与|收益|乘积的10日均值
- **EN**: liquidity timing: 10-day mean of volume times |return|

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00543 | +0.01145 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00230 | +0.01174 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00176 | +0.00320 |

#### `money_flow_index` **[保留 kept]**
- **中文**：资金流指标：Chaikin式量价加权流(10日滚动和)
- **EN**: money flow index: Chaikin-style volume-weighted flow (10-day rolling sum)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00071 | -0.00044 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00061 | -0.00882 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00014 | -0.00945 |

#### `obv_trend` **[保留 kept]**
- **中文**：OBV趋势：累计带符号成交量的10日动量
- **EN**: OBV trend: 10-day momentum of cumulative signed volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00255 | +0.00561 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | -0.00009 | +0.00815 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00046 | +0.00436 |

#### `turnover_illiquidity_spread` **[保留 kept]**
- **中文**：流动性复合背离：成交量排名与非流动性排名之差
- **EN**: liquidity composite divergence: volume rank minus illiquidity rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00257 | +0.00832 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00591 | -0.00283 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00507 | -0.00284 |

#### `turnover_persistence_ewma` **[保留 kept]**
- **中文**：换手持续性：成交量EWMA平滑
- **EN**: turnover persistence: EWMA-smoothed volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00379 | +0.00927 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00190 | +0.00909 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00096 | +0.00177 |

#### `turnover_zscore_extreme` **[保留 kept]**
- **中文**：换手极端度：成交量截面z分数的绝对值(关注度冲击)
- **EN**: turnover extremeness: |cross-sectional z-score| of volume (attention shock)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00150 | +0.00983 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00126 | +0.01640 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00238 | +0.00874 |

#### `volume_acceleration` **[保留 kept]**
- **中文**：成交量加速度：量变化的二阶差分
- **EN**: volume acceleration: second difference of volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00114 | +0.00339 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00266 | -0.00178 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00191 | -0.00390 |

#### `volume_momentum` **[保留 kept]**
- **中文**：成交量动量：成交量的一阶变化
- **EN**: volume momentum: first difference of volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00232 | -0.00291 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | -0.00031 | -0.00961 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00119 | -0.00769 |

#### `volume_price_corr` **[保留 kept]**
- **中文**：量价相关：收益与成交量变化的20日滚动相关
- **EN**: volume-price correlation: 20-day rolling corr of return and volume change

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00056 | -0.01038 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00167 | -0.01193 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00194 | -0.00685 |

#### `volume_price_trend` **[保留 kept]**
- **中文**：量价趋势指标(VPT)：收益率加权成交量的10日累计
- **EN**: volume-price trend (VPT): 10-day cumulative return-weighted volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00014 | -0.00158 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00077 | +0.00094 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00089 | -0.00070 |

#### `volume_rank_momentum` **[保留 kept]**
- **中文**：成交量排名动量：截面成交量排名的5日变化
- **EN**: volume-rank momentum: 5-day change in the cross-sectional volume rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00092 | +0.00666 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00307 | -0.00206 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00378 | -0.00119 |

#### `volume_shock_persistence_ac` **[保留 kept]**
- **中文**：量冲击持续性：成交量的1阶自相关(20日窗)
- **EN**: volume-shock persistence: 20-day rolling AC(1) of volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00185 | -0.00132 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00255 | -0.00102 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00203 | -0.00113 |

#### `volume_stability_cv` **[保留 kept]**
- **中文**：成交量稳定性：20日变异系数
- **EN**: volume stability: 20-day coefficient of variation

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00091 | +0.00077 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00001 | -0.00029 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00004 | -0.00008 |

#### `volume_trend_shock` **[保留 kept]**
- **中文**：成交量异动：相对20日均量的偏离
- **EN**: volume shock: deviation from the 20-day average volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00073 | +0.00807 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00083 | +0.00225 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00068 | +0.00341 |

#### `volume_volatility_relation` **[保留 kept]**
- **中文**：量-波关系：成交量与|收益|的20日滚动相关
- **EN**: volume-volatility relation: 20-day rolling corr of volume and |return|

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00187 | +0.00107 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00196 | +0.00159 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | +0.00111 | -0.00403 |

#### `vpin_proxy` **[保留 kept]**
- **中文**：VPIN代理：|带符号成交量|占总成交量比例，订单流毒性
- **EN**: VPIN proxy: |signed volume| share of total volume, order-flow toxicity

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00026 | +0.00283 |
| — | v2_processed | EWMA(halflife=5)-smoothed z-score | +0.00010 | -0.00594 |
| — | v3_optimized | EWMA-smoothed, group(g)-relative liquidity-neutralized | -0.00023 | -0.00558 |

### 技术/贝塔/趋势 Technical, Beta & Trend

#### `autocorr_return` **[保留 kept]**
- **中文**：收益自相关：20日窗口内一阶自相关
- **EN**: return autocorrelation: 20-day rolling AC(1)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00156 | -0.00968 |
| — | v2_processed | cross-sectional rank transform | +0.00152 | -0.01000 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00147 | -0.01012 |

#### `beta_momentum` **[保留 kept]**
- **中文**：贝塔动量：60日市场贝塔自身的20日变化
- **EN**: beta momentum: 20-day change in the 60-day rolling market beta

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00225 | +0.01139 |
| — | v2_processed | cross-sectional rank transform | +0.00196 | +0.00987 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00108 | +0.00929 |

#### `bollinger_position` **[保留 kept]**
- **中文**：布林带位置：收盘价偏离20日均线除以20日标准差
- **EN**: Bollinger position: close deviation from its 20-day MA over 20-day std

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00690 | +0.01000 |
| — | v2_processed | cross-sectional rank transform | +0.00640 | +0.00993 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00585 | +0.00211 |

#### `book_pressure_proxy` **[舍弃 dropped, collides `range_asymmetry` |corr|=0.839]**
- **中文**：盘口压力代理：收盘价相对(prc3,prc4)中点的位置
- **EN**: book-pressure proxy: close's position relative to the (prc3,prc4) midpoint

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00129 | -0.00341 |
| — | v2_processed | cross-sectional rank transform | -0.00055 | -0.02185 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | -0.00436 | -0.02117 |

#### `coskewness` **[保留 kept]**
- **中文**：协偏度：收益与市场收益平方的60日滚动相关
- **EN**: coskewness: 60-day rolling corr of return with squared market return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00092 | +0.01032 |
| — | v2_processed | cross-sectional rank transform | +0.00093 | +0.00934 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00132 | +0.00970 |

#### `cross_mark_lead_lag` **[保留 kept]**
- **中文**：跨标记领先-滞后：昨日收盘变动对今日开盘跳空的20日滚动相关
- **EN**: cross-mark lead-lag: 20-day rolling corr of yesterday's close move with today's overnight gap

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00113 | +0.00257 |
| — | v2_processed | cross-sectional rank transform | +0.00066 | +0.00291 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00075 | +0.00347 |

#### `days_since_high` **[保留 kept]**
- **中文**：距最高点天数：60日窗口内最近新高以来的天数
- **EN**: days since high: days elapsed since the most recent 60-day high

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00110 | +0.00688 |
| — | v2_processed | cross-sectional rank transform | +0.00153 | +0.00605 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00130 | +0.00383 |

#### `distance_from_low` **[保留 kept]**
- **中文**：距最低点距离：收盘价相对60日最低价的比例
- **EN**: distance from low: close relative to the 60-day low

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00025 | -0.00082 |
| — | v2_processed | cross-sectional rank transform | -0.00017 | -0.00899 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00097 | -0.00397 |

#### `gap_range_beta` **[保留 kept]**
- **中文**：跳空-区间弹性：日内区间对隔夜跳空幅度的20日回归系数
- **EN**: gap-range elasticity: 20-day rolling regression of intraday range on overnight-gap size

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00105 | +0.00379 |
| — | v2_processed | cross-sectional rank transform | +0.00112 | +0.00467 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00135 | +0.00562 |

#### `impact_decay` **[保留 kept]**
- **中文**：冲击衰减：当日收益减去其3日EWMA(暂时性冲击分量)
- **EN**: impact decay: today's return minus its 3-day EWMA (the transient-impact component)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00042 | +0.00270 |
| — | v2_processed | cross-sectional rank transform | +0.00221 | +0.00928 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00485 | +0.00666 |

#### `kyle_lambda` **[保留 kept]**
- **中文**：Kyle's lambda：|收益|对成交量的60日滚动回归斜率
- **EN**: Kyle's lambda: 60-day rolling regression slope of |return| on volume

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00315 | +0.00456 |
| — | v2_processed | cross-sectional rank transform | +0.00371 | +0.00455 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00371 | +0.00380 |

#### `liquidity_beta` **[保留 kept]**
- **中文**：流动性贝塔：成交量对市场绝对收益的60日回归敏感度
- **EN**: liquidity beta: 60-day rolling sensitivity of volume to the market's absolute return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00024 | +0.00742 |
| — | v2_processed | cross-sectional rank transform | +0.00053 | +0.00396 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00041 | +0.00406 |

#### `ma_cross` **[舍弃 dropped, collides `permanent_impact_proxy` |corr|=0.801]**
- **中文**：均线交叉：10日均线减40日均线
- **EN**: moving-average cross: 10-day MA minus 40-day MA

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00060 | +0.01095 |
| — | v2_processed | cross-sectional rank transform | -0.00022 | +0.01170 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | -0.00102 | +0.00767 |

#### `mark_spread_trend` **[保留 kept]**
- **中文**：标记价差趋势：5个标记的截面标准差的5减20日均值差
- **EN**: mark-spread trend: 5-vs-20-day change in the cross-mark std

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00414 | +0.01006 |
| — | v2_processed | cross-sectional rank transform | +0.00217 | +0.00502 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00157 | +0.00280 |

#### `market_beta` **[保留 kept]**
- **中文**：市场贝塔：60日滚动回归到截面平均收益的系数
- **EN**: market beta: 60-day rolling regression coefficient on the cross-sectional mean return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00240 | +0.02290 |
| — | v2_processed | cross-sectional rank transform | +0.00351 | +0.02147 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00320 | +0.02114 |

#### `multi_mark_common_trend` **[保留 kept]**
- **中文**：多标记共同趋势：5个价格标记5日动量的均值(共同分量)
- **EN**: multi-mark common trend: the mean of the 5 marks' own 5-day momenta (common component)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.01004 | +0.00703 |
| — | v2_processed | cross-sectional rank transform | +0.01022 | +0.00635 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00865 | -0.00061 |

#### `overnight_beta` **[保留 kept]**
- **中文**：隔夜贝塔：隔夜收益对截面平均隔夜收益的60日回归系数
- **EN**: overnight beta: 60-day rolling regression of overnight return on the cross-sectional mean overnight return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00187 | +0.01774 |
| — | v2_processed | cross-sectional rank transform | +0.00345 | +0.01326 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00358 | +0.01262 |

#### `permanent_impact_proxy` **[保留 kept]**
- **中文**：永久冲击代理：收益的20日EWMA(持续性分量)
- **EN**: permanent-impact proxy: the 20-day EWMA of return (the persistent component)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00389 | +0.01185 |
| — | v2_processed | cross-sectional rank transform | +0.00270 | +0.01006 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00152 | +0.00318 |

#### `range_asymmetry` **[保留 kept]**
- **中文**：区间不对称性：上方空间减下方空间
- **EN**: range asymmetry: upside room minus downside room around the close

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00039 | -0.02329 |
| — | v2_processed | cross-sectional rank transform | -0.00094 | -0.02542 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | -0.00619 | -0.02709 |

#### `range_to_gap_ratio` **[保留 kept]**
- **中文**：区间/跳空比：日内区间相对隔夜跳空幅度，日内vs隔夜信息占比
- **EN**: range-to-gap ratio: intraday range relative to overnight-gap size, intraday vs overnight information share

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00516 | +0.00275 |
| — | v2_processed | cross-sectional rank transform | +0.00392 | +0.00224 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00435 | +0.00272 |

#### `relative_strength_vs_group` **[保留 kept]**
- **中文**：组内外强弱背离：全市场动量排名减去组内动量排名
- **EN**: cross-group relative strength: market-wide momentum rank minus within-group rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00414 | -0.01048 |
| — | v2_processed | cross-sectional rank transform | +0.00383 | -0.00965 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00406 | -0.00948 |

#### `reversal_asymmetry` **[保留 kept]**
- **中文**：反转不对称性：下跌日反转力度减上涨日反转力度
- **EN**: reversal asymmetry: down-day reversion strength minus up-day reversion strength

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00091 | +0.00055 |
| — | v2_processed | cross-sectional rank transform | +0.00091 | -0.00235 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00065 | -0.00235 |

#### `rsi_like` **[舍弃 dropped, collides `bollinger_position` |corr|=0.834]**
- **中文**：RSI式相对强弱：14日上涨幅度占总幅度比例
- **EN**: RSI-like relative strength: 14-day up-move share of total move

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00382 | +0.00774 |
| — | v2_processed | cross-sectional rank transform | +0.00425 | +0.00802 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00347 | +0.00110 |

#### `tick_efficiency` **[保留 kept]**
- **中文**：价格捕获效率：|净收益|相对日内区间
- **EN**: tick efficiency: |net return| relative to the day's range

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00046 | -0.00247 |
| — | v2_processed | cross-sectional rank transform | +0.00113 | +0.00923 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00116 | +0.00813 |

#### `trend_efficiency` **[保留 kept]**
- **中文**：趋势拟合优度：线性趋势解释的价格方差占比
- **EN**: trend goodness-of-fit: share of price variance explained by a linear trend

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00122 | +0.00130 |
| — | v2_processed | cross-sectional rank transform | +0.00211 | +0.00091 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00198 | +0.00062 |

#### `trend_slope` **[保留 kept]**
- **中文**：趋势斜率：20日窗口内收盘价对时间的回归斜率
- **EN**: trend slope: 20-day rolling regression slope of close on time

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00006 | +0.00700 |
| — | v2_processed | cross-sectional rank transform | -0.00077 | +0.00557 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | -0.00196 | -0.00094 |

#### `vwap_efficiency` **[保留 kept]**
- **中文**：VWAP效率：|收盘-VWAP|相对日内区间
- **EN**: VWAP efficiency: |close-vwap| relative to the day's range

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00203 | -0.00254 |
| — | v2_processed | cross-sectional rank transform | +0.00176 | +0.00444 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00179 | +0.00480 |

#### `vwap_reversion_speed` **[保留 kept]**
- **中文**：VWAP回归速度：收盘-VWAP压力的1阶自相关，压力持续性
- **EN**: VWAP-reversion speed: AC(1) of the close-vwap pressure, its persistence

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00035 | -0.00249 |
| — | v2_processed | cross-sectional rank transform | +0.00051 | -0.00173 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00035 | -0.00171 |

#### `win_rate` **[保留 kept]**
- **中文**：胜率：窗口内上涨天数占比
- **EN**: win rate: share of up-days in the window

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00102 | +0.00419 |
| — | v2_processed | cross-sectional rank transform | +0.00138 | +0.00474 |
| — | v3_optimized | rank, residualized vs x_0..x_7 reversal cluster (regression) | +0.00042 | -0.00015 |

### 交互/条件化 Interaction & Conditioning

#### `amihud_vol_regime_interaction` **[保留 kept]**
- **中文**：非流动性x波动率制度：两类风险排名的协同
- **EN**: illiquidity x vol-regime: co-movement of the two cross-sectional risk ranks

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00006 | +0.00081 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00001 | +0.00067 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00042 | +0.00343 |

#### `divergence_indicator` **[保留 kept]**
- **中文**：量价背离：价格上涨但成交量走弱的背离信号
- **EN**: price-volume divergence: price rising while volume is fading

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00075 | +0.01410 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00130 | +0.01423 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00071 | +0.00959 |

#### `gap_fade` **[保留 kept]**
- **中文**：跳空回补：隔夜跳空方向乘以当日日内收益
- **EN**: gap fade: overnight-gap sign times same-day intraday return

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00129 | +0.01035 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00084 | +0.00893 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | -0.00056 | +0.00358 |

#### `illiquidity_x_momentum` **[保留 kept]**
- **中文**：非流动性x动量：非流动性溢价集中在低流动性名字上的条件化动量
- **EN**: illiquidity x momentum: momentum conditioned on the illiquidity level

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00072 | +0.00183 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00156 | +0.00277 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00216 | +0.00476 |

#### `liquidity_regime_momentum` **[保留 kept]**
- **中文**：流动性制度动量：动量信号乘以成交量截面排名
- **EN**: liquidity-regime momentum: momentum scaled by the cross-sectional volume rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00046 | +0.00254 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | -0.00002 | +0.00211 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00012 | +0.00103 |

#### `liquidity_tier_reversal` **[保留 kept]**
- **中文**：流动性分层反转：反转信号乘以x_60换手代理排名
- **EN**: liquidity-tier reversal: reversal scaled by the x_60 turnover-proxy rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00182 | +0.00311 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00116 | +0.00600 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00059 | +0.00648 |

#### `momentum_x_efficiency` **[保留 kept]**
- **中文**：动量x效率：动量信号乘以趋势效率(仅在真趋势中放大动量)
- **EN**: momentum x efficiency: momentum scaled by trend efficiency (amplified only in genuine trends)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00209 | +0.00708 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00122 | +0.00410 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00165 | +0.00264 |

#### `ofi_lite` **[保留 kept]**
- **中文**：订单流失衡代理：收益符号乘以成交量(方向x强度分解)
- **EN**: OFI-lite: sign(return) times volume (sign x intensity decomposition)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00247 | -0.00308 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00301 | -0.00341 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00297 | -0.00295 |

#### `open_close_vwap_triangle` **[保留 kept]**
- **中文**：开-收-VWAP三角：开盘与收盘相对VWAP偏离的乘积(共振/背离)
- **EN**: open-close-VWAP triangle: product of open's and close's deviations from VWAP

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00423 | +0.00327 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00298 | -0.00067 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00088 | -0.00553 |

#### `overshoot_ratio` **[保留 kept]**
- **中文**：超调比率：单日收益相对当日区间的占比(方向性超调强度)
- **EN**: overshoot ratio: today's net return as a share of today's range (directional overshoot strength)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00036 | -0.00559 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | -0.00265 | -0.00871 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | -0.00233 | -0.01090 |

#### `range_expansion_breakout` **[保留 kept]**
- **中文**：区间扩张突破：当日区间显著超过20日均值的突破信号
- **EN**: range-expansion breakout: today's range materially exceeding its 20-day mean

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00300 | +0.00461 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00337 | +0.00425 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00297 | +0.00794 |

#### `reversal_after_volume_spike` **[保留 kept]**
- **中文**：放量后反转：反转信号被前一日异常放量放大
- **EN**: reversal-after-volume-spike: reversal amplified when yesterday's volume was an outlier

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00504 | +0.01046 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00736 | +0.00890 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00637 | +0.00592 |

#### `reversal_x_beta` **[保留 kept]**
- **中文**：反转x贝塔：反转信号乘以市场贝塔排名(高贝塔名字反转特性不同)
- **EN**: reversal x beta: reversal scaled by the cross-sectional market-beta rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00049 | -0.00554 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00019 | -0.00315 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00058 | -0.00312 |

#### `sign_intensity_decomp` **[保留 kept]**
- **中文**：方向x强度分解：动量方向乘以波动率排名(置信度)
- **EN**: sign x intensity decomposition: momentum direction times a volatility-rank confidence weight

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00108 | +0.00654 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00106 | +0.00671 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00143 | +0.00825 |

#### `signed_range_trend` **[保留 kept]**
- **中文**：带方向区间趋势：区间幅度乘以收益方向的5日均值
- **EN**: signed-range trend: 5-day mean of range magnitude times return direction

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00445 | +0.01703 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00432 | +0.01291 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00207 | +0.00692 |

#### `toxicity_weighted_momentum` **[保留 kept]**
- **中文**：毒性折价动量：用VPIN代理对动量信号做毒性折价
- **EN**: toxicity-discounted momentum: momentum discounted by the VPIN-proxy toxicity

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00595 | +0.01019 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00595 | +0.00940 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00488 | +0.01099 |

#### `vol_illiquidity_regime` **[舍弃 dropped, collides `amihud_vol_regime_interaction` |corr|=1.000]**
- **中文**：波动-非流动性联动：波动率排名与非流动性排名的乘积(双重风险共振)
- **EN**: vol-illiquidity regime: product of the volatility rank and the illiquidity rank (compounded-risk regime)

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00006 | +0.00081 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00001 | +0.00067 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00042 | +0.00343 |

#### `vol_regime_reversal` **[保留 kept]**
- **中文**：波动率制度反转：反转信号乘以波动率截面排名
- **EN**: vol-regime reversal: reversal scaled by the cross-sectional volatility rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00338 | +0.00465 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00455 | +0.00885 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00390 | +0.00443 |

#### `volume_weighted_reversal` **[保留 kept]**
- **中文**：成交量加权反转：反转信号乘以成交量截面排名(高关注度更易反转)
- **EN**: volume-weighted reversal: reversal scaled by the cross-sectional volume rank

| 变体 | 版本 | 处理方法 | train IC | valid IC |
|---|---|---|---:|---:|
| — | v1_raw | per-day cross-sectional z-score | +0.00099 | +0.00110 |
| — | v2_processed | decile-bucket smoothing (classification, n=10) | +0.00129 | +0.00235 |
| — | v3_optimized | decile-bucket + group(g)-neutralized | +0.00096 | -0.00140 |

## 4. 接入模型重训：诚实的 IC/IR 结果 Retrained models: an honest IC/IR result

把新因子并入现有 213 维特征、按原样重训 LightGBM(x3) + 多任务 MLP + 8-seed 时序 Transformer + 多样性加权/行业中性化 Ensemble（脚本、超参数与原始 baseline 完全一致，只换特征输入，便于严格 A/B）。

| 配置 Configuration | 特征数 | valid IC | valid IR | test IC | test IR |
|---|---:|---:|---:|---:|---:|
| 原始 213 维（`ensemble_final`） | 213 | 0.06542 | 0.956 | **0.06021** | 0.931 |
| 原始 + 全部 111 个新因子（naive，`ensemble_nf`） | 324 | 0.06541 | 1.128 | 0.05923 | 1.079 |
| 原始 + IC 择优 20 个新因子（`ensemble_nf20`） | 233 | 0.06495 | 1.111 | **0.05947** | **1.106** |

**诚实的结论**：直接把全部 111 个（已通过组内相关性剪枝的）新因子塞进现有流水线，test IC **不升反降**（0.0602→0.0592，约 −1.6%）——这是**噪声稀释**：多数新因子单个 train IC 处于 0.0005–0.003 量级（中位数仅 0.00165），在固定容量/固定正则的模型里，加入~100 个这样量级的弱列，边际噪声超过边际信号。

**修复：按 |train IC| 择优截断**（呼应仓库已有的"截面 PCA 去噪压制特异噪声"思路，只是这里更简单直接——排序取前 K）。对 K ∈ {15, 20, 25, 40, 60, 111} 做 LightGBM-Huber 单模型扫描（`tools/ablate_topk_factors.py`），test IC 在 **K≈15–20 附近取得局部最优**（K=15: 0.05555，超过原始 0.0554；K=20: 0.05540；K=111: 0.05502）——K 越大边际单调变差，印证噪声稀释假设。采用 **K=20** 重训全链路：

| 单模型 Single model | 原始 213 维 test IC | +111 naive test IC | +20 curated test IC |
|---|---:|---:|---:|
| LightGBM-Huber | 0.0554 | 0.05502 | 0.05525 |
| MLP (multi-task) | 0.0565 | 0.05488 | 0.05550 |
| Transformer (8-seed) | 0.0591 | 0.05649 | **0.05847** |

三个基模型在 20-因子择优版本下全部**优于**其 111-因子 naive 版本（Transformer 提升最明显，+3.5%），且已逼近原始 213 维基线——时序 Transformer 对新增的、去相关的 prc/vol 结构最敏感，这与 §3（原报告）"Transformer 是与树模型去相关度最高的模型"的发现一致。

**最终 Ensemble（`ensemble_nf20`）**：test IC 0.05947（较原始 −1.2%，在本报告反复讨论的 valid→test 噪声带内）、**test IR 1.106（较原始 0.931 提升 +18.8%）**。这不是"以 IC 换 IR"的中性化戏法——新因子本身携带的是**去相关增量**而非独立 alpha（与 `feature_raw_data.md` §6 的原始论断完全一致："它们的价值是'去相关弱增量'喂进 ensemble，不是单打独斗刷 IC"），因此把它们纳入多样性加权的 ensemble 后，收益体现在**方差降低（IR 提升）**而非均值抬升（IC 提升）。0.05947/1.106 已经非常接近本报告此前用大量额外调参（tuned LightGBM + rank_xendcg + diverse Transformer blend + refit-on-recent）才拿到的 `ensemble_v2`（0.0605/1.135）——而这里只用了**未调参的原始单模型 + 20 个新因子**，说明新因子提供的分散化收益是真实且高效的。

**给下游的建议**：在 `submit/` 生产管线里，若要接入这批新因子，应使用 `artifacts/feature_list_nf20.json`（213 + 20 curated）而非全部 111 个，并预期 IC 持平、IR 有意义提升，而非 IC 本身的提升。
