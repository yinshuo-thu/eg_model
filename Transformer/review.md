# Transformer 优化 Review（v1 → v2 深度路线图）

> 面向 `eg_model` 的**每日横截面 alpha** 任务。本文档结合数据/特征语义、评测指标数学、
> 集成与部署约束、以及全部已证伪实验，给出**可核验的量化目标**与**分层的实现角度**。
> 路径以仓库根 `eg_model/` 为基准；代码引用形如 `文件:行`。
>
> 阅读顺序：**§1 场景 → §2 目标（GOALS）→ §3 别重复的死路 → §4 优化路线 → §5 执行顺序 → §6 记录规范**。

---

## 0. 版本与目录约定

- **`Transformer/v1/`** = 已冻结的第 1 代（`run_transformer.py` 单塔 base、`run_transformer_v2.py` 实验壳、`run_transformer_v3.py` 横截面注意力[已证伪]、`build_v2_final.py` 多样性 blend+中性化）。**不再原地改。**
- 新一轮成体系改动 → 新建 `Transformer/v2/`，`scripts/ metrics/ …` 同构；`review.md` 常驻 `Transformer/` 根。
- **纪律**：同一 split（train ≤760 / valid 761–880 / test ≥881）、同一 `daily_ic` 口径（`ML_single/scripts/common.py:26`）复评；**决策只看 test，且每次都要跑一遍集成终审（§2.3）。**

---

## 1. 场景全景（先把目标函数搞对，再谈技巧）

### 1.1 数据 / 任务 / 指标（精确定义）
- **面板**：1,333 标的 × 1,259 天 ≈ 1.68M 行，近似满秩（几乎每个标的每天都在）——这正是"把横截面面板当作 per-instrument 日频时序"这一 Transformer 设计的前提。
- **标的**：`y` = 次日原始收益（std≈0.094，∈[−1,1]）；`y_xs` = 每日横截面 z-score（±6 截断）。**训练回归 `y_xs`，但评测在原始 `y` 上算 IC。**
- **特征**：213 个，**全部逐日横截面 z-score + ±6 截断 + NaN→0**（`tools/build_features.py`）。家族构成：
  - 86 个匿名因子 `x_0..x_85`（本身已 CS z-score）；
  - 90 个 **top-30 因子的时序展开**（`lag1 / mom5 / r10`）；
  - 11 个 **y 的自相关特征**（`y_lag1/2/3/5`、`y_roll/vol 5/10/20`、`y_ewm`）；
  - 11 个 价量（`prc1..prc5` 聚合、`vol0`）；
  - 3 个 分组（`grp_ylag_mean`、`grp_xbest_mean`、`grp_te` 泄漏安全的目标编码）；
  - 12 个 PCA（仅在 train 上拟合）。
  - `g` = 行业分组（−1 未分类，0–72，共 ~74 组）——中性化就针对它。
- **指标**（`common.py:26-46`）：
  - **IC** = 每日 `corr(pred, y)`（Pearson）在天上的均值；
  - **IR** = `mean(每日IC) / std(每日IC)`——**是稳定性/一致性比率，不是收益率**；
  - **spear** = 每日 Spearman 均值；**pos** = 日 IC 为正的天数占比。

### 1.2 现状全景（test，days 881–1259）

| 主体 | IC | IR | σ_daily=IC/IR | 说明 |
|---|---|---|---|---|
| **baseline `y_hat0`** | 0.0560 | **1.10** | 0.0509 | 低 IC、**高稳定**——这是我们要超越的对象 |
| 线性(Ridge/ENet) | ~0.042 | ~0.58 | ~0.073 | 信号强非线性 |
| LightGBM 家族 | 0.05589 | 0.953 | 0.0587 | dart 单体 IR 最高(1.051) |
| MLP(多任务,6-seed) | 0.05647 | 0.866 | 0.0652 | valid 最高但 test 掉队(过拟合) |
| **Transformer(8-seed)** | **0.05906** | **0.846** | **0.0698** | **IC 全场最高，IR/稳定性全场最差** |
| Ensemble(等权 3 家族) | 0.06021 | 0.931 | 0.0647 | |
| **Ensemble(divN+中性化α0.35)** | **0.06050** | **1.135** | 0.0533 | **当前生产/最优** |
| Transformer 多样性 blend(raw) | 0.06012 | 0.907 | 0.0663 | 纯 Transformer 家族上限 |
| Transformer blend+中性化α0.62 | 0.05901 | 1.101 | 0.0536 | 过了家族内闸口 |

**跨模型相关（test，`family_corr`）**：Transformer↔LightGBM **0.794**（全场最低，最值钱）；Transformer↔MLP **0.921**（高度冗余）；MLP↔LightGBM 0.829。
**部署**（`submit/weights/ensemble_config.json`）：集成权重 LGB **0.445** / MLP **0.276** / Transformer **0.279**；`neut_alpha=0.35`；**K=32 与 213-特征 schema 均为冻结契约**；Transformer 8 seeds × 2.2MB。

### 1.3 三条必须内化的结论（决定优化方向）

1. **Transformer 在这套系统里的身份是"集成的去相关器"**，不是"单打冠军"。它对 LGB 只有 0.794 相关（部署里 27.9% 权重多半来自这份正交性），但**对 MLP 高达 0.921——两台神经网络在同一张 213 特征表上高度重复**。
2. **它的病灶是 IR 不是 IC**：IC 0.0591 全场第一，但 σ_daily=0.070 **全场最高（比 baseline 高 37%）**，导致 IR 0.85 垫底；而下游闸口恰恰卡在 IR>1.10，只能靠中性化把 IC 换 IR 硬凑。
3. **瓶颈是 valid→test gap，不是容量**（报告与 `transformer_ineffective.md` 反复证实）：加宽/加深、加横截面注意力都把 valid 抬高却把 test 压低。**只有"多样性 + refit-on-recent + 中性化"真正抬过 test。**

> **⇒ 正确的北极星不是"单塔 test IC"，而是"对最终集成的边际贡献"**，可近似为
> **`边际贡献 ≈ 单塔IC × (1 − 对其他base的平均相关) ，且不拉低集成 IR`**。
> 这也把用户的约束——"**不大幅降低 IR 的前提下提高 IC**"——落到了实处：
> 对本模型而言，**抬 IR / 降相关 与 抬 IC 同等重要**，因为 IR 是它最稀缺的资源、相关性是它最大的杠杆。

---

## 2. 具体、可核验的目标（GOALS）

所有目标在 **test（days 881–1259）** 上度量；每条给出**验收阈值**。基线为当前 v1：
单塔 `IC 0.05906 / IR 0.846 / spear 0.0576 / pos 0.805 / corr-LGB 0.794 / corr-MLP 0.921`。

### 2.1 目标阶梯

| 编号 | 目标（一句话） | 硬指标（验收） | 主攻方向 |
|---|---|---|---|
| **G1** | **抬 IC，且 IR 不掉**（用户主约束） | 单塔 test **IC ≥ 0.0600**（+0.0009）**且 IR ≥ 0.846** | §4-B refit/正则、§4-A 目标对齐 |
| **G2** | **抬 IR，且 IC 不掉**（补最稀缺资源） | 单塔 test **IR ≥ 0.95**（σ_daily ≤ 0.0619，−11%）**且 IC ≥ 0.0588** | §4-A IR 损失、多 seed/SWA |
| **G3** | **降冗余、增正交**（放大集成价值） | **corr-LGB ≤ 0.80** 保持 **且 corr-MLP 从 0.921 → ≤ 0.88** | §4-C 时序视图去相关 |
| **G4** | **Transformer 家族 blend 上台阶** | raw blend **IC ≥ 0.0610 / IR ≥ 0.90**；中性化后 **IC ≥ 0.0595 & IR > 1.10**（优于现 0.0590/1.10） | §4-C 成员多样化 |
| **G5** | **集成终审（唯一真赢）** | 重训集成后 **IC ≥ 0.0610 且 IR ≥ 1.135**（都不降）；**冲刺：IC > 0.062**（干净闸口） | 以上综合 |

> **σ_daily 换算表**（held IC=0.0588，用于把"抬 IR"翻译成"每日 IC 波动要降多少"）：
> IR≥0.90 → σ≤0.0653（−6%）｜IR≥0.95 → σ≤0.0619（−11%）｜IR≥1.00 → σ≤0.0588（−16%）｜IR≥1.10 → σ≤0.0535（−23%）。

### 2.2 单次改动的 accept / reject 规则（避免"抬 IC 砸 IR"的假进步）
- **接受**：满足其一——(a) `ΔIC ≥ +0.0005 且 ΔIR ≥ −0.02`；或 (b) `ΔIR ≥ +0.05 且 ΔIC ≥ −0.0003`；或 (c) `Δcorr-MLP ≤ −0.02 且 IC、IR 均不降`。
- **拒绝**：`IC 与 IR 同降`，或 `IR 跌幅 > 0.03 而 IC 未显著上升`。
- 单塔 IC 的"显著"以 **≥0.0005**（≈日 IC 均值一个标准误量级）为准，低于此视为噪声，需靠多 seed 复现确认。

### 2.3 集成终审（每个候选改动的最后一关，**决定去留**）
> 单塔数字只是中间量。**任何改动都要把新的 Transformer 预测喂回 `ML_ensemble/scripts/run_ensemble_v2.py` 重训集成**，看 G5：集成 test IC/IR 是否**不降且最好上升**。
> 记录**四元组**：`(单塔IC, 单塔IR, corr-LGB, corr-MLP)` + `(集成IC, 集成IR)`。一个单塔 IC 涨但与 LGB/MLP 更相关的改动，可能对集成是负的——**以集成 delta 为准**。

### 2.4 部署硬边界（方案必须落在框内，否则标注"破坏性变更")
- **K=32 冻结**、**213-特征 schema 冻结**（`submit/predict.py` 按 `feature_artifacts.json` 对齐）。任何需要 K>32 或新特征的方案 = 需重训全链路 + 改推理面板构造，必须显式标注成本。
- 8 seeds × 2.2MB、有 CPU 回退：**容量还有很大余量**（可放心加 seed / 适度调宽），但"加深加宽"已证无益，别把预算花在这。

---

## 3. 已证伪 / 不要重复（两级，详见 [`transformer_ineffective.md`](../artifacts/notes/transformer_ineffective.md) 与 [`ensemble_ineffective.md`](../artifacts/notes/ensemble_ineffective.md)）

**Transformer 级**：横截面注意力(过拟合)、慢 EMA decay0.999(拖后腿)、K=48/64(IC 更低)、group-g 嵌入(双降)、`ICW=3.0 + ~4天/批`(训练崩)、pred 后处理[EMA平滑/winsor/rank/gauss](均更差)、PCA 中性化(毁 IC)、稳定倾斜式中性化(更差)、valid 调 blend 权重(封顶 IC)。
**集成级**：valid-优化权重(过拟合堆到 MLP)、full 中性化 α=1(IC 破线)、掺线性基(拖 IC)、rank/percentile 基替代 z-score(丢量级、掉 IC)、blend 前 winsor/clip(掉 IC)、full PCA 中性化(毁 alpha)、最终信号跨日 EMA 平滑(净零且增泄漏面)。

**共性教训**（贯穿全篇）：① 任何"逐日再压缩/再变换预测"的后处理都在**削 Pearson 奖励的量级信息**——别碰；② 任何"在 valid 上择优/调权"都会被 valid→test 漂移反噬——**多样性用固定 inverse-corr 权重**；③ 提升来自"**新的、正交的、且泛化到 881+ 的信号**"，不来自"更强地拟合旧分布"。

---

## 4. 优化路线（按优先级；每条：为什么｜怎么做｜对应目标｜影响/工作量/风险｜怎么量）

> 优先级排序原则：**先打最稀缺的 IR 与最大的杠杆（正交性），再打公认瓶颈（gap），最后才是归纳偏置与特征**——因为"加强单塔拟合"已被反复证伪。

### Tier A — 直接优化 IR / 对齐评测目标（最高优先：正中用户约束与 §1.3-②）

**A1　IR-aware / 排序对齐损失（这次做对版）** → G1+G2，且产出去相关成员(G4)
- **为什么**：训练用逐点 `smooth_l1(y_xs)`（`run_transformer.py:147`），评测却是**逐日横截面相关**——目标错配，且逐点损失完全不管"每天 IC 的稳定性"（正是 IR 的定义）。上次横截面损失失败（`ICW=3.0`、~4 天/批）是**权重过强 + 每步天数太少致梯度噪声**，不是目标错。
- **怎么做**（安全配方，逐项都可消融）：
  1. **整天大截面**：`DAYBATCH=1`，每步聚合**多天**（≥8–16 天，别再用 4 天），每天在 ~1000+ 标的上算 per-day **Pearson 或 CCC**（`run_transformer_v2.py:166 corr_loss` 已有雏形）。
  2. **显式惩罚 IR 的分母**：损失 = `−mean_d(IC_d) + λ·std_d(IC_d)`（Sharpe/IR 的可微代理）——直接压 σ_daily，直击 G2。
  3. **中等权重 + warmup + 逐点锚点**：`loss = smooth_l1 + w·L_rank`，`w∈[0.1,0.5]`，前若干 epoch 线性升 w，保留逐点项防塌缩。
  4. **更贴 IC 的替代**：可微 Spearman / soft-rank（`SoftRank / NeuralSort / fast-soft-sort`）做 listwise 排序损失，天然对齐秩相关且比硬 corr 平滑。
- **影响 高（唯一同时直击 IC 与 IR 的杠杆）｜工作量 中｜风险 中**（前车之鉴：配错会崩——务必小 w + 多天/批 + warmup）。
- **量**：单塔 IC/IR/σ_daily；**即便 solo IC 不涨，只要 IR 涨或与逐点塔去相关，就进 §4-C 的 blend 池**。

**A2　多 seed + 快照/SWA 集成降方差 → 抬 IR** → G2（便宜、低风险）
- **为什么**：IR=IC/σ_daily，**多 seed 平均直接降 σ_daily**（报告已述"8-seed 抬 IR"）。死路是**慢 EMA(0.999)**，与之不同的是：**SWA（对 valid 峰值附近 epoch 4–8 权重做平均）/ snapshot ensembling / 快 EMA(0.99~0.999 之外更快档)**。当前 seed 在 ep4–7 达峰后骤降——正好适合在峰区做权重平均。
- **怎么做**：把部署的 `NSEED=8` 试到 12–16 看 IR 边际；叠加 SWA（cosine restart 收集快照）。
- **影响 中（稳态抬 IR）｜工作量 低｜风险 低**。**量**：σ_daily 是否下降、IR 是否上台阶、集成 IR delta。

**A3　波动日归一 / 样本再加权** → G2
- **为什么**：某些日的横截面天然更难（σ_daily 被少数坏日撑大）。对每日损失做**规模归一**或对高噪声日降权，能压 IR 分母。
- **怎么做**：day-batched 下按 `1/该日损失尺度` 或按历史该 regime 的日 IC 方差反比加权（注意别用未来信息，用滚动估计）。
- **影响 低-中｜工作量 低｜风险 低（可能轻微降 IC，需 accept 规则把关）**。

### Tier B — 缩小 valid→test gap（公认瓶颈；抬 IC 的主路径）

**B1　refit-on-recent 设为默认** → G1（已证最佳单基，部署已这么做）
- **为什么**：`v3_refit880`（在 train+valid ≤880 上重训）是**最佳单基**（test IC 0.05947>0.05906），部署也是"训练全部有标签数据"。当前研究默认只训 ≤760。生产拿不到 test，就该默认吃满近期数据。
- **怎么做**：v2 base 默认 `REFIT=1`（`run_transformer_v2.py:61,193`），早停 holdout **用最近若干天**而非随机对（更贴 test 的时间结构）。
- **影响 中｜工作量 低（开关已在）｜风险 低**。**量**：test IC/IR + 集成 delta。

**B2　recency-weighted loss（时间指数衰减）** → G1+G2
- **为什么**：test 在 881+，训练分布偏旧且存在 regime 漂移。给近期交易日更高样本权重，让拟合重心贴近 test。经典、便宜、**不在死路清单**。
- **怎么做**：`w(day)=γ^{(D_end−day)}` 或线性衰减，乘到 `run_transformer.py:147` 的 loss；小范围扫 γ。可与 A3 合并实现。
- **影响 中｜工作量 低｜风险 低**。

**B3　purged/embargoed walk-forward + 扩窗多折集成** → G1+G4
- **为什么**：单一 760 切点既浪费数据又给不出稳健早停；**多个"在不同截止日结束的扩张窗口"模型**既正则化又产出**去相关成员**（喂 §4-C）。purge/embargo（López de Prado）避免窗口边界的时间泄漏。
- **怎么做**：截止日 ∈ {700,760,820,880} 各训一版，折间 inverse-corr 融合；每折间留 embargo（≥K 天）。
- **影响 中｜工作量 中｜风险 低**。**量**：折融合 test IC/IR、成员间相关。

**B4　针对 gap 的正则组合（容量非瓶颈 → 该往正则走）** → G1+G2
- **为什么**：clean≤760 在 ep~6 达峰即过拟合；报告明言"deeper/wider hurts"。该做的是**更强正则**而非更大模型。
- **怎么做**（逐个消融）：**feature dropout**（整特征随机置零，抗 regime，**不在死路清单**）、输入噪声/mixup（时间维或同日标的间）、启用 stochastic depth（`SD` 默认 0，`run_transformer_v2.py:53`）、加大 `WD`、**适度缩小 d/nl**、更早停。
- **影响 中｜工作量 低｜风险 低**。

**B5（进阶）　分布漂移正则** → G1
- **为什么**：直接对"train 与 recent 的表征差异"施压，逼模型学 regime-invariant 表征。
- **怎么做**：DANN 式——加一个判别"样本属于旧/新 regime"的小头，主干对它做梯度反转惩罚；或对 batchnorm/统计做 test-time 适配（谨慎，防泄漏）。
- **影响 中｜工作量 中-高｜风险 中（易过设计，先做 B1–B4）**。

### Tier C — 提升对集成的边际价值（去相关；本仓库唯一兑现过 test 增益的路径）

**C1　降低与 MLP 的冗余（0.921）：喂 Transformer "MLP 看不到的东西"** → G3（**新洞见，高价值**）
- **为什么**：Transformer 与 MLP 都是"同一张 213 CS 特征表上的神经网络"，故 0.921 高相关、彼此冗余；集成因此把 MLP 降权。Transformer 的**独有价值是时序**——但当前它其实也在读"已经被展开成 lag/mom/roll 的表格特征"，时序独特性被稀释。让它更"纯时序"就能与 MLP 拉开。
- **怎么做**（K=32 冻结内可实现，非破坏性）：
  - 给时间编码器**侧重原始序列视图**：K 日的 `y` 自相关序列、价量 `prc/vol` 的**日间差分/收益序列**，弱化那些"已是横截面快照"的因子；或用**双分支**（时序分支 + 表格快照分支）并在损失里鼓励时序分支携带增量信息。
  - 训练时加**去相关正则**：penalize `corr(pred_xfmr, pred_mlp_detached)`（用已训好的 MLP 预测作参照），显式把 Transformer 推离 MLP 子空间。
- **影响 中-高（直接抬集成边际贡献）｜工作量 中｜风险 中（别把 IC 也一起推没了——用 accept 规则 c 把关）**。**量**：corr-MLP、集成 IC/IR delta。

**C2　系统化去相关成员池 + 固定 inverse-corr 权重** → G4
- **为什么**：唯一兑现过增益的是多样性 blend（0.0591→0.0602，`build_v2_final.py`）；而"valid 调权"是死路。增益来自**加入正交成员**。
- **怎么做**：成员来源要"目标/数据/读出/视图"真不同——A1 的 rank/IR-loss 塔、B3 的时间折、C1 的纯时序塔、不同 K、**特征 bagging**（随机子集）、last-token-only vs pool-only 读出。权重用 inverse-correlation（`build_v2_final.py:55`），**永不 valid 调权**。
- **影响 中-高（有实证）｜工作量 中｜风险 低**。**量**：raw blend IC≥0.0610、成员两两相关分布。

**C3　rank-loss 成员一箭双雕**：A1 的排序塔与逐点塔天然去相关，solo 成败都应进 C2 池——这是 A 与 C 的协同点。

### Tier D — 归纳偏置 / 正确性（便宜、原则性；报告证实 time-bias/multitask/dual-readout 有用，**别加深加宽**）

**D1　补时间维 padding mask** → 正确性，可能在悄悄吃信号
- **为什么**：`build_panel` 用 `np.zeros` 稠密填充缺失 (标的,日) 格（`run_transformer.py:33`），K 窗口对上市晚/停牌标的会混入**伪造全零"假日"**，而时间自注意力**没有 key_padding_mask**（`run_transformer.py:122-126`、`Block.forward:58`）。v3 已有 `present` 张量（`run_transformer_v3.py:179`）却没用于时间编码器。
- **怎么做**：把 `present[inst, win]` 作为 `key_padding_mask` 传入时间注意力，pooling 对缺失位置置零权重。先量"窗口零填充占比"估收益上界。
- **影响 中｜工作量 低｜风险 低**。

**D2　卷积改因果** → 去掉窗口内一步"偷看未来"
- **为什么**：`Conv1d(...,3,padding=1,groups=d)`（`run_transformer.py:73`）对称 padding 让位置 t 混入 t+1；虽全在 ≤day t 窗口内非标签泄漏，但每个非末位 token 掺了窗口内未来、再被 pool 读出。
- **怎么做**：左 pad `k−1`、丢右侧；或直接消融该 stem。**影响 低-中｜工作量 低｜风险 低**。

**D3　退化的时间偏置 → per-head 相对位置偏置 / RoPE** → 归纳偏置
- **为什么**：现偏置 `−softplus(bias)·|dist[-1]−dist|`（`run_transformer.py:56-62`）对**所有 query 行施加同一套**"偏好靠近末日的 key"，既非相对 `|i−j|`、也不随 query 变化、且**全 head 共用一个标量**——偏置退化。
- **怎么做**：换 **per-head ALiBi（`|i−j|`+每头斜率）** 或 **RoPE**；至少给每头一个可学衰减。**影响 中｜工作量 低-中｜风险 低**。

**D4　多任务头/dual-readout 保留 + 权重消融**：报告证实其有用；仅微调 `sign/mag` 的 0.3/0.3 权重（`run_transformer.py:147`）做小幅消融，别删。

### Tier E — 特征 / 标签工程（结合已知特征族；注意 213-schema 冻结契约）

**E1　训练标签的秩/稳健化**（**注意：训练目标 ≠ pred 后处理**）→ G1
- **为什么**：pred 后处理做 rank/winsor 是**死路**（丢量级）；但把**训练目标**换成 rank/uniform 或 winsor 的 `y_xs`，是改变优化几何，可能更抗离群、更贴 Spearman。
- **怎么做**：主头目标试 `rank(y_xs)` 或 winsor(±3σ) 版本作辅助头/或主损失小权重项；与 A1 排序损失同源，可合并试。**影响 低-中｜工作量 低｜风险 低**。

**E2　轻量特征选择 / 门控**（**别上重的 feature-attention**）→ G1+G3
- **为什么**：213 特征里 90 个是 top-30 的 lag/mom/roll 展开 + 12 PCA，**冗余度高**；regime 不稳的特征会放大 gap。但"横截面注意力"已证过拟合——**只能轻量**。
- **怎么做**：学习 per-feature 缩放（`nn.Parameter` gate）或对输入 `Linear(213→d)` 加 group-lasso/dropout 做隐式选择；配 B4 的 feature-dropout。**影响 中｜工作量 低-中｜风险 中（易过拟合，强正则）**。

**E3　不要再碰 group 条件化**：`grp_te/grp_ylag_mean/grp_xbest_mean` 已作为特征进模型；`group-g 嵌入`已证双降。分组信息交给**下游中性化**处理即可。

### Tier F — 工程卫生（非信号，纯效率/可复现）
- `torch.cuda.amp` → `torch.amp`（去 warning）；Ampere+ 上试 **bf16**（比 fp16 稳）；默认 `SAVESEEDS=1` 存 per-seed 预测矩阵，离线 blend 免重训；用 `configs/` 配置化、`weights/` 落盘 checkpoint。

---

## 5. 推荐执行顺序（一步步，每步带验收数字）

> 每一步做完都跑 **§2.3 集成终审**；接受/拒绝按 **§2.2 规则**；证伪追加进 notes。

1. **地基（D1+D2+B1+B2）** → 冻结为 `Transformer/v2` 的 `v2_base`。
   *验收*：单塔 **IC ≥ 0.0591 且 IR ≥ 0.88**（先把 IR 从 0.846 抬起来），集成不降。
2. **A1 IR-aware/排序损失** 单训一版 → *验收 G2*：**IR ≥ 0.95 且 IC ≥ 0.0588**；无论 solo 成败都收进 blend 池。
3. **A2 seeds↑ + SWA** → *验收*：σ_daily 下降、IR 再上一档。
4. **C1 纯时序去相关塔** → *验收 G3*：**corr-MLP ≤ 0.88**，集成 IR/IC 不降。
5. **C2 组 inverse-corr blend**（成员={v2_base, rank塔, 时序塔, K48, feature-bag}）→ *验收 G4*：**raw IC ≥ 0.0610 / IR ≥ 0.90**。
6. **B3 扩窗多折 + B4 正则** 压 gap，折模型回灌 C2 → *验收 G1*：**单塔/blend IC ≥ 0.0600**。
7. **D3 / E1-E2 / F 精修** → 收尾。
8. **集成终审汇总（G5）**：重训 `run_ensemble_v2` → 目标 **IC ≥ 0.0610 且 IR ≥ 1.135**（都不降）；冲刺 **IC > 0.062** 的干净闸口。

---

## 6. 度量与记录规范 + 诚实边界

- **统一口径**：同一 split、`common.daily_ic`；每个 idea 记**六元组** `(单塔IC, IR, spear, pos, corr-LGB, corr-MLP)` + `(集成IC, 集成IR)` 一行进 `Transformer/v2/metrics/leaderboard_v2.csv`。
- **决策以 test + 集成 delta 为准**；任何"在 valid 上择优/调权"的读数都要 test 复核（本问题病根就是 valid 高估 test）。
- 证伪的追加进 [`transformer_ineffective.md`](../artifacts/notes/transformer_ineffective.md) / [`ensemble_ineffective.md`](../artifacts/notes/ensemble_ineffective.md)（死路只踩一次）。
- **诚实边界**：v3-lineage 单塔的**架构类**改动已基本穷尽，单塔 test IC 逼近 ~0.059 的信号墙，单项现实增益多在**千分之几 IC**。**最可能赢的组合不是"更强的单塔"，而是"更稳(IR↑) + 更正交(corr-MLP↓) + refit-on-recent + 排序对齐"**——这恰好满足"**不降 IR 还要抬 IC**"的约束：把力气压在 **IR 与正交性**上，IC 通过**集成**顺势兑现。**不要再把预算投在"更大/更深/更花"的模型或任何"逐日再压缩预测"的后处理上。**
