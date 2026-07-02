"""
New-feature build for EG: ~115 economically-motivated factor IDEAS built only from
prc1..prc5 / vol0 (+ g, x_0..x_7, x_60 as conditioning/orthogonalization helpers),
each pushed through a uniform 3-round optimization ladder:

  v1 raw            -> per-day cross-sectional z-score of the causal signal
  v2 processing     -> a category-specific statistical treatment (robust z / EWMA
                       smoothing / rank transform / decile-bucket smoothing)
  v3 optimized      -> a category-specific structural treatment (group-g
                       neutralization / vol-residualization / orthogonalization
                       against the x_0..x_7 reversal cluster / decile+neutral combo)

Rules enforced:
  - causal only: all temporal ops use shift(L>=0) within instrument_id; all
    cross-sectional ops use only same-day rows. Nothing at t+1 is ever read.
  - same idea at different windows (k=1,2,3,5 momentum etc.) = ONE idea, many
    "variants"; only the FINAL v3 column of each variant is written to the output
    factor library (v1/v2 exist only to document the optimization ladder's IC).
  - decisions (sign, which version is "best") are made on TRAIN IC only (day<=760);
    VALID IC (761-880) is reported as a secondary stability check; TEST (881-1259)
    is never touched here (reserved for the downstream model retrain).
  - cross-idea correlation pruning: build one "representative" v3 series per idea,
    correlate on train+valid, greedily keep by |train IC| descending, drop an idea
    outright (all its variants) if it correlates > 0.8 with an already-kept idea.

Outputs:
  artifacts/new_factors.parquet       day, instrument_id + surviving idea v3 cols
  artifacts/new_factor_list.json      list of surviving column names
  artifacts/notes/new_factors_ic.json full per-idea/version/variant IC ledger
    (consumed by tools/write_new_factors_report.py to render the markdown doc)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/eg_model")
RAW = ROOT / "artifacts" / "panel_raw.parquet"
OUT_PARQUET = ROOT / "artifacts" / "new_factors.parquet"
OUT_LIST = ROOT / "artifacts" / "new_factor_list.json"
OUT_LEDGER = ROOT / "artifacts" / "notes" / "new_factors_ic.json"
TRAIN_END, VALID_END = 760, 880
EPS = 1e-9
CORR_THRESH = 0.80

t_start = time.time()
print("[nf] loading raw panel", flush=True)
cols = (["day", "instrument_id", "g", "vol0", "y", "x_60"]
        + [f"prc{i}" for i in range(1, 6)] + [f"x_{i}" for i in range(8)])
df = pd.read_parquet(RAW, columns=cols)
df = df.sort_values(["instrument_id", "day"]).reset_index(drop=True)
print(f"[nf] {len(df):,} rows, {df['instrument_id'].nunique()} instruments, {df['day'].nunique()} days", flush=True)

inst = df["instrument_id"]
day = df["day"]
grp = df["g"]
day_vals = day.to_numpy()
y_vals = df["y"].to_numpy(dtype="float64")

# ----------------------------------------------------------------------------
# fast vectorised per-day IC (train / valid) via np.add.reduceat
# ----------------------------------------------------------------------------
def _groups(mask):
    idx = np.where(mask)[0]
    d = day_vals[idx]
    order = idx[np.argsort(d, kind="stable")]
    d_sorted = day_vals[order]
    starts = np.r_[0, np.flatnonzero(np.diff(d_sorted)) + 1]
    return order, starts

tr_order, tr_starts = _groups(day_vals <= TRAIN_END)
va_order, va_starts = _groups((day_vals > TRAIN_END) & (day_vals <= VALID_END))
all_order, all_starts = _groups(np.ones(len(day_vals), dtype=bool))
all_ends = np.r_[all_starts[1:], len(all_order)]


def _fast_ic(v: np.ndarray, order: np.ndarray, starts: np.ndarray) -> float:
    vo = v[order]; yo = y_vals[order]
    n = np.diff(np.r_[starts, len(order)]).astype("float64")
    def red(a): return np.add.reduceat(a, starts)
    sv, sy = red(vo), red(yo)
    svv, syy, svy = red(vo * vo), red(yo * yo), red(vo * yo)
    mv, my = sv / n, sy / n
    cov = svy / n - mv * my
    varv = svv / n - mv * mv
    vary = syy / n - my * my
    denom = np.sqrt(np.clip(varv * vary, 1e-18, None))
    corr = np.where(denom > 1e-9, cov / denom, np.nan)
    return float(np.nanmean(corr))


def train_ic(s) -> float:
    return _fast_ic(np.asarray(s, dtype="float64"), tr_order, tr_starts)


def valid_ic(s) -> float:
    return _fast_ic(np.asarray(s, dtype="float64"), va_order, va_starts)


# ----------------------------------------------------------------------------
# causal transform toolkit
# ----------------------------------------------------------------------------
def ishift(s, L):
    return s.groupby(inst, sort=False).shift(L)


def rmean(s, w, minp=None):
    return s.groupby(inst, sort=False).rolling(w, min_periods=minp or max(2, w // 2)).mean().reset_index(level=0, drop=True)


def rstd(s, w, minp=None):
    return s.groupby(inst, sort=False).rolling(w, min_periods=minp or max(3, w // 2)).std().reset_index(level=0, drop=True)


def rmax(s, w, minp=None):
    return s.groupby(inst, sort=False).rolling(w, min_periods=minp or max(2, w // 2)).max().reset_index(level=0, drop=True)


def rmin(s, w, minp=None):
    return s.groupby(inst, sort=False).rolling(w, min_periods=minp or max(2, w // 2)).min().reset_index(level=0, drop=True)


def rsum(s, w, minp=None):
    return s.groupby(inst, sort=False).rolling(w, min_periods=minp or max(2, w // 2)).sum().reset_index(level=0, drop=True)


def rskew(s, w):
    return s.groupby(inst, sort=False).rolling(w, min_periods=max(6, w // 2)).skew().reset_index(level=0, drop=True)


def rkurt(s, w):
    return s.groupby(inst, sort=False).rolling(w, min_periods=max(6, w // 2)).kurt().reset_index(level=0, drop=True)


def ewm(s, halflife):
    return s.groupby(inst, sort=False).transform(lambda v: v.ewm(halflife=halflife, min_periods=2).mean())


def rcov(a, b, w):
    return rmean(a * b, w) - rmean(a, w) * rmean(b, w)


def rvar(a, w):
    return rcov(a, a, w)


def rcorr(a, b, w):
    return rcov(a, b, w) / (np.sqrt(np.clip(rvar(a, w) * rvar(b, w), 1e-18, None)) + EPS)


def cs_z(s):
    g = s.groupby(day, sort=False)
    return (s - g.transform("mean")) / (g.transform("std") + EPS)


def cs_rank(s):
    r = s.groupby(day, sort=False).rank(pct=True)
    return (r - 0.5) * 2.0


def cs_robust_z(s):
    med = s.groupby(day, sort=False).transform("median")
    mad = (s - med).abs().groupby(day, sort=False).transform("median")
    return (s - med) / (1.4826 * mad + EPS)


def group_neutral(s, alpha=1.0):
    gm = s.groupby([day, grp], sort=False).transform("mean")
    return cs_z(s - alpha * gm)


def group_rank(s):
    r = s.groupby([day, grp], sort=False).rank(pct=True)
    return (r - 0.5) * 2.0


def decile_bucket_smooth(s, n=10):
    rk = s.groupby(day, sort=False).rank(pct=True, method="first")
    bucket = np.minimum((rk.to_numpy() * n).astype(int), n - 1)
    tmp = pd.DataFrame({"day": day.to_numpy(), "b": bucket, "s": s.to_numpy()})
    bmean = tmp.groupby(["day", "b"], sort=False)["s"].transform("mean")
    return pd.Series(bmean.to_numpy(), index=s.index)


def residualize_single(s, x):
    sm = s.groupby(day, sort=False).transform("mean")
    xm = x.groupby(day, sort=False).transform("mean")
    cov = ((s - sm) * (x - xm)).groupby(day, sort=False).transform("mean")
    varx = ((x - xm) ** 2).groupby(day, sort=False).transform("mean")
    beta = cov / (varx + EPS)
    return cs_z((s - sm) - beta * (x - xm))


_X07 = [cs_z(df[f"x_{i}"]).fillna(0.0) for i in range(8)]


def residualize_x07(s):
    X = np.column_stack([np.ones(len(df))] + [x.to_numpy() for x in _X07])
    yv = s.to_numpy(dtype="float64")
    Xo, yo = X[all_order], yv[all_order]
    out_o = np.empty_like(yo)
    for st, en in zip(all_starts, all_ends):
        Xd = Xo[st:en]; yd = yo[st:en]
        beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        out_o[st:en] = yd - Xd @ beta
    out = np.empty_like(yv)
    out[all_order] = out_o
    return cs_z(pd.Series(out, index=s.index))


def fillz(s):
    return s.fillna(0.0)


# ----------------------------------------------------------------------------
# base causal series
# ----------------------------------------------------------------------------
print("[nf] building base causal series", flush=True)
prc1, prc2, prc3, prc4, prc5 = (df[f"prc{i}"] for i in range(1, 6))
prc_mean = (prc1 + prc2 + prc3 + prc4 + prc5) / 5.0
vol0 = df["vol0"]
x60 = df["x_60"]

ret = prc2 - ishift(prc2, 1)
overnight = prc1 - ishift(prc2, 1)
intraday = prc2 - prc1
rng = prc4 - prc3
close_vwap = prc2 - prc5
open_vwap = prc1 - prc5
clv = ((prc2 - prc3) / (rng.replace(0, np.nan))).clip(-3, 3)
dvol0 = vol0 - ishift(vol0, 1)
amihud = ret.abs() / (vol0.clip(lower=0.02) + 0.05)
mkt_ret = ret.groupby(day, sort=False).transform("mean")
rvol10 = rstd(ret, 10)
rvol20 = rstd(ret, 20)
ofi_raw = np.sign(ret) * vol0
tindex = df.groupby("instrument_id", sort=False).cumcount().astype("float64")
tindex = pd.Series(tindex.to_numpy(), index=df.index)
d1_5 = prc1 - ishift(prc1, 5)
d2_5 = prc2 - ishift(prc2, 5)
d3_5 = prc3 - ishift(prc3, 5)
d4_5 = prc4 - ishift(prc4, 5)
d5_5 = prc5 - ishift(prc5, 5)
marks_mom5 = pd.concat([d1_5, d2_5, d3_5, d4_5, d5_5], axis=1)
d1_1 = prc1 - ishift(prc1, 1)
d3_1 = prc3 - ishift(prc3, 1)
d4_1 = prc4 - ishift(prc4, 1)
d5_1 = prc5 - ishift(prc5, 1)
common1 = (ret + d1_1 + d3_1 + d4_1 + d5_1) / 5.0
print(f"[nf] base series ready in {time.time()-t_start:.0f}s", flush=True)

# ----------------------------------------------------------------------------
# recipes: category -> (v2 func, v2 label), (v3 func, v3 label)
# ----------------------------------------------------------------------------
RECIPES = {
    "reversal": {
        "v2": lambda s: (decile_bucket_smooth(s, 10), "decile-bucket smoothing (classification, n=10)"),
        "v3": lambda s2: (residualize_single(group_neutral(s2, 1.0), rvol10), "group(g)-neutral + vol10-residualized (regression)"),
    },
    "vol_liquidity": {
        "v2": lambda s: (cs_robust_z(s), "robust (median/MAD) z-score"),
        "v3": lambda s2: (group_neutral(cs_robust_z(np.sign(s2) * np.log1p(s2.abs())), 1.0), "log1p + robust z, group(g)-neutralized"),
    },
    "volume": {
        "v2": lambda s: (cs_z(ewm(s, 5)), "EWMA(halflife=5)-smoothed z-score"),
        "v3": lambda s2: (group_neutral(cs_z(ewm(s2, 5)), 1.0), "EWMA-smoothed, group(g)-relative liquidity-neutralized"),
    },
    "technical": {
        "v2": lambda s: (cs_rank(s), "cross-sectional rank transform"),
        "v3": lambda s2: (residualize_x07(cs_rank(s2)), "rank, residualized vs x_0..x_7 reversal cluster (regression)"),
    },
    "interaction": {
        "v2": lambda s: (decile_bucket_smooth(s, 10), "decile-bucket smoothing (classification, n=10)"),
        "v3": lambda s2: (group_neutral(decile_bucket_smooth(s2, 10), 1.0), "decile-bucket + group(g)-neutralized"),
    },
}


def run_variant(idea_key, variant_key, raw, category):
    """raw: un-normalised causal pd.Series -> dict of version results."""
    raw = fillz(raw.replace([np.inf, -np.inf], np.nan))
    v1 = fillz(cs_z(raw))
    ic1 = train_ic(v1.to_numpy())
    if ic1 < 0:            # canonical sign: positive train IC
        raw = -raw
        v1 = -v1
        ic1 = -ic1
    vic1 = valid_ic(v1.to_numpy())

    v2_raw, v2_label = RECIPES[category]["v2"](raw)
    v2 = fillz(v2_raw if category in ("reversal", "interaction") else cs_z(v2_raw))
    ic2 = train_ic(v2.to_numpy()); vic2 = valid_ic(v2.to_numpy())

    v3_raw, v3_label = RECIPES[category]["v3"](raw if category not in ("reversal", "interaction") else v2)
    v3 = fillz(v3_raw)
    ic3 = train_ic(v3.to_numpy()); vic3 = valid_ic(v3.to_numpy())

    col = f"{idea_key}_{variant_key}" if variant_key else idea_key
    return {
        "idea": idea_key, "variant": variant_key, "category": category, "col": col,
        "v1": v1, "v2": v2, "v3": v3,
        "versions": [
            {"name": "v1_raw", "formula": "per-day cross-sectional z-score", "train_ic": round(ic1, 5), "valid_ic": round(vic1, 5)},
            {"name": "v2_processed", "formula": v2_label, "train_ic": round(ic2, 5), "valid_ic": round(vic2, 5)},
            {"name": "v3_optimized", "formula": v3_label, "train_ic": round(ic3, 5), "valid_ic": round(vic3, 5)},
        ],
    }


# ============================================================================
# IDEA DEFINITIONS  (idea key, category, zh/en rationale, variants[(vkey, raw_fn)])
# ============================================================================
IDEAS = []


def add(idea, category, zh, en, variants):
    IDEAS.append(dict(idea=idea, category=category, zh=zh, en=en, variants=variants))


# ---- A. Reversal / momentum core -------------------------------------------------
add("reversal", "reversal", "短周期反转：过度反应后价格回吐", "short-horizon reversal: overreaction mean-reverts",
    [("k1", lambda: -ret), ("k2", lambda: -(prc2 - ishift(prc2, 2))),
     ("k3", lambda: -(prc2 - ishift(prc2, 3))), ("k5", lambda: -(prc2 - ishift(prc2, 5)))])
add("momentum", "reversal", "中长周期动量：趋势延续", "medium/long-horizon momentum: trend continuation",
    [("k10", lambda: prc2 - ishift(prc2, 10)), ("k20", lambda: prc2 - ishift(prc2, 20)),
     ("k40", lambda: prc2 - ishift(prc2, 40))])
add("momentum_term_spread", "reversal", "动量期限利差：长动量减短动量，捕捉加速/减速", "momentum term spread: long minus short momentum captures accel/decel",
    [(None, lambda: (prc2 - ishift(prc2, 20)) - (prc2 - ishift(prc2, 5)))])
add("reversal_risk_adj", "reversal", "风险调整反转：反转信号除以短期波动率（类夏普）", "risk-adjusted reversal: reversal scaled by short-horizon vol (Sharpe-like)",
    [(None, lambda: -ret / (rvol10 + EPS))])
add("reversal_consistency", "reversal", "反转一致性：窗口内下跌天数占比", "reversal consistency: fraction of down-days in window",
    [(None, lambda: rmean((ret < 0).astype(float), 5))])
add("overnight_gap", "reversal", "隔夜跳空：隔夜收益的动量分量", "overnight gap: the overnight-return momentum component",
    [(None, lambda: overnight)])
add("intraday_return", "reversal", "日内收益：日内部分的反转分量", "intraday return: the intraday-segment reversal component",
    [(None, lambda: -intraday)])
add("close_vwap_pressure", "reversal", "收盘相对VWAP压力：收盘价偏离日内均价", "close-vs-VWAP pressure: close deviation from intraday average price",
    [(None, lambda: -close_vwap)])
add("open_vwap_gap", "reversal", "开盘相对VWAP缺口", "open-vs-VWAP gap",
    [(None, lambda: open_vwap)])
add("vwap_dev_momentum", "reversal", "VWAP偏离的动量：close-vwap缺口的5日变化", "VWAP-deviation momentum: 5-day change of the close-vwap gap",
    [(None, lambda: close_vwap - ishift(close_vwap, 5))])
add("overnight_intraday_divergence", "reversal", "隔夜/日内背离：两类收益滚动均值之差", "overnight/intraday divergence: rolling-mean gap between the two return legs",
    [(None, lambda: rmean(overnight, 5) - rmean(intraday, 5))])
add("price_acceleration", "reversal", "价格加速度：收益的一阶差分", "price acceleration: first difference of daily return",
    [(None, lambda: ret - ishift(ret, 1))])
add("idio_reversal_beta", "reversal", "特质反转：剔除60日滚动市场beta后的残差反转", "idiosyncratic reversal: reversal net of a 60-day rolling market beta",
    [(None, lambda: -(ret - rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS) * mkt_ret))])
add("idio_reversal_pca", "reversal", "特质反转(共同分解)：5个价格标记同期变动的共同分量之外的close残差", "idiosyncratic reversal (common/idio split): close move net of the 5-mark common component",
    [(None, lambda: -(ret - common1))])
add("group_rank_reversal", "reversal", "组内反转：反转信号在行业g内部重新排名而非全市场", "within-group reversal: rank the reversal signal inside industry g, not the whole market",
    [(None, lambda: group_rank(-(prc2 - ishift(prc2, 3))))])
add("orthogonal_momentum_x07", "reversal", "正交动量：动量对x_0..x_7反转簇做截面回归后的残差", "orthogonalized momentum: momentum residualized against the x_0..x_7 reversal cluster",
    [(None, lambda: residualize_x07(prc2 - ishift(prc2, 10)))])
add("beta_asym_reversal", "reversal", "非对称贝塔反转：下跌市场beta高于上涨市场beta的名字更容易反转", "asymmetric-beta reversal: names with higher down-market than up-market beta revert more",
    [(None, lambda: -ret * np.sign(rcov(ret.where(mkt_ret < 0, 0.0), mkt_ret.where(mkt_ret < 0, 0.0), 60) - rcov(ret.where(mkt_ret >= 0, 0.0), mkt_ret.where(mkt_ret >= 0, 0.0), 60)))])

# ---- B. Range / volatility / distribution ----------------------------------------
add("intraday_range", "vol_liquidity", "日内区间：High-Low波动幅度", "intraday range: high-low swing amplitude",
    [(None, lambda: rng)])
add("range_pct", "vol_liquidity", "相对区间：区间幅度相对价格水平归一", "relative range: range normalised by the price level",
    [(None, lambda: rng / (prc_mean.abs() + EPS))])
add("close_location_value", "vol_liquidity", "收盘位置值：收盘价在日内区间中的位置(类%K)", "close location value: close's position inside the day's range (Stochastic %K-like)",
    [(None, lambda: clv)])
add("parkinson_vol", "vol_liquidity", "Parkinson波动率：基于区间平方的已实现波动率代理", "Parkinson volatility: realised-vol proxy from squared range",
    [(None, lambda: np.sqrt(rmean(rng ** 2, 10) + EPS))])
add("return_skew", "vol_liquidity", "收益偏度：20日滚动偏度，彩票/尾部偏好折价", "return skew: 20-day rolling skew, lottery/tail-preference discount",
    [(None, lambda: rskew(ret, 20))])
add("return_kurtosis", "vol_liquidity", "收益峰度：20日滚动峰度，尾部风险", "return kurtosis: 20-day rolling kurtosis, tail risk",
    [(None, lambda: rkurt(ret, 20))])
add("downside_vol", "vol_liquidity", "下行波动率：仅用下跌日计算的波动", "downside volatility: vol computed only from down-days",
    [(None, lambda: rstd(ret.where(ret < 0), 20))])
add("upside_vol", "vol_liquidity", "上行波动率：仅用上涨日计算的波动", "upside volatility: vol computed only from up-days",
    [(None, lambda: rstd(ret.where(ret > 0), 20))])
add("vol_of_vol", "vol_liquidity", "波动的波动：短期波动率自身的20日波动", "vol-of-vol: the 20-day volatility of the 10-day realised vol",
    [(None, lambda: rstd(rvol10, 20))])
add("max_range_lottery", "vol_liquidity", "MAX效应：窗口内最大单日区间，彩票股折价", "MAX effect: the largest single-day range in the window, lottery discount",
    [(None, lambda: rmax(rng, 20))])
add("worst_return", "vol_liquidity", "最差收益：窗口内最差单日收益", "worst return: the worst single-day return in the window",
    [(None, lambda: rmin(ret, 20))])
add("best_return", "vol_liquidity", "最佳收益：窗口内最佳单日收益", "best return: the best single-day return in the window",
    [(None, lambda: rmax(ret, 20))])
add("atr_pct", "vol_liquidity", "相对ATR：平均真实波幅相对价格水平", "relative ATR: average true range normalised by price level",
    [(None, lambda: rmean(rng, 14) / (prc_mean.abs() + EPS))])
add("range_skew", "vol_liquidity", "区间偏度：日内区间的滚动偏度", "range skew: rolling skew of the intraday range",
    [(None, lambda: rskew(rng, 20))])
add("compression_expansion", "vol_liquidity", "波动率制度：短窗区间/长窗区间，布林带收窄-放宽", "volatility regime: short/long range ratio, Bollinger squeeze-vs-expansion",
    [(None, lambda: rmean(rng, 5) / (rmean(rng, 20) + EPS))])
add("jump_frequency", "vol_liquidity", "跳跃频率：窗口内|收益|超过2倍波动率的天数占比", "jump frequency: share of days with |return| > 2x rolling vol",
    [(None, lambda: rmean((ret.abs() > 2 * rvol20).astype(float), 20))])
add("choppiness_index", "vol_liquidity", "震荡指数：收益符号翻转频率，趋势vs噪声", "choppiness index: sign-flip frequency of returns, trend vs noise",
    [(None, lambda: rmean((np.sign(ret) != np.sign(ishift(ret, 1))).astype(float), 20))])
add("efficiency_ratio", "vol_liquidity", "考夫曼效率比：净移动/累计绝对移动，趋势效率", "Kaufman efficiency ratio: net move / cumulative absolute move, trend efficiency",
    [(None, lambda: (prc2 - ishift(prc2, 20)).abs() / (rsum(ret.abs(), 20) + EPS))])
add("realized_vol_term_spread", "vol_liquidity", "已实现波动率期限利差：短窗-长窗波动率", "realised-vol term spread: short-window minus long-window vol",
    [(None, lambda: rstd(ret, 5) - rstd(ret, 60))])
add("multi_mark_divergence", "vol_liquidity", "多标记分歧度：5个价格标记5日动量的截面(同标的内)离散度", "multi-mark divergence: dispersion across the 5 marks' own 5-day momenta",
    [(None, lambda: marks_mom5.std(axis=1))])
add("chaikin_volatility", "vol_liquidity", "Chaikin波动率：区间均值的10日变化率", "Chaikin volatility: 10-day rate of change of the mean range",
    [(None, lambda: (rmean(rng, 10) - ishift(rmean(rng, 10), 10)) / (ishift(rmean(rng, 10), 10).abs() + EPS))])
add("hawkes_intensity_proxy", "vol_liquidity", "Hawkes强度代理：收益平方的EWMA，刻画波动的自激聚集", "Hawkes-intensity proxy: EWMA of squared returns, self-exciting vol clustering",
    [(None, lambda: ewm(ret ** 2, 5))])

# ---- C. Volume / turnover / liquidity ---------------------------------------------
add("volume_momentum", "volume", "成交量动量：成交量的一阶变化", "volume momentum: first difference of volume",
    [(None, lambda: dvol0)])
add("volume_trend_shock", "volume", "成交量异动：相对20日均量的偏离", "volume shock: deviation from the 20-day average volume",
    [(None, lambda: vol0 / (rmean(vol0, 20) + EPS) - 1)])
add("volume_acceleration", "volume", "成交量加速度：量变化的二阶差分", "volume acceleration: second difference of volume",
    [(None, lambda: dvol0 - ishift(dvol0, 1))])
add("volume_stability_cv", "volume", "成交量稳定性：20日变异系数", "volume stability: 20-day coefficient of variation",
    [(None, lambda: rstd(vol0, 20) / (rmean(vol0, 20).abs() + EPS))])
add("turnover_persistence_ewma", "volume", "换手持续性：成交量EWMA平滑", "turnover persistence: EWMA-smoothed volume",
    [(None, lambda: ewm(vol0, 10))])
add("volume_price_corr", "volume", "量价相关：收益与成交量变化的20日滚动相关", "volume-price correlation: 20-day rolling corr of return and volume change",
    [(None, lambda: rcorr(ret, dvol0, 20))])
add("volume_shock_persistence_ac", "volume", "量冲击持续性：成交量的1阶自相关(20日窗)", "volume-shock persistence: 20-day rolling AC(1) of volume",
    [(None, lambda: rcorr(vol0, ishift(vol0, 1), 20))])
add("obv_trend", "volume", "OBV趋势：累计带符号成交量的10日动量", "OBV trend: 10-day momentum of cumulative signed volume",
    [(None, lambda: ofi_raw.groupby(inst, sort=False).cumsum() - ishift(ofi_raw.groupby(inst, sort=False).cumsum(), 10))])
add("money_flow_index", "volume", "资金流指标：Chaikin式量价加权流(10日滚动和)", "money flow index: Chaikin-style volume-weighted flow (10-day rolling sum)",
    [(None, lambda: rsum(((prc2 - prc3) - (prc4 - prc2)) / (rng + EPS) * vol0, 10))])
add("volume_rank_momentum", "volume", "成交量排名动量：截面成交量排名的5日变化", "volume-rank momentum: 5-day change in the cross-sectional volume rank",
    [(None, lambda: cs_rank(vol0) - ishift(cs_rank(vol0), 5))])
add("volume_volatility_relation", "volume", "量-波关系：成交量与|收益|的20日滚动相关", "volume-volatility relation: 20-day rolling corr of volume and |return|",
    [(None, lambda: rcorr(vol0, ret.abs(), 20))])
add("turnover_zscore_extreme", "volume", "换手极端度：成交量截面z分数的绝对值(关注度冲击)", "turnover extremeness: |cross-sectional z-score| of volume (attention shock)",
    [(None, lambda: cs_z(vol0).abs())])
add("liquidity_timing", "volume", "流动性择时：成交量与|收益|乘积的10日均值", "liquidity timing: 10-day mean of volume times |return|",
    [(None, lambda: rmean(vol0 * ret.abs(), 10))])
add("gap_size_trend", "volume", "跳空幅度趋势：隔夜跳空绝对值的10日均值，事件代理", "gap-size trend: 10-day mean of |overnight gap|, an event/news proxy",
    [(None, lambda: rmean(overnight.abs(), 10))])
add("volume_price_trend", "volume", "量价趋势指标(VPT)：收益率加权成交量的10日累计", "volume-price trend (VPT): 10-day cumulative return-weighted volume",
    [(None, lambda: rsum(ret / (prc_mean.abs() + EPS) * vol0, 10))])
add("vpin_proxy", "volume", "VPIN代理：|带符号成交量|占总成交量比例，订单流毒性", "VPIN proxy: |signed volume| share of total volume, order-flow toxicity",
    [(None, lambda: rmean(ofi_raw.abs(), 10) / (rmean(vol0.abs(), 10) + EPS))])
add("turnover_illiquidity_spread", "volume", "流动性复合背离：成交量排名与非流动性排名之差", "liquidity composite divergence: volume rank minus illiquidity rank",
    [(None, lambda: cs_rank(vol0) - cs_rank(ret.abs() / (vol0.clip(lower=0.02) + 0.05)))])

# ---- D. Group / beta / technical / trend -------------------------------------------
add("market_beta", "technical", "市场贝塔：60日滚动回归到截面平均收益的系数", "market beta: 60-day rolling regression coefficient on the cross-sectional mean return",
    [(None, lambda: rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS))])
add("idio_vol", "vol_liquidity", "特质波动率：剔除市场beta后残差的60日波动", "idiosyncratic volatility: 60-day vol of the beta-residual return",
    [(None, lambda: rstd(ret - rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS) * mkt_ret, 60))])
add("coskewness", "technical", "协偏度：收益与市场收益平方的60日滚动相关", "coskewness: 60-day rolling corr of return with squared market return",
    [(None, lambda: rcorr(ret, mkt_ret ** 2, 60))])
add("group_relative_volume", "volume", "行业相对成交量：成交量减去(day,g)组内均值", "group-relative volume: volume minus its (day,g) group mean",
    [(None, lambda: vol0 - vol0.groupby([day, grp], sort=False).transform("mean"))])
add("group_relative_vol", "vol_liquidity", "行业相对波动率：20日波动率减去(day,g)组内均值", "group-relative volatility: 20-day vol minus its (day,g) group mean",
    [(None, lambda: rvol20 - rvol20.groupby([day, grp], sort=False).transform("mean"))])
add("relative_strength_vs_group", "technical", "组内外强弱背离：全市场动量排名减去组内动量排名", "cross-group relative strength: market-wide momentum rank minus within-group rank",
    [(None, lambda: cs_rank(prc2 - ishift(prc2, 5)) - group_rank(prc2 - ishift(prc2, 5)))])
add("days_since_high", "technical", "距最高点天数：60日窗口内最近新高以来的天数", "days since high: days elapsed since the most recent 60-day high",
    [(None, lambda: prc2.groupby(inst, sort=False).rolling(60, min_periods=10).apply(lambda a: len(a) - 1 - np.argmax(a), raw=True).reset_index(level=0, drop=True))])
add("distance_from_low", "technical", "距最低点距离：收盘价相对60日最低价的比例", "distance from low: close relative to the 60-day low",
    [(None, lambda: prc2 / (rmin(prc2, 60) + EPS) - 1)])
add("ma_cross", "technical", "均线交叉：10日均线减40日均线", "moving-average cross: 10-day MA minus 40-day MA",
    [(None, lambda: rmean(prc2, 10) - rmean(prc2, 40))])
add("trend_slope", "technical", "趋势斜率：20日窗口内收盘价对时间的回归斜率", "trend slope: 20-day rolling regression slope of close on time",
    [(None, lambda: rcov(prc2, tindex, 20) / (rvar(tindex, 20) + EPS))])
add("bollinger_position", "technical", "布林带位置：收盘价偏离20日均线除以20日标准差", "Bollinger position: close deviation from its 20-day MA over 20-day std",
    [(None, lambda: (prc2 - rmean(prc2, 20)) / (rstd(prc2, 20) + EPS))])
add("rsi_like", "technical", "RSI式相对强弱：14日上涨幅度占总幅度比例", "RSI-like relative strength: 14-day up-move share of total move",
    [(None, lambda: rmean(ret.clip(lower=0), 14) / (rmean(ret.abs(), 14) + EPS))])
add("win_rate", "technical", "胜率：窗口内上涨天数占比", "win rate: share of up-days in the window",
    [(None, lambda: rmean((ret > 0).astype(float), 20))])
add("autocorr_return", "technical", "收益自相关：20日窗口内一阶自相关", "return autocorrelation: 20-day rolling AC(1)",
    [(None, lambda: rcorr(ret, ishift(ret, 1), 20))])
add("trend_efficiency", "technical", "趋势拟合优度：线性趋势解释的价格方差占比", "trend goodness-of-fit: share of price variance explained by a linear trend",
    [(None, lambda: 1 - rvar(prc2 - rmean(prc2, 20), 20) / (rvar(prc2, 20) + EPS))])
add("vwap_efficiency", "technical", "VWAP效率：|收盘-VWAP|相对日内区间", "VWAP efficiency: |close-vwap| relative to the day's range",
    [(None, lambda: close_vwap.abs() / (rng.abs() + EPS))])
add("multi_mark_common_trend", "technical", "多标记共同趋势：5个价格标记5日动量的均值(共同分量)", "multi-mark common trend: the mean of the 5 marks' own 5-day momenta (common component)",
    [(None, lambda: marks_mom5.mean(axis=1))])
add("book_pressure_proxy", "technical", "盘口压力代理：收盘价相对(prc3,prc4)中点的位置", "book-pressure proxy: close's position relative to the (prc3,prc4) midpoint",
    [(None, lambda: (2 * prc2 - prc3 - prc4) / (rng.abs() + EPS))])
add("mark_spread_trend", "technical", "标记价差趋势：5个标记的截面标准差的5减20日均值差", "mark-spread trend: 5-vs-20-day change in the cross-mark std",
    [(None, lambda: rmean(pd.concat([prc1, prc2, prc3, prc4, prc5], axis=1).std(axis=1), 5) - rmean(pd.concat([prc1, prc2, prc3, prc4, prc5], axis=1).std(axis=1), 20))])
add("tick_efficiency", "technical", "价格捕获效率：|净收益|相对日内区间", "tick efficiency: |net return| relative to the day's range",
    [(None, lambda: ret.abs() / (rng.abs() + EPS))])
add("range_asymmetry", "technical", "区间不对称性：上方空间减下方空间", "range asymmetry: upside room minus downside room around the close",
    [(None, lambda: (prc4 - prc2) - (prc2 - prc3))])
add("kyle_lambda", "technical", "Kyle's lambda：|收益|对成交量的60日滚动回归斜率", "Kyle's lambda: 60-day rolling regression slope of |return| on volume",
    [(None, lambda: rcov(ret.abs(), vol0, 60) / (rvar(vol0, 60) + EPS))])
add("impact_decay", "technical", "冲击衰减：当日收益减去其3日EWMA(暂时性冲击分量)", "impact decay: today's return minus its 3-day EWMA (the transient-impact component)",
    [(None, lambda: ret - ewm(ret, 3))])
add("permanent_impact_proxy", "technical", "永久冲击代理：收益的20日EWMA(持续性分量)", "permanent-impact proxy: the 20-day EWMA of return (the persistent component)",
    [(None, lambda: ewm(ret, 20))])
add("reversal_asymmetry", "technical", "反转不对称性：下跌日反转力度减上涨日反转力度", "reversal asymmetry: down-day reversion strength minus up-day reversion strength",
    [(None, lambda: -rmean(ret.where(ret < 0), 10) - rmean(-ret.where(ret > 0), 10))])
add("amihud_illiquidity", "vol_liquidity", "Amihud非流动性：|收益|/成交量", "Amihud illiquidity: |return| / volume",
    [(None, lambda: amihud)])
add("amihud_trend", "vol_liquidity", "非流动性趋势：Amihud指标的10日均值", "illiquidity trend: 10-day mean of the Amihud measure",
    [(None, lambda: rmean(amihud, 10))])
add("signed_amihud", "vol_liquidity", "带符号非流动性：符号(收益)乘以非流动性", "signed illiquidity: sign(return) times illiquidity",
    [(None, lambda: np.sign(ret) * amihud)])
add("illiquidity_momentum", "vol_liquidity", "非流动性动量：Amihud指标的10日变化", "illiquidity momentum: 10-day change in the Amihud measure",
    [(None, lambda: amihud - ishift(amihud, 10))])
add("liquidity_adj_range", "vol_liquidity", "流动性调整区间：日内区间除以成交量", "liquidity-adjusted range: intraday range divided by volume",
    [(None, lambda: rng / (vol0.clip(lower=0.02) + 0.05))])
add("price_vol_elasticity", "vol_liquidity", "价格-成交量弹性：|收益|相对|成交量变化|(类价格冲击)", "price-volume elasticity: |return| relative to |volume change| (a price-impact proxy)",
    [(None, lambda: ret.abs() / (dvol0.abs() + EPS))])

# ---- E. Interaction / conditioning (Task-2 style templates) -----------------------
add("gap_fade", "interaction", "跳空回补：隔夜跳空方向乘以当日日内收益", "gap fade: overnight-gap sign times same-day intraday return",
    [(None, lambda: np.sign(overnight) * intraday)])
add("ofi_lite", "interaction", "订单流失衡代理：收益符号乘以成交量(方向x强度分解)", "OFI-lite: sign(return) times volume (sign x intensity decomposition)",
    [(None, lambda: ofi_raw)])
add("volume_weighted_reversal", "interaction", "成交量加权反转：反转信号乘以成交量截面排名(高关注度更易反转)", "volume-weighted reversal: reversal scaled by the cross-sectional volume rank",
    [(None, lambda: -(prc2 - ishift(prc2, 3)) * cs_rank(vol0))])
add("illiquidity_x_momentum", "interaction", "非流动性x动量：非流动性溢价集中在低流动性名字上的条件化动量", "illiquidity x momentum: momentum conditioned on the illiquidity level",
    [(None, lambda: (prc2 - ishift(prc2, 10)) * cs_rank(amihud))])
add("vol_regime_reversal", "interaction", "波动率制度反转：反转信号乘以波动率截面排名", "vol-regime reversal: reversal scaled by the cross-sectional volatility rank",
    [(None, lambda: -(prc2 - ishift(prc2, 3)) * cs_rank(rvol20))])
add("liquidity_regime_momentum", "interaction", "流动性制度动量：动量信号乘以成交量截面排名", "liquidity-regime momentum: momentum scaled by the cross-sectional volume rank",
    [(None, lambda: (prc2 - ishift(prc2, 10)) * cs_rank(vol0))])
add("divergence_indicator", "interaction", "量价背离：价格上涨但成交量走弱的背离信号", "price-volume divergence: price rising while volume is fading",
    [(None, lambda: (prc2 - ishift(prc2, 10)) * -np.sign(vol0 - ishift(vol0, 10)))])
add("toxicity_weighted_momentum", "interaction", "毒性折价动量：用VPIN代理对动量信号做毒性折价", "toxicity-discounted momentum: momentum discounted by the VPIN-proxy toxicity",
    [(None, lambda: (prc2 - ishift(prc2, 10)) / (1 + rmean(ofi_raw.abs(), 10) / (rmean(vol0.abs(), 10) + EPS)))])
add("sign_intensity_decomp", "interaction", "方向x强度分解：动量方向乘以波动率排名(置信度)", "sign x intensity decomposition: momentum direction times a volatility-rank confidence weight",
    [(None, lambda: np.sign(prc2 - ishift(prc2, 10)) * cs_rank(rvol20))])
add("liquidity_tier_reversal", "interaction", "流动性分层反转：反转信号乘以x_60换手代理排名", "liquidity-tier reversal: reversal scaled by the x_60 turnover-proxy rank",
    [(None, lambda: -(prc2 - ishift(prc2, 3)) * cs_rank(x60))])
add("range_expansion_breakout", "interaction", "区间扩张突破：当日区间显著超过20日均值的突破信号", "range-expansion breakout: today's range materially exceeding its 20-day mean",
    [(None, lambda: (rng > 1.5 * rmean(rng, 20)).astype(float) * np.sign(ret))])
add("open_close_vwap_triangle", "interaction", "开-收-VWAP三角：开盘与收盘相对VWAP偏离的乘积(共振/背离)", "open-close-VWAP triangle: product of open's and close's deviations from VWAP",
    [(None, lambda: open_vwap * close_vwap)])
add("amihud_vol_regime_interaction", "interaction", "非流动性x波动率制度：两类风险排名的协同", "illiquidity x vol-regime: co-movement of the two cross-sectional risk ranks",
    [(None, lambda: cs_rank(amihud) * cs_rank(rvol20))])
add("reversal_x_beta", "interaction", "反转x贝塔：反转信号乘以市场贝塔排名(高贝塔名字反转特性不同)", "reversal x beta: reversal scaled by the cross-sectional market-beta rank",
    [(None, lambda: -(prc2 - ishift(prc2, 3)) * cs_rank(rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS)))])

# ---- F. A few more distinct constructions (buffer above the 100-idea bar) ---------
add("vwap_reversion_speed", "technical", "VWAP回归速度：收盘-VWAP压力的1阶自相关，压力持续性", "VWAP-reversion speed: AC(1) of the close-vwap pressure, its persistence",
    [(None, lambda: rcorr(close_vwap, ishift(close_vwap, 1), 10))])
add("beta_momentum", "technical", "贝塔动量：60日市场贝塔自身的20日变化", "beta momentum: 20-day change in the 60-day rolling market beta",
    [(None, lambda: (rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS)) - ishift(rcov(ret, mkt_ret, 60) / (rvar(mkt_ret, 60) + EPS), 20))])
add("overnight_beta", "technical", "隔夜贝塔：隔夜收益对截面平均隔夜收益的60日回归系数", "overnight beta: 60-day rolling regression of overnight return on the cross-sectional mean overnight return",
    [(None, lambda: rcov(overnight, overnight.groupby(day, sort=False).transform("mean"), 60) / (rvar(overnight.groupby(day, sort=False).transform("mean"), 60) + EPS))])
add("range_to_gap_ratio", "technical", "区间/跳空比：日内区间相对隔夜跳空幅度，日内vs隔夜信息占比", "range-to-gap ratio: intraday range relative to overnight-gap size, intraday vs overnight information share",
    [(None, lambda: rmean(rng, 10) / (rmean(overnight.abs(), 10) + EPS))])

# ---- G. Second buffer batch (headroom above the 100-idea bar post-pruning) --------
add("overnight_drift_persistence", "reversal", "隔夜漂移持续性：隔夜跳空方向的5日平滑，是否持续同向", "overnight-drift persistence: 5-day smoothed overnight gap, whether the drift persists",
    [(None, lambda: rmean(overnight, 5))])
add("cross_mark_lead_lag", "technical", "跨标记领先-滞后：昨日收盘变动对今日开盘跳空的20日滚动相关", "cross-mark lead-lag: 20-day rolling corr of yesterday's close move with today's overnight gap",
    [(None, lambda: rcorr(overnight, ishift(ret, 1), 20))])
add("volatility_clustering_ac", "vol_liquidity", "波动聚集性：收益平方的1阶自相关(ARCH效应)，20日窗", "volatility clustering: AC(1) of squared returns (ARCH effect), 20-day window",
    [(None, lambda: rcorr(ret ** 2, ishift(ret ** 2, 1), 20))])
add("amihud_vol_ratio", "vol_liquidity", "单位风险非流动性：Amihud非流动性除以已实现波动率", "illiquidity-per-unit-risk: Amihud illiquidity divided by realised volatility",
    [(None, lambda: amihud / (rvol20 + EPS))])
add("range_meanreversion", "vol_liquidity", "波动率均值回归：当日区间偏离其10日均值的反向信号", "volatility mean-reversion: today's range deviation from its 10-day mean, sign-reversed",
    [(None, lambda: -(rng - rmean(rng, 10)))])
add("liquidity_beta", "technical", "流动性贝塔：成交量对市场绝对收益的60日回归敏感度", "liquidity beta: 60-day rolling sensitivity of volume to the market's absolute return",
    [(None, lambda: rcov(vol0, mkt_ret.abs(), 60) / (rvar(mkt_ret.abs(), 60) + EPS))])
add("gap_range_beta", "technical", "跳空-区间弹性：日内区间对隔夜跳空幅度的20日回归系数", "gap-range elasticity: 20-day rolling regression of intraday range on overnight-gap size",
    [(None, lambda: rcov(rng, overnight.abs(), 20) / (rvar(overnight.abs(), 20) + EPS))])
add("attention_shift_trend", "volume", "关注度转移趋势：行业相对成交量的10日变化", "attention-shift trend: 10-day change in the industry-relative volume position",
    [(None, lambda: (vol0 - vol0.groupby([day, grp], sort=False).transform("mean"))
                    - ishift(vol0 - vol0.groupby([day, grp], sort=False).transform("mean"), 10))])
add("group_volume_dispersion", "volume", "行业成交量分散度：组内成交量截面标准差(行业关注度分歧)", "group volume dispersion: within-(day,g) cross-sectional std of volume (industry attention disagreement)",
    [(None, lambda: vol0.groupby([day, grp], sort=False).transform("std"))])
add("overshoot_ratio", "interaction", "超调比率：单日收益相对当日区间的占比(方向性超调强度)", "overshoot ratio: today's net return as a share of today's range (directional overshoot strength)",
    [(None, lambda: ret / (rng.abs() + EPS))])
add("reversal_after_volume_spike", "interaction", "放量后反转：反转信号被前一日异常放量放大", "reversal-after-volume-spike: reversal amplified when yesterday's volume was an outlier",
    [(None, lambda: -(prc2 - ishift(prc2, 3)) * ishift(cs_z(vol0).abs(), 1))])
add("signed_range_trend", "interaction", "带方向区间趋势：区间幅度乘以收益方向的5日均值", "signed-range trend: 5-day mean of range magnitude times return direction",
    [(None, lambda: rmean(rng * np.sign(ret), 5))])
add("vol_illiquidity_regime", "interaction", "波动-非流动性联动：波动率排名与非流动性排名的乘积(双重风险共振)", "vol-illiquidity regime: product of the volatility rank and the illiquidity rank (compounded-risk regime)",
    [(None, lambda: cs_rank(rvol20) * cs_rank(amihud))])
add("momentum_x_efficiency", "interaction", "动量x效率：动量信号乘以趋势效率(仅在真趋势中放大动量)", "momentum x efficiency: momentum scaled by trend efficiency (amplified only in genuine trends)",
    [(None, lambda: (prc2 - ishift(prc2, 10)) * cs_rank((prc2 - ishift(prc2, 20)).abs() / (rsum(ret.abs(), 20) + EPS)))])
add("group_te_volume", "volume", "行业成交量目标编码：截至前一日的组内平均成交量(无泄漏)", "group volume target-encoding: expanding (day,g) mean volume up to the previous day (leak-free)",
    [(None, lambda: ishift(vol0.groupby([day, grp], sort=False).transform("mean"), 1))])

print(f"[nf] {len(IDEAS)} ideas defined, {sum(len(i['variants']) for i in IDEAS)} variants; computing...", flush=True)

# ----------------------------------------------------------------------------
# compute everything
# ----------------------------------------------------------------------------
results = []
out_cols = {"day": df["day"].astype("int16"), "instrument_id": df["instrument_id"].astype("int32")}
ledger = []
failures = []
t1 = time.time()
for k, idea in enumerate(IDEAS):
    for vkey, fn in idea["variants"]:
        try:
            raw = fn()
            if not isinstance(raw, pd.Series):
                raw = pd.Series(raw, index=df.index)
            res = run_variant(idea["idea"], vkey, raw, idea["category"])
        except Exception as e:
            print(f"[nf] FAILED idea={idea['idea']} variant={vkey}: {type(e).__name__}: {e}", flush=True)
            failures.append({"idea": idea["idea"], "variant": vkey, "error": f"{type(e).__name__}: {e}"})
            continue
        results.append(res)
        out_cols[f"{res['col']}_v3"] = res["v3"].astype("float32")
        ledger.append({
            "idea": idea["idea"], "variant": vkey, "category": idea["category"],
            "col": res["col"], "zh": idea["zh"], "en": idea["en"], "versions": res["versions"],
        })
    if (k + 1) % 20 == 0:
        print(f"[nf]  ...{k+1}/{len(IDEAS)} ideas done ({time.time()-t1:.0f}s)", flush=True)
print(f"[nf] all variants computed in {time.time()-t1:.0f}s ({len(failures)} failures)", flush=True)
if failures:
    print(f"[nf] FAILURES: {failures}", flush=True)

# ----------------------------------------------------------------------------
# correlation pruning at the IDEA level (representative = best-|train IC| v3 variant)
# ----------------------------------------------------------------------------
print("[nf] correlation pruning", flush=True)
rep_by_idea = {}
for idea in IDEAS:
    cands = [r for r in results if r["idea"] == idea["idea"]]
    if not cands:
        continue
    best = max(cands, key=lambda r: abs(r["versions"][-1]["train_ic"]))
    rep_by_idea[idea["idea"]] = best

idea_order = sorted(rep_by_idea, key=lambda k: -abs(rep_by_idea[k]["versions"][-1]["train_ic"]))
mask_tv = day_vals <= VALID_END
rep_mat = np.column_stack([rep_by_idea[k]["v3"].to_numpy()[mask_tv] for k in idea_order])
corr_full = np.corrcoef(rep_mat, rowvar=False)

kept, dropped = [], []
for i, k in enumerate(idea_order):
    if not kept:
        kept.append(k); continue
    kept_idx = [idea_order.index(kk) for kk in kept]
    corrs = np.abs(corr_full[i, kept_idx])
    j = int(np.argmax(corrs))
    if corrs[j] > CORR_THRESH:
        dropped.append({"idea": k, "collides_with": kept[j], "abs_corr": round(float(corrs[j]), 3)})
    else:
        kept.append(k)

print(f"[nf] kept {len(kept)} / {len(idea_order)} ideas after correlation pruning (threshold {CORR_THRESH})", flush=True)
if dropped:
    print("[nf] dropped:", [(d["idea"], d["collides_with"], d["abs_corr"]) for d in dropped], flush=True)

kept_set = set(kept)
keep_cols = ["day", "instrument_id"]
for r in results:
    if r["idea"] in kept_set:
        keep_cols.append(f"{r['col']}_v3")
keep_cols = list(dict.fromkeys(keep_cols))

out_df = pd.DataFrame(out_cols)[keep_cols]
out_df.to_parquet(OUT_PARQUET, index=False)
factor_cols = [c for c in keep_cols if c not in ("day", "instrument_id")]
OUT_LIST.write_text(json.dumps(factor_cols, indent=0))
print(f"[nf] wrote {OUT_PARQUET} ({len(out_df):,} rows, {len(factor_cols)} factor cols)", flush=True)

OUT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
OUT_LEDGER.write_text(json.dumps({
    "n_ideas": len(IDEAS), "n_variants": sum(len(i["variants"]) for i in IDEAS),
    "n_kept_ideas": len(kept), "n_final_cols": len(factor_cols),
    "corr_thresh": CORR_THRESH, "kept_ideas": kept, "dropped": dropped,
    "ledger": ledger,
}, indent=2))
print(f"[nf] wrote {OUT_LEDGER}", flush=True)
print(f"[nf] TOTAL done in {time.time()-t_start:.0f}s", flush=True)
