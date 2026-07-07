"""Reconstruction of the `charpay` (characteristic-payoff timing) factor family
as causal, deployable functions over a `core.Ctx`.

Recipe (see submit/genalpha/_charpay_engine.py, proven on cpf_dx3_ts20_80):
  carrier c_j = x_j level (carrier_h=None) or its h-day change (x_j[t]-x_j[t-h]).
  norm_j     = per-day cross-sectional standardisation of c_j: 'z' (cs z-score)
               or 'rank' (per-day pct-rank mapped to [-1,1]).   (reused verbatim
               from the engine's _carrier_mat / _norm_by_day.)
  daily payoff m_j[t] = mean_i( norm_j[i,t] * pay[i,t] ), pay = y or sign(y).
  m_j is EWM-smoothed over days (>=1 day label delay); on unlabelled (OOS) days
  the EWM state FREEZES.  Weight w_j(t):
     ts(a,b): (EWM_a[m]-EWM_b[m]) / sqrt(EWM_b[m^2]-EWM_b[m]^2)
     lvl(hl): EWM_hl[m]
     sgn(hl): sign(EWM_hl[m])
  signal[i,t] = sum_j w_j(t)*norm_j[i,t] / sum_k|w_k(t)|   (j restricted to a
  feature block when block!=None).  Final carrier = raw signal / cs_rank / cs_z,
  sign-flipped to positive train-IC (day<=760).

Two EWM conventions are needed to match the saved values exactly:
  * 'manual' — engine-style zero-init recursion with a 40-day warm-up gate and a
    label delay (used by every '..._s1' ic02 factor; final='raw' — NO cs transform).
  * 'pandas' — pandas .ewm(adjust=..,min_periods=0) with NO warm-up gate, emitting
    from day 0 (used by the two alpha_v5 agent factors; final='rank').  Implemented
    here as a freeze-capable recursion that equals pandas on fully-labelled days.

Verified corr vs saved parquet on days 1-1259 (see module docstring table below).
Supervision is frozen on unlabelled days so the module is leak-free OOS; on days
1-1259 every day is labelled so freezing is a no-op for validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from _charpay_engine import _carrier_mat, _norm_by_day

TRAIN_END = 760


# ------------------------------------------------------------------ primitives
def _norm(ctx, carrier_h, std, feat_idx=None):
    norm = _norm_by_day(ctx, _carrier_mat(ctx, carrier_h), std)          # (n,86)
    if feat_idx is not None:
        keep = np.zeros(norm.shape[1], bool); keep[list(feat_idx)] = True
        norm = np.where(keep[None, :], norm, 0.0)
    return norm


def _daily_payoff(ctx, norm, payoff):
    pay = ctx.y if payoff == "y" else np.sign(ctx.y)
    T, J = len(ctx.day_slices), norm.shape[1]
    m = np.full((T, J), np.nan)
    for t, ii in enumerate(ctx.day_slices):
        pp = pay[ii]; good = np.isfinite(pp)
        if good.any():
            m[t] = np.mean(norm[ii][good] * pp[good, None], axis=0)
    return m


def _weight(wmode, mean, second, wa, wb):
    if wmode == "ts":
        var = np.maximum(second[wb] - mean[wb] ** 2, 1e-8)
        return (mean[wa] - mean[wb]) / np.sqrt(var)
    if wmode == "lvl":
        return mean[wa]
    if wmode == "sgn":
        return np.sign(mean[wa])
    raise ValueError(wmode)


def _emit_manual(ctx, norm, m, wmode, wa, wb, delay, min_hist):
    """Engine-style zero-init EWM recursion; freeze on unlabelled days; warm-up
    gate of `min_hist` labelled days.  Returns the raw signal (n,)."""
    T, J = m.shape
    hls = sorted({wa} | ({wb} if wb else set()))
    decay = {hl: float(np.exp(np.log(0.5) / hl)) for hl in hls}
    mean = {hl: np.zeros(J) for hl in hls}
    second = {hl: np.zeros(J) for hl in hls}
    out = np.zeros(ctx.n); count = 0
    for t, ii in enumerate(ctx.day_slices):
        if count >= min_hist:
            w = _weight(wmode, mean, second, wa, wb)
            denom = np.sum(np.abs(w))
            if denom > 1e-12:
                out[ii] = norm[ii] @ (w / denom)
        upd = t - delay
        if upd >= 0 and ctx.day_has_y[upd]:                # FREEZE on unlabelled
            obs = np.nan_to_num(m[upd])
            for hl in hls:
                a = decay[hl]
                mean[hl] = a * mean[hl] + (1 - a) * obs
                second[hl] = a * second[hl] + (1 - a) * obs ** 2
            count += 1
    return out


def _ewm_frozen(ctx, x, hl, adjust):
    """EWM of each column of x over days, frozen on unlabelled days.  Equals
    pandas .ewm(halflife=hl, adjust=adjust, min_periods=0).mean() on days whose
    payoff is finite (all of 1-1259).  E[t] includes x[t]."""
    T, J = x.shape
    alpha = 1.0 - 0.5 ** (1.0 / hl)
    E = np.full((T, J), np.nan)
    if adjust:
        num = np.zeros(J); den = np.zeros(J); started = False
        for t in range(T):
            if ctx.day_has_y[t]:
                obs = np.nan_to_num(x[t])
                num = (1 - alpha) * num + obs
                den = (1 - alpha) * den + 1.0
                started = True
            if started:
                E[t] = num / den
    else:
        y = np.zeros(J); started = False
        for t in range(T):
            if ctx.day_has_y[t]:
                obs = np.nan_to_num(x[t])
                y = obs if not started else (1 - alpha) * y + alpha * obs
                started = True
            if started:
                E[t] = y
    return E


def _emit_pandas(ctx, norm, m, wmode, wa, wb, delay, adjust):
    """pandas-EWM convention (min_periods=0, no warm-up gate): the weight at day t
    uses the EWM through m[t-delay].  Returns the raw signal (n,)."""
    T, J = m.shape

    def roll(E):
        R = np.full_like(E, np.nan)
        if delay < T:
            R[delay:] = E[:-delay]
        return R

    Ea = roll(_ewm_frozen(ctx, m, wa, adjust))
    Eb = roll(_ewm_frozen(ctx, m, wb, adjust)) if wb else None
    Eb2 = roll(_ewm_frozen(ctx, m ** 2, wb, adjust)) if wb else None
    out = np.zeros(ctx.n)
    for t, ii in enumerate(ctx.day_slices):
        if not np.isfinite(Ea[t]).all():
            continue
        mean = {wa: Ea[t]}
        second = {}
        if wb:
            mean[wb] = Eb[t]; second[wb] = Eb2[t]
        w = _weight(wmode, mean, second, wa, wb)
        denom = np.sum(np.abs(w))
        if denom > 1e-12:
            out[ii] = norm[ii] @ (w / denom)
    return out


def _finalize(ctx, out, final):
    s = pd.Series(out, index=ctx.df.index)
    if final == "rank":
        fs = ctx.cs_rank(s)
    elif final == "z":
        fs = ctx.cs_z(s)
    else:                                                   # 'raw'
        fs = s
    d = pd.DataFrame({"p": fs.to_numpy(), "y": ctx.y, "day": ctx.day_vals}).dropna()
    d = d[d.day <= TRAIN_END]
    tic = _train_ic_scalar(d)          # scalar (nan if no train days -> keep sign)
    if np.isfinite(tic) and tic < 0:
        fs = -fs
    return np.nan_to_num(fs.to_numpy())


def _train_ic_scalar(d) -> float:
    """Mean per-day cross-sectional IC of columns p vs y over the rows in d, as a
    guaranteed scalar. Returns nan when there are no usable (train) days — e.g. an
    OOS-only slice — so callers keep the factor's sign unchanged instead of crashing
    on an empty/ambiguous groupby result."""
    ics = []
    for _, g in d.groupby("day"):
        if len(g) >= 3 and g["p"].std() > 0 and g["y"].std() > 0:
            ics.append(g["p"].corr(g["y"]))
    return float(np.nanmean(ics)) if ics else np.nan


def _charpay(ctx, carrier_h, std, payoff, wmode, wa, wb=None, *,
             ewm="manual", delay=1, min_hist=40, adjust=True,
             final="raw", feat_idx=None):
    norm = _norm(ctx, carrier_h, std, feat_idx)
    m = _daily_payoff(ctx, norm, payoff)
    if ewm == "manual":
        out = _emit_manual(ctx, norm, m, wmode, wa, wb, delay, min_hist)
    else:
        out = _emit_pandas(ctx, norm, m, wmode, wa, wb, delay, adjust)
    return _finalize(ctx, out, final)


# ------------------------------------------------------------- feature blocks
# The 'block=q?/t?/od' variants restrict the weighted sum to a subset of the 86
# features.  Decoded empirically against the saved values (a greedy oracle-subset
# search recovered feature sets that were exactly the index residue classes):
# the block is an INDEX STRIDE (round-robin over the 0..85 feature index) --
#   q<i>  -> {j : j % 4 == i}   (quartile stride, ~21-22 features)
#   t<i>  -> {j : j % 3 == i}   (tercile  stride, ~28-29 features)
#   od    -> {j : j % 2 == 1}   (odd indices)
# The signal is normalised by sum_{j in block}|w_j| (denominator over the block).
def _block_feat_idx(block):
    if block == "od":
        return [j for j in range(86) if j % 2 == 1]
    kind, i = block[0], int(block[1:])
    Q = 4 if kind == "q" else 3
    return [j for j in range(86) if j % Q == i]


# --------------------------------------------------------------------- driver
def gen(ctx) -> dict:
    out = {}

    # ---- alpha_v5 anchor (proven): engine manual recursion, final rank -------
    out["cpf_dx3_ts20_80"] = _charpay(
        ctx, 3, "z", "y", "ts", 20, 80, ewm="manual", final="rank")

    # ---- alpha_v5 agent charpay: pandas EWM, no warm-up gate, final rank -----
    out["charpay_dx5_ts40_160_rank_agent30"] = _charpay(
        ctx, 5, "z", "y", "ts", 40, 160, ewm="pandas", adjust=True,
        delay=1, final="rank")
    out["charpay_dx10_ts80_320_rank_agent43"] = _charpay(
        ctx, 10, "z", "y", "ts", 80, 320, ewm="pandas", adjust=False,
        delay=2, final="rank")

    # ---- ic02 '_s1' charpay: manual recursion, warm-up 40, final RAW ---------
    out["dx5_z_y_ts40_160_s1"] = _charpay(ctx, 5, "z", "y", "ts", 40, 160)
    out["dx10_z_y_ts80_320_s1"] = _charpay(ctx, 10, "z", "y", "ts", 80, 320)
    out["dx3_z_y_ts80_320_s1"] = _charpay(ctx, 3, "z", "y", "ts", 80, 320)
    out["lvl_z_y_ts20_120_s1"] = _charpay(ctx, None, "z", "y", "ts", 20, 120)
    out["lvl_rank_signy_ts40_160_s1"] = _charpay(ctx, None, "rank", "signy", "ts", 40, 160)
    out["dx5_z_signy_sgn60_s1"] = _charpay(ctx, 5, "z", "signy", "sgn", 60)
    out["lvl_rank_y_sgn60_s1"] = _charpay(ctx, None, "rank", "y", "sgn", 60)

    # ---- ic02 '_s1' charpay block variants (feat subset) ---------------------
    out["dx20_rank_y_lvl60_bq2_s1"] = _charpay(
        ctx, 20, "rank", "y", "lvl", 60, feat_idx=_block_feat_idx("q2"))
    out["dx3_rank_y_lvl60_bq2_s1"] = _charpay(
        ctx, 3, "rank", "y", "lvl", 60, feat_idx=_block_feat_idx("q2"))
    out["dx3_rank_y_lvl60_bt2_s1"] = _charpay(
        ctx, 3, "rank", "y", "lvl", 60, feat_idx=_block_feat_idx("t2"))
    out["dx10_rank_y_lvl60_bt0_s1"] = _charpay(
        ctx, 10, "rank", "y", "lvl", 60, feat_idx=_block_feat_idx("t0"))

    return {k: np.nan_to_num(v) for k, v in out.items()}
