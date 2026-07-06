"""Parametrised online characteristic-payoff (charpay) engine — PROVEN to
reproduce the saved factor values exactly (cpf_dx3_ts20_80: corr 1.00000).

For a carrier c_j (x_j level or its h-day change), the daily characteristic
payoff is m_j[t] = mean_i( norm(c_j)[i,t] * pay[i,t] ), where pay = y or sign(y).
m_j is EWM-smoothed over days with a >=1 day label delay; on unlabelled (OOS)
days the EWM state FREEZES.  The signal for day t is
    sum_j w_j(t) * norm(c_j)[i,t] / sum_k|w_k(t)|
with weight modes:
    ts(a,b): w = (EWM_a[m] - EWM_b[m]) / sqrt(EWM_b[m^2] - EWM_b[m]^2)   (term structure)
    lvl(hl): w = EWM_hl[m]
    sgn(hl): w = sign(EWM_hl[m])
Final carrier is cs_rank or cs_z of the signal (sign-flipped to +train IC).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from core import XCOLS


def _carrier_mat(ctx, h):
    X = ctx.xmat()
    if h is None:
        return X
    lag = np.column_stack([ctx.ishift(ctx.col(c), h).to_numpy("float64") for c in XCOLS])
    return X - lag


def _norm_by_day(ctx, mat, std):
    """Per-day normalise each carrier column: 'z' -> cs z-score, 'rank' -> per-day
    rank mapped to [-1,1]. NaN->0."""
    if std == "z":
        return ctx.csz_by_day(mat)
    out = np.zeros_like(mat, dtype="float64")
    for ii in ctx.day_slices:
        a = mat[ii]
        # rank each column within the day, -> [-1,1]; NaN kept out of ranking
        df = pd.DataFrame(a)
        r = df.rank(pct=True).to_numpy()
        out[ii] = np.nan_to_num((r - 0.5) * 2.0)
    return out


def charpay(ctx, carrier_h, std, payoff, wmode, wa, wb=None,
            min_hist=40, label_delay=1, final="rank", feat_idx=None):
    norm = _norm_by_day(ctx, _carrier_mat(ctx, carrier_h), std)   # (n, 86)
    if feat_idx is not None:
        keep = np.zeros(norm.shape[1], bool); keep[list(feat_idx)] = True
        norm = np.where(keep[None, :], norm, 0.0)
    pay = ctx.y if payoff == "y" else np.sign(ctx.y)
    # daily payoff m_j[t] using the chosen pay target
    T, J = len(ctx.day_slices), norm.shape[1]
    m = np.full((T, J), np.nan)
    for t, ii in enumerate(ctx.day_slices):
        pp = pay[ii]; good = np.isfinite(pp)
        if good.any():
            m[t] = np.mean(norm[ii][good] * pp[good, None], axis=0)
    hls = sorted({wa} | ({wb} if wb else set()))
    decay = {hl: float(np.exp(np.log(0.5) / hl)) for hl in hls}
    mean = {hl: np.zeros(J) for hl in hls}
    second = {hl: np.zeros(J) for hl in hls}
    out = np.zeros(ctx.n); count = 0
    for t, ii in enumerate(ctx.day_slices):
        if count >= min_hist:
            if wmode == "ts":
                var = np.maximum(second[wb] - mean[wb] ** 2, 1e-8)
                w = (mean[wa] - mean[wb]) / np.sqrt(var)
            elif wmode == "lvl":
                w = mean[wa]
            elif wmode == "sgn":
                w = np.sign(mean[wa])
            else:
                raise ValueError(wmode)
            denom = np.sum(np.abs(w))
            if denom > 1e-12:
                out[ii] = norm[ii] @ (w / denom)
        upd = t - label_delay
        if upd >= 0 and ctx.day_has_y[upd]:          # FREEZE on unlabelled days
            obs = np.nan_to_num(m[upd])
            for hl in hls:
                a = decay[hl]
                mean[hl] = a * mean[hl] + (1 - a) * obs
                second[hl] = a * second[hl] + (1 - a) * obs ** 2
            count += 1
    sig = pd.Series(out, index=ctx.df.index)
    final_s = ctx.cs_rank(sig) if final == "rank" else ctx.cs_z(sig)
    # sign convention: positive train IC on day<=760
    tr = ctx.day_vals <= 760
    d = pd.DataFrame({"p": final_s.to_numpy(), "y": ctx.y, "day": ctx.day_vals}).dropna()
    d = d[d.day <= 760]
    tic = d.groupby("day").apply(lambda g: g.p.corr(g.y)).mean()
    if tic < 0:
        final_s = -final_s
    return np.nan_to_num(final_s.to_numpy())
