"""factors_miscA.py -- reconstruction of 5 alpha_v5 "miscA" factors as causal,
OOS-safe functions over a raw panel (core.Ctx).

Factors:
  * pastpay_xpair_k100_sign_rank_agent34
  * pastpay_orthpair_k300_sign_rank_agent34
  * crossfamily_vrs_x29_followup_vrs_resid_on_x29m5_agent24
  * closer_xmkt_b10_agent47
  * dfhpaydyn_hl60_exk_rank_agent40

All supervised state (past-payoff EWMs, dynamic IC weights) FREEZES on days
where ctx.day_has_y[t] is False (label delay >= 1), so a signal for day t only
reads payoff/label info through day t-1.  Every factor is sign-flipped to a
positive train IC (day <= 760), matching the saved values' sign convention.

Everything is computed from the raw panel only -- no precomputed factor/feature
parquet is ever read.  The two composite factors recompute their dependency
factors from raw x too:
  * crossfamily_... = as_signal(residualize_single(vrs, x29m5))   (agent24) --
    vrs and x29m5 are rebuilt from raw x + pca controls (the pcaorth/vol-range
    keeper chain), reaching ~0.98 vs the saved value (the saved base pool predates
    the current pca snapshot / rolling lib).
  * closer_xmkt_... = decile_bucket_smooth(pastpay_xmkt_k120, 10) (agent47) --
    the pastpay_xmkt base is rebuilt from raw with the past-payoff machinery; see
    the caveat at that factor (its generator is absent and carriers under-specified).
The pca controls come from the OOS-portable ctx.pca() interface (attached to the
panel by compute_263, the same frozen-loading PCA the 213 base uses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

XCOLS = [f"x_{j}" for j in range(86)]
PCA_COLS = [f"pca{i}_z" for i in range(12)]
TRAIN_END = 760
EPS = 1e-9


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------
def _zscore_cols(a: np.ndarray) -> np.ndarray:
    """Per-column z-score over rows of a single day's matrix (NaN->0).
    Matches alpha_skill past-payoff scripts' zscore_cols (var>1e-12 guard)."""
    good = np.isfinite(a)
    n = np.maximum(good.sum(axis=0), 1)
    clean = np.where(good, a, 0.0)
    mu = clean.sum(axis=0) / n
    var = np.where(good, (a - mu) ** 2, 0.0).sum(axis=0) / n
    out = (a - mu) / np.where(var > 1e-12, np.sqrt(var), 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _csz_matrix(ctx, M: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional z-score of every column of M (n, J)."""
    out = np.zeros_like(M, dtype="float64")
    for ii in ctx.day_slices:
        out[ii] = _zscore_cols(M[ii])
    return out


def _load_pca(ctx) -> np.ndarray:
    """Production pca0_z..pca11_z controls, row-aligned to the panel (n, 12).
    Live + OOS-portable (attached to the panel by compute_263)."""
    return np.column_stack(ctx.pca())


def _orth_carriers(ctx, X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Orthogonalised carriers zo (n,J): per-day z-score of the OLS residual of
    the per-day-standardised x columns on the pca controls C, i.e.
        zo_j = z( resid( cs_z(x_j) ~ 1 + pca0_z..pca11_z ) ).
    x is standardised (NaN->0) first so every day is a full-rank multi-response
    regression on the same finite rows (matches the saved orthpair carriers)."""
    Z = _csz_matrix(ctx, X)                                    # per-day z(x), NaN->0
    resid = np.zeros_like(Z, dtype="float64")
    for ii in ctx.day_slices:
        D = np.column_stack([np.ones(len(ii)), C[ii]])         # (m, K+1)
        beta, *_ = np.linalg.lstsq(D, Z[ii], rcond=None)
        resid[ii] = Z[ii] - D @ beta
    return _csz_matrix(ctx, resid)


def _train_ic_sign(v: np.ndarray, ctx) -> float:
    """+1/-1 so mean per-day IC (corr with y) on train days (day<=760) is >=0."""
    y = ctx.y
    num = 0.0
    cnt = 0
    for t, ii in enumerate(ctx.day_slices):
        if ctx.udays[t] > TRAIN_END:
            continue
        vv = v[ii]
        yy = y[ii]
        m = np.isfinite(vv) & np.isfinite(yy)
        if m.sum() < 3:
            continue
        a = vv[m] - vv[m].mean()
        b = yy[m] - yy[m].mean()
        sa = np.sqrt((a * a).sum())
        sb = np.sqrt((b * b).sum())
        if sa > 1e-12 and sb > 1e-12:
            num += float((a * b).sum() / (sa * sb))
            cnt += 1
    ic = num / cnt if cnt else 0.0
    return -1.0 if ic < 0 else 1.0


def _flip(sig: pd.Series, ctx) -> pd.Series:
    return sig * _train_ic_sign(sig.to_numpy("float64"), ctx)


# ----------------------------------------------------------------------------
# shared past-payoff pair machinery (xpair / orthpair)
# ----------------------------------------------------------------------------
def _pastpay_pair_sign(ctx, carrier_cz: np.ndarray, topk: int,
                       hl: float = 120.0, rebal: int = 20, warmup: int = 60,
                       zscore_prod: bool = False) -> np.ndarray:
    """sum over top-`topk` pairs (by |EWM_hl payoff|, reselected every `rebal`
    days) of sign(EWM_hl payoff_jk[<=t-1]) * carrier_j[t] * carrier_k[t].
    `carrier_cz` is the per-day z-scored carrier matrix (n, J).  Payoff EWM
    freezes on unlabelled days."""
    n, J = carrier_cz.shape
    y = ctx.y
    jj, kk = np.triu_indices(J, 1)
    decay = float(np.exp(np.log(0.5) / hl))
    pair_mean = np.zeros((J, J), dtype="float64")
    out = np.zeros(n, dtype="float64")
    sel_l = sel_r = None
    for t, ii in enumerate(ctx.day_slices):
        cz = carrier_cz[ii]
        if t >= warmup and t % rebal == 0:
            absm = np.abs(pair_mean[jj, kk])
            top = np.argsort(-absm, kind="stable")[:topk]
            sel_l, sel_r = jj[top], kk[top]
        if t >= warmup and sel_l is not None:
            w = np.sign(pair_mean[sel_l, sel_r])
            denom = np.sum(np.abs(w))
            if denom > 1e-12:
                prod = cz[:, sel_l] * cz[:, sel_r]
                if zscore_prod:
                    prod = _zscore_cols(prod)     # authoritative agent3: z-score pair products per day
                out[ii] = prod @ (w / denom)
        # emit-then-update; freeze on unlabelled days
        if ctx.day_has_y[t]:
            yz = _zscore_cols(y[ii].reshape(-1, 1))[:, 0]
            pair_obs = cz.T @ (cz * yz[:, None]) / len(ii)
            pair_mean = decay * pair_mean + (1.0 - decay) * pair_obs
    return out


# ----------------------------------------------------------------------------
# Factor 1: pastpay_xpair_k100_sign_rank_agent34
# ----------------------------------------------------------------------------
def _pastpay_xpair_k100(ctx) -> np.ndarray:
    xz = _csz_matrix(ctx, ctx.xmat())
    raw = _pastpay_pair_sign(ctx, xz, topk=100, hl=120.0, rebal=20, warmup=60)
    final = ctx.cs_z(ctx.cs_rank(pd.Series(raw, index=ctx.df.index)))
    return np.nan_to_num(_flip(final, ctx).to_numpy("float64"))


# ----------------------------------------------------------------------------
# Factor 2: pastpay_orthpair_k300_sign_rank_agent34
# ----------------------------------------------------------------------------
def _pastpay_orthpair_k300(ctx, pca: np.ndarray) -> np.ndarray:
    zo = _orth_carriers(ctx, ctx.xmat(), pca)           # per-day z of per-column pca residuals
    raw = _pastpay_pair_sign(ctx, zo, topk=300, hl=120.0, rebal=20, warmup=60)
    final = ctx.cs_z(ctx.cs_rank(pd.Series(raw, index=ctx.df.index)))
    return np.nan_to_num(_flip(final, ctx).to_numpy("float64"))


# ----------------------------------------------------------------------------
# Factor 3: crossfamily_vrs_x29_followup_vrs_resid_on_x29m5_agent24
# Recomputed ENTIRELY from raw x + pca controls (no precomputed-factor reads).
#   target = as_signal(residualize_single(vrs, x29m5))
#   vrs   = crossfam_vrs_resid_g0_agent22      = flip(cs_z(resid(vol_range_state, g0)))
#   x29m5 = keeper_transform_x29_mean5_agent16 = flip(cs_z(cs_z(rmean(x29keep,5))))
#   g0     = group_rank(pcaorth_x0 residual)          (sign-invariant as control)
#   x29keep= conditional_rank5(pcaorth_x0 | pcaorth_x29 quintile)
#   vol_range_state = xblock_vol_range_state_ensemble_v1 (raw x/prc/vol formula)
# Every branch is followed by a flip-to-+train-IC, so the (arbitrary) global sign
# of each leaf is irrelevant; the flips pin every sign to the saved convention.
# ----------------------------------------------------------------------------
def _pca_resid(ctx, X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-day multi-response OLS residual of raw target columns X (n,J) on the
    pca controls C (n,K) with intercept (finite rows only)."""
    out = np.full_like(X, np.nan, dtype="float64")
    for ii in ctx.day_slices:
        xd, cd = X[ii], C[ii]
        good = np.isfinite(cd).all(axis=1) & np.isfinite(xd).all(axis=1)
        if int(good.sum()) < C.shape[1] + 3:
            continue
        Dg = np.column_stack([np.ones(int(good.sum())), cd[good]])
        beta, *_ = np.linalg.lstsq(Dg, xd[good], rcond=None)
        rows = np.where(good)[0]
        Dall = np.column_stack([np.ones(len(rows)), cd[rows]])
        out[ii[rows]] = xd[rows] - Dall @ beta
    return out


def _conditional_rank(ctx, value: pd.Series, conditioner: pd.Series, n: int = 5) -> pd.Series:
    """Rank `value` within same-day `conditioner` quantile buckets -> [-1,1]."""
    q = conditioner.groupby(ctx.day, sort=False).rank(pct=True, method="average")
    bucket = np.minimum(np.floor(q.fillna(0.5).to_numpy() * n), n - 1).astype("int16")
    tmp = pd.DataFrame({"day": ctx.day_vals, "bucket": bucket, "value": value.to_numpy()})
    r = tmp.groupby(["day", "bucket"], sort=False)["value"].rank(pct=True, method="average")
    return pd.Series((r.to_numpy() - 0.5) * 2.0, index=value.index)


def _vol_range_state(ctx) -> pd.Series:
    prc = ctx.df[[f"prc{i}" for i in range(1, 6)]].to_numpy("float64")
    rng = pd.Series(np.nanmax(prc, axis=1) - np.nanmin(prc, axis=1), index=ctx.df.index)
    vol0 = ctx.col("vol0")
    csz, csr = ctx.cs_z, ctx.cs_rank
    x = lambda j: ctx.col(f"x_{j}")
    rng_shock = rng - ctx.ishift(ctx.rmean(rng, 10), 1)
    vol_shock = vol0 - ctx.ishift(ctx.rmean(vol0, 10), 1)
    t1 = csz(csz(x(46)) * csr(vol0))
    t2 = csz(csz(x(43)) * csr(rng_shock))
    t3 = csz(csz(x(76) - ctx.ishift(x(76), 10)) * csr(rng_shock))
    t4 = csz(csz(x(69) - ctx.ishift(x(69), 10)) * csr(vol_shock))
    t5 = csz(csz(x(78)) * csr(ctx.ishift(rng, 1)))
    return csz(-t1 - t2 - t3 + t4 + t5)


def _crossfamily(ctx, pca: np.ndarray) -> np.ndarray:
    idx = ctx.df.index
    tgt = np.column_stack([ctx.col("x_0").to_numpy("float64"),
                           ctx.col("x_29").to_numpy("float64")])
    resid = _pca_resid(ctx, tgt, pca)
    resid_x0 = pd.Series(resid[:, 0], index=idx)
    resid_x29 = pd.Series(resid[:, 1], index=idx)

    g0 = ctx.group_rank(resid_x0)                                    # control (sign-free)
    x29keep = _conditional_rank(ctx, resid_x0, resid_x29, 5)
    vrs_base = _vol_range_state(ctx)

    vrs = _flip(ctx.cs_z(ctx.residualize_single(vrs_base, g0)).fillna(0.0), ctx)
    x29m5 = _flip(ctx.cs_z(ctx.cs_z(ctx.rmean(x29keep, 5))).fillna(0.0), ctx)

    raw = ctx.residualize_single(vrs, x29m5)
    sig = _flip(ctx.cs_z(raw).fillna(0.0), ctx)          # as_signal: cs_z, flip, then cs_z
    final = ctx.cs_z(sig).fillna(0.0)
    return np.nan_to_num(final.to_numpy("float64"))


# ----------------------------------------------------------------------------
# Factor 4: closer_xmkt_b10_agent47  (escalation rung 2)
#   = decile_bucket_smooth( pastpay_xmkt_k120_raw_rank_agent34 , 10 )
# Recomputed from raw: the base is the past-payoff timing of z(x_j)*m_k cells
# (86 x's x 6 market carriers = 516 cells; top-120 by |EWM120 payoff|, level
# weighted, rebal 20, warmup 60), then decile-bucket-smoothed.
# NOTE: the base factor's generator (agent34 xmkt) is not present in the repo and
# its 6 market carriers are only loosely specified ("vol level/shock, range
# level/shock, mom5, rev1; all cs_z; shocks lag-1; clip5").  The carriers below
# are the faithful reading of that spec but only span ~R^2 0.16 of the saved base,
# so this factor's match to the *old* saved value is limited (~0.5).  It is a
# valid, fully-OOS-causal x-by-market payoff-timing signal; a self-consistent
# retrain re-fits downstream weights on it.
# ----------------------------------------------------------------------------
def _market_carriers(ctx) -> np.ndarray:
    prc = ctx.df[[f"prc{i}" for i in range(1, 6)]].to_numpy("float64")
    pbar = pd.Series(np.nanmean(prc, axis=1), index=ctx.df.index)
    rng = pd.Series(np.nanmax(prc, axis=1) - np.nanmin(prc, axis=1), index=ctx.df.index)
    vol0 = ctx.col("vol0")

    def cz_clip(s):
        return np.clip(np.nan_to_num(ctx.cs_z(s).to_numpy("float64")), -5.0, 5.0)

    return np.column_stack([
        cz_clip(vol0),
        cz_clip(vol0 - ctx.ishift(ctx.rmean(vol0, 10), 1)),
        cz_clip(rng),
        cz_clip(rng - ctx.ishift(ctx.rmean(rng, 10), 1)),
        cz_clip(pbar - ctx.ishift(pbar, 5)),
        cz_clip(-(pbar - ctx.ishift(pbar, 1))),
    ])


def _pastpay_xmkt_base(ctx, hl: float = 120.0, topk: int = 120,
                       rebal: int = 20, warmup: int = 60) -> np.ndarray:
    xz = _csz_matrix(ctx, ctx.xmat())          # (n, 86) z(x_j)
    mk = _market_carriers(ctx)                 # (n, 6)
    J, K = 86, mk.shape[1]
    C = J * K
    y = ctx.y
    decay = float(np.exp(np.log(0.5) / hl))
    cell_mean = np.zeros(C, dtype="float64")
    out = np.zeros(ctx.n, dtype="float64")
    sel = None
    for t, ii in enumerate(ctx.day_slices):
        m = len(ii)
        cells = (xz[ii][:, :, None] * mk[ii][:, None, :]).reshape(m, C)   # (m, 516)
        if t >= warmup and t % rebal == 0:
            sel = np.argsort(-np.abs(cell_mean), kind="stable")[:topk]
        if t >= warmup and sel is not None:
            w = cell_mean[sel]
            denom = np.sum(np.abs(w))
            if denom > 1e-12:
                out[ii] = cells[:, sel] @ (w / denom)
        if ctx.day_has_y[t]:                    # freeze on unlabelled days
            yz = _zscore_cols(y[ii].reshape(-1, 1))[:, 0]
            obs = (cells * yz[:, None]).sum(axis=0) / m
            cell_mean = decay * cell_mean + (1.0 - decay) * np.nan_to_num(obs)
    base = ctx.cs_z(ctx.cs_rank(pd.Series(out, index=ctx.df.index)))
    return np.nan_to_num(_flip(base, ctx).to_numpy("float64"))


def _closer_xmkt_b10(ctx) -> np.ndarray:
    base = pd.Series(_pastpay_xmkt_base(ctx), index=ctx.df.index)
    smoothed = ctx.decile_bucket_smooth(base, 10)
    return np.nan_to_num(smoothed.to_numpy("float64"))


# ----------------------------------------------------------------------------
# Factor 5: dfhpaydyn_hl60_exk_rank_agent40
#   dfh_j_60 = (x_j - rmax(x_j,60)) / (|rmax-rmin| + eps)
#   carrier_j = cs_rank(dfh_j_60) ; m_j[t] = daily corr(carrier_j, y)
#   w_j = EWM(hl=60) of m_j (delay 1, freeze OOS)
#   comp = sum_j w_j(t-1) carrier_j / sum_j|w_j| ; over 81 x_j excluding {1,2,5,6,7}
#   final = cs_rank(comp), flipped to +train IC.
# ----------------------------------------------------------------------------
def _dfhpaydyn(ctx, hl: float = 60.0, warmup: int = 40) -> np.ndarray:
    exclude = {1, 2, 5, 6, 7}
    keep = [j for j in range(86) if j not in exclude]
    n = ctx.n
    # carriers cs_rank(dfh_j_60)
    C = np.zeros((n, len(keep)), dtype="float64")
    for c, j in enumerate(keep):
        xj = ctx.col(f"x_{j}")
        hi = ctx.rmax(xj, 60)
        lo = ctx.rmin(xj, 60)
        dfh = (xj - hi) / ((hi - lo).abs() + EPS)
        C[:, c] = np.nan_to_num(ctx.cs_rank(dfh).to_numpy("float64"))
    y = ctx.y
    T = len(ctx.day_slices)
    Jk = len(keep)
    # daily cross-sectional IC of each carrier
    payoff = np.zeros((T, Jk), dtype="float64")
    for t, ii in enumerate(ctx.day_slices):
        cc = C[ii]
        yy = y[ii]
        good = np.isfinite(yy)
        if not good.any():
            continue
        cg = cc[good]
        yg = yy[good]
        ca = cg - cg.mean(axis=0)
        ya = yg - yg.mean()
        cov = (ca * ya[:, None]).sum(axis=0)
        vc = np.sqrt((ca * ca).sum(axis=0))
        vy = np.sqrt((ya * ya).sum())
        payoff[t] = np.where((vc > 1e-12) & (vy > 1e-12), cov / (vc * vy + 1e-18), 0.0)
    # EWM level weights with delay 1 + freeze
    decay = float(np.exp(np.log(0.5) / hl))
    mean = np.zeros(Jk, dtype="float64")
    out = np.zeros(n, dtype="float64")
    count = 0
    for t, ii in enumerate(ctx.day_slices):
        if count >= warmup:
            w = mean
            denom = np.sum(np.abs(w))
            if denom > 1e-12:
                out[ii] = C[ii] @ (w / denom)
        upd = t - 1
        if upd >= 0 and ctx.day_has_y[upd]:
            mean = decay * mean + (1.0 - decay) * payoff[upd]
            count += 1
    final = ctx.cs_rank(pd.Series(out, index=ctx.df.index))
    return np.nan_to_num(_flip(final, ctx).to_numpy("float64"))


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def gen(ctx) -> dict:
    pca = _load_pca(ctx)
    return {
        "pastpay_xpair_k100_sign_rank_agent34": _pastpay_xpair_k100(ctx),
        "pastpay_orthpair_k300_sign_rank_agent34": _pastpay_orthpair_k300(ctx, pca),
        "crossfamily_vrs_x29_followup_vrs_resid_on_x29m5_agent24": _crossfamily(ctx, pca),
        "closer_xmkt_b10_agent47": _closer_xmkt_b10(ctx),
        "dfhpaydyn_hl60_exk_rank_agent40": _dfhpaydyn(ctx),
    }
