"""factors_miscB — reconstruction of 5 alpha_v5 causal factors.

gen(ctx) -> {name: np.ndarray length ctx.n, in ctx row order, NaN->0}

Families:
  tsrr_corrected_anchor_a2000         (agent1)  term-structure reversal x range ridge
  newxj_gates_ens5_rank_agent36       (agent36) vol/range-gated x interactions (5-term ens)
  pvgres_dyn7_z_agent41               (agent41) pcaorth-resid dynamics gated by vol/range shocks
  pair2d_allpairs_m25_rank_agent46    (agent46) 2D lookup-cell interaction over all pairs
  pair2d_als_m25_rank_agent46         (agent46) 2D lookup-cell interaction over top-25-x_j pairs

All ops mirror submit/genalpha/core.Ctx (== alpha_skill/lib/ops.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------- helpers
def _nz(s):
    if isinstance(s, pd.Series):
        return np.nan_to_num(s.to_numpy("float64"))
    return np.nan_to_num(np.asarray(s, "float64"))


def _rng(ctx):
    """max(prc1..5) - min(prc1..5) as a pd.Series aligned to ctx rows."""
    P = ctx.df[[f"prc{i}" for i in range(1, 6)]]
    return P.max(axis=1) - P.min(axis=1)


def _train_ic(ctx, s):
    """mean per-day corr(s, y) over day<=760 (for sign convention)."""
    p = np.asarray(s.to_numpy("float64") if isinstance(s, pd.Series) else s, "float64")
    d = pd.DataFrame({"p": p, "y": ctx.y, "day": ctx.day_vals}).dropna()
    d = d[d.day <= 760]
    return d.groupby("day").apply(lambda g: g.p.corr(g.y)).mean()


def _load_pca(ctx):
    return ctx.pca()   # live, OOS-portable PCA controls (attached by compute_263)


# ============================================================ tsrr (agent1)
_TSRR_HORIZONS = (2, 3, 5, 8, 13, 21, 34, 55)


def _tsrr(ctx):
    """cs_z(Ridge(alpha=2000, day<=760) over
       [cs_z(rev_L), cs_z(rangelvl_L), cs_z(cs_z(rev_L)*cs_z(rangelvl_L))]),
       reversal_base=prc2, range_base=|prc4-prc3|, range_mode='level'.
    Exact replica of mine_term_structure_range_agent1.ridge_combine."""
    rev_base = ctx.col("prc2")
    rng_base = (ctx.col("prc4") - ctx.col("prc3")).abs()
    feats = []
    for L in _TSRR_HORIZONS:
        rev = ctx.cs_z(-(rev_base - ctx.ishift(rev_base, L)))
        level = ctx.rmean(rng_base, L)
        sec = ctx.cs_z(level)
        feats.append(rev)
        feats.append(sec)
        feats.append(ctx.cs_z(rev * sec))
    X = np.column_stack([_nz(s) for s in feats])
    train = ctx.day_vals <= 760
    Xt, yt = X[train], ctx.y[train]
    good = np.isfinite(yt)
    Xt, yt = Xt[good], yt[good]
    xm = Xt.mean(0)
    ym = float(yt.mean())
    Xc = Xt - xm
    yc = yt - ym
    gram = Xc.T @ Xc
    rhs = Xc.T @ yc
    coef = np.linalg.solve(gram + 2000.0 * np.eye(X.shape[1]), rhs)
    pred = (X - xm) @ coef + ym
    s = ctx.cs_z(pd.Series(pred, index=ctx.df.index))
    if _train_ic(ctx, s) < 0:
        s = -s
    return _nz(s)


# ======================================================== newxj gates (agent36)
def _newxj_gates(ctx):
    """cs_rank(cs_z( t1+t2-t3-t4-t5 )) ; see agent36 formula."""
    x2 = ctx.col("x_2")
    x32 = ctx.col("x_32")
    vol0 = np.log1p(ctx.col("vol0").clip(lower=0))   # vol gates use log1p(vol0)
    rng = _rng(ctx)
    vol_sh = ctx.cs_rank(vol0)
    dvol = ctx.cs_rank(vol0 - ctx.ishift(ctx.rmean(vol0, 10), 1))
    drng = ctx.cs_rank(rng - ctx.ishift(ctx.rmean(rng, 10), 1))
    t1 = ctx.cs_z(ctx.cs_z(x2) * vol_sh)
    t2 = ctx.cs_z(ctx.cs_z(x2 - ctx.ishift(x2, 10)) * dvol)
    t3 = ctx.cs_z(ctx.cs_z(x32) * drng)
    t4 = ctx.cs_z(ctx.cs_z(x2) * drng)
    t5 = ctx.cs_z(ctx.cs_z(x32) * vol_sh)
    s = t1 + t2 - t3 - t4 - t5
    return _nz(ctx.cs_rank(ctx.cs_z(s)))


# ======================================================== pvgres dyn7 (agent41)
def _pvgres(ctx, pca):
    """cs_z(mean(t1..t7)); R_j=residualize_multi(x_j, pca0..11); D5_j=cs_z(R_j-ishift(R_j,5))."""
    lvol = np.log1p(ctx.col("vol0"))
    rng = _rng(ctx)
    vsh60 = ctx.cs_rank((lvol - ctx.rmean(lvol, 60)) / (ctx.rstd(lvol, 60) + EPS))
    rsh60 = ctx.cs_rank((rng - ctx.rmean(rng, 60)) / (ctx.rstd(rng, 60) + EPS))
    rsh10 = ctx.cs_rank((rng - ctx.rmean(rng, 10)) / (ctx.rstd(rng, 10) + EPS))

    def R(j):
        return ctx.residualize_multi(ctx.col(f"x_{j}"), pca)

    def D5(j):
        r = R(j)
        return ctx.cs_z(r - ctx.ishift(r, 5))

    # each term nan->0 BEFORE averaging so early days (only rsh10-gated t6
    # available) still contribute, matching the saved values.
    t1 = _nz(ctx.cs_z(R(24) * vsh60))
    t2 = _nz(ctx.cs_z(D5(24) * vsh60))
    t3 = _nz(ctx.cs_z(D5(14) * vsh60))
    t4 = _nz(ctx.cs_z(D5(19) * vsh60))
    t5 = _nz(ctx.cs_z(R(18) * rsh60))
    t6 = _nz(ctx.cs_z(D5(15) * rsh10))
    t7 = _nz(ctx.cs_z(R(48) * vsh60))
    mean = (t1 + t2 - t3 - t4 - t5 - t6 - t7) / 7.0
    return _nz(ctx.cs_z(pd.Series(mean, index=ctx.df.index)))


# =================================================== pair2d engine (agent46)
# 2D lookup-cell interaction factor.  Each x_j is quantised into 5 per-day
# quintile buckets q5.  For a pair (j,k) a 5x5 cell table T holds the expanding,
# label-delayed (d2), lam=200-shrunk mean of y for instruments in cell
# (q5(x_j),q5(x_k)).  Additive row/col margins are removed by count-weighted ALS
# and the rank-1 bilinear (centered-quantile-code outer product) is projected
# out, leaving the pure interaction table I2.  A per-day, per-pair z-score of the
# instrument's cell value is the pair lookup.  The composite averages the lookups
# of the top-25 pairs ranked by causal EWM(hl60) daily-payoff (rebal every 20d,
# burn-in 250d).  Final carrier = cs_z(cs_rank(.)); days<=250 forced to 0.
# SUPERVISED: table/EWM updates use a 2-day label delay and freeze on unlabelled
# days, so OOS rows never read their own/future y.
_U = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_BIL = np.outer(_U, _U).reshape(25)


def _pair2d_buckets(ctx):
    """(n,86) int quintile bucket per row (per-day rank pct -> bucket 0..4)."""
    X = ctx.xmat()
    B = np.zeros((ctx.n, 86), dtype=np.int16)
    for ii in ctx.day_slices:
        a = X[ii]
        ns = len(ii)
        order = np.argsort(a, axis=0, kind="stable")
        ranks = np.empty_like(order)
        ar = np.arange(ns)
        for c in range(86):
            ranks[order[:, c], c] = ar
        B[ii] = np.minimum(((ranks + 1) / ns * 5).astype(int), 4).astype(np.int16)
    return B


def _pair2d_top25_xj(ctx, X):
    """top-25 x_j by |train rank-payoff| over day<=760 (als 300-pair universe).
    payoff_j = mean over train days of mean_i( cs_rank(x_j)[i] * y[i] )."""
    tp = np.zeros(86)
    cnt = 0
    for ii in ctx.day_slices:
        if ctx.day_vals[ii][0] > 760:
            continue
        yy = ctx.y[ii]
        good = np.isfinite(yy)
        if good.sum() < 10:
            continue
        a = X[ii]
        ns = len(ii)
        order = np.argsort(a, axis=0, kind="stable")
        z = np.empty_like(a)
        ar = np.arange(ns)
        for c in range(86):
            rr = np.empty(ns)
            rr[order[:, c]] = ar
            z[:, c] = (rr + 1) / ns * 2 - 1          # cs_rank in (-1,1]
        tp += (z[good] * yy[good, None]).mean(0)
        cnt += 1
    tp /= max(cnt, 1)
    return sorted(np.argsort(-np.abs(tp))[:25].tolist())


def _pair2d_composite(ctx, B, pairs, lam=200.0, delay=2, hl=60.0, rebal=20,
                      burn=250, topk=25, als_iter=4):
    n = ctx.n
    slices = ctx.day_slices
    T = len(slices)
    y = ctx.y
    day = ctx.day_vals
    P = len(pairs)
    J = np.array([p[0] for p in pairs])
    K = np.array([p[1] for p in pairs])
    parP = np.arange(P)
    Ssum = np.zeros((P, 25))
    Scnt = np.zeros((P, 25))
    decay = 0.5 ** (1.0 / hl)
    em = np.zeros(P)
    ecnt = 0
    payoff_hist = [None] * T
    comp = np.zeros(n)
    cur_top = np.arange(min(topk, P))
    for ti in range(T):
        ii = slices[ti]
        dv = day[ii][0]
        s_in = ti - delay
        if s_in >= 0:
            jj = slices[s_in]
            ys = y[jj]
            good = np.isfinite(ys)
            if good.any():                       # FREEZE on unlabelled days
                bj = B[jj][good]
                yv = ys[good]
                cell = 5 * bj[:, J].astype(np.int64) + bj[:, K].astype(np.int64)
                flat = (parP[None, :] * 25 + cell).ravel()
                Ssum += np.bincount(flat, weights=np.repeat(yv, P),
                                    minlength=P * 25).reshape(P, 25)
                Scnt += np.bincount(flat, minlength=P * 25).reshape(P, 25).astype(float)
                ph = payoff_hist[s_in]
                if ph is not None:
                    em = decay * em + (1 - decay) * ph
                    ecnt += 1
        if ecnt > 0 and (ti % rebal == 0):
            cur_top = np.argsort(-em)[:topk]
        # interaction table: shrunk cell mean -> count-weighted ALS margins -> bilinear
        Tt = Ssum / (Scnt + lam)
        Tg = Tt.reshape(P, 5, 5)
        Wg = Scnt.reshape(P, 5, 5)
        g = np.zeros(P)
        r = np.zeros((P, 5))
        c = np.zeros((P, 5))
        Ws = np.where(Wg.sum((1, 2)) > 0, Wg.sum((1, 2)), 1)
        Wrs = np.where(Wg.sum(2) > 0, Wg.sum(2), 1)
        Wcs = np.where(Wg.sum(1) > 0, Wg.sum(1), 1)
        for _ in range(als_iter):
            g = (Wg * (Tg - r[:, :, None] - c[:, None, :])).sum((1, 2)) / Ws
            r = (Wg * (Tg - g[:, None, None] - c[:, None, :])).sum(2) / Wrs
            c = (Wg * (Tg - g[:, None, None] - r[:, :, None])).sum(1) / Wcs
        I = (Tg - g[:, None, None] - r[:, :, None] - c[:, None, :]).reshape(P, 25)
        num = (Scnt * I * _BIL[None, :]).sum(1)
        den = (Scnt * _BIL[None, :] ** 2).sum(1)
        I2 = I - (num / np.where(den > 0, den, 1))[:, None] * _BIL[None, :]
        # per-day per-pair z-score lookup for this day's instruments
        bi = B[ii]
        cell = 5 * bi[:, J].astype(np.int64) + bi[:, K].astype(np.int64)
        look = I2[parP[None, :], cell]
        z = (look - look.mean(0)) / (look.std(0) + EPS)
        yy = y[ii]
        gp = np.isfinite(yy)
        if gp.any():
            payoff_hist[ti] = (z[gp] * yy[gp, None]).mean(0)
        if dv > burn:
            comp[ii] = z[:, cur_top].mean(1)
    return comp


def _pair2d_final(ctx, comp):
    s = pd.Series(comp, index=ctx.df.index)
    fin = np.nan_to_num(ctx.cs_z(ctx.cs_rank(s)).to_numpy())
    fin[ctx.day_vals <= 250] = 0.0
    return fin


# ==================================================================== gen
def gen(ctx) -> dict:
    out = {}
    pca = _load_pca(ctx)
    out["tsrr_corrected_anchor_a2000"] = _tsrr(ctx)
    out["newxj_gates_ens5_rank_agent36"] = _newxj_gates(ctx)
    out["pvgres_dyn7_z_agent41"] = _pvgres(ctx, pca)
    # pair2d family (best-effort reconstruction of the supervised interaction factor)
    B = _pair2d_buckets(ctx)
    top25 = _pair2d_top25_xj(ctx, ctx.xmat())
    import itertools
    pairs_als = list(itertools.combinations(top25, 2))
    pairs_all = list(itertools.combinations(range(86), 2))
    out["pair2d_als_m25_rank_agent46"] = _pair2d_final(
        ctx, _pair2d_composite(ctx, B, pairs_als))
    out["pair2d_allpairs_m25_rank_agent46"] = _pair2d_final(
        ctx, _pair2d_composite(ctx, B, pairs_all))
    return out
