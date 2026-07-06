"""genalpha reconstruction — icw (IC-weighted composite) + sup (supervised ridge/tilt)
families.  Causal, OOS-safe, self-contained on the raw panel (no features.parquet
dependency): every engineered input (prc_mean, vol0) is recomputed from the panel.

Reconstructed to match the saved ic02 values (days 1..1259) from formula strings;
the original ic02 generator source is lost.  Achieved corr vs saved:
    icw_x_rank_w30_s7     ~0.9998
    icw_x_pearson_w8_s7   ~0.9997
    icw_all_pearson_w60_s7~0.9999
    sup_ridge_sm3_s4      ~0.9949
    sup_tilt_sm1_x42_s4   ~0.9988
    sup_tilt_sm1_x55_s4   ~0.9897   (shared-master limited; see note in report)
    sup_tilt_sm1_x60_s4   ~0.9918

ICW recipe: per-day normalise each feature (rank->[-1,1] or z), take the daily
cross-sectional pearson IC vs y, smooth it with rolling{w}(min_periods=w//3) using
only days <= t-1 (lag1), and form composite[i,t] = sum_j smoothIC_j(t)*norm_j[i,t].
IC weights FREEZE on unlabelled (OOS) days.

SUP recipe: ridge-on-past.  carriers c_j = cs_z(rmean(x_j,w)) (w=3 master 'sm3',
w=1 == cs_z(x_j) for the tilt master 'sm1').  Each day t fit beta on a trailing
252 labelled-day Gram (delay1): beta=(sumZZ+lam*trace/K*I)^-1 sumZy, lam=10, K=86;
predict signal[i,t]=c[i,t]@beta.  Gram updates FREEZE on unlabelled days.
    sup_ridge_sm3_s4 = cs_z(pred_sm3)
    sup_tilt_sm1_xNN = cs_z( cs_z(pred_sm1) + cs_z(x_NN) )
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core import XCOLS

K = 86


# ---------------------------------------------------------------- helpers
def _rank_norm_by_day(ctx, mat):
    """per-day rank of each column -> [-1,1], NaN->0 (matches _charpay_engine)."""
    out = np.zeros_like(mat, dtype="float64")
    for ii in ctx.day_slices:
        r = pd.DataFrame(mat[ii]).rank(pct=True).to_numpy()
        out[ii] = np.nan_to_num((r - 0.5) * 2.0)
    return out


def _daily_ic(ctx, norm, y):
    """(T,J) per-day cross-sectional pearson IC of each norm column vs y."""
    T, J = len(ctx.day_slices), norm.shape[1]
    ic = np.full((T, J), np.nan)
    for t, ii in enumerate(ctx.day_slices):
        yv = y[ii]
        g = np.isfinite(yv)
        if not g.any():
            continue                      # unlabelled day -> leave NaN (freeze)
        A = norm[ii][g]
        yy = yv[g]
        Ac = A - A.mean(0)
        yc = yy - yy.mean()
        num = Ac.T @ yc
        den = np.sqrt((Ac ** 2).sum(0) * (yc ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ic[t] = np.where(den > 1e-12, num / den, 0.0)
    return ic


def _smooth_lag1(ic, w):
    """rolling-mean(window=w, min_periods=w//3) of the daily IC, then 1-day lag.
    NaN rows (unlabelled days) are held (ffill) so weights FREEZE on OOS days."""
    s = pd.DataFrame(ic).ffill()
    return np.nan_to_num(
        s.rolling(w, min_periods=max(1, w // 3)).mean().shift(1).to_numpy())


def _icw(ctx, norm, y, w):
    """IC-weighted composite over the columns of `norm`."""
    sm = _smooth_lag1(_daily_ic(ctx, norm, y), w)
    out = np.zeros(ctx.n)
    for t, ii in enumerate(ctx.day_slices):
        out[ii] = norm[ii] @ sm[t]
    return out


def _cszcol(ctx, arr):
    """per-day cross-sectional z of a single 1-D array (NaN->0)."""
    return ctx.csz_by_day(np.asarray(arr, "float64").reshape(-1, 1))[:, 0]


def _ridge_pred(ctx, norm, y, W=252, lam=10.0):
    """ridge-on-past prediction.  Trailing W labelled-day Gram, delay1, freeze OOS."""
    T = len(ctx.day_slices)
    # per-day Gram / cross-moment (only labelled rows contribute)
    G = np.zeros((T, K, K))
    B = np.zeros((T, K))
    lab = np.zeros(T, bool)
    for t, ii in enumerate(ctx.day_slices):
        yv = y[ii]
        g = np.isfinite(yv)
        if g.any():
            Z = norm[ii][g]
            G[t] = Z.T @ Z
            B[t] = Z.T @ yv[g]
            lab[t] = True
    pred = np.zeros(ctx.n)
    Ieye = np.eye(K)
    cumG = [np.zeros((K, K))]          # cumulative over labelled days only
    cumB = [np.zeros(K)]
    nlab = 0
    for t, ii in enumerate(ctx.day_slices):
        if nlab > 0:                    # beta from the last <=W labelled days < t
            lo = max(0, nlab - W)
            SZZ = cumG[nlab] - cumG[lo]
            SZy = cumB[nlab] - cumB[lo]
            tr = np.trace(SZZ)
            if tr > 1e-12:
                beta = np.linalg.solve(SZZ + lam * tr / K * Ieye, SZy)
                pred[ii] = norm[ii] @ beta
        if lab[t]:                      # advance the frozen history (labelled only)
            cumG.append(cumG[-1] + G[t])
            cumB.append(cumB[-1] + B[t])
            nlab += 1
    return pred


# ---------------------------------------------------------------- entry point
def gen(ctx) -> dict:
    y = ctx.y
    X = ctx.xmat()
    out = {}

    # ===== ICW =====================================================
    xz = ctx.csz_by_day(X)               # per-day z of the 86 x  (kind=pearson)
    xr = _rank_norm_by_day(ctx, X)       # per-day rank of the 86 x (kind=rank)

    out["icw_x_rank_w30_s7"]   = np.nan_to_num(_icw(ctx, xr, y, 30))
    out["icw_x_pearson_w8_s7"] = np.nan_to_num(_icw(ctx, xz, y, 8))

    # subset="all": 86 x  +  vol0  +  prc_mean (recomputed from the panel).
    # prc_mean carries a ~5x loading in the lost original; 5.0 reproduces the
    # saved values to corr ~0.9999 (see report note).
    prc_mean = ctx.df[[f"prc{i}" for i in range(1, 6)]].mean(1).to_numpy("float64")
    vol0 = ctx.col("vol0").to_numpy("float64")
    extra = ctx.csz_by_day(np.column_stack([vol0, prc_mean]))   # (n,2): vol0, prc_mean
    norm_all = np.column_stack([xz, extra])                      # (n,88)
    sm_all = _smooth_lag1(_daily_ic(ctx, norm_all, y), 60)
    w_all = sm_all.copy()
    w_all[:, 87] *= 5.0                  # amplify prc_mean loading
    comp = np.zeros(ctx.n)
    for t, ii in enumerate(ctx.day_slices):
        comp[ii] = norm_all[ii] @ w_all[t]
    out["icw_all_pearson_w60_s7"] = np.nan_to_num(comp)

    # ===== SUP =====================================================
    # sm1 carriers = cs_z(x_j); sm3 carriers = cs_z(rmean(x_j,3))
    R3 = np.column_stack([ctx.rmean(ctx.col(c), 3).to_numpy("float64") for c in XCOLS])
    n3 = ctx.csz_by_day(R3)
    n1 = xz                              # cs_z(rmean(x,1)) == cs_z(x)

    pred3 = _ridge_pred(ctx, n3, y)
    out["sup_ridge_sm3_s4"] = np.nan_to_num(
        ctx.cs_z(pd.Series(pred3, index=ctx.df.index)).to_numpy())

    master1 = ctx.cs_z(pd.Series(_ridge_pred(ctx, n1, y),
                                 index=ctx.df.index)).to_numpy()   # cs_z(pred_sm1)
    for NN in (42, 55, 60):
        cszx = ctx.cs_z(ctx.col(f"x_{NN}")).to_numpy()
        tilt = ctx.cs_z(pd.Series(np.nan_to_num(master1) + np.nan_to_num(cszx),
                                  index=ctx.df.index)).to_numpy()
        out[f"sup_tilt_sm1_x{NN}_s4"] = np.nan_to_num(tilt)

    return out
