"""genalpha 'gx' family — group-conditional characteristic-payoff factors.

gxpay = the charpay recipe (see _charpay_engine.py) with a GROUP-AWARE payoff.
For carrier c_j = per-day cross-sectional z-score of x_j, and a payoff target
(y, or the per-day rank of y), we form two daily characteristic payoffs

    m_glob[t,j]   = mean_i ( z_j[i,t] * target[i,t] )                 (all instruments)
    m_grp[t,g,j]  = mean_{i in g} ( z_j[i,t] * target[i,t] )          (per group g)

both EWM-smoothed over days (halflife hl) with a >=1 day label delay and FROZEN
on unlabelled (OOS) days.  For an instrument in group g the per-feature payoff is
a group-size-shrunk blend of the two

    B[g,j] = (1 - w_g) * EWM_glob[j] + w_g * EWM_grp[g,j],
    w_g    = ewm_grpsize_g / (ewm_grpsize_g + k)        ("k" = shrink constant)

so a large/stable group leans on its own payoff, a small one on the global one.
The signal is  s[i,t] = sum_j B[g_i, j] * z_j[i,t]  and the factor is a final
cross-sectional transform (raw / cs_rank / group_rank) sign-flipped to +train IC
(day<=760).

Causality: EWMs only look back (pandas ewm is causal; the shift adds the label
delay); on unlabelled days the daily payoff is NaN -> ignored -> the EWM freezes,
so no OOS row ever reads its own/future y.  Groups use the point-in-time `g`
column (verified: current-day g beats lagged / static-modal / first / last).

Two EWM conventions are used, matched empirically to each source generator:
  * ic02 / agent 's3' factors (gxb1, gxyr, gxz)  -> pandas EWM (adjust=False,
    ignore_na=True): the first observation seeds the state as itself (y0=x0).
  * alpha_v5 / agent28 (gxpay_shrunk)            -> a manual zero-init EWM
    (y0=(1-a)*x0), which is what that generator used (cs_rank output is scale
    sensitive, so the init convention is visible; it is not for group_rank).

Validation corr vs saved (days 1-1259):
    gxz_hl20_k8_gr_s3                     0.99979
    gxb1_hl60_k5_s3                       0.99779
    gxpay_shrunk_g_hl60_k30_d2_rank...   0.99470
    gxyr_hl60_k60_s3                      0.98367  (see note below)

gxyr note: recipe is exact — its converged periods (days ~1-100 and ~600-1259)
match at daily corr 1.0000 — but days ~100-400 dip because 850+ instruments are
re-assigned to new groups there (a reshuffle in the `g` column). The per-instrument
error in that window is 7x larger on group-changing instruments; the original
generator's exact per-group EWM bookkeeping across those membership transitions is
not fully recoverable from the saved values, capping the overall match at 0.984.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core import XCOLS

# featblock1: the 17-feature block used by gxb1 (recovered by regressing the saved
# raw signal onto the 86 per-feature contributions -> exactly these 17 have coef ~1).
_BLOCK1 = [0, 5, 9, 15, 19, 30, 31, 42, 53, 55, 56, 64, 65, 70, 74, 78, 85]


def _prep(ctx):
    """Shared, factor-independent precompute (build once per ctx)."""
    normZ = ctx.csz_by_day(ctx.xmat())                       # (n,86) per-day z carriers
    gvals = ctx.grp.to_numpy()
    ug = np.unique(gvals)
    g2i = {int(v): k for k, v in enumerate(ug)}
    gcode = np.array([g2i[int(v)] for v in gvals])           # 0..G-1 group index
    G = len(ug)
    y = ctx.y
    yrank = ctx.cs_rank(pd.Series(y, index=ctx.df.index)).to_numpy()   # per-day rank in [-1,1], NaN where y NaN
    return {"normZ": normZ, "gcode": gcode, "G": G, "T": len(ctx.day_slices),
            "y": y, "yrank": yrank}


def _ewm_manual(A, hl):
    """Zero-init unadjusted EWM over axis 0; per-element freeze on NaN (ignore_na).
    y_t = a*y_{t-1} + (1-a)*x_t, y_{-1}=0 -> y0 = (1-a)*x0."""
    a = float(np.exp(np.log(0.5) / hl))
    K = A.shape[1]
    st = np.zeros(K)
    started = np.zeros(K, bool)
    out = np.full_like(A, np.nan)
    for t in range(A.shape[0]):
        row = A[t]
        obs = np.isfinite(row)
        st[obs] = a * st[obs] + (1.0 - a) * row[obs]
        started |= obs
        out[t] = np.where(started, st, np.nan)
    return out


def _ewm_pandas(A, hl):
    """pandas causal EWM, adjust=False (y0=x0), ignore_na=True (freeze on NaN)."""
    return pd.DataFrame(A).ewm(halflife=hl, adjust=False, ignore_na=True).mean().to_numpy()


def _shift(A, d):
    B = np.full_like(A, np.nan)
    if d > 0:
        B[d:] = A[:-d]
    else:
        B[:] = A
    return B


def _gxpay(ctx, prep, target, hl, k, final, delay, kind, feat_idx=None):
    normZ, gcode, G, T = prep["normZ"], prep["gcode"], prep["G"], prep["T"]
    feat = list(range(normZ.shape[1])) if feat_idx is None else list(feat_idx)
    A = len(feat)
    nm = normZ[:, feat]                                      # (n,A) active carriers

    # ---- daily global / per-group payoff and group size (NaN where undefined) ----
    Mg = np.full((T, A), np.nan)
    Mgr = np.full((T, G, A), np.nan)
    S = np.full((T, G), np.nan)
    for t, ii in enumerate(ctx.day_slices):
        tg = target[ii]
        good = np.isfinite(tg)
        if not good.any():
            continue                                         # unlabelled day -> NaN -> EWM freezes
        P = nm[ii] * np.nan_to_num(tg)[:, None]
        Pg = P[good]
        Mg[t] = Pg.mean(0)
        grp = gcode[ii][good]
        sums = np.zeros((G, A))
        cnts = np.zeros(G)
        np.add.at(sums, grp, Pg)
        np.add.at(cnts, grp, 1)
        pres = cnts > 0
        Mgr[t, pres] = sums[pres] / cnts[pres, None]
        S[t, pres] = cnts[pres]

    ewm = _ewm_pandas if kind == "pandas" else _ewm_manual
    EG = _shift(ewm(Mg, hl), delay)                          # (T,A)
    EGR = _shift(ewm(Mgr.reshape(T, G * A), hl).reshape(T, G, A), delay)
    ES = _shift(ewm(S, hl), delay)                           # (T,G)

    # ---- signal: project carriers onto the group-shrunk blended payoff ----
    out = np.zeros(ctx.n)
    for t, ii in enumerate(ctx.day_slices):
        eg = EG[t]
        if not np.isfinite(eg).any():
            continue
        eg = np.nan_to_num(eg)
        egr = np.nan_to_num(EGR[t])
        es = ES[t]
        wg = np.where(np.isfinite(es), es / (es + k), 0.0)   # group-size shrink weight
        B = (1.0 - wg)[:, None] * eg[None, :] + wg[:, None] * egr
        out[ii] = (nm[ii] * B[gcode[ii]]).sum(1)

    sig = pd.Series(out, index=ctx.df.index)
    fs = {"raw": sig,
          "rank": ctx.cs_rank(sig),
          "gr": ctx.group_rank(sig),
          "z": ctx.cs_z(sig)}[final].to_numpy()

    # sign convention: positive train IC on day <= 760
    d = pd.DataFrame({"p": fs, "y": ctx.y, "day": ctx.day_vals}).dropna()
    d = d[d.day <= 760]
    tic = d.groupby("day").apply(lambda g: g.p.corr(g.y)).mean()
    if tic < 0:
        fs = -fs
    return np.nan_to_num(fs)


def gen(ctx) -> dict:
    prep = _prep(ctx)
    out = {}
    # ic02 / s3 : pandas EWM, 1-day label delay
    out["gxb1_hl60_k5_s3"] = _gxpay(
        ctx, prep, prep["y"], hl=60, k=5, final="raw", delay=1, kind="pandas", feat_idx=_BLOCK1)
    out["gxyr_hl60_k60_s3"] = _gxpay(
        ctx, prep, prep["yrank"], hl=60, k=60, final="raw", delay=1, kind="pandas")
    out["gxz_hl20_k8_gr_s3"] = _gxpay(
        ctx, prep, prep["y"], hl=20, k=8, final="gr", delay=1, kind="pandas")
    # alpha_v5 / agent28 : manual zero-init EWM, delay-2 recipe (shift 3 in this framework)
    out["gxpay_shrunk_g_hl60_k30_d2_rank_agent28"] = _gxpay(
        ctx, prep, prep["y"], hl=60, k=30, final="rank", delay=3, kind="manual")
    return out
