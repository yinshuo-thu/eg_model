"""Payoff-weighted pair-quadratic (pw) factor family — z^T W(t) z.

For each instrument i on day t the signal is a quadratic form in the per-day
normalised carrier vector z[i] (86 features):

    F[i,t] = sum_{j != k} W_jk(t) * z_j[i,t] * z_k[i,t]        (offdiag, symmetric)

W(t) is an 86x86 payoff-weighted kernel accumulated CAUSALLY over past days.
The daily payoff-pair matrix is a label-weighted outer product,

    G_jk[t] = mean_i( z_j[i,t] * z_k[i,t] * label[i,t] )       (mean over all rows)

exponentially smoothed across days (pandas `adjust=False` convention: the state
is initialised at the first observation, then
    state = alpha*G_t + (1-alpha)*state,  alpha = 1 - 0.5**(1/halflife) ),
and the kernel used on day t is the smoothed state built from days STRICTLY
before t (delay 1).  On unlabelled (OOS) days the state FREEZES so a row never
reads its own / future y.  The diagonal of W is zeroed (offdiag), and optionally
only the top-k off-diagonal pairs (by |W|, upper-triangle threshold applied
symmetrically) are kept.

carrier z in {z: cs_z, rank: cs_rank->[-1,1], rob: clip(cs_robust_z,+/-5)},
applied per feature per day (NaN->0).  The SAME carrier is used to form G and the
quadratic form.
label in {raw: y - day_mean(y), sign: sign(y - day_mean(y)), rank: cs_rank(y)}.

The factor value is the raw quadratic form F, sign-flipped so the mean daily
Pearson IC vs y on train days (day<=760) is positive, then NaN->0.

Exact reconstruction of the ic02 `s5` pw generator (verified against the saved
values).  The kernel tuple in the factor name is (mode, halflife, topk): e.g.
`ewm15_150` = EWM halflife 15, top-150 pairs (NOT a fast/slow term structure).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core import XCOLS

EPS = 1e-9
NF = 86
TRAIN_END = 760

# name -> (carrier, label, halflife, topk_or_None)
PW_SPECS = {
    "pw_pair_hl10_kall_s5":       ("z",    "raw",  10, None),
    "pw_pair_hl60_k100_s5":       ("z",    "raw",  60, 100),
    "pw_rank_rank_ewm60_all_s5":  ("rank", "rank", 60, None),
    "pw_z_rank_ewm15_150_s5":     ("z",    "rank", 15, 150),
    "pw_rob_raw_ewm60_all_s5":    ("rob",  "raw",  60, None),
    "pw_z_sign_ewm15_150_s5":     ("z",    "sign", 15, 150),
    "pw_rob_sign_hl60_kall_s5":   ("rob",  "sign", 60, None),
    "pw_rank_sign_hl60_kall_s5":  ("rank", "sign", 60, None),
    "pw_rank_sign_hl20_k150_s5":  ("rank", "sign", 20, 150),
}


# ------------------------------------------------------------------ carriers
def _carrier_mat(ctx, kind):
    """Per-day, per-feature normalised (n,86) carrier matrix; NaN->0.

    Matches alpha_skill ops: cs_z uses pandas std (ddof=1)+EPS; cs_rank ->[-1,1];
    cs_robust_z clipped to +/-5.  pandas groupby-transform skips NaN per column."""
    Xdf = ctx.df[XCOLS]
    g = Xdf.groupby(ctx.day, sort=False)
    if kind == "z":
        mean = g.transform("mean")
        std = g.transform("std")
        M = (Xdf - mean) / (std + EPS)
    elif kind == "rank":
        r = g.rank(pct=True)
        M = (r - 0.5) * 2.0
    elif kind == "rob":
        med = g.transform("median")
        mad = (Xdf - med).abs().groupby(ctx.day, sort=False).transform("median")
        M = (Xdf - med) / (1.4826 * mad + EPS)
    else:
        raise ValueError(kind)
    M = np.nan_to_num(M.to_numpy("float64"))
    if kind == "rob":
        M = np.clip(M, -5.0, 5.0)
    return M


def _label_vec(ctx, kind):
    """Per-day label vector; NaN->0.  raw = y - day_mean(y); sign = sign(raw);
    rank = cs_rank(y) in [-1,1]."""
    y = ctx.df["y"]
    ymean = y.groupby(ctx.day, sort=False).transform("mean")
    yc = np.nan_to_num((y - ymean).to_numpy("float64"))
    if kind == "raw":
        return yc
    if kind == "sign":
        return np.sign(yc)
    if kind == "rank":
        r = y.groupby(ctx.day, sort=False).rank(pct=True)
        return np.nan_to_num(((r - 0.5) * 2.0).to_numpy("float64"))
    raise ValueError(kind)


def _daily_pair_matrix(ctx, Z, lab):
    """G[t] = (Z_t * label_t).T @ Z_t / m  (m = rows that day) -> (T,86,86)."""
    T = len(ctx.day_slices)
    G = np.zeros((T, NF, NF), dtype="float64")
    for t, ii in enumerate(ctx.day_slices):
        Zd = Z[ii]
        m = len(ii)
        if m:
            G[t] = (Zd * lab[ii][:, None]).T @ Zd / m
    return G


def _quad_signal(ctx, Z, G, halflife, topk):
    """EWM-smooth G causally (delay 1, freeze on OOS) and evaluate the offdiag
    (optionally top-k) quadratic form F = z^T W z per row."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    iu = np.triu_indices(NF, k=1) if topk is not None else None
    state = None
    out = np.zeros(ctx.n, dtype="float64")
    for t, ii in enumerate(ctx.day_slices):
        if state is not None:                       # W = EWM state through t-1
            Wd = state.copy()
            np.fill_diagonal(Wd, 0.0)
            if topk is not None:
                vals = np.abs(Wd[iu])
                if topk < vals.size:
                    thr = np.partition(vals, vals.size - topk)[vals.size - topk]
                    mask = np.abs(Wd) >= thr
                    np.fill_diagonal(mask, False)
                    Wd = Wd * mask
            Zd = Z[ii]
            out[ii] = np.einsum("si,ij,sj->s", Zd, Wd, Zd, optimize=True)
        if ctx.day_has_y[t]:                        # update state, FREEZE on OOS
            g = G[t]
            state = g.copy() if state is None else alpha * g + (1.0 - alpha) * state
    return out


def _sign_flip(ctx, F):
    """Flip so mean daily Pearson IC vs y on day<=760 is positive (ops.train_ic)."""
    y = ctx.y
    days = ctx.udays
    trmask = days <= TRAIN_END
    ics = []
    for t, ii in enumerate(ctx.day_slices):
        if not trmask[t]:
            continue
        a = F[ii]; b = y[ii]
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() > 2:
            aa = a[good]; bb = b[good]
            if np.std(aa) > 1e-9 and np.std(bb) > 1e-9:
                ics.append(np.corrcoef(aa, bb)[0, 1])
    tic = np.nanmean(ics) if ics else 0.0
    if tic < 0:
        F = -F
    return np.nan_to_num(F)


def gen(ctx) -> dict:
    """Return {factor_name: np.ndarray(len==ctx.n)} for the pw family."""
    out = {}
    Zcache = {}   # carrier -> (n,86)
    Gcache = {}   # (carrier,label) -> G
    for name, (carrier, label, halflife, topk) in PW_SPECS.items():
        if carrier not in Zcache:
            Zcache[carrier] = _carrier_mat(ctx, carrier)
        Z = Zcache[carrier]
        key = (carrier, label)
        if key not in Gcache:
            Gcache[key] = _daily_pair_matrix(ctx, Z, _label_vec(ctx, label))
        F = _quad_signal(ctx, Z, Gcache[key], halflife, topk)
        out[name] = _sign_flip(ctx, F)
    return out
