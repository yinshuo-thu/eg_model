"""Reconstruction of the `sc` (x-block IC-combo) and `ff` (feature-family
composite) factor families as causal, deployable functions over a `core.Ctx`.

Verified corr vs saved parquet on days 1-1259 (all labelled): 1.0 for all six.

  SC family (source: alpha_skill/lib/s10_register.py; ic02 registry):
    signal = sum_{j in subset} cs_z_day(x_j) * W[day-1, j]
    W[t,j] = EWM(halflife=hl, causal shift 1) of the daily cross-sectional
             pearson-IC series  IC[t,j] = mean_i( cs_z_day(x_j)[i] * z_day(y)[i] ).
    - sc_halfa_s10            : subset = x_0..x_42,        hl=40
    - sc_even_s10             : subset = even idx x_0,2,..84, hl=40
    - sc_halfa_tx50_b09_s10   : global_z(raw halfa combo) + 0.9*cs_z_day(x_50)
    Final = sign-fix to positive train-IC (day<=760), NaN->0.  No other norm.

  FF family (source: alpha_skill/scripts/factor_factory.py; alpha_v5 registry):
    pcaorth_x_j = ctx.residualize_multi(x_j, [pca0_z..pca11_z])  (= cs_z of the
                  per-day OLS residual of RAW x_j on the 12 PCA base cols).
    - ff_comp_eq3_x3_45_47   : mean over j in {3,45,47}  of sign-fixed cs_rank(pcaorth_x_j)
    - ff_comp_eq4_x0_3_32_47 : mean over j in {0,3,32,47} of sign-fixed cs_rank(pcaorth_x_j)
    - ff_pcaorth_x3_bucket   : decile_bucket_smooth(pcaorth_x_3, 10)
    Each per-j cs_rank is sign-fixed to +train-IC before averaging; the composite
    (and the bucket) is then sign-fixed to +train-IC; NaN->0.

Supervision (y-driven EWM weights, per-j sign fixes) is frozen on unlabelled
days so the module is leak-free on a future OOS block; on days 1-1259 every day
is labelled so this is a no-op for validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRAIN_END = 760


def gen(ctx) -> dict:
    N = ctx.n
    yv = ctx.y
    slices = ctx.day_slices
    # day value of each day-block (blocks are day-sorted ascending)
    blk_day = np.array([int(ctx.day_vals[ii[0]]) for ii in slices])
    blk_labelled = np.asarray(ctx.day_has_y, dtype=bool)
    train_blocks = [k for k in range(len(slices)) if blk_day[k] <= TRAIN_END]

    # ---- train IC on day<=760 (population moments; matches ops._fast_ic) -----
    def train_ic(vec):
        vec = np.asarray(vec, dtype="float64")
        acc = []
        for k in train_blocks:
            ii = slices[k]
            v = vec[ii]; y = yv[ii]
            good = np.isfinite(v) & np.isfinite(y)
            if good.sum() < 3:
                continue
            v = v[good]; y = y[good]
            mv = v.mean(); my = y.mean()
            cov = (v * y).mean() - mv * my
            varv = (v * v).mean() - mv * mv
            vary = (y * y).mean() - my * my
            den = np.sqrt(max(varv * vary, 1e-18))
            if den > 1e-9:
                acc.append(cov / den)
        return float(np.nanmean(acc)) if acc else np.nan

    def signfix(vec):
        t = train_ic(vec)
        return -np.asarray(vec, "float64") if (np.isfinite(t) and t < 0) else np.asarray(vec, "float64")

    # =====================================================================
    # SC family
    # =====================================================================
    Z = ctx.csz_by_day(ctx.xmat())                       # (N,86) per-day ddof0 z, NaN->0
    D = len(slices)
    IC = np.zeros((D, 86))
    for k, ii in enumerate(slices):
        if not blk_labelled[k]:
            continue
        yb = yv[ii]
        good = np.isfinite(yb)
        yb2 = np.where(good, yb, 0.0)
        mu = yb2[good].mean(); sd = yb2[good].std()
        zy = np.where(good, (yb - mu) / (sd if sd > 1e-9 else 1.0), 0.0)
        IC[k] = (Z[ii] * zy[:, None]).mean(0)

    def ewm_weights(M, hl, shift=1):
        """causal EWM over day-blocks, FROZEN on unlabelled days, then shifted."""
        a = 1 - 0.5 ** (1.0 / hl)
        es = np.zeros_like(M)
        cur = None
        for t in range(len(M)):
            if blk_labelled[t]:
                cur = M[t].copy() if cur is None else a * M[t] + (1 - a) * cur
            es[t] = cur if cur is not None else 0.0
        W = np.zeros_like(M)
        if shift < len(M):
            W[shift:] = es[:-shift]
        return W

    W40 = ewm_weights(IC, 40, 1)
    rdi = np.empty(N, dtype=int)
    for k, ii in enumerate(slices):
        rdi[ii] = k
    Wrow = W40[rdi]

    def combo(subset):
        mask = np.zeros(86, bool); mask[list(subset)] = True
        return (Z[:, mask] * Wrow[:, mask]).sum(1)

    def global_z(c):
        c = np.asarray(c, dtype="float64"); m = c - c.mean()
        return m / (c.std() + 1e-12)

    halfa_raw = combo(range(0, 43))
    sc_halfa = np.nan_to_num(signfix(halfa_raw))
    sc_even = np.nan_to_num(signfix(combo(range(0, 86, 2))))
    sc_tilt = np.nan_to_num(signfix(global_z(halfa_raw) + 0.9 * Z[:, 50]))

    # =====================================================================
    # FF family
    # =====================================================================
    pcas = ctx.pca()   # live, OOS-portable PCA controls (attached by compute_263)

    def pcaorth(j):
        return ctx.residualize_multi(ctx.col(f"x_{j}"), pcas)

    def signed_rank(j):
        return signfix(ctx.cs_rank(pcaorth(j)).to_numpy())

    def comp(js):
        c = sum(signed_rank(j) for j in js) / len(js)
        return np.nan_to_num(signfix(c))

    ff_eq3 = comp([3, 45, 47])
    ff_eq4 = comp([0, 3, 32, 47])
    ff_bucket = np.nan_to_num(signfix(ctx.decile_bucket_smooth(pcaorth(3), 10).to_numpy()))

    return {
        "sc_halfa_s10": sc_halfa,
        "sc_even_s10": sc_even,
        "sc_halfa_tx50_b09_s10": sc_tilt,
        "ff_comp_eq3_x3_45_47": ff_eq3,
        "ff_comp_eq4_x0_3_32_47": ff_eq4,
        "ff_pcaorth_x3_bucket": ff_bucket,
    }
