"""genalpha core: a causal factor-construction context over an ARBITRARY raw panel.

This is a deployable fork of tools/factor_mining_lib.py + alpha_skill/lib/ops.py
(the mining toolkit), generalised so every primitive runs on a panel passed in
at construction time instead of a hard-coded panel_raw.parquet.  It is the shared
engine behind genalpha's 50 curated factors, so the SAME code that produced the
training feature values also regenerates them on a brand-new OOS block.

Causality contract (identical to the mining toolkit):
  * temporal ops shift/roll WITHIN instrument_id, using only past rows;
  * cross-sectional ops (cs_z / cs_rank / group means) use only the SAME day;
  * supervised (label-using) helpers read y with a >=1 day delay and, when y is
    unavailable (OOS rows), FREEZE their state at the last labelled day -- so a
    row on an unlabelled OOS day never reads its own or any future y.

Row order: the context preserves the input df's row order (index reset to
0..n-1).  On the training panel that order == panel_raw.parquet, so factor
values reproduce the saved parquet values position-for-position.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-9
XCOLS = [f"x_{j}" for j in range(86)]


class Ctx:
    """Holds one raw panel + cached day/instrument/group keys, exposes the causal
    primitive toolkit as methods.  Construct once, reuse across all factors."""

    def __init__(self, df: pd.DataFrame):
        # keep caller's row order; work on a 0..n-1 RangeIndex so rolling ops can
        # be restored to row order via .sort_index()
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.day = self.df["day"]
        self.inst = self.df["instrument_id"]
        self.grp = self.df["g"] if "g" in self.df else pd.Series(np.zeros(self.n), name="g")
        self.day_vals = self.day.to_numpy()
        self.y = (self.df["y"].to_numpy("float64") if "y" in self.df
                  else np.full(self.n, np.nan))
        # sorted unique days + per-day row-index slices (in row order)
        self.udays = np.sort(self.day.unique())
        self._day_to_pos = {int(d): k for k, d in enumerate(self.udays)}
        self.day_slices = list(self.df.groupby("day", sort=True).indices.values())
        # y availability per day (a day is "labelled" if any finite y)
        self.day_has_y = np.array(
            [np.isfinite(self.y[ii]).any() for ii in self.day_slices])

    # -------------------------------------------------- accessors
    def col(self, name) -> pd.Series:
        return self.df[name]

    def xmat(self) -> np.ndarray:
        return self.df[XCOLS].to_numpy("float64")

    def pca(self):
        """The 12 cross-sectional PCA controls (pca0_z..pca11_z), aligned to rows.
        In production these are attached to the panel by compute_263 (the SAME
        frozen-loading PCA the 213 base uses), so they extend to OOS causally.
        Dev fallback (standalone factor validation only): read features.parquet."""
        cols = [f"pca{i}_z" for i in range(12)]
        if all(c in self.df.columns for c in cols):
            return [self.df[c].to_numpy("float64") for c in cols]
        # fallback for standalone validation on the 1-1259 dev panel
        import os
        fp = os.path.join(os.path.dirname(__file__), "..", "..",
                          "artifacts", "features.parquet")
        ref = pd.read_parquet(fp, columns=["day", "instrument_id"] + cols)
        ref = ref.set_index(["day", "instrument_id"])
        key = pd.MultiIndex.from_arrays([self.day_vals, self.inst.to_numpy()])
        ref = ref.reindex(key)
        return [np.nan_to_num(ref[c].to_numpy("float64")) for c in cols]

    # -------------------------------------------------- temporal (per instrument)
    def ishift(self, s, L):
        return s.groupby(self.inst, sort=False).shift(L)

    def _roll(self, s, w, fn, minp):
        r = s.groupby(self.inst, sort=False).rolling(w, min_periods=minp)
        return getattr(r, fn)().reset_index(level=0, drop=True).sort_index()

    def rmean(self, s, w, minp=None): return self._roll(s, w, "mean", minp or max(2, w // 2))
    def rstd(self, s, w, minp=None):  return self._roll(s, w, "std",  minp or max(3, w // 2))
    def rmax(self, s, w, minp=None):  return self._roll(s, w, "max",  minp or max(2, w // 2))
    def rmin(self, s, w, minp=None):  return self._roll(s, w, "min",  minp or max(2, w // 2))
    def rsum(self, s, w, minp=None):  return self._roll(s, w, "sum",  minp or max(2, w // 2))
    def rskew(self, s, w):            return self._roll(s, w, "skew", max(6, w // 2))
    def rkurt(self, s, w):            return self._roll(s, w, "kurt", max(6, w // 2))

    def ewm(self, s, halflife):
        return s.groupby(self.inst, sort=False).transform(
            lambda v: v.ewm(halflife=halflife, min_periods=2).mean())

    def rcov(self, a, b, w): return self.rmean(a * b, w) - self.rmean(a, w) * self.rmean(b, w)
    def rvar(self, a, w):    return self.rcov(a, a, w)
    def rcorr(self, a, b, w):
        return self.rcov(a, b, w) / (np.sqrt(np.clip(self.rvar(a, w) * self.rvar(b, w), 1e-18, None)) + EPS)

    # -------------------------------------------------- cross-sectional (per day)
    def cs_z(self, s):
        g = s.groupby(self.day, sort=False)
        return (s - g.transform("mean")) / (g.transform("std") + EPS)

    def cs_rank(self, s):
        r = s.groupby(self.day, sort=False).rank(pct=True)
        return (r - 0.5) * 2.0

    def cs_robust_z(self, s):
        med = s.groupby(self.day, sort=False).transform("median")
        mad = (s - med).abs().groupby(self.day, sort=False).transform("median")
        return (s - med) / (1.4826 * mad + EPS)

    def group_neutral(self, s, alpha=1.0):
        gm = s.groupby([self.day, self.grp], sort=False).transform("mean")
        return self.cs_z(s - alpha * gm)

    def group_rank(self, s):
        r = s.groupby([self.day, self.grp], sort=False).rank(pct=True)
        return (r - 0.5) * 2.0

    def decile_bucket_smooth(self, s, n=10):
        rk = s.groupby(self.day, sort=False).rank(pct=True, method="first")
        bucket = np.minimum((rk.to_numpy() * n).astype(int), n - 1)
        tmp = pd.DataFrame({"day": self.day_vals, "b": bucket, "s": s.to_numpy()})
        bmean = tmp.groupby(["day", "b"], sort=False)["s"].transform("mean")
        return pd.Series(bmean.to_numpy(), index=s.index)

    def residualize_single(self, s, x):
        sm = s.groupby(self.day, sort=False).transform("mean")
        xm = x.groupby(self.day, sort=False).transform("mean")
        cov = ((s - sm) * (x - xm)).groupby(self.day, sort=False).transform("mean")
        varx = ((x - xm) ** 2).groupby(self.day, sort=False).transform("mean")
        beta = cov / (varx + EPS)
        return self.cs_z((s - sm) - beta * (x - xm))

    def residualize_multi(self, s, xs):
        yv = np.asarray(s, dtype="float64")
        out = np.full(len(yv), np.nan)
        tmp = pd.DataFrame({"day": self.day_vals, "y": yv,
                            **{f"c{i}": np.asarray(x, "float64") for i, x in enumerate(xs)}})
        cc = [f"c{i}" for i in range(len(xs))]
        for _, idx in tmp.groupby("day").groups.items():
            ii = idx.to_numpy()
            Xd = np.column_stack([np.ones(len(ii))] + [tmp.loc[ii, c].to_numpy() for c in cc])
            yd = tmp.loc[ii, "y"].to_numpy()
            good = np.isfinite(Xd).all(1) & np.isfinite(yd)
            if good.sum() < len(cc) + 2:
                continue
            beta, *_ = np.linalg.lstsq(Xd[good], yd[good], rcond=None)
            out[ii] = yd - Xd @ beta
        return self.cs_z(pd.Series(out, index=s.index))

    # -------------------------------------------------- per-day cross-sectional z of a matrix
    def csz_by_day(self, mat: np.ndarray) -> np.ndarray:
        """Per-day cross-sectional z-score of every column of `mat` (n,J); NaN->0."""
        out = np.zeros_like(mat, dtype=np.float64)
        for ii in self.day_slices:
            a = mat[ii]
            good = np.isfinite(a)
            nn = np.maximum(good.sum(0), 1)
            clean = np.where(good, a, 0.0)
            mean = clean.sum(0) / nn
            var = np.where(good, (a - mean) ** 2, 0.0).sum(0) / nn
            sd = np.sqrt(var)
            out[ii] = np.nan_to_num((a - mean) / np.where(sd > 1e-9, sd, 1.0))
        return out

    # -------------------------------------------------- supervised daily payoff (charpay engine)
    def daily_payoff(self, csz_mat: np.ndarray) -> np.ndarray:
        """payoff[t, j] = mean_i( csz_mat[i,j] * y[i] ) over instruments on day t
        (only rows with finite y contribute).  Rows on unlabelled days -> NaN so
        callers can FREEZE their EWM there (OOS-safe)."""
        T, J = len(self.day_slices), csz_mat.shape[1]
        payoff = np.full((T, J), np.nan)
        for t, ii in enumerate(self.day_slices):
            yy = self.y[ii]; good = np.isfinite(yy)
            if good.any():
                payoff[t] = np.mean(csz_mat[ii][good] * yy[good, None], axis=0)
        return payoff

    def scatter_days(self, per_day_row_mat: dict) -> np.ndarray:
        """(helper) build an (n,) array from a {row_slice_index -> vector} mapping.
        Not typically needed; day loops usually write into a preallocated (n,)."""
        raise NotImplementedError
