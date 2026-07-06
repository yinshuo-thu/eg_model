# genalpha factor reconstruction — subagent guide

## Mission
Reconstruct curated alpha factors as **causal** Python functions that regenerate
the **exact saved values** from the raw panel, so the same code works on a future
OOS block. You are assigned ONE family. Ground-truth values exist for every
factor — your job is to match them.

## Success bar
For each assigned factor, `corr(reconstruction, saved_values)` on days 1–1259
must be **≥ 0.99** (aim 1.0). Report the achieved corr per factor. If a factor
resists exact match after real effort, report the best corr + what's ambiguous.

## The core API — `submit/genalpha/core.py`  (class `Ctx`)
```python
import sys; sys.path.insert(0, "/root/autodl-tmp/eg_model/submit/genalpha")
from core import Ctx, XCOLS          # XCOLS = ["x_0",...,"x_85"]
import pandas as pd
cols = ["day","instrument_id","g","vol0","y"]+[f"prc{i}" for i in range(1,6)]+XCOLS
df  = pd.read_parquet("/root/autodl-tmp/eg_model/artifacts/panel_raw.parquet", columns=cols)
ctx = Ctx(df)                        # row order == panel_raw == saved parquet order
```
`Ctx` methods (all causal, all return a pd.Series aligned to ctx.df.index unless noted):
- temporal (per instrument): `ishift(s,L)`, `rmean/rstd/rmax/rmin/rsum(s,w)`, `rskew/rkurt(s,w)`, `ewm(s,halflife)`, `rcov/rvar/rcorr(a,b,w)`
- cross-sectional (per day): `cs_z(s)`, `cs_rank(s)`→[-1,1], `cs_robust_z(s)`, `group_neutral(s,alpha)`, `group_rank(s)`, `decile_bucket_smooth(s,n)`, `residualize_single(s,x)`, `residualize_multi(s,[x0,x1,...])`
- matrix helpers: `xmat()`→(n,86), `csz_by_day(mat)`→per-day z of each col (NaN→0), `daily_payoff(csz_mat)`→(T,86) daily cross-sectional payoff (NaN on unlabelled days)
- attrs: `n, day, inst, grp, day_vals, y, udays, day_slices` (row-index list per day, day-sorted), `day_has_y` (bool per day)

## CAUSALITY — non-negotiable
- temporal ops use only past rows (shift L≥1 for anything label-derived); cross-sectional ops use only the same day.
- **Supervised factors (use y): freeze on OOS.** Any EWM/rolling state driven by y must SKIP its update on days where `ctx.day_has_y[t]` is False (see `_charpay_engine.py`). On days 1–1259 all y present, so this is a no-op for validation but makes OOS leak-free.
- Label delay ≥ 1: a signal for day t may only use payoff/label info through day t−1.

## Proven template — `submit/genalpha/_charpay_engine.py`
`charpay(ctx, carrier_h, std, payoff, wmode, wa, wb, final, feat_idx)` reproduces
the characteristic-payoff family exactly (verified corr 1.0). Read it as the
canonical pattern for supervised, EWM-over-days, freeze-on-OOS factors.

## Sign convention
Many factors are sign-flipped to **positive train IC on days ≤ 760** (see the
charpay engine's tail). Apply the same rule so your sign matches the saved values.

## Your deliverable
Write `submit/genalpha/factors_<FAMILY>.py` exposing:
```python
def gen(ctx) -> dict:   # {factor_name: np.ndarray of length ctx.n, in ctx row order}
    ...
```
NaN→0 the outputs. Then run your own validation loop:
```python
import numpy as np, pandas as pd
saved = pd.read_parquet(SAVED_PATH)["value"].to_numpy()
rec   = out[name]
m = np.isfinite(saved)
corr = np.corrcoef(np.nan_to_num(rec[m]), np.nan_to_num(saved[m]))[0,1]
```
Return a table {name: corr}. Do NOT edit core.py or _charpay_engine.py.

## Formula strings & source
Your prompt lists your factors' names, formula strings, saved-value paths, and
(for alpha_v5) the original generator script. The formula strings are precise —
decode `EWM(hl)`, `delay1`, `offdiag`, `topk`, `roll{w}`, `carrier=`, `std=`,
`payoff=`, term-structure `ts{a}_{b}`, etc. Validate relentlessly against saved.
