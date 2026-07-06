# EG Model — OOS Prediction Package (`submit/`)

A self-contained, **causal** alpha model for the Engineering-Gates take-home. It
regenerates **263 leak-free features** (213 base + 50 curated `genalpha` factors)
from the raw panel, runs a diversity-weighted **ensemble** of LightGBM + a
multi-task MLP + a temporal Transformer, and emits the daily cross-sectional
signal `y_hat`.

**The shipped weights are trained on ALL labelled days (1–1259)** — the whole
`data.csv` — because the real score is out-of-sample. Nothing here is held out;
every labelled row is used so the model is as strong as possible on your hidden
OOS period.

---

## Input schema (identical to `data.csv`)

A raw panel with the training columns — the same fields you already have:

| column | meaning |
|---|---|
| `day` | integer time index (chronological) |
| `instrument_id` | instrument id (consistent across days) |
| `x_0 … x_85` | 86 features |
| `prc1 … prc5` | price-related columns |
| `vol0` | volume column |
| `g` | group id (`-1` = unclassified) |
| `y` | target — **optional / may be withheld**; only used to pick which days to score, never in the computation |

Everything the model needs is derived from `x_0..x_85`, `prc1..prc5`, `vol0`, `g`.
**`y` is never read to produce `y_hat`.**

### History requirement
Features and factors are causal (per-instrument temporal windows + supervised
factors that use only *past* `y`). Each instrument you want scored needs enough
**prior history** in the panel: hard floor **K = 32 days** (the Transformer's
lookback) plus rolling/EWM warm-up. **Pass the entire panel you have** (all of
`1..1259` plus the new OOS rows) — the supervised factors freeze their state at
the last labelled day, so unlabelled OOS rows never read their own or any future
`y`.

---

## Output

`predict(df) -> DataFrame[day, instrument_id, y_hat]`, one row per scored
`(day, instrument_id)`. The metric is the **mean daily cross-sectional Pearson
IC**, so only the **per-day ranking** of `y_hat` matters (it is per-day
z-scored; level/scale are irrelevant).

---

## Quickstart

### CLI
```bash
# default = ensemble
python submit/predict.py --input oos_panel.parquet --output preds.parquet

# a specific sub-model instead of the ensemble
python submit/predict.py --input oos_panel.parquet --output preds.parquet --model transformer

# only score specific days
python submit/predict.py --input panel.parquet --output preds.parquet --days 1260,1261,1262

# force CPU (e.g. if the GPU is busy/small) — or set EG_DEVICE=cpu
python submit/predict.py --input panel.parquet --output preds.parquet --device cpu
```
`--input` / `--output` accept `.parquet` or `.csv`. Device is auto (GPU if
available) unless `--device cpu` / `EG_DEVICE=cpu` is set; the neural nets fit
comfortably on CPU.

### Python
```python
from submit.predict import predict
import pandas as pd

df   = pd.read_parquet("oos_panel.parquet")     # raw schema above, y withheld
yhat = predict(df)                               # ensemble (default)
yhat = predict(df, model="lightgbm")             # or a sub-model
# yhat: columns [day, instrument_id, y_hat]
```

### Choosing the model (`--model` / `model=`)
| value | signal |
|---|---|
| `ensemble` *(default)* | diversity-weighted, group-neutralised blend of all three families — **the recommended model** |
| `lightgbm` | LightGBM-DART seed-bag only (per-day z-scored) |
| `mlp` | multi-task DCN-MLP seed-bag only |
| `transformer` | temporal Transformer seed-bag only |

Omit `--model` to get the ensemble.

---

## How it works (causal, leak-free)

1. **`genalpha.compute_263(panel)`** rebuilds the 263 features from raw:
   - 213 base features (`features_core.py`): per-day cross-sectional z-scores,
     per-instrument temporal lags/momentum/rolling stats, group aggregates &
     leak-free expanding target-encoding, and 12 cross-sectional PCA factors
     (frozen loadings).
   - 50 curated `genalpha` factors (`genalpha/factors_*.py`): characteristic-
     payoff timing, payoff-weighted pair quadratics, IC-weighted composites,
     supervised ridge/tilt, group-payoff, feature-family composites, etc. — every
     one recomputed from raw with **supervised state frozen on unlabelled days**.
2. Each family's frozen models score the rows; each family is **per-day
   z-scored**, blended with **diversity (inverse-correlation) weights**, then
   **group-`g` neutralised** — parameters frozen from the in-sample fit.

All learned parameters (model weights, blend weights, neutralisation α, PCA
loadings, supervised factor states) are **fixed at training time**; no OOS label
is ever used.

---

## Reproduce the training
```bash
python submit/train_submit.py        # retrains on all labelled days -> submit/weights/
```
Reads the raw panel, builds the 263 features (fit mode), trains the LightGBM /
MLP / Transformer seed bags, fits the ensemble config, and writes everything to
`submit/weights/`.

---

## Files
```
submit/
  predict.py            # predict(df, model=...) -> [day, instrument_id, y_hat]  (+ CLI)
  train_submit.py       # retrain all models on all labelled days
  features_core.py      # 213 causal base features (fit/apply, frozen artifacts)
  genalpha/             # the 50 curated factors, regenerated from raw
    __init__.py         #   compute_263(panel) -> 263-feature matrix
    core.py             #   Ctx: causal primitive toolkit over any panel
    factors_*.py        #   the factor families
    confpos50.json      #   the 50 factor names (report-canonical order)
  models.py             # MTMLP + EGTransformer definitions
  weights/              # frozen: lgb_dart_seed*.txt, mlp_seed*.pt, xfmr_seed*.pt,
                        #         feature_artifacts.json, ensemble_config.json
  requirements.txt
```

## Requirements
See `requirements.txt` (numpy, pandas, pyarrow, scikit-learn, lightgbm, torch).
A GPU is used if available for the neural nets; CPU works (slower).
