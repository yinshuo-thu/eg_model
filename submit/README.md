# EG Model — OOS Prediction Package (`submit/`)

A self-contained, **causal** alpha model for the Engineering-Gates take-home. It
regenerates **250 leak-free features** (200 base + 50 curated `genalpha` factors)
from the raw panel, runs a diversity-weighted **ensemble** of LightGBM + a
multi-task MLP + a **temporal + cross-sectional Transformer** (v2), and emits the
daily cross-sectional signal `y_hat`.

**No feature reads `y` at inference for its own row.** The 200 base features are
pure functions of `x / prc / vol / g` (the autoregressive y-history features were
removed). The 50 curated factors are *supervised*: they learn cross-sectional
weights/signs/payoffs from **past** `y` (≥1-day delay) and freeze that state on
unlabelled days — so a withheld-label OOS row never uses its own or any future `y`.

**No GPU required.** Inference uses a GPU if one is available and **falls back to
CPU automatically** on any CUDA error/OOM (or force it with `--device cpu`).
Progress is printed as it runs.

**The shipped weights are trained on ALL labelled days (1–1259)** — the whole
`data.csv` — because the real score is out-of-sample. Nothing is held out; every
labelled row is used so the model is as strong as possible on your hidden OOS period.

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
| `y` | target — **optional**. If present it is used **only** to (a) auto-pick the days to score and (b) report IC/IR afterwards. It is **never** read to produce a row's own `y_hat`. |

Everything the model needs for `y_hat` is derived from `x_0..x_85`, `prc1..prc5`,
`vol0`, `g` (plus *past* `y` for the frozen supervised-factor weights).

### History requirement
Features and factors are causal (per-instrument temporal windows + supervised
factors that use only *past* `y`). Each instrument you want scored needs enough
**prior history** in the panel: hard floor **K = 32 days** (the Transformer's
lookback) plus rolling/EWM warm-up. **Pass the entire panel you have** (all of
`1..1259` plus the new OOS rows) — the supervised factors freeze their state at
the last labelled day, so unlabelled OOS rows never read their own or any future `y`.

---

## Output & scoring

`predict(df) -> DataFrame[day, instrument_id, y_hat]`, one row per scored
`(day, instrument_id)`. The metric is the **mean daily cross-sectional Pearson
IC**, so only the **per-day ranking** of `y_hat` matters (it is per-day z-scored;
level/scale are irrelevant).

The CLI then checks the input for a realised `y` on the scored rows:
- **`y` present** → it prints the score (**Pearson IC / IR / Spearman IC / % positive
  days**) and writes `<output>_metrics.json` alongside the predictions.
- **`y` absent / withheld** → it writes the predictions only (no IC/IR).

---

## Quickstart

Point it straight at a `data.csv`-format file:

### Easiest — the wrapper
```bash
# ./run_predict.sh INPUT OUTPUT [MODEL] [DEVICE]
./submit/run_predict.sh data.csv preds.csv                    # ensemble, auto device
./submit/run_predict.sh data.csv preds.csv ensemble cpu       # force CPU
./submit/run_predict.sh oos_panel.parquet preds.parquet       # parquet works too
```

### CLI
```bash
# default = ensemble; CSV in, CSV out
python submit/predict.py --input data.csv --output preds.csv

# a specific sub-model instead of the ensemble
python submit/predict.py --input data.csv --output preds.csv --model transformer

# only score specific days (e.g. the new OOS block)
python submit/predict.py --input data.csv --output preds.csv --days 1260,1261,1262

# force CPU (e.g. if the GPU is busy/small) — or set EG_DEVICE=cpu
python submit/predict.py --input data.csv --output preds.csv --device cpu
```
`--input` / `--output` accept `.csv` or `.parquet`. If the input `data.csv`
still carries `y`, you get IC/IR printed and `preds_metrics.json` written; if `y`
is withheld you get predictions only.

### Python
```python
from submit.predict import predict
import pandas as pd

df   = pd.read_csv("data.csv")        # raw schema above (y may be withheld)
yhat = predict(df)                    # ensemble (default)
yhat = predict(df, model="lightgbm")  # or a sub-model
# yhat: columns [day, instrument_id, y_hat]
```

### Choosing the model (`--model` / `model=`)
| value | signal |
|---|---|
| `ensemble` *(default)* | diversity-weighted, group-neutralised blend of all three families — **the recommended model** |
| `lightgbm` | LightGBM-DART seed-bag only (per-day z-scored) |
| `mlp` | multi-task DCN-MLP seed-bag only |
| `transformer` | temporal + cross-sectional Transformer (v2) seed-bag only |

Omit `--model` to get the ensemble.

---

## How it works (causal, leak-free)

1. **`genalpha.compute(panel)`** rebuilds the 250 features from raw:
   - 200 base features (`features_core.py`): per-day cross-sectional z-scores,
     per-instrument temporal lags/momentum/rolling stats of the strongest `x`'s,
     price/volume statistics, a pure-`x` group tilt, and 12 cross-sectional PCA
     factors (frozen loadings). **No feature reads `y`** (the autoregressive
     y-history and past-y group encodings were removed).
   - 50 curated `genalpha` factors (`genalpha/factors_*.py`): characteristic-
     payoff timing, payoff-weighted pair quadratics, IC-weighted composites,
     supervised ridge/tilt, group-payoff, feature-family composites, etc. — each a
     supervised factor recomputed from raw with **its past-`y`-learned state frozen
     on unlabelled days** (so it is computable on a withheld-label OOS block and
     never reads a scored row's own `y`).
2. Three frozen model families score the rows:
   - **LightGBM-DART** seed-bag (robust L1 gradient boosting),
   - **multi-task DCN-MLP** seed-bag (cross-network + sign/magnitude heads),
   - **v2 temporal + cross-sectional Transformer** seed-bag — a per-instrument
     K=32-day temporal encoder (ALiBi time-bias attention, SwiGLU, stochastic
     depth) followed by **cross-sectional attention across instruments within a
     day**, scaled up (d=176, depth 4) and trained with R-Drop consistency
     (Liang et al. 2021). Inference runs **per day** over the day's cross-section.
3. Each family is **per-day z-scored**, blended with **diversity
   (inverse-correlation) weights**, then **group-`g` neutralised** — all
   parameters frozen from the in-sample fit.

All learned parameters (model weights, blend weights, neutralisation α, PCA
loadings, supervised factor states) are **fixed at training time**; no scored
row's own `y` is ever used. On any CUDA failure the neural nets fall back to CPU.

---

## Performance

The honest OOS estimate is the **research ensemble** on the sealed test split
(days 881–1259, never used for selection) — see the report at
<https://autoalpha.cn/eg_model/>. Removing the 13 autoregressive y-history
features costs only a little IC (the supervised factors and the `x`-signal carry
the model), and the diversity-weighted, group-neutralised ensemble still clears
the `y_hat0` baseline (IC 0.056 / IR 1.1).

The shipped weights are trained on **all** labelled days, so an in-sample
wiring-check (days 1200–1259, which the model *did* train on) scores an
optimistically-high IC; treat the report's sealed-test number as the OOS
expectation.

## Reproduce the training
```bash
python submit/train_submit.py                 # retrains on all labelled days -> submit/weights/
EG_RAW=/path/to/data.csv python submit/train_submit.py   # or train straight from a CSV
```
Reads the raw panel, builds the 250 features (fit mode, cached to
`artifacts/genalpha_250.parquet`), trains the LightGBM / MLP / Transformer seed
bags, fits the ensemble config, and writes everything to `submit/weights/`.

---

## Files
```
submit/
  run_predict.sh        # convenience wrapper: ./run_predict.sh IN OUT [MODEL] [DEVICE]
  predict.py            # predict(df, model=..., device=...) -> [day, instrument_id, y_hat] (+ CLI, + IC/IR when y present)
  train_submit.py       # retrain all models on all labelled days (EG_RAW to point at a CSV)
  features_core.py      # 200 causal base features (fit/apply, frozen artifacts; no y-history)
  genalpha/             # the 50 curated supervised factors, regenerated from raw
    __init__.py         #   compute(panel) -> 250-feature matrix
    core.py             #   Ctx: causal primitive toolkit over any panel
    factors_*.py        #   the factor families
    confpos50.json      #   the 50 factor names (report-canonical order)
  models.py             # MTMLP + EGCSTransformer (v2 temporal+cross-sectional)
  weights/              # frozen: lgb_dart_seed*.txt, mlp_seed*.pt, xfmr_seed*.pt,
                        #         feature_artifacts.json, ensemble_config.json
  requirements.txt
```

## Requirements
See `requirements.txt` (numpy, pandas, pyarrow, scikit-learn, lightgbm, torch).
A GPU is used if available for the neural nets; CPU works (slower).
