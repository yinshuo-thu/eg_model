# Engineering-Gates OOS prediction package

A deployable, self-contained ensemble that predicts a weak forward return signal
`y` on a **daily cross-sectional panel** of instruments. It is the final,
out-of-sample (OOS) build of the Engineering-Gates project: the same models and
winning blend recipe used in research, **retrained on ALL labelled days (1..1259)**
so that nothing is held out from the fit.

The scored metric is the **mean daily cross-sectional Pearson IC** — the average,
over days, of the within-day correlation between the prediction and the realised
`y`. Only the *cross-sectional ranking per day* matters, so the package returns a
per-day z-scored, blended and group-neutralised score `y_hat`.

## What's inside

Three base models are blended:

| model        | what it is                                                            |
|--------------|-----------------------------------------------------------------------|
| LightGBM     | tuned DART gradient-boosting, 5-seed bag (`weights/lgb_dart_seed*.txt`)|
| MLP          | multi-task DCN-style tabular net, 6-seed bag (`weights/mlp_seed*.pt`)  |
| Transformer  | v3-lineage daily temporal Transformer, K=32 lookback, 8-seed bag (`weights/xfmr_seed*.pt`) |

**Ensemble recipe** (fitted on the in-sample predictions, saved in
`weights/ensemble_config.json`):

1. per-day z-score each base prediction,
2. blend with **inverse-correlation "diversity" weights** over
   {lightgbm, mlp, transformer},
3. **partial group-`g` neutralisation**: `y_hat = perday_z(blend − α·groupmean_g(blend))`.

All 213 input features are **leak-free and causal** (temporal ops use only past
days; cross-sectional ops use only the same day). The feature recipe's fit-time
state (the strongest-`x` ranking and the 12 PCA loadings) is frozen in
`weights/feature_artifacts.json`, so OOS rows get exactly the features the models
were trained on.

## Install

```bash
pip install -r submit/requirements.txt
```

Requires: pandas, numpy, pyarrow, lightgbm, torch, scikit-learn. A CUDA GPU is
used automatically if available (falls back to CPU).

## Input schema

A **raw panel** DataFrame / parquet / csv with the *same schema as training*:

```
day            int    trading day index
instrument_id  int    instrument id
x_0 .. x_85    float  86 raw features
prc1 .. prc5   float  price snapshots
vol0           float  volume
g              int    group id
y              float  forward return  (OPTIONAL — may be absent/NaN for the days you predict)
```

### History requirement (important)

The features are causal and the Transformer looks back **K=32 days**, so to
predict day `t` you must also supply the **prior days** for those instruments.

* Provide **at least ~60 prior days of history** before the first day you want
  predicted (Transformer needs 32 days; the rolling features warm up over ~20).
* **More history is better.** Passing the entire available history makes the
  group target-encoding feature exact. Instruments/days without a full K-day
  lookback are skipped (a warning is printed).

## Usage

### Python API

```python
import pandas as pd
from submit.predict import predict          # or: import predict (from inside submit/)

panel = pd.read_parquet("my_panel.parquet") # history + the days to predict
preds = predict(panel)                       # -> DataFrame[day, instrument_id, y_hat]
```

Choosing which days to score:

* `predict(panel)` — if `y` is present with both observed **and** missing values,
  it scores the days that contain missing `y` (the OOS rows). If there is no `y`
  (or it is all-NaN), it scores **every** day that has a full K-day lookback.
* `predict(panel, predict_days=[1260, 1261])` — score exactly those days.

### CLI

```bash
python submit/predict.py --input my_panel.parquet --output preds.parquet
# optional: --days 1260,1261    --weights submit/weights
```

`--input`/`--output` accept `.parquet` or `.csv`.

## Output

One row per predicted `(day, instrument_id)`:

```
day  instrument_id  y_hat
```

`y_hat` is the final ensemble signal — **per-day z-scored, diversity-blended and
group-neutralised**. Higher `y_hat` = more positive expected forward return.
Scores are only comparable *within a day* (that is what the daily-IC metric
rewards); there is no meaningful cross-day scale.

## Re-training

To rebuild every weight from the raw panel (`artifacts/panel_raw.parquet`):

```bash
python submit/train_submit.py
```

This recomputes the 213 causal features (fit mode), trains the LightGBM DART bag,
the MLP bag and the Transformer bag on **all labelled days**, fits the diversity
weights + neutralisation-α on the in-sample predictions, and writes everything to
`submit/weights/`. A GPU is strongly recommended. A tiny random slice of days is
held out **only** as an early-stopping monitor for the neural nets; LightGBM uses
a fixed round budget.

## Files

```
submit/
├── predict.py             callable inference API + CLI
├── train_submit.py        full-in-sample retraining
├── features_core.py       causal 213-feature engineering (fit / apply)
├── models.py              MLP + Transformer definitions
├── requirements.txt
├── README.md
└── weights/
    ├── feature_artifacts.json   top_x ranking + PCA loadings + feature order
    ├── ensemble_config.json     diversity weights + neutralisation α + K
    ├── lgb_dart_seed{0..4}.txt  LightGBM boosters
    ├── mlp_seed{0..5}.pt        MLP state_dicts
    └── xfmr_seed{0..7}.pt       Transformer state_dicts
```
