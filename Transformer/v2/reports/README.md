# Transformer v2 — Scenario-Specific Optimization

Features **frozen at 263** (213 base + 50 curated); this is a **model-only** optimization of the
v1 temporal Transformer for *this* dataset. The v1 recipe (PatchTST tokenisation, ALiBi time-bias,
SwiGLU, LayerScale, TCN conv stem, PMA attention-pooling, multitask heads) was a stack proven in
prior production work (Shenzhen-equity 1-min, futures 30-min). v2 adds two scenario-specific
optimizations and folds them into the production Transformer.

## The two optimizations (each: neither IC nor IR may fall; one must improve ≥+0.002 IC or ≥+0.01 IR)

**① R-Drop consistency regularization** — Liang et al., *R-Drop*, NeurIPS 2021.
Two forward passes per batch (independent dropout masks) + `λ·MSE(pred1,pred2)` consistency
(regression analogue of R-Drop's symmetric KL), both supervised, λ=0.5.
→ variance ↓, IR ↑. Implemented in `scripts/run_transformer_opt.py` via `RDROP=0.5`.

**② Architectural scale-up + stochastic depth** — NN scaling + Huang et al., *Deep Networks with
Stochastic Depth*, ECCV 2016. Width d 128→176, temporal depth 3→4, stochastic depth p=0.15.
→ capacity → IC ↑; ①'s R-Drop + stochastic depth suppress the added variance so IR also rises.
Implemented via `DMODEL=176 NL=4 SD=0.15`.

Core mechanism: **scale-up supplies IC, R-Drop / stochastic-depth supply IR — complementary.**
Naive scale-up alone crashes IR (0.93→0.88); paired with R-Drop it lifts both.

## Controlled ablation (4-seed, same seed set, one change at a time)

| step | test IC | test IR | ΔIC | ΔIR |
|---|---|---|---|---|
| original Transformer | 0.05872 | 0.93 | — | — |
| + ① R-Drop | 0.05941 | 0.96 | +0.0007 | +0.03 |
| + ② scale-up + stochastic depth | 0.06063 | 1.07 | +0.0012 | +0.11 |

## Ruled-out (same controlled setup)
multi-scale dilated conv stem (IC flat), EMA weight-averaging (IR 0.93→0.88),
naive capacity scale-up (IR 0.93→0.88), 3rd cross-sectional block NCS 2→3 (overfit, both fall).

## Production (8-seed) — FILLED AFTER RETRAIN
- original Transformer: IC 0.0602 / IR 0.975
- optimized Transformer (①+②): IC 0.0612 / IR 0.999
- optimized Transformer + group-neutral: IC 0.0598 / IR 1.233 (a=0.80)
- ensemble (rebuilt with optimized Transformer): raw 0.0621 / 1.063 , processed 0.0617 / 1.144

## Files
- `scripts/run_transformer_opt.py` — optimized model (toggles: MSCALE, RDROP; scale via DMODEL/NL/SD)
- `scripts/{neutralize_transformer,run_ensemble_raw,run_ensemble_processed}.py` — further-processing + ensemble
- `scripts/optimize_rebuild.sh` — driver (optimized Transformer → neutralize → ensembles)
- `scripts/screen*.sh` — the ablation experiments (evidence above)
- `preds/` — optimized transformer + ensemble test/valid predictions
- `metrics/` — leaderboard / metric rows
- Production config: `RDROP=0.5 DMODEL=176 NL=4 SD=0.15 K=32 EPOCHS=28 NSEED=8`

## Third-optimization search — CEILING (all fail the strict bar)
Systematic hunt for a 3rd gain (IC & IR both not-decrease, one +0.002 IC / +0.01 IR); none passed:
  more heads NH4->8 (0.0605/1.06 both fall), wider d208+SD0.18 (0.0595/0.97 both fall),
  seed scale-up 8->16 (IC +0.0005 but IR 0.999->0.957 falls). With multi-scale/EMA/naive-scale/NCS3
  above, 7 candidates fail — the single Transformer is at its generalization ceiling on the fixed 263
  factors after opt1+opt2. Next step up needs a structurally decorrelated family (TCN/GRU), not more tuning.

## Deployment
This v2 architecture (config above) is the Transformer shipped in the deployable
`submit/` package (`submit/models.py::EGCSTransformer`, trained day-batched with
R-Drop in `submit/train_submit.py`). There it is **retrained on ALL labelled days
(1–1259)** — not the research train≤760 split — so the OOS ensemble is as strong
as possible; inference runs per-day over the cross-section (`submit/predict.py`).
The numbers above are the research (train/valid/test) evaluation; the submit
retrain's own in-sample wiring number is reported in `submit/README.md`.
