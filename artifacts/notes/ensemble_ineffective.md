# Ensemble v2 — ineffective / rejected methods

Baseline to beat: equal-weight 3-family blend — test IC 0.06021 / IR 0.931 (valid 0.06542 / 0.956).
Gate: test IC > 0.062, OR (test IC >= 0.06021 AND test IR > 1.10).
WINNER: divN (5-member inverse-correlation diversity) + partial group-`g` neutralization (alpha=0.35)
        — test IC 0.06050 / IR 1.135 (valid 0.06540 / 1.149). Clears the IR route.

Dead-ends (all use per-day z-scored bases; valid/test via common.daily_ic):

- valid-IC-optimized simplex weights (random + grid search over {lgb,mlp,transformer}) — test IC 0.05927 / IR 0.909 — overfits the valid->test regime shift (piles onto mlp, the best valid model); worse than equal on test.
- valid-IR-optimized simplex weights — test IC 0.05960 / IR 0.935 — same overfit; no IR gain on test vs equal.
- full group-`g` neutralization (alpha=1.0) on equal blend — test IC 0.05840 / IR 1.248 — IR soars but IC falls below the 0.06021 gate floor; the group tilt carries real cross-sectional IC, so only PARTIAL neutralization (alpha~0.35) keeps IC up while lifting IR.
- adding linear models (ridge/elasticnet avg) at small weight (0.05–0.20) to the diversity blend — test IC 0.06008 -> 0.05859 (monotone down) / IR ~0.93 — linear bases are too weak (IC ~0.04) and drag IC with no IR benefit.
- rank/percentile-transform bases before blending (instead of z-score) — test IC 0.052 / IR ~0.79 — discards magnitude information that Pearson IC rewards; large IC loss.
- per-day winsorize/clip the blend (|z|<2.5–3 or 1/99 pct) before neutralization — test IC ~0.0586 / IR ~1.09 — clips the informative tails of an already-standardized signal; IC drops well below the floor.
- PCA-neutralize the blend against the top 1–3 cross-sectional PCs of the base-pred matrix (full removal) — test IC 0.012 / IR 0.17 — the top PC IS essentially the consensus signal, so removing it destroys the alpha. (A mild 0.3x removal stacked AFTER g-neutralization does work: test IC 0.06044 / IR 1.201 — extra IR but per-day SVD adds cost+overfit risk, so not adopted.)
- temporal EMA smoothing of the final per-instrument signal across days (lam 0.1–0.3) — lam=0.1: test IC 0.06055 / IR 1.131 (vs 0.06044/1.150 unsmoothed) — net wash (tiny IC up, IR down) and adds a cross-day dependency / leakage surface; not worth the complexity.
