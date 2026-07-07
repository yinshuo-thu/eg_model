#!/bin/bash
set -e
cd /root/autodl-tmp/eg_model/alpha_skill/retrain
export RETRAIN_TAG=confpos50
export RETRAIN_FEATURES=/root/autodl-tmp/eg_model/artifacts/features_confpos50.parquet
export RETRAIN_FEATURE_LIST=/root/autodl-tmp/eg_model/artifacts/feature_list_confpos50.json
export NSEED=8 EPOCHS=28 K=32
echo "=== [$(date +%H:%M:%S)] OPTIMIZED transformer (R-Drop + scale-up d176/depth4 + stochastic depth) ==="
RDROP=0.5 DMODEL=176 NL=4 SD=0.15 OUTPRED=transformer_confpos50 python3 run_transformer_opt.py 2>&1 | grep -aE "valid IC|test IC|done in|MSCALE|RDROP" | grep -vaE "seed. ep"
echo "=== [$(date +%H:%M:%S)] neutralize ==="
python3 neutralize_transformer.py 2>&1 | grep -aE "neut|IC|alpha" | grep -vaE "LightGBM" | tail -3
echo "=== [$(date +%H:%M:%S)] ensembles ==="
python3 run_ensemble_raw.py 2>&1 | grep -aE "ensemble.*IC|raw" | grep -vaE "LightGBM" | tail -3
python3 run_ensemble_processed.py 2>&1 | grep -aE "FINAL|ensemble.*IC|divN" | grep -vaE "LightGBM" | tail -4
echo "=== [$(date +%H:%M:%S)] OPTIMIZE_REBUILD DONE ==="
