#!/bin/bash
set -e
cd /root/autodl-tmp/eg_model/alpha_skill/retrain
export RETRAIN_TAG=confpos50
export RETRAIN_FEATURES=/root/autodl-tmp/eg_model/artifacts/features_confpos50.parquet
export RETRAIN_FEATURE_LIST=/root/autodl-tmp/eg_model/artifacts/feature_list_confpos50.json
export EPOCHS=28 K=32 NSEED=4
run(){ OUT=$1; shift; echo "=== [$(date +%H:%M:%S)] $OUT : $* ==="; env "$@" OUTPRED=$OUT python3 run_transformer_opt.py 2>&1 | grep -aE "valid IC|test IC|done in" | grep -vaE "seed. ep"; }
run s4_o2  RDROP=0.5 DMODEL=176 NL=4 SD=0.15                   # opt2: R-Drop + scale-up
run s4_o3  RDROP=0.5 DMODEL=176 NL=4 SD=0.15 NCS=3            # opt3: + deeper cross-sectional
echo "=== SCREEN2 DONE ==="
