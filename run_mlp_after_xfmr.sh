#!/usr/bin/env bash
cd /root/autodl-tmp/eg_model
until grep -aq "xfmr] done" Transformer/log_train.txt 2>/dev/null; do sleep 20; done
echo "[chain] transformer done -> MLP $(date '+%H:%M:%S')"
OMP_NUM_THREADS=8 python ML_single/scripts/run_mlp.py > ML_single/log_mlp.txt 2>&1
echo "[chain] MLP done $(date '+%H:%M:%S')"
