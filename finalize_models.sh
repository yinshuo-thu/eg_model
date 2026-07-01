#!/usr/bin/env bash
cd /root/autodl-tmp/eg_model
until grep -aq "xfmr] done" Transformer/log_train8.txt 2>/dev/null; do sleep 20; done
echo "[final] transformer8 done $(date '+%H:%M:%S')" >> finalize.log
python ML_ensemble/scripts/run_ensemble.py >> finalize.log 2>&1
python tools/gen_figs.py >> finalize.log 2>&1
echo "[final] DONE $(date '+%H:%M:%S')" >> finalize.log
echo "--- transformer ---" >> finalize.log; cat Transformer/metrics/leaderboard.csv >> finalize.log
echo "--- ensemble ---" >> finalize.log; cat artifacts/metrics/ensemble_leaderboard.csv >> finalize.log
