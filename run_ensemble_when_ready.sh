#!/usr/bin/env bash
cd /root/autodl-tmp/eg_model
need="ridge elasticnet lightgbm_l1 lightgbm_huber lightgbm_dart mlp transformer"
while :; do
  miss=""
  for m in $need; do [ -f "artifacts/preds/$m.parquet" ] || miss="$miss $m"; done
  [ -z "$miss" ] && break
  sleep 20
done
echo "[ens-chain] all preds ready $(date '+%H:%M:%S'); running ensemble" >> ens_chain.log
python ML_ensemble/scripts/run_ensemble.py >> ens_chain.log 2>&1
echo "[ens-chain] DONE $(date '+%H:%M:%S')" >> ens_chain.log
