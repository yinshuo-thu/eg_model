#!/usr/bin/env bash
# Convenience wrapper for OOS prediction with the EG model.
#
# Usage:
#   ./run_predict.sh INPUT OUTPUT [MODEL] [DEVICE]
#
#   INPUT   raw panel in the data.csv schema (.csv or .parquet)
#           (day, instrument_id, x_0..x_85, prc1..prc5, vol0, g[, y]); y optional.
#   OUTPUT  where to write predictions (.csv or .parquet): [day, instrument_id, y_hat]
#   MODEL   ensemble (default) | lightgbm | mlp | transformer
#   DEVICE  auto (default) | cpu | cuda      (CUDA failures fall back to CPU automatically)
#
# If INPUT still carries y for the scored rows, IC/IR are printed and an
# <OUTPUT>_metrics.json is written; otherwise predictions only.
#
# Examples:
#   ./run_predict.sh data.csv preds.csv                 # point straight at a data.csv
#   ./run_predict.sh data.csv preds.csv ensemble cpu
#   ./run_predict.sh oos.parquet preds.parquet
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
fi

IN="$1"; OUT="$2"; MODEL="${3:-ensemble}"; DEVICE="${4:-}"
ARGS=(--input "$IN" --output "$OUT" --model "$MODEL")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

echo "[run_predict] model=$MODEL device=${DEVICE:-auto}  $IN -> $OUT"
exec python3 "$HERE/predict.py" "${ARGS[@]}"
