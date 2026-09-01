#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/lookaroundnet_native18_tusz_train_dev_v1.json}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a new output directory}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to a cache directory with at least 65 GiB free}
WORKERS=${WORKERS:-4}

export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/lookaroundnet_mpl_cache}

rtk python3 -m src.lookaroundnet_native18.cli prepare-rosters \
  --config "$CONFIG" --run-dir "$RUN_DIR"
rtk python3 -m src.lookaroundnet_native18.cli estimate \
  --config "$CONFIG" --run-dir "$RUN_DIR"
rtk python3 -m src.lookaroundnet_native18.cli preprocess \
  --config "$CONFIG" --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" \
  --split source_train --workers "$WORKERS"
rtk python3 -m src.lookaroundnet_native18.cli preprocess \
  --config "$CONFIG" --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" \
  --split source_dev --workers "$WORKERS"
TRAIN_REQUIRED=1
if [[ -f "$RUN_DIR/training_summary.json" ]] && \
  rtk python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("status") == "complete" and value.get("epochs_completed") == value.get("protocol_epochs") == 200 else 1)' \
  "$RUN_DIR/training_summary.json"; then
  TRAIN_REQUIRED=0
fi
if [[ "$TRAIN_REQUIRED" -eq 1 ]]; then
  TRAIN_RESUME=()
  if [[ -f "$RUN_DIR/checkpoints/last.pt" ]]; then
    TRAIN_RESUME=(--resume)
  fi
  rtk python3 -m src.lookaroundnet_native18.cli train \
    --config "$CONFIG" --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" \
    --device cuda --workers 0 --micro-batch-size 64 "${TRAIN_RESUME[@]}"
fi

OFFICIAL_INVENTORY="$RUN_DIR/predictions/source_dev/native_literal/inventory_full.jsonl"
OFFICIAL_SUMMARY="$RUN_DIR/predictions/source_dev/native_literal/inventory_full.summary.json"
if [[ ! -f "$OFFICIAL_SUMMARY" ]] || ! \
  rtk python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("status") == "complete_prediction_first" else 1)' \
  "$OFFICIAL_SUMMARY"; then
  rtk python3 -m src.lookaroundnet_native18.cli infer-source-dev \
    --config "$CONFIG" --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" \
    --checkpoint "$RUN_DIR/checkpoints/best_segment_f1_at_0.85.pt" \
    --device cuda --batch-size 64 --resume
fi

COVERAGE_INVENTORY="$RUN_DIR/predictions/source_dev/coverage_complete_short_pad/inventory_full.jsonl"
COVERAGE_SUMMARY="$RUN_DIR/predictions/source_dev/coverage_complete_short_pad/inventory_full.summary.json"
if [[ ! -f "$COVERAGE_SUMMARY" ]] || ! \
  rtk python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("status") == "complete_prediction_first" else 1)' \
  "$COVERAGE_SUMMARY"; then
  rtk python3 -m src.lookaroundnet_native18.cli infer-source-dev \
    --config "$CONFIG" --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" \
    --checkpoint "$RUN_DIR/checkpoints/best_segment_f1_at_0.85.pt" \
    --device cuda --batch-size 64 --resume --coverage-complete-short-pad
fi

rtk python3 scripts/score_lookaroundnet_native18_corrected_source_dev.py \
  --config "$CONFIG" --run-dir "$RUN_DIR" \
  --official-inventory "$OFFICIAL_INVENTORY" \
  --coverage-inventory "$COVERAGE_INVENTORY"
rtk python3 -m src.lookaroundnet_native18.cli score-released-literal-bug \
  --config "$CONFIG" --run-dir "$RUN_DIR"
