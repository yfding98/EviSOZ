#!/usr/bin/env bash
set -u

# Recoverable single-GPU queue for label-neutral screening followed by the
# offline Qwen/Qwen3.5-35B-A3B multimodal SOZ review.  Private and TUSZ runs
# use separate output directories so that --resume can safely reuse completed
# records after an interruption.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRIVATE_MAX_EVENTS="${PRIVATE_MAX_EVENTS:-3}"
TUSZ_MAX_EVENTS="${TUSZ_MAX_EVENTS:-3}"
PRIVATE_START_ROW="${PRIVATE_START_ROW:-0}"
TUSZ_START_ROW="${TUSZ_START_ROW:-0}"
RUN_TAG="${RUN_TAG:-20260722}"

cd "$PROJECT_ROOT" || exit 2
rtk mkdir -p outputs/qwen35_batch_logs

rtk .venv-qwen35/bin/python -u scripts/run_qwen35_soz_annotation.py \
  --dataset private \
  --start-row "$PRIVATE_START_ROW" \
  --max-events "$PRIVATE_MAX_EVENTS" \
  --max-review-rounds 2 \
  --local-validation-retries 2 \
  --local-narrative-max-tokens 2400 \
  --output-dir "outputs/qwen35_soz_private_batch_${RUN_TAG}" \
  --resume \
  --no-fail-fast \
  > "outputs/qwen35_batch_logs/private_${RUN_TAG}.log" 2>&1
PRIVATE_EXIT=$?

rtk .venv-qwen35/bin/python -u scripts/run_qwen35_soz_annotation.py \
  --dataset tusz \
  --start-row "$TUSZ_START_ROW" \
  --max-events "$TUSZ_MAX_EVENTS" \
  --max-review-rounds 2 \
  --local-validation-retries 2 \
  --local-narrative-max-tokens 2400 \
  --output-dir "outputs/qwen35_soz_tusz_batch_${RUN_TAG}" \
  --resume \
  --no-fail-fast \
  > "outputs/qwen35_batch_logs/tusz_${RUN_TAG}.log" 2>&1
TUSZ_EXIT=$?

if (( PRIVATE_EXIT != 0 || TUSZ_EXIT != 0 )); then
  exit 1
fi
