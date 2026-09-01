#!/usr/bin/env bash
set -u

# Full single-GPU Qwen3.5 SOZ pipeline.
#
# Direct event-local review:
#   private, TUSZ FNSZ, CHB-MIT
#
# Label-neutral full-record preselection followed by local Qwen review:
#   TUEV, TUEP
#
# VEPISET is intentionally excluded: its short IED snippets can enrich motif
# pretraining but cannot support t0/spread/end SOZ labels.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_TAG="${RUN_TAG:-20260723}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/qwen35_all_data_$RUN_TAG}"

RUN_PRIVATE="${RUN_PRIVATE:-1}"
RUN_TUSZ="${RUN_TUSZ:-1}"
RUN_CHBMIT="${RUN_CHBMIT:-1}"
RUN_TUEV="${RUN_TUEV:-1}"
RUN_TUEP="${RUN_TUEP:-1}"
AUTO_RETRY="${AUTO_RETRY:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"

PRIVATE_START_ROW="${PRIVATE_START_ROW:-0}"
TUSZ_START_ROW="${TUSZ_START_ROW:-0}"
CHBMIT_START_ROW="${CHBMIT_START_ROW:-0}"
MAX_EVENTS="${MAX_EVENTS:--1}"
UNANCHORED_MAX_FILES="${UNANCHORED_MAX_FILES:--1}"
UNANCHORED_MAX_PER_FILE="${UNANCHORED_MAX_PER_FILE:-2}"
LOCAL_VALIDATION_RETRIES="${LOCAL_VALIDATION_RETRIES:-2}"

CHBMIT_ROOT="${CHBMIT_ROOT:-/mnt/hd1/dyf/dataset/CHB-MIT}"
TUEV_ROOT="${TUEV_ROOT:-/mnt/hd1/dyf/dataset/tuh_eeg_events}"
TUEP_ROOT="${TUEP_ROOT:-/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy}"
CHBMIT_MANIFEST="${CHBMIT_MANIFEST:-$PROJECT_ROOT/outputs/llm_soz_public_manifests/chbmit_seizure_anchors.csv}"

PYTHON="$PROJECT_ROOT/.venv-qwen35/bin/python"
ANNOTATOR="$PROJECT_ROOT/scripts/run_qwen35_soz_annotation.py"
SCANNER="$PROJECT_ROOT/scripts/scan_public_soz_candidates.py"
SELECTOR="$PROJECT_ROOT/scripts/select_full_record_soz_review_queue.py"
INDEXER="$PROJECT_ROOT/scripts/index_qwen35_results.py"
RETRY="$PROJECT_ROOT/scripts/retry_qwen35_failures.py"
PROGRESS="$PROJECT_ROOT/scripts/watch_qwen35_progress.py"

PIPELINE_FAILURES=0

cd "$PROJECT_ROOT" || exit 2
rtk mkdir -p "$RUN_ROOT/logs"

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  rtk "$PYTHON" -u "$PROGRESS" \
    --run-root "$RUN_ROOT" \
    --interval "$PROGRESS_INTERVAL" \
    --watch-pid "$$" &
fi

run_logged() {
  local name="$1"
  shift
  "$@" > "$RUN_ROOT/logs/$name.log" 2>&1
  local returncode=$?
  if (( returncode != 0 )); then
    PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
  fi
  return 0
}

refresh_index() {
  rtk "$PYTHON" "$INDEXER" \
    --run-root "$RUN_ROOT" \
    --output-dir "$RUN_ROOT/indexes" \
    >> "$RUN_ROOT/logs/index.log" 2>&1
}

# Preserve a usable success/failure ledger even when a long all-data run is
# interrupted between datasets.
trap 'refresh_index' EXIT

if [[ "$RUN_PRIVATE" == "1" ]]; then
  run_logged private \
    rtk "$PYTHON" -u "$ANNOTATOR" \
      --dataset private \
      --start-row "$PRIVATE_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --max-review-rounds 2 \
      --local-validation-retries "$LOCAL_VALIDATION_RETRIES" \
      --local-narrative-max-tokens 2400 \
      --output-dir "$RUN_ROOT/private" \
      --resume \
      --no-fail-fast
  refresh_index
fi

if [[ "$RUN_TUSZ" == "1" ]]; then
  run_logged tusz \
    rtk "$PYTHON" -u "$ANNOTATOR" \
      --dataset tusz \
      --start-row "$TUSZ_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --max-review-rounds 2 \
      --local-validation-retries "$LOCAL_VALIDATION_RETRIES" \
      --local-narrative-max-tokens 2400 \
      --output-dir "$RUN_ROOT/tusz" \
      --resume \
      --no-fail-fast
  refresh_index
fi

if [[ "$RUN_CHBMIT" == "1" ]]; then
  run_logged chbmit \
    rtk "$PYTHON" -u "$ANNOTATOR" \
      --dataset generic \
      --dataset-name chbmit \
      --manifest "$CHBMIT_MANIFEST" \
      --eeg-root "$CHBMIT_ROOT" \
      --generic-anchor-fields seizure_start_s \
      --start-row "$CHBMIT_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --max-review-rounds 2 \
      --local-validation-retries "$LOCAL_VALIDATION_RETRIES" \
      --local-narrative-max-tokens 2400 \
      --output-dir "$RUN_ROOT/chbmit" \
      --resume \
      --no-fail-fast
  refresh_index
fi

run_unanchored_dataset() {
  local dataset="$1"
  local eeg_root="$2"
  local scan_dir="$RUN_ROOT/${dataset}_scan"
  local queue_manifest="$scan_dir/qwen_review_queue.csv"

  run_logged "${dataset}_scan" \
    rtk "$PYTHON" -u "$SCANNER" \
      --dataset-name "$dataset" \
      --eeg-root "$eeg_root" \
      --output-dir "$scan_dir" \
      --max-files "$UNANCHORED_MAX_FILES" \
      --max-candidates-per-file 3 \
      --resume \
      --no-fail-fast

  if [[ -f "$scan_dir/full_record_candidates.csv" ]]; then
    run_logged "${dataset}_select" \
      rtk "$PYTHON" "$SELECTOR" \
        --input-manifest "$scan_dir/full_record_candidates.csv" \
        --output-manifest "$queue_manifest" \
        --max-per-file "$UNANCHORED_MAX_PER_FILE" \
        --hard-max-per-file 3
  else
    PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
  fi

  if [[ -f "$queue_manifest" ]]; then
    run_logged "$dataset" \
      rtk "$PYTHON" -u "$ANNOTATOR" \
        --dataset generic \
        --dataset-name "$dataset" \
        --manifest "$queue_manifest" \
        --eeg-root "$eeg_root" \
        --generic-anchor-fields coarse_candidate_s \
        --max-events "$MAX_EVENTS" \
        --max-review-rounds 2 \
        --local-validation-retries "$LOCAL_VALIDATION_RETRIES" \
        --local-narrative-max-tokens 2400 \
        --output-dir "$RUN_ROOT/$dataset" \
        --resume \
        --no-fail-fast
    refresh_index
  else
    PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
  fi
}

if [[ "$RUN_TUEV" == "1" ]]; then
  run_unanchored_dataset tuev "$TUEV_ROOT"
fi

if [[ "$RUN_TUEP" == "1" ]]; then
  run_unanchored_dataset tuep "$TUEP_ROOT"
fi

refresh_index

if [[ "$AUTO_RETRY" == "1" && -s "$RUN_ROOT/indexes/failures.jsonl" ]]; then
  run_logged retry \
    rtk "$PYTHON" -u "$RETRY" \
      --failures "$RUN_ROOT/indexes/failures.jsonl" \
      --output-root "$RUN_ROOT/retries"
fi

if (( PIPELINE_FAILURES != 0 )); then
  exit 1
fi
