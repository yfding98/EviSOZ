#!/usr/bin/env bash
set -u

# Full SOZ pipeline backed by one already-running Qwen3.6 vLLM server.
# Event workers are isolated into shards, so concurrent writes cannot collide.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_TAG="${RUN_TAG:-full_20260723}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/qwen36_vllm_all_data_$RUN_TAG}"
PRIOR_RUN_ROOT="${PRIOR_RUN_ROOT:-$PROJECT_ROOT/outputs/qwen35_all_data_full_20260723}"
PRIOR_SUCCESS_INDEX="${PRIOR_SUCCESS_INDEX:-$PRIOR_RUN_ROOT/indexes/successes.jsonl}"

WORKERS="${VLLM_EVENT_WORKERS:-4}"
MAX_EVENTS="${MAX_EVENTS:--1}"
RUN_PRIVATE="${RUN_PRIVATE:-1}"
RUN_TUSZ="${RUN_TUSZ:-1}"
RUN_CHBMIT="${RUN_CHBMIT:-1}"
RUN_TUEV="${RUN_TUEV:-1}"
RUN_TUEP="${RUN_TUEP:-1}"
PRIVATE_START_ROW="${PRIVATE_START_ROW:-0}"
TUSZ_START_ROW="${TUSZ_START_ROW:-0}"
CHBMIT_START_ROW="${CHBMIT_START_ROW:-0}"
UNANCHORED_MAX_FILES="${UNANCHORED_MAX_FILES:--1}"
UNANCHORED_MAX_PER_FILE="${UNANCHORED_MAX_PER_FILE:-2}"

PYTHON="$PROJECT_ROOT/.venv-qwen35/bin/python"
QUEUE="$PROJECT_ROOT/scripts/run_qwen36_vllm_queue.py"
SCANNER="$PROJECT_ROOT/scripts/scan_public_soz_candidates.py"
SELECTOR="$PROJECT_ROOT/scripts/select_full_record_soz_review_queue.py"
INDEXER="$PROJECT_ROOT/scripts/index_qwen35_results.py"
CHBMIT_ROOT="${CHBMIT_ROOT:-/mnt/hd1/dyf/dataset/CHB-MIT}"
TUEV_ROOT="${TUEV_ROOT:-/mnt/hd1/dyf/dataset/tuh_eeg_events}"
TUEP_ROOT="${TUEP_ROOT:-/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy}"
CHBMIT_MANIFEST="${CHBMIT_MANIFEST:-$PROJECT_ROOT/outputs/llm_soz_public_manifests/chbmit_seizure_anchors.csv}"
PIPELINE_FAILURES=0

cd "$PROJECT_ROOT" || exit 2
rtk mkdir -p "$RUN_ROOT/logs"

PRIOR_ARGS=()
if [[ -s "$PRIOR_SUCCESS_INDEX" ]]; then
  PRIOR_ARGS=(--prior-success-index "$PRIOR_SUCCESS_INDEX")
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
  local roots=(--run-root "$RUN_ROOT")
  if [[ -d "$PRIOR_RUN_ROOT" ]]; then
    roots+=(--run-root "$PRIOR_RUN_ROOT")
  fi
  rtk "$PYTHON" "$INDEXER" \
    "${roots[@]}" \
    --output-dir "$RUN_ROOT/indexes" \
    >> "$RUN_ROOT/logs/index.log" 2>&1
}

trap 'refresh_index' EXIT

if [[ "$RUN_PRIVATE" == "1" ]]; then
  run_logged private \
    rtk "$PYTHON" -u "$QUEUE" \
      --dataset private \
      --start-row "$PRIVATE_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --workers "$WORKERS" \
      --output-dir "$RUN_ROOT/private" \
      "${PRIOR_ARGS[@]}"
  refresh_index
fi

if [[ "$RUN_TUSZ" == "1" ]]; then
  run_logged tusz \
    rtk "$PYTHON" -u "$QUEUE" \
      --dataset tusz \
      --start-row "$TUSZ_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --workers "$WORKERS" \
      --output-dir "$RUN_ROOT/tusz" \
      "${PRIOR_ARGS[@]}"
  refresh_index
fi

if [[ "$RUN_CHBMIT" == "1" ]]; then
  run_logged chbmit \
    rtk "$PYTHON" -u "$QUEUE" \
      --dataset generic \
      --dataset-name chbmit \
      --manifest "$CHBMIT_MANIFEST" \
      --eeg-root "$CHBMIT_ROOT" \
      --generic-anchor-fields seizure_start_s \
      --start-row "$CHBMIT_START_ROW" \
      --max-events "$MAX_EVENTS" \
      --workers "$WORKERS" \
      --output-dir "$RUN_ROOT/chbmit" \
      "${PRIOR_ARGS[@]}"
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
      rtk "$PYTHON" -u "$QUEUE" \
        --dataset generic \
        --dataset-name "$dataset" \
        --manifest "$queue_manifest" \
        --eeg-root "$eeg_root" \
        --generic-anchor-fields coarse_candidate_s \
        --max-events "$MAX_EVENTS" \
        --workers "$WORKERS" \
        --output-dir "$RUN_ROOT/$dataset" \
        "${PRIOR_ARGS[@]}"
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
if (( PIPELINE_FAILURES != 0 )); then
  exit 1
fi
