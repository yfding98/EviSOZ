#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${1:-${REPO_ROOT}/outputs/deepsoz_607_llm_qwen36_full_v3_20260801}"
PYTHON_BIN="${REPO_ROOT}/.venv-qwen35/bin/python"
MANIFEST="${REPO_ROOT}/outputs/deepsoz_607_llm_batch_20260801/tusz_deepsoz_607_events_manifest.csv"
EEG_ROOT="/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf"

# 1566 events split into eight non-overlapping, exhaustive ranges.
START_ROWS=(0 196 392 588 784 980 1176 1371)
EVENT_COUNTS=(196 196 196 196 196 196 195 195)

rtk mkdir -p "${RUN_ROOT}"

PIDS=()
for SHARD_INDEX in "${!START_ROWS[@]}"; do
    SHARD_NAME="$(printf 'shard_%02d' "${SHARD_INDEX}")"
    SHARD_DIR="${RUN_ROOT}/${SHARD_NAME}"
    rtk mkdir -p "${SHARD_DIR}"
    rtk env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u \
        -m code.auto_annotate.llm_soz_annotator \
        --dataset tusz \
        --manifest "${MANIFEST}" \
        --eeg-root "${EEG_ROOT}" \
        --all-tusz-events \
        --output-dir "${SHARD_DIR}" \
        --inference-backend vllm \
        --api-base-url http://127.0.0.1:8000/v1 \
        --model qwen36-soz \
        --require-local-model-release qwen3.6 \
        --analysis-sfreq 128 \
        --context-pre-s 20 \
        --context-post-s 45 \
        --api-image-max-edge 1280 \
        --api-image-jpeg-quality 75 \
        --max-review-rounds 2 \
        --local-validation-retries 2 \
        --local-core-max-tokens 1300 \
        --start-row "${START_ROWS[SHARD_INDEX]}" \
        --max-events "${EVENT_COUNTS[SHARD_INDEX]}" \
        --resume \
        --no-fail-fast \
        >"${SHARD_DIR}/run.log" 2>&1 &
    PIDS+=("$!")
done

EXIT_STATUS=0
for PROCESS_ID in "${PIDS[@]}"; do
    if ! wait "${PROCESS_ID}"; then
        EXIT_STATUS=1
    fi
done
exit "${EXIT_STATUS}"
