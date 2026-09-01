#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTROLLED_WORKSPACE="${EVISOZ_CONTROLLED_WORKSPACE:-/mnt/hd1/dyf/workspace/laptop/EEG_Seizure}"
LEGACY_VIEWER_ROOT="${EVISOZ_LEGACY_VIEWER_ROOT:-$CONTROLLED_WORKSPACE}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/deepsoz_607_llm_qwen36_full_v3_20260801}"
HOST="${VIEWER_HOST:-0.0.0.0}"
PORT="${VIEWER_PORT:-8767}"
PYTHON_BIN="${VIEWER_PYTHON:-$PROJECT_ROOT/.venv-qwen35/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

rtk env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
  "$LEGACY_VIEWER_ROOT/code/data_preprocess/llm_soz_result_viewer.py" \
  --run-root "$RUN_ROOT" \
  --host "$HOST" \
  --port "$PORT"
