#!/usr/bin/env bash
set -euo pipefail

# One-command launcher: start the local Qwen3.6 vLLM server, wait until its
# OpenAI endpoint is ready, resume all datasets, then stop only the server
# process created by this script.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_TAG="${RUN_TAG:-full_20260723}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/qwen36_vllm_all_data_$RUN_TAG}"
BASE_URL="${QWEN36_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
READY_URL="${BASE_URL%/}/models"
WAIT_SECONDS="${VLLM_STARTUP_TIMEOUT_S:-1800}"
SERVER_LOG="$RUN_ROOT/logs/vllm_server.log"
SERVER_PID=""
OWN_SERVER=0

cd "$PROJECT_ROOT"
rtk mkdir -p "$RUN_ROOT/logs"

cleanup() {
  if [[ "$OWN_SERVER" == "1" && -n "$SERVER_PID" ]]; then
    rtk kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! rtk curl -fsS "$READY_URL" >/dev/null 2>&1; then
  rtk bash scripts/serve_qwen36_vllm.sh >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  OWN_SERVER=1
  deadline=$((SECONDS + WAIT_SECONDS))
  until rtk curl -fsS "$READY_URL" >/dev/null 2>&1; do
    if ! rtk kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited during startup; inspect $SERVER_LOG" >&2
      exit 2
    fi
    if (( SECONDS >= deadline )); then
      echo "vLLM startup timed out after ${WAIT_SECONDS}s" >&2
      exit 2
    fi
    rtk sleep 5
  done
fi

rtk env \
  RUN_TAG="$RUN_TAG" \
  RUN_ROOT="$RUN_ROOT" \
  QWEN36_VLLM_BASE_URL="$BASE_URL" \
  bash scripts/run_qwen36_vllm_all_data.sh
