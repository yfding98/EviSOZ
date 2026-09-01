#!/usr/bin/env bash
set -euo pipefail

# vLLM 0.19.1 is the version explicitly verified by the local Qwen3.6 GPTQ
# model card.  Keep it outside the nearly-full workspace filesystem and do not
# modify the working Transformers environment.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_ENV="${VLLM_ENV:-/tmp/eeg_qwen36_vllm_0_19_1}"
VLLM_LINK="$PROJECT_ROOT/.venv-vllm-qwen36"
PIP_INDEX_URL="${VLLM_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TIMEOUT="${VLLM_PIP_TIMEOUT:-180}"
PIP_RETRIES="${VLLM_PIP_RETRIES:-8}"
PIP_CACHE_DIR="${VLLM_PIP_CACHE_DIR:-/tmp/eeg_qwen36_vllm_pip_cache}"

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  rtk python3 -m venv "$VLLM_ENV"
fi

rtk "$VLLM_ENV/bin/python" -m pip install \
  --index-url "$PIP_INDEX_URL" \
  --timeout "$PIP_TIMEOUT" \
  --retries "$PIP_RETRIES" \
  --cache-dir "$PIP_CACHE_DIR" \
  --upgrade pip
rtk "$VLLM_ENV/bin/python" -m pip install \
  --index-url "$PIP_INDEX_URL" \
  --timeout "$PIP_TIMEOUT" \
  --retries "$PIP_RETRIES" \
  --cache-dir "$PIP_CACHE_DIR" \
  "vllm==0.19.1"
rtk "$VLLM_ENV/bin/python" -c \
  "import vllm; print('vLLM', vllm.__version__)"

if [[ -e "$VLLM_LINK" || -L "$VLLM_LINK" ]]; then
  if [[ "$(rtk readlink -f "$VLLM_LINK")" != "$(rtk readlink -f "$VLLM_ENV")" ]]; then
    echo "Refusing to replace existing $VLLM_LINK" >&2
    exit 2
  fi
else
  rtk ln -s "$VLLM_ENV" "$VLLM_LINK"
fi
