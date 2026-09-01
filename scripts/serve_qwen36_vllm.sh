#!/usr/bin/env bash
set -euo pipefail

# Continuous-batching Qwen3.6 VLM server for the SOZ annotation workers.
# The model card warns not to force --quantization; vLLM must auto-detect GPTQ.

PROJECT_ROOT="${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-$PROJECT_ROOT/.venv-vllm-qwen36/bin/python}"
MODEL_PATH="${QWEN36_MODEL_PATH:-$PROJECT_ROOT/models/Qwen3.6-35B-A3B-GPTQ-Int4}"
SERVED_MODEL_NAME="${QWEN36_VLLM_MODEL_NAME:-qwen36-soz}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
# vLLM 0.19.1 temporarily allocates `max_cudagraph_capture_size` cache blocks
# while profiling hybrid attention/Mamba models.  With max-num-seqs=1 the
# inferred value is exactly 2, which makes an attention cache shaped
# (2, 2, ...) layout-ambiguous and aborts Qwen3.6 startup.  One capture block
# avoids that upstream edge case without changing generated text or KV-cache
# capacity used after profiling.
MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-1}"
# RTX A6000 (SM86/Ampere) cannot compile the fp8e4nv KV-cache kernels used by
# vLLM.  `auto` selects the model dtype and works across Ampere and newer GPUs.
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-4}"
ENABLE_MTP="${VLLM_ENABLE_MTP:-0}"
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

# These are launcher-facing compatibility variables that have already been
# translated into explicit CLI arguments.  Do not leak them into vLLM's own
# VLLM_* environment validator (and do not leak the batch worker setting from
# the parent stack into the server process).
unset \
  VLLM_HOST \
  VLLM_PORT \
  VLLM_TENSOR_PARALLEL_SIZE \
  VLLM_MAX_MODEL_LEN \
  VLLM_MAX_NUM_SEQS \
  VLLM_MAX_NUM_BATCHED_TOKENS \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE \
  VLLM_KV_CACHE_DTYPE \
  VLLM_MM_PROCESSOR_CACHE_GB \
  VLLM_ENABLE_MTP \
  VLLM_ATTENTION_BACKEND \
  VLLM_NUM_GPU_BLOCKS_OVERRIDE \
  VLLM_EVENT_WORKERS

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "vLLM environment not found: $VLLM_PYTHON" >&2
  echo "Run: rtk bash scripts/install_qwen36_vllm.sh" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Qwen3.6 model is incomplete: $MODEL_PATH" >&2
  exit 2
fi

# FlashInfer JIT compilation launches the `ninja` executable as a subprocess.
# Calling the venv's Python directly does not activate the venv, so expose all
# companion executables explicitly.
VLLM_BIN_DIR="${VLLM_PYTHON%/python}"
export PATH="$VLLM_BIN_DIR:$PATH"

# The host login environment may contain a path-list style CUDA_HOME such as
# `:/usr/local/cuda-12.1`.  CUDA_HOME is a single directory, and passing that
# malformed value to FlashInfer produces an invalid `:/.../bin/nvcc` command.
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
while [[ "$CUDA_ROOT" == :* ]]; do
  CUDA_ROOT="${CUDA_ROOT#:}"
done
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  CUDA_ROOT="/usr/local/cuda"
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "CUDA compiler not found under CUDA_HOME=$CUDA_ROOT" >&2
  exit 2
fi
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export PATH="$CUDA_ROOT/bin:$PATH"

ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --dtype bfloat16
  --attention-backend "$ATTENTION_BACKEND"
  --reasoning-parser qwen3
  --enable-prefix-caching
  --enable-chunked-prefill
  --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB"
  --mm-processor-cache-type shm
  --limit-mm-per-prompt '{"image":24}'
  --trust-remote-code
  --no-enable-log-requests
)

if [[ "$ENABLE_MTP" == "1" ]]; then
  ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
fi

cd "$PROJECT_ROOT"
exec rtk "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
