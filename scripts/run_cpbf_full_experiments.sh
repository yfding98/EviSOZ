#!/usr/bin/env bash
set -euo pipefail

SELECTED_MODE="${SELECTED_MODE:-router}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-30}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_full}"
BASELINE_PREFIX="${BASELINE_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval}"

run_one() {
  local mode="$1"
  local seed="$2"
  local baseline_root="${BASELINE_PREFIX}_seed${seed}"
  local output_root="${OUTPUT_PREFIX}_${mode}_seed${seed}"

  env \
    TOKENIZER_EPOCHS=0 \
    CLASSIFIER_EPOCHS="$CLASSIFIER_EPOCHS" \
    BATCH_SIZE=8 \
    VAL_PATIENTS="$VAL_PATIENTS" \
    SEED="$seed" \
    SPLIT_SEED="$SPLIT_SEED" \
    INIT_LOPO_ROOT="$baseline_root" \
    OUTPUT_ROOT="$output_root" \
    USE_REGION_ATTENTION_POOLING=1 \
    USE_REGION_EMBEDDING_HEAD=1 \
    USE_CPBF_GRAPH=1 \
    CPBF_MODE="$mode" \
    CPBF_TOPK=6 \
    CPBF_RESIDUAL_INIT=0.0 \
    FREEZE_BASE_FOR_CPBF=1 \
    SKIP_EXISTING=1 \
    QUIET_SUMMARY=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh

  python3 code/tfm_soz/summarize_lopo.py \
    --root "$output_root" \
    --split test \
    --bootstrap-iters 5000 \
    --bootstrap-seed "$seed" \
    --output "${output_root}/lopo_test_summary.json" \
    --quiet
  python3 code/tfm_soz/summarize_lopo.py \
    --root "$output_root" \
    --split val \
    --bootstrap-iters 5000 \
    --bootstrap-seed "$seed" \
    --output "${output_root}/lopo_val_summary.json" \
    --quiet
}

run_head_only() {
  local seed="2028"
  local baseline_root="${BASELINE_PREFIX}_seed${seed}"
  local output_root="${OUTPUT_PREFIX}_head_only_seed${seed}"

  env \
    TOKENIZER_EPOCHS=0 \
    CLASSIFIER_EPOCHS="$CLASSIFIER_EPOCHS" \
    BATCH_SIZE=8 \
    VAL_PATIENTS="$VAL_PATIENTS" \
    SEED="$seed" \
    SPLIT_SEED="$SPLIT_SEED" \
    INIT_LOPO_ROOT="$baseline_root" \
    OUTPUT_ROOT="$output_root" \
    USE_REGION_ATTENTION_POOLING=1 \
    USE_REGION_EMBEDDING_HEAD=1 \
    FREEZE_BASE_CHANNEL_HEAD_ONLY=1 \
    SKIP_EXISTING=1 \
    QUIET_SUMMARY=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh

  python3 code/tfm_soz/summarize_lopo.py \
    --root "$output_root" \
    --split test \
    --bootstrap-iters 5000 \
    --bootstrap-seed "$seed" \
    --output "${output_root}/lopo_test_summary.json" \
    --quiet
  python3 code/tfm_soz/summarize_lopo.py \
    --root "$output_root" \
    --split val \
    --bootstrap-iters 5000 \
    --bootstrap-seed "$seed" \
    --output "${output_root}/lopo_val_summary.json" \
    --quiet
}

if [[ "$RUN_ABLATIONS" == "1" && "$RUN_PARALLEL" == "1" ]]; then
  run_head_only & job_head=$!
  run_one temporal 2028 & job_temporal=$!
  run_one learned_graph 2028 & job_learned=$!
  wait "$job_head"
  wait "$job_temporal"
  wait "$job_learned"

  run_one context_graph 2028 & job_context=$!
  run_one router 2028 & job_router=$!
  run_one "$SELECTED_MODE" 2029 & job_seed2029=$!
  wait "$job_context"
  wait "$job_router"
  wait "$job_seed2029"

  run_one "$SELECTED_MODE" 2030
  exit 0
elif [[ "$RUN_ABLATIONS" == "1" ]]; then
  run_head_only
  for mode in temporal learned_graph context_graph router; do
    run_one "$mode" 2028
  done
else
  run_one "$SELECTED_MODE" 2028
fi

run_one "$SELECTED_MODE" 2029
run_one "$SELECTED_MODE" 2030
