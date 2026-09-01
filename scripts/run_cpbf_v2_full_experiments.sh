#!/usr/bin/env bash
set -euo pipefail

VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-12}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_v2_full}"
BASELINE_PREFIX="${BASELINE_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval}"

common_env() {
  local seed="$1"
  local baseline_root="${BASELINE_PREFIX}_seed${seed}"
  printf '%s\n' \
    "TOKENIZER_EPOCHS=0" \
    "CLASSIFIER_EPOCHS=${CLASSIFIER_EPOCHS}" \
    "BATCH_SIZE=8" \
    "VAL_PATIENTS=${VAL_PATIENTS}" \
    "SEED=${seed}" \
    "SPLIT_SEED=${SPLIT_SEED}" \
    "INIT_LOPO_ROOT=${baseline_root}" \
    "USE_REGION_ATTENTION_POOLING=1" \
    "USE_REGION_EMBEDDING_HEAD=1" \
    "SELECTION_PRIMARY_SOURCE=channel_onset" \
    "SKIP_EXISTING=1" \
    "QUIET_SUMMARY=1"
}

summarize_root() {
  local root="$1"
  local seed="$2"
  python3 code/tfm_soz/summarize_lopo.py \
    --root "$root" \
    --split test \
    --bootstrap-iters 0 \
    --bootstrap-seed "$seed" \
    --output "$root/lopo_test_summary.json" \
    --quiet
  python3 code/tfm_soz/summarize_lopo.py \
    --root "$root" \
    --split val \
    --bootstrap-iters 0 \
    --bootstrap-seed "$seed" \
    --output "$root/lopo_val_summary.json" \
    --quiet
}

run_head_only() {
  local seed="$1"
  local root="${OUTPUT_PREFIX}_head_only_rank_seed${seed}"
  local -a env_args=()
  mapfile -t env_args < <(common_env "$seed")
  env "${env_args[@]}" \
    OUTPUT_ROOT="$root" \
    FREEZE_BASE_CHANNEL_HEAD_ONLY=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh
  summarize_root "$root" "$seed"
}

run_graph() {
  local name="$1"
  local seed="$2"
  local input_stage="$3"
  local disable_region_bias="$4"
  local root="${OUTPUT_PREFIX}_${name}_seed${seed}"
  local -a env_args=()
  mapfile -t env_args < <(common_env "$seed")
  env "${env_args[@]}" \
    OUTPUT_ROOT="$root" \
    USE_CPBF_GRAPH=1 \
    CPBF_MODE=context_graph \
    CPBF_TOPK=6 \
    CPBF_RESIDUAL_INIT=0.0 \
    CPBF_CANDIDATE_POLICY=compact \
    CPBF_INPUT_STAGE="$input_stage" \
    DISABLE_CPBF_REGION_BIAS="$disable_region_bias" \
    FREEZE_BASE_FOR_CPBF=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh
  summarize_root "$root" "$seed"
}

run_head_only 2028 & pid_head2028=$!
run_head_only 2029 & pid_head2029=$!
run_head_only 2030 & pid_head2030=$!
run_graph compact_post 2028 post_global 0 & pid_post=$!
run_graph compact_pre_noregion 2028 pre_global_temporal 1 & pid_noregion=$!
run_graph compact_pre 2028 pre_global_temporal 0 & pid_pre2028=$!
run_graph compact_pre 2029 pre_global_temporal 0 & pid_pre2029=$!
run_graph compact_pre 2030 pre_global_temporal 0 & pid_pre2030=$!

status=0
for pid in "$pid_head2028" "$pid_head2029" "$pid_head2030" "$pid_post" "$pid_noregion" "$pid_pre2028" "$pid_pre2029" "$pid_pre2030"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
