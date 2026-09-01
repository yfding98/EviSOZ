#!/usr/bin/env bash
set -euo pipefail

SCREEN_PATIENTS="${SCREEN_PATIENTS:-刘娟,庄芷端,朱涵栖,杨朵,江仁坤,确干,赖冬微,陈妙玲,黄建和,龙娇}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SEED="${SEED:-2028}"
EPOCHS="${CLASSIFIER_EPOCHS:-12}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_v2_screen}"

common_env=(
  TOKENIZER_EPOCHS=0 CLASSIFIER_EPOCHS="$EPOCHS" BATCH_SIZE=8
  PATIENTS="$SCREEN_PATIENTS" VAL_PATIENTS="$VAL_PATIENTS"
  SEED="$SEED" SPLIT_SEED=2026 INIT_LOPO_ROOT="$BASELINE_ROOT"
  USE_REGION_ATTENTION_POOLING=1 USE_REGION_EMBEDDING_HEAD=1
  SELECTION_PRIMARY_SOURCE=channel_onset SKIP_EXISTING=1 QUIET_SUMMARY=1
)

run_head_only() {
  env "${common_env[@]}" \
    OUTPUT_ROOT="${OUTPUT_PREFIX}_head_only_seed${SEED}" \
    FREEZE_BASE_CHANNEL_HEAD_ONLY=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh
}

run_graph() {
  local name="$1"
  local input_stage="$2"
  env "${common_env[@]}" \
    OUTPUT_ROOT="${OUTPUT_PREFIX}_${name}_seed${SEED}" \
    USE_CPBF_GRAPH=1 CPBF_MODE=context_graph CPBF_TOPK=6 \
    CPBF_RESIDUAL_INIT=0.0 CPBF_CANDIDATE_POLICY=compact \
    CPBF_INPUT_STAGE="$input_stage" FREEZE_BASE_FOR_CPBF=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh
}

run_head_only & job_head=$!
run_graph compact_post post_global & job_post=$!
run_graph compact_pre pre_global_temporal & job_pre=$!
wait "$job_head"
wait "$job_post"
wait "$job_pre"

for name in head_only compact_post compact_pre; do
  root="${OUTPUT_PREFIX}_${name}_seed${SEED}"
  python3 code/tfm_soz/summarize_lopo.py \
    --root "$root" --split val --bootstrap-iters 0 \
    --output "$root/lopo_val_screen_summary.json" --quiet
done
