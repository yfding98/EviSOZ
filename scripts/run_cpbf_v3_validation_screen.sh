#!/usr/bin/env bash
set -euo pipefail

# Architecture and hyperparameter selection is restricted to these predeclared
# validation folds. The trainer must not construct the test dataset or write
# test predictions; stale non-blind output roots are rejected by the runner.
SCREEN_PATIENTS="${SCREEN_PATIENTS:-刘娟,庄芷端,朱涵栖,杨朵,江仁坤,确干,赖冬微,陈妙玲,黄建和,龙娇}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SEED="${SEED:-2028}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
EPOCHS="${CLASSIFIER_EPOCHS:-12}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028}"
HEAD_ONLY_ROOT="${HEAD_ONLY_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_v2_screen_head_only_seed2028}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_v3_screen}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-outputs/tfm_soz/private_0622_fix_rows119_teacher_logits_tfm_2028_2029_2030_cpbf_v3_screen10}"
DISTILL_CHANNEL_WEIGHT="${DISTILL_CHANNEL_WEIGHT:-0.2}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"

common_env=(
  TOKENIZER_EPOCHS=0 CLASSIFIER_EPOCHS="$EPOCHS" BATCH_SIZE=8
  PATIENTS="$SCREEN_PATIENTS" VAL_PATIENTS="$VAL_PATIENTS"
  SEED="$SEED" SPLIT_SEED="$SPLIT_SEED" INIT_LOPO_ROOT="$BASELINE_ROOT"
  SKIP_TEST_EVAL=1
  USE_REGION_ATTENTION_POOLING=1 USE_REGION_EMBEDDING_HEAD=1
  SELECTION_PRIMARY_SOURCE=channel_onset SKIP_EXISTING=1 QUIET_SUMMARY=1
  USE_CPBF_GRAPH=1 CPBF_MODE=context_graph CPBF_TOPK=6
  CPBF_RESIDUAL_INIT=0.0 CPBF_CANDIDATE_POLICY=compact
  CPBF_INPUT_STAGE=pre_global_temporal DISABLE_CPBF_REGION_BIAS=1
  FREEZE_BASE_FOR_CPBF=1
)

run_v2_no_region() {
  rtk env "${common_env[@]}" \
    OUTPUT_ROOT="${OUTPUT_PREFIX}_v2_no_region_seed${SEED}" \
    CPBF_RESIDUAL_POLICY=signed_scalar USE_CPBF_TEMPORAL_ADAPTER=0 \
    rtk bash scripts/run_tfm_soz_private_segments_lopo.sh
}

run_v3() {
  local name="$1"
  local distill="$2"
  local -a distill_env=()
  if [[ "$distill" == "1" ]]; then
    distill_env=(
      TEACHER_CACHE_DIR="$TEACHER_CACHE_DIR"
      TEACHER_DISTILL_CHANNEL_WEIGHT="$DISTILL_CHANNEL_WEIGHT"
      TEACHER_DISTILL_REGION_WEIGHT=0
      TEACHER_DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE"
    )
  fi
  rtk env "${common_env[@]}" "${distill_env[@]}" \
    OUTPUT_ROOT="${OUTPUT_PREFIX}_${name}_seed${SEED}" \
    CPBF_RESIDUAL_POLICY=confidence_gate CPBF_CONFIDENCE_MAX_SCALE=0.5 \
    USE_CPBF_TEMPORAL_ADAPTER=1 CPBF_ADAPTER_BOTTLENECK=16 \
    rtk bash scripts/run_tfm_soz_private_segments_lopo.sh
}

run_v2_no_region & job_v2=$!
run_v3 gate_adapter 0 & job_v3=$!
run_v3 gate_adapter_distill 1 & job_distill=$!

status=0
for job in "$job_v2" "$job_v3" "$job_distill"; do
  if ! wait "$job"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  exit "$status"
fi

for root in \
  "$HEAD_ONLY_ROOT" \
  "${OUTPUT_PREFIX}_v2_no_region_seed${SEED}" \
  "${OUTPUT_PREFIX}_gate_adapter_seed${SEED}" \
  "${OUTPUT_PREFIX}_gate_adapter_distill_seed${SEED}"; do
  rtk python3 code/tfm_soz/summarize_lopo.py \
    --root "$root" --split val --bootstrap-iters 0 \
    --output "$root/lopo_val_screen_summary.json" --quiet
done
