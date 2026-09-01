#!/usr/bin/env bash
set -euo pipefail

# Predeclared deterministic folds. Candidate selection must use only the
# validation summaries produced below; test metrics remain sealed until the
# CPBF mode is fixed.
SCREEN_PATIENTS="${SCREEN_PATIENTS:-刘娟,庄芷端,朱涵栖,杨朵,江仁坤,确干,赖冬微,陈妙玲,黄建和,龙娇}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SEED="${SEED:-2028}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-12}"
MODES="${MODES:-temporal learned_graph context_graph router}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/tfm_soz/private_0622_fix_rows119_cpbf_screen}"

for mode in $MODES; do
  output_root="${OUTPUT_PREFIX}_${mode}_seed${SEED}"
  env \
    TOKENIZER_EPOCHS=0 \
    CLASSIFIER_EPOCHS="$CLASSIFIER_EPOCHS" \
    BATCH_SIZE=8 \
    PATIENTS="$SCREEN_PATIENTS" \
    VAL_PATIENTS="$VAL_PATIENTS" \
    SEED="$SEED" \
    SPLIT_SEED="$SPLIT_SEED" \
    INIT_LOPO_ROOT="$BASELINE_ROOT" \
    OUTPUT_ROOT="$output_root" \
    USE_REGION_ATTENTION_POOLING=1 \
    USE_REGION_EMBEDDING_HEAD=1 \
    USE_CPBF_GRAPH=1 \
    CPBF_MODE="$mode" \
    CPBF_TOPK=6 \
    CPBF_RESIDUAL_INIT=0.0 \
    FREEZE_BASE_FOR_CPBF=1 \
    SKIP_EXISTING=1 \
    bash scripts/run_tfm_soz_private_segments_lopo.sh

  python3 code/tfm_soz/summarize_lopo.py \
    --root "$output_root" \
    --split val \
    --folds "$SCREEN_PATIENTS" \
    --bootstrap-iters 2000 \
    --bootstrap-seed 2028 \
    --output "${output_root}/lopo_val_screen_summary.json" \
    --quiet
done
