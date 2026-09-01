#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-2042}"
VARIANT="${VARIANT:-graph}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/tfm_soz/private_0622_fix_regiongt_rows119_segments_15s}"
MANIFEST="${MANIFEST:-private_sz_union_relabel_manifest_0622_fix_region_annotation.csv}"
INIT_LOPO_ROOT="${INIT_LOPO_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
SPLIT_SEED="${SPLIT_SEED:-2028}"

case "$VARIANT" in
  graph)
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tfm_soz/private_0622_fix_regiongt_cpbf_lineageclean_channel_graph_top1joint_fixedval_seed${SEED}}"
    variant_env=(
      USE_CPBF_GRAPH=1
      CPBF_MODE=context_graph
      CPBF_TOPK=6
      CPBF_CANDIDATE_POLICY=compact
      CPBF_INPUT_STAGE=pre_global_temporal
      DISABLE_CPBF_REGION_BIAS=1
      FREEZE_BASE_FOR_CPBF=1
    )
    ;;
  head)
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tfm_soz/private_0622_fix_regiongt_cpbf_lineageclean_head_only_top1joint_fixedval_seed${SEED}}"
    variant_env=(FREEZE_BASE_CHANNEL_HEAD_ONLY=1)
    ;;
  *)
    echo "VARIANT must be graph or head, got: $VARIANT" >&2
    exit 2
    ;;
esac

env \
  PREPROCESSED_DIR="$PREPROCESSED_DIR" \
  MANIFEST="$MANIFEST" \
  REGION_LABEL_SOURCE=soz_region \
  TOKENIZER_EPOCHS=0 \
  CLASSIFIER_EPOCHS=12 \
  BATCH_SIZE=8 \
  SEED="$SEED" \
  SPLIT_SEED="$SPLIT_SEED" \
  VAL_PATIENTS="$VAL_PATIENTS" \
  INIT_LOPO_ROOT="$INIT_LOPO_ROOT" \
  USE_REGION_ATTENTION_POOLING=1 \
  USE_REGION_EMBEDDING_HEAD=1 \
  TRAINING_OBJECTIVE=top1_joint \
  CHANNEL_TOP1_MARGIN_WEIGHT=0.25 \
  TRAIN_REGION_PATH_WITH_FROZEN_BASE=1 \
  SKIP_EXISTING=1 \
  QUIET_SUMMARY=1 \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  "${variant_env[@]}" \
  bash scripts/run_tfm_soz_private_segments_lopo.sh
