#!/usr/bin/env bash
set -euo pipefail

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/tfm_soz/private_0622_fix_segments_15s}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tfm_soz/private_0622_fix_segment_hardneg_run}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-16}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-2026}"
SPLIT_SEED="${SPLIT_SEED:-$SEED}"
VAL_PATIENTS="${VAL_PATIENTS:-}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
HARD_NEGATIVE_FRACTION="${HARD_NEGATIVE_FRACTION:-0.25}"
CHANNEL_HARD_NEGATIVE_WEIGHT="${CHANNEL_HARD_NEGATIVE_WEIGHT:-0.05}"
REGION_HARD_NEGATIVE_WEIGHT="${REGION_HARD_NEGATIVE_WEIGHT:-0.25}"
CHANNEL_TOP1_MARGIN_WEIGHT="${CHANNEL_TOP1_MARGIN_WEIGHT:-0.0}"
REGION_TOP1_MARGIN_WEIGHT="${REGION_TOP1_MARGIN_WEIGHT:-0.0}"
TOP1_MARGIN="${TOP1_MARGIN:-0.2}"
CHANNEL_POSITIVE_SET_MARGIN_WEIGHT="${CHANNEL_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
REGION_POSITIVE_SET_MARGIN_WEIGHT="${REGION_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
POSITIVE_SET_MARGIN="${POSITIVE_SET_MARGIN:-0.2}"
AUGMENT_SEGMENT_RECONSTRUCT_PROB="${AUGMENT_SEGMENT_RECONSTRUCT_PROB:-0.0}"
AUGMENT_SEGMENT_RECONSTRUCT_PIECES="${AUGMENT_SEGMENT_RECONSTRUCT_PIECES:-4}"
AUGMENT_NEGATIVE_CHANNEL_DROP_PROB="${AUGMENT_NEGATIVE_CHANNEL_DROP_PROB:-0.0}"
AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION="${AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION:-0.25}"
AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB="${AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB:-0.0}"
AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC="${AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC:-0.5}"
USE_CONTEXT_DELTA_HEADS="${USE_CONTEXT_DELTA_HEADS:-0}"
USE_REGION_ATTENTION_POOLING="${USE_REGION_ATTENTION_POOLING:-0}"
USE_REGION_EMBEDDING_HEAD="${USE_REGION_EMBEDDING_HEAD:-0}"
USE_CBRAMOD_CRISS_CROSS="${USE_CBRAMOD_CRISS_CROSS:-0}"
USE_ADAPTIVE_POSITION_ENCODING="${USE_ADAPTIVE_POSITION_ENCODING:-0}"
DISABLE_ADAPTIVE_POSITION_ENCODING="${DISABLE_ADAPTIVE_POSITION_ENCODING:-0}"
ADAPTIVE_POSITION_SPATIAL_KERNEL="${ADAPTIVE_POSITION_SPATIAL_KERNEL:-19}"
ADAPTIVE_POSITION_TEMPORAL_KERNEL="${ADAPTIVE_POSITION_TEMPORAL_KERNEL:-7}"

EXTRA_ARGS=()
if [[ "$USE_CONTEXT_DELTA_HEADS" == "1" ]]; then
  EXTRA_ARGS+=(--use-context-delta-heads)
fi
if [[ "$USE_REGION_ATTENTION_POOLING" == "1" ]]; then
  EXTRA_ARGS+=(--use-region-attention-pooling)
fi
if [[ "$USE_REGION_EMBEDDING_HEAD" == "1" ]]; then
  EXTRA_ARGS+=(--use-region-embedding-head)
fi
if [[ "$USE_CBRAMOD_CRISS_CROSS" == "1" ]]; then
  EXTRA_ARGS+=(--use-cbramod-criss-cross)
fi
if [[ "$USE_ADAPTIVE_POSITION_ENCODING" == "1" ]]; then
  EXTRA_ARGS+=(--use-adaptive-position-encoding)
fi
if [[ "$DISABLE_ADAPTIVE_POSITION_ENCODING" == "1" ]]; then
  EXTRA_ARGS+=(--disable-adaptive-position-encoding)
fi
EXTRA_ARGS+=(
  --adaptive-position-spatial-kernel "$ADAPTIVE_POSITION_SPATIAL_KERNEL"
  --adaptive-position-temporal-kernel "$ADAPTIVE_POSITION_TEMPORAL_KERNEL"
)

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$PREPROCESSED_DIR/index.csv" ]]; then
  python3 -u code/tfm_soz/preprocess_private_tfm_soz_segments.py \
    --manifest private_sz_union_relabel_manifest_0622_fix.csv \
    --output-dir "$PREPROCESSED_DIR" \
    --pre-sec 5 \
    --onset-sec 5 \
    --post-sec 5
else
  echo "Using existing preprocessed data: $PREPROCESSED_DIR"
fi

python3 -u code/tfm_soz/train_private_soz_segments.py \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer-epochs "$TOKENIZER_EPOCHS" \
  --classifier-epochs "$CLASSIFIER_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --split-seed "$SPLIT_SEED" \
  --val-patients "$VAL_PATIENTS" \
  --hard-negative-fraction "$HARD_NEGATIVE_FRACTION" \
  --channel-hard-negative-weight "$CHANNEL_HARD_NEGATIVE_WEIGHT" \
  --region-hard-negative-weight "$REGION_HARD_NEGATIVE_WEIGHT" \
  --channel-top1-margin-weight "$CHANNEL_TOP1_MARGIN_WEIGHT" \
  --region-top1-margin-weight "$REGION_TOP1_MARGIN_WEIGHT" \
  --top1-margin "$TOP1_MARGIN" \
  --channel-positive-set-margin-weight "$CHANNEL_POSITIVE_SET_MARGIN_WEIGHT" \
  --region-positive-set-margin-weight "$REGION_POSITIVE_SET_MARGIN_WEIGHT" \
  --positive-set-margin "$POSITIVE_SET_MARGIN" \
  --augment-segment-reconstruct-prob "$AUGMENT_SEGMENT_RECONSTRUCT_PROB" \
  --augment-segment-reconstruct-pieces "$AUGMENT_SEGMENT_RECONSTRUCT_PIECES" \
  --augment-negative-channel-drop-prob "$AUGMENT_NEGATIVE_CHANNEL_DROP_PROB" \
  --augment-negative-channel-drop-max-fraction "$AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION" \
  --augment-label-preserving-time-mask-prob "$AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB" \
  --augment-label-preserving-time-mask-max-sec "$AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC" \
  "${EXTRA_ARGS[@]}"
