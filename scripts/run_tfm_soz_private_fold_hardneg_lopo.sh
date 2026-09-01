#!/usr/bin/env bash
set -euo pipefail

# Fold-local hard-negative LOPO runner.
#
# For each held-out patient, this script:
# 1. uses the matching source LOPO fold to scan only that fold's train patients,
# 2. converts high-scoring non-onset windows to all-zero hard-negative NPZ files,
# 3. retrains the same held-out fold with only its own fold-local hard negatives.
#
# This avoids the subtle leakage that would happen if hard negatives mined by a
# model trained with the current held-out patient were reused for that patient.

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/tfm_soz/private_0622_fix_rows119_segments_15s}"
SOURCE_LOPO_ROOT="${SOURCE_LOPO_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2030}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_foldhardneg_seed2030}"
HARDNEG_ROOT="${HARDNEG_ROOT:-outputs/tfm_soz/private_0622_fix_rows119_foldhardneg_seed2030}"

PATIENTS="${PATIENTS:-}"
MAX_PATIENTS="${MAX_PATIENTS:-0}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_MINE="${FORCE_MINE:-0}"
FORCE_HARDNEG="${FORCE_HARDNEG:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INIT_FROM_SOURCE="${INIT_FROM_SOURCE:-0}"

SEED="${SEED:-2030}"
SPLIT_SEED="${SPLIT_SEED:-2028}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-16}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMB_SIZE="${EMB_SIZE:-64}"
CODE_BOOK_SIZE="${CODE_BOOK_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

MINING_DEVICE="${MINING_DEVICE:-$DEVICE}"
MINING_THRESHOLD_POLICY="${MINING_THRESHOLD_POLICY:-fixed_window}"
MINING_MAX_TRAIN_FILES="${MINING_MAX_TRAIN_FILES:-0}"
MINING_MAX_VAL_FILES="${MINING_MAX_VAL_FILES:-0}"
MINING_STEP_SEC="${MINING_STEP_SEC:-5}"
MINING_BATCH_SIZE="${MINING_BATCH_SIZE:-8}"
MINING_WINDOW_CHUNK_SIZE="${MINING_WINDOW_CHUNK_SIZE:-256}"
MINING_READ_WINDOW_CHUNK_SIZE="${MINING_READ_WINDOW_CHUNK_SIZE:-64}"
MINING_TARGET_FPR_PER_HOUR="${MINING_TARGET_FPR_PER_HOUR:-10}"
MINING_MIN_SENSITIVITY="${MINING_MIN_SENSITIVITY:-0.8}"

HARDNEG_SCORE_COLUMN="${HARDNEG_SCORE_COLUMN:-max_region_score}"
HARDNEG_MIN_SCORE="${HARDNEG_MIN_SCORE:-0.70}"
HARDNEG_SELECTION_MODE="${HARDNEG_SELECTION_MODE:-window}"
HARDNEG_CLUSTER_GAP_SEC="${HARDNEG_CLUSTER_GAP_SEC:-15}"
HARDNEG_CLUSTER_MIN_SIZE="${HARDNEG_CLUSTER_MIN_SIZE:-1}"
HARDNEG_WINDOWS_PER_CLUSTER="${HARDNEG_WINDOWS_PER_CLUSTER:-3}"
HARDNEG_MAX_WINDOWS="${HARDNEG_MAX_WINDOWS:-0}"
HARDNEG_MAX_WINDOWS_PER_PATIENT="${HARDNEG_MAX_WINDOWS_PER_PATIENT:-20}"
HARDNEG_MAX_WINDOWS_PER_FILE="${HARDNEG_MAX_WINDOWS_PER_FILE:-5}"
HARDNEG_MIN_GAP_SEC="${HARDNEG_MIN_GAP_SEC:-30}"

BACKGROUND_NEGATIVE_RATIO="${BACKGROUND_NEGATIVE_RATIO:-1.0}"
BACKGROUND_NEGATIVE_MAX_SAMPLES="${BACKGROUND_NEGATIVE_MAX_SAMPLES:-0}"
BACKGROUND_NEGATIVE_SEED="${BACKGROUND_NEGATIVE_SEED:-$SEED}"
BACKGROUND_NEGATIVE_SAMPLING_MODE="${BACKGROUND_NEGATIVE_SAMPLING_MODE:-random}"

HARD_NEGATIVE_FRACTION="${HARD_NEGATIVE_FRACTION:-0.25}"
CHANNEL_HARD_NEGATIVE_WEIGHT="${CHANNEL_HARD_NEGATIVE_WEIGHT:-0.05}"
REGION_HARD_NEGATIVE_WEIGHT="${REGION_HARD_NEGATIVE_WEIGHT:-0.25}"
CHANNEL_TOP1_MARGIN_WEIGHT="${CHANNEL_TOP1_MARGIN_WEIGHT:-0.0}"
REGION_TOP1_MARGIN_WEIGHT="${REGION_TOP1_MARGIN_WEIGHT:-0.0}"
TOP1_MARGIN="${TOP1_MARGIN:-0.2}"
CHANNEL_POSITIVE_SET_MARGIN_WEIGHT="${CHANNEL_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
REGION_POSITIVE_SET_MARGIN_WEIGHT="${REGION_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
POSITIVE_SET_MARGIN="${POSITIVE_SET_MARGIN:-0.2}"
CHANNEL_CONTEXT_MAX_WEIGHT="${CHANNEL_CONTEXT_MAX_WEIGHT:-0.0}"
REGION_CONTEXT_MAX_WEIGHT="${REGION_CONTEXT_MAX_WEIGHT:-0.0}"
CHANNEL_CONTEXT_FOCAL_WEIGHT="${CHANNEL_CONTEXT_FOCAL_WEIGHT:-0.0}"
REGION_CONTEXT_FOCAL_WEIGHT="${REGION_CONTEXT_FOCAL_WEIGHT:-0.0}"
CHANNEL_POSITIVE_CONTEXT_MARGIN_WEIGHT="${CHANNEL_POSITIVE_CONTEXT_MARGIN_WEIGHT:-0.0}"
REGION_POSITIVE_CONTEXT_MARGIN_WEIGHT="${REGION_POSITIVE_CONTEXT_MARGIN_WEIGHT:-0.0}"
REGION_EMBEDDING_CONTRASTIVE_WEIGHT="${REGION_EMBEDDING_CONTRASTIVE_WEIGHT:-0.0}"
REGION_EMBEDDING_CONTRASTIVE_MARGIN="${REGION_EMBEDDING_CONTRASTIVE_MARGIN:-0.5}"
REGION_BACKGROUND_EMBEDDING_COMPACT_WEIGHT="${REGION_BACKGROUND_EMBEDDING_COMPACT_WEIGHT:-0.25}"
POSITIVE_CONTEXT_MARGIN="${POSITIVE_CONTEXT_MARGIN:-0.2}"
CONTEXT_FOCAL_GAMMA="${CONTEXT_FOCAL_GAMMA:-2.0}"
USE_SEGMENT_EVENTNESS_HEAD="${USE_SEGMENT_EVENTNESS_HEAD:-0}"
SEGMENT_EVENTNESS_LOSS_WEIGHT="${SEGMENT_EVENTNESS_LOSS_WEIGHT:-0.0}"
SEGMENT_EVENTNESS_CONTRASTIVE_WEIGHT="${SEGMENT_EVENTNESS_CONTRASTIVE_WEIGHT:-0.0}"
SEGMENT_EVENTNESS_CONTRASTIVE_MARGIN="${SEGMENT_EVENTNESS_CONTRASTIVE_MARGIN:-0.2}"
SEGMENT_EVENTNESS_BACKGROUND_MAX_WEIGHT="${SEGMENT_EVENTNESS_BACKGROUND_MAX_WEIGHT:-0.0}"
SEGMENT_EVENTNESS_CLUSTER_BACKGROUND_MAX_WEIGHT="${SEGMENT_EVENTNESS_CLUSTER_BACKGROUND_MAX_WEIGHT:-0.0}"
CHANNEL_BACKGROUND_MAX_WEIGHT="${CHANNEL_BACKGROUND_MAX_WEIGHT:-0.0}"
CHANNEL_CLUSTER_BACKGROUND_MAX_WEIGHT="${CHANNEL_CLUSTER_BACKGROUND_MAX_WEIGHT:-0.0}"
REGION_BACKGROUND_MAX_WEIGHT="${REGION_BACKGROUND_MAX_WEIGHT:-0.0}"
REGION_CLUSTER_BACKGROUND_MAX_WEIGHT="${REGION_CLUSTER_BACKGROUND_MAX_WEIGHT:-0.0}"
AUGMENT_NOISE_STD="${AUGMENT_NOISE_STD:-0.03}"
AUGMENT_SCALE_STD="${AUGMENT_SCALE_STD:-0.08}"
AUGMENT_SEGMENT_RECONSTRUCT_PROB="${AUGMENT_SEGMENT_RECONSTRUCT_PROB:-0.0}"
AUGMENT_SEGMENT_RECONSTRUCT_PIECES="${AUGMENT_SEGMENT_RECONSTRUCT_PIECES:-4}"
AUGMENT_NEGATIVE_CHANNEL_DROP_PROB="${AUGMENT_NEGATIVE_CHANNEL_DROP_PROB:-0.0}"
AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION="${AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION:-0.25}"
AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB="${AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB:-0.0}"
AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC="${AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC:-0.5}"

USE_CONTEXT_DELTA_HEADS="${USE_CONTEXT_DELTA_HEADS:-0}"
USE_REGION_ATTENTION_POOLING="${USE_REGION_ATTENTION_POOLING:-1}"
USE_REGION_EMBEDDING_HEAD="${USE_REGION_EMBEDDING_HEAD:-1}"
USE_CBRAMOD_CRISS_CROSS="${USE_CBRAMOD_CRISS_CROSS:-0}"
USE_ADAPTIVE_POSITION_ENCODING="${USE_ADAPTIVE_POSITION_ENCODING:-0}"
DISABLE_ADAPTIVE_POSITION_ENCODING="${DISABLE_ADAPTIVE_POSITION_ENCODING:-0}"
ADAPTIVE_POSITION_SPATIAL_KERNEL="${ADAPTIVE_POSITION_SPATIAL_KERNEL:-19}"
ADAPTIVE_POSITION_TEMPORAL_KERNEL="${ADAPTIVE_POSITION_TEMPORAL_KERNEL:-7}"

VALIDATION_BACKGROUND_NEGATIVE_DIR="${VALIDATION_BACKGROUND_NEGATIVE_DIR:-}"
VALIDATION_BACKGROUND_MAX_SAMPLES="${VALIDATION_BACKGROUND_MAX_SAMPLES:-0}"
VALIDATION_BACKGROUND_SEED="${VALIDATION_BACKGROUND_SEED:--1}"
VALIDATION_BACKGROUND_SELECTION_WEIGHT="${VALIDATION_BACKGROUND_SELECTION_WEIGHT:-0}"

SUMMARIZE_LOPO="${SUMMARIZE_LOPO:-1}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$OUTPUT_ROOT/lopo_test_summary.json}"

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

patient_list="$(mktemp)"
cleanup() {
  rm -f "$patient_list"
}
trap cleanup EXIT

if [[ -n "$PATIENTS" ]]; then
  printf '%s\n' "$PATIENTS" | tr ',' '\n' | sed '/^$/d' > "$patient_list"
else
python3 - <<'PY' "$PREPROCESSED_DIR" "$MAX_PATIENTS" > "$patient_list"
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader((Path(sys.argv[1]) / "index.csv").open(encoding="utf-8-sig")))
patients = sorted({row["base_patient_id"] for row in rows})
max_patients = int(sys.argv[2])
if max_patients > 0:
    patients = patients[:max_patients]
for patient in patients:
    print(patient)
PY
fi

common_model_args=(
  --adaptive-position-spatial-kernel "$ADAPTIVE_POSITION_SPATIAL_KERNEL"
  --adaptive-position-temporal-kernel "$ADAPTIVE_POSITION_TEMPORAL_KERNEL"
)
if [[ "$USE_CONTEXT_DELTA_HEADS" == "1" ]]; then
  common_model_args+=(--use-context-delta-heads)
fi
if [[ "$USE_SEGMENT_EVENTNESS_HEAD" == "1" ]]; then
  common_model_args+=(--use-segment-eventness-head)
fi
if [[ "$USE_REGION_ATTENTION_POOLING" == "1" ]]; then
  common_model_args+=(--use-region-attention-pooling)
fi
if [[ "$USE_REGION_EMBEDDING_HEAD" == "1" ]]; then
  common_model_args+=(--use-region-embedding-head)
fi
if [[ "$USE_CBRAMOD_CRISS_CROSS" == "1" ]]; then
  common_model_args+=(--use-cbramod-criss-cross)
fi
if [[ "$USE_ADAPTIVE_POSITION_ENCODING" == "1" ]]; then
  common_model_args+=(--use-adaptive-position-encoding)
fi
if [[ "$DISABLE_ADAPTIVE_POSITION_ENCODING" == "1" ]]; then
  common_model_args+=(--disable-adaptive-position-encoding)
fi

validation_background_args=()
if [[ -n "$VALIDATION_BACKGROUND_NEGATIVE_DIR" ]]; then
  validation_background_args=(
    --validation-background-negative-dir "$VALIDATION_BACKGROUND_NEGATIVE_DIR"
    --validation-background-max-samples "$VALIDATION_BACKGROUND_MAX_SAMPLES"
    --validation-background-seed "$VALIDATION_BACKGROUND_SEED"
    --validation-background-selection-weight "$VALIDATION_BACKGROUND_SELECTION_WEIGHT"
  )
fi

init_args=()

while IFS= read -r patient; do
  echo "== Fold-local hard-negative LOPO patient: ${patient} =="
  source_fold="$SOURCE_LOPO_ROOT/$patient"
  if [[ ! -f "$source_fold/run_config.json" ]]; then
    echo "Missing source fold for mining: $source_fold" >&2
    exit 1
  fi
  init_args=()
  if [[ "$INIT_FROM_SOURCE" == "1" ]]; then
    init_args=(--init-run-dir "$source_fold")
  fi

  fold_mining_dir="$HARDNEG_ROOT/$patient"
  mining_json="$fold_mining_dir/train_fullfile.json"
  candidate_csv="${mining_json%.json}_candidates.csv"
  hardneg_dir="$fold_mining_dir/hardneg"
  mkdir -p "$fold_mining_dir"

  need_hardneg=0
  if [[ "$FORCE_HARDNEG" == "1" || ! -f "$hardneg_dir/index.csv" ]]; then
    need_hardneg=1
  fi

  if [[ "$FORCE_MINE" == "1" || ( "$need_hardneg" == "1" && ! -f "$candidate_csv" ) ]]; then
    python3 -u code/tfm_soz/evaluate_lopo_continuous_fullfile.py \
      --lopo-root "$SOURCE_LOPO_ROOT" \
      --preprocessed-dir "$PREPROCESSED_DIR" \
      --output "$mining_json" \
      --threshold-policy "$MINING_THRESHOLD_POLICY" \
      --target-fpr-per-hour "$MINING_TARGET_FPR_PER_HOUR" \
      --min-sensitivity "$MINING_MIN_SENSITIVITY" \
      --scan-split train \
      --folds "$patient" \
      --max-test-files "$MINING_MAX_TRAIN_FILES" \
      --max-val-files "$MINING_MAX_VAL_FILES" \
      --step-sec "$MINING_STEP_SEC" \
      --batch-size "$MINING_BATCH_SIZE" \
      --window-chunk-size "$MINING_WINDOW_CHUNK_SIZE" \
      --read-window-chunk-size "$MINING_READ_WINDOW_CHUNK_SIZE" \
      --device "$MINING_DEVICE" \
      --quiet
  elif [[ -f "$candidate_csv" ]]; then
    echo "Using existing mining candidates: $candidate_csv"
  else
    echo "Using existing hard negatives without mining candidates: $hardneg_dir"
  fi

  if [[ "$need_hardneg" == "1" ]]; then
    if [[ ! -f "$candidate_csv" ]]; then
      echo "Missing mining candidates for hard-negative generation: $candidate_csv" >&2
      exit 1
    fi
    python3 -u code/tfm_soz/preprocess_private_tfm_soz_hard_negatives.py \
      --candidates-csv "$candidate_csv" \
      --output-dir "$hardneg_dir" \
      --score-column "$HARDNEG_SCORE_COLUMN" \
      --min-score "$HARDNEG_MIN_SCORE" \
      --selection-mode "$HARDNEG_SELECTION_MODE" \
      --cluster-gap-sec "$HARDNEG_CLUSTER_GAP_SEC" \
      --cluster-min-size "$HARDNEG_CLUSTER_MIN_SIZE" \
      --windows-per-cluster "$HARDNEG_WINDOWS_PER_CLUSTER" \
      --max-windows "$HARDNEG_MAX_WINDOWS" \
      --max-windows-per-patient "$HARDNEG_MAX_WINDOWS_PER_PATIENT" \
      --max-windows-per-file "$HARDNEG_MAX_WINDOWS_PER_FILE" \
      --min-gap-sec "$HARDNEG_MIN_GAP_SEC" \
      --require-scan-split train
  else
    echo "Using existing hard negatives: $hardneg_dir"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$OUTPUT_ROOT/$patient/metrics.json" ]]; then
    echo "Skipping existing fold training: $OUTPUT_ROOT/$patient"
    continue
  fi

  python3 -u code/tfm_soz/train_private_soz_segments.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --output-dir "$OUTPUT_ROOT/$patient" \
    --tokenizer-epochs "$TOKENIZER_EPOCHS" \
    --classifier-epochs "$CLASSIFIER_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --emb-size "$EMB_SIZE" \
    --code-book-size "$CODE_BOOK_SIZE" \
    --seed "$SEED" \
    --split-seed "$SPLIT_SEED" \
    --val-patients "$VAL_PATIENTS" \
    --test-patient "$patient" \
    --background-negative-dir "$hardneg_dir" \
    --background-negative-ratio "$BACKGROUND_NEGATIVE_RATIO" \
    --background-negative-max-samples "$BACKGROUND_NEGATIVE_MAX_SAMPLES" \
    --background-negative-seed "$BACKGROUND_NEGATIVE_SEED" \
    --background-negative-sampling-mode "$BACKGROUND_NEGATIVE_SAMPLING_MODE" \
    --hard-negative-fraction "$HARD_NEGATIVE_FRACTION" \
    --channel-hard-negative-weight "$CHANNEL_HARD_NEGATIVE_WEIGHT" \
    --region-hard-negative-weight "$REGION_HARD_NEGATIVE_WEIGHT" \
    --channel-top1-margin-weight "$CHANNEL_TOP1_MARGIN_WEIGHT" \
    --region-top1-margin-weight "$REGION_TOP1_MARGIN_WEIGHT" \
    --top1-margin "$TOP1_MARGIN" \
    --channel-positive-set-margin-weight "$CHANNEL_POSITIVE_SET_MARGIN_WEIGHT" \
    --region-positive-set-margin-weight "$REGION_POSITIVE_SET_MARGIN_WEIGHT" \
	    --positive-set-margin "$POSITIVE_SET_MARGIN" \
	    --channel-context-max-weight "$CHANNEL_CONTEXT_MAX_WEIGHT" \
	    --region-context-max-weight "$REGION_CONTEXT_MAX_WEIGHT" \
	    --channel-context-focal-weight "$CHANNEL_CONTEXT_FOCAL_WEIGHT" \
	    --region-context-focal-weight "$REGION_CONTEXT_FOCAL_WEIGHT" \
	    --channel-positive-context-margin-weight "$CHANNEL_POSITIVE_CONTEXT_MARGIN_WEIGHT" \
	    --region-positive-context-margin-weight "$REGION_POSITIVE_CONTEXT_MARGIN_WEIGHT" \
	    --region-embedding-contrastive-weight "$REGION_EMBEDDING_CONTRASTIVE_WEIGHT" \
	    --region-embedding-contrastive-margin "$REGION_EMBEDDING_CONTRASTIVE_MARGIN" \
	    --region-background-embedding-compact-weight "$REGION_BACKGROUND_EMBEDDING_COMPACT_WEIGHT" \
	    --positive-context-margin "$POSITIVE_CONTEXT_MARGIN" \
	    --context-focal-gamma "$CONTEXT_FOCAL_GAMMA" \
	    --segment-eventness-loss-weight "$SEGMENT_EVENTNESS_LOSS_WEIGHT" \
	    --segment-eventness-contrastive-weight "$SEGMENT_EVENTNESS_CONTRASTIVE_WEIGHT" \
	    --segment-eventness-contrastive-margin "$SEGMENT_EVENTNESS_CONTRASTIVE_MARGIN" \
	    --segment-eventness-background-max-weight "$SEGMENT_EVENTNESS_BACKGROUND_MAX_WEIGHT" \
	    --segment-eventness-cluster-background-max-weight "$SEGMENT_EVENTNESS_CLUSTER_BACKGROUND_MAX_WEIGHT" \
	    --channel-background-max-weight "$CHANNEL_BACKGROUND_MAX_WEIGHT" \
	    --channel-cluster-background-max-weight "$CHANNEL_CLUSTER_BACKGROUND_MAX_WEIGHT" \
	    --region-background-max-weight "$REGION_BACKGROUND_MAX_WEIGHT" \
	    --region-cluster-background-max-weight "$REGION_CLUSTER_BACKGROUND_MAX_WEIGHT" \
	    --augment-noise-std "$AUGMENT_NOISE_STD" \
	    --augment-scale-std "$AUGMENT_SCALE_STD" \
	    --augment-segment-reconstruct-prob "$AUGMENT_SEGMENT_RECONSTRUCT_PROB" \
	    --augment-segment-reconstruct-pieces "$AUGMENT_SEGMENT_RECONSTRUCT_PIECES" \
	    --augment-negative-channel-drop-prob "$AUGMENT_NEGATIVE_CHANNEL_DROP_PROB" \
	    --augment-negative-channel-drop-max-fraction "$AUGMENT_NEGATIVE_CHANNEL_DROP_MAX_FRACTION" \
	    --augment-label-preserving-time-mask-prob "$AUGMENT_LABEL_PRESERVING_TIME_MASK_PROB" \
	    --augment-label-preserving-time-mask-max-sec "$AUGMENT_LABEL_PRESERVING_TIME_MASK_MAX_SEC" \
	    --device "$DEVICE" \
    "${init_args[@]}" \
    "${common_model_args[@]}" \
    "${validation_background_args[@]}"
done < "$patient_list"

if [[ "$SUMMARIZE_LOPO" == "1" ]]; then
  python3 -u code/tfm_soz/summarize_lopo.py \
    --root "$OUTPUT_ROOT" \
    --split test \
    --output "$SUMMARY_OUTPUT"
fi
