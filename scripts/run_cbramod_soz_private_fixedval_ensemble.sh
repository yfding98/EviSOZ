#!/usr/bin/env bash
set -euo pipefail

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/tfm_soz/private_0622_fix_rows119_segments_15s}"
MANIFEST="${MANIFEST:-private_sz_union_relabel_manifest_0622_fix.csv}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbramod_fixedval}"
SEEDS="${SEEDS:-2033,2034,2035}"
SPLIT_SEED="${SPLIT_SEED:-2028}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-16}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
MAX_PATIENTS="${MAX_PATIENTS:-0}"
PATIENTS="${PATIENTS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
HARD_NEGATIVE_FRACTION="${HARD_NEGATIVE_FRACTION:-0.25}"
CHANNEL_HARD_NEGATIVE_WEIGHT="${CHANNEL_HARD_NEGATIVE_WEIGHT:-0.05}"
REGION_HARD_NEGATIVE_WEIGHT="${REGION_HARD_NEGATIVE_WEIGHT:-0.25}"
CHANNEL_POSITIVE_SET_MARGIN_WEIGHT="${CHANNEL_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
REGION_POSITIVE_SET_MARGIN_WEIGHT="${REGION_POSITIVE_SET_MARGIN_WEIGHT:-0.0}"
POSITIVE_SET_MARGIN="${POSITIVE_SET_MARGIN:-0.2}"
AUGMENT_SEGMENT_RECONSTRUCT_PROB="${AUGMENT_SEGMENT_RECONSTRUCT_PROB:-0.0}"
RUN_EVAL="${RUN_EVAL:-1}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-2000}"

IFS=',' read -r -a SEED_ITEMS <<< "$SEEDS"
ROOTS=()
for seed in "${SEED_ITEMS[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [[ -z "$seed" ]]; then
    continue
  fi
  root="${OUTPUT_BASE}_seed${seed}"
  ROOTS+=("$root")
  echo "== CBraMod-SOZ LOPO seed ${seed}: ${root} =="
  PREPROCESSED_DIR="$PREPROCESSED_DIR" \
  MANIFEST="$MANIFEST" \
  OUTPUT_ROOT="$root" \
  TOKENIZER_EPOCHS="$TOKENIZER_EPOCHS" \
  CLASSIFIER_EPOCHS="$CLASSIFIER_EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  DEVICE="$DEVICE" \
  MAX_PATIENTS="$MAX_PATIENTS" \
  PATIENTS="$PATIENTS" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  SEED="$seed" \
  SPLIT_SEED="$SPLIT_SEED" \
  VAL_PATIENTS="$VAL_PATIENTS" \
  HARD_NEGATIVE_FRACTION="$HARD_NEGATIVE_FRACTION" \
  CHANNEL_HARD_NEGATIVE_WEIGHT="$CHANNEL_HARD_NEGATIVE_WEIGHT" \
  REGION_HARD_NEGATIVE_WEIGHT="$REGION_HARD_NEGATIVE_WEIGHT" \
  CHANNEL_POSITIVE_SET_MARGIN_WEIGHT="$CHANNEL_POSITIVE_SET_MARGIN_WEIGHT" \
  REGION_POSITIVE_SET_MARGIN_WEIGHT="$REGION_POSITIVE_SET_MARGIN_WEIGHT" \
  POSITIVE_SET_MARGIN="$POSITIVE_SET_MARGIN" \
  AUGMENT_SEGMENT_RECONSTRUCT_PROB="$AUGMENT_SEGMENT_RECONSTRUCT_PROB" \
  USE_CBRAMOD_CRISS_CROSS=1 \
  USE_REGION_ATTENTION_POOLING="${USE_REGION_ATTENTION_POOLING:-1}" \
  USE_REGION_EMBEDDING_HEAD="${USE_REGION_EMBEDDING_HEAD:-1}" \
  scripts/run_tfm_soz_private_segments_lopo.sh
done

if [[ "$RUN_EVAL" != "1" ]]; then
  exit 0
fi

mkdir -p "$OUTPUT_BASE"
roots_csv="$(IFS=,; echo "${ROOTS[*]}")"

python3 code/tfm_soz/evaluate_lopo_global_fusion.py \
  --roots "$roots_csv" \
  --output "$OUTPUT_BASE/lopo_test_global_fusion_equal.json" \
  --objective top1 \
  --alpha-grid-step 0.01

python3 code/tfm_soz/evaluate_lopo_ensemble_operating_points.py \
  --roots "$roots_csv" \
  --output "$OUTPUT_BASE/lopo_test_global_equal_operating_points.json" \
  --fusion-mode global \
  --seed-weight-mode equal \
  --ensemble-objective top1 \
  --alpha-grid-step 0.01

python3 code/tfm_soz/evaluate_lopo_ensemble_operating_points.py \
  --roots "$roots_csv" \
  --output "$OUTPUT_BASE/lopo_test_global_equal_top1_augmented_operating_points.json" \
  --fusion-mode global \
  --seed-weight-mode equal \
  --ensemble-objective top1 \
  --alpha-grid-step 0.01 \
  --policy-family top1_augmented

python3 code/tfm_soz/evaluate_lopo_ensemble_constrained_policies.py \
  --roots "$roots_csv" \
  --output "$OUTPUT_BASE/lopo_test_global_equal_top1_augmented_constrained_policies.json" \
  --fusion-mode global \
  --seed-weight-mode equal \
  --ensemble-objective top1 \
  --alpha-grid-step 0.01 \
  --policy-family top1_augmented

python3 code/tfm_soz/evaluate_lopo_ensemble_bootstrap.py \
  --roots "$roots_csv" \
  --output "$OUTPUT_BASE/lopo_test_global_equal_bootstrap_ci.json" \
  --fusion-mode global \
  --seed-weight-mode equal \
  --ensemble-objective top1 \
  --alpha-grid-step 0.01 \
  --bootstrap-iters "$BOOTSTRAP_ITERS"
