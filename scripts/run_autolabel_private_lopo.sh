#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   rtk bash scripts/run_autolabel_private_lopo.sh [DATA_DIR] [OUTPUT_DIR] [EPOCHS] ["SEEDS"]

AUTOLABEL_DATA_DIR="${1:-outputs/soz_crossfilter/autolabel_pseudo_private_recentered_v1_20260719}"
AUTOLABEL_OUTPUT_DIR="${2:-outputs/soz_crossfilter/autolabel_private_lopo_v1_20260719}"
AUTOLABEL_EPOCHS="${3:-20}"
AUTOLABEL_SEEDS="${4:-2026 2027 2028}"

rtk mkdir -p "$AUTOLABEL_OUTPUT_DIR"

for AUTOLABEL_SEED in $AUTOLABEL_SEEDS; do
  AUTOLABEL_SEED_DIR="$AUTOLABEL_OUTPUT_DIR/seed_${AUTOLABEL_SEED}"
  AUTOLABEL_TFM_ROOT="$AUTOLABEL_SEED_DIR/tfm"
  rtk python -u -m code.soz_crossfilter.run_lopo_private_rows119 \
    --preprocessed-dir "$AUTOLABEL_DATA_DIR" \
    --output-root "$AUTOLABEL_TFM_ROOT" \
    --sources private \
    --epochs "$AUTOLABEL_EPOCHS" \
    --batch-size 16 \
    --seed "$AUTOLABEL_SEED" \
    --model tfm \
    --tfm-tokenizer-epochs 5 \
    --selection-metric vote_balanced_joint \
    --balanced-sampler both \
    --augment-noise-std 0.02 \
    --augment-scale-std 0.05 \
    --augment-time-mask-prob 0.1 \
    --augment-time-mask-max-fraction 0.05 \
    --channel-pos-weight 1.75 \
    --region-pos-weight 1.15 \
    --channel-f1-weight 0.35 \
    --region-f1-weight 0.25 \
    --skip-existing \
    --device cuda

  rtk python -u -m code.soz_crossfilter.run_lopo_private_rows119 \
    --preprocessed-dir "$AUTOLABEL_DATA_DIR" \
    --output-root "$AUTOLABEL_SEED_DIR/cpbf_v3" \
    --sources private \
    --epochs "$AUTOLABEL_EPOCHS" \
    --batch-size 16 \
    --seed "$AUTOLABEL_SEED" \
    --model cpbf_v3 \
    --tfm-init-root "$AUTOLABEL_TFM_ROOT" \
    --selection-metric vote_balanced_joint \
    --balanced-sampler both \
    --augment-noise-std 0.02 \
    --augment-scale-std 0.05 \
    --augment-time-mask-prob 0.1 \
    --augment-time-mask-max-fraction 0.05 \
    --channel-pos-weight 1.75 \
    --region-pos-weight 1.15 \
    --channel-f1-weight 0.35 \
    --region-f1-weight 0.25 \
    --skip-existing \
    --device cuda

  for AUTOLABEL_MODEL in crossfilter deepsoz eegnet szloc; do
    rtk python -u -m code.soz_crossfilter.run_lopo_private_rows119 \
      --preprocessed-dir "$AUTOLABEL_DATA_DIR" \
      --output-root "$AUTOLABEL_SEED_DIR/$AUTOLABEL_MODEL" \
      --sources private \
      --epochs "$AUTOLABEL_EPOCHS" \
      --batch-size 16 \
      --seed "$AUTOLABEL_SEED" \
      --model "$AUTOLABEL_MODEL" \
      --selection-metric vote_balanced_joint \
      --balanced-sampler both \
      --augment-noise-std 0.02 \
      --augment-scale-std 0.05 \
      --augment-time-mask-prob 0.1 \
      --augment-time-mask-max-fraction 0.05 \
      --channel-pos-weight 1.75 \
      --region-pos-weight 1.15 \
      --channel-f1-weight 0.35 \
      --region-f1-weight 0.25 \
      --skip-existing \
      --device cuda
  done
done
