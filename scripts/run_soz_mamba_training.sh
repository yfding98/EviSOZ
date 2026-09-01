#!/usr/bin/env bash
set -euo pipefail

# Mamba-inspired SOZ localization experiments.
#
# Usage:
#   bash scripts/run_soz_mamba_training.sh tusz
#   bash scripts/run_soz_mamba_training.sh private_lopo
#   bash scripts/run_soz_mamba_training.sh private_from_tusz
#   bash scripts/run_soz_mamba_training.sh tusz_private
#   bash scripts/run_soz_mamba_training.sh all

MODE="${1:-all}"

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/soz_pre/preprocessed_ica_fnsz}"
RUN_ROOT="${RUN_ROOT:-outputs/soz_mamba}"

TUSZ_DIR="${TUSZ_DIR:-${RUN_ROOT}/tusz_mamba}"
PRIVATE_LOPO_DIR="${PRIVATE_LOPO_DIR:-${RUN_ROOT}/private_lopo_mamba}"
PRIVATE_FROM_TUSZ_DIR="${PRIVATE_FROM_TUSZ_DIR:-${RUN_ROOT}/private_lopo_from_tusz_mamba}"
MIXED_DIR="${MIXED_DIR:-${RUN_ROOT}/tusz_private_mamba}"

EPOCHS="${EPOCHS:-40}"
TUSZ_EPOCHS="${TUSZ_EPOCHS:-${EPOCHS}}"
PRIVATE_EPOCHS="${PRIVATE_EPOCHS:-${EPOCHS}}"
MIXED_EPOCHS="${MIXED_EPOCHS:-${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
MAX_PRIVATE_PATIENTS="${MAX_PRIVATE_PATIENTS:-0}"

D_MODEL="${D_MODEL:-64}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-3}"
SPATIAL_LAYERS="${SPATIAL_LAYERS:-2}"
EXPANSION="${EXPANSION:-2}"
DROPOUT="${DROPOUT:-0.15}"

TUSZ_SPATIAL_WEIGHT_SCALE="${TUSZ_SPATIAL_WEIGHT_SCALE:-0.25}"
PRIVATE_SPATIAL_WEIGHT_SCALE="${PRIVATE_SPATIAL_WEIGHT_SCALE:-1.0}"
REGION_POOL_BLEND="${REGION_POOL_BLEND:-0.5}"
REGION_POS_WEIGHT_MODE="${REGION_POS_WEIGHT_MODE:-balanced}"
CHANNEL_REGION_LOSS_WEIGHT="${CHANNEL_REGION_LOSS_WEIGHT:-0.3}"
SELECTION_METRIC="${SELECTION_METRIC:-region_macro_f1}"
PATIENT_REGION_THRESHOLD="${PATIENT_REGION_THRESHOLD:-0.5}"
PATIENT_MAX_REGIONS="${PATIENT_MAX_REGIONS:-3}"

common_args=(
  --preprocessed_dir "${PREPROCESSED_DIR}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --d_model "${D_MODEL}"
  --temporal_layers "${TEMPORAL_LAYERS}"
  --spatial_layers "${SPATIAL_LAYERS}"
  --expansion "${EXPANSION}"
  --dropout "${DROPOUT}"
  --tusz_spatial_weight_scale "${TUSZ_SPATIAL_WEIGHT_SCALE}"
  --private_spatial_weight_scale "${PRIVATE_SPATIAL_WEIGHT_SCALE}"
  --region_pool_blend "${REGION_POOL_BLEND}"
  --region_pos_weight_mode "${REGION_POS_WEIGHT_MODE}"
  --channel_region_loss_weight "${CHANNEL_REGION_LOSS_WEIGHT}"
  --selection_metric "${SELECTION_METRIC}"
  --patient_region_threshold "${PATIENT_REGION_THRESHOLD}"
  --patient_max_regions "${PATIENT_MAX_REGIONS}"
)

run_tusz() {
  mkdir -p "${TUSZ_DIR}"
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    --output_dir "${TUSZ_DIR}" \
    --train_splits train \
    --val_splits dev \
    --train_sources tusz \
    --val_sources tusz \
    --source_balance none \
    --epochs "${TUSZ_EPOCHS}"
}

run_private_lopo() {
  mkdir -p "${PRIVATE_LOPO_DIR}"
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    --mode lopo \
    --output_dir "${PRIVATE_LOPO_DIR}" \
    --epochs "${PRIVATE_EPOCHS}" \
    --max_patients "${MAX_PRIVATE_PATIENTS}"
}

run_private_from_tusz() {
  mkdir -p "${PRIVATE_FROM_TUSZ_DIR}"
  local checkpoint="${TUSZ_DIR}/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing TUSZ checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_mamba_training.sh tusz" >&2
    exit 1
  fi
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    --mode lopo \
    --output_dir "${PRIVATE_FROM_TUSZ_DIR}" \
    --init_checkpoint "${checkpoint}" \
    --epochs "${PRIVATE_EPOCHS}" \
    --max_patients "${MAX_PRIVATE_PATIENTS}"
}

run_mixed() {
  mkdir -p "${MIXED_DIR}"
  local checkpoint="${TUSZ_DIR}/best_model.pt"
  local init_args=()
  if [[ -f "${checkpoint}" ]]; then
    init_args=(--init_checkpoint "${checkpoint}")
  fi
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    "${init_args[@]}" \
    --output_dir "${MIXED_DIR}" \
    --train_splits train,dev,private \
    --val_splits eval \
    --train_sources tusz,private \
    --val_sources tusz \
    --source_balance source \
    --epochs "${MIXED_EPOCHS}"
}

case "${MODE}" in
  tusz)
    run_tusz
    ;;
  private_lopo)
    run_private_lopo
    ;;
  private_from_tusz)
    run_private_from_tusz
    ;;
  tusz_private)
    run_mixed
    ;;
  all)
    run_tusz
    run_private_lopo
    run_private_from_tusz
    run_mixed
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: tusz, private_lopo, private_from_tusz, tusz_private, all" >&2
    exit 2
    ;;
esac

