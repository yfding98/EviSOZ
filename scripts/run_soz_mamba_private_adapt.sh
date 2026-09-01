#!/usr/bin/env bash
set -euo pipefail

# Private clinical-SOZ adaptation for SOZ-Mamba.
#
# This entrypoint treats TUSZ as weak-supervised representation pretraining and
# private labels as a separate clinical target: load only the encoder from TUSZ,
# reinitialize all task heads, warm up heads with the encoder frozen, and avoid
# propagation/channel-region losses that are poorly supported by private masks.
#
# Usage:
#   bash scripts/run_soz_mamba_private_adapt.sh pretrain
#   bash scripts/run_soz_mamba_private_adapt.sh private_lopo_encoder
#   bash scripts/run_soz_mamba_private_adapt.sh private_lopo_scratch
#   bash scripts/run_soz_mamba_private_adapt.sh all

MODE="${1:-private_lopo_encoder}"

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/soz_mamba/event_sequences_128s}"
RUN_ROOT="${RUN_ROOT:-outputs/soz_mamba_private_adapt_128s}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${RUN_ROOT}/tusz_mamba}"
PRIVATE_ENCODER_LOPO_DIR="${PRIVATE_ENCODER_LOPO_DIR:-${RUN_ROOT}/private_lopo_encoder}"
PRIVATE_SCRATCH_LOPO_DIR="${PRIVATE_SCRATCH_LOPO_DIR:-${RUN_ROOT}/private_lopo_scratch}"

EPOCHS="${EPOCHS:-40}"
TUSZ_EPOCHS="${TUSZ_EPOCHS:-${EPOCHS}}"
PRIVATE_EPOCHS="${PRIVATE_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
PRIVATE_LR="${PRIVATE_LR:-5e-5}"
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
REGION_POOL_BLEND="${REGION_POOL_BLEND:-0.0}"
REGION_POS_WEIGHT_MODE="${REGION_POS_WEIGHT_MODE:-balanced}"
CHANNEL_POS_WEIGHT_MODE="${CHANNEL_POS_WEIGHT_MODE:-none}"
SELECTION_METRIC="${SELECTION_METRIC:-patient_region_top1_hit}"
PATIENT_REGION_THRESHOLD="${PATIENT_REGION_THRESHOLD:-0.85}"
PATIENT_MAX_REGIONS="${PATIENT_MAX_REGIONS:-2}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-3}"

PRIVATE_CHANNEL_LOSS_WEIGHT="${PRIVATE_CHANNEL_LOSS_WEIGHT:-0.2}"
PRIVATE_REGION_LOSS_WEIGHT="${PRIVATE_REGION_LOSS_WEIGHT:-2.0}"
PRIVATE_PROPAGATION_LOSS_WEIGHT="${PRIVATE_PROPAGATION_LOSS_WEIGHT:-0.0}"
PRIVATE_SEIZURE_LOSS_WEIGHT="${PRIVATE_SEIZURE_LOSS_WEIGHT:-0.3}"
PRIVATE_HEMISPHERE_LOSS_WEIGHT="${PRIVATE_HEMISPHERE_LOSS_WEIGHT:-0.5}"
PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT="${PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT:-0.0}"
PRIVATE_REGION_RANKING_LOSS_WEIGHT="${PRIVATE_REGION_RANKING_LOSS_WEIGHT:-0.2}"
PRIVATE_CHANNEL_REGION_LOSS_WEIGHT="${PRIVATE_CHANNEL_REGION_LOSS_WEIGHT:-0.0}"

common_args=(
  --preprocessed_dir "${PREPROCESSED_DIR}"
  --batch_size "${BATCH_SIZE}"
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
  --channel_pos_weight_mode "${CHANNEL_POS_WEIGHT_MODE}"
  --patient_region_threshold "${PATIENT_REGION_THRESHOLD}"
  --patient_max_regions "${PATIENT_MAX_REGIONS}"
)

private_args=(
  --lr "${PRIVATE_LR}"
  --channel_loss_weight "${PRIVATE_CHANNEL_LOSS_WEIGHT}"
  --region_loss_weight "${PRIVATE_REGION_LOSS_WEIGHT}"
  --propagation_loss_weight "${PRIVATE_PROPAGATION_LOSS_WEIGHT}"
  --seizure_loss_weight "${PRIVATE_SEIZURE_LOSS_WEIGHT}"
  --hemisphere_loss_weight "${PRIVATE_HEMISPHERE_LOSS_WEIGHT}"
  --channel_ranking_loss_weight "${PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT}"
  --region_ranking_loss_weight "${PRIVATE_REGION_RANKING_LOSS_WEIGHT}"
  --channel_region_loss_weight "${PRIVATE_CHANNEL_REGION_LOSS_WEIGHT}"
  --selection_metric "${SELECTION_METRIC}"
  --epochs "${PRIVATE_EPOCHS}"
  --max_patients "${MAX_PRIVATE_PATIENTS}"
)

resolve_checkpoint() {
  if [[ -n "${TUSZ_CHECKPOINT:-}" ]]; then
    echo "${TUSZ_CHECKPOINT}"
    return
  fi
  if [[ -f "${PRETRAIN_DIR}/best_model.pt" ]]; then
    echo "${PRETRAIN_DIR}/best_model.pt"
    return
  fi
  if [[ -f "outputs/soz_mamba_long_128s/tusz_mamba/best_model.pt" ]]; then
    echo "outputs/soz_mamba_long_128s/tusz_mamba/best_model.pt"
    return
  fi
  echo "${PRETRAIN_DIR}/best_model.pt"
}

run_pretrain() {
  mkdir -p "${PRETRAIN_DIR}"
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    --lr "${LR}" \
    --output_dir "${PRETRAIN_DIR}" \
    --train_splits train \
    --val_splits dev \
    --train_sources tusz \
    --val_sources tusz \
    --source_balance none \
    --selection_metric region_macro_f1 \
    --epochs "${TUSZ_EPOCHS}"
}

run_private_lopo_encoder() {
  mkdir -p "${PRIVATE_ENCODER_LOPO_DIR}"
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing TUSZ checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_mamba_private_adapt.sh pretrain" >&2
    exit 1
  fi
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    "${private_args[@]}" \
    --mode lopo \
    --output_dir "${PRIVATE_ENCODER_LOPO_DIR}" \
    --init_checkpoint "${checkpoint}" \
    --init_checkpoint_mode encoder_only \
    --freeze_encoder_epochs "${FREEZE_ENCODER_EPOCHS}"
}

run_private_lopo_scratch() {
  mkdir -p "${PRIVATE_SCRATCH_LOPO_DIR}"
  python3 code/soz_mamba/train_mamba_soz.py \
    "${common_args[@]}" \
    "${private_args[@]}" \
    --mode lopo \
    --output_dir "${PRIVATE_SCRATCH_LOPO_DIR}" \
    --freeze_encoder_epochs 0
}

case "${MODE}" in
  pretrain)
    run_pretrain
    ;;
  private_lopo_encoder)
    run_private_lopo_encoder
    ;;
  private_lopo_scratch)
    run_private_lopo_scratch
    ;;
  all)
    run_pretrain
    run_private_lopo_encoder
    run_private_lopo_scratch
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: pretrain, private_lopo_encoder, private_lopo_scratch, all" >&2
    exit 2
    ;;
esac
