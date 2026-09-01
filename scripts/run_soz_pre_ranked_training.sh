#!/usr/bin/env bash
set -euo pipefail

# DeepSOZ-style SOZ training entrypoint for the ICA FNSZ TUSZ+private dataset.
#
# Usage:
#   bash scripts/run_soz_pre_ranked_training.sh pretrain
#   bash scripts/run_soz_pre_ranked_training.sh lopo
#   bash scripts/run_soz_pre_ranked_training.sh final
#   VAL_PATIENTS="PATIENT_A,PATIENT_B" bash scripts/run_soz_pre_ranked_training.sh holdout
#   bash scripts/run_soz_pre_ranked_training.sh all

MODE="${1:-all}"

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/soz_pre/preprocessed_ica_fnsz}"
RUN_ROOT="${RUN_ROOT:-outputs/soz_pre/ranked_soz}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${RUN_ROOT}/tusz_pretrain}"
LOPO_DIR="${LOPO_DIR:-${RUN_ROOT}/private_lopo_from_tusz}"
FINAL_DIR="${FINAL_DIR:-${RUN_ROOT}/final_tusz_private}"
HOLDOUT_DIR="${HOLDOUT_DIR:-${RUN_ROOT}/private_holdout}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-40}"
LOPO_EPOCHS="${LOPO_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

TUSZ_SPATIAL_WEIGHT_SCALE="${TUSZ_SPATIAL_WEIGHT_SCALE:-0.25}"
PRIVATE_SPATIAL_WEIGHT_SCALE="${PRIVATE_SPATIAL_WEIGHT_SCALE:-1.0}"
CHANNEL_RANKING_LOSS_WEIGHT="${CHANNEL_RANKING_LOSS_WEIGHT:-0.2}"
REGION_RANKING_LOSS_WEIGHT="${REGION_RANKING_LOSS_WEIGHT:-0.1}"
CHANNEL_REGION_LOSS_WEIGHT="${CHANNEL_REGION_LOSS_WEIGHT:-0.3}"
REGION_POOL_BLEND="${REGION_POOL_BLEND:-0.5}"
REGION_POS_WEIGHT_MODE="${REGION_POS_WEIGHT_MODE:-balanced}"
CHANNEL_POS_WEIGHT_MODE="${CHANNEL_POS_WEIGHT_MODE:-none}"
MAX_POS_WEIGHT="${MAX_POS_WEIGHT:-5.0}"
SAMPLER_WEIGHT_CAP="${SAMPLER_WEIGHT_CAP:-10.0}"
SELECTION_METRIC="${SELECTION_METRIC:-region_macro_f1}"
PATIENT_REGION_THRESHOLD="${PATIENT_REGION_THRESHOLD:-0.5}"
PATIENT_MAX_REGIONS="${PATIENT_MAX_REGIONS:-3}"

VAL_PATIENTS="${VAL_PATIENTS:-}"
MAX_LOPO_PATIENTS="${MAX_LOPO_PATIENTS:-0}"

common_args=(
  --preprocessed_dir "${PREPROCESSED_DIR}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --tusz_spatial_weight_scale "${TUSZ_SPATIAL_WEIGHT_SCALE}"
  --private_spatial_weight_scale "${PRIVATE_SPATIAL_WEIGHT_SCALE}"
  --channel_ranking_loss_weight "${CHANNEL_RANKING_LOSS_WEIGHT}"
  --region_ranking_loss_weight "${REGION_RANKING_LOSS_WEIGHT}"
  --channel_region_loss_weight "${CHANNEL_REGION_LOSS_WEIGHT}"
  --region_pool_blend "${REGION_POOL_BLEND}"
  --region_pos_weight_mode "${REGION_POS_WEIGHT_MODE}"
  --channel_pos_weight_mode "${CHANNEL_POS_WEIGHT_MODE}"
  --max_pos_weight "${MAX_POS_WEIGHT}"
  --sampler_weight_cap "${SAMPLER_WEIGHT_CAP}"
  --selection_metric "${SELECTION_METRIC}"
  --patient_region_threshold "${PATIENT_REGION_THRESHOLD}"
  --patient_max_regions "${PATIENT_MAX_REGIONS}"
)

run_pretrain() {
  mkdir -p "${PRETRAIN_DIR}"
  python3 code/soz_pre/train_region_soz.py \
    "${common_args[@]}" \
    --output_dir "${PRETRAIN_DIR}" \
    --train_splits train \
    --val_splits dev \
    --train_sources tusz \
    --val_sources tusz \
    --source_balance none \
    --epochs "${PRETRAIN_EPOCHS}"
}

run_lopo() {
  mkdir -p "${LOPO_DIR}"
  local checkpoint="${PRETRAIN_DIR}/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing pretrain checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_pre_ranked_training.sh pretrain" >&2
    exit 1
  fi
  python3 code/soz_pre/run_private_lopo.py \
    "${common_args[@]}" \
    --output_dir "${LOPO_DIR}" \
    --init_checkpoint "${checkpoint}" \
    --epochs "${LOPO_EPOCHS}" \
    --max_patients "${MAX_LOPO_PATIENTS}"
}

run_final() {
  mkdir -p "${FINAL_DIR}"
  local checkpoint="${PRETRAIN_DIR}/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing pretrain checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_pre_ranked_training.sh pretrain" >&2
    exit 1
  fi
  python3 code/soz_pre/train_region_soz.py \
    "${common_args[@]}" \
    --output_dir "${FINAL_DIR}" \
    --init_checkpoint "${checkpoint}" \
    --train_splits train,dev,private \
    --val_splits eval \
    --train_sources tusz,private \
    --val_sources tusz \
    --source_balance source \
    --epochs "${FINETUNE_EPOCHS}"
}

run_holdout() {
  if [[ -z "${VAL_PATIENTS}" ]]; then
    echo "Set VAL_PATIENTS before holdout mode, for example:" >&2
    echo "VAL_PATIENTS=\"PATIENT_A,PATIENT_B\" bash scripts/run_soz_pre_ranked_training.sh holdout" >&2
    exit 1
  fi
  mkdir -p "${HOLDOUT_DIR}"
  local checkpoint="${PRETRAIN_DIR}/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing pretrain checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_pre_ranked_training.sh pretrain" >&2
    exit 1
  fi
  python3 code/soz_pre/train_region_soz.py \
    "${common_args[@]}" \
    --output_dir "${HOLDOUT_DIR}" \
    --init_checkpoint "${checkpoint}" \
    --train_splits train,private \
    --val_splits private \
    --train_sources tusz,private \
    --val_sources private \
    --exclude_patients "${VAL_PATIENTS}" \
    --val_patients "${VAL_PATIENTS}" \
    --source_balance source \
    --epochs "${FINETUNE_EPOCHS}" \
    --selection_metric region_macro_f1
}

case "${MODE}" in
  pretrain)
    run_pretrain
    ;;
  lopo)
    run_lopo
    ;;
  final)
    run_final
    ;;
  holdout)
    run_holdout
    ;;
  all)
    run_pretrain
    run_lopo
    run_final
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: pretrain, lopo, final, holdout, all" >&2
    exit 2
    ;;
esac
