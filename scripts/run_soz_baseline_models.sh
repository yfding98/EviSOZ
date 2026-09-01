#!/usr/bin/env bash
set -euo pipefail

# Baseline SOZ experiments on the unified TUSZ/private preprocessing output.
#
# Usage:
#   bash scripts/run_soz_baseline_models.sh deepsoz_tusz
#   bash scripts/run_soz_baseline_models.sh deepsoz_private_lopo
#   bash scripts/run_soz_baseline_models.sh deepsoz_private_from_tusz
#   bash scripts/run_soz_baseline_models.sh eegnet_tusz
#   bash scripts/run_soz_baseline_models.sh eegnet_private_lopo
#   bash scripts/run_soz_baseline_models.sh eegnet_private_from_tusz
#   bash scripts/run_soz_baseline_models.sh all

MODE="${1:-all}"

PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/soz_pre/preprocessed_ica_fnsz}"
TUSZ_PREPROCESSED_DIR="${TUSZ_PREPROCESSED_DIR:-${PREPROCESSED_DIR}}"
PRIVATE_PREPROCESSED_DIR="${PRIVATE_PREPROCESSED_DIR:-${PREPROCESSED_DIR}}"
RUN_ROOT="${RUN_ROOT:-outputs/soz_baselines}"

EPOCHS="${EPOCHS:-40}"
TUSZ_EPOCHS="${TUSZ_EPOCHS:-${EPOCHS}}"
PRIVATE_EPOCHS="${PRIVATE_EPOCHS:-${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DEVICE="${DEVICE:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
MAX_PRIVATE_PATIENTS="${MAX_PRIVATE_PATIENTS:-0}"

D_MODEL="${D_MODEL:-64}"
NHEAD="${NHEAD:-4}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
LSTM_HIDDEN_DIM="${LSTM_HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.15}"
ATTENTION_TEMPERATURE="${ATTENTION_TEMPERATURE:-1.0}"

EEGNET_TEMPORAL_FILTERS="${EEGNET_TEMPORAL_FILTERS:-16}"
EEGNET_DEPTH_MULTIPLIER="${EEGNET_DEPTH_MULTIPLIER:-2}"
EEGNET_POINTWISE_FILTERS="${EEGNET_POINTWISE_FILTERS:-32}"
EEGNET_KERNEL_LENGTH="${EEGNET_KERNEL_LENGTH:-64}"
EEGNET_SEPARABLE_KERNEL_LENGTH="${EEGNET_SEPARABLE_KERNEL_LENGTH:-16}"
EEGNET_POOL1="${EEGNET_POOL1:-4}"
EEGNET_POOL2="${EEGNET_POOL2:-8}"

TUSZ_SPATIAL_WEIGHT_SCALE="${TUSZ_SPATIAL_WEIGHT_SCALE:-1.0}"
PRIVATE_SPATIAL_WEIGHT_SCALE="${PRIVATE_SPATIAL_WEIGHT_SCALE:-1.0}"
REGION_POOL_BLEND="${REGION_POOL_BLEND:-0.5}"
REGION_POS_WEIGHT_MODE="${REGION_POS_WEIGHT_MODE:-balanced}"
CHANNEL_POS_WEIGHT_MODE="${CHANNEL_POS_WEIGHT_MODE:-none}"
MAX_POS_WEIGHT="${MAX_POS_WEIGHT:-5.0}"
SELECTION_METRIC="${SELECTION_METRIC:-region_macro_f1}"
PATIENT_REGION_THRESHOLD="${PATIENT_REGION_THRESHOLD:-0.5}"
PATIENT_MAX_REGIONS="${PATIENT_MAX_REGIONS:-3}"

TUSZ_CHANNEL_LOSS_WEIGHT="${TUSZ_CHANNEL_LOSS_WEIGHT:-1.0}"
TUSZ_REGION_LOSS_WEIGHT="${TUSZ_REGION_LOSS_WEIGHT:-1.5}"
TUSZ_PROPAGATION_LOSS_WEIGHT="${TUSZ_PROPAGATION_LOSS_WEIGHT:-0.5}"
TUSZ_SEIZURE_LOSS_WEIGHT="${TUSZ_SEIZURE_LOSS_WEIGHT:-0.5}"
TUSZ_HEMISPHERE_LOSS_WEIGHT="${TUSZ_HEMISPHERE_LOSS_WEIGHT:-0.7}"
TUSZ_CHANNEL_RANKING_LOSS_WEIGHT="${TUSZ_CHANNEL_RANKING_LOSS_WEIGHT:-0.2}"
TUSZ_REGION_RANKING_LOSS_WEIGHT="${TUSZ_REGION_RANKING_LOSS_WEIGHT:-0.1}"
TUSZ_CHANNEL_REGION_LOSS_WEIGHT="${TUSZ_CHANNEL_REGION_LOSS_WEIGHT:-0.3}"

PRIVATE_CHANNEL_LOSS_WEIGHT="${PRIVATE_CHANNEL_LOSS_WEIGHT:-0.2}"
PRIVATE_REGION_LOSS_WEIGHT="${PRIVATE_REGION_LOSS_WEIGHT:-2.0}"
PRIVATE_PROPAGATION_LOSS_WEIGHT="${PRIVATE_PROPAGATION_LOSS_WEIGHT:-0.0}"
PRIVATE_SEIZURE_LOSS_WEIGHT="${PRIVATE_SEIZURE_LOSS_WEIGHT:-0.3}"
PRIVATE_HEMISPHERE_LOSS_WEIGHT="${PRIVATE_HEMISPHERE_LOSS_WEIGHT:-0.3}"
PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT="${PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT:-0.0}"
PRIVATE_REGION_RANKING_LOSS_WEIGHT="${PRIVATE_REGION_RANKING_LOSS_WEIGHT:-0.2}"
PRIVATE_CHANNEL_REGION_LOSS_WEIGHT="${PRIVATE_CHANNEL_REGION_LOSS_WEIGHT:-0.0}"

common_args=(
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --d_model "${D_MODEL}"
  --nhead "${NHEAD}"
  --transformer_layers "${TRANSFORMER_LAYERS}"
  --dim_feedforward "${DIM_FEEDFORWARD}"
  --lstm_hidden_dim "${LSTM_HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --attention_temperature "${ATTENTION_TEMPERATURE}"
  --eegnet_temporal_filters "${EEGNET_TEMPORAL_FILTERS}"
  --eegnet_depth_multiplier "${EEGNET_DEPTH_MULTIPLIER}"
  --eegnet_pointwise_filters "${EEGNET_POINTWISE_FILTERS}"
  --eegnet_kernel_length "${EEGNET_KERNEL_LENGTH}"
  --eegnet_separable_kernel_length "${EEGNET_SEPARABLE_KERNEL_LENGTH}"
  --eegnet_pool1 "${EEGNET_POOL1}"
  --eegnet_pool2 "${EEGNET_POOL2}"
  --region_pool_blend "${REGION_POOL_BLEND}"
  --region_pos_weight_mode "${REGION_POS_WEIGHT_MODE}"
  --channel_pos_weight_mode "${CHANNEL_POS_WEIGHT_MODE}"
  --max_pos_weight "${MAX_POS_WEIGHT}"
  --selection_metric "${SELECTION_METRIC}"
  --patient_region_threshold "${PATIENT_REGION_THRESHOLD}"
  --patient_max_regions "${PATIENT_MAX_REGIONS}"
)

run_tusz() {
  local model="$1"
  local output_dir="${RUN_ROOT}/${model}/tusz"
  mkdir -p "${output_dir}"
  python3 code/soz_pre/train_region_soz.py \
    "${common_args[@]}" \
    --preprocessed_dir "${TUSZ_PREPROCESSED_DIR}" \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --train_splits train \
    --val_splits dev \
    --train_sources tusz \
    --val_sources tusz \
    --source_balance none \
    --tusz_spatial_weight_scale "${TUSZ_SPATIAL_WEIGHT_SCALE}" \
    --private_spatial_weight_scale "${PRIVATE_SPATIAL_WEIGHT_SCALE}" \
    --channel_loss_weight "${TUSZ_CHANNEL_LOSS_WEIGHT}" \
    --region_loss_weight "${TUSZ_REGION_LOSS_WEIGHT}" \
    --propagation_loss_weight "${TUSZ_PROPAGATION_LOSS_WEIGHT}" \
    --seizure_loss_weight "${TUSZ_SEIZURE_LOSS_WEIGHT}" \
    --hemisphere_loss_weight "${TUSZ_HEMISPHERE_LOSS_WEIGHT}" \
    --channel_ranking_loss_weight "${TUSZ_CHANNEL_RANKING_LOSS_WEIGHT}" \
    --region_ranking_loss_weight "${TUSZ_REGION_RANKING_LOSS_WEIGHT}" \
    --channel_region_loss_weight "${TUSZ_CHANNEL_REGION_LOSS_WEIGHT}" \
    --epochs "${TUSZ_EPOCHS}"
}

run_private_lopo() {
  local model="$1"
  local output_dir="${RUN_ROOT}/${model}/private_lopo"
  mkdir -p "${output_dir}"
  python3 code/soz_pre/run_private_lopo.py \
    "${common_args[@]}" \
    --preprocessed_dir "${PRIVATE_PREPROCESSED_DIR}" \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --epochs "${PRIVATE_EPOCHS}" \
    --max_patients "${MAX_PRIVATE_PATIENTS}" \
    --source_balance none \
    --tusz_spatial_weight_scale "${TUSZ_SPATIAL_WEIGHT_SCALE}" \
    --private_spatial_weight_scale "${PRIVATE_SPATIAL_WEIGHT_SCALE}" \
    --channel_loss_weight "${PRIVATE_CHANNEL_LOSS_WEIGHT}" \
    --region_loss_weight "${PRIVATE_REGION_LOSS_WEIGHT}" \
    --propagation_loss_weight "${PRIVATE_PROPAGATION_LOSS_WEIGHT}" \
    --seizure_loss_weight "${PRIVATE_SEIZURE_LOSS_WEIGHT}" \
    --hemisphere_loss_weight "${PRIVATE_HEMISPHERE_LOSS_WEIGHT}" \
    --channel_ranking_loss_weight "${PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT}" \
    --region_ranking_loss_weight "${PRIVATE_REGION_RANKING_LOSS_WEIGHT}" \
    --channel_region_loss_weight "${PRIVATE_CHANNEL_REGION_LOSS_WEIGHT}"
}

run_private_from_tusz() {
  local model="$1"
  local output_dir="${RUN_ROOT}/${model}/private_lopo_from_tusz"
  local checkpoint="${RUN_ROOT}/${model}/tusz/best_model.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing TUSZ checkpoint: ${checkpoint}" >&2
    echo "Run: bash scripts/run_soz_baseline_models.sh ${model}_tusz" >&2
    exit 1
  fi
  mkdir -p "${output_dir}"
  python3 code/soz_pre/run_private_lopo.py \
    "${common_args[@]}" \
    --preprocessed_dir "${PRIVATE_PREPROCESSED_DIR}" \
    --model "${model}" \
    --output_dir "${output_dir}" \
    --init_checkpoint "${checkpoint}" \
    --epochs "${PRIVATE_EPOCHS}" \
    --max_patients "${MAX_PRIVATE_PATIENTS}" \
    --source_balance none \
    --tusz_spatial_weight_scale "${TUSZ_SPATIAL_WEIGHT_SCALE}" \
    --private_spatial_weight_scale "${PRIVATE_SPATIAL_WEIGHT_SCALE}" \
    --channel_loss_weight "${PRIVATE_CHANNEL_LOSS_WEIGHT}" \
    --region_loss_weight "${PRIVATE_REGION_LOSS_WEIGHT}" \
    --propagation_loss_weight "${PRIVATE_PROPAGATION_LOSS_WEIGHT}" \
    --seizure_loss_weight "${PRIVATE_SEIZURE_LOSS_WEIGHT}" \
    --hemisphere_loss_weight "${PRIVATE_HEMISPHERE_LOSS_WEIGHT}" \
    --channel_ranking_loss_weight "${PRIVATE_CHANNEL_RANKING_LOSS_WEIGHT}" \
    --region_ranking_loss_weight "${PRIVATE_REGION_RANKING_LOSS_WEIGHT}" \
    --channel_region_loss_weight "${PRIVATE_CHANNEL_REGION_LOSS_WEIGHT}"
}

case "${MODE}" in
  deepsoz_tusz)
    run_tusz deepsoz
    ;;
  deepsoz_private_lopo)
    run_private_lopo deepsoz
    ;;
  deepsoz_private_from_tusz)
    run_private_from_tusz deepsoz
    ;;
  eegnet_tusz)
    run_tusz eegnet
    ;;
  eegnet_private_lopo)
    run_private_lopo eegnet
    ;;
  eegnet_private_from_tusz)
    run_private_from_tusz eegnet
    ;;
  all)
    run_tusz deepsoz
    run_private_lopo deepsoz
    run_private_from_tusz deepsoz
    run_tusz eegnet
    run_private_lopo eegnet
    run_private_from_tusz eegnet
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: deepsoz_tusz, deepsoz_private_lopo, deepsoz_private_from_tusz, eegnet_tusz, eegnet_private_lopo, eegnet_private_from_tusz, all" >&2
    exit 2
    ;;
esac
