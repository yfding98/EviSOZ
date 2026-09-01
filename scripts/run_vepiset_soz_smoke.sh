#!/usr/bin/env bash
set -euo pipefail

# Smoke-test VEPiSet on the two SOZ model families:
#   1) code/soz_pre/model.py via train_region_soz.py
#   2) code/models/integration_model.py via train_soz_locator_with_brain_networks.py

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
VEP_ROOT="${VEP_ROOT:-/mnt/hd1/dyf/dataset/vepiset-dataset/opensource-dataset}"
RUN_ROOT="${RUN_ROOT:-outputs/vepiset_smoke}"
TARGET_SAMPLES="${TARGET_SAMPLES:-2000}"
SEED="${SEED:-42}"

SOZ_PRE_MAX_PER_CLASS="${SOZ_PRE_MAX_PER_CLASS:-4}"
SOZ_PRE_MAX_NON_IED="${SOZ_PRE_MAX_NON_IED:-4}"
BRAIN_MAX_PER_CLASS="${BRAIN_MAX_PER_CLASS:-1}"
BRAIN_MAX_NON_IED="${BRAIN_MAX_NON_IED:-1}"

run_soz_pre() {
  "${PYTHON_BIN}" code/soz_pre/train_region_soz.py \
    --dataset_format vepiset \
    --vepiset_root "${VEP_ROOT}" \
    --vepiset_max_samples_per_class "${SOZ_PRE_MAX_PER_CLASS}" \
    --vepiset_max_non_ied_samples "${SOZ_PRE_MAX_NON_IED}" \
    --vepiset_target_samples "${TARGET_SAMPLES}" \
    --output_dir "${RUN_ROOT}/soz_pre" \
    --epochs 1 \
    --batch_size 4 \
    --d_model 16 \
    --nhead 2 \
    --transformer_layers 1 \
    --dim_feedforward 32 \
    --lstm_hidden_dim 16 \
    --num_workers 0 \
    --source_balance none \
    --region_pool_blend 0.0 \
    --seizure_loss_weight 0.2 \
    --hemisphere_loss_weight 0.0 \
    --propagation_loss_weight 0.0 \
    --seed "${SEED}"
}

run_brain_networks() {
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    --dataset-format vepiset \
    --vepiset-root "${VEP_ROOT}" \
    --vepiset-max-samples-per-class "${BRAIN_MAX_PER_CLASS}" \
    --vepiset-max-non-ied-samples "${BRAIN_MAX_NON_IED}" \
    --vepiset-target-samples "${TARGET_SAMPLES}" \
    --output-dir "${RUN_ROOT}/brain_networks" \
    --fs 500 \
    --patch-duration 0.5 \
    --pre-onset-sec 2 \
    --post-onset-sec 2 \
    --finetune-epochs 1 \
    --batch-size 1 \
    --workers 0 \
    --embed-dim 40 \
    --labram-frozen-layers 12 \
    --freeze-labram \
    --no-eeg-augment \
    --augment-minority-oversample 0 \
    --w-transition 0 \
    --w-pattern 0 \
    --w-hemisphere 0 \
    --task-training-mode multitask \
    --generalized-sample-weight 1.0 \
    --save-every 100 \
    --skip-test-eval \
    --seed "${SEED}"
}

case "${1:-all}" in
  soz_pre)
    run_soz_pre
    ;;
  brain)
    run_brain_networks
    ;;
  all)
    run_soz_pre
    run_brain_networks
    ;;
  *)
    echo "Usage: $0 [all|soz_pre|brain]" >&2
    exit 2
    ;;
esac
