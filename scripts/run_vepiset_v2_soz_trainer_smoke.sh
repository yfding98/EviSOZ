#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the formal SOZ trainer with the VEPiSet-trained integration_model_v2
# shared-backbone initialization. This is an interface/reproducibility check,
# not a clinical SOZ evaluation.

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
VEP_ROOT="${VEP_ROOT:-/mnt/hd1/dyf/dataset/vepiset-dataset/opensource-dataset}"
FORCE="${FORCE:-0}"

INIT_CKPT="${INIT_CKPT:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20/vepiset_v2_backbone_init.pt}"
SMOKE_DIR="${SMOKE_DIR:-outputs/vepiset_v2_soz_trainer_smoke}"

run_if_needed() {
  local name="$1"
  local sentinel="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -e "${sentinel}" ]]; then
    printf '[skip] %s: %s exists\n' "${name}" "${sentinel}"
    return 0
  fi
  printf '[run] %s\n' "${name}"
  "$@"
}

run_if_needed "export VEPiSet v2 backbone init" "${INIT_CKPT}" \
  "${PYTHON_BIN}" scripts/export_vepiset_v2_backbone_init.py --verify-load \
  --output "${INIT_CKPT}"

run_if_needed "v2 SOZ trainer smoke" "${SMOKE_DIR}/train.log" \
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
  --dataset-format vepiset \
  --vepiset-root "${VEP_ROOT}" \
  --model-arch v2_lightweight \
  --init-soz-ckpt "${INIT_CKPT}" \
  --output-dir "${SMOKE_DIR}" \
  --vepiset-max-samples-per-class 1 \
  --vepiset-max-non-ied-samples 1 \
  --finetune-epochs 0 \
  --batch-size 1 \
  --workers 0 \
  --no-eeg-augment \
  --no-private-balanced-sampler \
  --no-private-channel-loss-weight \
  --skip-test-eval

"${PYTHON_BIN}" scripts/audit_vepiset_sota_comparability.py \
  --output-json outputs/vepiset_sota_comparability_audit.json

"${PYTHON_BIN}" scripts/audit_vepiset_goal_requirements.py \
  --backbone-init "${INIT_CKPT}" \
  --trainer-smoke-dir "${SMOKE_DIR}" \
  --sota-comparability outputs/vepiset_sota_comparability_audit.json \
  --output-json outputs/vepiset_goal_requirements_audit.json
