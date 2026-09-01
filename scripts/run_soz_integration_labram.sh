#!/usr/bin/env bash
set -euo pipefail

# Two-stage SOZ experiment using code/models/integration_model.py:
#   1) binary seizure/non-seizure stage pretraining to adapt LaBraM weights
#   2) full LaBraM + TimeFilter + brain-network integration model for SOZ
#
# Examples:
#   bash scripts/run_soz_integration_labram.sh stage
#   bash scripts/run_soz_integration_labram.sh tusz
#   FOLD_INDEX=0 bash scripts/run_soz_integration_labram.sh private_fold
#   MAX_PRIVATE_FOLDS=3 bash scripts/run_soz_integration_labram.sh private_lopo

MODE="${1:-tusz}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
if [[ -d "${PYTHON_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

MANIFEST="${MANIFEST:-outputs/soz_pre/unified_region_soz_manifest_tusz_fnsz_ica_private_ica.csv}"
TUSZ_DATA_ROOT="${TUSZ_DATA_ROOT:-outputs/tusz_fnsz_soz_preprocessed}"
PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-outputs/soz_pre/private_ica_preprocessed}"
RUN_ROOT="${RUN_ROOT:-outputs/soz_integration_labram}"

LABRAM_CKPT="${LABRAM_CKPT:-/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth}"
OUTPUT_MODE="${OUTPUT_MODE:-monopolar}"
REGION_LABEL_MODE="${REGION_LABEL_MODE:-coarse}"
BRAIN_NETWORK_FEATURES="${BRAIN_NETWORK_FEATURES:-gc,te,aec,wpli}"

STAGE_EPOCHS="${STAGE_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
STAGE_LR="${STAGE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WORKERS="${WORKERS:-0}"
SEED="${SEED:-42}"
AMP_FLAG="${AMP_FLAG:-}"
SKIP_TEST_EVAL="${SKIP_TEST_EVAL:-0}"

PRE_ONSET_SEC="${PRE_ONSET_SEC:-5}"
POST_ONSET_SEC="${POST_ONSET_SEC:-5}"
STAGE_PRE_ONSET_SEC="${STAGE_PRE_ONSET_SEC:-8}"
STAGE_POST_ONSET_SEC="${STAGE_POST_ONSET_SEC:-4}"

FOLD_INDEX="${FOLD_INDEX:-0}"
PRIVATE_LOO_VAL_OFFSET="${PRIVATE_LOO_VAL_OFFSET:-1}"
MAX_PRIVATE_FOLDS="${MAX_PRIVATE_FOLDS:-0}"
STAGE_OUTPUT_DIR="${STAGE_OUTPUT_DIR:-${RUN_ROOT}/stage_pretrain}"
STAGE_CKPT="${STAGE_CKPT:-${STAGE_OUTPUT_DIR}/best_pretrain_ckpt.pth}"

common_args=(
  --manifest "${MANIFEST}"
  --tusz-data-root "${TUSZ_DATA_ROOT}"
  --private-data-root "${PRIVATE_DATA_ROOT}"
  --labram-ckpt "${LABRAM_CKPT}"
  --output-mode "${OUTPUT_MODE}"
  --region-label-mode "${REGION_LABEL_MODE}"
  --brain-network-features "${BRAIN_NETWORK_FEATURES}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --stage-lr "${STAGE_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --stage-epochs "${STAGE_EPOCHS}"
  --finetune-epochs "${FINETUNE_EPOCHS}"
  --pre-onset-sec "${PRE_ONSET_SEC}"
  --post-onset-sec "${POST_ONSET_SEC}"
  --stage-pre-onset-sec "${STAGE_PRE_ONSET_SEC}"
  --stage-post-onset-sec "${STAGE_POST_ONSET_SEC}"
)

if [[ -n "${AMP_FLAG}" ]]; then
  common_args+=(--amp)
fi

if [[ "${SKIP_TEST_EVAL}" == "1" ]]; then
  common_args+=(--skip-test-eval)
fi

run_stage() {
  mkdir -p "${STAGE_OUTPUT_DIR}"
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    --stage-only \
    --source tusz \
    --split-strategy manifest_split \
    --output-dir "${STAGE_OUTPUT_DIR}"
}

ensure_stage_ckpt() {
  if [[ ! -f "${STAGE_CKPT}" ]]; then
    echo "Stage checkpoint not found: ${STAGE_CKPT}" >&2
    echo "Running stage pretraining first..." >&2
    run_stage
  fi
  if [[ ! -f "${STAGE_CKPT}" ]]; then
    echo "Stage checkpoint was not created: ${STAGE_CKPT}" >&2
    echo "Check STAGE_EPOCHS and the stage training log before running SOZ-from-stage modes." >&2
    exit 1
  fi
}

run_tusz() {
  local out_dir="${RUN_ROOT}/tusz"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    --use-pretrain-stage \
    --source tusz \
    --split-strategy manifest_split \
    --output-dir "${out_dir}"
}

run_tusz_from_stage() {
  ensure_stage_ckpt
  local out_dir="${RUN_ROOT}/tusz_from_stage"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    --stage-pretrain-ckpt "${STAGE_CKPT}" \
    --source tusz \
    --split-strategy manifest_split \
    --output-dir "${out_dir}"
}

run_private_fold() {
  ensure_stage_ckpt
  local fold="$1"
  local out_dir="${RUN_ROOT}/private_lopo/fold_${fold}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    --stage-pretrain-ckpt "${STAGE_CKPT}" \
    --source private \
    --split-strategy private_loo \
    --private-loo-fold-index "${fold}" \
    --private-loo-val-offset "${PRIVATE_LOO_VAL_OFFSET}" \
    --output-dir "${out_dir}"
}

count_private_patients() {
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import csv
import sys

manifest = sys.argv[1]
patients = set()
with open(manifest, "r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if str(row.get("source", "")).strip().lower() != "private":
            continue
        patient = str(row.get("base_patient_id") or row.get("patient_id") or "").strip()
        if patient:
            patients.add(patient)
print(len(patients))
PY
}

run_private_lopo() {
  ensure_stage_ckpt
  local n_folds
  n_folds="$(count_private_patients)"
  if [[ "${MAX_PRIVATE_FOLDS}" != "0" ]]; then
    n_folds="${MAX_PRIVATE_FOLDS}"
  fi
  for ((fold = 0; fold < n_folds; fold++)); do
    run_private_fold "${fold}"
  done
}

case "${MODE}" in
  stage)
    run_stage
    ;;
  tusz)
    run_tusz
    ;;
  tusz_from_stage)
    run_tusz_from_stage
    ;;
  private_fold)
    run_private_fold "${FOLD_INDEX}"
    ;;
  private_lopo)
    run_private_lopo
    ;;
  all)
    run_stage
    run_tusz_from_stage
    run_private_lopo
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected: stage, tusz, tusz_from_stage, private_fold, private_lopo, all" >&2
    exit 2
    ;;
esac
