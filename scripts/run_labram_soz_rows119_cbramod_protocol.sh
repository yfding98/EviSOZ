#!/usr/bin/env bash
set -euo pipefail

# LaBraM two-stage SOZ experiment aligned to the CBraMod/TFM rows119 protocol:
#   - input samples: outputs/tfm_soz/private_0622_fix_rows119_segments_15s/*.npz
#   - channel space: 32 bipolar rows119 channels
#   - sample layout: 15 s = pre 5 s + onset 5 s + post 5 s
#   - stage1 detection uses a full seizure start/end manifest; stage2 SOZ labels use onset y_segments[1]
#   - split: LOPO test patient with fixed validation patients, matching CBraMod/TFM scripts
#
# Examples:
#   FOLD_INDEX=0 bash scripts/run_labram_soz_rows119_cbramod_protocol.sh fold
#   MAX_FOLDS=3 bash scripts/run_labram_soz_rows119_cbramod_protocol.sh lopo
#   FOLD_INDEX=0 bash scripts/run_labram_soz_rows119_cbramod_protocol.sh stage_fold

MODE="${1:-fold}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
if [[ -d "${PYTHON_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TFM_SEGMENTS_DIR="${TFM_SEGMENTS_DIR:-outputs/tfm_soz/private_0622_fix_rows119_segments_15s}"
TFM_SEGMENTS_INDEX="${TFM_SEGMENTS_INDEX:-index.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/soz_integration_labram/rows119_cbramod_protocol_lopo}"
LABRAM_CKPT="${LABRAM_CKPT:-/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
STAGE_MANIFEST="${STAGE_MANIFEST:-outputs/soz_pre/private_edf_soz_manifest_soft_ica.csv}"
STAGE_DATA_ROOT="${STAGE_DATA_ROOT:-outputs/soz_pre/private_ica_preprocessed}"
STAGE_SOURCE="${STAGE_SOURCE:-private}"

FOLD_INDEX="${FOLD_INDEX:-0}"
MAX_FOLDS="${MAX_FOLDS:-0}"
SEED="${SEED:-42}"
STAGE_EPOCHS="${STAGE_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-0}"
LR="${LR:-1e-4}"
STAGE_LR="${STAGE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BRAIN_NETWORK_FEATURES="${BRAIN_NETWORK_FEATURES:-gc,te,aec,wpli}"
STAGE_SAMPLE_ROLES="${STAGE_SAMPLE_ROLES:-onset mid offset}"
AMP_FLAG="${AMP_FLAG:-}"
SKIP_TEST_EVAL="${SKIP_TEST_EVAL:-0}"
TWO_STAGE="${TWO_STAGE:-1}"

read -r -a STAGE_ROLE_ITEMS <<< "${STAGE_SAMPLE_ROLES}"

common_args=(
  --dataset-format tfm_segments
  --tfm-segments-dir "${TFM_SEGMENTS_DIR}"
  --tfm-segments-index "${TFM_SEGMENTS_INDEX}"
  --split-strategy private_loo
  --private-fixed-val-patients "${VAL_PATIENTS}"
  --labram-ckpt "${LABRAM_CKPT}"
  --stage-manifest "${STAGE_MANIFEST}"
  --stage-data-root "${STAGE_DATA_ROOT}"
  --stage-source "${STAGE_SOURCE}"
  --stage-sample-roles "${STAGE_ROLE_ITEMS[@]}"
  --brain-network-features "${BRAIN_NETWORK_FEATURES}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --lr "${LR}"
  --stage-lr "${STAGE_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --stage-epochs "${STAGE_EPOCHS}"
  --finetune-epochs "${FINETUNE_EPOCHS}"
)

if [[ -n "${AMP_FLAG}" ]]; then
  common_args+=(--amp)
fi

if [[ "${SKIP_TEST_EVAL}" == "1" ]]; then
  common_args+=(--skip-test-eval)
fi

count_rows119_patients() {
  "${PYTHON_BIN}" -c "import csv,sys; rows=csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')); print(len({(r.get('base_patient_id') or r.get('patient_id') or '').strip() for r in rows if (r.get('base_patient_id') or r.get('patient_id') or '').strip()}))" \
    "${TFM_SEGMENTS_DIR}/${TFM_SEGMENTS_INDEX}"
}

run_stage_fold() {
  local fold="$1"
  local out_dir="${OUTPUT_ROOT}/fold_${fold}_stage"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    --private-loo-fold-index "${fold}" \
    --stage-only \
    --output-dir "${out_dir}"
}

run_fold() {
  local fold="$1"
  local out_dir="${OUTPUT_ROOT}/fold_${fold}"
  mkdir -p "${out_dir}"
  local stage_args=()
  if [[ "${TWO_STAGE}" == "1" ]]; then
    stage_args+=(--use-pretrain-stage)
  fi
  "${PYTHON_BIN}" code/models/train_soz_locator_with_brain_networks.py \
    "${common_args[@]}" \
    "${stage_args[@]}" \
    --private-loo-fold-index "${fold}" \
    --output-dir "${out_dir}"
}

run_lopo() {
  local n_folds
  n_folds="$(count_rows119_patients)"
  if [[ "${MAX_FOLDS}" != "0" ]]; then
    n_folds="${MAX_FOLDS}"
  fi
  for ((fold = 0; fold < n_folds; fold++)); do
    run_fold "${fold}"
  done
}

case "${MODE}" in
  stage_fold)
    run_stage_fold "${FOLD_INDEX}"
    ;;
  fold)
    run_fold "${FOLD_INDEX}"
    ;;
  lopo)
    run_lopo
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected: stage_fold, fold, lopo" >&2
    exit 2
    ;;
esac
