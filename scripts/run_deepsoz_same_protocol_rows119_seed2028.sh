#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PRIVATE_DIR="outputs/tfm_soz/private_0622_fix_rows119_deepsoz_baseline_npz"
OUTPUT_DIR="outputs/tfm_soz/private_0622_fix_rows119_external_baselines/deepsoz_same_protocol"
LOG="${OUTPUT_DIR}/run_seed2028.log"

mkdir -p "${OUTPUT_DIR}"
exec >>"${LOG}" 2>&1

echo "[$(date '+%F %T')] start DeepSOZ-style same-protocol rows119 baseline seed2028"
python3 code/baseline/train_soz_lopo.py \
  --private_dir "${PRIVATE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --region_level \
  --input_mode bipolar \
  --region_model standard \
  --epochs 40 \
  --batch_size 16 \
  --seed 2028 \
  --workers 0 \
  --mc_samples 0
echo "[$(date '+%F %T')] complete DeepSOZ-style same-protocol rows119 baseline seed2028"
