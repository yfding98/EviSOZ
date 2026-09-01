#!/usr/bin/env bash
set -euo pipefail

python3 code/tfm_soz/preprocess_private_tfm_soz.py \
  --manifest private_sz_union_relabel_manifest_0622_fix.csv \
  --output-dir outputs/tfm_soz/private_0622_fix_events_10s \
  --window-sec 10 \
  --pre-onset-sec 2

python3 code/tfm_soz/train_private_soz.py \
  --preprocessed-dir outputs/tfm_soz/private_0622_fix_events_10s \
  --output-dir outputs/tfm_soz/private_0622_fix_tfm_run \
  --tokenizer-epochs 12 \
  --classifier-epochs 60 \
  --batch-size 8 \
  --seed 2026
