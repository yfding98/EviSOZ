#!/usr/bin/env bash
set -euo pipefail

# Development-only sequential OOF runner. Do not launch while the formal I
# recovery owns the GPU. Each fold is written atomically to a distinct path.
MORPH_PREFLIGHT_SHA="d4f54d79c5e9db162cf671497a84d3903f3379e34b0786029bfcdf8d3415e57e"

for MORPH_SELECTION in fold0 fold1 fold2 fold3 fold4; do
  rtk python3 scripts/train_labram_morphology_hierarchical_recovery_oof_v1.py \
    --selection "${MORPH_SELECTION}" \
    --expected-preflight-receipt-sha256 "${MORPH_PREFLIGHT_SHA}" \
    --device cuda
done

