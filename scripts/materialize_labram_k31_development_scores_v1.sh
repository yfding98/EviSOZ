#!/usr/bin/env bash
set -euo pipefail

# The six recovery manifests do not exist until their training jobs publish
# atomically, so their independently recorded hashes are mandatory positional
# arguments.  No target table/source CSV, score, label, mask, event row, or
# fold assignment can be supplied to the Python materializer.

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 FOLD0_SHA FOLD1_SHA FOLD2_SHA FOLD3_SHA FOLD4_SHA FINAL_SHA [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

OUTPUT_DIRECTORY="${7:-outputs/labram_k31_development_scores_v1_20260810}"

rtk python3 scripts/materialize_labram_k31_development_scores.py \
  --recovery-run "fold0=outputs/labram_ictal_k31_oof_recovery_v1_20260810/fold0=$1" \
  --recovery-run "fold1=outputs/labram_ictal_k31_oof_recovery_v1_20260810/fold1=$2" \
  --recovery-run "fold2=outputs/labram_ictal_k31_oof_recovery_v1_20260810/fold2=$3" \
  --recovery-run "fold3=outputs/labram_ictal_k31_oof_recovery_v1_20260810/fold3=$4" \
  --recovery-run "fold4=outputs/labram_ictal_k31_oof_recovery_v1_20260810/fold4=$5" \
  --recovery-run "final=outputs/labram_ictal_k31_oof_recovery_v1_20260810/final=$6" \
  --oof-protocol outputs/ictal_concept_oof_protocol_v2_20260808 \
  --expected-oof-protocol-artifact-sha256 cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0 \
  --expected-oof-protocol-receipt-sha256 a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803 \
  --signal-preflight-bundle outputs/deepsoz_signal_preflight_v2_20260809_current \
  --expected-signal-preflight-artifact-sha256 a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66 \
  --expected-signal-preflight-receipt-sha256 10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446 \
  --expected-target-v2-artifact-sha256 5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed \
  --expected-target-v2-receipt-sha256 80f2b71cfdf23d604849b2d1a52cc36f0b01c593906e3cef74e79d425cc442d3 \
  --expected-target-v2-policy-sha256 bc953272edf638150a7800b01be01261d7b96dfc6db5def5b98cfd6b93dea237 \
  --preprocessing-selection-bundle outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability \
  --expected-preprocessing-selection-artifact-sha256 b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485 \
  --expected-preprocessing-protocol-receipt-sha256 9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196 \
  --source-train-token-corpus outputs/tusz_ictal_token_corpus_formal_v4_20260809/master \
  --expected-source-train-token-index-sha256 a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26 \
  --source-dev-manifest-bundle outputs/tusz_ictal_native_eval_manifest_source_dev_v2_20260809_current \
  --expected-source-dev-manifest-artifact-sha256 6c9a5cbcf91033bd1dc83d6565d87162cf8f669d25c34cb7c30887afad01fbf1 \
  --expected-source-dev-manifest-receipt-sha256 8f041ea7b5d5196ff0f33e559beefd1ac5fe37ec328047d52e137c9ff295bf72 \
  --source-dev-token-corpus outputs/tusz_ictal_native_eval_token_corpus_source_dev_v2_20260809_current \
  --expected-source-dev-token-index-sha256 c0f3348e5335721d61ba4cb77a2bcf37361d3edec50cf99e57feb5ba1c5611b7 \
  --edf-root /mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf \
  --output-directory "$OUTPUT_DIRECTORY"
