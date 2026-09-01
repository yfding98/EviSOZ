#!/usr/bin/env bash
set -euo pipefail

# Rebuild the two distinct authorization-projected TUEV manifests.  The
# holding bundle owns the shared token corpus and deliberately has no fit/held
# roles; the fold bundle is derived from it and is used only by head training.

rtk python3 scripts/build_tuev_morphology_manifest.py \
  --target-v2-directory outputs/deepsoz_target_v2 \
  --deepsoz-source-csv outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv \
  --deepsoz-split-csv outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv \
  --expected-target-v2-target-artifact-sha256 5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed \
  --expected-target-v2-summary-artifact-sha256 1def41d4af3b3446db8a64cac1db658eff9c32c574e838e3a3b8e9b1bb93ec39 \
  --expected-target-v2-readme-artifact-sha256 e8b88190b0c8b10f05f2a67ffe572aa64b3c4ee47d61b2a1ed01b95aa1520196 \
  --expected-deepsoz-source-sha256 4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c \
  --expected-deepsoz-split-sha256 5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57 \
  --public-ledger-bundle outputs/tusz_deepsoz_public_ledger_v1_1_0_20260808 \
  --oof-protocol-bundle outputs/ictal_concept_oof_protocol_v2_20260808 \
  --expected-public-ledger-bundle-sha256 7dbce1daf514c53a5256c31a73571e1b37feda1cea2b8351a8401541d1e5ff87 \
  --expected-public-ledger-build-sha256 d38eda7fc38798903110553fc65aec1ad65e5536e21b69282b1fa1225d5b586f \
  --expected-oof-protocol-artifact-sha256 cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0 \
  --expected-oof-protocol-sha256 a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803 \
  --edf-root /mnt/hd1/dyf/dataset/tuh_eeg_events/v2.0.1/edf \
  --preflight-bundle outputs/tuev_morphology_preflight_v2_20260808/bundle \
  --external-metadata-json outputs/tuev_morphology_preflight_v2_20260808/external_metadata.json \
  --expected-preflight-bundle-manifest-sha256 e0cd694f7d4eb5b0f2f4ad4cce88cc15df4163eb06824d9dc167d84dba6f5bfb \
  --expected-preflight-receipt-sha256 61736badedf91987e8449c44884b21ae9a6c2ce46f9f51395f986e10d568b421 \
  --expected-external-metadata-sha256 76cb1e448d0d5f8919e3870da4c04a7bbe587dfdfa5a210555bfc1bb87d039bb \
  --expected-producer-source-sha256 38d8f3a7fdfad589983731638c967025d0071213473fc8c9639218fd3f77ac4f \
  --expected-preprocessing-policy-sha256 b0aa16b25e3bff9c4a12192eaa4dbbf2ae552f08e9011bd6e841638dc8ac7e39 \
  --expected-standard19-mapping-policy-sha256 fedbb95f12d33056bbc4844692f44fa0b7c26291e41c8bca1f4fe5bd60554b71 \
  --holding-output-directory outputs/tuev_morphology_holding_manifest_v3_20260810 \
  --output-directory outputs/tuev_morphology_fold_manifest_v3_20260810
