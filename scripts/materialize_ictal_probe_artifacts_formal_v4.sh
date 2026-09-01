#!/usr/bin/env bash
set -euo pipefail

# Replay the frozen six-producer formal-v4 ensemble.  This wrapper exposes no
# score, mask, feature, label, fold assignment, or statistical parameter.

rtk python3 scripts/materialize_ictal_probe_artifacts.py \
  --production-run fold0=outputs/tusz_ictal_concept_formal_v4_20260809/fold0=b6509080cd6057c0ef5837ecce1d4fb981d55c6956ebe21fa229ccf39b66ba61 \
  --production-run fold1=outputs/tusz_ictal_concept_formal_v4_20260809/fold1=4f4b5cc14644d2ff8fefbb00e345624203b91de857ad692505366684e2716e31 \
  --production-run fold2=outputs/tusz_ictal_concept_formal_v4_20260809/fold2=1de7ff60206e1011fd864f44274b81470668828da34c0704b2c4fc8f32b8449a \
  --production-run fold3=outputs/tusz_ictal_concept_formal_v4_20260809/fold3=826f7a87c7af276d8cf6af46e7b2cc8b5655cd601ca7366b4beb2ff678e684ae \
  --production-run fold4=outputs/tusz_ictal_concept_formal_v4_20260809/fold4=1842830b2dc480c15de9ce1b1caedeb715ac8970728d44e8e040ee8a064323ba \
  --production-run final=outputs/tusz_ictal_concept_formal_v4_20260809/final=ca18f7bd498bd913eb213f5cbe2452ce48056345da603fe9e324aa060a78160f \
  --target-v2-directory outputs/deepsoz_target_v2 \
  --deepsoz-source-csv outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv \
  --deepsoz-split-csv outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv \
  --expected-target-v2-target-artifact-sha256 5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed \
  --expected-target-v2-summary-artifact-sha256 1def41d4af3b3446db8a64cac1db658eff9c32c574e838e3a3b8e9b1bb93ec39 \
  --expected-target-v2-readme-artifact-sha256 e8b88190b0c8b10f05f2a67ffe572aa64b3c4ee47d61b2a1ed01b95aa1520196 \
  --expected-deepsoz-source-sha256 4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c \
  --expected-deepsoz-split-sha256 5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57 \
  --public-ledger outputs/tusz_deepsoz_public_ledger_v1_1_0_20260808 \
  --expected-public-ledger-bundle-sha256 7dbce1daf514c53a5256c31a73571e1b37feda1cea2b8351a8401541d1e5ff87 \
  --expected-public-ledger-build-sha256 d38eda7fc38798903110553fc65aec1ad65e5536e21b69282b1fa1225d5b586f \
  --oof-protocol outputs/ictal_concept_oof_protocol_v2_20260808 \
  --expected-oof-protocol-artifact-sha256 cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0 \
  --expected-oof-protocol-receipt-sha256 a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803 \
  --signal-preflight-bundle outputs/deepsoz_signal_preflight_v2_20260809_current \
  --expected-signal-preflight-artifact-sha256 a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66 \
  --expected-signal-preflight-receipt-sha256 10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446 \
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
  --scale-output-directory outputs/tusz_ictal_scale_probe_formal_v4_20260810 \
  --fold-id-output-directory outputs/tusz_ictal_fold_id_probe_formal_v4_20260810
