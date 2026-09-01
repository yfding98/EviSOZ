#!/usr/bin/env bash
set -euo pipefail

# Resume the locked formal-v4 production sequence after completed fold0/fold1.
# This wrapper exposes no training hyperparameters and only supplies the
# externally pinned artifact paths and hashes already validated by each CLI.

COMMON_ARGS=(
  --promotion-gate-policy-bundle outputs/ictal_promotion_gate_policy_v1_20260808
  --expected-promotion-gate-policy-artifact-sha256 bbe78d2fa62aa6a21f641227e5fcb05b69252a717fb00f0ecf6c4eb3b8e9bb46
  --expected-promotion-gate-policy-bundle-receipt-sha256 288016771dc8a86f5bd638ecadd61b53008d7d061b04cabaf7ec79737f4fbbb4
  --oof-protocol outputs/ictal_concept_oof_protocol_v2_20260808
  --expected-oof-protocol-artifact-sha256 cd1893031873b81053678316ed36145c1ba572d33ae332d221bc0907e1e0bca0
  --expected-oof-protocol-receipt-sha256 a1668bfaa9b3489851251924d618e2c107503455183bf54e0b44ae1613ed4803
  --public-ledger outputs/tusz_deepsoz_public_ledger_v1_1_0_20260808
  --expected-public-ledger-bundle-sha256 7dbce1daf514c53a5256c31a73571e1b37feda1cea2b8351a8401541d1e5ff87
  --expected-public-ledger-build-sha256 d38eda7fc38798903110553fc65aec1ad65e5536e21b69282b1fa1225d5b586f
  --deepsoz-source-csv outputs/deepsoz_llm_tusz_all_607_20260801/source/TUH_manifest_final.csv
  --expected-deepsoz-source-sha256 4d08552dbb94f1e8e8a3931249d2bd29538233e2282b8d21a39d0f5dd873fd5c
  --deepsoz-split-csv outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv
  --expected-split-manifest-sha256 5062e894ec139ffaf7abc1b8f45b326f50a118cfcb8907bb25ff81dbbaa91d57
  --preprocessing-selection-bundle outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability
  --expected-preprocessing-selection-artifact-sha256 b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485
  --expected-preprocessing-protocol-receipt-sha256 9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196
  --edf-root /mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf
  --device cuda
)

MASTER_EVALUATION_ARGS=(
  --native-evaluation-manifest-bundle outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight
  --expected-native-evaluation-manifest-bundle-sha256 73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f
  --expected-native-evaluation-manifest-source-sha256 d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8
  --native-evaluation-token-corpus outputs/tusz_ictal_token_corpus_formal_v4_20260809/master
  --expected-native-evaluation-token-corpus-index-sha256 a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26
)

run_fold() {
  local selection="$1"
  local fold_directory="$2"
  local manifest_bundle_sha256="$3"
  local manifest_source_sha256="$4"
  local token_index_sha256="$5"

  rtk python3 scripts/train_tusz_ictal_concept.py \
    "${COMMON_ARGS[@]}" \
    "${MASTER_EVALUATION_ARGS[@]}" \
    --selection "$selection" \
    --training-manifest-bundle "outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$fold_directory" \
    --expected-training-manifest-bundle-sha256 "$manifest_bundle_sha256" \
    --expected-training-manifest-source-sha256 "$manifest_source_sha256" \
    --training-token-corpus "outputs/tusz_ictal_token_corpus_formal_v4_20260809/$fold_directory" \
    --expected-training-token-corpus-index-sha256 "$token_index_sha256" \
    --output-directory "outputs/tusz_ictal_concept_formal_v4_20260809/$selection"
}

run_fold \
  fold2 fold_2 \
  15365ee222f2ef041e8ba54266a82208d7798f425a860a04e61b560cf01678ca \
  a0712976693e64dba17b09538f064f2b7197d3464447c90b5ef450a800775b6b \
  f52fc4d9d5f88ca6cd94d358a1ad385d955ac1d3c2b83b27f9ba677a890aab52

run_fold \
  fold3 fold_3 \
  d0b158c80c76f5302ce47e8074b5901bf81e646855493be7760657c70f49bbb4 \
  06af3bec1714c93a110ca7bcdcd83aeac6cb2940492ed442e58f82e836704094 \
  39ae5bd86a21bec577e01ebd8d68bf6d31b783b1d3ed1ac1a0f98b26601c701f

run_fold \
  fold4 fold_4 \
  81eb77a10479778a149fbf02546c039b4141a242a24e93eaf0d46af7ec5e7381 \
  d4de03763212278463da6a67373b0af6f084e9abc1c67b91e41c7ab7787923d9 \
  431b13e959384da011ac7b2353836c74d8b1a3c3e41556871fa3bb4983c40fcf

rtk python3 scripts/train_tusz_ictal_concept.py \
  "${COMMON_ARGS[@]}" \
  --selection final \
  --training-manifest-bundle outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight \
  --expected-training-manifest-bundle-sha256 73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f \
  --expected-training-manifest-source-sha256 d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8 \
  --training-token-corpus outputs/tusz_ictal_token_corpus_formal_v4_20260809/master \
  --expected-training-token-corpus-index-sha256 a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26 \
  --native-evaluation-manifest-bundle outputs/tusz_ictal_native_eval_manifest_source_dev_v2_20260809_current \
  --expected-native-evaluation-manifest-bundle-sha256 6c9a5cbcf91033bd1dc83d6565d87162cf8f669d25c34cb7c30887afad01fbf1 \
  --expected-native-evaluation-manifest-source-sha256 8f041ea7b5d5196ff0f33e559beefd1ac5fe37ec328047d52e137c9ff295bf72 \
  --native-evaluation-token-corpus outputs/tusz_ictal_native_eval_token_corpus_source_dev_v2_20260809_current \
  --expected-native-evaluation-token-corpus-index-sha256 c0f3348e5335721d61ba4cb77a2bcf37361d3edec50cf99e57feb5ba1c5611b7 \
  --native-evaluation-signal-preflight-bundle outputs/deepsoz_signal_preflight_v2_20260809_current \
  --expected-native-evaluation-signal-preflight-artifact-sha256 a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66 \
  --expected-native-evaluation-signal-preflight-receipt-sha256 10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446 \
  --output-directory outputs/tusz_ictal_concept_formal_v4_20260809/final
