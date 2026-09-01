#!/usr/bin/env bash
set -euo pipefail

# Strictly replay one completed formal-v4 producer and publish the native,
# time-only, and mask-only prediction artifacts required by the pre-registered
# ictal promotion gate.  This wrapper does not fit, select, calibrate, or read
# DeepSOZ SOZ targets.  The completed production-run manifest identity must be
# supplied explicitly by the caller.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {fold0|fold1|fold2|fold3|fold4|final} PRODUCTION_RUN_MANIFEST_SHA256" >&2
  exit 2
fi

SELECTION="$1"
PRODUCTION_SHA256="$2"
if [[ ! "$PRODUCTION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PRODUCTION_RUN_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi

case "$SELECTION" in
  fold0)
    FOLD_DIRECTORY="fold_0"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$FOLD_DIRECTORY"
    TRAINING_BUNDLE_SHA256="88857946163e6583795079810d65d078d0d3c325a7dfe1bca86fe29c68e78200"
    TRAINING_SOURCE_SHA256="0cd023bcee1d58dd7e254427837dc0a942b2310a31882134b80d61469ec6510a"
    TRAINING_TOKEN_SHA256="fae930dbbaed5e80af12723909110ab05b3ab7ce86562aa357aa78e8bbbd59b1"
    ;;
  fold1)
    FOLD_DIRECTORY="fold_1"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$FOLD_DIRECTORY"
    TRAINING_BUNDLE_SHA256="f03cb001fcfe32ccb05a646813a6941014212f71f8bc6e11a887c852e3f585c7"
    TRAINING_SOURCE_SHA256="b86a7d982fc1e060cd0dea9086f6adac6a4f3ea9874ca4d1c6206da960e9ac7f"
    TRAINING_TOKEN_SHA256="bb130ae50e857798a80be2cd551565251db85d6fb3ba95a8ba5c0eb49033de72"
    ;;
  fold2)
    FOLD_DIRECTORY="fold_2"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$FOLD_DIRECTORY"
    TRAINING_BUNDLE_SHA256="15365ee222f2ef041e8ba54266a82208d7798f425a860a04e61b560cf01678ca"
    TRAINING_SOURCE_SHA256="a0712976693e64dba17b09538f064f2b7197d3464447c90b5ef450a800775b6b"
    TRAINING_TOKEN_SHA256="f52fc4d9d5f88ca6cd94d358a1ad385d955ac1d3c2b83b27f9ba677a890aab52"
    ;;
  fold3)
    FOLD_DIRECTORY="fold_3"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$FOLD_DIRECTORY"
    TRAINING_BUNDLE_SHA256="d0b158c80c76f5302ce47e8074b5901bf81e646855493be7760657c70f49bbb4"
    TRAINING_SOURCE_SHA256="06af3bec1714c93a110ca7bcdcd83aeac6cb2940492ed442e58f82e836704094"
    TRAINING_TOKEN_SHA256="39ae5bd86a21bec577e01ebd8d68bf6d31b783b1d3ed1ac1a0f98b26601c701f"
    ;;
  fold4)
    FOLD_DIRECTORY="fold_4"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_oof_fold_manifests_v4_1_20260809_current_preflight/$FOLD_DIRECTORY"
    TRAINING_BUNDLE_SHA256="81eb77a10479778a149fbf02546c039b4141a242a24e93eaf0d46af7ec5e7381"
    TRAINING_SOURCE_SHA256="d4de03763212278463da6a67373b0af6f084e9abc1c67b91e41c7ab7787923d9"
    TRAINING_TOKEN_SHA256="431b13e959384da011ac7b2353836c74d8b1a3c3e41556871fa3bb4983c40fcf"
    ;;
  final)
    FOLD_DIRECTORY="master"
    TRAINING_MANIFEST_BUNDLE="outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight"
    TRAINING_BUNDLE_SHA256="73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f"
    TRAINING_SOURCE_SHA256="d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8"
    TRAINING_TOKEN_SHA256="a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26"
    ;;
  *)
    echo "selection must be fold0, fold1, fold2, fold3, fold4, or final" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --production-run "outputs/tusz_ictal_concept_formal_v4_20260809/$SELECTION"
  --expected-production-run-manifest-sha256 "$PRODUCTION_SHA256"
  --training-manifest-bundle "$TRAINING_MANIFEST_BUNDLE"
  --expected-training-manifest-bundle-sha256 "$TRAINING_BUNDLE_SHA256"
  --expected-training-manifest-source-sha256 "$TRAINING_SOURCE_SHA256"
  --training-token-corpus "outputs/tusz_ictal_token_corpus_formal_v4_20260809/$FOLD_DIRECTORY"
  --expected-training-token-corpus-index-sha256 "$TRAINING_TOKEN_SHA256"
  --preprocessing-selection-bundle outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability
  --expected-preprocessing-selection-artifact-sha256 b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485
  --expected-preprocessing-protocol-receipt-sha256 9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196
  --edf-root /mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf
  --output-directory "outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809/$SELECTION"
)

mkdir -p outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809

if [[ "$SELECTION" == "final" ]]; then
  rtk python3 scripts/materialize_ictal_prediction_artifacts.py \
    "${COMMON_ARGS[@]}" \
    --native-evaluation-manifest-bundle outputs/tusz_ictal_native_eval_manifest_source_dev_v2_20260809_current \
    --expected-native-evaluation-manifest-bundle-sha256 6c9a5cbcf91033bd1dc83d6565d87162cf8f669d25c34cb7c30887afad01fbf1 \
    --expected-native-evaluation-manifest-source-sha256 8f041ea7b5d5196ff0f33e559beefd1ac5fe37ec328047d52e137c9ff295bf72 \
    --native-evaluation-token-corpus outputs/tusz_ictal_native_eval_token_corpus_source_dev_v2_20260809_current \
    --expected-native-evaluation-token-corpus-index-sha256 c0f3348e5335721d61ba4cb77a2bcf37361d3edec50cf99e57feb5ba1c5611b7 \
    --native-evaluation-signal-preflight-bundle outputs/deepsoz_signal_preflight_v2_20260809_current \
    --expected-native-evaluation-signal-preflight-artifact-sha256 a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66 \
    --expected-native-evaluation-signal-preflight-receipt-sha256 10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446
else
  rtk python3 scripts/materialize_ictal_prediction_artifacts.py \
    "${COMMON_ARGS[@]}" \
    --native-evaluation-manifest-bundle outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight \
    --expected-native-evaluation-manifest-bundle-sha256 73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f \
    --expected-native-evaluation-manifest-source-sha256 d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8 \
    --native-evaluation-token-corpus outputs/tusz_ictal_token_corpus_formal_v4_20260809/master \
    --expected-native-evaluation-token-corpus-index-sha256 a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26
fi
