#!/usr/bin/env bash
set -euo pipefail

# Development-only CPU replay.  The formal-v4 comparator is the original
# independent-second context=1 producer; this is not a k5 comparison.  The
# command exposes no source-dev, source-eval, private, DeepSOZ SOZ, final, or
# device argument.
if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

rtk python3 scripts/evaluate_labram_k31_vs_formal_v4_paired.py \
  --formal-v4-root outputs/tusz_ictal_concept_formal_v4_20260809 \
  --k31-v1-2-root outputs/labram_ictal_k31_oof_recovery_v1_2_20260810 \
  --expected-formal-v4-manifest-sha256 fold0=b6509080cd6057c0ef5837ecce1d4fb981d55c6956ebe21fa229ccf39b66ba61 \
  --expected-formal-v4-manifest-sha256 fold1=4f4b5cc14644d2ff8fefbb00e345624203b91de857ad692505366684e2716e31 \
  --expected-formal-v4-manifest-sha256 fold2=1de7ff60206e1011fd864f44274b81470668828da34c0704b2c4fc8f32b8449a \
  --expected-formal-v4-manifest-sha256 fold3=826f7a87c7af276d8cf6af46e7b2cc8b5655cd601ca7366b4beb2ff678e684ae \
  --expected-formal-v4-manifest-sha256 fold4=1842830b2dc480c15de9ce1b1caedeb715ac8970728d44e8e040ee8a064323ba \
  --expected-k31-v1-2-manifest-sha256 fold0=c183acd41ea91eb0164180e80e61fe67820c84d0cd72311493327b462741064d \
  --expected-k31-v1-2-manifest-sha256 fold1=1ffd565f5a80259e202f62878ac7585ef28e6fccff44bb1fdaea4aae79a7e1eb \
  --expected-k31-v1-2-manifest-sha256 fold2=c3147f4542a02fdb255e3e33ae396674b744b0fe18a60e15340927c28685c468 \
  --expected-k31-v1-2-manifest-sha256 fold3=d7474077616be3aad24c843f8a39df2b9c8e6e5e16133aff22576fb5e8cc0efc \
  --expected-k31-v1-2-manifest-sha256 fold4=6236fad9ad53951a39976f41093f55f8a85be5c51d2fcd788a3333adb8b03cb9 \
  --master-manifest-bundle outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight \
  --expected-master-manifest-bundle-sha256 73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f \
  --expected-master-manifest-source-sha256 d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8 \
  --master-token-corpus outputs/tusz_ictal_token_corpus_formal_v4_20260809/master \
  --expected-master-token-corpus-index-sha256 a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26 \
  --preprocessing-selection-bundle outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability \
  --expected-preprocessing-selection-artifact-sha256 b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485 \
  --expected-preprocessing-protocol-receipt-sha256 9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196 \
  --target-snapshot outputs/tusz_ictal_prediction_artifacts_formal_v4_20260809/final/native \
  --expected-target-snapshot-manifest-sha256 bc22681928e596ef6564af51f54215e96a9560a21cdeaedef043ccd324596cba \
  --expected-target-snapshot-receipt-sha256 e216338d5112a67d20fcba5d545834af2b84c8896a8713b9919866e839c7953a \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 20260810 \
  --event-microbatch-size 8 \
  --device cuda \
  --output-directory outputs/labram_k31_vs_formal_v4_paired_patient_v1_20260810
