#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${EVISOZ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tfm_soz/private_0622_fix_rows119_cbramod_vs_tfm_confusions_final_ensemble}"
LOG="${LOG:-${OUTPUT_DIR}/watcher_v2.log}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
EXPECTED_FOLDS="${EXPECTED_FOLDS:-43}"

CBRAMOD_2033="${CBRAMOD_2033:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbramod_fixedval_seed2033}"
CBRAMOD_2034="${CBRAMOD_2034:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbramod_fixedval_seed2034}"
CBRAMOD_2035="${CBRAMOD_2035:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbramod_fixedval_seed2035}"

TFM_2028="${TFM_2028:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028}"
TFM_2029="${TFM_2029:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2029}"
TFM_2030="${TFM_2030:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2030}"

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

count_metrics() {
  local root="$1"
  find "$root" -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null | wc -l | tr -d ' '
}

while true; do
  n2033="$(count_metrics "$CBRAMOD_2033")"
  n2034="$(count_metrics "$CBRAMOD_2034")"
  n2035="$(count_metrics "$CBRAMOD_2035")"
  printf '[%s] seed2033=%s seed2034=%s seed2035=%s\n' \
    "$(date '+%F %T')" "$n2033" "$n2034" "$n2035" >> "$LOG"

  if [[ "$n2033" == "$EXPECTED_FOLDS" && "$n2034" == "$EXPECTED_FOLDS" && "$n2035" == "$EXPECTED_FOLDS" ]]; then
    python3 code/tfm_soz/compare_lopo_top1_confusions.py \
      --model "CBraMod_2033_2034_2035=${CBRAMOD_2033},${CBRAMOD_2034},${CBRAMOD_2035}" \
      --model "TFM_ensemble_2028_2029_2030=${TFM_2028},${TFM_2029},${TFM_2030}" \
      --output-dir "$OUTPUT_DIR" >> "$LOG" 2>&1
    break
  fi

  sleep "$INTERVAL_SEC"
done
