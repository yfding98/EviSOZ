#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT_DIR="outputs/tfm_soz/private_0622_fix_rows119_continuous_fullfile"
GOAL_DIR="outputs/tfm_soz/private_0622_fix_rows119_goal_readiness"
mkdir -p "${OUT_DIR}" "${GOAL_DIR}"

LOG="${OUT_DIR}/cbramod_seed2034_continuous_after_training_idle.log"
exec >>"${LOG}" 2>&1

TRAIN_MARKER="${TRAIN_MARKER:-code/tfm_soz/train_private_soz_segments.py}"
WAIT_SEC="${WAIT_SEC:-600}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-5000}"

LOPO_ROOT="outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbramod_fixedval_seed2034"
PREPROCESSED_DIR="outputs/tfm_soz/private_0622_fix_rows119_segments_15s"
FIXED_SUMMARY="${LOPO_ROOT}/lopo_test_summary_partial.json"

BASE_STEM="${OUT_DIR}/cbramod_seed2034_alltest_file1val_onsetregion_sens80_causal_step5_fullfile"
BASE_JSON="${BASE_STEM}.json"
POST_STEM="${OUT_DIR}/cbramod_seed2034_alltest_file1val_contextclusterrank_fpr4_sens80_causal_step5"
POST_JSON="${POST_STEM}.json"
BOOTSTRAP_JSON="${POST_STEM}_bootstrap.json"
ERROR_PREFIX="${GOAL_DIR}/cbramod_seed2034_continuous_error_hotspots"
BUNDLE_JSON="${GOAL_DIR}/cbramod_seed2034_continuous_candidate_bundle.json"

echo "[$(date '+%F %T')] queued CBraMOD seed2034 continuous evaluation"
echo "log=${LOG}"

active_training_processes() {
  ps -eo pid=,comm=,args= | awk -v marker="${TRAIN_MARKER}" '
    $2 ~ /^python[0-9.]*$/ &&
    index($0, marker) &&
    $0 !~ /python[0-9.]*[[:space:]]+-m[[:space:]]+py_compile/ {
      print
    }
  '
}

while [[ -n "$(active_training_processes)" ]]; do
  echo "[$(date '+%F %T')] waiting for active training processes matching ${TRAIN_MARKER}"
  active_training_processes || true
  sleep "${WAIT_SEC}"
done

echo "[$(date '+%F %T')] no active training processes found; starting continuous evaluation"

if [[ ! -s "${BASE_JSON}" ]]; then
  python3 code/tfm_soz/evaluate_lopo_continuous_fullfile.py \
    --lopo-root "${LOPO_ROOT}" \
    --preprocessed-dir "${PREPROCESSED_DIR}" \
    --output "${BASE_JSON}" \
    --threshold-policy sensitivity_floor \
    --score-source onset_region \
    --min-sensitivity 0.80 \
    --max-val-files 1 \
    --max-test-files 0 \
    --step-sec 5 \
    --event-time-policy causal_end \
    --device cuda \
    --quiet
else
  echo "[$(date '+%F %T')] reuse existing ${BASE_JSON}"
fi

if [[ ! -s "${POST_JSON}" ]]; then
  python3 code/tfm_soz/evaluate_lopo_continuous_fullfile_context_cluster_rank.py \
    --source-json "${BASE_JSON}" \
    --output "${POST_JSON}" \
    --min-sensitivity 0.80 \
    --target-fpr-per-hour 4.0 \
    --quiet
else
  echo "[$(date '+%F %T')] reuse existing ${POST_JSON}"
fi

if [[ ! -s "${BOOTSTRAP_JSON}" ]]; then
  python3 code/tfm_soz/bootstrap_continuous_fullfile.py \
    --summary-json "${POST_JSON}" \
    --event-csv "${POST_STEM}.csv" \
    --candidate-csv "${POST_STEM}_candidates.csv" \
    --output "${BOOTSTRAP_JSON}" \
    --bootstrap-iters "${BOOTSTRAP_ITERS}"
else
  echo "[$(date '+%F %T')] reuse existing ${BOOTSTRAP_JSON}"
fi

if [[ ! -s "${ERROR_PREFIX}_summary.json" ]]; then
  python3 code/tfm_soz/analyze_continuous_fullfile_errors.py \
    --events-csv "${POST_STEM}.csv" \
    --candidates-csv "${POST_STEM}_candidates.csv" \
    --output-prefix "${ERROR_PREFIX}"
else
  echo "[$(date '+%F %T')] reuse existing ${ERROR_PREFIX}_summary.json"
fi

python3 code/tfm_soz/audit_rows119_continuous_candidate_bundle.py \
  --candidate-fixed-summary-json "${FIXED_SUMMARY}" \
  --candidate-continuous-json "${POST_JSON}" \
  --candidate-continuous-bootstrap-json "${BOOTSTRAP_JSON}" \
  --candidate-false-alarm-clusters-csv "${ERROR_PREFIX}_false_alarm_clusters.csv" \
  --candidate-missed-csv "${ERROR_PREFIX}_missed.csv" \
  --output-json "${BUNDLE_JSON}"

python3 code/tfm_soz/summarize_rows119_continuous_frontier.py \
  --output "${GOAL_DIR}/continuous_frontier_summary.json"

python3 code/tfm_soz/make_rows119_goal_readiness_bundle.py

echo "[$(date '+%F %T')] CBraMOD seed2034 continuous queue complete"
