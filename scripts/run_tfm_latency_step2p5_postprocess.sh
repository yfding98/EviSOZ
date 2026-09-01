#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="outputs/tfm_soz/private_0622_fix_rows119_continuous_fullfile/ensemble_2028_2029_2030_alltest_file1val_onsetcontext_sens80_causal_step2p5_fullfile.json"
POST="outputs/tfm_soz/private_0622_fix_rows119_continuous_fullfile/ensemble_2028_2029_2030_alltest_file1val_onsetcontext_step2p5_contextclusterrank_fpr4_sens80_causal.json"
BOOT="outputs/tfm_soz/private_0622_fix_rows119_continuous_fullfile/ensemble_2028_2029_2030_alltest_file1val_onsetcontext_step2p5_contextclusterrank_fpr4_sens80_causal_bootstrap.json"
ERR="outputs/tfm_soz/private_0622_fix_rows119_goal_readiness/onsetcontext_step2p5_contextclusterrank_fpr4_sens80_causal_error_hotspots"
BUNDLE="outputs/tfm_soz/private_0622_fix_rows119_goal_readiness/latency_step2p5_continuous_candidate_bundle.json"
EXTAUD="outputs/tfm_soz/private_0622_fix_rows119_goal_readiness/latency_step2p5_external_baseline_audit.json"

while [[ ! -s "$BASE" ]]; do
  sleep 60
done

python3 code/tfm_soz/evaluate_lopo_continuous_fullfile_context_cluster_rank.py \
  --source-json "$BASE" \
  --output "$POST" \
  --min-sensitivity 0.80 \
  --target-fpr-per-hour 4.0 \
  --quiet

python3 code/tfm_soz/bootstrap_continuous_fullfile.py \
  --summary-json "$POST" \
  --event-csv "${POST%.json}.csv" \
  --candidate-csv "${POST%.json}_candidates.csv" \
  --output "$BOOT" \
  --bootstrap-iters 5000

python3 code/tfm_soz/analyze_continuous_fullfile_errors.py \
  --events-csv "${POST%.json}.csv" \
  --candidates-csv "${POST%.json}_candidates.csv" \
  --output-prefix "$ERR"

python3 code/tfm_soz/audit_rows119_continuous_candidate_bundle.py \
  --candidate-fixed-summary-json outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_global_fusion_fixedval_ensemble_2028_2029_2030/lopo_test_global_fusion_equal.json \
  --candidate-continuous-json "$POST" \
  --candidate-continuous-bootstrap-json "$BOOT" \
  --candidate-false-alarm-clusters-csv "${ERR}_false_alarm_clusters.csv" \
  --candidate-missed-csv "${ERR}_missed.csv" \
  --output-json "$BUNDLE"

python3 code/tfm_soz/audit_rows119_external_baseline.py \
  --continuous-json "$POST" \
  --baseline-json outputs/tfm_soz/private_0622_fix_rows119_goal_readiness/external_baselines/deepsoz_same_protocol_rows119_filled_baseline.json \
  --output "$EXTAUD"

python3 code/tfm_soz/summarize_rows119_continuous_frontier.py \
  --output outputs/tfm_soz/private_0622_fix_rows119_goal_readiness/continuous_frontier_summary.json

python3 code/tfm_soz/make_rows119_goal_readiness_bundle.py
