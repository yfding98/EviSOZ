#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION_NAME="${SESSION_NAME:-cbratfm_hardbg_targeted_full43_seed2055_2056_2057}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_cbratfm_hardbg_target_temporal_full43_topconfdistill}"
WAIT_SESSIONS="${WAIT_SESSIONS:-cbramod_rows119_seed2035_resume,cbramod_seed2034_continuous_after_idle,cbratfm_distill_topconf_full43,cbratfm_topconf_distill_full43}"
TRAIN_MARKER="${TRAIN_MARKER:-code/tfm_soz/train_private_soz_segments.py}"
WAIT_SEC="${WAIT_SEC:-600}"

MANIFEST="${MANIFEST:-private_sz_union_relabel_manifest_0622_fix.csv}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-outputs/tfm_soz/private_0622_fix_rows119_segments_15s}"
BACKGROUND_NEGATIVE_DIR="${BACKGROUND_NEGATIVE_DIR:-outputs/tfm_soz/private_0622_fix_rows119_segments_15s_background_negatives}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-outputs/tfm_soz/private_0622_fix_rows119_teacher_logits_tfm_2028_2029_2030_full43}"
TEACHER_ROOTS="${TEACHER_ROOTS:-outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2028,outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2029,outputs/tfm_soz/private_0622_fix_rows119_segment_lopo_regionattn_fullfast_fixedval_seed2030}"

SEEDS="${SEEDS:-2055,2056,2057}"
SPLIT_SEED="${SPLIT_SEED:-2028}"
VAL_PATIENTS="${VAL_PATIENTS:-曾静君,李伟恺,杜克华,薛少林,陈芳}"
MAX_PATIENTS="${MAX_PATIENTS:-0}"
TOKENIZER_EPOCHS="${TOKENIZER_EPOCHS:-16}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-5000}"

CONTINUOUS_DIR="${CONTINUOUS_DIR:-outputs/tfm_soz/private_0622_fix_rows119_continuous_fullfile}"
GOAL_DIR="${GOAL_DIR:-outputs/tfm_soz/private_0622_fix_rows119_goal_readiness}"
MODEL_LEVEL_GATING_CONTRACT_JSON="${MODEL_LEVEL_GATING_CONTRACT_JSON:-$GOAL_DIR/model_level_hardbg_gating_contract.json}"
MODEL_LEVEL_GATING_SCHEMA="${MODEL_LEVEL_GATING_SCHEMA:-rows119_model_level_hardbg_gating_contract_v1}"
MODEL_LEVEL_GATING_NEGATIVE_PRIORITY="${MODEL_LEVEL_GATING_NEGATIVE_PRIORITY:-hard_background_or_targeted_temporal_training}"
MODEL_LEVEL_GATING_NON_EXECUTING="${MODEL_LEVEL_GATING_NON_EXECUTING:-1}"
REGION_HARD_BACKGROUND_TARGET_WEIGHT="${REGION_HARD_BACKGROUND_TARGET_WEIGHT:-0.05}"
REGION_HARD_BACKGROUND_TARGET_REGIONS="${REGION_HARD_BACKGROUND_TARGET_REGIONS:-right_temporal,left_temporal}"
REGION_BUDGET_NEAR_MISS_ALIAS="${REGION_BUDGET_NEAR_MISS_ALIAS:-top1_truemargin_diag_causal}"
REGION_BUDGET_NEAR_MISS_TEMPORAL_CLUSTERS="${REGION_BUDGET_NEAR_MISS_TEMPORAL_CLUSTERS:-155}"
REGION_BUDGET_MAX_TEMPORAL_CLUSTERS="${REGION_BUDGET_MAX_TEMPORAL_CLUSTERS:-128}"
REGION_BUDGET_REQUIRED_TEMPORAL_REDUCTION="${REGION_BUDGET_REQUIRED_TEMPORAL_REDUCTION:-27}"
REGION_BUDGET_NEAR_MISS_MISSED_EVENTS="${REGION_BUDGET_NEAR_MISS_MISSED_EVENTS:-0}"
REGION_BUDGET_MAX_MISSED_EVENTS="${REGION_BUDGET_MAX_MISSED_EVENTS:-2}"
REGION_BUDGET_NEAR_MISS_MISSED_EXCESS="${REGION_BUDGET_NEAR_MISS_MISSED_EXCESS:-0}"
REGION_BUDGET_TARGET_SOURCE="${REGION_BUDGET_TARGET_SOURCE:-diagnostic_heldout_test_near_miss}"
REGION_BUDGET_TARGET_PROMOTION_POLICY="${REGION_BUDGET_TARGET_PROMOTION_POLICY:-diagnostic_only_until_fold_validation_target_selection_proven}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME"
  exit 0
fi

mkdir -p "$REPO_ROOT/$OUTPUT_BASE" "$REPO_ROOT/$CONTINUOUS_DIR" "$REPO_ROOT/$GOAL_DIR"

cat > "$REPO_ROOT/$OUTPUT_BASE/queued_command.txt" <<EOF
SESSION_NAME=$SESSION_NAME
WAIT_SESSIONS=$WAIT_SESSIONS
TRAIN_MARKER=$TRAIN_MARKER
WAIT_SEC=$WAIT_SEC
MANIFEST=$MANIFEST
PREPROCESSED_DIR=$PREPROCESSED_DIR
BACKGROUND_NEGATIVE_DIR=$BACKGROUND_NEGATIVE_DIR
TEACHER_CACHE_DIR=$TEACHER_CACHE_DIR
TEACHER_ROOTS=$TEACHER_ROOTS
OUTPUT_BASE=$OUTPUT_BASE
SEEDS=$SEEDS
SPLIT_SEED=$SPLIT_SEED
VAL_PATIENTS=$VAL_PATIENTS
MAX_PATIENTS=$MAX_PATIENTS
TOKENIZER_EPOCHS=$TOKENIZER_EPOCHS
CLASSIFIER_EPOCHS=$CLASSIFIER_EPOCHS
BATCH_SIZE=$BATCH_SIZE
DEVICE=$DEVICE
SKIP_EXISTING=$SKIP_EXISTING
BOOTSTRAP_ITERS=$BOOTSTRAP_ITERS
CONTINUOUS_DIR=$CONTINUOUS_DIR
GOAL_DIR=$GOAL_DIR
MODEL_LEVEL_GATING_CONTRACT_JSON=$MODEL_LEVEL_GATING_CONTRACT_JSON
MODEL_LEVEL_GATING_SCHEMA=$MODEL_LEVEL_GATING_SCHEMA
MODEL_LEVEL_GATING_NEGATIVE_PRIORITY=$MODEL_LEVEL_GATING_NEGATIVE_PRIORITY
MODEL_LEVEL_GATING_NON_EXECUTING=$MODEL_LEVEL_GATING_NON_EXECUTING
TEACHER_DISTILL_REGION_WEIGHT=0.25
TEACHER_DISTILL_CHANNEL_WEIGHT=0.05
TEACHER_DISTILL_TEMPERATURE=2.0
BACKGROUND_NEGATIVE_RATIO=0.75
BACKGROUND_NEGATIVE_SAMPLING_MODE=cluster_balanced
VALIDATION_BACKGROUND_NEGATIVE_DIR=$BACKGROUND_NEGATIVE_DIR
VALIDATION_BACKGROUND_SELECTION_WEIGHT=0.75
SEGMENT_EVENTNESS_HARD_BACKGROUND_MAX_WEIGHT=0.10
SEGMENT_EVENTNESS_CLUSTER_BACKGROUND_MAX_WEIGHT=0.08
CHANNEL_HARD_BACKGROUND_MAX_WEIGHT=0.05
CHANNEL_CLUSTER_BACKGROUND_MAX_WEIGHT=0.03
REGION_HARD_BACKGROUND_MAX_WEIGHT=0.12
REGION_HARD_BACKGROUND_TARGET_WEIGHT=$REGION_HARD_BACKGROUND_TARGET_WEIGHT
REGION_HARD_BACKGROUND_TARGET_REGIONS=$REGION_HARD_BACKGROUND_TARGET_REGIONS
REGION_CLUSTER_BACKGROUND_MAX_WEIGHT=0.10
REGION_BUDGET_NEAR_MISS_ALIAS=$REGION_BUDGET_NEAR_MISS_ALIAS
REGION_BUDGET_NEAR_MISS_TEMPORAL_CLUSTERS=$REGION_BUDGET_NEAR_MISS_TEMPORAL_CLUSTERS
REGION_BUDGET_MAX_TEMPORAL_CLUSTERS=$REGION_BUDGET_MAX_TEMPORAL_CLUSTERS
REGION_BUDGET_REQUIRED_TEMPORAL_REDUCTION=$REGION_BUDGET_REQUIRED_TEMPORAL_REDUCTION
REGION_BUDGET_NEAR_MISS_MISSED_EVENTS=$REGION_BUDGET_NEAR_MISS_MISSED_EVENTS
REGION_BUDGET_MAX_MISSED_EVENTS=$REGION_BUDGET_MAX_MISSED_EVENTS
REGION_BUDGET_NEAR_MISS_MISSED_EXCESS=$REGION_BUDGET_NEAR_MISS_MISSED_EXCESS
REGION_BUDGET_TARGET_SOURCE=$REGION_BUDGET_TARGET_SOURCE
REGION_BUDGET_TARGET_PROMOTION_POLICY=$REGION_BUDGET_TARGET_PROMOTION_POLICY
USE_CONTEXT_DELTA_HEADS=1
USE_CBRAMOD_RESIDUAL_BRANCH=1
USE_CBRAMOD_CRISS_CROSS=0
DISABLE_REGION_BIAS_CALIBRATION=1
SELECTION_PRIMARY_SOURCE=region_onset
SELECTION_MIN_PRIMARY_TOP1=0.80
SELECTION_MIN_PRIMARY_TOP2=0.95
SELECTION_PRIMARY_TOP1_PENALTY_WEIGHT=2.0
SELECTION_PRIMARY_TOP2_PENALTY_WEIGHT=1.5
SELECTION_BACKGROUND_REQUIRES_PRIMARY_TOP1=1
SELECTION_BACKGROUND_REQUIRES_PRIMARY_TOP2=1
CONTINUOUS_THRESHOLD_POLICY=sensitivity_floor
CONTINUOUS_SCORE_SOURCE=onset_context_region
CONTINUOUS_MIN_SENSITIVITY=0.80
CONTINUOUS_TARGET_FPR_PER_HOUR=4.0
CONTINUOUS_MAX_VAL_FILES=1
CONTINUOUS_MAX_TEST_FILES=0
FOLD_VALIDATION_TARGET_SELECTION_MAX_VAL_FILES=0
FOLD_VALIDATION_TARGET_SELECTION_SCAN_SPLIT=val
CONTINUOUS_STEP_SEC=5
CONTINUOUS_EVENT_TIME_POLICY=causal_end
EOF

tmux new-session -d -s "$SESSION_NAME" "
set -euo pipefail
cd '$REPO_ROOT'
LOG='$OUTPUT_BASE/run.log'
exec >>\"\$LOG\" 2>&1

echo \"[\$(date '+%F %T')] queued full43 targeted hard-background CBraTFM-SOZ run\"
echo \"session=$SESSION_NAME\"

IFS=',' read -r -a wait_sessions <<< '$WAIT_SESSIONS'
for session in \"\${wait_sessions[@]}\"; do
  session=\"\${session//[[:space:]]/}\"
  while [[ -n \"\$session\" ]] && tmux list-sessions -F '#S' 2>/dev/null | grep -qx \"\$session\"; do
    echo \"[\$(date '+%F %T')] waiting for tmux session \$session\"
    sleep '$WAIT_SEC'
  done
done

active_training_processes() {
  ps -eo pid=,comm=,args= | awk -v marker='$TRAIN_MARKER' '
    \$2 ~ /^python[0-9.]*$/ &&
    index(\$0, marker) &&
    \$0 !~ /python[0-9.]*[[:space:]]+-m[[:space:]]+py_compile/ {
      print
    }
  '
}

while [[ -n \"\$(active_training_processes)\" ]]; do
  echo \"[\$(date '+%F %T')] waiting for active training processes matching $TRAIN_MARKER\"
  active_training_processes || true
  sleep '$WAIT_SEC'
done

env \
  MANIFEST='$MANIFEST' \
  PREPROCESSED_DIR='$PREPROCESSED_DIR' \
  BACKGROUND_NEGATIVE_DIR='$BACKGROUND_NEGATIVE_DIR' \
  TEACHER_CACHE_DIR='$TEACHER_CACHE_DIR' \
  TEACHER_ROOTS='$TEACHER_ROOTS' \
  OUTPUT_BASE='$OUTPUT_BASE' \
  SEEDS='$SEEDS' \
  SPLIT_SEED='$SPLIT_SEED' \
  VAL_PATIENTS='$VAL_PATIENTS' \
  MAX_PATIENTS='$MAX_PATIENTS' \
  TOKENIZER_EPOCHS='$TOKENIZER_EPOCHS' \
  CLASSIFIER_EPOCHS='$CLASSIFIER_EPOCHS' \
  BATCH_SIZE='$BATCH_SIZE' \
  DEVICE='$DEVICE' \
  SKIP_EXISTING='$SKIP_EXISTING' \
  RUN_EVAL=1 \
  BOOTSTRAP_ITERS='$BOOTSTRAP_ITERS' \
  MODEL_LEVEL_GATING_CONTRACT_JSON='$MODEL_LEVEL_GATING_CONTRACT_JSON' \
  MODEL_LEVEL_GATING_SCHEMA='$MODEL_LEVEL_GATING_SCHEMA' \
  MODEL_LEVEL_GATING_NEGATIVE_PRIORITY='$MODEL_LEVEL_GATING_NEGATIVE_PRIORITY' \
  MODEL_LEVEL_GATING_NON_EXECUTING='$MODEL_LEVEL_GATING_NON_EXECUTING' \
  TEACHER_DISTILL_REGION_WEIGHT=0.25 \
  TEACHER_DISTILL_CHANNEL_WEIGHT=0.05 \
  TEACHER_DISTILL_TEMPERATURE=2.0 \
  BACKGROUND_NEGATIVE_RATIO=0.75 \
  BACKGROUND_NEGATIVE_SAMPLING_MODE=cluster_balanced \
  VALIDATION_BACKGROUND_NEGATIVE_DIR='$BACKGROUND_NEGATIVE_DIR' \
  VALIDATION_BACKGROUND_SELECTION_WEIGHT=0.75 \
  SEGMENT_EVENTNESS_HARD_BACKGROUND_MAX_WEIGHT=0.10 \
  SEGMENT_EVENTNESS_CLUSTER_BACKGROUND_MAX_WEIGHT=0.08 \
  CHANNEL_HARD_BACKGROUND_MAX_WEIGHT=0.05 \
  CHANNEL_CLUSTER_BACKGROUND_MAX_WEIGHT=0.03 \
  REGION_HARD_BACKGROUND_MAX_WEIGHT=0.12 \
  REGION_HARD_BACKGROUND_TARGET_WEIGHT='$REGION_HARD_BACKGROUND_TARGET_WEIGHT' \
  REGION_HARD_BACKGROUND_TARGET_REGIONS='$REGION_HARD_BACKGROUND_TARGET_REGIONS' \
  REGION_CLUSTER_BACKGROUND_MAX_WEIGHT=0.10 \
  USE_CONTEXT_DELTA_HEADS=1 \
  USE_CBRAMOD_RESIDUAL_BRANCH=1 \
  USE_CBRAMOD_CRISS_CROSS=0 \
  DISABLE_REGION_BIAS_CALIBRATION=1 \
  SELECTION_PRIMARY_SOURCE=region_onset \
  SELECTION_MIN_PRIMARY_TOP1=0.80 \
  SELECTION_MIN_PRIMARY_TOP2=0.95 \
  SELECTION_PRIMARY_TOP1_PENALTY_WEIGHT=2.0 \
  SELECTION_PRIMARY_TOP2_PENALTY_WEIGHT=1.5 \
  SELECTION_BACKGROUND_REQUIRES_PRIMARY_TOP1=1 \
  SELECTION_BACKGROUND_REQUIRES_PRIMARY_TOP2=1 \
  bash scripts/run_cbratfm_soz_topconf_distill_full43.sh

IFS=',' read -r -a seed_items <<< '$SEEDS'
roots=()
for seed in \"\${seed_items[@]}\"; do
  seed=\"\${seed//[[:space:]]/}\"
  [[ -n \"\$seed\" ]] && roots+=(\"$OUTPUT_BASE\"_seed\"\$seed\")
done
roots_csv=\"\$(IFS=,; echo \"\${roots[*]}\")\"

python3 code/tfm_soz/audit_private_soz_protocol.py \
  --manifest '$MANIFEST' \
  --preprocessed-dir '$PREPROCESSED_DIR' \
  --lopo-roots \"\$roots_csv\" \
  --output '$GOAL_DIR/cbratfm_targeted_hardbg_full43_protocol_audit.json' \
  --quiet

foldval_target_stem='$CONTINUOUS_DIR/cbratfm_targeted_hardbg_full43_foldval_target_selection_ensemble_onsetcontext_sens80_causal_step5_fullfile'
foldval_target_json=\"\${foldval_target_stem}.json\"
foldval_region_screen_json='$GOAL_DIR/targeted_full43_fold_validation_region_screen.json'
foldval_selection_route_json='$GOAL_DIR/targeted_full43_fold_validation_target_selection_route.json'
foldval_selection_audit_json='$GOAL_DIR/targeted_full43_fold_validation_target_selection_audit.json'

if [[ ! -s \"\$foldval_target_json\" ]]; then
  python3 code/tfm_soz/evaluate_lopo_continuous_fullfile_ensemble.py \
    --roots \"\$roots_csv\" \
    --preprocessed-dir '$PREPROCESSED_DIR' \
    --output \"\$foldval_target_json\" \
    --threshold-policy sensitivity_floor \
    --score-source onset_context_region \
    --min-sensitivity 0.80 \
    --max-val-files 0 \
    --max-test-files 0 \
    --scan-split val \
    --step-sec 5 \
    --event-time-policy causal_end \
    --device '$DEVICE' \
    --quiet
fi

python3 code/tfm_soz/screen_rows119_fold_validation_target_regions.py \
  --search-root \"\${foldval_target_stem}_val_candidates.csv\" \
  --output-json \"\$foldval_region_screen_json\"

python3 code/tfm_soz/build_rows119_fold_validation_target_selection_route.py \
  --search-root '$CONTINUOUS_DIR' \
  --region-screen-json \"\$foldval_region_screen_json\" \
  --output-json \"\$foldval_selection_route_json\" \
  --write-selection

python3 code/tfm_soz/audit_rows119_fold_validation_target_selection.py \
  --output-json \"\$foldval_selection_audit_json\"

python3 code/tfm_soz/make_rows119_fold_validation_target_selection_coverage_report.py \
  --region-screen-json \"\$foldval_region_screen_json\" \
  --route-json \"\$foldval_selection_route_json\" \
  --output-json '$GOAL_DIR/fold_validation_target_selection_coverage_report.json'

base_stem='$CONTINUOUS_DIR/cbratfm_targeted_hardbg_full43_alltest_file1val_ensemble_onsetcontext_sens80_causal_step5_fullfile'
base_json=\"\${base_stem}.json\"
post_stem='$CONTINUOUS_DIR/cbratfm_targeted_hardbg_full43_alltest_file1val_contextclusterrank_fpr4_sens80_causal_step5'
post_json=\"\${post_stem}.json\"
bootstrap_json=\"\${post_stem}_bootstrap.json\"
error_prefix='$GOAL_DIR/cbratfm_targeted_hardbg_full43_continuous_error_hotspots'
bundle_json='$GOAL_DIR/cbratfm_targeted_hardbg_full43_continuous_candidate_bundle.json'

if [[ ! -s \"\$base_json\" ]]; then
  python3 code/tfm_soz/evaluate_lopo_continuous_fullfile_ensemble.py \
    --roots \"\$roots_csv\" \
    --preprocessed-dir '$PREPROCESSED_DIR' \
    --output \"\$base_json\" \
    --threshold-policy sensitivity_floor \
    --score-source onset_context_region \
    --min-sensitivity 0.80 \
    --max-val-files 1 \
    --max-test-files 0 \
    --step-sec 5 \
    --event-time-policy causal_end \
    --device '$DEVICE' \
    --quiet
fi

if [[ ! -s \"\$post_json\" ]]; then
  python3 code/tfm_soz/evaluate_lopo_continuous_fullfile_context_cluster_rank.py \
    --source-json \"\$base_json\" \
    --output \"\$post_json\" \
    --min-sensitivity 0.80 \
    --target-fpr-per-hour 4.0 \
    --quiet
fi

if [[ ! -s \"\$bootstrap_json\" ]]; then
  python3 code/tfm_soz/bootstrap_continuous_fullfile.py \
    --summary-json \"\$post_json\" \
    --event-csv \"\${post_stem}.csv\" \
    --candidate-csv \"\${post_stem}_candidates.csv\" \
    --output \"\$bootstrap_json\" \
    --bootstrap-iters '$BOOTSTRAP_ITERS'
fi

python3 code/tfm_soz/analyze_continuous_fullfile_errors.py \
  --events-csv \"\${post_stem}.csv\" \
  --candidates-csv \"\${post_stem}_candidates.csv\" \
  --output-prefix \"\$error_prefix\"

python3 code/tfm_soz/audit_rows119_continuous_candidate_bundle.py \
  --candidate-fixed-summary-json '$OUTPUT_BASE/lopo_test_global_fusion_equal.json' \
  --candidate-continuous-json \"\$post_json\" \
  --candidate-continuous-bootstrap-json \"\$bootstrap_json\" \
  --candidate-false-alarm-clusters-csv \"\${error_prefix}_false_alarm_clusters.csv\" \
  --candidate-missed-csv \"\${error_prefix}_missed.csv\" \
  --output-json \"\$bundle_json\"

python3 code/tfm_soz/summarize_rows119_continuous_frontier.py \
  --output '$GOAL_DIR/continuous_frontier_summary.json'
python3 code/tfm_soz/make_rows119_goal_readiness_bundle.py

echo \"[\$(date '+%F %T')] full43 targeted hard-background queue complete\"
"

echo "Queued full43 targeted hard-background run in tmux session: $SESSION_NAME"
echo "Output: $OUTPUT_BASE"
