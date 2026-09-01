#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current strict VEPiSet IED / SOZ-like proxy main line.
#
# Default behavior is resume-friendly: each expensive step is skipped when its
# sentinel artifact already exists. Set FORCE=1 to rerun every step in-place.
#
# This script uses only validation labels for checkpoint/calibration choices.
# The final test set is audited by scripts/verify_vepiset_strict_main.py.

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
VEP_ROOT="${VEP_ROOT:-/mnt/hd1/dyf/dataset/vepiset-dataset/opensource-dataset}"
FORCE="${FORCE:-0}"

BASE_RUN="${BASE_RUN:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20}"
BASE_EXPORT="${BASE_EXPORT:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20_region_export}"
BASE_REGION="${BASE_REGION:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_macroselect_regionfusion_macro_valacc85}"
BASE_SMOOTH="${BASE_SMOOTH:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_macroselect_regionfusion_smooth_macro_valacc85}"

REGIONCONTRAST_RUN="${REGIONCONTRAST_RUN:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_regioncontrast_noamp20}"
REGIONCONTRAST_EXPORT="${REGIONCONTRAST_EXPORT:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_regioncontrast_noamp20_region_export}"
REGIONCONTRAST_REGION="${REGIONCONTRAST_REGION:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_regioncontrast_regionfusion_balanced_valacc85}"

ENSEMBLE_RUN="${ENSEMBLE_RUN:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_ensemble_smooth_regioncontrast_macro_valacc85}"
MAIN_RUN="${MAIN_RUN:-outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87}"
HIGH_ACCURACY_AUDIT_RUN="${HIGH_ACCURACY_AUDIT_RUN:-outputs/vepiset_ied_v2_full6_patientclasssplit_main_singlebias_accuracy_valmacro43}"
POSITIVE_SPATIAL_BIAS_AUDIT_RUN="${POSITIVE_SPATIAL_BIAS_AUDIT_RUN:-outputs/vepiset_ied_v2_full6_patientclasssplit_main_positive_spatial_bias_weighted_tiny}"

run_if_needed() {
  local name="$1"
  local sentinel="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -e "${sentinel}" ]]; then
    printf '[skip] %s: %s exists\n' "${name}" "${sentinel}"
    return 0
  fi
  printf '[run] %s\n' "${name}"
  "$@"
}

train_common_args=(
  --vepiset-root "${VEP_ROOT}"
  --task multiclass
  --montage monopolar19
  --brain-network-features aec,wpli
  --epochs 20
  --early-stop-patience 8
  --min-epochs 8
  --batch-size 64
  --workers 4
  --lr 0.0005
  --sampler class
  --class-weight-cap 10
  --sample-weight-mode none
  --selection-metric macro_f1
  --class-focal-gamma 0.5
  --label-smoothing 0.02
  --logit-adjust-tau 0.25
  --eval-logit-adjust-tau 0.0
  --w-binary 0.2
  --w-region 0.05
  --w-moe 0.01
  --seed 2026
  --split-seed 2026
  --split-strategy patient_class_balanced
  --split-search-trials 8192
  --train-max-non-ied-samples 3000
  --embed-dim 96
  --transformer-layers 1
  --transformer-heads 4
  --out-chans 8
  --tf-heads 4
  --timefilter-blocks 1
  --brain-tf-blocks 1
  --brain-tf-hidden 32
  --gru-hidden 64
  --gru-layers 1
  --gcn-hidden 32
)

run_if_needed "base v2 training" "${BASE_RUN}/best_model.pt" \
  "${PYTHON_BIN}" code/models/train_vepiset_ied_v2.py \
  "${train_common_args[@]}" \
  --output-dir "${BASE_RUN}"

run_if_needed "base region-probability export" "${BASE_EXPORT}/export_manifest.json" \
  "${PYTHON_BIN}" code/models/export_vepiset_ied_predictions.py \
  --run-dir "${BASE_RUN}" \
  --output-dir "${BASE_EXPORT}" \
  --splits val,test \
  --batch-size 128 \
  --workers 4

run_if_needed "base validation-only region fusion" "${BASE_REGION}/region_fusion_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_region_fusion.py \
  --prediction-dir "${BASE_EXPORT}" \
  --output-dir "${BASE_REGION}" \
  --selector macro_f1 \
  --min-val-accuracy 0.85

run_if_needed "base validation-only patient/window smoothing" "${BASE_SMOOTH}/smoothed_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_patient_smoothing.py \
  --run-dir "${BASE_REGION}" \
  --output-dir "${BASE_SMOOTH}" \
  --selector macro_f1 \
  --min-val-accuracy 0.85 \
  --non-ied-biases=-0.5,-0.25,0,0.25,0.5

run_if_needed "region-contrast v2 training" "${REGIONCONTRAST_RUN}/best_model.pt" \
  "${PYTHON_BIN}" code/models/train_vepiset_ied_v2.py \
  "${train_common_args[@]}" \
  --morph-region-contrast \
  --output-dir "${REGIONCONTRAST_RUN}"

run_if_needed "region-contrast region-probability export" "${REGIONCONTRAST_EXPORT}/export_manifest.json" \
  "${PYTHON_BIN}" code/models/export_vepiset_ied_predictions.py \
  --run-dir "${REGIONCONTRAST_RUN}" \
  --output-dir "${REGIONCONTRAST_EXPORT}" \
  --splits val,test \
  --batch-size 128 \
  --workers 4

run_if_needed "region-contrast validation-only region fusion" "${REGIONCONTRAST_REGION}/region_fusion_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_region_fusion.py \
  --prediction-dir "${REGIONCONTRAST_EXPORT}" \
  --output-dir "${REGIONCONTRAST_REGION}" \
  --selector balanced_accuracy \
  --min-val-accuracy 0.85

run_if_needed "validation-only smoothing/region-contrast ensemble" "${ENSEMBLE_RUN}/calibrated_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_ensemble.py \
  --run-dir-a "${BASE_SMOOTH}" \
  --run-dir-b "${REGIONCONTRAST_REGION}" \
  --output-dir "${ENSEMBLE_RUN}" \
  --selector macro_f1 \
  --min-val-accuracy 0.85

run_if_needed "conservative validation-only patient prior" "${MAIN_RUN}/patient_prior_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_patient_prior.py \
  --run-dir "${ENSEMBLE_RUN}" \
  --output-dir "${MAIN_RUN}" \
  --selector macro_f1 \
  --modes spatial \
  --top-fracs 0.05,0.1,0.2 \
  --evidences prob,spatial \
  --spatial-alphas 0,0.05,0.1,0.15 \
  --mass-alphas 0 \
  --non-ied-biases=-0.1,-0.05,0,0.05,0.1 \
  --min-val-accuracy 0.87 \
  --min-val-weighted-f1 0.88

run_if_needed "high-accuracy single-bias audit" "${HIGH_ACCURACY_AUDIT_RUN}/calibrated_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_single_model.py \
  --run-dir "${MAIN_RUN}" \
  --output-dir "${HIGH_ACCURACY_AUDIT_RUN}" \
  --selector accuracy \
  --bias-min -1.0 \
  --bias-max 2.0 \
  --bias-steps 601 \
  --min-val-accuracy 0.87 \
  --min-val-weighted-f1 0.87 \
  --min-val-macro-f1 0.43

run_if_needed "high-accuracy positive localization audit" "${HIGH_ACCURACY_AUDIT_RUN}/positive_localization_metrics.json" \
  "${PYTHON_BIN}" scripts/evaluate_vepiset_positive_localization.py \
  --run-dir "${HIGH_ACCURACY_AUDIT_RUN}" \
  --output-json "${HIGH_ACCURACY_AUDIT_RUN}/positive_localization_metrics.json"

run_if_needed "high-accuracy patient proxy audit" "${HIGH_ACCURACY_AUDIT_RUN}/patient_proxy_metrics_mean_positive_selector.json" \
  "${PYTHON_BIN}" scripts/evaluate_vepiset_patient_proxy.py \
  --run-dir "${HIGH_ACCURACY_AUDIT_RUN}" \
  --output-json "${HIGH_ACCURACY_AUDIT_RUN}/patient_proxy_metrics_mean_positive_selector.json" \
  --aggregation mean \
  --selector positive_hit_accuracy

run_if_needed "positive-spatial tiny bias audit" "${POSITIVE_SPATIAL_BIAS_AUDIT_RUN}/positive_spatial_bias_metrics.json" \
  "${PYTHON_BIN}" code/models/calibrate_vepiset_positive_spatial_bias.py \
  --run-dir "${MAIN_RUN}" \
  --output-dir "${POSITIVE_SPATIAL_BIAS_AUDIT_RUN}" \
  --selector weighted_f1 \
  --selection-aggregation loo_mean \
  --min-val-accuracy 0.70 \
  --min-val-macro-f1 0.70 \
  --max-abs-bias 0.1 \
  --steps 0.05,0.025,0.01 \
  --passes-per-step 8

run_if_needed "positive-spatial tiny bias localization audit" "${POSITIVE_SPATIAL_BIAS_AUDIT_RUN}/positive_localization_metrics.json" \
  "${PYTHON_BIN}" scripts/evaluate_vepiset_positive_localization.py \
  --run-dir "${POSITIVE_SPATIAL_BIAS_AUDIT_RUN}" \
  --output-json "${POSITIVE_SPATIAL_BIAS_AUDIT_RUN}/positive_localization_metrics.json"

"${PYTHON_BIN}" scripts/verify_vepiset_strict_main.py
