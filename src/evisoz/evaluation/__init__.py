"""Leakage-safe EviSOZ evaluation utilities."""

from .metrics import (
    brier_score_multiclass,
    correction_corruption_rates,
    expected_calibration_error,
    mean_reciprocal_rank,
    onset_spread_order_accuracy,
    risk_coverage_curve,
    top_k_candidate_hit,
    unsupported_claim_rate,
)
from .bound_evidence_eval import (
    SHADOW_EVALUATION_SCHEMA_VERSION,
    evaluate_bound_evidence_shadow_predictions,
    validate_bound_evidence_shadow_evaluation,
)
from .patient_qwen_shadow_eval import (
    PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION,
    PATIENT_QWEN_SHADOW_EVALUATION_STATUS,
    evaluate_bound_patient_qwen_shadow_inputs,
    validate_patient_qwen_shadow_evaluation,
)
from .clinical_localization import (
    NODE_LATERALITY,
    NODE_REGION,
    aggregate_event_probabilities_by_patient,
    evaluate_localization_predictions,
    evaluate_patient_localization_predictions,
    extract_released_node_target,
)
from .report_factuality import (
    REPORT_CLAIM_TYPES,
    evaluate_evisoz_report_factuality,
)

__all__ = [
    "brier_score_multiclass",
    "correction_corruption_rates",
    "expected_calibration_error",
    "mean_reciprocal_rank",
    "onset_spread_order_accuracy",
    "risk_coverage_curve",
    "top_k_candidate_hit",
    "unsupported_claim_rate",
    "SHADOW_EVALUATION_SCHEMA_VERSION",
    "evaluate_bound_evidence_shadow_predictions",
    "validate_bound_evidence_shadow_evaluation",
    "PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION",
    "PATIENT_QWEN_SHADOW_EVALUATION_STATUS",
    "evaluate_bound_patient_qwen_shadow_inputs",
    "validate_patient_qwen_shadow_evaluation",
    "NODE_LATERALITY",
    "NODE_REGION",
    "aggregate_event_probabilities_by_patient",
    "evaluate_localization_predictions",
    "evaluate_patient_localization_predictions",
    "extract_released_node_target",
    "REPORT_CLAIM_TYPES",
    "evaluate_evisoz_report_factuality",
]
