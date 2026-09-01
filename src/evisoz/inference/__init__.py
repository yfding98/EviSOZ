"""Loader-backed and guarded EviSOZ inference entry points."""

from .shadow import (
    ShadowInferenceResult,
    run_bound_evidence_shadow_inference,
)
from .patient import (
    aggregate_bound_shadow_predictions,
    build_bound_patient_qwen_shadow_inputs,
)
from .localization import run_authorized_residual_forward

__all__ = [
    "ShadowInferenceResult",
    "run_bound_evidence_shadow_inference",
    "aggregate_bound_shadow_predictions",
    "build_bound_patient_qwen_shadow_inputs",
    "run_authorized_residual_forward",
]
