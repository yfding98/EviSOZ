"""Evidence-grounded, deterministic report planning for EviSOZ."""

from .predicted_report_plan import (
    PREDICTED_REPORT_PLAN_SCHEMA_VERSION,
    build_predicted_report_plan,
    validate_predicted_report_plan,
)
from .bound_shadow_report import (
    build_bound_shadow_report_plan,
    selected_knowledge_card_ids,
)
from .qwen_structured_input import (
    QWEN_EVIDENCE_TOKEN_COUNT,
    QWEN_HIDDEN_SIZE,
    QWEN_STRUCTURED_INPUT_MODE,
    QWEN_STRUCTURED_INPUT_SCHEMA_VERSION,
    QWEN_STRUCTURED_INPUT_STATUS,
    build_qwen_structured_input,
    validate_qwen_structured_input,
)
from .qwen_patient_input import (
    QWEN_PATIENT_INPUT_MODE,
    QWEN_PATIENT_INPUT_SCHEMA_VERSION,
    QWEN_PATIENT_INPUT_STATUS,
    build_qwen_patient_input,
    validate_qwen_patient_input,
)

__all__ = [
    "PREDICTED_REPORT_PLAN_SCHEMA_VERSION",
    "build_predicted_report_plan",
    "validate_predicted_report_plan",
    "build_bound_shadow_report_plan",
    "selected_knowledge_card_ids",
    "QWEN_EVIDENCE_TOKEN_COUNT",
    "QWEN_HIDDEN_SIZE",
    "QWEN_STRUCTURED_INPUT_MODE",
    "QWEN_STRUCTURED_INPUT_SCHEMA_VERSION",
    "QWEN_STRUCTURED_INPUT_STATUS",
    "build_qwen_structured_input",
    "validate_qwen_structured_input",
    "QWEN_PATIENT_INPUT_MODE",
    "QWEN_PATIENT_INPUT_SCHEMA_VERSION",
    "QWEN_PATIENT_INPUT_STATUS",
    "build_qwen_patient_input",
    "validate_qwen_patient_input",
]
