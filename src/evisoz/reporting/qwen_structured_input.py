"""Validated, candidate-only input packet for the future Qwen route.

The packet is deliberately a *shadow* interface.  It carries a validated
predicted report plan and the already-selected ``knowledge/eeg`` card IDs,
but never card text, raw EEG, physician text, or a permission to generate a
clinical report.  A future Qwen adapter may consume this packet only after
the aggregate Stage-0 gate authorizes the corresponding operation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.forge.findings_claims_reports import (
    KNOWLEDGE_SELECTION_SCHEMA_VERSION,
    validate_knowledge_selection_receipt,
)

from .predicted_report_plan import (
    PREDICTED_REPORT_PLAN_SCHEMA_VERSION,
    validate_predicted_report_plan,
)


QWEN_STRUCTURED_INPUT_SCHEMA_VERSION = "evisoz_qwen_structured_input_v1"
QWEN_STRUCTURED_INPUT_STATUS = "shadow_input_no_generation"
QWEN_STRUCTURED_INPUT_MODE = "candidate_lexicalization_shadow"
QWEN_EVIDENCE_TOKEN_COUNT = 32
QWEN_HIDDEN_SIZE = 5120
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-QINPUT-"

_SYSTEM_POLICY = (
    "Only lexicalize the supplied model-candidate plan. Do not add patient "
    "facts, channels, times, morphology, localization, certainty, or "
    "clinical conclusions. Preserve candidate-only status and require "
    "physician review."
)

_PERMISSIONS = {
    "qwen_generation_allowed": False,
    "qwen_may_add_patient_facts": False,
    "qwen_may_change_localization": False,
    "qwen_may_change_uncertainty": False,
    "knowledge_may_create_patient_facts": False,
    "raw_eeg_included": False,
    "physician_text_included": False,
    "requires_physician_review": True,
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["input_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _selection_card_ids(selection: Mapping[str, object]) -> list[str]:
    cards = selection["selected_cards"]
    result = sorted({str(row["card_id"]) for row in cards})
    if not result:
        raise ValueError("Qwen input requires a non-empty knowledge selection")
    return result


def _plan_card_ids(plan: Mapping[str, object]) -> list[str]:
    result: set[str] = set()
    for section in plan["sections"]:
        result.update(str(item) for item in section["knowledge_card_ids"])
    return sorted(result)


def build_qwen_structured_input(
    *,
    report_plan: Mapping[str, object],
    knowledge_selection: Mapping[str, object],
) -> dict[str, Any]:
    """Build a no-generation Qwen input from validated upstream artifacts.

    ``knowledge_selection`` is the only permitted bridge into ``knowledge/eeg``.
    The packet carries card identity and bundle provenance, never card text.
    The report plan remains candidate-only and is copied verbatim so its
    content-addressed reference can be replayed by a future consumer.
    """

    plan = validate_predicted_report_plan(dict(report_plan))
    selection = validate_knowledge_selection_receipt(dict(knowledge_selection))
    selected_ids = _selection_card_ids(selection)
    plan_ids = _plan_card_ids(plan)
    if plan_ids != selected_ids:
        raise ValueError("Qwen input knowledge cards do not replay the selection receipt")

    plan_ref = build_json_artifact_ref(
        plan,
        artifact_kind="evisoz_predicted_report_plan",
        payload_schema_version=PREDICTED_REPORT_PLAN_SCHEMA_VERSION,
    )
    selection_ref = build_json_artifact_ref(
        selection,
        artifact_kind="evisoz_knowledge_selection_receipt",
        payload_schema_version=KNOWLEDGE_SELECTION_SCHEMA_VERSION,
    )
    body: dict[str, Any] = {
        "schema_version": QWEN_STRUCTURED_INPUT_SCHEMA_VERSION,
        "input_id": _HASH_PLACEHOLDER,
        "event_id": plan["event_id"],
        "stage0_status": plan["stage0_status"],
        "status": QWEN_STRUCTURED_INPUT_STATUS,
        "mode": QWEN_STRUCTURED_INPUT_MODE,
        "source_plan_ref": plan_ref,
        "knowledge_selection_ref": selection_ref,
        "prompt": {
            "system_policy": _SYSTEM_POLICY,
            "task": "lexicalize_candidate_only",
            "knowledge_context": {
                "knowledge_version": selection["knowledge_version"],
                "bundle_sha256": selection["knowledge_manifest_sha256"],
                "card_ids": selected_ids,
                "card_text_included": False,
                "patient_fact_creation_allowed": False,
            },
            "report_plan": plan,
        },
        "evidence_token_contract": {
            "source": "runtime_evidence_token_resampler",
            "token_count": QWEN_EVIDENCE_TOKEN_COUNT,
            "hidden_size": QWEN_HIDDEN_SIZE,
            "embeddings_included": False,
            "requires_runtime_resampler": True,
        },
        "permissions": deepcopy(_PERMISSIONS),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["input_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_qwen_structured_input(
        body,
        trusted_plan=plan,
        trusted_selection=selection,
    )


def validate_qwen_structured_input(
    value: object,
    *,
    trusted_plan: Mapping[str, object] | None = None,
    trusted_selection: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate a Qwen packet and optionally replay its source payloads."""

    required = {
        "schema_version",
        "input_id",
        "event_id",
        "stage0_status",
        "status",
        "mode",
        "source_plan_ref",
        "knowledge_selection_ref",
        "prompt",
        "evidence_token_contract",
        "permissions",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("Qwen structured input fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != QWEN_STRUCTURED_INPUT_SCHEMA_VERSION:
        raise ValueError("Qwen structured input schema drifted")
    if data["status"] != QWEN_STRUCTURED_INPUT_STATUS or data["mode"] != QWEN_STRUCTURED_INPUT_MODE:
        raise ValueError("Qwen structured input remains shadow-only")
    if not isinstance(data["event_id"], str) or not data["event_id"]:
        raise ValueError("Qwen structured input event ID is invalid")
    if data["stage0_status"] not in {"GO", "NO_GO"}:
        raise ValueError("Qwen structured input Stage-0 status drifted")

    plan_ref = validate_artifact_ref(data["source_plan_ref"])
    selection_ref = validate_artifact_ref(data["knowledge_selection_ref"])
    if plan_ref["artifact_kind"] != "evisoz_predicted_report_plan":
        raise ValueError("Qwen structured input plan reference kind drifted")
    if plan_ref["payload_schema_version"] != PREDICTED_REPORT_PLAN_SCHEMA_VERSION:
        raise ValueError("Qwen structured input plan reference version drifted")
    if selection_ref["artifact_kind"] != "evisoz_knowledge_selection_receipt":
        raise ValueError("Qwen structured input knowledge reference kind drifted")
    if selection_ref["payload_schema_version"] != KNOWLEDGE_SELECTION_SCHEMA_VERSION:
        raise ValueError("Qwen structured input knowledge reference version drifted")

    prompt = data["prompt"]
    if type(prompt) is not dict or set(prompt) != {
        "system_policy", "task", "knowledge_context", "report_plan"
    }:
        raise ValueError("Qwen structured input prompt fields drifted")
    if prompt["system_policy"] != _SYSTEM_POLICY or prompt["task"] != "lexicalize_candidate_only":
        raise ValueError("Qwen structured input prompt policy drifted")
    context = prompt["knowledge_context"]
    if type(context) is not dict or set(context) != {
        "knowledge_version", "bundle_sha256", "card_ids", "card_text_included",
        "patient_fact_creation_allowed",
    }:
        raise ValueError("Qwen structured input knowledge context drifted")
    if (
        not isinstance(context["knowledge_version"], str)
        or not isinstance(context["bundle_sha256"], str)
        or len(context["bundle_sha256"]) != 64
        or context["card_text_included"] is not False
        or context["patient_fact_creation_allowed"] is not False
    ):
        raise ValueError("Qwen structured input knowledge context is unsafe")
    if not isinstance(context["card_ids"], list) or any(
        not isinstance(item, str) or not item for item in context["card_ids"]
    ) or context["card_ids"] != sorted(set(context["card_ids"])):
        raise ValueError("Qwen structured input card roster drifted")

    plan = validate_predicted_report_plan(prompt["report_plan"])
    selection = None
    if trusted_plan is not None:
        selection_plan = validate_predicted_report_plan(dict(trusted_plan))
        verify_artifact_content(plan_ref, selection_plan)
        if plan != selection_plan:
            raise ValueError("Qwen structured input plan replay drifted")
    if trusted_selection is not None:
        selection = validate_knowledge_selection_receipt(dict(trusted_selection))
        verify_artifact_content(selection_ref, selection)
    if plan["event_id"] != data["event_id"] or plan["stage0_status"] != data["stage0_status"]:
        raise ValueError("Qwen structured input plan linkage drifted")
    plan_ids = _plan_card_ids(plan)
    context_ids = list(context["card_ids"])
    if plan_ids != context_ids:
        raise ValueError("Qwen structured input plan/card linkage drifted")
    if selection is not None:
        if _selection_card_ids(selection) != context_ids:
            raise ValueError("Qwen structured input selection/card linkage drifted")
        if context["knowledge_version"] != selection["knowledge_version"]:
            raise ValueError("Qwen structured input knowledge version drifted")
        if context["bundle_sha256"] != selection["knowledge_manifest_sha256"]:
            raise ValueError("Qwen structured input knowledge bundle drifted")

    token_contract = data["evidence_token_contract"]
    if type(token_contract) is not dict or set(token_contract) != {
        "source", "token_count", "hidden_size", "embeddings_included",
        "requires_runtime_resampler",
    }:
        raise ValueError("Qwen structured input token contract drifted")
    if token_contract != {
        "source": "runtime_evidence_token_resampler",
        "token_count": QWEN_EVIDENCE_TOKEN_COUNT,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "embeddings_included": False,
        "requires_runtime_resampler": True,
    }:
        raise ValueError("Qwen structured input token contract is unsafe")
    if data["permissions"] != _PERMISSIONS:
        raise ValueError("Qwen structured input permissions drifted")

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["input_id"] != expected_id:
        raise ValueError("Qwen structured input ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("Qwen structured input receipt drifted")
    return data


__all__ = [
    "QWEN_EVIDENCE_TOKEN_COUNT",
    "QWEN_HIDDEN_SIZE",
    "QWEN_STRUCTURED_INPUT_MODE",
    "QWEN_STRUCTURED_INPUT_SCHEMA_VERSION",
    "QWEN_STRUCTURED_INPUT_STATUS",
    "build_qwen_structured_input",
    "validate_qwen_structured_input",
]
