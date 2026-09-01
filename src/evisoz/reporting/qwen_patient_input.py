"""Patient-level candidate-only Qwen input.

This is the patient-level sibling of :mod:`qwen_structured_input`.  It binds
the signal-derived candidate claim graph, deterministic canonical shadow
report, and the same ``knowledge/eeg`` selection receipt.  No physician text,
raw EEG, or clinical SOZ conclusion is admitted to this packet.
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
    CANONICAL_REPORT_SCHEMA_VERSION,
    KNOWLEDGE_SELECTION_SCHEMA_VERSION,
    SIGNAL_GRAPH_SCHEMA_VERSION,
    validate_canonical_report,
    validate_knowledge_selection_receipt,
    validate_signal_candidate_claim_graph,
)

from .qwen_structured_input import (
    QWEN_EVIDENCE_TOKEN_COUNT,
    QWEN_HIDDEN_SIZE,
    _PERMISSIONS,
    _SYSTEM_POLICY,
)


QWEN_PATIENT_INPUT_SCHEMA_VERSION = "evisoz_qwen_patient_input_v1"
QWEN_PATIENT_INPUT_STATUS = "shadow_input_no_generation"
QWEN_PATIENT_INPUT_MODE = "patient_candidate_lexicalization_shadow"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-QPATIENT-"

_PATIENT_PERMISSIONS = {
    **_PERMISSIONS,
    "qwen_may_change_patient_aggregation": False,
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
    cards = sorted({str(row["card_id"]) for row in selection["selected_cards"]})
    if not cards:
        raise ValueError("patient Qwen input requires a non-empty knowledge selection")
    return cards


def _report_card_ids(report: Mapping[str, object]) -> list[str]:
    cards: set[str] = set()
    for section in report["sections"]:
        cards.update(str(item) for item in section["knowledge_card_ids"])
    return sorted(cards)


def build_qwen_patient_input(
    *,
    signal_graph: Mapping[str, object],
    canonical_report: Mapping[str, object],
    knowledge_selection: Mapping[str, object],
) -> dict[str, Any]:
    """Build a patient-level no-generation packet from bound shadow artifacts."""

    graph = validate_signal_candidate_claim_graph(dict(signal_graph))
    selection = validate_knowledge_selection_receipt(dict(knowledge_selection))
    report = validate_canonical_report(
        dict(canonical_report),
        trusted_graph=graph,
        trusted_selection=selection,
    )
    if report["linkage_group_id"] != graph["linkage_group_id"]:
        raise ValueError("patient Qwen report and claim graph linkage drifted")
    selected_ids = _selection_card_ids(selection)
    if _report_card_ids(report) != selected_ids:
        raise ValueError("patient Qwen report cards do not replay the selection receipt")

    graph_ref = build_json_artifact_ref(
        graph,
        artifact_kind="evisoz_signal_candidate_claim_graph",
        payload_schema_version=SIGNAL_GRAPH_SCHEMA_VERSION,
    )
    report_ref = build_json_artifact_ref(
        report,
        artifact_kind="evisoz_canonical_report",
        payload_schema_version=CANONICAL_REPORT_SCHEMA_VERSION,
    )
    selection_ref = build_json_artifact_ref(
        selection,
        artifact_kind="evisoz_knowledge_selection_receipt",
        payload_schema_version=KNOWLEDGE_SELECTION_SCHEMA_VERSION,
    )
    body: dict[str, Any] = {
        "schema_version": QWEN_PATIENT_INPUT_SCHEMA_VERSION,
        "input_id": _HASH_PLACEHOLDER,
        "linkage_group_id": graph["linkage_group_id"],
        "stage0_status": "NO_GO",
        "status": QWEN_PATIENT_INPUT_STATUS,
        "mode": QWEN_PATIENT_INPUT_MODE,
        "source_graph_ref": graph_ref,
        "source_report_ref": report_ref,
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
            "claim_graph": graph,
            "canonical_report": report,
        },
        "evidence_token_contract": {
            "source": "runtime_evidence_token_resampler",
            "token_count": QWEN_EVIDENCE_TOKEN_COUNT,
            "hidden_size": QWEN_HIDDEN_SIZE,
            "embeddings_included": False,
            "requires_runtime_resampler": True,
        },
        "permissions": deepcopy(_PATIENT_PERMISSIONS),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["input_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_qwen_patient_input(
        body,
        trusted_graph=graph,
        trusted_report=report,
        trusted_selection=selection,
    )


def validate_qwen_patient_input(
    value: object,
    *,
    trusted_graph: Mapping[str, object] | None = None,
    trusted_report: Mapping[str, object] | None = None,
    trusted_selection: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate a patient-level packet and optionally replay all source objects."""

    required = {
        "schema_version", "input_id", "linkage_group_id", "stage0_status",
        "status", "mode", "source_graph_ref", "source_report_ref",
        "knowledge_selection_ref", "prompt", "evidence_token_contract",
        "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("patient Qwen input fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != QWEN_PATIENT_INPUT_SCHEMA_VERSION:
        raise ValueError("patient Qwen input schema drifted")
    if data["status"] != QWEN_PATIENT_INPUT_STATUS or data["mode"] != QWEN_PATIENT_INPUT_MODE:
        raise ValueError("patient Qwen input remains shadow-only")
    if data["stage0_status"] != "NO_GO":
        raise ValueError("patient Qwen input must remain Stage-0 NO_GO shadow")
    if not isinstance(data["linkage_group_id"], str) or not data["linkage_group_id"]:
        raise ValueError("patient Qwen input linkage group is invalid")

    graph_ref = validate_artifact_ref(data["source_graph_ref"])
    report_ref = validate_artifact_ref(data["source_report_ref"])
    selection_ref = validate_artifact_ref(data["knowledge_selection_ref"])
    if graph_ref["artifact_kind"] != "evisoz_signal_candidate_claim_graph":
        raise ValueError("patient Qwen graph reference kind drifted")
    if report_ref["artifact_kind"] != "evisoz_canonical_report":
        raise ValueError("patient Qwen report reference kind drifted")
    if selection_ref["artifact_kind"] != "evisoz_knowledge_selection_receipt":
        raise ValueError("patient Qwen knowledge reference kind drifted")

    prompt = data["prompt"]
    if type(prompt) is not dict or set(prompt) != {
        "system_policy", "task", "knowledge_context", "claim_graph", "canonical_report"
    }:
        raise ValueError("patient Qwen prompt fields drifted")
    if prompt["system_policy"] != _SYSTEM_POLICY or prompt["task"] != "lexicalize_candidate_only":
        raise ValueError("patient Qwen prompt policy drifted")
    context = prompt["knowledge_context"]
    if type(context) is not dict or set(context) != {
        "knowledge_version", "bundle_sha256", "card_ids", "card_text_included",
        "patient_fact_creation_allowed",
    }:
        raise ValueError("patient Qwen knowledge context drifted")
    if (
        not isinstance(context["knowledge_version"], str)
        or not isinstance(context["bundle_sha256"], str)
        or len(context["bundle_sha256"]) != 64
        or context["card_text_included"] is not False
        or context["patient_fact_creation_allowed"] is not False
        or not isinstance(context["card_ids"], list)
        or context["card_ids"] != sorted(set(context["card_ids"]))
        or any(not isinstance(item, str) or not item for item in context["card_ids"])
    ):
        raise ValueError("patient Qwen knowledge context is unsafe")

    graph = validate_signal_candidate_claim_graph(prompt["claim_graph"])
    report = validate_canonical_report(prompt["canonical_report"])
    if trusted_graph is not None:
        graph_trusted = validate_signal_candidate_claim_graph(dict(trusted_graph))
        verify_artifact_content(graph_ref, graph_trusted)
        if graph != graph_trusted:
            raise ValueError("patient Qwen claim graph replay drifted")
    if trusted_report is not None:
        report_trusted = validate_canonical_report(dict(trusted_report))
        verify_artifact_content(report_ref, report_trusted)
        if report != report_trusted:
            raise ValueError("patient Qwen report replay drifted")
    if trusted_selection is not None:
        selection_trusted = validate_knowledge_selection_receipt(dict(trusted_selection))
        verify_artifact_content(selection_ref, selection_trusted)
        if _selection_card_ids(selection_trusted) != context["card_ids"]:
            raise ValueError("patient Qwen selection/card linkage drifted")
        if context["knowledge_version"] != selection_trusted["knowledge_version"]:
            raise ValueError("patient Qwen knowledge version drifted")
        if context["bundle_sha256"] != selection_trusted["knowledge_manifest_sha256"]:
            raise ValueError("patient Qwen knowledge bundle drifted")
    if graph["linkage_group_id"] != data["linkage_group_id"] or report["linkage_group_id"] != data["linkage_group_id"]:
        raise ValueError("patient Qwen linkage group drifted")
    if report["source_graph_ref"] != graph_ref or report["knowledge_selection_ref"] != selection_ref:
        raise ValueError("patient Qwen report sources do not replay packet refs")
    if _report_card_ids(report) != context["card_ids"]:
        raise ValueError("patient Qwen report/card linkage drifted")

    token_contract = data["evidence_token_contract"]
    if type(token_contract) is not dict or token_contract != {
        "source": "runtime_evidence_token_resampler",
        "token_count": QWEN_EVIDENCE_TOKEN_COUNT,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "embeddings_included": False,
        "requires_runtime_resampler": True,
    }:
        raise ValueError("patient Qwen token contract is unsafe")
    if data["permissions"] != _PATIENT_PERMISSIONS:
        raise ValueError("patient Qwen permissions drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["input_id"] != expected_id:
        raise ValueError("patient Qwen input ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("patient Qwen input receipt drifted")
    return data


__all__ = [
    "QWEN_PATIENT_INPUT_MODE",
    "QWEN_PATIENT_INPUT_SCHEMA_VERSION",
    "QWEN_PATIENT_INPUT_STATUS",
    "build_qwen_patient_input",
    "validate_qwen_patient_input",
]
