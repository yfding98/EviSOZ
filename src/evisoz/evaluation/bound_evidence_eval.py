"""Structural evaluation for loader-backed EviSOZ shadow predictions.

This evaluator intentionally does not open evaluator-only clinical labels.  It
checks that predictions and candidate report plans remain bound to the same
loader-replayed event, preserve node/edge masks, and stay candidate-only.
Clinical SOZ metrics belong to a separately authorized evaluator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from src.evisoz.data.artifact_ref import canonical_json_sha256
from src.evisoz.data.bound_evidence_loader import (
    BoundEvidenceRecord,
    validate_bound_evidence_loader_receipt,
)
from src.evisoz.models.predicted_evidence import validate_predicted_evidence_packet
from src.evisoz.reporting.predicted_report_plan import validate_predicted_report_plan
from src.evisoz.reporting.qwen_structured_input import validate_qwen_structured_input


SHADOW_EVALUATION_SCHEMA_VERSION = "evisoz_bound_evidence_shadow_evaluation_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-SHADOW-EVAL-"


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["evaluation_id"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _prediction_parts(value: object) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if hasattr(value, "predicted_evidence") and hasattr(value, "report_plan"):
        packet = getattr(value, "predicted_evidence")
        plan = getattr(value, "report_plan")
    elif isinstance(value, Mapping):
        packet = value.get("predicted_evidence", value.get("packet"))
        plan = value.get("report_plan", value.get("plan"))
    else:
        raise TypeError("prediction must be a ShadowInferenceResult or mapping")
    if not isinstance(packet, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("prediction packet and report plan are required")
    return packet, plan


def _qwen_input(value: object) -> Mapping[str, Any] | None:
    if hasattr(value, "qwen_structured_input"):
        candidate = getattr(value, "qwen_structured_input")
    elif isinstance(value, Mapping):
        candidate = value.get("qwen_structured_input")
    else:
        candidate = None
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise ValueError("prediction Qwen input must be a mapping")
    return candidate


def evaluate_bound_evidence_shadow_predictions(
    records: Iterable[BoundEvidenceRecord],
    predictions: Iterable[object],
    *,
    loader_receipt: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Evaluate loader/prediction/report linkage without opening labels."""

    replayed = list(records)
    predicted = list(predictions)
    if not replayed or len(replayed) != len(predicted):
        raise ValueError("records and predictions must be non-empty and aligned")
    if loader_receipt is not None:
        validate_bound_evidence_loader_receipt(dict(loader_receipt))
    event_ids = [record.event_id for record in replayed]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("loader records contain duplicate events")

    packet_ids: set[str] = set()
    mask_ok = 0
    plan_ok = 0
    claim_units_ok = 0
    for record, value in zip(replayed, predicted):
        packet_raw, plan_raw = _prediction_parts(value)
        packet = validate_predicted_evidence_packet(dict(packet_raw))
        plan = validate_predicted_report_plan(dict(plan_raw))
        qwen_input = _qwen_input(value)
        if qwen_input is not None:
            validate_qwen_structured_input(
                dict(qwen_input),
                trusted_plan=plan,
                trusted_selection=record.knowledge_selection,
            )
            if qwen_input["event_id"] != record.event_id:
                raise ValueError("Qwen input event identity does not match loader record")
        if packet["event_id"] != record.event_id or plan["event_id"] != record.event_id:
            raise ValueError("prediction event identity does not match loader record")
        if packet["packet_id"] in packet_ids:
            raise ValueError("prediction packet IDs are duplicated")
        packet_ids.add(packet["packet_id"])
        if plan["source_packet"] != {
            "packet_id": packet["packet_id"],
            "receipt_sha256": packet["receipt_sha256"],
        }:
            raise ValueError("report plan is not bound to its evidence packet")
        inputs = record.checkout_inputs()
        expected_node_mask = list(inputs["standard19_observed_mask"])
        expected_edge_mask = list(inputs["tcp22_observed_mask"])
        if packet["observed_node_mask"] == expected_node_mask and packet["observed_edge_mask"] == expected_edge_mask:
            mask_ok += 1
        for claim in plan["claims"]:
            concept = claim["concept"]
            if concept == "onset_candidate_node":
                allowed = set(packet["node_units"])
            elif concept == "spread_candidate_edge":
                allowed = set(packet["edge_units"])
            else:
                allowed = set()
            if any(str(unit) not in allowed for unit in claim["support_units"]):
                raise ValueError("report claim references an unknown support unit")
        claim_units_ok += 1
        plan_ok += 1

    body: dict[str, Any] = {
        "schema_version": SHADOW_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": _HASH_PLACEHOLDER,
        "status": "structural_shadow_only",
        "source": {
            "event_ids": event_ids,
            "loader_receipt_sha256": (
                str(loader_receipt["receipt_sha256"])
                if loader_receipt is not None
                else None
            ),
        },
        "counts": {
            "event_count": len(replayed),
            "packet_count": len(packet_ids),
            "report_plan_count": plan_ok,
        },
        "metrics": {
            "mask_consistency_rate": float(mask_ok / len(replayed)),
            "report_plan_linkage_rate": float(plan_ok / len(replayed)),
            "claim_support_unit_validity_rate": float(claim_units_ok / len(replayed)),
        },
        "runtime_policy": {
            "clinical_labels_opened": False,
            "physician_report_text_opened": False,
            "teacher_runtime_opened": False,
            "clinical_soz_metric_computed": False,
            "patient_fact_created": False,
            "candidate_only": True,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["evaluation_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_bound_evidence_shadow_evaluation(body)


def validate_bound_evidence_shadow_evaluation(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "evaluation_id", "status", "source", "counts",
        "metrics", "runtime_policy", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("shadow evaluation fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != SHADOW_EVALUATION_SCHEMA_VERSION or data["status"] != "structural_shadow_only":
        raise ValueError("shadow evaluation identity drifted")
    if not isinstance(data["source"].get("event_ids"), list) or not data["source"]["event_ids"]:
        raise ValueError("shadow evaluation source event roster is empty")
    counts = data["counts"]
    if counts != {
        "event_count": len(data["source"]["event_ids"]),
        "packet_count": len(data["source"]["event_ids"]),
        "report_plan_count": len(data["source"]["event_ids"]),
    }:
        raise ValueError("shadow evaluation counts drifted")
    metrics = data["metrics"]
    if set(metrics) != {"mask_consistency_rate", "report_plan_linkage_rate", "claim_support_unit_validity_rate"}:
        raise ValueError("shadow evaluation metric roster drifted")
    if any(not isinstance(value, (int, float)) or not 0 <= float(value) <= 1 for value in metrics.values()):
        raise ValueError("shadow evaluation metrics are invalid")
    if data["runtime_policy"] != {
        "clinical_labels_opened": False,
        "physician_report_text_opened": False,
        "teacher_runtime_opened": False,
        "clinical_soz_metric_computed": False,
        "patient_fact_created": False,
        "candidate_only": True,
    }:
        raise ValueError("shadow evaluation runtime policy drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["evaluation_id"] != expected_id or data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("shadow evaluation receipt hash drifted")
    return data


__all__ = [
    "SHADOW_EVALUATION_SCHEMA_VERSION",
    "evaluate_bound_evidence_shadow_predictions",
    "validate_bound_evidence_shadow_evaluation",
]
