"""Structural evaluator for patient-level candidate-only Qwen packets."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

from src.evisoz.data.artifact_ref import canonical_json_sha256
from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord
from src.evisoz.reporting.qwen_patient_input import (
    build_qwen_patient_input,
    validate_qwen_patient_input,
)


PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION = (
    "evisoz_patient_qwen_shadow_evaluation_v1"
)
PATIENT_QWEN_SHADOW_EVALUATION_STATUS = "patient_structural_shadow_only"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-PATIENT-QWEN-EVAL-"

_RUNTIME_POLICY = {
    "clinical_labels_opened": False,
    "physician_report_text_opened": False,
    "teacher_runtime_opened": False,
    "clinical_soz_metric_computed": False,
    "patient_fact_created": False,
    "candidate_only": True,
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["evaluation_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def evaluate_bound_patient_qwen_shadow_inputs(
    records: Iterable[BoundEvidenceRecord],
    patient_packets: Mapping[str, Mapping[str, object]],
    *,
    loader_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay each patient packet against loader-bound graph/report sources.

    This evaluator is structural only.  It does not open field-release label
    values, physician DOCX text, or any teacher runtime, and it never computes
    a clinical SOZ metric.
    """

    rows = list(records)
    if not rows:
        raise ValueError("patient Qwen evaluator received no records")
    grouped: dict[str, list[BoundEvidenceRecord]] = defaultdict(list)
    for record in rows:
        if not isinstance(record, BoundEvidenceRecord):
            raise TypeError("records must come from bound_evidence_loader")
        grouped[record.linkage_group_id].append(record)
    if set(patient_packets) != set(grouped):
        raise ValueError("patient packet roster does not match loader groups")

    event_ids = sorted(record.event_id for record in rows)
    graph_ok = report_ok = selection_ok = packet_ok = 0
    for linkage_group_id in sorted(grouped):
        group = grouped[linkage_group_id]
        first = group[0]
        graph = first.signal_candidate_claim_graph
        report = first.canonical_report
        selection = first.knowledge_selection
        if graph is None or report is None or selection is None:
            raise ValueError("patient lacks complete graph/report/selection source")
        for record in group[1:]:
            if (
                record.signal_candidate_claim_graph is None
                or record.canonical_report is None
                or record.knowledge_selection is None
                or record.signal_candidate_claim_graph["graph_id"] != graph["graph_id"]
                or record.canonical_report["report_id"] != report["report_id"]
                or record.knowledge_selection["selection_id"] != selection["selection_id"]
            ):
                raise ValueError("patient source graph/report/selection drifted")
        expected = build_qwen_patient_input(
            signal_graph=graph,
            canonical_report=report,
            knowledge_selection=selection,
        )
        packet = validate_qwen_patient_input(
            dict(patient_packets[linkage_group_id]),
            trusted_graph=graph,
            trusted_report=report,
            trusted_selection=selection,
        )
        if packet != expected:
            raise ValueError("patient Qwen packet differs from trusted source replay")
        graph_ok += 1
        report_ok += 1
        selection_ok += 1
        packet_ok += 1

    if loader_receipt_sha256 is not None and (
        not isinstance(loader_receipt_sha256, str)
        or len(loader_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in loader_receipt_sha256)
    ):
        raise ValueError("loader receipt SHA-256 is invalid")
    patient_count = len(grouped)
    body: dict[str, Any] = {
        "schema_version": PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION,
        "evaluation_id": _HASH_PLACEHOLDER,
        "status": PATIENT_QWEN_SHADOW_EVALUATION_STATUS,
        "source": {
            "event_ids": event_ids,
            "linkage_group_ids": sorted(grouped),
            "loader_receipt_sha256": loader_receipt_sha256,
        },
        "counts": {
            "event_count": len(rows),
            "patient_count": patient_count,
            "packet_count": len(patient_packets),
        },
        "metrics": {
            "graph_replay_rate": float(graph_ok / patient_count),
            "canonical_report_replay_rate": float(report_ok / patient_count),
            "knowledge_selection_replay_rate": float(selection_ok / patient_count),
            "packet_replay_rate": float(packet_ok / patient_count),
        },
        "runtime_policy": deepcopy(_RUNTIME_POLICY),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["evaluation_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_patient_qwen_shadow_evaluation(body)


def validate_patient_qwen_shadow_evaluation(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "evaluation_id", "status", "source", "counts",
        "metrics", "runtime_policy", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("patient Qwen shadow evaluation fields drifted")
    data = deepcopy(value)
    if (
        data["schema_version"] != PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION
        or data["status"] != PATIENT_QWEN_SHADOW_EVALUATION_STATUS
    ):
        raise ValueError("patient Qwen shadow evaluation identity drifted")
    source = data["source"]
    if type(source) is not dict or set(source) != {
        "event_ids", "linkage_group_ids", "loader_receipt_sha256"
    }:
        raise ValueError("patient Qwen shadow evaluation source fields drifted")
    if (
        not isinstance(source["event_ids"], list)
        or not source["event_ids"]
        or source["event_ids"] != sorted(set(source["event_ids"]))
        or not isinstance(source["linkage_group_ids"], list)
        or not source["linkage_group_ids"]
        or source["linkage_group_ids"] != sorted(set(source["linkage_group_ids"]))
    ):
        raise ValueError("patient Qwen shadow evaluation source roster is invalid")
    receipt = source["loader_receipt_sha256"]
    if receipt is not None and (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or any(char not in "0123456789abcdef" for char in receipt)
    ):
        raise ValueError("patient Qwen shadow evaluation loader receipt is invalid")
    counts = data["counts"]
    if counts != {
        "event_count": len(source["event_ids"]),
        "patient_count": len(source["linkage_group_ids"]),
        "packet_count": len(source["linkage_group_ids"]),
    }:
        raise ValueError("patient Qwen shadow evaluation counts drifted")
    metrics = data["metrics"]
    if set(metrics) != {
        "graph_replay_rate", "canonical_report_replay_rate",
        "knowledge_selection_replay_rate", "packet_replay_rate",
    } or any(
        not isinstance(item, (int, float)) or not 0 <= float(item) <= 1
        for item in metrics.values()
    ):
        raise ValueError("patient Qwen shadow evaluation metrics drifted")
    if data["runtime_policy"] != _RUNTIME_POLICY:
        raise ValueError("patient Qwen shadow evaluation runtime policy drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["evaluation_id"] != expected_id:
        raise ValueError("patient Qwen shadow evaluation ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("patient Qwen shadow evaluation receipt drifted")
    return data


__all__ = [
    "PATIENT_QWEN_SHADOW_EVALUATION_SCHEMA_VERSION",
    "PATIENT_QWEN_SHADOW_EVALUATION_STATUS",
    "evaluate_bound_patient_qwen_shadow_inputs",
    "validate_patient_qwen_shadow_evaluation",
]
