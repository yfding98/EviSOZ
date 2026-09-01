"""Content-closed materialization for patient-level Qwen shadow packets.

The inference smoke already emits one ``qwen_patient_input.json`` per opaque
linkage group.  This module gives that directory an independently replayable
manifest.  It is deliberately a Stage-0 shadow artifact: packets remain
candidate-only, contain no raw EEG or physician text, and cannot authorize
Qwen generation or training.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from src.evisoz.data.artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from src.evisoz.data.bound_evidence_loader import (
    BOUND_LOADER_RECEIPT_SCHEMA_VERSION,
    BoundEvidenceRecord,
    validate_bound_evidence_loader_receipt,
)
from src.evisoz.evaluation.bound_evidence_eval import (
    SHADOW_EVALUATION_SCHEMA_VERSION,
    validate_bound_evidence_shadow_evaluation,
)
from .qwen_patient_input import (
    QWEN_PATIENT_INPUT_SCHEMA_VERSION,
    validate_qwen_patient_input,
)


PATIENT_SHADOW_MATERIALIZATION_SCHEMA_VERSION = (
    "evisoz_qwen_patient_shadow_materialization_v1"
)
PATIENT_SHADOW_MATERIALIZATION_STATUS = "real_loader_patient_shadow_materialized"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-QPATIENT-MAT-"

_RUNTIME_POLICY = {
    "physician_report_text_opened": False,
    "canonical_shadow_report_opened": True,
    "teacher_runtime_opened": False,
    "training_allowed": False,
    "prompt_or_rag_allowed": False,
    "patient_fact_created": False,
    "candidate_only": True,
}
_PERMISSIONS = {
    "qwen_generation_allowed": False,
    "patient_fact_creation_allowed": False,
    "embeddings_materialized": False,
    "requires_physician_review": True,
}


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["materialization_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("patient shadow relative_path must be a string")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or not parsed.parts or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError("patient shadow relative_path is unsafe")
    if len(parsed.parts) != 3 or parsed.parts[0] != "patients":
        raise ValueError("patient shadow relative_path must be patients/<id>/qwen_patient_input.json")
    if parsed.parts[2] != "qwen_patient_input.json":
        raise ValueError("patient shadow packet filename drifted")
    return value


def _read_packet(root: Path, relative_path: str) -> dict[str, Any]:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("patient shadow packet is missing") from exc
    resolved.relative_to(root.resolve(strict=True))
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError("patient shadow packet must be a regular file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("patient shadow packet must be a JSON object")
    return value


def _rows_by_group(
    records: Iterable[BoundEvidenceRecord],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[BoundEvidenceRecord]] = {}
    for record in records:
        if not isinstance(record, BoundEvidenceRecord):
            raise TypeError("records must come from bound_evidence_loader")
        grouped.setdefault(record.linkage_group_id, []).append(record)
    if not grouped:
        raise ValueError("patient shadow materialization received no records")
    result: dict[str, dict[str, Any]] = {}
    for group_id, group in sorted(grouped.items()):
        event_ids = sorted({record.event_id for record in group})
        if len(event_ids) != len(group):
            raise ValueError("patient shadow records contain duplicate events")
        roles = {record.evisoz_role for record in group}
        if len(roles) != 1:
            raise ValueError("one patient cannot mix Stage-0 split roles")
        role = next(iter(roles))
        folds = {record.bound_evidence["outer_holdout_fold"] for record in group}
        if len(folds) != 1:
            raise ValueError("one patient cannot mix outer holdout folds")
        result[group_id] = {
            "event_ids": event_ids,
            "evisoz_role": role,
            "outer_holdout_fold": next(iter(folds)),
        }
    return result


def build_qwen_patient_shadow_materialization(
    *,
    records: Iterable[BoundEvidenceRecord],
    patient_packets: Mapping[str, Mapping[str, object]],
    loader_receipt: Mapping[str, object],
    shadow_evaluation: Mapping[str, object],
) -> dict[str, Any]:
    """Build a patient packet manifest from loader-bound records.

    ``patient_packets`` must be the output of
    :func:`build_bound_patient_qwen_shadow_inputs`; callers cannot provide a
    packet for a patient absent from the replayed record roster.
    """

    records_by_group = _rows_by_group(records)
    loader = validate_bound_evidence_loader_receipt(dict(loader_receipt))
    evaluation = validate_bound_evidence_shadow_evaluation(dict(shadow_evaluation))
    if evaluation["source"]["event_ids"] != loader["selection"]["event_ids"]:
        raise ValueError("patient shadow evaluation and loader event rosters drifted")
    if set(patient_packets) != set(records_by_group):
        raise ValueError("patient shadow packet roster does not match loader groups")

    loader_ref = build_json_artifact_ref(
        loader,
        artifact_kind="bound_evidence_loader_receipt",
        payload_schema_version=BOUND_LOADER_RECEIPT_SCHEMA_VERSION,
    )
    evaluation_ref = build_json_artifact_ref(
        evaluation,
        artifact_kind="bound_evidence_shadow_evaluation",
        payload_schema_version=SHADOW_EVALUATION_SCHEMA_VERSION,
    )
    rows: list[dict[str, Any]] = []
    for group_id in sorted(records_by_group):
        packet = validate_qwen_patient_input(dict(patient_packets[group_id]))
        if packet["linkage_group_id"] != group_id:
            raise ValueError("patient shadow packet linkage group drifted")
        row = records_by_group[group_id]
        rows.append(
            {
                "linkage_group_id": group_id,
                "event_ids": row["event_ids"],
                "evisoz_role": row["evisoz_role"],
                "outer_holdout_fold": row["outer_holdout_fold"],
                "qwen_patient_input_ref": build_json_artifact_ref(
                    packet,
                    artifact_kind="evisoz_qwen_patient_input",
                    payload_schema_version=QWEN_PATIENT_INPUT_SCHEMA_VERSION,
                ),
                "relative_path": f"patients/{group_id}/qwen_patient_input.json",
            }
        )

    body: dict[str, Any] = {
        "schema_version": PATIENT_SHADOW_MATERIALIZATION_SCHEMA_VERSION,
        "materialization_id": _HASH_PLACEHOLDER,
        "status": PATIENT_SHADOW_MATERIALIZATION_STATUS,
        "source_refs": {
            "loader_receipt": loader_ref,
            "shadow_evaluation": evaluation_ref,
        },
        "rows": rows,
        "counts": {
            "event_count": len(loader["selection"]["event_ids"]),
            "patient_count": len(rows),
            "packet_count": len(rows),
            "development_patient_count": sum(
                row["evisoz_role"] == "development_cv" for row in rows
            ),
            "locked_test_patient_count": sum(
                row["evisoz_role"] == "locked_test" for row in rows
            ),
        },
        "runtime_policy": deepcopy(_RUNTIME_POLICY),
        "permissions": deepcopy(_PERMISSIONS),
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["materialization_id"] = _ID_PREFIX + canonical_json_sha256(
        _id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_qwen_patient_shadow_materialization(
        body,
        trusted_packets=patient_packets,
        trusted_loader_receipt=loader,
        trusted_evaluation=evaluation,
    )


def validate_qwen_patient_shadow_materialization(
    value: object,
    *,
    trusted_packets: Mapping[str, Mapping[str, object]] | None = None,
    trusted_loader_receipt: Mapping[str, object] | None = None,
    trusted_evaluation: Mapping[str, object] | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a packet manifest and optionally replay its source files."""

    required = {
        "schema_version", "materialization_id", "status", "source_refs",
        "rows", "counts", "runtime_policy", "permissions", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("patient shadow materialization fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PATIENT_SHADOW_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("patient shadow materialization schema drifted")
    if data["status"] != PATIENT_SHADOW_MATERIALIZATION_STATUS:
        raise ValueError("patient shadow materialization status drifted")

    refs = data["source_refs"]
    if type(refs) is not dict or set(refs) != {"loader_receipt", "shadow_evaluation"}:
        raise ValueError("patient shadow source refs drifted")
    loader_ref = validate_artifact_ref(refs["loader_receipt"])
    evaluation_ref = validate_artifact_ref(refs["shadow_evaluation"])
    if (
        loader_ref["artifact_kind"] != "bound_evidence_loader_receipt"
        or loader_ref["payload_schema_version"] != BOUND_LOADER_RECEIPT_SCHEMA_VERSION
        or evaluation_ref["artifact_kind"] != "bound_evidence_shadow_evaluation"
        or evaluation_ref["payload_schema_version"] != SHADOW_EVALUATION_SCHEMA_VERSION
    ):
        raise ValueError("patient shadow source reference kind/version drifted")

    if trusted_loader_receipt is not None:
        loader = validate_bound_evidence_loader_receipt(dict(trusted_loader_receipt))
        verify_artifact_content(loader_ref, loader)
    else:
        loader = None
    if trusted_evaluation is not None:
        evaluation = validate_bound_evidence_shadow_evaluation(dict(trusted_evaluation))
        verify_artifact_content(evaluation_ref, evaluation)
    else:
        evaluation = None
    if loader is not None and evaluation is not None:
        if evaluation["source"]["event_ids"] != loader["selection"]["event_ids"]:
            raise ValueError("patient shadow source event rosters drifted")

    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("patient shadow materialization rows are empty")
    group_ids: list[str] = []
    event_ids: list[str] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "linkage_group_id", "event_ids", "evisoz_role", "outer_holdout_fold",
            "qwen_patient_input_ref", "relative_path",
        }:
            raise ValueError("patient shadow materialization row fields drifted")
        group_id = row["linkage_group_id"]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("patient shadow linkage group is invalid")
        if group_id in group_ids:
            raise ValueError("patient shadow linkage groups are duplicated")
        group_ids.append(group_id)
        row_events = row["event_ids"]
        if not isinstance(row_events, list) or not row_events or row_events != sorted(set(row_events)):
            raise ValueError("patient shadow event roster is not sorted/unique")
        if any(not isinstance(event_id, str) or not event_id for event_id in row_events):
            raise ValueError("patient shadow event ID is invalid")
        event_ids.extend(row_events)
        if row["evisoz_role"] not in {"development_cv", "locked_test"}:
            raise ValueError("patient shadow split role drifted")
        if row["evisoz_role"] == "locked_test" and row["outer_holdout_fold"] is not None:
            raise ValueError("locked patient shadow row must not expose an outer fold")
        if row["evisoz_role"] == "development_cv" and (
            isinstance(row["outer_holdout_fold"], bool)
            or not isinstance(row["outer_holdout_fold"], int)
            or row["outer_holdout_fold"] < 0
        ):
            raise ValueError("development patient shadow row has invalid outer fold")
        packet_ref = validate_artifact_ref(row["qwen_patient_input_ref"])
        if (
            packet_ref["artifact_kind"] != "evisoz_qwen_patient_input"
            or packet_ref["payload_schema_version"] != QWEN_PATIENT_INPUT_SCHEMA_VERSION
        ):
            raise ValueError("patient shadow packet reference drifted")
        relative_path = _safe_relative_path(row["relative_path"])
        if relative_path != f"patients/{group_id}/qwen_patient_input.json":
            raise ValueError("patient shadow packet path/linkage drifted")
        packet: Mapping[str, object] | None = None
        if trusted_packets is not None:
            if group_id not in trusted_packets:
                raise ValueError("trusted patient shadow packet is missing")
            packet = validate_qwen_patient_input(dict(trusted_packets[group_id]))
            if packet["linkage_group_id"] != group_id:
                raise ValueError("trusted patient shadow packet linkage drifted")
            verify_artifact_content(packet_ref, packet)
        if output_root is not None:
            root = Path(output_root).resolve(strict=True)
            on_disk = validate_qwen_patient_input(_read_packet(root, relative_path))
            if on_disk["linkage_group_id"] != group_id:
                raise ValueError("on-disk patient shadow packet linkage drifted")
            verify_artifact_content(packet_ref, on_disk)
            if packet is not None and on_disk != packet:
                raise ValueError("on-disk patient shadow packet differs from trusted packet")

    if len(event_ids) != len(set(event_ids)):
        raise ValueError("patient shadow event roster must be globally unique")
    if loader is not None and set(event_ids) != set(loader["selection"]["event_ids"]):
        raise ValueError("patient shadow event roster does not match loader selection")
    counts = data["counts"]
    expected_counts = {
        "event_count": len(event_ids),
        "patient_count": len(rows),
        "packet_count": len(rows),
        "development_patient_count": sum(
            row["evisoz_role"] == "development_cv" for row in rows
        ),
        "locked_test_patient_count": sum(
            row["evisoz_role"] == "locked_test" for row in rows
        ),
    }
    if counts != expected_counts:
        raise ValueError("patient shadow materialization counts drifted")
    if data["runtime_policy"] != _RUNTIME_POLICY or data["permissions"] != _PERMISSIONS:
        raise ValueError("patient shadow runtime permissions drifted")
    if data["materialization_id"] != _ID_PREFIX + canonical_json_sha256(
        _id_source(data)
    )[:24]:
        raise ValueError("patient shadow materialization ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("patient shadow materialization receipt drifted")
    return data


__all__ = [
    "PATIENT_SHADOW_MATERIALIZATION_SCHEMA_VERSION",
    "PATIENT_SHADOW_MATERIALIZATION_STATUS",
    "build_qwen_patient_shadow_materialization",
    "validate_qwen_patient_shadow_materialization",
]
