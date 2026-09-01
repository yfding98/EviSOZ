"""Fail-closed import contract for public near/partial-overlap audits.

The overlap audit is an external evidence artifact.  This module validates
its shape and content binding, but it never grants training permission.  The
Stage-0 gate may use a complete receipt to remove only the corresponding
exposure blockers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifact_ref import canonical_json_sha256, validate_artifact_ref


PUBLIC_OVERLAP_AUDIT_SCHEMA_VERSION = "evisoz_public_overlap_audit_receipt_v1"
_HASH_PLACEHOLDER = "0" * 64
_ID_PREFIX = "EVISOZ-PUBOVER-"


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["audit_id"] = "CONTENT-ADDRESS-PENDING"
    return body


def validate_public_overlap_audit_receipt(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "audit_id", "status", "source_projection_ref",
        "authority_receipt_ref", "near_partial_overlap", "tuev_eval_identity",
        "tuev_label_fold", "permissions", "missing_closure_codes",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("public overlap audit receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PUBLIC_OVERLAP_AUDIT_SCHEMA_VERSION:
        raise ValueError("public overlap audit schema drifted")
    if data["status"] not in {"complete", "pending"}:
        raise ValueError("public overlap audit status is invalid")
    source_ref = validate_artifact_ref(data["source_projection_ref"])
    if source_ref["artifact_kind"] != "public_auxiliary_exposure_projection":
        raise ValueError("public overlap source projection reference drifted")
    authority_ref = validate_artifact_ref(data["authority_receipt_ref"])
    if authority_ref["artifact_kind"] != "dataset_authority_receipt":
        raise ValueError("public overlap authority reference drifted")
    for key in ("near_partial_overlap", "tuev_eval_identity", "tuev_label_fold"):
        row = data[key]
        if type(row) is not dict or set(row) != {"status", "evidence_ref"}:
            raise ValueError(f"public overlap {key} fields drifted")
        if row["status"] not in {"complete", "pending"}:
            raise ValueError(f"public overlap {key} status is invalid")
        evidence_ref = validate_artifact_ref(row["evidence_ref"])
        if evidence_ref["artifact_kind"] not in {
            "public_overlap_evidence", "tuev_identity_crosswalk", "tuev_label_fold_receipt"
        }:
            raise ValueError(f"public overlap {key} evidence kind is invalid")
    permissions = data["permissions"]
    if permissions != {
        "training_authorized": False,
        "patient_identity_creation_by_inference": False,
        "label_promotion_authorized": False,
    }:
        raise ValueError("public overlap permissions drifted")
    complete = all(
        data[key]["status"] == "complete"
        for key in ("near_partial_overlap", "tuev_eval_identity", "tuev_label_fold")
    )
    expected_status = "complete" if complete else "pending"
    if data["status"] != expected_status:
        raise ValueError("public overlap aggregate status drifted")
    expected_missing = [] if complete else [
        key for key in ("near_partial_overlap", "tuev_eval_identity", "tuev_label_fold")
        if data[key]["status"] != "complete"
    ]
    if data["missing_closure_codes"] != expected_missing:
        raise ValueError("public overlap missing closure codes drifted")
    if not isinstance(data["audit_id"], str) or not data["audit_id"].startswith(_ID_PREFIX):
        raise ValueError("public overlap audit ID is invalid")
    if data["audit_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("public overlap audit ID does not bind content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("public overlap audit receipt drifted")
    return data


__all__ = [
    "PUBLIC_OVERLAP_AUDIT_SCHEMA_VERSION",
    "validate_public_overlap_audit_receipt",
]
