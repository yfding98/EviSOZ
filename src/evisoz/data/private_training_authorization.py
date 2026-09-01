"""Explicit external authorization for private field-level training.

The private physician-report release contract is intentionally separate from
this contract.  A report-text authorization cannot grant clinical-label loss
ports.  This module validates the small, content-addressed receipt required
before a private Stage-0 materializer may enable any direct-label loss.

The receipt contains no raw patient identifiers or label values.  It binds the
exact source ledgers and split roster through artifact references and carries a
detached-signature verification receipt supplied by the external data
controller.  In particular, a self-attested boolean or a report-release
authorization is never sufficient.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Mapping

from .artifact_ref import (
    CANONICAL_JSON_HASH_DOMAIN,
    RAW_BYTES_HASH_DOMAIN,
    canonical_json_sha256,
    validate_artifact_ref,
)


PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION = (
    "evisoz_private_training_authorization_v1"
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-PAUTH-"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_ISO_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_LOSS_PORTS = ("typed_slot_loss", "node_localization_loss")
_FORBIDDEN_FIELD_IDS = {"PRIVATE-PHYSICIAN-REPORT-TEXT"}
_REFERENCE_KINDS = {
    "split_roster_ref": ("split_roster", CANONICAL_JSON_HASH_DOMAIN),
    "signal_roster_ref": ("private_signal_roster", RAW_BYTES_HASH_DOMAIN),
    "target_ledger_ref": ("private_target_ledger", RAW_BYTES_HASH_DOMAIN),
    "source_manifest_ref": (
        "private_label_authority_manifest",
        RAW_BYTES_HASH_DOMAIN,
    ),
}


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable identifier")
    return value


def _iso_utc(value: object, context: str) -> str:
    if not isinstance(value, str) or _ISO_UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include an explicit UTC designator")
    return value


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["authorization_id"] = _PENDING_ID
    return result


def _reference(
    value: object,
    *,
    context: str,
    expected_kind: str | None = None,
    expected_domain: str | None = None,
) -> dict[str, Any]:
    reference = validate_artifact_ref(value)
    if expected_kind is not None and reference["artifact_kind"] != expected_kind:
        raise ValueError(f"{context} artifact kind drifted")
    if expected_domain is not None and reference["content_hash"]["domain"] != expected_domain:
        raise ValueError(f"{context} hash domain drifted")
    return reference


def _normalize_field_scope(value: object) -> dict[str, Any]:
    required = {
        "allowed_evisoz_roles",
        "field_permissions",
        "locked_test_training_allowed",
        "report_text_loss_allowed",
        "prompt_or_rag_allowed",
        "report_text_can_supervise_localization",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("private training authorization field_scope fields drifted")
    roles = value["allowed_evisoz_roles"]
    if roles != ["development_cv"]:
        raise ValueError(
            "private clinical-label training authorization may only target development_cv"
        )
    raw_permissions = value["field_permissions"]
    if not isinstance(raw_permissions, list) or not raw_permissions:
        raise ValueError("private training authorization field_permissions must be non-empty")
    permissions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_permissions):
        context = f"field_scope.field_permissions[{index}]"
        if type(raw) is not dict or set(raw) != {"field_id", "loss_ports"}:
            raise ValueError(f"{context} fields drifted")
        field_id = _identifier(raw["field_id"], f"{context}.field_id")
        if field_id in _FORBIDDEN_FIELD_IDS:
            raise ValueError("physician report text cannot be authorized by private label training")
        ports = raw["loss_ports"]
        if (
            not isinstance(ports, list)
            or not ports
            or ports != sorted(set(ports))
            or any(port not in _LOSS_PORTS for port in ports)
        ):
            raise ValueError(f"{context}.loss_ports must be sorted and limited to EEG losses")
        permissions.append({"field_id": field_id, "loss_ports": list(ports)})
    if permissions != sorted(permissions, key=lambda row: row["field_id"]):
        raise ValueError("private training authorization field_permissions must be sorted")
    if len({row["field_id"] for row in permissions}) != len(permissions):
        raise ValueError("private training authorization field IDs must be unique")
    if value["locked_test_training_allowed"] is not False:
        raise ValueError("locked_test training is permanently forbidden")
    if value["report_text_loss_allowed"] is not False:
        raise ValueError("private label authorization cannot grant report_text_loss")
    if value["prompt_or_rag_allowed"] is not False:
        raise ValueError("private labels cannot be used for prompt or RAG")
    if value["report_text_can_supervise_localization"] is not False:
        raise ValueError("report text cannot supervise localization")
    return {
        "allowed_evisoz_roles": ["development_cv"],
        "field_permissions": permissions,
        "locked_test_training_allowed": False,
        "report_text_loss_allowed": False,
        "prompt_or_rag_allowed": False,
        "report_text_can_supervise_localization": False,
    }


def _normalize_data_binding(value: object) -> dict[str, Any]:
    required = {
        "dataset_id",
        "patient_roster_sha256",
        "split_roster_ref",
        "signal_roster_ref",
        "target_ledger_ref",
        "source_manifest_ref",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("private training authorization data_binding fields drifted")
    if value["dataset_id"] != "private":
        raise ValueError("private training authorization dataset_id must be private")
    normalized: dict[str, Any] = {
        "dataset_id": "private",
        "patient_roster_sha256": _sha256(
            value["patient_roster_sha256"],
            "data_binding.patient_roster_sha256",
        ),
    }
    for key, (kind, domain) in _REFERENCE_KINDS.items():
        normalized[key] = _reference(
            value[key],
            context=f"data_binding.{key}",
            expected_kind=kind,
            expected_domain=domain,
        )
    return normalized


def validate_private_training_authorization(
    value: object,
    *,
    expected_bindings: Mapping[str, object] | None = None,
    expected_field_ids: set[str] | None = None,
    as_of_utc: str | None = None,
) -> dict[str, Any]:
    """Validate an external authorization and optionally replay its bindings."""

    required = {
        "schema_version",
        "authorization_id",
        "status",
        "issuer",
        "signature",
        "effective_window",
        "data_binding",
        "field_scope",
        "permissions",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("private training authorization fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("private training authorization schema_version drifted")
    if data["status"] != "active":
        raise ValueError("private training authorization is not active")
    issuer = data["issuer"]
    if type(issuer) is not dict or set(issuer) != {
        "institution",
        "role",
        "approval_reference",
    }:
        raise ValueError("private training authorization issuer fields drifted")
    for key in ("institution", "approval_reference"):
        _identifier(issuer[key], f"issuer.{key}")
    if issuer["role"] != "data_controller":
        raise ValueError("private training authorization issuer must be data_controller")

    signature = data["signature"]
    if type(signature) is not dict or set(signature) != {
        "scheme",
        "signature_reference",
        "signed_payload_sha256",
        "verification_receipt_ref",
        "verification_status",
    }:
        raise ValueError("private training authorization signature fields drifted")
    if signature["scheme"] != "detached_signature":
        raise ValueError("private training authorization must use a detached signature")
    _identifier(signature["signature_reference"], "signature.signature_reference")
    _sha256(signature["signed_payload_sha256"], "signature.signed_payload_sha256")
    _reference(
        signature["verification_receipt_ref"],
        context="signature.verification_receipt_ref",
        expected_kind="governance_signature_verification",
        expected_domain=RAW_BYTES_HASH_DOMAIN,
    )
    if signature["verification_status"] != "verified":
        raise ValueError("detached signature verification has not passed")

    window = data["effective_window"]
    if type(window) is not dict or set(window) != {"effective_from", "effective_until"}:
        raise ValueError("private training authorization effective_window fields drifted")
    effective_from = _iso_utc(window["effective_from"], "effective_window.effective_from")
    effective_until = _iso_utc(window["effective_until"], "effective_window.effective_until")
    from_dt = datetime.fromisoformat(effective_from[:-1] + "+00:00")
    until_dt = datetime.fromisoformat(effective_until[:-1] + "+00:00")
    if until_dt <= from_dt:
        raise ValueError("private training authorization effective window is empty")
    if as_of_utc is not None:
        current = datetime.fromisoformat(_iso_utc(as_of_utc, "as_of_utc")[:-1] + "+00:00")
        if current < from_dt or current >= until_dt:
            raise ValueError("private training authorization is outside its effective window")

    binding = _normalize_data_binding(data["data_binding"])
    if expected_bindings is not None:
        for key in (
            "dataset_id",
            "patient_roster_sha256",
            "split_roster_ref",
            "signal_roster_ref",
            "target_ledger_ref",
            "source_manifest_ref",
        ):
            if key not in expected_bindings or binding[key] != expected_bindings[key]:
                raise ValueError(f"private training authorization {key} binding drifted")

    scope = _normalize_field_scope(data["field_scope"])
    authorized_ids = {row["field_id"] for row in scope["field_permissions"]}
    if expected_field_ids is not None and not authorized_ids.issubset(expected_field_ids):
        raise ValueError("private training authorization names an unknown field")

    permissions = data["permissions"]
    expected_permissions = {
        "training_authorized": True,
        "private_labels_can_supervise_eeg": True,
        "locked_test_training": False,
        "report_text_loss": False,
        "qwen_text_training": False,
        "prompt_or_rag": False,
        "knowledge_can_create_patient_facts": False,
    }
    if permissions != expected_permissions:
        raise ValueError("private training authorization permissions drifted")

    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["authorization_id"] != expected_id:
        raise ValueError("private training authorization ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("private training authorization receipt hash drifted")
    data["issuer"] = {
        "institution": issuer["institution"],
        "role": issuer["role"],
        "approval_reference": issuer["approval_reference"],
    }
    data["data_binding"] = binding
    data["field_scope"] = scope
    return data


def build_private_training_authorization(
    *,
    issuer: Mapping[str, object],
    signature: Mapping[str, object],
    effective_window: Mapping[str, object],
    data_binding: Mapping[str, object],
    field_permissions: list[Mapping[str, object]],
) -> dict[str, Any]:
    """Build a canonical receipt for external-controller tooling/tests."""

    body: dict[str, Any] = {
        "schema_version": PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": _PENDING_ID,
        "status": "active",
        "issuer": deepcopy(dict(issuer)),
        "signature": deepcopy(dict(signature)),
        "effective_window": deepcopy(dict(effective_window)),
        "data_binding": deepcopy(dict(data_binding)),
        "field_scope": {
            "allowed_evisoz_roles": ["development_cv"],
            "field_permissions": [deepcopy(dict(row)) for row in field_permissions],
            "locked_test_training_allowed": False,
            "report_text_loss_allowed": False,
            "prompt_or_rag_allowed": False,
            "report_text_can_supervise_localization": False,
        },
        "permissions": {
            "training_authorized": True,
            "private_labels_can_supervise_eeg": True,
            "locked_test_training": False,
            "report_text_loss": False,
            "qwen_text_training": False,
            "prompt_or_rag": False,
            "knowledge_can_create_patient_facts": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["authorization_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_training_authorization(body)


__all__ = [
    "PRIVATE_TRAINING_AUTHORIZATION_SCHEMA_VERSION",
    "build_private_training_authorization",
    "validate_private_training_authorization",
]
