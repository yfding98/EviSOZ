"""Field-level supervision and report-scope permissions for EviSOZ."""

from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Any, Mapping, Sequence

from src.soz.geometry import STANDARD_19

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)
from .event_identity import (
    EVENT_IDENTITY_SCHEMA_VERSION,
    validate_event_identity,
)


FIELD_RELEASE_SCHEMA_VERSION = "evisoz_field_release_v1"

REPORT_SCOPES = (
    "atomic_event",
    "ictal_findings",
    "soz_localization",
    "full_soz",
    "nonlocalizing",
)
FIELD_STATES = ("provided", "not_provided", "not_evaluable", "technical_failure")
AUTHORITIES = (
    "physician",
    "physician_authored_text",
    "dataset_direct",
    "teacher_programmatic",
    "signal_derived",
    "generated_text",
    "knowledge_rule",
)
QUALITY_TIERS = ("gold_lite", "silver", "programmatic", "uncertain", "not_applicable")
SEMANTIC_ROLES = (
    "node_label",
    "region_label",
    "laterality_label",
    "morphology",
    "spread",
    "evolution",
    "quality",
    "localizability",
    "text",
    "knowledge_rule",
    "other",
)
CLAIM_PERMISSIONS = ("direct", "candidate_only", "none")
LOSS_PORTS = (
    "typed_slot_loss",
    "node_localization_loss",
    "report_text_loss",
)
EVISOZ_ROLES = ("development_cv", "locked_test", "external_evaluation")

NODE_LABEL_VALUE_SCHEMA_VERSION = "evisoz_node_label_value_v1"
CANDIDATE_VALUE_SCHEMA_VERSION = "evisoz_candidate_value_v1"
REPORT_TEXT_VALUE_SCHEMA_VERSION = "evisoz_report_text_value_v1"
CHANNEL_SET_VALUE_SCHEMA_VERSION = "evisoz_channel_set_value_v1"
REGION_SET_VALUE_SCHEMA_VERSION = "evisoz_region_set_value_v1"
CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION = "evisoz_categorical_label_value_v1"

_CANONICAL_REGIONS = {
    "left_frontal",
    "right_frontal",
    "left_temporal",
    "right_temporal",
    "central_parietal",
}

_DIRECT_AUTHORITIES = {"physician", "dataset_direct"}
_CANDIDATE_AUTHORITIES = {"teacher_programmatic", "signal_derived"}
_SCOPE_DIRECT_ROLES = {
    "atomic_event": {"morphology", "quality"},
    "ictal_findings": {"morphology", "evolution", "quality"},
    "soz_localization": {
        "node_label",
        "region_label",
        "laterality_label",
        "localizability",
        "quality",
    },
    "full_soz": {
        "node_label",
        "region_label",
        "laterality_label",
        "morphology",
        "spread",
        "evolution",
        "quality",
        "localizability",
    },
    "nonlocalizing": {"morphology", "evolution", "quality", "localizability"},
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a stable identifier")
    return value


def _field_path(value: object, context: str) -> str:
    if not isinstance(value, str) or _PATH_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a dotted lowercase field path")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _loss_permissions(value: object, context: str) -> dict[str, bool]:
    if type(value) is not dict or set(value) != set(LOSS_PORTS):
        raise ValueError(f"{context} fields drifted")
    if any(type(item) is not bool for item in value.values()):
        raise TypeError(f"{context} must be boolean")
    return {key: bool(value[key]) for key in LOSS_PORTS}


def _normalize_dataset_capability(
    value: object,
    *,
    expected_dataset_id: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "dataset_id",
        "patient_roster_sha256",
        "field_roster",
    }:
        raise ValueError("dataset_capability fields drifted")
    dataset_id = _identifier(value["dataset_id"], "dataset_capability.dataset_id")
    if dataset_id != expected_dataset_id:
        raise ValueError("dataset_capability dataset_id drifted from the release")
    roster_sha256 = _sha256(
        value["patient_roster_sha256"],
        "dataset_capability.patient_roster_sha256",
    )
    raw_roster = value["field_roster"]
    if not isinstance(raw_roster, list) or not raw_roster:
        raise ValueError("dataset_capability.field_roster must be non-empty")
    roster: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_roster):
        context = f"dataset_capability.field_roster[{index}]"
        if type(raw) is not dict or set(raw) != {
            "field_id",
            "field_path",
            "semantic_role",
            "payload_schema_version",
            "allowed_roles",
            "loss_allowed",
            "report_target_allowed",
            "prompt_or_rag_allowed",
        }:
            raise ValueError(f"{context} fields drifted")
        field_id = _identifier(raw["field_id"], f"{context}.field_id")
        field_path = _field_path(raw["field_path"], f"{context}.field_path")
        role = raw["semantic_role"]
        if role not in SEMANTIC_ROLES:
            raise ValueError(f"{context}.semantic_role is unsupported")
        payload_schema_version = _identifier(
            raw["payload_schema_version"],
            f"{context}.payload_schema_version",
        )
        raw_roles = raw["allowed_roles"]
        if not isinstance(raw_roles, list) or not raw_roles:
            raise ValueError(f"{context}.allowed_roles must be non-empty")
        if any(role_name not in EVISOZ_ROLES for role_name in raw_roles):
            raise ValueError(f"{context}.allowed_roles contains an unsupported role")
        allowed_roles = sorted(set(raw_roles))
        if len(allowed_roles) != len(raw_roles):
            raise ValueError(f"{context}.allowed_roles must be unique")
        loss_allowed = _loss_permissions(
            raw["loss_allowed"],
            f"{context}.loss_allowed",
        )
        if any(loss_allowed.values()) and "development_cv" not in allowed_roles:
            raise ValueError(
                f"{context} cannot authorize training loss outside development_cv"
            )
        report_allowed = raw["report_target_allowed"]
        prompt_allowed = raw["prompt_or_rag_allowed"]
        if type(report_allowed) is not bool or type(prompt_allowed) is not bool:
            raise TypeError(f"{context} report/prompt permissions must be boolean")
        roster.append(
            {
                "field_id": field_id,
                "field_path": field_path,
                "semantic_role": role,
                "payload_schema_version": payload_schema_version,
                "allowed_roles": allowed_roles,
                "loss_allowed": loss_allowed,
                "report_target_allowed": report_allowed,
                "prompt_or_rag_allowed": prompt_allowed,
            }
        )
    if roster != sorted(roster, key=lambda row: row["field_id"]):
        raise ValueError("dataset capability field roster must be canonically sorted")
    if len({row["field_id"] for row in roster}) != len(roster) or len(
        {row["field_path"] for row in roster}
    ) != len(roster):
        raise ValueError("dataset capability field IDs/paths must be unique")
    return {
        "dataset_id": dataset_id,
        "patient_roster_sha256": roster_sha256,
        "field_roster": roster,
    }


def _validate_typed_value_payload(
    value: object,
    *,
    payload_schema_version: str,
    semantic_role: str,
    context: str,
) -> object:
    if type(value) is not dict:
        raise TypeError(f"{context} must be a typed JSON object")
    payload = deepcopy(value)
    if payload_schema_version == NODE_LABEL_VALUE_SCHEMA_VERSION:
        if semantic_role != "node_label" or set(payload) != {"values", "semantics"}:
            raise ValueError(f"{context} is not a closed node-label value")
        values = payload["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{context}.values must be a non-empty node array")
        if any(type(node) is not str or node not in STANDARD_19 for node in values):
            raise ValueError(
                f"{context}.values must contain only canonical Standard19 nodes"
            )
        if len(values) != len(set(values)):
            raise ValueError(f"{context}.values must be unique")
        if payload["semantics"] not in {
            "exhaustive",
            "incomplete_positive",
            "unknown",
        }:
            raise ValueError(f"{context}.semantics is unsupported")
        return payload
    if payload_schema_version == CANDIDATE_VALUE_SCHEMA_VERSION:
        if semantic_role not in {
            "region_label",
            "laterality_label",
            "morphology",
            "spread",
            "evolution",
            "quality",
            "localizability",
            "other",
        } or set(payload) != {"concept", "confidence"}:
            raise ValueError(f"{context} is not a closed candidate value")
        _identifier(payload["concept"], f"{context}.concept")
        confidence = payload["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(f"{context}.confidence must be finite in [0,1]")
        return payload
    if payload_schema_version == CHANNEL_SET_VALUE_SCHEMA_VERSION:
        if semantic_role != "spread" or set(payload) != {"values", "semantics"}:
            raise ValueError(f"{context} is not a closed channel-set value")
        values = payload["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{context}.values must be a non-empty channel array")
        if any(type(node) is not str or node not in STANDARD_19 for node in values):
            raise ValueError(
                f"{context}.values must contain only canonical Standard19 nodes"
            )
        if values != sorted(set(values)):
            raise ValueError(f"{context}.values must be unique and canonically sorted")
        if payload["semantics"] not in {
            "exhaustive",
            "incomplete_positive",
            "unknown",
        }:
            raise ValueError(f"{context}.semantics is unsupported")
        return payload
    if payload_schema_version == REGION_SET_VALUE_SCHEMA_VERSION:
        if semantic_role != "region_label" or set(payload) != {"values", "semantics"}:
            raise ValueError(f"{context} is not a closed region-set value")
        values = payload["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{context}.values must be a non-empty region array")
        if any(type(region) is not str or region not in _CANONICAL_REGIONS for region in values):
            raise ValueError(f"{context}.values contains a non-canonical region")
        if values != sorted(set(values)):
            raise ValueError(f"{context}.values must be unique and canonically sorted")
        if payload["semantics"] not in {
            "exhaustive",
            "incomplete_positive",
            "unknown",
        }:
            raise ValueError(f"{context}.semantics is unsupported")
        return payload
    if payload_schema_version == CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION:
        if semantic_role not in {
            "laterality_label",
            "morphology",
            "evolution",
            "quality",
            "localizability",
            "spread",
            "other",
        } or set(payload) != {"value", "certainty"}:
            raise ValueError(f"{context} is not a closed categorical-label value")
        _identifier(payload["value"], f"{context}.value")
        if payload["certainty"] not in {"low", "medium", "high"}:
            raise ValueError(f"{context}.certainty is unsupported")
        return payload
    if payload_schema_version == REPORT_TEXT_VALUE_SCHEMA_VERSION:
        if semantic_role != "text" or set(payload) != {"text"}:
            raise ValueError(f"{context} is not a closed report-text value")
        text = payload["text"]
        if not isinstance(text, str) or not text or text != text.strip():
            raise ValueError(f"{context}.text must be a non-empty trimmed string")
        return payload
    raise ValueError(
        f"{context} uses an unregistered typed value schema: {payload_schema_version}"
    )


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["release_id"] = _PENDING_ID
    source["receipt_sha256"] = _HASH_PLACEHOLDER
    return source


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["receipt_sha256"] = _HASH_PLACEHOLDER
    return source


def _normalize_field(
    value: Mapping[str, object],
    *,
    report_scope: str,
    capability: Mapping[str, object],
    context: str,
    trusted_values_by_artifact_id: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    expected = {
        "field_id",
        "field_path",
        "state",
        "authority",
        "quality_tier",
        "semantic_role",
        "value_ref",
        "value_payload",
        "claim_permission",
        "loss_permissions",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{context} fields drifted")
    field_id = _identifier(value["field_id"], f"{context}.field_id")
    path = _field_path(value["field_path"], f"{context}.field_path")
    state = value["state"]
    authority = value["authority"]
    quality = value["quality_tier"]
    role = value["semantic_role"]
    claim = value["claim_permission"]
    if state not in FIELD_STATES or authority not in AUTHORITIES:
        raise ValueError(f"{context} state/authority is unsupported")
    if quality not in QUALITY_TIERS or role not in SEMANTIC_ROLES or claim not in CLAIM_PERMISSIONS:
        raise ValueError(f"{context} quality/semantic/claim value is unsupported")
    if not capability:
        raise ValueError(f"{context} is absent from the dataset field roster")
    if (
        capability["field_id"] != field_id
        or capability["field_path"] != path
        or capability["semantic_role"] != role
    ):
        raise ValueError(f"{context} identity is outside the dataset field roster")
    permissions = _loss_permissions(
        value["loss_permissions"],
        f"{context}.loss_permissions",
    )
    if any(
        permissions[port] and not capability["loss_allowed"][port]
        for port in LOSS_PORTS
    ):
        raise ValueError(f"{context} loss permission exceeds dataset capability")
    value_ref = (
        validate_artifact_ref(value["value_ref"])
        if value["value_ref"] is not None
        else None
    )
    value_payload = deepcopy(value["value_payload"])
    if state == "provided" and (value_ref is None or value_payload is None):
        raise ValueError(f"{context} provided field requires value_ref and value_payload")
    if value_ref is not None and (
        value_ref["artifact_kind"] != "field_value"
        or value_ref["content_hash"]["domain"] != "canonical_json_v1"
        or value_ref["payload_schema_version"] is None
    ):
        raise ValueError(
            f"{context} field value must be typed canonical JSON"
        )
    if value_ref is not None and (
        value_ref["payload_schema_version"]
        != capability["payload_schema_version"]
    ):
        raise ValueError(f"{context} value schema is outside the dataset field roster")
    if state != "provided":
        if (
            value_ref is not None
            or value_payload is not None
            or claim != "none"
            or any(permissions.values())
        ):
            raise ValueError(f"{context} unavailable field must disable values, claims and losses")
        if quality != "not_applicable":
            raise ValueError(f"{context} unavailable field must use not_applicable quality")
    else:
        verify_artifact_content(value_ref, value_payload)
        if trusted_values_by_artifact_id is not None:
            artifact_id = str(value_ref["artifact_id"])
            if artifact_id not in trusted_values_by_artifact_id:
                raise ValueError(f"{context} value is absent from the trusted values map")
            trusted_payload = deepcopy(trusted_values_by_artifact_id[artifact_id])
            verify_artifact_content(value_ref, trusted_payload)
            if trusted_payload != value_payload:
                raise ValueError(f"{context} embedded value differs from trusted content")
        value_payload = _validate_typed_value_payload(
            value_payload,
            payload_schema_version=str(value_ref["payload_schema_version"]),
            semantic_role=str(role),
            context=f"{context}.value_payload",
        )
        if quality == "not_applicable":
            raise ValueError(f"{context} provided field needs an applicable quality tier")
        if authority in _DIRECT_AUTHORITIES:
            if claim not in {"direct", "none"}:
                raise ValueError(f"{context} direct authority cannot emit candidate-only claims")
        elif authority in _CANDIDATE_AUTHORITIES:
            if claim not in {"candidate_only", "none"}:
                raise ValueError(f"{context} programmatic/derived authority cannot emit direct claims")
            if permissions["node_localization_loss"]:
                raise ValueError(f"{context} programmatic/derived evidence cannot supervise node localization")
        elif authority == "physician_authored_text":
            if role != "text" or claim != "none":
                raise ValueError(f"{context} physician-authored text is not a structured patient label")
            if permissions != {
                "typed_slot_loss": False,
                "node_localization_loss": False,
                "report_text_loss": True,
            }:
                raise ValueError(
                    f"{context} physician-authored text may only supervise report_text_loss"
                )
        elif authority == "generated_text":
            if role != "text" or claim != "none":
                raise ValueError(f"{context} generated text is not a patient fact")
            if permissions != {
                "typed_slot_loss": False,
                "node_localization_loss": False,
                "report_text_loss": True,
            }:
                raise ValueError(f"{context} generated text may only supervise report_text_loss")
        elif authority == "knowledge_rule":
            if role != "knowledge_rule" or claim != "none" or any(permissions.values()):
                raise ValueError(f"{context} knowledge rules cannot create patient labels or losses")
        if quality == "uncertain":
            if claim != "none" or any(permissions.values()):
                raise ValueError(
                    f"{context} uncertain field cannot supervise a released loss or claim"
                )
        if role == "other" and claim == "direct":
            raise ValueError(f"{context} semantic_role=other cannot emit a direct claim")
        if permissions["node_localization_loss"] and (
            authority not in _DIRECT_AUTHORITIES
            or role != "node_label"
            or report_scope not in {"soz_localization", "full_soz"}
        ):
            raise ValueError(f"{context} node localization permission exceeds field authority/scope")
        if permissions["report_text_loss"] and authority not in {
            "generated_text",
            "physician_authored_text",
        }:
            raise ValueError(
                f"{context} report_text_loss requires generated or physician-authored text authority"
            )
        if claim == "direct" and role not in _SCOPE_DIRECT_ROLES[report_scope]:
            raise ValueError(f"{context} direct claim exceeds report_scope")
        if (
            claim != "none" or permissions["report_text_loss"]
        ) and capability["report_target_allowed"] is not True:
            raise ValueError(f"{context} exceeds dataset report-target capability")
        if role in {"text", "knowledge_rule"} and permissions["typed_slot_loss"]:
            raise ValueError(f"{context} text/knowledge cannot supervise EEG typed slots")
        if (
            role == "node_label"
            and permissions["node_localization_loss"]
            and value_payload["semantics"] == "unknown"
        ):
            raise ValueError(f"{context} unknown positive-set semantics cannot supervise localization")
    return {
        "field_id": field_id,
        "field_path": path,
        "state": state,
        "authority": authority,
        "quality_tier": quality,
        "semantic_role": role,
        "value_ref": value_ref,
        "value_payload": value_payload,
        "claim_permission": claim,
        "loss_permissions": permissions,
    }


def _materialize_value_payload(
    value: Mapping[str, object],
    *,
    trusted_values_by_artifact_id: Mapping[str, object],
    context: str,
) -> dict[str, object]:
    expected = {
        "field_id",
        "field_path",
        "state",
        "authority",
        "quality_tier",
        "semantic_role",
        "value_ref",
        "claim_permission",
        "loss_permissions",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{context} fields drifted")
    result = deepcopy(dict(value))
    if value["value_ref"] is None:
        result["value_payload"] = None
        return result
    reference = validate_artifact_ref(value["value_ref"])
    artifact_id = str(reference["artifact_id"])
    if artifact_id not in trusted_values_by_artifact_id:
        raise ValueError(f"{context} value is absent from the trusted values map")
    payload = deepcopy(trusted_values_by_artifact_id[artifact_id])
    verify_artifact_content(reference, payload)
    result["value_payload"] = payload
    return result


def build_field_release(
    *,
    dataset_id: str,
    sample_id: str,
    report_scope: str,
    fields: Sequence[Mapping[str, object]],
    event_identity: Mapping[str, object],
    dataset_capability: Mapping[str, object],
    trusted_values_by_artifact_id: Mapping[str, object],
) -> dict[str, Any]:
    if report_scope not in REPORT_SCOPES:
        raise ValueError("unsupported report_scope")
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise TypeError("fields must be an array")
    if not isinstance(trusted_values_by_artifact_id, Mapping):
        raise TypeError("trusted_values_by_artifact_id must be a mapping")
    dataset = _identifier(dataset_id, "dataset_id")
    sample = _identifier(sample_id, "sample_id")
    identity = validate_event_identity(event_identity)
    if identity["dataset_id"] != dataset or identity["sample_id"] != sample:
        raise ValueError("field release and event identity dataset/sample drifted")
    event_identity_ref = build_json_artifact_ref(
        identity,
        artifact_kind="event_identity",
        payload_schema_version=EVENT_IDENTITY_SCHEMA_VERSION,
    )
    capability = _normalize_dataset_capability(
        dataset_capability,
        expected_dataset_id=dataset,
    )
    capability_by_id = {
        row["field_id"]: row for row in capability["field_roster"]
    }
    prepared = [
        _materialize_value_payload(
            field,
            trusted_values_by_artifact_id=trusted_values_by_artifact_id,
            context=f"fields[{index}]",
        )
        for index, field in enumerate(fields)
    ]
    normalized = [
        _normalize_field(
            field,
            report_scope=report_scope,
            capability=capability_by_id.get(field.get("field_id"), {}),
            context=f"fields[{index}]",
            trusted_values_by_artifact_id=trusted_values_by_artifact_id,
        )
        for index, field in enumerate(prepared)
    ]
    normalized.sort(key=lambda row: row["field_id"])
    if not normalized or len({row["field_id"] for row in normalized}) != len(normalized):
        raise ValueError("field release rows must be non-empty with unique field IDs")
    if len({row["field_path"] for row in normalized}) != len(normalized):
        raise ValueError("field release paths must be unique")
    if {row["field_id"] for row in normalized} != set(capability_by_id):
        raise ValueError("field release must materialize the complete dataset field roster")
    body: dict[str, Any] = {
        "schema_version": FIELD_RELEASE_SCHEMA_VERSION,
        "release_id": _PENDING_ID,
        "dataset_id": dataset,
        "sample_id": sample,
        "event_identity_ref": event_identity_ref,
        "dataset_capability": capability,
        "report_scope": report_scope,
        "fields": normalized,
        "generated_text_is_not_a_label": True,
        "knowledge_creates_patient_facts": False,
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["release_id"] = "EVISOZ-FIELDS-" + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_field_release(
        body,
        trusted_event_identity=identity,
        trusted_values_by_artifact_id=trusted_values_by_artifact_id,
    )


def validate_field_release(
    value: object,
    *,
    trusted_event_identity: Mapping[str, object] | None = None,
    trusted_values_by_artifact_id: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "release_id",
        "dataset_id",
        "sample_id",
        "event_identity_ref",
        "dataset_capability",
        "report_scope",
        "fields",
        "generated_text_is_not_a_label",
        "knowledge_creates_patient_facts",
        "receipt_sha256",
    }:
        raise ValueError("field release fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != FIELD_RELEASE_SCHEMA_VERSION:
        raise ValueError("field release schema_version drifted")
    dataset_id = _identifier(data["dataset_id"], "dataset_id")
    sample_id = _identifier(data["sample_id"], "sample_id")
    event_identity_ref = validate_artifact_ref(data["event_identity_ref"])
    if (
        event_identity_ref["artifact_kind"] != "event_identity"
        or event_identity_ref["content_hash"]["domain"] != "canonical_json_v1"
        or event_identity_ref["payload_schema_version"]
        != EVENT_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("field release event_identity_ref type drifted")
    if trusted_event_identity is not None:
        identity = validate_event_identity(trusted_event_identity)
        if identity["dataset_id"] != dataset_id or identity["sample_id"] != sample_id:
            raise ValueError("field release and trusted event identity drifted")
        expected_identity_ref = build_json_artifact_ref(
            identity,
            artifact_kind="event_identity",
            payload_schema_version=EVENT_IDENTITY_SCHEMA_VERSION,
        )
        if event_identity_ref != expected_identity_ref:
            raise ValueError("field release does not bind the trusted event identity")
    capability = _normalize_dataset_capability(
        data["dataset_capability"],
        expected_dataset_id=dataset_id,
    )
    if capability != data["dataset_capability"]:
        raise ValueError("field release dataset capability is not canonical")
    capability_by_id = {
        row["field_id"]: row for row in capability["field_roster"]
    }
    scope = data["report_scope"]
    if scope not in REPORT_SCOPES:
        raise ValueError("unsupported report_scope")
    fields = data["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError("field release rows must be non-empty")
    normalized = [
        _normalize_field(
            field,
            report_scope=scope,
            capability=capability_by_id.get(field.get("field_id"), {}),
            context=f"fields[{index}]",
            trusted_values_by_artifact_id=trusted_values_by_artifact_id,
        )
        for index, field in enumerate(fields)
    ]
    if normalized != sorted(normalized, key=lambda row: row["field_id"]):
        raise ValueError("field release rows must be canonically sorted")
    if len({row["field_id"] for row in normalized}) != len(normalized) or len(
        {row["field_path"] for row in normalized}
    ) != len(normalized):
        raise ValueError("field release IDs/paths must be unique")
    if {row["field_id"] for row in normalized} != set(capability_by_id):
        raise ValueError("field release does not cover the complete dataset field roster")
    if data["generated_text_is_not_a_label"] is not True or data["knowledge_creates_patient_facts"] is not False:
        raise ValueError("field release safety policy drifted")
    expected_id = "EVISOZ-FIELDS-" + canonical_json_sha256(_id_source(data))[:24]
    if data["release_id"] != expected_id:
        raise ValueError("field release_id does not bind its content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("field release receipt hash drifted")
    data["dataset_id"] = dataset_id
    data["sample_id"] = sample_id
    data["event_identity_ref"] = event_identity_ref
    data["dataset_capability"] = capability
    data["fields"] = normalized
    return data


__all__ = [
    "FIELD_RELEASE_SCHEMA_VERSION",
    "REPORT_SCOPES",
    "FIELD_STATES",
    "AUTHORITIES",
    "QUALITY_TIERS",
    "SEMANTIC_ROLES",
    "CLAIM_PERMISSIONS",
    "LOSS_PORTS",
    "NODE_LABEL_VALUE_SCHEMA_VERSION",
    "CANDIDATE_VALUE_SCHEMA_VERSION",
    "REPORT_TEXT_VALUE_SCHEMA_VERSION",
    "CHANNEL_SET_VALUE_SCHEMA_VERSION",
    "REGION_SET_VALUE_SCHEMA_VERSION",
    "CATEGORICAL_LABEL_VALUE_SCHEMA_VERSION",
    "build_field_release",
    "validate_field_release",
]
