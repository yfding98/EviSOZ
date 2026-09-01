"""Fail-closed quarantine receipts for private reports with no safe linkage.

An exclusion receipt is deliberately different from a linkage or a release
receipt.  It binds the exact report bytes to an operational decision that the
report must not create a patient linkage, split assignment, preprocessing row,
or text release.  It contains no source path, filename, patient name, or raw
report text.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from .private_physician_reports import (
    PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    validate_private_physician_report_inventory,
)


PRIVATE_REPORT_EXCLUSION_SCHEMA_VERSION = "evisoz_private_report_exclusion_v1"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-REPORT-EXCL-"
_REPORT_ID_RE = re.compile(r"^EVISOZ-PRPT-[0-9a-f]{24}$")
_OPERATOR_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["exclusion_id"] = _PENDING_ID
    return body


def build_private_report_exclusion(
    *,
    report_inventory: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    operator: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Build a content-addressed operational exclusion receipt.

    ``entries`` must reference only currently unresolved inventory rows.  The
    function intentionally does not accept a linkage group or split assignment.
    """

    inventory = validate_private_physician_report_inventory(report_inventory)
    if not isinstance(operator, str) or _OPERATOR_RE.fullmatch(operator) is None:
        raise ValueError("exclusion operator must be a short non-empty identifier")
    if not isinstance(recorded_at_utc, str) or not recorded_at_utc.strip():
        raise ValueError("exclusion recorded_at_utc must be a non-empty string")
    unresolved = {
        str(row["report_id"]): row
        for row in inventory["reports"]
        if row["association"]["status"] == "unresolved"
    }
    if not entries:
        raise ValueError("private report exclusion entries must be non-empty")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in entries:
        if type(item) is not dict or set(item) != {
            "report_id",
            "document_ref",
            "exclusion_status",
            "exclusion_code",
            "downstream_policy",
        }:
            raise ValueError("private report exclusion entry fields drifted")
        report_id = item["report_id"]
        if not isinstance(report_id, str) or _REPORT_ID_RE.fullmatch(report_id) is None:
            raise ValueError("private report exclusion report ID is invalid")
        if report_id in seen:
            raise ValueError("private report exclusion report IDs are duplicated")
        seen.add(report_id)
        source = unresolved.get(report_id)
        if source is None:
            raise ValueError("only unresolved reports may be excluded")
        ref = validate_artifact_ref(item["document_ref"])
        if ref != source["document_ref"]:
            raise ValueError("private report exclusion document reference drifted")
        status = item["exclusion_status"]
        code = item["exclusion_code"]
        if status != "excluded_unresolved":
            raise ValueError("private report exclusion status drifted")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,120}", code):
            raise ValueError("private report exclusion code is invalid")
        policy = item["downstream_policy"]
        expected_policy = {
            "create_linkage": False,
            "create_split_assignment": False,
            "admit_to_signal_preprocessing": False,
            "admit_to_event_training": False,
            "admit_to_qwen_training": False,
            "admit_to_language_evaluation": False,
        }
        if policy != expected_policy:
            raise ValueError("private report exclusion downstream policy drifted")
        normalized.append(
            {
                "report_id": report_id,
                "document_ref": ref,
                "exclusion_status": status,
                "exclusion_code": code,
                "downstream_policy": expected_policy,
            }
        )
    normalized.sort(key=lambda row: str(row["report_id"]))
    body: dict[str, Any] = {
        "schema_version": PRIVATE_REPORT_EXCLUSION_SCHEMA_VERSION,
        "exclusion_id": _PENDING_ID,
        "inventory_ref": build_json_artifact_ref(
            inventory,
            artifact_kind="physician_report_inventory",
            payload_schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
        ),
        "entries": normalized,
        "counts": {
            "excluded_count": len(normalized),
            "unresolved_inventory_count": len(unresolved),
        },
        "decision": {
            "decision_type": "operational_quarantine",
            "operator": operator,
            "recorded_at_utc": recorded_at_utc,
            "institutional_training_authorization": False,
        },
        "policy": {
            "report_is_not_linked": True,
            "report_is_not_split": True,
            "report_is_not_preprocessed": True,
            "report_is_not_released": True,
            "raw_identifiers_stored": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["exclusion_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_report_exclusion(body, report_inventory=inventory)


def validate_private_report_exclusion(
    value: object,
    *,
    report_inventory: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "exclusion_id",
        "inventory_ref",
        "entries",
        "counts",
        "decision",
        "policy",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("private report exclusion fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PRIVATE_REPORT_EXCLUSION_SCHEMA_VERSION:
        raise ValueError("private report exclusion schema drifted")
    inventory_ref = validate_artifact_ref(data["inventory_ref"])
    if (
        inventory_ref["artifact_kind"] != "physician_report_inventory"
        or inventory_ref["payload_schema_version"] != PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION
        or inventory_ref["content_hash"]["domain"] != "canonical_json_v1"
    ):
        raise ValueError("private report exclusion inventory reference drifted")
    entries = data["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("private report exclusion entries must be non-empty")
    if entries != sorted(entries, key=lambda row: row["report_id"]):
        raise ValueError("private report exclusion entries are not sorted")
    if len({row.get("report_id") for row in entries}) != len(entries):
        raise ValueError("private report exclusion entries are not unique")
    for item in entries:
        if type(item) is not dict or set(item) != {
            "report_id", "document_ref", "exclusion_status", "exclusion_code", "downstream_policy"
        }:
            raise ValueError("private report exclusion entry fields drifted")
        if not isinstance(item["report_id"], str) or _REPORT_ID_RE.fullmatch(item["report_id"]) is None:
            raise ValueError("private report exclusion report ID is invalid")
        ref = validate_artifact_ref(item["document_ref"])
        if ref["artifact_kind"] != "physician_authored_report_source":
            raise ValueError("private report exclusion document reference kind drifted")
        if item["exclusion_status"] != "excluded_unresolved":
            raise ValueError("private report exclusion status drifted")
        if not isinstance(item["exclusion_code"], str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,120}", item["exclusion_code"]):
            raise ValueError("private report exclusion code is invalid")
        if item["downstream_policy"] != {
            "create_linkage": False,
            "create_split_assignment": False,
            "admit_to_signal_preprocessing": False,
            "admit_to_event_training": False,
            "admit_to_qwen_training": False,
            "admit_to_language_evaluation": False,
        }:
            raise ValueError("private report exclusion policy drifted")
    counts = data["counts"]
    if type(counts) is not dict or set(counts) != {"excluded_count", "unresolved_inventory_count"}:
        raise ValueError("private report exclusion counts drifted")
    if counts["excluded_count"] != len(entries) or not isinstance(counts["unresolved_inventory_count"], int) or counts["unresolved_inventory_count"] < len(entries):
        raise ValueError("private report exclusion counts are invalid")
    decision = data["decision"]
    if type(decision) is not dict or set(decision) != {"decision_type", "operator", "recorded_at_utc", "institutional_training_authorization"}:
        raise ValueError("private report exclusion decision fields drifted")
    if decision["decision_type"] != "operational_quarantine" or not isinstance(decision["operator"], str) or _OPERATOR_RE.fullmatch(decision["operator"]) is None or not isinstance(decision["recorded_at_utc"], str) or not decision["recorded_at_utc"].strip() or decision["institutional_training_authorization"] is not False:
        raise ValueError("private report exclusion decision is invalid")
    if data["policy"] != {
        "report_is_not_linked": True,
        "report_is_not_split": True,
        "report_is_not_preprocessed": True,
        "report_is_not_released": True,
        "raw_identifiers_stored": False,
    }:
        raise ValueError("private report exclusion policy drifted")
    if not isinstance(data["exclusion_id"], str) or not data["exclusion_id"].startswith(_ID_PREFIX):
        raise ValueError("private report exclusion ID drifted")
    if data["exclusion_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("private report exclusion ID does not bind content")
    if not isinstance(data["receipt_sha256"], str) or _SHA256_RE.fullmatch(data["receipt_sha256"]) is None or data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("private report exclusion receipt drifted")
    if report_inventory is not None:
        inventory = validate_private_physician_report_inventory(report_inventory)
        expected_ref = build_json_artifact_ref(
            inventory,
            artifact_kind="physician_report_inventory",
            payload_schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
        )
        if data["inventory_ref"] != expected_ref:
            raise ValueError("private report exclusion inventory binding drifted")
        unresolved = {
            row["report_id"]: row["document_ref"]
            for row in inventory["reports"]
            if row["association"]["status"] == "unresolved"
        }
        excluded = {row["report_id"]: row["document_ref"] for row in entries}
        if any(report_id not in unresolved for report_id in excluded):
            raise ValueError("private report exclusion contains a resolved report")
        if any(unresolved[report_id] != ref for report_id, ref in excluded.items()):
            raise ValueError("private report exclusion document does not match inventory")
        if counts["unresolved_inventory_count"] != len(unresolved):
            raise ValueError("private report exclusion unresolved count drifted")
    return data


__all__ = [
    "PRIVATE_REPORT_EXCLUSION_SCHEMA_VERSION",
    "build_private_report_exclusion",
    "validate_private_report_exclusion",
]
