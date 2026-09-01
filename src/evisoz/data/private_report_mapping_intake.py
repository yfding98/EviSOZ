"""Privacy-safe intake for authoritative physician-report linkage closure.

The intake is deliberately a request artifact, not a guessed mapping.  It
contains only content-addressed report references and empty mapping slots so
an authorized reviewer can complete the association in a separate controlled
workflow without copying names, paths, or report text into the repository.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from .artifact_ref import (
    build_json_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)
from .private_physician_reports import (
    PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    validate_private_physician_report_inventory,
)


PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION = (
    "evisoz_private_report_mapping_intake_v1"
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-REPORT-MAP-"
_REPORT_ID_RE = re.compile(r"^EVISOZ-PRPT-[0-9a-f]{24}$")


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    body = deepcopy(dict(value))
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    body = _hash_source(value)
    body["intake_id"] = _PENDING_ID
    return body


def build_private_report_mapping_intake(
    *,
    report_inventory: Mapping[str, object],
    split_roster: Mapping[str, object],
) -> dict[str, Any]:
    """Build an immutable request list for unresolved report associations."""

    inventory = validate_private_physician_report_inventory(report_inventory)
    if type(split_roster) is not dict:
        raise TypeError("split roster must be an object")
    unresolved = [
        row for row in inventory["reports"]
        if row["association"]["status"] == "unresolved"
    ]
    if not unresolved:
        raise ValueError("report mapping intake has no unresolved reports")
    inventory_ref = build_json_artifact_ref(
        inventory,
        artifact_kind="physician_report_inventory",
        payload_schema_version=PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION,
    )
    split_roster_ref = build_json_artifact_ref(
        split_roster,
        artifact_kind="split_roster",
        payload_schema_version="evisoz_split_roster_v1",
    )
    requests = [
        {
            "report_id": str(row["report_id"]),
            "document_ref": deepcopy(row["document_ref"]),
            "requested_action": "provide_authoritative_linkage_or_exclude",
            "proposed_linkage_group_id": None,
            "authority_attestation": None,
        }
        for row in unresolved
    ]
    requests.sort(key=lambda row: row["report_id"])
    body: dict[str, Any] = {
        "schema_version": PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION,
        "intake_id": _PENDING_ID,
        "status": "awaiting_authoritative_mapping",
        "inventory_ref": inventory_ref,
        "split_roster_ref": split_roster_ref,
        "requests": requests,
        "counts": {
            "request_count": len(requests),
            "unresolved_count": len(requests),
            "resolved_count": 0,
        },
        "permissions": {
            "mapping_authority_required": True,
            "training_authorized": False,
            "report_text_release_authorized": False,
            "raw_patient_identifiers_stored": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["intake_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_private_report_mapping_intake(body)


def validate_private_report_mapping_intake(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "intake_id",
        "status",
        "inventory_ref",
        "split_roster_ref",
        "requests",
        "counts",
        "permissions",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("private report mapping intake fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION:
        raise ValueError("private report mapping intake schema drifted")
    if data["status"] != "awaiting_authoritative_mapping":
        raise ValueError("private report mapping intake status drifted")
    inventory_ref = validate_artifact_ref(data["inventory_ref"])
    if (
        inventory_ref["artifact_kind"] != "physician_report_inventory"
        or inventory_ref["payload_schema_version"]
        != PHYSICIAN_REPORT_INVENTORY_SCHEMA_VERSION
    ):
        raise ValueError("private report mapping inventory reference drifted")
    split_ref = validate_artifact_ref(data["split_roster_ref"])
    if (
        split_ref["artifact_kind"] != "split_roster"
        or split_ref["payload_schema_version"] != "evisoz_split_roster_v1"
    ):
        raise ValueError("private report mapping split reference drifted")
    requests = data["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValueError("private report mapping requests must be non-empty")
    ids: list[str] = []
    for index, row in enumerate(requests):
        if type(row) is not dict or set(row) != {
            "report_id",
            "document_ref",
            "requested_action",
            "proposed_linkage_group_id",
            "authority_attestation",
        }:
            raise ValueError(f"private report mapping request {index} fields drifted")
        report_id = row["report_id"]
        if not isinstance(report_id, str) or _REPORT_ID_RE.fullmatch(report_id) is None:
            raise ValueError("private report mapping report ID drifted")
        ids.append(report_id)
        document_ref = validate_artifact_ref(row["document_ref"])
        if document_ref["artifact_kind"] != "physician_authored_report_source":
            raise ValueError("private report mapping document reference drifted")
        if row["requested_action"] != "provide_authoritative_linkage_or_exclude":
            raise ValueError("private report mapping requested action drifted")
        if row["proposed_linkage_group_id"] is not None:
            raise ValueError("private report mapping contains a guessed linkage")
        if row["authority_attestation"] is not None:
            raise ValueError("private report mapping contains an unverified attestation")
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("private report mapping requests are not sorted/unique")
    counts = data["counts"]
    if counts != {
        "request_count": len(requests),
        "unresolved_count": len(requests),
        "resolved_count": 0,
    }:
        raise ValueError("private report mapping counts drifted")
    if data["permissions"] != {
        "mapping_authority_required": True,
        "training_authorized": False,
        "report_text_release_authorized": False,
        "raw_patient_identifiers_stored": False,
    }:
        raise ValueError("private report mapping permissions drifted")
    if not isinstance(data["intake_id"], str) or not data["intake_id"].startswith(_ID_PREFIX):
        raise ValueError("private report mapping intake ID drifted")
    if data["intake_id"] != _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]:
        raise ValueError("private report mapping intake ID does not bind content")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("private report mapping intake receipt drifted")
    return data


__all__ = [
    "PRIVATE_REPORT_MAPPING_INTAKE_SCHEMA_VERSION",
    "build_private_report_mapping_intake",
    "validate_private_report_mapping_intake",
]
