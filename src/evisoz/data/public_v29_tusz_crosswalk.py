"""Privacy-safe public canonical-v29 to TUSZ identity crosswalk.

The public v29 cache and the TUSZ/DeepSOZ identity recovery artifacts use
different namespaces.  This module proves the patient-level join without
persisting the source patient identifiers or EDF paths.  The resulting
receipt is an audit aid only; it never grants a training permission.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
from typing import Any, Mapping, Sequence

from .artifact_ref import (
    build_json_artifact_ref,
    build_raw_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
)


PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION = (
    "evisoz_public_v29_tusz_crosswalk_v1"
)
IDENTITY_HASH_DOMAIN = "evisoz-public-v29-tusz-patient-v1"
LOCAL_IDENTITY_HASH_DOMAIN = "evisoz-public-v29-tusz-local-patient-v1"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_CROSSWALK_ID_PREFIX = "EVISOZ-XWALK-"
_ROW_ID_PREFIX = "EVISOZ-XWALK-ROW-"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity_hash(value: str, *, domain: str = IDENTITY_HASH_DOMAIN) -> str:
    return _sha256(domain.encode("ascii") + b"\x00" + value.encode("ascii"))


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["crosswalk_id"] = _PENDING_ID
    return result


def _row_id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["crosswalk_row_id"] = _PENDING_ID
    return result


def _require_ascii_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f"{context} must be a non-empty ASCII identifier")
    return value


def _source_ref(value: object, context: str) -> dict[str, Any]:
    return validate_artifact_ref(value)


def build_public_v29_tusz_crosswalk(
    *,
    v29_roster: Mapping[str, object],
    identity_rows: Sequence[Mapping[str, object]],
    v29_roster_ref: Mapping[str, object] | None = None,
    identity_crosswalk_ref: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a closed crosswalk from a v29 roster and identity-v2 rows.

    ``identity_rows`` is intentionally accepted as an in-memory authority so
    callers can read a source CSV inside a controlled process.  Only hashes,
    counts and split names are emitted.
    """

    if type(v29_roster) is not dict:
        raise TypeError("v29 roster must be an object")
    if v29_roster.get("schema_version") != "evisoz_v29_patient_identity_roster_projection_v1":
        raise ValueError("v29 roster schema drifted")
    roster = v29_roster.get("patients")
    if not isinstance(roster, list) or not roster:
        raise ValueError("v29 roster is empty")
    if not isinstance(identity_rows, Sequence) or isinstance(identity_rows, (str, bytes)):
        raise TypeError("identity rows must be a sequence")
    if v29_roster_ref is None or identity_crosswalk_ref is None:
        raise ValueError("source artifact references are required")
    _source_ref(v29_roster_ref, "v29 roster reference")
    _source_ref(identity_crosswalk_ref, "identity crosswalk reference")

    roster_by_patient: dict[str, Mapping[str, object]] = {}
    expected_indices: list[int] = []
    namespace: str | None = None
    for item in roster:
        if type(item) is not dict or set(item) != {
            "identity_namespace",
            "identity_sha256",
            "patient_id",
            "patient_index",
        }:
            raise ValueError("v29 roster row fields drifted")
        patient = _require_ascii_identifier(item["patient_id"], "v29 patient ID")
        if patient in roster_by_patient:
            raise ValueError("v29 roster contains duplicate patients")
        if not isinstance(item["identity_namespace"], str) or not item["identity_namespace"]:
            raise ValueError("v29 identity namespace is invalid")
        if namespace is None:
            namespace = item["identity_namespace"]
        elif namespace != item["identity_namespace"]:
            raise ValueError("v29 identity namespaces are mixed")
        identity_sha = item["identity_sha256"]
        if not isinstance(identity_sha, str) or len(identity_sha) != 64:
            raise ValueError("v29 identity hash is invalid")
        index = item["patient_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("v29 patient index is invalid")
        roster_by_patient[patient] = item
        expected_indices.append(index)
    if sorted(expected_indices) != list(range(len(roster_by_patient))):
        raise ValueError("v29 patient indices are not contiguous")

    rows_by_patient: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in identity_rows:
        if not isinstance(row, Mapping):
            raise TypeError("identity crosswalk row must be an object")
        patient = row.get("deepsoz_patient_id")
        if patient is None:
            patient = row.get("patient_id")
        if patient is None:
            raise ValueError("identity crosswalk row has no DeepSOZ patient ID")
        patient = _require_ascii_identifier(str(patient), "identity crosswalk patient ID")
        if patient in roster_by_patient:
            rows_by_patient[patient].append(row)

    output_rows: list[dict[str, Any]] = []
    unmatched = 0
    ambiguous = 0
    for patient, roster_item in roster_by_patient.items():
        source_rows = rows_by_patient.get(patient, [])
        statuses = {str(row.get("mapping_status", "")) for row in source_rows}
        unique_rows = [row for row in source_rows if row.get("mapping_status") == "unique"]
        local_patients = {
            str(row.get("local_patient_id"))
            for row in unique_rows
            if row.get("local_patient_id") not in (None, "")
        }
        if not unique_rows:
            if source_rows or statuses.intersection({"ambiguous", "unmapped"}):
                ambiguous += 1
                status = "ambiguous_or_unmapped"
            else:
                unmatched += 1
                status = "unmatched"
            local_patient_hash = None
            split_names: list[str] = []
            record_hashes: list[str] = []
        else:
            if len(local_patients) != 1:
                ambiguous += 1
                status = "ambiguous_local_patient"
            else:
                status = "unique"
            local_patient_hash = (
                _identity_hash(next(iter(local_patients)), domain=LOCAL_IDENTITY_HASH_DOMAIN)
                if len(local_patients) == 1
                else None
            )
            split_names = sorted(
                {
                    str(row.get("local_official_split") or row.get("source_official_split"))
                    for row in unique_rows
                    if row.get("local_official_split") or row.get("source_official_split")
                }
            )
            record_hashes = sorted(
                {
                    str(row.get("source_edf_container_sha256"))
                    for row in unique_rows
                    if isinstance(row.get("source_edf_container_sha256"), str)
                    and len(str(row.get("source_edf_container_sha256"))) == 64
                }
            )
            if not record_hashes:
                record_hashes = sorted(
                    {
                        _sha256(str(row.get("local_edf_path")).encode("utf-8"))
                        for row in unique_rows
                        if row.get("local_edf_path")
                    }
                )
        row_body: dict[str, Any] = {
            "crosswalk_row_id": _PENDING_ID,
            "v29_identity_namespace": str(roster_item["identity_namespace"]),
            "v29_identity_sha256": str(roster_item["identity_sha256"]),
            "tusz_identity_namespace": "deepsoz_tusz_identity_v2",
            "tusz_identity_sha256": _identity_hash(patient),
            "tusz_local_patient_identity_sha256": local_patient_hash,
            "official_splits": split_names,
            "mapped_record_count": len(unique_rows),
            "unique_container_count": len(record_hashes),
            "record_roster_sha256": canonical_json_sha256(record_hashes),
            "mapping_status": status,
            "identity_match_method": "exact_numeric_patient_id_via_identity_v2",
            "raw_patient_identifiers_stored": False,
        }
        row_body["crosswalk_row_id"] = _ROW_ID_PREFIX + canonical_json_sha256(
            _row_id_source(row_body)
        )[:24]
        output_rows.append(row_body)

    output_rows.sort(key=lambda row: row["v29_identity_sha256"])
    matched = sum(row["mapping_status"] == "unique" for row in output_rows)
    body: dict[str, Any] = {
        "schema_version": PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION,
        "crosswalk_id": _PENDING_ID,
        "status": (
            "complete_audit_only_training_disabled"
            if matched == len(output_rows)
            else "incomplete_audit_only_training_disabled"
        ),
        "identity_hash_domain": IDENTITY_HASH_DOMAIN,
        "local_identity_hash_domain": LOCAL_IDENTITY_HASH_DOMAIN,
        "v29_roster_ref": deepcopy(dict(v29_roster_ref)),
        "identity_crosswalk_ref": deepcopy(dict(identity_crosswalk_ref)),
        "rows": output_rows,
        "counts": {
            "v29_patient_count": len(output_rows),
            "matched_patient_count": matched,
            "unmatched_patient_count": unmatched,
            "ambiguous_patient_count": ambiguous,
            "mapped_record_count": sum(row["mapped_record_count"] for row in output_rows),
        },
        "permissions": {
            "crosswalk_for_leakage_audit": True,
            "crosswalk_for_split_assignment": False,
            "training_authorized": False,
            "task_label_training_authorized": False,
            "raw_patient_identifiers_stored": False,
            "can_create_patient_fact": False,
        },
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["crosswalk_id"] = _CROSSWALK_ID_PREFIX + canonical_json_sha256(
        _id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_public_v29_tusz_crosswalk(body)


def validate_public_v29_tusz_crosswalk(value: object) -> dict[str, Any]:
    """Validate a privacy-safe crosswalk and its content-addressed rows."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "crosswalk_id",
        "status",
        "identity_hash_domain",
        "local_identity_hash_domain",
        "v29_roster_ref",
        "identity_crosswalk_ref",
        "rows",
        "counts",
        "permissions",
        "receipt_sha256",
    }:
        raise ValueError("public v29/TUSZ crosswalk fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION:
        raise ValueError("public v29/TUSZ crosswalk schema drifted")
    if data["identity_hash_domain"] != IDENTITY_HASH_DOMAIN or data[
        "local_identity_hash_domain"
    ] != LOCAL_IDENTITY_HASH_DOMAIN:
        raise ValueError("public v29/TUSZ crosswalk hash domain drifted")
    _source_ref(data["v29_roster_ref"], "v29 roster reference")
    _source_ref(data["identity_crosswalk_ref"], "identity crosswalk reference")
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("public v29/TUSZ crosswalk rows are empty")
    previous = ""
    matched = 0
    unmatched = 0
    ambiguous = 0
    records = 0
    seen_ids: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {
            "crosswalk_row_id",
            "v29_identity_namespace",
            "v29_identity_sha256",
            "tusz_identity_namespace",
            "tusz_identity_sha256",
            "tusz_local_patient_identity_sha256",
            "official_splits",
            "mapped_record_count",
            "unique_container_count",
            "record_roster_sha256",
            "mapping_status",
            "identity_match_method",
            "raw_patient_identifiers_stored",
        }:
            raise ValueError("public v29/TUSZ crosswalk row fields drifted")
        row_id = row["crosswalk_row_id"]
        if not isinstance(row_id, str) or not row_id.startswith(_ROW_ID_PREFIX):
            raise ValueError("public v29/TUSZ crosswalk row ID is invalid")
        if row_id in seen_ids:
            raise ValueError("public v29/TUSZ crosswalk row ID duplicated")
        seen_ids.add(row_id)
        v29_hash = row["v29_identity_sha256"]
        tusz_hash = row["tusz_identity_sha256"]
        if (
            not isinstance(v29_hash, str)
            or len(v29_hash) != 64
            or not isinstance(tusz_hash, str)
            or len(tusz_hash) != 64
        ):
            raise ValueError("public v29/TUSZ crosswalk identity hash is invalid")
        if v29_hash <= previous:
            raise ValueError("public v29/TUSZ crosswalk rows are not sorted")
        previous = v29_hash
        local_hash = row["tusz_local_patient_identity_sha256"]
        if local_hash is not None and (not isinstance(local_hash, str) or len(local_hash) != 64):
            raise ValueError("public v29/TUSZ local identity hash is invalid")
        splits = row["official_splits"]
        if not isinstance(splits, list) or len(splits) != len(set(splits)) or any(
            not isinstance(item, str) or not item for item in splits
        ):
            raise ValueError("public v29/TUSZ official split roster is invalid")
        for field in ("mapped_record_count", "unique_container_count"):
            number = row[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError("public v29/TUSZ crosswalk count is invalid")
        if row["unique_container_count"] > row["mapped_record_count"]:
            raise ValueError("public v29/TUSZ container count exceeds record count")
        if not isinstance(row["record_roster_sha256"], str) or len(row["record_roster_sha256"]) != 64:
            raise ValueError("public v29/TUSZ record roster hash is invalid")
        if row["mapping_status"] not in {"unique", "unmatched", "ambiguous_or_unmapped", "ambiguous_local_patient"}:
            raise ValueError("public v29/TUSZ mapping status is invalid")
        if row["identity_match_method"] != "exact_numeric_patient_id_via_identity_v2":
            raise ValueError("public v29/TUSZ identity match method drifted")
        if row["raw_patient_identifiers_stored"] is not False:
            raise ValueError("public v29/TUSZ crosswalk stores raw patient identifiers")
        expected_row_id = _ROW_ID_PREFIX + canonical_json_sha256(
            _row_id_source(row)
        )[:24]
        if row_id != expected_row_id:
            raise ValueError("public v29/TUSZ crosswalk row ID drifted")
        if row["mapping_status"] == "unique":
            if local_hash is None or not splits or row["mapped_record_count"] <= 0:
                raise ValueError("unique crosswalk row lacks mapped identity evidence")
            matched += 1
        elif row["mapping_status"] == "unmatched":
            if local_hash is not None or splits or row["mapped_record_count"] != 0:
                raise ValueError("unmatched crosswalk row contains mapped evidence")
            unmatched += 1
        else:
            ambiguous += 1
        records += row["mapped_record_count"]
    counts = data["counts"]
    if type(counts) is not dict or counts != {
        "v29_patient_count": len(rows),
        "matched_patient_count": matched,
        "unmatched_patient_count": unmatched,
        "ambiguous_patient_count": ambiguous,
        "mapped_record_count": records,
    }:
        raise ValueError("public v29/TUSZ crosswalk counts drifted")
    expected_status = (
        "complete_audit_only_training_disabled"
        if matched == len(rows)
        else "incomplete_audit_only_training_disabled"
    )
    if data["status"] != expected_status:
        raise ValueError("public v29/TUSZ crosswalk status drifted")
    if data["permissions"] != {
        "crosswalk_for_leakage_audit": True,
        "crosswalk_for_split_assignment": False,
        "training_authorized": False,
        "task_label_training_authorized": False,
        "raw_patient_identifiers_stored": False,
        "can_create_patient_fact": False,
    }:
        raise ValueError("public v29/TUSZ crosswalk permissions drifted")
    expected_id = _CROSSWALK_ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["crosswalk_id"] != expected_id:
        raise ValueError("public v29/TUSZ crosswalk ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("public v29/TUSZ crosswalk receipt drifted")
    return data


def build_raw_source_refs(*, v29_roster_payload: object, identity_crosswalk_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build source refs without retaining source paths in the receipt."""

    if not isinstance(identity_crosswalk_bytes, bytes):
        raise TypeError("identity crosswalk source must be bytes")
    return (
        build_json_artifact_ref(
            v29_roster_payload,
            artifact_kind="v29_patient_identity_roster",
            payload_schema_version="evisoz_v29_patient_identity_roster_projection_v1",
        ),
        build_raw_artifact_ref(
            identity_crosswalk_bytes,
            artifact_kind="deepsoz_tusz_identity_v2_crosswalk",
            media_type="text/csv",
        ),
    )


__all__ = [
    "PUBLIC_V29_TUSZ_CROSSWALK_SCHEMA_VERSION",
    "IDENTITY_HASH_DOMAIN",
    "LOCAL_IDENTITY_HASH_DOMAIN",
    "build_public_v29_tusz_crosswalk",
    "build_raw_source_refs",
    "validate_public_v29_tusz_crosswalk",
]
