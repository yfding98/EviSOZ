"""Identity-only adapter from the complete TUSZ roster to EventNet.

The adapter consumes an already validated ``tusz_complete_detector_roster_v1``
object.  It does not accept a TUSZ root or a reference-sidecar path and cannot
open ``csv_bi`` contents.  The exported rows contain only split, patient,
recording-path and EDF-container identity.  Official evaluation identities are
retained in the projection, but EventNet execution remains admission-gated.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Final, Mapping, Sequence

from .tusz_complete_detector_roster_v1 import (
    TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION,
    validate_tusz_complete_detector_roster_v1,
)


EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_SCHEMA_VERSION = (
    "eventnet_tusz_complete_identity_projection_v1"
)
EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_METHOD_ID = (
    "validated_complete_roster_reference_free_identity_only_export_v1"
)
EVENTNET_TUSZ_COMPLETE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "model_split",
    "official_split",
    "local_patient_id",
    "local_edf_path",
    "source_edf_container_sha256",
)

_OFFICIAL_TO_MODEL_SPLIT: Final[dict[str, str]] = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_MODEL_TO_OFFICIAL_SPLIT: Final[dict[str, str]] = {
    value: key for key, value in _OFFICIAL_TO_MODEL_SPLIT.items()
}
_HEX = frozenset("0123456789abcdef")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed string")
    if len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _split_permissions() -> dict[str, dict[str, bool]]:
    return {
        split: {
            "identity_export_authorized": True,
            "eventnet_model_execution_authorized": split != "source_eval",
            "eventnet_model_execution_admission_required": split == "source_eval",
            "host_admission_receipt_present": False,
            "official_reference_access_authorized": False,
        }
        for split in sorted(_MODEL_TO_OFFICIAL_SPLIT)
    }


def _project_records(
    roster_records: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    records = [
        {
            "model_split": str(row["benchmark_split"]),
            "official_split": str(row["official_split"]),
            "local_patient_id": str(row["patient_id"]),
            "local_edf_path": str(row["recording_id"]),
            "source_edf_container_sha256": str(row["container_sha256"]),
        }
        for row in roster_records
    ]
    records.sort(
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        )
    )
    return records


def _summarize_projection(
    records: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for model_split in sorted(_MODEL_TO_OFFICIAL_SPLIT):
        selected = [row for row in records if row["model_split"] == model_split]
        patients = sorted({row["local_patient_id"] for row in selected})
        recording_ids = sorted(row["local_edf_path"] for row in selected)
        bindings = sorted(
            [
                [row["local_edf_path"], row["source_edf_container_sha256"]]
                for row in selected
            ]
        )
        summaries[model_split] = {
            "official_split": _MODEL_TO_OFFICIAL_SPLIT[model_split],
            "patient_count": len(patients),
            "recording_count": len(selected),
            "patient_roster_sha256": _canonical_sha256(patients),
            "recording_roster_sha256": _canonical_sha256(recording_ids),
            "container_binding_roster_sha256": _canonical_sha256(bindings),
        }
    return summaries


def _projection_from_validated_roster(roster: Mapping[str, Any]) -> dict[str, Any]:
    records = _project_records(roster["records"])
    summaries = _summarize_projection(records)
    observed = roster["observed_inventory"]
    source_split_bindings = {
        _OFFICIAL_TO_MODEL_SPLIT[official_split]: {
            "patient_roster_sha256": row["patient_roster_sha256"],
            "recording_roster_sha256": row["recording_roster_sha256"],
            "container_binding_roster_sha256": row[
                "container_binding_roster_sha256"
            ],
        }
        for official_split, row in observed["split_summaries"].items()
    }
    body: dict[str, Any] = {
        "schema_version": (
            EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_SCHEMA_VERSION
        ),
        "method_id": EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_METHOD_ID,
        "projection_id": "EVENTNET-TUSZ-IDENTITY-PROJECTION-PENDING",
        "source_roster_binding": {
            "source_schema_version": roster["schema_version"],
            "source_roster_id": roster["roster_id"],
            "source_roster_receipt_sha256": roster["receipt_sha256"],
            "source_release_id": roster["expected_inventory"]["release_id"],
            "source_records_payload_sha256": observed["records_payload_sha256"],
            "source_split_bindings_sha256": _canonical_sha256(
                source_split_bindings
            ),
            "source_patient_count": observed["total_patient_count"],
            "source_recording_count": observed["total_recording_count"],
        },
        "projection_fields": list(EVENTNET_TUSZ_COMPLETE_IDENTITY_FIELDS),
        "records": records,
        "split_summaries": summaries,
        "reference_access_receipt": {
            "reference_path_argument_accepted": False,
            "reference_files_opened": 0,
            "csv_bi_files_opened": 0,
            "csv_bi_bytes_read": 0,
            "csv_bi_contents_read": False,
            "seizure_interval_or_label_values_read": False,
            "edf_annotations_read": False,
            "spreadsheet_or_clinical_text_read": False,
        },
        "split_permissions": _split_permissions(),
        "scope_receipt": {
            "complete_train_dev_eval_identity_inventory_projected": True,
            "identity_only_projection": True,
            "edf_container_hash_used_for_input_integrity_only": True,
            "identity_fields_authorized_as_model_features": False,
            "source_eval_identity_export_authorized": True,
            "source_eval_model_execution_requires_host_admission": True,
            "source_eval_model_execution_authorized_by_this_receipt": False,
            "source_eval_reference_join_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["projection_id"] = "EVNTUSZID-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _validate_projection_record(value: object, index: int) -> dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != set(EVENTNET_TUSZ_COMPLETE_IDENTITY_FIELDS)
    ):
        raise ValueError(f"EventNet identity record {index} fields drifted")
    row = deepcopy(value)
    model_split = row["model_split"]
    official_split = row["official_split"]
    if (
        model_split not in _MODEL_TO_OFFICIAL_SPLIT
        or official_split != _MODEL_TO_OFFICIAL_SPLIT[model_split]
    ):
        raise ValueError("EventNet identity split mapping drifted")
    patient_id = _identifier(row["local_patient_id"], "local patient ID")
    recording_id = _identifier(row["local_edf_path"], "local EDF path")
    path = PurePosixPath(recording_id)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in recording_id
        or len(path.parts) < 3
        or path.parts[0] != official_split
        or path.parts[1] != patient_id
        or path.suffix.lower() != ".edf"
    ):
        raise ValueError("EventNet identity split/patient/path binding drifted")
    _sha256(row["source_edf_container_sha256"], "source EDF container SHA-256")
    return row


def validate_eventnet_tusz_complete_identity_projection_v1(
    payload: object,
    *,
    source_roster: object | None = None,
) -> dict[str, Any]:
    """Validate an identity projection, optionally replaying its source roster."""

    required = {
        "schema_version",
        "method_id",
        "projection_id",
        "source_roster_binding",
        "projection_fields",
        "records",
        "split_summaries",
        "reference_access_receipt",
        "split_permissions",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet complete-roster identity projection fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_SCHEMA_VERSION
        or data["method_id"]
        != EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_METHOD_ID
    ):
        raise ValueError("EventNet identity projection schema/method drifted")

    binding_fields = {
        "source_schema_version",
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_release_id",
        "source_records_payload_sha256",
        "source_split_bindings_sha256",
        "source_patient_count",
        "source_recording_count",
    }
    binding = data["source_roster_binding"]
    if type(binding) is not dict or set(binding) != binding_fields:
        raise ValueError("EventNet identity source-roster binding fields drifted")
    if binding["source_schema_version"] != TUSZ_COMPLETE_DETECTOR_ROSTER_SCHEMA_VERSION:
        raise ValueError("EventNet identity source-roster schema drifted")
    _identifier(binding["source_roster_id"], "source roster ID")
    _identifier(binding["source_release_id"], "source release ID")
    for field in (
        "source_roster_receipt_sha256",
        "source_records_payload_sha256",
        "source_split_bindings_sha256",
    ):
        _sha256(binding[field], field)
    patient_count = _positive_integer(
        binding["source_patient_count"], "source patient count"
    )
    recording_count = _positive_integer(
        binding["source_recording_count"], "source recording count"
    )

    if data["projection_fields"] != list(EVENTNET_TUSZ_COMPLETE_IDENTITY_FIELDS):
        raise ValueError("EventNet identity projection field allowlist drifted")
    if type(data["records"]) is not list or not data["records"]:
        raise ValueError("EventNet identity projection has no records")
    records = [
        _validate_projection_record(row, index)
        for index, row in enumerate(data["records"])
    ]
    expected_order = sorted(
        records,
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        ),
    )
    if records != expected_order:
        raise ValueError("EventNet identity records are not canonically sorted")
    recording_ids = [row["local_edf_path"] for row in records]
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError("EventNet identity recording IDs are not unique")
    patient_splits: dict[str, str] = {}
    for row in records:
        previous = patient_splits.setdefault(
            row["local_patient_id"], row["model_split"]
        )
        if previous != row["model_split"]:
            raise ValueError("EventNet identity patient crosses source splits")
    if len(records) != recording_count or len(patient_splits) != patient_count:
        raise ValueError("EventNet identity projection totals drifted")

    summaries = _summarize_projection(records)
    if any(row["recording_count"] < 1 for row in summaries.values()):
        raise ValueError("EventNet identity projection must retain train/dev/eval")
    if data["split_summaries"] != summaries:
        raise ValueError("EventNet identity split summaries are not replayable")
    source_split_bindings = {
        split: {
            "patient_roster_sha256": row["patient_roster_sha256"],
            "recording_roster_sha256": row["recording_roster_sha256"],
            "container_binding_roster_sha256": row[
                "container_binding_roster_sha256"
            ],
        }
        for split, row in summaries.items()
    }
    if (
        _canonical_sha256(source_split_bindings)
        != binding["source_split_bindings_sha256"]
    ):
        raise ValueError("EventNet identity split/container binding drifted")

    expected_access = {
        "reference_path_argument_accepted": False,
        "reference_files_opened": 0,
        "csv_bi_files_opened": 0,
        "csv_bi_bytes_read": 0,
        "csv_bi_contents_read": False,
        "seizure_interval_or_label_values_read": False,
        "edf_annotations_read": False,
        "spreadsheet_or_clinical_text_read": False,
    }
    if data["reference_access_receipt"] != expected_access:
        raise ValueError("EventNet identity reference-access receipt drifted")
    if data["split_permissions"] != _split_permissions():
        raise ValueError("EventNet identity split permissions drifted")
    expected_scope = {
        "complete_train_dev_eval_identity_inventory_projected": True,
        "identity_only_projection": True,
        "edf_container_hash_used_for_input_integrity_only": True,
        "identity_fields_authorized_as_model_features": False,
        "source_eval_identity_export_authorized": True,
        "source_eval_model_execution_requires_host_admission": True,
        "source_eval_model_execution_authorized_by_this_receipt": False,
        "source_eval_reference_join_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("EventNet identity projection scope drifted")

    digest = deepcopy(data)
    digest["projection_id"] = "EVENTNET-TUSZ-IDENTITY-PROJECTION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["projection_id"] != "EVNTUSZID-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet identity projection ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet identity projection receipt hash drifted")

    if source_roster is not None:
        validated_roster = validate_tusz_complete_detector_roster_v1(source_roster)
        expected = _projection_from_validated_roster(validated_roster)
        if data != expected:
            raise ValueError("EventNet identity projection disagrees with source roster")
    return data


def build_eventnet_tusz_complete_identity_projection_v1(
    source_roster: Mapping[str, object],
) -> dict[str, Any]:
    """Project a complete validated roster without accepting reference paths."""

    roster = validate_tusz_complete_detector_roster_v1(source_roster)
    projection = _projection_from_validated_roster(roster)
    return validate_eventnet_tusz_complete_identity_projection_v1(
        projection,
        source_roster=roster,
    )


# Short aliases keep script/test call sites readable without weakening schema IDs.
build_eventnet_complete_roster_identity_projection_v1 = (
    build_eventnet_tusz_complete_identity_projection_v1
)
validate_eventnet_complete_roster_identity_projection_v1 = (
    validate_eventnet_tusz_complete_identity_projection_v1
)


__all__ = [
    "EVENTNET_TUSZ_COMPLETE_IDENTITY_FIELDS",
    "EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_METHOD_ID",
    "EVENTNET_TUSZ_COMPLETE_IDENTITY_PROJECTION_SCHEMA_VERSION",
    "build_eventnet_complete_roster_identity_projection_v1",
    "build_eventnet_tusz_complete_identity_projection_v1",
    "validate_eventnet_complete_roster_identity_projection_v1",
    "validate_eventnet_tusz_complete_identity_projection_v1",
]
