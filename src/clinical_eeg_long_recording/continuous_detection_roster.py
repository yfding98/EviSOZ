"""Content-bound patient/recording rosters for detector experiments.

Only identity and frozen split metadata are projected from a manifest.  Event
times, SOZ labels, EDF annotations, Excel fields, and clinical text never enter
the returned artifact.  This roster is an inventory/isolation control, not a
reference label artifact or detector qualification receipt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Final, Mapping, Sequence


CONTINUOUS_DETECTOR_ROSTER_SCHEMA_VERSION = "continuous_detector_split_roster_v1"
CONTINUOUS_DETECTOR_ROSTER_METHOD_ID = (
    "identity_split_projection_with_patient_isolation_v1"
)
CONTINUOUS_DETECTOR_ROSTER_ALLOWED_PATIENT_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "patient_id",
            "patient_uid",
            "patient_key",
            "subject_id",
            "local_patient_id",
            "deepsoz_patient_id",
        }
    )
)
CONTINUOUS_DETECTOR_ROSTER_ALLOWED_RECORDING_FIELDS: Final[frozenset[str]] = (
    frozenset(
        {
            "recording_id",
            "recording_key",
            "record_id",
            "edf_path",
            "relative_edf_path",
            "local_edf_path",
        }
    )
)
CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_FIELDS: Final[frozenset[str]] = (
    frozenset({"model_split", "dataset_split", "official_split", "split"})
)
CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES: Final[frozenset[str]] = (
    frozenset(
        {
            "source_train",
            "source_dev",
            "source_eval",
            "quarantine",
            "private_inference",
            "train",
            "dev",
            "validation",
            "val",
            "eval",
            "test",
        }
    )
)
CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES: Final[frozenset[str]] = frozenset(
    {
        "manifest_scope_not_claimed",
        "explicit_research_smoke_subset",
        "development_subset",
        "full_manifest_projection_not_completeness_qualified",
    }
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def validate_continuous_detector_projection_fields(
    *,
    patient_field: str,
    recording_field: str,
    split_field: str,
) -> tuple[str, str, str]:
    """Authorize only identity/split fields, never label-bearing projections.

    The allowlist is intentionally semantic rather than a substring filter:
    ``deepsoz_patient_id`` is a valid identity even though the project name
    contains ``soz``.  Conversely, onset/SOZ/channel targets, annotations,
    spreadsheet fields and clinical text have no approved projection slot.
    """

    patient_key = _identifier(patient_field, "patient_field")
    recording_key = _identifier(recording_field, "recording_field")
    split_key = _identifier(split_field, "split_field")
    if patient_key not in CONTINUOUS_DETECTOR_ROSTER_ALLOWED_PATIENT_FIELDS:
        raise ValueError(
            "patient_field is not an allowlisted identity field; target, annotation, "
            "Excel and clinical fields cannot be projected"
        )
    if recording_key not in CONTINUOUS_DETECTOR_ROSTER_ALLOWED_RECORDING_FIELDS:
        raise ValueError(
            "recording_field is not an allowlisted recording identity field; target, "
            "annotation, Excel and clinical fields cannot be projected"
        )
    if split_key not in CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_FIELDS:
        raise ValueError(
            "split_field is not an allowlisted frozen split field; target, annotation, "
            "Excel and clinical fields cannot be projected"
        )
    if len({patient_key, recording_key, split_key}) != 3:
        raise ValueError("patient, recording and split projection fields must be distinct")
    return patient_key, recording_key, split_key


def build_continuous_detector_split_roster(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    manifest_file_sha256: str,
    patient_field: str = "local_patient_id",
    recording_field: str = "local_edf_path",
    split_field: str = "model_split",
    inventory_scope: str = "manifest_scope_not_claimed",
) -> dict[str, Any]:
    """Project a manifest into disjoint, content-bound split rosters."""

    manifest_sha256 = _sha256(manifest_file_sha256, "manifest_file_sha256")
    patient_key, recording_key, split_key = (
        validate_continuous_detector_projection_fields(
            patient_field=patient_field,
            recording_field=recording_field,
            split_field=split_field,
        )
    )
    inventory = _identifier(inventory_scope, "inventory_scope")
    if inventory not in CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES:
        raise ValueError("continuous detector roster inventory scope is unsupported")
    if not isinstance(manifest_rows, Sequence) or isinstance(
        manifest_rows, (str, bytes)
    ) or not manifest_rows:
        raise TypeError("manifest_rows must be a non-empty sequence")

    projected: list[dict[str, str]] = []
    recording_ids: set[str] = set()
    patient_split: dict[str, str] = {}
    for index, raw in enumerate(manifest_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"manifest row {index} must be an object")
        missing = {
            name
            for name in (patient_key, recording_key, split_key)
            if name not in raw
        }
        if missing:
            raise ValueError(f"manifest row {index} lacks fields {sorted(missing)}")
        patient_id = _identifier(raw[patient_key], f"manifest row {index} patient")
        recording_id = _identifier(
            raw[recording_key], f"manifest row {index} recording"
        )
        split = _identifier(raw[split_key], f"manifest row {index} split")
        if split not in CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES:
            raise ValueError(
                f"manifest row {index} split is not a frozen supported split value"
            )
        if recording_id in recording_ids:
            raise ValueError("recording IDs must be globally unique")
        recording_ids.add(recording_id)
        previous = patient_split.setdefault(patient_id, split)
        if previous != split:
            raise ValueError("one patient occurs in multiple detector splits")
        projected.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "split": split,
            }
        )
    projected.sort(
        key=lambda row: (row["split"], row["patient_id"], row["recording_id"])
    )

    split_payload: dict[str, dict[str, Any]] = {}
    for split in sorted({row["split"] for row in projected}):
        selected = [row for row in projected if row["split"] == split]
        patients = sorted({row["patient_id"] for row in selected})
        recordings = sorted(row["recording_id"] for row in selected)
        split_payload[split] = {
            "patient_count": len(patients),
            "recording_count": len(recordings),
            "patient_ids": patients,
            "recording_ids": recordings,
            "patient_roster_sha256": _canonical_sha256(patients),
            "recording_roster_sha256": _canonical_sha256(recordings),
        }

    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_DETECTOR_ROSTER_SCHEMA_VERSION,
        "roster_id": "CONTINUOUS-DETECTOR-ROSTER-PENDING",
        "method_id": CONTINUOUS_DETECTOR_ROSTER_METHOD_ID,
        "inventory_scope": inventory,
        "source_manifest_file_sha256": manifest_sha256,
        "source_field_projection": {
            "patient_field": patient_key,
            "recording_field": recording_key,
            "split_field": split_key,
        },
        "projected_rows_sha256": _canonical_sha256(projected),
        "total_patient_count": len(patient_split),
        "total_recording_count": len(projected),
        "split_rosters": split_payload,
        "patient_split_isolation_verified": True,
        "scope_receipt": {
            "identity_and_split_metadata_only": True,
            "projection_fields_allowlisted": True,
            "target_annotation_excel_fields_projected": False,
            "seizure_intervals_retained": False,
            "soz_or_channel_labels_retained": False,
            "edf_annotations_used": False,
            "excel_or_clinical_text_used": False,
            "roster_is_reference_or_performance_evidence": False,
            "production_or_sota_claim_authorized": False,
            "complete_split_inventory_verified": False,
        },
    }
    body["roster_id"] = "CONTROSTER-" + _canonical_sha256(body)[:24]
    return validate_continuous_detector_split_roster(body)


def validate_continuous_detector_split_roster(payload: object) -> dict[str, Any]:
    """Validate roster content binding, isolation, and label-free scope."""

    required = {
        "schema_version",
        "roster_id",
        "method_id",
        "inventory_scope",
        "source_manifest_file_sha256",
        "source_field_projection",
        "projected_rows_sha256",
        "total_patient_count",
        "total_recording_count",
        "split_rosters",
        "patient_split_isolation_verified",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("continuous detector roster has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != CONTINUOUS_DETECTOR_ROSTER_SCHEMA_VERSION
        or data["method_id"] != CONTINUOUS_DETECTOR_ROSTER_METHOD_ID
    ):
        raise ValueError("continuous detector roster schema/method drifted")
    inventory = _identifier(data["inventory_scope"], "inventory_scope")
    if inventory not in CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES:
        raise ValueError("continuous detector roster inventory scope drifted")
    _sha256(data["source_manifest_file_sha256"], "source manifest SHA-256")
    _sha256(data["projected_rows_sha256"], "projected rows SHA-256")
    projection = data["source_field_projection"]
    if type(projection) is not dict or set(projection) != {
        "patient_field",
        "recording_field",
        "split_field",
    }:
        raise ValueError("continuous detector roster field projection drifted")
    validate_continuous_detector_projection_fields(
        patient_field=projection["patient_field"],
        recording_field=projection["recording_field"],
        split_field=projection["split_field"],
    )
    if data["patient_split_isolation_verified"] is not True:
        raise ValueError("continuous detector roster lacks patient isolation")
    rosters = data["split_rosters"]
    if type(rosters) is not dict or not rosters:
        raise ValueError("continuous detector roster has no splits")
    all_patients: set[str] = set()
    all_recordings: set[str] = set()
    patient_total = 0
    recording_total = 0
    if (
        type(data["total_patient_count"]) is not int
        or data["total_patient_count"] < 1
        or type(data["total_recording_count"]) is not int
        or data["total_recording_count"] < 1
    ):
        raise ValueError("continuous detector total counts must be positive integers")
    for split in sorted(rosters):
        _identifier(split, "split")
        if split not in CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES:
            raise ValueError("continuous detector split value drifted")
        row = rosters[split]
        required_row = {
            "patient_count",
            "recording_count",
            "patient_ids",
            "recording_ids",
            "patient_roster_sha256",
            "recording_roster_sha256",
        }
        if type(row) is not dict or set(row) != required_row:
            raise ValueError("continuous detector split roster fields drifted")
        patients = row["patient_ids"]
        recordings = row["recording_ids"]
        if (
            not isinstance(patients, list)
            or not patients
            or any(not isinstance(value, str) for value in patients)
            or patients != sorted(set(patients))
            or not isinstance(recordings, list)
            or not recordings
            or any(not isinstance(value, str) for value in recordings)
            or recordings != sorted(set(recordings))
        ):
            raise ValueError("continuous detector split roster is not canonical")
        for value in patients:
            _identifier(value, "patient ID")
        for value in recordings:
            _identifier(value, "recording ID")
        if all_patients.intersection(patients):
            raise ValueError("continuous detector patient crosses split rosters")
        if all_recordings.intersection(recordings):
            raise ValueError("continuous detector recording crosses split rosters")
        all_patients.update(patients)
        all_recordings.update(recordings)
        if (
            type(row["patient_count"]) is not int
            or type(row["recording_count"]) is not int
            or row["patient_count"] != len(patients)
            or row["recording_count"] != len(recordings)
        ):
            raise ValueError("continuous detector split roster counts drifted")
        if row["patient_roster_sha256"] != _canonical_sha256(patients) or row[
            "recording_roster_sha256"
        ] != _canonical_sha256(recordings):
            raise ValueError("continuous detector split roster hash drifted")
        patient_total += len(patients)
        recording_total += len(recordings)
    if data["total_patient_count"] != patient_total or data[
        "total_recording_count"
    ] != recording_total:
        raise ValueError("continuous detector total roster counts drifted")
    expected_scope = {
        "identity_and_split_metadata_only": True,
        "projection_fields_allowlisted": True,
        "target_annotation_excel_fields_projected": False,
        "seizure_intervals_retained": False,
        "soz_or_channel_labels_retained": False,
        "edf_annotations_used": False,
        "excel_or_clinical_text_used": False,
        "roster_is_reference_or_performance_evidence": False,
        "production_or_sota_claim_authorized": False,
        "complete_split_inventory_verified": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("continuous detector roster scope drifted")
    digest = deepcopy(data)
    digest["roster_id"] = "CONTINUOUS-DETECTOR-ROSTER-PENDING"
    expected_id = "CONTROSTER-" + _canonical_sha256(digest)[:24]
    if data["roster_id"] != expected_id:
        raise ValueError("continuous detector roster is not content-bound")
    return data


__all__ = [
    "CONTINUOUS_DETECTOR_ROSTER_ALLOWED_PATIENT_FIELDS",
    "CONTINUOUS_DETECTOR_ROSTER_ALLOWED_RECORDING_FIELDS",
    "CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_FIELDS",
    "CONTINUOUS_DETECTOR_ROSTER_ALLOWED_SPLIT_VALUES",
    "CONTINUOUS_DETECTOR_ROSTER_INVENTORY_SCOPES",
    "CONTINUOUS_DETECTOR_ROSTER_METHOD_ID",
    "CONTINUOUS_DETECTOR_ROSTER_SCHEMA_VERSION",
    "build_continuous_detector_split_roster",
    "validate_continuous_detector_projection_fields",
    "validate_continuous_detector_split_roster",
]
