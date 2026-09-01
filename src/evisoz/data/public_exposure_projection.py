"""Privacy-safe EviSOZ projection of the frozen public exposure registry."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from typing import Any, Mapping

from .artifact_ref import build_json_artifact_ref, canonical_json_sha256
from .split_ledger import (
    SPLIT_ROSTER_SCHEMA_VERSION,
    build_patient_linkage_group,
    build_split_roster,
    validate_patient_linkage_group,
    validate_split_roster,
)


PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION = (
    "evisoz_public_auxiliary_exposure_projection_v1"
)
_SOURCE_SCHEMA_VERSION = "clinical_eeg_full_stack_nested_exposure_registry_v1"
_SOURCE_STATUS = "patient_rosters_and_nested_schedule_materialized_artifacts_untrained"
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_PATIENT_HASH_DOMAIN = b"evisoz-tuh-public-patient-v1\x00"


def _source_patient_sha256(raw_patient_id: str) -> str:
    return hashlib.sha256(
        _PATIENT_HASH_DOMAIN + raw_patient_id.encode("ascii")
    ).hexdigest()


def _patient_pseudonym(raw_patient_id: str) -> str:
    return "TUH-P-" + _source_patient_sha256(raw_patient_id)[:16]


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["projection_id"] = _PENDING_ID
    return result


def build_public_auxiliary_exposure_projection(
    source_registry: Mapping[str, object],
) -> dict[str, Any]:
    """Project source-train patient folds without persisting TUH patient IDs."""

    if type(source_registry) is not dict:
        raise TypeError("public exposure registry must be an object")
    if source_registry.get("schema_version") != _SOURCE_SCHEMA_VERSION:
        raise ValueError("public exposure registry schema drifted")
    if source_registry.get("status") != _SOURCE_STATUS:
        raise ValueError("public exposure registry status drifted")
    partition_groups = source_registry.get("partition_groups")
    if not isinstance(partition_groups, list) or not partition_groups:
        raise ValueError("public exposure registry partition groups are empty")
    raw_fold_by_patient: dict[str, int] = {}
    deepsoz_overlap: set[str] = set()
    tuev_overlap: set[str] = set()
    source_group_counts: Counter[str] = Counter()
    for group in partition_groups:
        if not isinstance(group, Mapping):
            raise TypeError("public exposure partition group must be an object")
        group_id = group.get("group_id")
        fold = group.get("source_fold_id")
        patients = group.get("patient_ids")
        if (
            not isinstance(group_id, str)
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold < 0
            or not isinstance(patients, list)
            or not patients
        ):
            raise ValueError("public exposure partition group identity drifted")
        if len(patients) != len(set(patients)) or group.get("patient_count") != len(patients):
            raise ValueError("public exposure partition patient roster drifted")
        for patient in patients:
            if (
                not isinstance(patient, str)
                or not patient
                or not patient.isascii()
                or patient in raw_fold_by_patient
            ):
                raise ValueError("public exposure patient identity is invalid or duplicated")
            raw_fold_by_patient[patient] = fold
        deepsoz = group.get("deepsoz_source_train_overlay_patient_ids")
        tuev = group.get("tuev_train_visible_overlap_patient_ids")
        if not isinstance(deepsoz, list) or not isinstance(tuev, list):
            raise ValueError("public exposure overlap rosters are missing")
        if not set(deepsoz).issubset(patients) or not set(tuev).issubset(patients):
            raise ValueError("public exposure overlap roster escapes its partition")
        deepsoz_overlap.update(deepsoz)
        tuev_overlap.update(tuev)
        source_group_counts[group_id] = len(patients)
    if len(raw_fold_by_patient) != 579:
        raise ValueError("public TUSZ development patient denominator drifted")

    groups_by_raw: dict[str, dict[str, Any]] = {}
    for raw_patient in sorted(raw_fold_by_patient):
        groups_by_raw[raw_patient] = build_patient_linkage_group(
            members=[
                {
                    "dataset_id": "tusz",
                    "patient_key": _patient_pseudonym(raw_patient),
                    "source_patient_sha256": _source_patient_sha256(raw_patient),
                }
            ],
            linkage_status="singleton",
        )
    assignments = [
        {
            "linkage_group_id": group["linkage_group_id"],
            "official_splits": [
                {"dataset_id": "tusz", "official_split": "source_train"}
            ],
            "evisoz_role": "development_cv",
            "outer_holdout_fold": raw_fold_by_patient[raw_patient],
            "locked": False,
        }
        for raw_patient, group in groups_by_raw.items()
    ]
    split = build_split_roster(
        linkage_groups=list(groups_by_raw.values()),
        assignments=assignments,
    )
    exposure_rows = [
        {
            "linkage_group_id": groups_by_raw[raw_patient]["linkage_group_id"],
            "outer_fold": raw_fold_by_patient[raw_patient],
            "deepsoz_source_train_overlap": raw_patient in deepsoz_overlap,
            "tuev_train_visible_overlap": raw_patient in tuev_overlap,
            "self_supervised_signal_use_requires_outer_fold_exclusion": True,
            "task_label_use_authorized": False,
        }
        for raw_patient in sorted(raw_fold_by_patient)
    ]
    exposure_rows.sort(key=lambda row: row["linkage_group_id"])
    linkage_groups = sorted(
        groups_by_raw.values(), key=lambda group: group["linkage_group_id"]
    )
    fold_counts = Counter(raw_fold_by_patient.values())
    body: dict[str, Any] = {
        "schema_version": PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION,
        "projection_id": _PENDING_ID,
        "status": "patient_split_projection_materialized_training_disabled",
        "source_registry_ref": build_json_artifact_ref(
            source_registry,
            artifact_kind="public_auxiliary_exposure_registry",
            payload_schema_version=_SOURCE_SCHEMA_VERSION,
        ),
        "linkage_groups": linkage_groups,
        "split_roster": split,
        "patient_exposure_rows": exposure_rows,
        "counts": {
            "tusz_source_train_patient_count": len(raw_fold_by_patient),
            "outer_fold_patient_counts": {
                str(key): fold_counts[key] for key in sorted(fold_counts)
            },
            "source_partition_group_patient_counts": dict(
                sorted(source_group_counts.items())
            ),
            "deepsoz_source_train_overlap_patient_count": len(deepsoz_overlap),
            "tuev_train_visible_overlap_patient_count": len(tuev_overlap),
        },
        "permissions": {
            "training_authorized_by_projection": False,
            "task_label_training_authorized": False,
            "self_supervised_signal_pretraining_authorized": False,
            "outer_fold_exclusion_required_if_later_authorized": True,
            "public_v29_to_tusz_crosswalk_closed": False,
            "tuev_eval_patient_identity_closed": False,
            "raw_patient_identifiers_stored": False,
        },
        "missing_closure_codes": [
            "auxiliary_field_releases_not_materialized",
            "near_or_partial_content_overlap_not_materialized",
            "public_v29_to_tusz_patient_crosswalk_not_materialized",
            "tuev_eval_patient_identity_opaque",
        ],
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["projection_id"] = "EVISOZ-PUBEXP-" + canonical_json_sha256(
        _id_source(body)
    )[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_public_auxiliary_exposure_projection(body)


def validate_public_auxiliary_exposure_projection(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "projection_id",
        "status",
        "source_registry_ref",
        "linkage_groups",
        "split_roster",
        "patient_exposure_rows",
        "counts",
        "permissions",
        "missing_closure_codes",
        "receipt_sha256",
    }:
        raise ValueError("public auxiliary exposure projection fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION:
        raise ValueError("public auxiliary exposure projection schema drifted")
    if data["status"] != "patient_split_projection_materialized_training_disabled":
        raise ValueError("public auxiliary exposure projection status drifted")
    groups = data["linkage_groups"]
    if not isinstance(groups, list) or len(groups) != 579:
        raise ValueError("public auxiliary exposure linkage denominator drifted")
    normalized_groups = [validate_patient_linkage_group(group) for group in groups]
    if normalized_groups != sorted(
        normalized_groups, key=lambda group: group["linkage_group_id"]
    ) or len({group["linkage_group_id"] for group in normalized_groups}) != len(
        normalized_groups
    ):
        raise ValueError("public auxiliary exposure linkage groups are not uniquely sorted")
    trusted = {group["linkage_group_id"]: group for group in normalized_groups}
    split = validate_split_roster(
        data["split_roster"],
        trusted_linkage_groups=trusted,
    )
    rows = data["patient_exposure_rows"]
    if not isinstance(rows, list) or len(rows) != len(groups):
        raise ValueError("public auxiliary exposure patient rows drifted")
    if rows != sorted(rows, key=lambda row: row["linkage_group_id"]) or {
        row["linkage_group_id"] for row in rows
    } != set(trusted):
        raise ValueError("public auxiliary exposure patient rows are not split-complete")
    assignment_by_group = {
        row["linkage_group_id"]: row for row in split["assignments"]
    }
    fold_counts: Counter[int] = Counter()
    deepsoz_count = 0
    tuev_count = 0
    for row in rows:
        if type(row) is not dict or set(row) != {
            "linkage_group_id",
            "outer_fold",
            "deepsoz_source_train_overlap",
            "tuev_train_visible_overlap",
            "self_supervised_signal_use_requires_outer_fold_exclusion",
            "task_label_use_authorized",
        }:
            raise ValueError("public auxiliary exposure patient row fields drifted")
        assignment = assignment_by_group[row["linkage_group_id"]]
        if (
            row["outer_fold"] != assignment["outer_holdout_fold"]
            or assignment["evisoz_role"] != "development_cv"
            or row["self_supervised_signal_use_requires_outer_fold_exclusion"] is not True
            or row["task_label_use_authorized"] is not False
        ):
            raise ValueError("public auxiliary exposure patient row policy drifted")
        fold_counts[row["outer_fold"]] += 1
        deepsoz_count += int(row["deepsoz_source_train_overlap"] is True)
        tuev_count += int(row["tuev_train_visible_overlap"] is True)
    counts = data["counts"]
    if counts["tusz_source_train_patient_count"] != len(rows):
        raise ValueError("public auxiliary exposure patient count drifted")
    if counts["outer_fold_patient_counts"] != {
        str(key): fold_counts[key] for key in sorted(fold_counts)
    }:
        raise ValueError("public auxiliary exposure fold counts drifted")
    if counts["deepsoz_source_train_overlap_patient_count"] != deepsoz_count or counts[
        "tuev_train_visible_overlap_patient_count"
    ] != tuev_count:
        raise ValueError("public auxiliary exposure overlap counts drifted")
    if data["permissions"] != {
        "training_authorized_by_projection": False,
        "task_label_training_authorized": False,
        "self_supervised_signal_pretraining_authorized": False,
        "outer_fold_exclusion_required_if_later_authorized": True,
        "public_v29_to_tusz_crosswalk_closed": False,
        "tuev_eval_patient_identity_closed": False,
        "raw_patient_identifiers_stored": False,
    }:
        raise ValueError("public auxiliary exposure permissions drifted")
    expected_missing = [
        "auxiliary_field_releases_not_materialized",
        "near_or_partial_content_overlap_not_materialized",
        "public_v29_to_tusz_patient_crosswalk_not_materialized",
        "tuev_eval_patient_identity_opaque",
    ]
    if data["missing_closure_codes"] != expected_missing:
        raise ValueError("public auxiliary exposure missing closures drifted")
    expected_id = "EVISOZ-PUBEXP-" + canonical_json_sha256(_id_source(data))[:24]
    if data["projection_id"] != expected_id:
        raise ValueError("public auxiliary exposure projection ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("public auxiliary exposure projection hash drifted")
    data["linkage_groups"] = normalized_groups
    data["split_roster"] = split
    return data


__all__ = [
    "PUBLIC_EXPOSURE_PROJECTION_SCHEMA_VERSION",
    "build_public_auxiliary_exposure_projection",
    "validate_public_auxiliary_exposure_projection",
]
