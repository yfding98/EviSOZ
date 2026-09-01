"""Complete TUSZ audit roster with exact-container equivalence handling.

Version 1 intentionally fails closed when two official EDF paths contain the
same bytes.  Version 2 is additive: it retains every official path in an audit
roster, materializes exact-container equivalence classes, and emits a separate
analysis identity projection.  A class confined to one official split and one
patient directory alias contributes exactly one canonical unit.  A class that
crosses a patient alias or official split is quarantined in full.

Only EDF bytes/acquisition headers and the *identity/existence* of each
``csv_bi`` sidecar are inspected.  Sidecar contents and labels are never read.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

from .tusz_complete_detector_roster_v1 import (
    TUSZ_V203_EXPECTED_INVENTORY,
    _canonical_sha256,
    _identifier,
    _positive_integer,
    _safe_regular_file,
    _sha256,
    _sha256_file,
    _summarize_records,
    _validate_record_row,
    inspect_edf_container_header_v1,
    validate_tusz_complete_expected_inventory_v1,
)


TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION = (
    "tusz_complete_detector_roster_v2"
)
TUSZ_COMPLETE_DETECTOR_ROSTER_V2_METHOD_ID = (
    "full_path_audit_exact_container_equivalence_and_quarantine_v2"
)
TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION = (
    "tusz_analysis_identity_projection_v2"
)
TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_METHOD_ID = (
    "one_unit_per_safe_exact_container_class_cross_boundary_quarantine_v2"
)

TUSZ_ANALYSIS_IDENTITY_FIELDS_V2: Final[tuple[str, ...]] = (
    "analysis_identity_id",
    "model_split",
    "official_split",
    "local_patient_id",
    "local_edf_path",
    "source_edf_container_sha256",
    "exact_container_equivalence_id",
    "source_official_path_multiplicity",
    "analysis_unit_weight",
)

_OFFICIAL_TO_MODEL_SPLIT: Final[dict[str, str]] = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_MODEL_TO_OFFICIAL_SPLIT: Final[dict[str, str]] = {
    value: key for key, value in _OFFICIAL_TO_MODEL_SPLIT.items()
}


def _equivalence_id(container_sha256: str) -> str:
    return f"TUSZEXACT-{container_sha256}"


def _analysis_identity_id(container_sha256: str) -> str:
    return f"TUSZANALYSIS-{container_sha256}"


def _analysis_policy() -> dict[str, Any]:
    return {
        "equivalence_key": "full_edf_container_sha256",
        "audit_roster_retains_every_official_path": True,
        "singleton_class_policy": "one_analysis_unit_weight_one",
        "same_split_same_patient_alias_policy": (
            "lexicographically_first_path_is_one_unit_weight_one_others_excluded"
        ),
        "cross_patient_alias_or_official_split_policy": (
            "quarantine_entire_equivalence_class"
        ),
        "canonical_selection_uses_reference_values": False,
        "reference_sidecar_reconciliation_authorized": False,
        "audit_roster_paths_may_be_used_as_analysis_denominator": False,
    }


def _role_permissions() -> dict[str, dict[str, bool]]:
    return {
        "source_train": {
            "model_fit_identity_authorized": True,
            "development_calibration_identity_authorized": False,
            "locked_evaluation_identity_export_authorized": False,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": False,
            "reference_access_authorized": False,
        },
        "source_dev": {
            "model_fit_identity_authorized": False,
            "development_calibration_identity_authorized": True,
            "locked_evaluation_identity_export_authorized": False,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": False,
            "reference_access_authorized": False,
        },
        "source_eval": {
            "model_fit_identity_authorized": False,
            "development_calibration_identity_authorized": False,
            "locked_evaluation_identity_export_authorized": True,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": True,
            "reference_access_authorized": False,
        },
    }


def _reference_access_receipt() -> dict[str, Any]:
    return {
        "reference_path_argument_accepted": False,
        "reference_files_opened": 0,
        "csv_bi_files_opened": 0,
        "csv_bi_bytes_read": 0,
        "csv_bi_contents_read": False,
        "seizure_interval_or_label_values_read": False,
        "edf_annotations_read": False,
        "spreadsheet_or_clinical_text_read": False,
    }


def _equivalence_classes(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_hash.setdefault(str(row["container_sha256"]), []).append(row)

    classes: list[dict[str, Any]] = []
    for container_sha256, members in by_hash.items():
        ordered = sorted(members, key=lambda row: str(row["recording_id"]))
        metadata_bindings = {
            (
                row["container_bytes"],
                row["header_bytes"],
                row["header_sha256"],
                row["signal_count"],
                tuple(row["data_record_duration_fraction"]),
                tuple(row["recording_duration_fraction"]),
            )
            for row in ordered
        }
        if len(metadata_bindings) != 1:
            raise ValueError(
                "one exact-container hash has inconsistent acquisition metadata"
            )
        recording_ids = [str(row["recording_id"]) for row in ordered]
        official_splits = sorted({str(row["official_split"]) for row in ordered})
        patient_aliases = sorted({str(row["patient_id"]) for row in ordered})
        if len(ordered) == 1:
            boundary_type = "singleton"
            analysis_eligible = True
        elif len(official_splits) == 1 and len(patient_aliases) == 1:
            boundary_type = "same_split_same_patient_alias"
            analysis_eligible = True
        elif len(official_splits) > 1 and len(patient_aliases) > 1:
            boundary_type = "cross_official_split_and_patient_alias"
            analysis_eligible = False
        elif len(official_splits) > 1:
            boundary_type = "cross_official_split"
            analysis_eligible = False
        else:
            boundary_type = "cross_patient_alias_same_split"
            analysis_eligible = False
        audit_canonical = recording_ids[0]
        analysis_canonical = audit_canonical if analysis_eligible else None
        excluded_aliases = (
            recording_ids[1:]
            if boundary_type == "same_split_same_patient_alias"
            else []
        )
        classes.append(
            {
                "equivalence_class_id": _equivalence_id(container_sha256),
                "container_sha256": container_sha256,
                "container_bytes": ordered[0]["container_bytes"],
                "member_count": len(ordered),
                "member_recording_ids": recording_ids,
                "member_roster_sha256": _canonical_sha256(recording_ids),
                "official_splits": official_splits,
                "patient_aliases": patient_aliases,
                "boundary_type": boundary_type,
                "audit_canonical_recording_id": audit_canonical,
                "analysis_canonical_recording_id": analysis_canonical,
                "analysis_eligible": analysis_eligible,
                "excluded_same_patient_alias_recording_ids": excluded_aliases,
                "quarantine_reason": (
                    None
                    if analysis_eligible
                    else "exact_container_equivalence_crosses_patient_or_split"
                ),
            }
        )
    classes.sort(key=lambda row: row["equivalence_class_id"])
    return classes


def _equivalence_inventory(
    classes: Sequence[Mapping[str, Any]], total_recording_count: int
) -> dict[str, Any]:
    duplicate_classes = [row for row in classes if row["member_count"] > 1]
    safe_alias_classes = [
        row
        for row in classes
        if row["boundary_type"] == "same_split_same_patient_alias"
    ]
    quarantine_classes = [row for row in classes if not row["analysis_eligible"]]
    eligible_classes = [row for row in classes if row["analysis_eligible"]]
    excluded_alias_count = sum(
        len(row["excluded_same_patient_alias_recording_ids"])
        for row in safe_alias_classes
    )
    quarantined_recording_count = sum(
        row["member_count"] for row in quarantine_classes
    )
    inventory = {
        "equivalence_class_count": len(classes),
        "singleton_class_count": sum(
            row["boundary_type"] == "singleton" for row in classes
        ),
        "exact_duplicate_class_count": len(duplicate_classes),
        "exact_duplicate_member_recording_count": sum(
            row["member_count"] for row in duplicate_classes
        ),
        "exact_duplicate_excess_path_count": sum(
            row["member_count"] - 1 for row in duplicate_classes
        ),
        "same_split_same_patient_alias_class_count": len(safe_alias_classes),
        "same_patient_alias_excluded_recording_count": excluded_alias_count,
        "boundary_quarantine_class_count": len(quarantine_classes),
        "cross_patient_same_split_class_count": sum(
            row["boundary_type"] == "cross_patient_alias_same_split"
            for row in quarantine_classes
        ),
        "cross_official_split_class_count": sum(
            row["boundary_type"]
            in {"cross_official_split", "cross_official_split_and_patient_alias"}
            for row in quarantine_classes
        ),
        "quarantined_recording_count": quarantined_recording_count,
        "analysis_eligible_class_count": len(eligible_classes),
        "analysis_eligible_canonical_recording_count": len(eligible_classes),
        "class_roster_sha256": _canonical_sha256(classes),
        "analysis_canonical_binding_sha256": _canonical_sha256(
            [
                [row["equivalence_class_id"], row["analysis_canonical_recording_id"]]
                for row in eligible_classes
            ]
        ),
        "quarantine_class_roster_sha256": _canonical_sha256(
            [row["equivalence_class_id"] for row in quarantine_classes]
        ),
        "path_accounting_verified": (
            len(eligible_classes)
            + excluded_alias_count
            + quarantined_recording_count
            == total_recording_count
        ),
    }
    if not inventory["path_accounting_verified"]:
        raise ValueError("exact-container analysis path accounting does not close")
    return inventory


def _expected_sidecar_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sidecar_ids = sorted(str(row["reference_sidecar_id"]) for row in records)
    return {
        "sidecar_count": len(sidecar_ids),
        "sidecar_identity_roster_sha256": _canonical_sha256(sidecar_ids),
        "one_to_one_with_edf_verified": True,
        "sidecar_contents_opened": False,
    }


def validate_tusz_complete_detector_roster_v2(payload: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "roster_id",
        "expected_inventory",
        "records",
        "observed_inventory",
        "reference_sidecar_inventory",
        "exact_container_equivalence_classes",
        "exact_container_equivalence_inventory",
        "analysis_eligibility_policy",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("TUSZ complete detector roster v2 fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION
        or data["method_id"] != TUSZ_COMPLETE_DETECTOR_ROSTER_V2_METHOD_ID
    ):
        raise ValueError("TUSZ complete detector roster v2 schema/method drifted")
    expected = validate_tusz_complete_expected_inventory_v1(data["expected_inventory"])
    records = data["records"]
    if type(records) is not list or not records:
        raise ValueError("TUSZ complete detector roster v2 has no records")
    validated_records = [
        _validate_record_row(row, index) for index, row in enumerate(records)
    ]
    canonical_order = sorted(
        validated_records,
        key=lambda row: (row["official_split"], row["patient_id"], row["recording_id"]),
    )
    if validated_records != canonical_order:
        raise ValueError("TUSZ roster v2 records are not canonically sorted")
    recording_ids = [row["recording_id"] for row in validated_records]
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError("TUSZ roster v2 official recording paths are not unique")

    observed = _summarize_records(validated_records)
    if data["observed_inventory"] != observed:
        raise ValueError("TUSZ roster v2 observed inventory is not replayable")
    if (
        observed["total_patient_count"] != expected["total_patient_count"]
        or observed["total_recording_count"] != expected["total_recording_count"]
    ):
        raise ValueError("TUSZ roster v2 total inventory differs from release")
    for split, expected_row in expected["split_expectations"].items():
        row = observed["split_summaries"][split]
        if (
            row["patient_count"] != expected_row["patient_count"]
            or row["recording_count"] != expected_row["recording_count"]
        ):
            raise ValueError("TUSZ roster v2 split inventory differs from release")

    if data["reference_sidecar_inventory"] != _expected_sidecar_inventory(records):
        raise ValueError("TUSZ roster v2 sidecar identity inventory drifted")
    classes = _equivalence_classes(validated_records)
    if data["exact_container_equivalence_classes"] != classes:
        raise ValueError("TUSZ roster v2 equivalence classes are not replayable")
    equivalence_inventory = _equivalence_inventory(
        classes, observed["total_recording_count"]
    )
    if data["exact_container_equivalence_inventory"] != equivalence_inventory:
        raise ValueError("TUSZ roster v2 equivalence inventory drifted")
    if data["analysis_eligibility_policy"] != _analysis_policy():
        raise ValueError("TUSZ roster v2 analysis eligibility policy drifted")
    expected_scope = {
        "complete_official_path_inventory_retained": True,
        "exact_container_duplicates_retained_for_audit": True,
        "exact_container_equivalence_classes_complete": True,
        "analysis_requires_separate_deduplicated_projection": True,
        "patient_official_split_isolation_verified": True,
        "csv_bi_one_to_one_identity_verified": True,
        "csv_bi_contents_read": False,
        "reference_labels_retained": False,
        "edf_annotations_used_as_model_input": False,
        "excel_doctor_or_clinical_text_used": False,
        "canonical_physical_signal_duplicate_audit_complete": False,
        "official_eval_reference_access_authorized": False,
        "detector_performance_or_sota_claim_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("TUSZ roster v2 scope receipt drifted")

    digest = deepcopy(data)
    digest["roster_id"] = "TUSZ-COMPLETE-ROSTER-V2-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["roster_id"] != "TUSZCROSTERV2-" + _canonical_sha256(digest)[:24]:
        raise ValueError("TUSZ roster v2 ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("TUSZ roster v2 receipt hash drifted")
    return data


def build_tusz_complete_detector_roster_v2(
    *,
    tusz_root: str | Path,
    expected_inventory: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Hash a complete TUSZ tree while retaining exact duplicate paths."""

    root_path = Path(tusz_root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("TUSZ root must be a regular non-symlink directory")
    root = root_path.resolve(strict=True)
    expectation = validate_tusz_complete_expected_inventory_v1(
        deepcopy(
            dict(expected_inventory)
            if expected_inventory is not None
            else TUSZ_V203_EXPECTED_INVENTORY
        )
    )
    edf_paths = sorted(
        root.rglob("*.edf"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not edf_paths:
        raise ValueError("TUSZ root contains no EDF files")
    discovered_sidecars = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.csv_bi")
        if _safe_regular_file(path, root, "TUSZ csv_bi sidecar")
    }
    records: list[dict[str, Any]] = []
    expected_sidecars: set[str] = set()
    for edf_path in edf_paths:
        edf = _safe_regular_file(edf_path, root, "TUSZ EDF")
        relative = edf.relative_to(root)
        if (
            len(relative.parts) < 3
            or relative.parts[0] not in _OFFICIAL_TO_MODEL_SPLIT
        ):
            raise ValueError("EDF path is outside train/dev/eval official layout")
        split = relative.parts[0]
        patient_id = relative.parts[1]
        recording_id = relative.as_posix()
        sidecar = edf.with_suffix(".csv_bi")
        _safe_regular_file(sidecar, root, "TUSZ csv_bi sidecar")
        sidecar_id = sidecar.relative_to(root).as_posix()
        expected_sidecars.add(sidecar_id)
        before = edf.stat()
        metadata = inspect_edf_container_header_v1(edf)
        container_sha256 = _sha256_file(edf)
        after = edf.stat()
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if stable_before != stable_after:
            raise ValueError("TUSZ EDF changed while v2 was inventorying it")
        records.append(
            {
                "recording_id": recording_id,
                "patient_id": patient_id,
                "official_split": split,
                "benchmark_split": _OFFICIAL_TO_MODEL_SPLIT[split],
                "montage": relative.parent.name,
                "container_sha256": container_sha256,
                **metadata,
                "reference_sidecar_id": sidecar_id,
                "reference_sidecar_exists": True,
            }
        )
    if expected_sidecars != discovered_sidecars:
        missing = sorted(expected_sidecars - discovered_sidecars)[:5]
        orphan = sorted(discovered_sidecars - expected_sidecars)[:5]
        raise ValueError(
            f"EDF/csv_bi inventory is not one-to-one; missing={missing}, orphan={orphan}"
        )
    records.sort(
        key=lambda row: (row["official_split"], row["patient_id"], row["recording_id"])
    )
    observed = _summarize_records(records)
    classes = _equivalence_classes(records)
    body: dict[str, Any] = {
        "schema_version": TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION,
        "method_id": TUSZ_COMPLETE_DETECTOR_ROSTER_V2_METHOD_ID,
        "roster_id": "TUSZ-COMPLETE-ROSTER-V2-PENDING",
        "expected_inventory": expectation,
        "records": records,
        "observed_inventory": observed,
        "reference_sidecar_inventory": _expected_sidecar_inventory(records),
        "exact_container_equivalence_classes": classes,
        "exact_container_equivalence_inventory": _equivalence_inventory(
            classes, observed["total_recording_count"]
        ),
        "analysis_eligibility_policy": _analysis_policy(),
        "scope_receipt": {
            "complete_official_path_inventory_retained": True,
            "exact_container_duplicates_retained_for_audit": True,
            "exact_container_equivalence_classes_complete": True,
            "analysis_requires_separate_deduplicated_projection": True,
            "patient_official_split_isolation_verified": True,
            "csv_bi_one_to_one_identity_verified": True,
            "csv_bi_contents_read": False,
            "reference_labels_retained": False,
            "edf_annotations_used_as_model_input": False,
            "excel_doctor_or_clinical_text_used": False,
            "canonical_physical_signal_duplicate_audit_complete": False,
            "official_eval_reference_access_authorized": False,
            "detector_performance_or_sota_claim_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["roster_id"] = "TUSZCROSTERV2-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_complete_detector_roster_v2(body)


def _projection_records(roster: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_recording = {row["recording_id"]: row for row in roster["records"]}
    rows: list[dict[str, Any]] = []
    for equivalence in roster["exact_container_equivalence_classes"]:
        canonical_id = equivalence["analysis_canonical_recording_id"]
        if canonical_id is None:
            continue
        source = by_recording[canonical_id]
        container_sha256 = source["container_sha256"]
        rows.append(
            {
                "analysis_identity_id": _analysis_identity_id(container_sha256),
                "model_split": source["benchmark_split"],
                "official_split": source["official_split"],
                "local_patient_id": source["patient_id"],
                "local_edf_path": source["recording_id"],
                "source_edf_container_sha256": container_sha256,
                "exact_container_equivalence_id": equivalence[
                    "equivalence_class_id"
                ],
                "source_official_path_multiplicity": equivalence["member_count"],
                "analysis_unit_weight": 1,
            }
        )
    rows.sort(
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        )
    )
    return rows


def _projection_split_summaries(
    roster: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_recording = {row["recording_id"]: row for row in roster["records"]}
    alias_excluded = {split: 0 for split in _MODEL_TO_OFFICIAL_SPLIT}
    quarantined = {split: 0 for split in _MODEL_TO_OFFICIAL_SPLIT}
    for equivalence in roster["exact_container_equivalence_classes"]:
        if equivalence["analysis_eligible"]:
            canonical = by_recording[equivalence["analysis_canonical_recording_id"]]
            alias_excluded[canonical["benchmark_split"]] += len(
                equivalence["excluded_same_patient_alias_recording_ids"]
            )
        else:
            for recording_id in equivalence["member_recording_ids"]:
                quarantined[by_recording[recording_id]["benchmark_split"]] += 1

    summaries: dict[str, dict[str, Any]] = {}
    for model_split in sorted(_MODEL_TO_OFFICIAL_SPLIT):
        official_split = _MODEL_TO_OFFICIAL_SPLIT[model_split]
        selected = [row for row in records if row["model_split"] == model_split]
        identities = [row["analysis_identity_id"] for row in selected]
        patients = sorted({row["local_patient_id"] for row in selected})
        audit_count = roster["observed_inventory"]["split_summaries"][
            official_split
        ]["recording_count"]
        summaries[model_split] = {
            "official_split": official_split,
            "audit_official_path_count": audit_count,
            "analysis_identity_count": len(selected),
            "analysis_patient_alias_count": len(patients),
            "same_patient_alias_excluded_path_count": alias_excluded[model_split],
            "quarantined_path_count": quarantined[model_split],
            "analysis_identity_roster_sha256": _canonical_sha256(identities),
            "analysis_patient_alias_roster_sha256": _canonical_sha256(patients),
            "path_count_closure_verified": (
                len(selected)
                + alias_excluded[model_split]
                + quarantined[model_split]
                == audit_count
            ),
        }
        if not summaries[model_split]["path_count_closure_verified"]:
            raise ValueError("analysis projection split path accounting does not close")
    return summaries


def _projection_from_validated_roster(roster: Mapping[str, Any]) -> dict[str, Any]:
    records = _projection_records(roster)
    summaries = _projection_split_summaries(roster, records)
    equivalence_inventory = roster["exact_container_equivalence_inventory"]
    quarantined_classes = [
        row
        for row in roster["exact_container_equivalence_classes"]
        if not row["analysis_eligible"]
    ]
    quarantine_ids = [row["equivalence_class_id"] for row in quarantined_classes]
    body: dict[str, Any] = {
        "schema_version": TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION,
        "method_id": TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_METHOD_ID,
        "projection_id": "TUSZ-ANALYSIS-IDENTITY-V2-PENDING",
        "source_roster_binding": {
            "source_schema_version": roster["schema_version"],
            "source_roster_id": roster["roster_id"],
            "source_roster_receipt_sha256": roster["receipt_sha256"],
            "source_release_id": roster["expected_inventory"]["release_id"],
            "source_records_payload_sha256": roster["observed_inventory"][
                "records_payload_sha256"
            ],
            "source_equivalence_class_roster_sha256": equivalence_inventory[
                "class_roster_sha256"
            ],
            "source_audit_recording_count": roster["observed_inventory"][
                "total_recording_count"
            ],
            "source_equivalence_class_count": equivalence_inventory[
                "equivalence_class_count"
            ],
            "source_analysis_eligible_class_count": equivalence_inventory[
                "analysis_eligible_class_count"
            ],
            "source_same_patient_alias_excluded_path_count": equivalence_inventory[
                "same_patient_alias_excluded_recording_count"
            ],
            "source_quarantined_path_count": equivalence_inventory[
                "quarantined_recording_count"
            ],
            "source_split_accounting_sha256": _canonical_sha256(summaries),
        },
        "identity_fields": list(TUSZ_ANALYSIS_IDENTITY_FIELDS_V2),
        "records": records,
        "split_summaries": summaries,
        "exclusion_receipt": {
            "same_patient_alias_equivalence_class_count": equivalence_inventory[
                "same_split_same_patient_alias_class_count"
            ],
            "same_patient_alias_excluded_path_count": equivalence_inventory[
                "same_patient_alias_excluded_recording_count"
            ],
            "quarantined_equivalence_class_count": equivalence_inventory[
                "boundary_quarantine_class_count"
            ],
            "quarantined_path_count": equivalence_inventory[
                "quarantined_recording_count"
            ],
            "quarantined_equivalence_class_ids": quarantine_ids,
            "quarantined_equivalence_class_roster_sha256": _canonical_sha256(
                quarantine_ids
            ),
            "noncanonical_alias_paths_retained_as_analysis_rows": False,
            "quarantined_paths_retained_as_analysis_rows": False,
        },
        "role_permissions": _role_permissions(),
        "reference_access_receipt": _reference_access_receipt(),
        "scope_receipt": {
            "identity_only_projection": True,
            "exact_container_equivalence_is_analysis_unit": True,
            "every_analysis_unit_has_weight_one": True,
            "same_patient_alias_duplicates_cannot_be_double_counted": True,
            "cross_patient_or_split_duplicates_are_fully_quarantined": True,
            "official_split_roles_are_mutually_separated": True,
            "source_eval_execution_requires_host_admission": True,
            "source_eval_execution_authorized_by_projection": False,
            "reference_join_authorized": False,
            "findings_soz_or_performance_claim_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["projection_id"] = "TUSZANALYSISV2-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _validate_projection_record(value: object, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(TUSZ_ANALYSIS_IDENTITY_FIELDS_V2):
        raise ValueError(f"TUSZ analysis identity row {index} fields drifted")
    row = deepcopy(value)
    model_split = row["model_split"]
    official_split = row["official_split"]
    if (
        model_split not in _MODEL_TO_OFFICIAL_SPLIT
        or official_split != _MODEL_TO_OFFICIAL_SPLIT[model_split]
    ):
        raise ValueError("TUSZ analysis identity split mapping drifted")
    patient_id = _identifier(row["local_patient_id"], "analysis patient alias")
    recording_id = _identifier(row["local_edf_path"], "analysis EDF path")
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
        raise ValueError("TUSZ analysis identity split/patient/path binding drifted")
    container_sha256 = _sha256(
        row["source_edf_container_sha256"], "analysis EDF container SHA-256"
    )
    if (
        row["exact_container_equivalence_id"] != _equivalence_id(container_sha256)
        or row["analysis_identity_id"] != _analysis_identity_id(container_sha256)
    ):
        raise ValueError("TUSZ analysis exact-container identity binding drifted")
    _positive_integer(
        row["source_official_path_multiplicity"], "source path multiplicity"
    )
    if type(row["analysis_unit_weight"]) is not int or row[
        "analysis_unit_weight"
    ] != 1:
        raise ValueError("TUSZ analysis identity unit weight must equal one")
    return row


def validate_tusz_analysis_identity_projection_v2(
    payload: object,
    *,
    source_roster: object | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "projection_id",
        "source_roster_binding",
        "identity_fields",
        "records",
        "split_summaries",
        "exclusion_receipt",
        "role_permissions",
        "reference_access_receipt",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("TUSZ analysis identity projection v2 fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION
        or data["method_id"] != TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_METHOD_ID
    ):
        raise ValueError("TUSZ analysis projection v2 schema/method drifted")
    binding_fields = {
        "source_schema_version",
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_release_id",
        "source_records_payload_sha256",
        "source_equivalence_class_roster_sha256",
        "source_audit_recording_count",
        "source_equivalence_class_count",
        "source_analysis_eligible_class_count",
        "source_same_patient_alias_excluded_path_count",
        "source_quarantined_path_count",
        "source_split_accounting_sha256",
    }
    binding = data["source_roster_binding"]
    if type(binding) is not dict or set(binding) != binding_fields:
        raise ValueError("TUSZ analysis source-roster binding fields drifted")
    if binding["source_schema_version"] != TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION:
        raise ValueError("TUSZ analysis source-roster schema drifted")
    _identifier(binding["source_roster_id"], "source roster ID")
    _identifier(binding["source_release_id"], "source release ID")
    for field in (
        "source_roster_receipt_sha256",
        "source_records_payload_sha256",
        "source_equivalence_class_roster_sha256",
        "source_split_accounting_sha256",
    ):
        _sha256(binding[field], field)
    for field in ("source_audit_recording_count", "source_equivalence_class_count"):
        _positive_integer(binding[field], field)
    for field in (
        "source_analysis_eligible_class_count",
        "source_same_patient_alias_excluded_path_count",
        "source_quarantined_path_count",
    ):
        if type(binding[field]) is not int or binding[field] < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if data["identity_fields"] != list(TUSZ_ANALYSIS_IDENTITY_FIELDS_V2):
        raise ValueError("TUSZ analysis identity field allowlist drifted")
    if type(data["records"]) is not list:
        raise ValueError("TUSZ analysis identity records must be a list")
    records = [
        _validate_projection_record(row, index)
        for index, row in enumerate(data["records"])
    ]
    canonical_order = sorted(
        records,
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        ),
    )
    if records != canonical_order:
        raise ValueError("TUSZ analysis identities are not canonically sorted")
    paths = [row["local_edf_path"] for row in records]
    containers = [row["source_edf_container_sha256"] for row in records]
    if len(paths) != len(set(paths)) or len(containers) != len(set(containers)):
        raise ValueError("TUSZ analysis identities are not unique")
    if len(records) != binding["source_analysis_eligible_class_count"]:
        raise ValueError("TUSZ analysis identity count disagrees with source binding")

    summaries = data["split_summaries"]
    if type(summaries) is not dict or set(summaries) != set(_MODEL_TO_OFFICIAL_SPLIT):
        raise ValueError("TUSZ analysis split summaries drifted")
    if _canonical_sha256(summaries) != binding["source_split_accounting_sha256"]:
        raise ValueError("TUSZ analysis split accounting hash drifted")
    total_audit = 0
    total_alias_excluded = 0
    total_quarantined = 0
    for model_split, summary in summaries.items():
        required_summary = {
            "official_split",
            "audit_official_path_count",
            "analysis_identity_count",
            "analysis_patient_alias_count",
            "same_patient_alias_excluded_path_count",
            "quarantined_path_count",
            "analysis_identity_roster_sha256",
            "analysis_patient_alias_roster_sha256",
            "path_count_closure_verified",
        }
        if type(summary) is not dict or set(summary) != required_summary:
            raise ValueError("TUSZ analysis split summary fields drifted")
        if summary["official_split"] != _MODEL_TO_OFFICIAL_SPLIT[model_split]:
            raise ValueError("TUSZ analysis split summary mapping drifted")
        selected = [row for row in records if row["model_split"] == model_split]
        patients = sorted({row["local_patient_id"] for row in selected})
        for field in (
            "audit_official_path_count",
            "analysis_identity_count",
            "analysis_patient_alias_count",
            "same_patient_alias_excluded_path_count",
            "quarantined_path_count",
        ):
            if type(summary[field]) is not int or summary[field] < 0:
                raise ValueError(
                    "TUSZ analysis split counts must be non-negative integers"
                )
        if (
            summary["analysis_identity_count"] != len(selected)
            or summary["analysis_patient_alias_count"] != len(patients)
            or summary["analysis_identity_roster_sha256"]
            != _canonical_sha256([row["analysis_identity_id"] for row in selected])
            or summary["analysis_patient_alias_roster_sha256"]
            != _canonical_sha256(patients)
        ):
            raise ValueError("TUSZ analysis split identity summary is not replayable")
        audit_count = summary["audit_official_path_count"]
        alias_count = summary["same_patient_alias_excluded_path_count"]
        quarantine_count = summary["quarantined_path_count"]
        if (
            summary["path_count_closure_verified"] is not True
            or len(selected) + alias_count + quarantine_count != audit_count
        ):
            raise ValueError("TUSZ analysis split path accounting does not close")
        total_audit += audit_count
        total_alias_excluded += alias_count
        total_quarantined += quarantine_count
    if (
        total_audit != binding["source_audit_recording_count"]
        or total_alias_excluded
        != binding["source_same_patient_alias_excluded_path_count"]
        or total_quarantined != binding["source_quarantined_path_count"]
    ):
        raise ValueError("TUSZ analysis global path accounting drifted")

    exclusion = data["exclusion_receipt"]
    exclusion_fields = {
        "same_patient_alias_equivalence_class_count",
        "same_patient_alias_excluded_path_count",
        "quarantined_equivalence_class_count",
        "quarantined_path_count",
        "quarantined_equivalence_class_ids",
        "quarantined_equivalence_class_roster_sha256",
        "noncanonical_alias_paths_retained_as_analysis_rows",
        "quarantined_paths_retained_as_analysis_rows",
    }
    if type(exclusion) is not dict or set(exclusion) != exclusion_fields:
        raise ValueError("TUSZ analysis exclusion receipt fields drifted")
    for field in (
        "same_patient_alias_equivalence_class_count",
        "same_patient_alias_excluded_path_count",
        "quarantined_equivalence_class_count",
        "quarantined_path_count",
    ):
        if type(exclusion[field]) is not int or exclusion[field] < 0:
            raise ValueError("TUSZ analysis exclusion counts drifted")
    quarantine_ids = exclusion["quarantined_equivalence_class_ids"]
    if (
        type(quarantine_ids) is not list
        or quarantine_ids != sorted(quarantine_ids)
        or len(quarantine_ids) != len(set(quarantine_ids))
        or exclusion["quarantined_equivalence_class_count"] != len(quarantine_ids)
        or exclusion["quarantined_equivalence_class_roster_sha256"]
        != _canonical_sha256(quarantine_ids)
        or exclusion["same_patient_alias_excluded_path_count"]
        != total_alias_excluded
        or exclusion["quarantined_path_count"] != total_quarantined
        or exclusion["noncanonical_alias_paths_retained_as_analysis_rows"] is not False
        or exclusion["quarantined_paths_retained_as_analysis_rows"] is not False
    ):
        raise ValueError("TUSZ analysis exclusion receipt drifted")
    if any(
        row["exact_container_equivalence_id"] in set(quarantine_ids)
        for row in records
    ):
        raise ValueError("quarantined equivalence class leaked into analysis rows")
    if data["role_permissions"] != _role_permissions():
        raise ValueError("TUSZ analysis split role permissions drifted")
    if data["reference_access_receipt"] != _reference_access_receipt():
        raise ValueError("TUSZ analysis reference access receipt drifted")
    expected_scope = {
        "identity_only_projection": True,
        "exact_container_equivalence_is_analysis_unit": True,
        "every_analysis_unit_has_weight_one": True,
        "same_patient_alias_duplicates_cannot_be_double_counted": True,
        "cross_patient_or_split_duplicates_are_fully_quarantined": True,
        "official_split_roles_are_mutually_separated": True,
        "source_eval_execution_requires_host_admission": True,
        "source_eval_execution_authorized_by_projection": False,
        "reference_join_authorized": False,
        "findings_soz_or_performance_claim_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("TUSZ analysis scope receipt drifted")

    digest = deepcopy(data)
    digest["projection_id"] = "TUSZ-ANALYSIS-IDENTITY-V2-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["projection_id"] != "TUSZANALYSISV2-" + _canonical_sha256(digest)[:24]:
        raise ValueError("TUSZ analysis projection ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("TUSZ analysis projection receipt hash drifted")
    if source_roster is not None:
        validated_roster = validate_tusz_complete_detector_roster_v2(source_roster)
        expected_projection = _projection_from_validated_roster(validated_roster)
        if data != expected_projection:
            raise ValueError("TUSZ analysis projection disagrees with source roster")
    return data


def build_tusz_analysis_identity_projection_v2(
    source_roster: Mapping[str, object],
) -> dict[str, Any]:
    """Build the sole analysis-eligible identity denominator from roster v2."""

    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = _projection_from_validated_roster(roster)
    return validate_tusz_analysis_identity_projection_v2(
        projection, source_roster=roster
    )


__all__ = [
    "TUSZ_ANALYSIS_IDENTITY_FIELDS_V2",
    "TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_METHOD_ID",
    "TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION",
    "TUSZ_COMPLETE_DETECTOR_ROSTER_V2_METHOD_ID",
    "TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION",
    "build_tusz_analysis_identity_projection_v2",
    "build_tusz_complete_detector_roster_v2",
    "validate_tusz_analysis_identity_projection_v2",
    "validate_tusz_complete_detector_roster_v2",
]
