"""Target-blind patient-disjoint five-fold plan for a clean-room TUSZ detector.

The planner consumes four already frozen identity artifacts: the complete
container roster v2, its analysis identity projection, the complete canonical
physical duplicate audit, and the audit-derived physical analysis projection.
It never accepts a dataset path or a reference/annotation object.  Fold
assignment is based only on each source-train patient's total closed-EDF
duration and projected recording count.

This is a planning and permission artifact, not a trained model.  In
particular, every checkpoint slot remains unmaterialized, source-dev is limited
to post-freeze calibration, and source-eval execution remains locked pending a
separate host-admission receipt.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

from .tusz_canonical_physical_signal_audit_v1 import (
    TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION,
    TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION,
    validate_tusz_canonical_physical_analysis_projection_v1,
    validate_tusz_canonical_physical_duplicate_audit_v1,
)
from .tusz_complete_detector_roster_v2 import (
    TUSZ_ANALYSIS_IDENTITY_FIELDS_V2,
    TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION,
    TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION,
    validate_tusz_analysis_identity_projection_v2,
    validate_tusz_complete_detector_roster_v2,
)


TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_SCHEMA_VERSION: Final[
    str
] = "tusz_detector_cleanroom_patient_five_fold_plan_v1"
TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_METHOD_ID: Final[
    str
] = "target_blind_exact_fraction_duration_record_count_greedy_v1"
TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1: Final[int] = 5

_MODEL_TO_OFFICIAL_SPLIT: Final[dict[str, str]] = {
    "source_train": "train",
    "source_dev": "dev",
    "source_eval": "eval",
}
_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_SOURCE_DURATION_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "analysis_identity_id",
        "model_split",
        "official_split",
        "local_patient_id",
        "local_edf_path",
        "recording_duration_seconds_fraction",
    }
)
_SOURCE_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "source_roster_schema_version",
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_analysis_projection_schema_version",
        "source_analysis_projection_id",
        "source_analysis_projection_receipt_sha256",
        "source_canonical_physical_projection_schema_version",
        "source_canonical_physical_projection_id",
        "source_canonical_physical_projection_receipt_sha256",
        "source_canonical_physical_audit_schema_version",
        "source_canonical_physical_audit_id",
        "source_canonical_physical_audit_receipt_sha256",
        "source_canonical_physical_audit_shard_count",
        "source_canonical_physical_audit_shard_receipt_roster_sha256",
        "source_canonical_physical_audit_all_partition_indices_present_once",
        "source_canonical_physical_outcome_receipt_roster_sha256",
        "source_analysis_identity_count",
        "canonical_physical_projected_identity_count",
        "canonical_physical_projected_identity_roster_sha256",
        "source_record_duration_binding_sha256",
        "canonical_physical_signal_duplicate_audit_complete",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(
    value: object, fields: frozenset[str] | set[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a non-empty normalized string")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _positive_integer(value: object, context: str) -> int:
    result = _nonnegative_integer(value, context)
    if result < 1:
        raise ValueError(f"{context} must be positive")
    return result


def _fraction(value: object, context: str, *, positive: bool) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
        or (positive and value[0] <= 0)
        or (not positive and value[0] < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be a {qualifier} fraction")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} fraction must be reduced")
    return result


def _fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _source_duration_rows_from_validated(
    *,
    roster: Mapping[str, Any],
    projection: Mapping[str, Any],
    physical_audit: Mapping[str, Any],
    physical_projection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    physical_binding = _strict_object(
        physical_projection["source_binding"],
        {
            "source_roster_id",
            "source_roster_receipt_sha256",
            "source_analysis_projection_id",
            "source_analysis_projection_receipt_sha256",
            "source_canonical_physical_audit_id",
            "source_canonical_physical_audit_receipt_sha256",
        },
        "canonical physical projection source binding",
    )
    expected_physical_binding = {
        "source_roster_id": roster["roster_id"],
        "source_roster_receipt_sha256": roster["receipt_sha256"],
        "source_analysis_projection_id": projection["projection_id"],
        "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
        "source_canonical_physical_audit_id": physical_audit["audit_id"],
        "source_canonical_physical_audit_receipt_sha256": physical_audit[
            "receipt_sha256"
        ],
    }
    if physical_binding != expected_physical_binding:
        raise ValueError(
            "canonical physical projection does not bind the supplied v2 sources"
        )
    _identifier(
        physical_binding["source_canonical_physical_audit_id"],
        "canonical physical audit ID",
    )
    _sha256(
        physical_binding["source_canonical_physical_audit_receipt_sha256"],
        "canonical physical audit receipt",
    )
    if physical_projection["role_permissions"] != projection["role_permissions"]:
        raise ValueError("canonical physical projection split permissions drifted")
    if (
        physical_projection["reference_access_receipt"]
        != projection["reference_access_receipt"]
    ):
        raise ValueError("canonical physical projection reference scope drifted")
    physical_scope = physical_projection["scope_receipt"]
    if (
        physical_scope.get("canonical_physical_signal_duplicate_audit_complete")
        is not True
        or physical_scope.get("one_unit_per_safe_physical_equivalence_class")
        is not True
        or physical_scope.get("cross_patient_or_split_physical_duplicates_quarantined")
        is not True
        or physical_scope.get("same_patient_same_split_physical_aliases_deduplicated")
        is not True
        or physical_scope.get("reference_join_authorized") is not False
        or physical_scope.get("model_performance_claim_authorized") is not False
    ):
        raise PermissionError(
            "fold planning requires a completed canonical physical projection"
        )

    projection_by_identity = {
        row["analysis_identity_id"]: row for row in projection["records"]
    }
    roster_by_path = {row["recording_id"]: row for row in roster["records"]}
    rows: list[dict[str, Any]] = []
    for physical_row in physical_projection["records"]:
        identity = physical_row["analysis_identity_id"]
        projected = projection_by_identity.get(identity)
        if projected is None or any(
            physical_row[field] != projected[field]
            for field in TUSZ_ANALYSIS_IDENTITY_FIELDS_V2
        ):
            raise ValueError(
                "canonical physical row does not replay its analysis identity"
            )
        roster_row = roster_by_path.get(physical_row["local_edf_path"])
        if roster_row is None:
            raise ValueError("canonical physical row is absent from roster v2")
        if (
            roster_row["patient_id"] != physical_row["local_patient_id"]
            or roster_row["official_split"] != physical_row["official_split"]
            or roster_row["benchmark_split"] != physical_row["model_split"]
            or roster_row["container_sha256"]
            != physical_row["source_edf_container_sha256"]
        ):
            raise ValueError("roster duration source identity binding drifted")
        duration = _fraction(
            roster_row["recording_duration_fraction"],
            "closed EDF recording duration",
            positive=True,
        )
        rows.append(
            {
                "analysis_identity_id": identity,
                "model_split": physical_row["model_split"],
                "official_split": physical_row["official_split"],
                "local_patient_id": physical_row["local_patient_id"],
                "local_edf_path": physical_row["local_edf_path"],
                "recording_duration_seconds_fraction": _fraction_json(duration),
            }
        )
    rows.sort(
        key=lambda row: (
            row["model_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        )
    )
    if not rows:
        raise ValueError("canonical physical projection has no analysis records")
    identities = [row["analysis_identity_id"] for row in rows]
    paths = [row["local_edf_path"] for row in rows]
    if len(identities) != len(set(identities)) or len(paths) != len(set(paths)):
        raise ValueError("canonical physical duration rows are not unique")
    inventory = physical_projection["projection_inventory"]
    if (
        inventory["source_analysis_identity_count"] != len(projection["records"])
        or inventory["projected_analysis_identity_count"] != len(rows)
        or inventory["projected_identity_roster_sha256"]
        != _canonical_sha256(identities)
        or inventory["path_accounting_verified"] is not True
    ):
        raise ValueError("canonical physical projection denominator drifted")
    for split in _MODEL_TO_OFFICIAL_SPLIT:
        if not any(row["model_split"] == split for row in rows):
            raise ValueError(f"canonical physical projection has no {split} rows")
    return rows


def _source_binding_from_validated(
    *,
    roster: Mapping[str, Any],
    projection: Mapping[str, Any],
    physical_audit: Mapping[str, Any],
    physical_projection: Mapping[str, Any],
    source_duration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = [row["analysis_identity_id"] for row in source_duration_rows]
    shard_inventory = physical_audit["shard_inventory"]
    return {
        "source_roster_schema_version": roster["schema_version"],
        "source_roster_id": roster["roster_id"],
        "source_roster_receipt_sha256": roster["receipt_sha256"],
        "source_analysis_projection_schema_version": projection["schema_version"],
        "source_analysis_projection_id": projection["projection_id"],
        "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
        "source_canonical_physical_projection_schema_version": physical_projection[
            "schema_version"
        ],
        "source_canonical_physical_projection_id": physical_projection["projection_id"],
        "source_canonical_physical_projection_receipt_sha256": physical_projection[
            "receipt_sha256"
        ],
        "source_canonical_physical_audit_schema_version": physical_audit[
            "schema_version"
        ],
        "source_canonical_physical_audit_id": physical_audit["audit_id"],
        "source_canonical_physical_audit_receipt_sha256": physical_audit[
            "receipt_sha256"
        ],
        "source_canonical_physical_audit_shard_count": shard_inventory[
            "shard_count"
        ],
        "source_canonical_physical_audit_shard_receipt_roster_sha256": (
            shard_inventory["shard_receipt_roster_sha256"]
        ),
        "source_canonical_physical_audit_all_partition_indices_present_once": (
            shard_inventory["all_partition_indices_present_once"]
        ),
        "source_canonical_physical_outcome_receipt_roster_sha256": (
            _canonical_sha256(
                [row["receipt_sha256"] for row in physical_audit["outcomes"]]
            )
        ),
        "source_analysis_identity_count": len(projection["records"]),
        "canonical_physical_projected_identity_count": len(source_duration_rows),
        "canonical_physical_projected_identity_roster_sha256": _canonical_sha256(
            identities
        ),
        "source_record_duration_binding_sha256": _canonical_sha256(
            source_duration_rows
        ),
        "canonical_physical_signal_duplicate_audit_complete": True,
    }


def _validate_sources(
    *,
    source_roster: object,
    source_analysis_projection: object,
    source_canonical_physical_audit: object,
    source_canonical_physical_projection: object,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = validate_tusz_analysis_identity_projection_v2(
        source_analysis_projection,
        source_roster=roster,
    )
    physical_audit = validate_tusz_canonical_physical_duplicate_audit_v1(
        source_canonical_physical_audit,
        source_roster=roster,
        source_projection=projection,
    )
    if (
        physical_audit["scope_receipt"][
            "canonical_physical_signal_duplicate_audit_complete"
        ]
        is not True
        or physical_audit["scope_receipt"]["analysis_projection_authorized"]
        is not True
        or physical_audit["shard_inventory"][
            "all_partition_indices_present_once"
        ]
        is not True
    ):
        raise PermissionError(
            "clean-room planning requires a complete canonical physical audit"
        )
    physical = validate_tusz_canonical_physical_analysis_projection_v1(
        source_canonical_physical_projection,
        audit=physical_audit,
        source_roster=roster,
        source_projection=projection,
    )
    rows = _source_duration_rows_from_validated(
        roster=roster,
        projection=projection,
        physical_audit=physical_audit,
        physical_projection=physical,
    )
    binding = _source_binding_from_validated(
        roster=roster,
        projection=projection,
        physical_audit=physical_audit,
        physical_projection=physical,
        source_duration_rows=rows,
    )
    return binding, rows


def _record_roster_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["local_patient_id"],
            row["local_edf_path"],
            row["analysis_identity_id"],
        ),
    )
    patients = sorted({str(row["local_patient_id"]) for row in ordered})
    identities = sorted(str(row["analysis_identity_id"]) for row in ordered)
    paths = sorted(str(row["local_edf_path"]) for row in ordered)
    duration = sum(
        (
            _fraction(
                row["recording_duration_seconds_fraction"],
                "recording duration",
                positive=True,
            )
            for row in ordered
        ),
        Fraction(0, 1),
    )
    return {
        "patient_count": len(patients),
        "recording_count": len(ordered),
        "duration_seconds_fraction": _fraction_json(duration),
        "patient_ids": patients,
        "analysis_identity_ids": identities,
        "local_edf_paths": paths,
        "patient_roster_sha256": _canonical_sha256(patients),
        "analysis_identity_roster_sha256": _canonical_sha256(identities),
        "local_edf_path_roster_sha256": _canonical_sha256(paths),
        "record_duration_binding_sha256": _canonical_sha256(ordered),
    }


def _source_split_rosters(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split, official_split in _MODEL_TO_OFFICIAL_SPLIT.items():
        selected = [row for row in rows if row["model_split"] == split]
        result[split] = {
            "model_split": split,
            "official_split": official_split,
            **_record_roster_view(selected),
        }
    return result


def _patient_balance_rows(
    train_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in train_rows:
        by_patient.setdefault(str(row["local_patient_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for patient in sorted(by_patient):
        selected = sorted(
            by_patient[patient], key=lambda row: str(row["local_edf_path"])
        )
        duration = sum(
            (
                _fraction(
                    row["recording_duration_seconds_fraction"],
                    "patient recording duration",
                    positive=True,
                )
                for row in selected
            ),
            Fraction(0, 1),
        )
        identities = sorted(str(row["analysis_identity_id"]) for row in selected)
        paths = sorted(str(row["local_edf_path"]) for row in selected)
        result.append(
            {
                "local_patient_id": patient,
                "record_count": len(selected),
                "total_duration_seconds_fraction": _fraction_json(duration),
                "analysis_identity_ids": identities,
                "local_edf_paths": paths,
                "analysis_identity_roster_sha256": _canonical_sha256(identities),
                "local_edf_path_roster_sha256": _canonical_sha256(paths),
                "record_duration_binding_sha256": _canonical_sha256(selected),
            }
        )
    return result


def _assignment_order(
    patient_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    total_duration = sum(
        (
            _fraction(
                row["total_duration_seconds_fraction"],
                "patient total duration",
                positive=True,
            )
            for row in patient_rows
        ),
        Fraction(0, 1),
    )
    total_records = sum(int(row["record_count"]) for row in patient_rows)
    if total_duration <= 0 or total_records <= 0:
        raise ValueError("source-train balance denominator is empty")

    def key(row: Mapping[str, Any]) -> tuple[Fraction, Fraction, Fraction, str]:
        duration_share = (
            _fraction(
                row["total_duration_seconds_fraction"],
                "patient total duration",
                positive=True,
            )
            / total_duration
        )
        record_share = Fraction(int(row["record_count"]), total_records)
        normalized_squared_mass = duration_share**2 + record_share**2
        return (
            -normalized_squared_mass,
            -duration_share,
            -record_share,
            str(row["local_patient_id"]),
        )

    return [str(row["local_patient_id"]) for row in sorted(patient_rows, key=key)]


def _global_balance_objective(
    *,
    fold_durations: Sequence[Fraction],
    fold_record_counts: Sequence[int],
    total_duration: Fraction,
    total_record_count: int,
) -> Fraction:
    target = Fraction(1, TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1)
    return sum(
        (
            (duration / total_duration - target) ** 2
            + (Fraction(records, total_record_count) - target) ** 2
            for duration, records in zip(fold_durations, fold_record_counts)
        ),
        Fraction(0, 1),
    )


def _assign_patients(
    patient_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, int], list[Fraction], list[int]]:
    if len(patient_rows) < TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1:
        raise ValueError("five-fold planning requires at least five train patients")
    by_patient = {str(row["local_patient_id"]): row for row in patient_rows}
    order = _assignment_order(patient_rows)
    total_duration = sum(
        (
            _fraction(
                row["total_duration_seconds_fraction"],
                "patient duration",
                positive=True,
            )
            for row in patient_rows
        ),
        Fraction(0, 1),
    )
    total_records = sum(int(row["record_count"]) for row in patient_rows)
    fold_durations = [Fraction(0, 1)] * TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    fold_records = [0] * TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    assignments: dict[str, int] = {}
    for patient in order:
        row = by_patient[patient]
        duration = _fraction(
            row["total_duration_seconds_fraction"],
            "patient duration",
            positive=True,
        )
        records = int(row["record_count"])
        candidates: list[tuple[Fraction, int]] = []
        for fold_id in range(TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1):
            candidate_durations = list(fold_durations)
            candidate_records = list(fold_records)
            candidate_durations[fold_id] += duration
            candidate_records[fold_id] += records
            objective = _global_balance_objective(
                fold_durations=candidate_durations,
                fold_record_counts=candidate_records,
                total_duration=total_duration,
                total_record_count=total_records,
            )
            candidates.append((objective, fold_id))
        _objective, selected_fold = min(candidates)
        assignments[patient] = selected_fold
        fold_durations[selected_fold] += duration
        fold_records[selected_fold] += records
    if set(assignments) != set(by_patient):
        raise ValueError("patient assignment denominator does not close")
    if any(value == 0 for value in fold_records):
        raise ValueError("deterministic balance produced an empty held-out fold")
    return order, assignments, fold_durations, fold_records


def _balance_contract() -> dict[str, Any]:
    return {
        "fold_count": TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1,
        "assignment_unit": "local_patient_id",
        "balance_metric_allowlist": [
            "patient_total_eeg_duration_seconds_fraction",
            "patient_projected_record_count",
        ],
        "arithmetic": "exact_rational_fraction_no_float_rounding",
        "patient_priority_rule": (
            "descending_sum_of_squared_normalized_duration_and_record_mass;"
            "then_descending_duration_share;then_descending_record_share;"
            "then_local_patient_id_ascending"
        ),
        "fold_choice_rule": (
            "minimum_global_sum_over_folds_of_squared_duration_share_and_"
            "record_share_deviation_from_one_fifth;then_fold_id_ascending"
        ),
        "random_seed_used": False,
        "randomized_search_used": False,
        "patient_identity_used_only_for_grouping_and_final_metric_tie_break": True,
        "event_or_channel_target_used": False,
        "reference_label_used": False,
    }


def _role_permissions() -> dict[str, dict[str, Any]]:
    return {
        "source_train": {
            "fold_assignment_authorized": True,
            "detector_parameter_fit_authorized": True,
            "fit_scope": "current_fold_training_patients_and_records_only",
            "out_of_fold_inference_authorized_after_fold_freeze": True,
            "development_calibration_authorized": False,
            "locked_evaluation_authorized": False,
            "reference_access_authorized_by_this_plan": False,
        },
        "source_dev": {
            "fold_assignment_authorized": False,
            "detector_parameter_fit_authorized": False,
            "preprocessing_fit_authorized": False,
            "checkpoint_selection_or_update_authorized": False,
            "calibration_only": True,
            "calibration_requires_all_fold_artifacts_frozen": True,
            "prediction_execution_after_all_fold_artifacts_frozen": True,
            "locked_evaluation_authorized": False,
            "reference_access_authorized_by_this_plan": False,
        },
        "source_eval": {
            "fold_assignment_authorized": False,
            "detector_parameter_fit_authorized": False,
            "preprocessing_fit_authorized": False,
            "calibration_authorized": False,
            "execution_authorized_by_this_plan": False,
            "host_admission_required": True,
            "host_admission_present": False,
            "separate_content_addressed_host_admission_required": True,
            "reference_access_authorized_by_this_plan": False,
        },
    }


def _data_access_receipt() -> dict[str, Any]:
    return {
        "planner_public_api_accepts_dataset_path": False,
        "planner_public_api_accepts_reference_or_target_payload": False,
        "complete_canonical_physical_audit_required": True,
        "canonical_audit_values_used_for_lineage_only": True,
        "raw_edf_samples_read_by_planner": False,
        "csv_bi_files_opened": 0,
        "csv_bi_bytes_read": 0,
        "edf_annotations_read": False,
        "seizure_event_intervals_or_labels_read": False,
        "soz_or_channel_targets_read": False,
        "spreadsheet_or_doctor_information_read": False,
        "clinical_text_or_report_read": False,
        "balance_values_read": [
            "closed_edf_recording_duration_fraction",
            "canonical_physical_projected_record_count",
        ],
        "identity_values_read_for_grouping_and_tie_break_only": True,
    }


def _scope_receipt() -> dict[str, Any]:
    return {
        "target_blind_fold_assignment": True,
        "patient_disjoint_five_fold_source_train": True,
        "canonical_physical_analysis_denominator_only": True,
        "fold_local_preprocessing_fit_only": True,
        "fold_local_checkpoint_training_exposure_only": True,
        "source_dev_calibration_only": True,
        "source_eval_locked_pending_host_admission": True,
        "checkpoint_artifacts_materialized": False,
        "detector_trained": False,
        "performance_or_sota_claim_authorized": False,
        "clinical_use_authorized": False,
    }


def _fold_permission(
    *,
    train_roster: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "fit_scope": "source_train_current_fold_training_records_only",
        "fit_patient_roster_sha256": train_roster["patient_roster_sha256"],
        "fit_analysis_identity_roster_sha256": train_roster[
            "analysis_identity_roster_sha256"
        ],
        "fit_local_edf_path_roster_sha256": train_roster[
            "local_edf_path_roster_sha256"
        ],
        "held_out_fit_authorized": False,
        "source_dev_fit_authorized": False,
        "source_eval_fit_authorized": False,
        "cross_fold_shared_fitted_statistics_authorized": False,
        "fixed_stateless_operations_may_be_shared": True,
        "fold_training_transform_authorized": True,
        "fold_held_out_transform_authorized_after_fit_freeze": True,
        "source_dev_transform_authorized_after_all_fold_artifacts_freeze": True,
        "source_eval_transform_authorized_by_this_plan": False,
        "source_eval_transform_requires_host_admission": True,
        "preprocessing_artifact_must_be_fold_content_addressed": True,
    }


def _checkpoint_contract(
    *,
    fold_id: int,
    train_roster: Mapping[str, Any],
    held_out_roster: Mapping[str, Any],
    source_split_rosters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_slot_id": f"TUSZ-CLEANROOM-DETECTOR-FOLD-{fold_id:02d}",
        "materialization_status": "planned_untrained",
        "checkpoint_artifact_sha256": None,
        "permitted_update_patient_roster_sha256": train_roster["patient_roster_sha256"],
        "permitted_update_analysis_identity_roster_sha256": train_roster[
            "analysis_identity_roster_sha256"
        ],
        "held_out_inference_patient_roster_sha256": held_out_roster[
            "patient_roster_sha256"
        ],
        "held_out_inference_analysis_identity_roster_sha256": held_out_roster[
            "analysis_identity_roster_sha256"
        ],
        "source_dev_patient_roster_sha256": source_split_rosters["source_dev"][
            "patient_roster_sha256"
        ],
        "source_dev_analysis_identity_roster_sha256": source_split_rosters[
            "source_dev"
        ]["analysis_identity_roster_sha256"],
        "source_eval_patient_roster_sha256": source_split_rosters["source_eval"][
            "patient_roster_sha256"
        ],
        "source_eval_analysis_identity_roster_sha256": source_split_rosters[
            "source_eval"
        ]["analysis_identity_roster_sha256"],
        "held_out_exposure_mode": "frozen_checkpoint_inference_only",
        "source_dev_exposure_mode": (
            "calibration_inference_only_after_all_five_fold_artifacts_freeze"
        ),
        "source_eval_exposure_mode": "locked_no_execution_without_host_admission",
        "held_out_gradient_or_checkpoint_update_authorized": False,
        "source_dev_gradient_or_checkpoint_update_authorized": False,
        "source_eval_any_exposure_authorized_by_this_plan": False,
        "checkpoint_content_address_required_before_inference": True,
        "training_run_receipt_required": True,
        "actual_exposure_roster_receipt_required": True,
    }


def _fold_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    fold_durations: Sequence[Fraction],
    fold_records: Sequence[int],
    source_split_rosters: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    total_duration = sum(fold_durations, Fraction(0, 1))
    total_records = sum(fold_records)
    folds: list[dict[str, Any]] = []
    all_patients = {str(row["local_patient_id"]) for row in train_rows}
    for fold_id in range(TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1):
        held_patients = sorted(
            patient for patient, assigned in assignments.items() if assigned == fold_id
        )
        train_patients = sorted(all_patients.difference(held_patients))
        held_set = set(held_patients)
        training_set = set(train_patients)
        held_rows = [row for row in train_rows if row["local_patient_id"] in held_set]
        fit_rows = [
            row for row in train_rows if row["local_patient_id"] in training_set
        ]
        held_roster = _record_roster_view(held_rows)
        fit_roster = _record_roster_view(fit_rows)
        if set(held_roster["patient_ids"]).intersection(
            fit_roster["patient_ids"]
        ) or set(held_roster["analysis_identity_ids"]).intersection(
            fit_roster["analysis_identity_ids"]
        ):
            raise ValueError("one fold leaks a held-out patient or recording")
        fold: dict[str, Any] = {
            "fold_id": fold_id,
            "train_roster": fit_roster,
            "held_out_roster": held_roster,
            "balance_receipt": {
                "held_out_duration_share_fraction": _fraction_json(
                    fold_durations[fold_id] / total_duration
                ),
                "held_out_record_share_fraction": _fraction_json(
                    Fraction(fold_records[fold_id], total_records)
                ),
                "normalized_joint_squared_deviation_fraction": _fraction_json(
                    (fold_durations[fold_id] / total_duration - Fraction(1, 5)) ** 2
                    + (Fraction(fold_records[fold_id], total_records) - Fraction(1, 5))
                    ** 2
                ),
            },
            "preprocessing_fit_permission": _fold_permission(train_roster=fit_roster),
            "checkpoint_exposure_contract": _checkpoint_contract(
                fold_id=fold_id,
                train_roster=fit_roster,
                held_out_roster=held_roster,
                source_split_rosters=source_split_rosters,
            ),
            "fold_receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
        digest = deepcopy(fold)
        digest["fold_receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
        fold["fold_receipt_sha256"] = _canonical_sha256(digest)
        folds.append(fold)
    return folds


def _materialize_plan(
    *,
    source_binding: Mapping[str, Any],
    source_duration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = deepcopy(list(source_duration_rows))
    split_rosters = _source_split_rosters(rows)
    train_rows = [row for row in rows if row["model_split"] == "source_train"]
    patient_rows = _patient_balance_rows(train_rows)
    order, assignments, fold_durations, fold_records = _assign_patients(patient_rows)
    assignment_rows = [
        {
            "local_patient_id": patient,
            "held_out_fold_id": assignments[patient],
        }
        for patient in sorted(assignments)
    ]
    folds = _fold_rows(
        train_rows=train_rows,
        assignments=assignments,
        fold_durations=fold_durations,
        fold_records=fold_records,
        source_split_rosters=split_rosters,
    )
    total_train_duration = sum(fold_durations, Fraction(0, 1))
    total_train_records = sum(fold_records)
    objective = _global_balance_objective(
        fold_durations=fold_durations,
        fold_record_counts=fold_records,
        total_duration=total_train_duration,
        total_record_count=total_train_records,
    )
    body: dict[str, Any] = {
        "schema_version": TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_SCHEMA_VERSION,
        "method_id": TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_METHOD_ID,
        "plan_id": "TUSZ-DETECTOR-CLEANROOM-FOLD-PLAN-V1-PENDING",
        "source_binding": deepcopy(dict(source_binding)),
        "fold_count": TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1,
        "balance_contract": _balance_contract(),
        "source_record_duration_rows": rows,
        "source_split_rosters": split_rosters,
        "source_train_patient_balance_rows": patient_rows,
        "assignment_order_patient_ids": order,
        "patient_fold_assignments": assignment_rows,
        "balance_outcome": {
            "source_train_total_duration_seconds_fraction": _fraction_json(
                total_train_duration
            ),
            "source_train_total_record_count": total_train_records,
            "fold_held_out_duration_seconds_fractions": [
                _fraction_json(value) for value in fold_durations
            ],
            "fold_held_out_record_counts": list(fold_records),
            "final_global_normalized_squared_deviation_fraction": _fraction_json(
                objective
            ),
            "every_patient_held_out_exactly_once": True,
            "every_record_held_out_exactly_once": True,
            "duration_and_record_denominators_close": True,
        },
        "folds": folds,
        "role_permissions": _role_permissions(),
        "data_access_receipt": _data_access_receipt(),
        "scope_receipt": _scope_receipt(),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_digest = deepcopy(body)
    body["plan_id"] = "TUSZDETCLEANFOLDV1-" + _canonical_sha256(id_digest)[:24]
    receipt_digest = deepcopy(body)
    receipt_digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    body["receipt_sha256"] = _canonical_sha256(receipt_digest)
    return body


def _validate_source_binding(value: object) -> dict[str, Any]:
    binding = _strict_object(value, _SOURCE_BINDING_FIELDS, "source binding")
    if (
        binding["source_roster_schema_version"]
        != TUSZ_COMPLETE_DETECTOR_ROSTER_V2_SCHEMA_VERSION
        or binding["source_analysis_projection_schema_version"]
        != TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION
        or binding["source_canonical_physical_projection_schema_version"]
        != TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION
        or binding["source_canonical_physical_audit_schema_version"]
        != TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION
    ):
        raise ValueError("clean-room fold source schema drifted")
    for field in (
        "source_roster_id",
        "source_analysis_projection_id",
        "source_canonical_physical_projection_id",
        "source_canonical_physical_audit_id",
    ):
        _identifier(binding[field], field)
    for field in (
        "source_roster_receipt_sha256",
        "source_analysis_projection_receipt_sha256",
        "source_canonical_physical_projection_receipt_sha256",
        "source_canonical_physical_audit_receipt_sha256",
        "source_canonical_physical_audit_shard_receipt_roster_sha256",
        "source_canonical_physical_outcome_receipt_roster_sha256",
        "canonical_physical_projected_identity_roster_sha256",
        "source_record_duration_binding_sha256",
    ):
        _sha256(binding[field], field)
    _positive_integer(
        binding["source_analysis_identity_count"], "source analysis identity count"
    )
    _positive_integer(
        binding["canonical_physical_projected_identity_count"],
        "canonical physical projected identity count",
    )
    _positive_integer(
        binding["source_canonical_physical_audit_shard_count"],
        "canonical physical audit shard count",
    )
    if (
        binding[
            "source_canonical_physical_audit_all_partition_indices_present_once"
        ]
        is not True
    ):
        raise ValueError("canonical physical audit shard partition is incomplete")
    if binding["canonical_physical_signal_duplicate_audit_complete"] is not True:
        raise PermissionError("canonical physical duplicate audit is incomplete")
    return binding


def _validate_source_duration_rows(
    value: object, *, source_binding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source duration rows must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for index, value_row in enumerate(value):
        row = _strict_object(
            value_row, _SOURCE_DURATION_ROW_FIELDS, f"source duration row {index}"
        )
        _identifier(row["analysis_identity_id"], "analysis identity")
        split = row["model_split"]
        if (
            split not in _MODEL_TO_OFFICIAL_SPLIT
            or row["official_split"] != _MODEL_TO_OFFICIAL_SPLIT[split]
        ):
            raise ValueError("source duration split mapping drifted")
        patient = _identifier(row["local_patient_id"], "patient ID")
        path = _identifier(row["local_edf_path"], "local EDF path")
        path_parts = path.split("/")
        if (
            path.startswith("/")
            or "\\" in path
            or ".." in path_parts
            or len(path_parts) < 3
            or path_parts[0] != row["official_split"]
            or path_parts[1] != patient
            or not path.lower().endswith(".edf")
        ):
            raise ValueError("source duration row path binding drifted")
        _fraction(
            row["recording_duration_seconds_fraction"],
            "recording duration",
            positive=True,
        )
        rows.append(row)
    canonical = sorted(
        rows,
        key=lambda row: (
            row["model_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        ),
    )
    if rows != canonical:
        raise ValueError("source duration rows are not canonically sorted")
    identities = [row["analysis_identity_id"] for row in rows]
    paths = [row["local_edf_path"] for row in rows]
    if len(identities) != len(set(identities)) or len(paths) != len(set(paths)):
        raise ValueError("source duration identities or paths repeat")
    if (
        len(rows) != source_binding["canonical_physical_projected_identity_count"]
        or _canonical_sha256(identities)
        != source_binding["canonical_physical_projected_identity_roster_sha256"]
        or _canonical_sha256(rows)
        != source_binding["source_record_duration_binding_sha256"]
    ):
        raise ValueError("source duration rows disagree with source binding")
    for split in _MODEL_TO_OFFICIAL_SPLIT:
        if not any(row["model_split"] == split for row in rows):
            raise ValueError(f"source duration rows have no {split} records")
    return rows


def validate_tusz_detector_cleanroom_fold_plan_v1(
    payload: object,
    *,
    source_roster: object | None = None,
    source_analysis_projection: object | None = None,
    source_canonical_physical_audit: object | None = None,
    source_canonical_physical_projection: object | None = None,
) -> dict[str, Any]:
    """Strictly replay a clean-room plan, optionally against all four sources."""

    required = {
        "schema_version",
        "method_id",
        "plan_id",
        "source_binding",
        "fold_count",
        "balance_contract",
        "source_record_duration_rows",
        "source_split_rosters",
        "source_train_patient_balance_rows",
        "assignment_order_patient_ids",
        "patient_fold_assignments",
        "balance_outcome",
        "folds",
        "role_permissions",
        "data_access_receipt",
        "scope_receipt",
        "receipt_sha256",
    }
    data = _strict_object(payload, required, "clean-room fold plan")
    if (
        data["schema_version"] != TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_SCHEMA_VERSION
        or data["method_id"] != TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_METHOD_ID
        or data["fold_count"] != TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1
    ):
        raise ValueError("clean-room five-fold schema or method drifted")
    binding = _validate_source_binding(data["source_binding"])
    rows = _validate_source_duration_rows(
        data["source_record_duration_rows"], source_binding=binding
    )
    expected = _materialize_plan(source_binding=binding, source_duration_rows=rows)
    if data != expected:
        raise ValueError(
            "clean-room fold plan is not a deterministic target-blind replay"
        )

    supplied = (
        source_roster,
        source_analysis_projection,
        source_canonical_physical_audit,
        source_canonical_physical_projection,
    )
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError("all four clean-room source artifacts are required")
        expected_binding, expected_rows = _validate_sources(
            source_roster=source_roster,
            source_analysis_projection=source_analysis_projection,
            source_canonical_physical_audit=source_canonical_physical_audit,
            source_canonical_physical_projection=(source_canonical_physical_projection),
        )
        if binding != expected_binding or rows != expected_rows:
            raise ValueError("clean-room fold plan disagrees with frozen sources")
    return data


def build_tusz_detector_cleanroom_fold_plan_v1(
    *,
    source_roster: object,
    source_analysis_projection: object,
    source_canonical_physical_audit: object,
    source_canonical_physical_projection: object,
) -> dict[str, Any]:
    """Build the sole deterministic five-fold plan from the four sources."""

    binding, rows = _validate_sources(
        source_roster=source_roster,
        source_analysis_projection=source_analysis_projection,
        source_canonical_physical_audit=source_canonical_physical_audit,
        source_canonical_physical_projection=source_canonical_physical_projection,
    )
    plan = _materialize_plan(source_binding=binding, source_duration_rows=rows)
    return validate_tusz_detector_cleanroom_fold_plan_v1(
        plan,
        source_roster=source_roster,
        source_analysis_projection=source_analysis_projection,
        source_canonical_physical_audit=source_canonical_physical_audit,
        source_canonical_physical_projection=source_canonical_physical_projection,
    )


__all__ = [
    "TUSZ_DETECTOR_CLEANROOM_FOLD_COUNT_V1",
    "TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_METHOD_ID",
    "TUSZ_DETECTOR_CLEANROOM_FOLD_PLAN_V1_SCHEMA_VERSION",
    "build_tusz_detector_cleanroom_fold_plan_v1",
    "validate_tusz_detector_cleanroom_fold_plan_v1",
]
