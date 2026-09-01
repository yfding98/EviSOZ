"""Post-freeze source-development calibration for EventNet's native decoder.

Stage 1 validates a complete prediction-only EventNet decoder grid.  Only
after that exact sealed carrier is revalidated does this module derive and
open public TUSZ ``dev/*.csv_bi`` files.  It projects exact global
``TERM,seiz`` intervals, scores every policy on every expected recording, and
selects a high-recall navigation operating point when the predeclared pooled
and patient-macro sensitivity floors are met.

No EDF annotation stream, channel-level annotation, Excel field, physician
label, report text, private data or source-evaluation reference has an input
slot.  This source-development receipt is not itself a source-evaluation
admission, production qualification, SOTA claim, clinical seizure assertion,
or SOZ/Finding fact source.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from .continuous_detection_benchmark import (
    DEFAULT_TOLERANCES_SECONDS,
    aggregate_continuous_detection_metrics,
)
from .continuous_detection_source_dev_join import (
    parse_tusz_term_seiz_reference_bytes,
)
from .eventnet_full_record_adapter import EVENTNET_PROVIDER_ID
from .eventnet_native_decoder_grid import (
    ValidatedEventNetDecoderGridBundle,
    eventnet_decoder_policy_rows,
    revalidate_eventnet_native_decoder_grid_bundle_without_references,
    validate_eventnet_native_decoder_policy,
)


EVENTNET_SOURCE_DEV_CALIBRATION_SCHEMA_VERSION = (
    "eventnet_native_decoder_source_dev_calibration_v1"
)
EVENTNET_SOURCE_DEV_CALIBRATION_METHOD_ID = (
    "postfreeze_global_term_seiz_native_grid_high_recall_selection_v1"
)
EVENTNET_SOURCE_DEV_CALIBRATION_FILENAME = "calibration_receipt.json"
EVENTNET_SOURCE_DEV_REFERENCE_PARSER_ID = "tusz_csv_bi_exact_TERM_seiz_projection_v1"
EVENTNET_SOURCE_DEV_REFERENCE_MAPPING_ID = (
    "source_dev_recording_relative_edf_suffix_to_csv_bi_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SELECTION_FIELDS = {
    "minimum_pooled_event_sensitivity",
    "minimum_patient_macro_event_sensitivity",
    "false_alarm_budgets_per_24h",
    "onset_tie_tolerance_seconds",
}
_CALIBRATION_SCOPE = {
    "prediction_grid_validated_before_first_reference_open": True,
    "source_dev_global_term_seiz_intervals_used_for_calibration_only": True,
    "channel_annotations_used": False,
    "edf_annotations_used": False,
    "excel_or_clinical_labels_used": False,
    "private_data_used": False,
    "source_eval_used": False,
    "detector_or_raw_prediction_rerun": False,
    "raw_prediction_or_grid_artifacts_mutated": False,
    "zero_alarm_and_seizure_free_records_retained": True,
    "eventnet_alarm_start_is_clinical_onset": False,
    "findings_or_soz_evidence_authorized": False,
    "source_eval_scoring_authorized_by_this_receipt": False,
    "production_or_sota_claim_authorized": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def eventnet_native_calibration_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _finite(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{context} must be finite and >= {minimum}")
    return result


def _safe_reference_path(root: Path, recording_id: str) -> Path:
    relative = PurePosixPath(recording_id)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 3
        or relative.parts[0] != "dev"
        or relative.suffix.lower() != ".edf"
        or any(part in {"eval", "source_eval", "private"} for part in relative.parts)
    ):
        raise ValueError("EventNet calibration recording is not safe source_dev")
    reference_relative = relative.with_suffix(".csv_bi")
    resolved_root = root.resolve(strict=True)
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise ValueError(
            "EventNet source-dev reference root must be a regular directory"
        )
    path = resolved_root.joinpath(*reference_relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("EventNet source-dev reference must be a regular file")
    resolved = path.resolve(strict=True)
    resolved.relative_to(resolved_root)
    return resolved


def _prediction_intervals(value: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "start_seconds": float(row["start_offset_seconds"]),
            "stop_seconds": float(row["stop_offset_seconds"]),
        }
        for row in value
    ]


def _candidate_scalar(
    candidate: Mapping[str, Any], path: Sequence[str]
) -> float | None:
    value: Any = candidate
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    return float(value)


def _finite_or_infinity(value: float | None) -> float:
    if value is not None and math.isfinite(value):
        return value
    return math.inf


def calibrate_eventnet_native_decoder_grid_source_dev(
    grid_bundle: ValidatedEventNetDecoderGridBundle,
    *,
    source_dev_reference_root: str | Path,
) -> dict[str, Any]:
    """Join source-dev truth only after the complete prediction grid freezes."""

    # Ordering is part of the evidence boundary: no reference-root resolution,
    # path construction or open occurs before this exact sealed-type replay.
    frozen = revalidate_eventnet_native_decoder_grid_bundle_without_references(
        grid_bundle
    )
    grid = frozen.grid_definition()
    grid_receipt = frozen.bundle_receipt()
    policies = eventnet_decoder_policy_rows(grid)
    prediction_by_pair = {
        (row.recording_id, row.policy_id): row for row in frozen.predictions
    }
    recording_ids = sorted({row.recording_id for row in frozen.predictions})
    metadata: dict[str, tuple[str, float]] = {}
    for row in frozen.predictions:
        prior = metadata.setdefault(
            row.recording_id,
            (row.patient_alias, row.recording_duration_seconds),
        )
        if prior != (row.patient_alias, row.recording_duration_seconds):
            raise ValueError("EventNet grid has inconsistent record metadata")

    # Stage 2 begins here.  Only exact public source-dev global seizure rows
    # survive the parser projection.
    reference_root = Path(source_dev_reference_root)
    reference_by_recording: dict[str, list[dict[str, float]]] = {}
    reference_inventory: list[dict[str, Any]] = []
    reference_event_inventory: list[list[Any]] = []
    ignored_non_term_rows = 0
    first_reference_open_after_grid_validation = True
    for recording_id in recording_ids:
        patient_alias, duration = metadata[recording_id]
        path = _safe_reference_path(reference_root, recording_id)
        payload = path.read_bytes()
        parsed = parse_tusz_term_seiz_reference_bytes(
            payload,
            duration_seconds=duration,
        )
        events = parsed.events()
        reference_by_recording[recording_id] = events
        reference_inventory.append(
            {
                "recording_id": recording_id,
                "patient_alias": patient_alias,
                "reference_relative_path": str(
                    PurePosixPath(recording_id).with_suffix(".csv_bi")
                ),
                "reference_file_sha256": parsed.reference_file_sha256,
                "selected_term_seiz_row_count": parsed.selected_term_seiz_row_count,
                "ignored_non_term_seiz_row_count": parsed.ignored_non_term_seiz_row_count,
            }
        )
        ignored_non_term_rows += parsed.ignored_non_term_seiz_row_count
        for event in events:
            reference_event_inventory.append(
                [
                    recording_id,
                    float(event["start_seconds"]),
                    float(event["stop_seconds"]),
                ]
            )

    selection = grid["selection_definition"]
    pooled_floor = float(selection["minimum_pooled_event_sensitivity"])
    macro_floor = float(selection["minimum_patient_macro_event_sensitivity"])
    candidate_results: list[dict[str, Any]] = []
    for policy_row in policies:
        policy_id = policy_row["policy_id"]
        benchmark_rows: list[dict[str, Any]] = []
        zero_alarm_records = 0
        zero_alarm_patients: set[str] = set()
        patients_with_alarm: set[str] = set()
        all_patients: set[str] = set()
        for recording_id in recording_ids:
            prediction = prediction_by_pair[(recording_id, policy_id)]
            patient_alias, duration = metadata[recording_id]
            alarms = _prediction_intervals(prediction.merged_alarms())
            all_patients.add(patient_alias)
            if alarms:
                patients_with_alarm.add(patient_alias)
            else:
                zero_alarm_records += 1
                zero_alarm_patients.add(patient_alias)
            benchmark_rows.append(
                {
                    "patient_id": patient_alias,
                    "recording_id": recording_id,
                    "split": "source_dev",
                    "duration_seconds": duration,
                    "reference_events": deepcopy(reference_by_recording[recording_id]),
                    "predicted_events": alarms,
                }
            )
        metrics = aggregate_continuous_detection_metrics(
            benchmark_rows,
            tolerances_seconds=DEFAULT_TOLERANCES_SECONDS,
        )
        pooled = metrics["event_sensitivity"]
        macro = metrics["patient_macro"]["event_sensitivity_macro"]
        feasible = (
            pooled is not None
            and float(pooled) >= pooled_floor - 1e-12
            and macro is not None
            and float(macro) >= macro_floor - 1e-12
        )
        candidate_results.append(
            {
                "candidate_id": "EVNCALCAND-" + policy_row["policy_sha256"][:20],
                "policy_id": policy_id,
                "policy_sha256": policy_row["policy_sha256"],
                "decoder_policy": deepcopy(policy_row["decoder_policy"]),
                "prediction_row_roster_sha256": _canonical_sha256(
                    [
                        prediction_by_pair[
                            (recording_id, policy_id)
                        ].prediction_row_receipt_sha256
                        for recording_id in recording_ids
                    ]
                ),
                "metrics": metrics,
                "coverage_accounting": {
                    "recording_count": len(recording_ids),
                    "patient_count": len(all_patients),
                    "zero_alarm_recording_count": zero_alarm_records,
                    "zero_alarm_patient_count": len(
                        all_patients.difference(patients_with_alarm)
                    ),
                    "seizure_free_recording_count": sum(
                        not reference_by_recording[recording_id]
                        for recording_id in recording_ids
                    ),
                    "all_source_dev_records_scored": True,
                    "zero_alarm_and_missed_records_retained": True,
                },
                "high_recall_constraints_met": feasible,
            }
        )

    def high_recall_ranking(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            _finite_or_infinity(
                _candidate_scalar(candidate, ("metrics", "alarm_false_alarms_per_24h"))
            ),
            _finite_or_infinity(
                _candidate_scalar(
                    candidate,
                    ("metrics", "time_in_warning_fraction_of_recording"),
                )
            ),
            _finite_or_infinity(
                _candidate_scalar(
                    candidate,
                    ("metrics", "onset_latency_seconds", "absolute_mean_matched_only"),
                )
            ),
            -float(
                _candidate_scalar(
                    candidate,
                    ("metrics", "patient_macro", "event_sensitivity_macro"),
                )
                or 0.0
            ),
            str(candidate["policy_sha256"]),
        )

    feasible = [
        row for row in candidate_results if row["high_recall_constraints_met"] is True
    ]
    selected_candidate = (
        None if not feasible else min(feasible, key=high_recall_ranking)
    )

    def best_effort_ranking(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        macro = _candidate_scalar(
            candidate, ("metrics", "patient_macro", "event_sensitivity_macro")
        )
        pooled = _candidate_scalar(candidate, ("metrics", "event_sensitivity"))
        return (
            -float(macro or 0.0),
            -float(pooled or 0.0),
            _finite_or_infinity(
                _candidate_scalar(candidate, ("metrics", "alarm_false_alarms_per_24h"))
            ),
            _finite_or_infinity(
                _candidate_scalar(
                    candidate,
                    ("metrics", "time_in_warning_fraction_of_recording"),
                )
            ),
            str(candidate["policy_sha256"]),
        )

    best_effort = min(candidate_results, key=best_effort_ranking)
    selected_operating_point: dict[str, Any] | None = None
    if selected_candidate is not None:
        selected_operating_point = {
            "operating_point_id": "EVNNATIVEOP-"
            + _canonical_sha256(
                {
                    "grid_bundle_id": grid_receipt["bundle_id"],
                    "reference_inventory_sha256": _canonical_sha256(
                        reference_inventory
                    ),
                    "candidate_id": selected_candidate["candidate_id"],
                    "method_id": EVENTNET_SOURCE_DEV_CALIBRATION_METHOD_ID,
                }
            )[:24],
            "candidate_id": selected_candidate["candidate_id"],
            "policy_id": selected_candidate["policy_id"],
            "policy_sha256": selected_candidate["policy_sha256"],
            "decoder_policy": deepcopy(selected_candidate["decoder_policy"]),
            "metrics": deepcopy(selected_candidate["metrics"]),
        }

    budget_operating_points: list[dict[str, Any]] = []
    for budget in selection["false_alarm_budgets_per_24h"]:
        within_budget = [
            candidate
            for candidate in candidate_results
            if (
                _candidate_scalar(candidate, ("metrics", "alarm_false_alarms_per_24h"))
                is not None
                and float(
                    _candidate_scalar(
                        candidate, ("metrics", "alarm_false_alarms_per_24h")
                    )
                )
                <= float(budget) + 1e-12
            )
        ]
        if within_budget:
            chosen = min(within_budget, key=best_effort_ranking)
            budget_operating_points.append(
                {
                    "false_alarm_budget_per_24h": float(budget),
                    "candidate_id": chosen["candidate_id"],
                    "policy_id": chosen["policy_id"],
                    "observed_false_alarms_per_24h": chosen["metrics"][
                        "alarm_false_alarms_per_24h"
                    ],
                    "event_sensitivity": chosen["metrics"]["event_sensitivity"],
                    "patient_macro_event_sensitivity": chosen["metrics"][
                        "patient_macro"
                    ]["event_sensitivity_macro"],
                    "time_in_warning_fraction": chosen["metrics"][
                        "time_in_warning_fraction_of_recording"
                    ],
                }
            )
        else:
            budget_operating_points.append(
                {
                    "false_alarm_budget_per_24h": float(budget),
                    "candidate_id": None,
                    "policy_id": None,
                    "observed_false_alarms_per_24h": None,
                    "event_sensitivity": None,
                    "patient_macro_event_sensitivity": None,
                    "time_in_warning_fraction": None,
                }
            )

    roster_binding = grid_receipt["source_dev_roster_binding"]
    official_dev_complete = roster_binding["complete_split_inventory_verified"]
    if selected_candidate is None:
        research_admission_status = "not_admitted_no_policy_met_high_recall_constraints"
    else:
        research_admission_status = (
            "selected_overlay_navigation_operating_point_research_only"
            if official_dev_complete is not True
            else "selected_navigation_operating_point_pending_separate_source_eval_admission"
        )
    limitations = [
        "released_checkpoint_exact_patient_and_record_exposure_unverified",
        "no_independent_source_eval_or_external_validation",
        "no_patient_bootstrap_confidence_intervals_in_calibration_selection",
        "eventnet_alarm_start_is_center_minus_half_duration_not_clinical_onset",
        "calibration_grid_and_released_policy_diagnostic_are_development_analyses",
    ]
    if selected_candidate is None:
        limitations.append("no_policy_met_preregistered_high_recall_constraints")
    if official_dev_complete is not True:
        limitations.append(
            "frozen_roster_is_not_complete_official_TUSZ_dev_opportunity"
        )
    if sum(not events for events in reference_by_recording.values()) < 100:
        limitations.append(
            "too_few_seizure_free_records_for_stable_false_alarm_transport"
        )

    body: dict[str, Any] = {
        "schema_version": EVENTNET_SOURCE_DEV_CALIBRATION_SCHEMA_VERSION,
        "calibration_id": "EVENTNET-SOURCE-DEV-CALIBRATION-PENDING",
        "method_id": EVENTNET_SOURCE_DEV_CALIBRATION_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "calibration_code_sha256": eventnet_native_calibration_code_sha256(),
        "decoder_code_sha256": grid_receipt["decoder_code_sha256"],
        "calibration_split": "source_dev",
        "grid_bundle_id": grid_receipt["bundle_id"],
        "grid_bundle_receipt_sha256": grid_receipt["receipt_sha256"],
        "grid_definition_sha256": grid_receipt["grid_definition_sha256"],
        "raw_bundle_validation_id": grid_receipt["raw_bundle_validation_id"],
        "raw_bundle_validation_receipt_sha256": grid_receipt[
            "raw_bundle_validation_receipt_sha256"
        ],
        "source_dev_roster_binding": deepcopy(roster_binding),
        "reference_parser_id": EVENTNET_SOURCE_DEV_REFERENCE_PARSER_ID,
        "reference_mapping_id": EVENTNET_SOURCE_DEV_REFERENCE_MAPPING_ID,
        "reference_file_inventory_sha256": _canonical_sha256(reference_inventory),
        "reference_event_inventory_sha256": _canonical_sha256(
            reference_event_inventory
        ),
        "recording_roster_sha256": _canonical_sha256(recording_ids),
        "patient_alias_roster_sha256": _canonical_sha256(
            sorted({metadata[recording_id][0] for recording_id in recording_ids})
        ),
        "recording_count": len(recording_ids),
        "patient_alias_count": len(
            {metadata[recording_id][0] for recording_id in recording_ids}
        ),
        "reference_file_count": len(reference_inventory),
        "reference_event_count": len(reference_event_inventory),
        "seizure_free_recording_count": sum(
            not reference_by_recording[recording_id] for recording_id in recording_ids
        ),
        "ignored_non_term_seiz_row_count": ignored_non_term_rows,
        "selection_definition": deepcopy(selection),
        "candidate_results": candidate_results,
        "selected_operating_point": selected_operating_point,
        "best_effort_diagnostic_candidate": {
            "candidate_id": best_effort["candidate_id"],
            "policy_id": best_effort["policy_id"],
            "policy_sha256": best_effort["policy_sha256"],
            "high_recall_constraints_met": best_effort["high_recall_constraints_met"],
            "metrics": deepcopy(best_effort["metrics"]),
        },
        "false_alarm_budget_operating_points": budget_operating_points,
        "constraint_status": (
            "met_selected_one_native_operating_point"
            if selected_operating_point is not None
            else "not_met_no_native_operating_point_frozen"
        ),
        "research_navigation_admission_status": research_admission_status,
        "complete_official_source_dev_inventory_verified": official_dev_complete,
        "future_source_eval_admission_eligible": bool(
            selected_operating_point is not None and official_dev_complete is True
        ),
        "source_eval_use_authorized": False,
        "qualification_status": "research_only_not_accuracy_primary",
        "qualification_limitations": limitations,
        "stage_order_receipt": {
            "decoder_grid_validation_completed_before_first_reference_open": first_reference_open_after_grid_validation,
            "prediction_grid_mutated_after_reference_open": False,
            "reference_files_opened": len(reference_inventory),
            "source_eval_reference_files_opened": 0,
        },
        "scope_receipt": deepcopy(_CALIBRATION_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["calibration_id"] = "EVNNATIVECAL-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_eventnet_source_dev_native_calibration_receipt(body)


def validate_eventnet_source_dev_native_calibration_receipt(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "calibration_id",
        "method_id",
        "provider_id",
        "calibration_code_sha256",
        "decoder_code_sha256",
        "calibration_split",
        "grid_bundle_id",
        "grid_bundle_receipt_sha256",
        "grid_definition_sha256",
        "raw_bundle_validation_id",
        "raw_bundle_validation_receipt_sha256",
        "source_dev_roster_binding",
        "reference_parser_id",
        "reference_mapping_id",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "recording_roster_sha256",
        "patient_alias_roster_sha256",
        "recording_count",
        "patient_alias_count",
        "reference_file_count",
        "reference_event_count",
        "seizure_free_recording_count",
        "ignored_non_term_seiz_row_count",
        "selection_definition",
        "candidate_results",
        "selected_operating_point",
        "best_effort_diagnostic_candidate",
        "false_alarm_budget_operating_points",
        "constraint_status",
        "research_navigation_admission_status",
        "complete_official_source_dev_inventory_verified",
        "future_source_eval_admission_eligible",
        "source_eval_use_authorized",
        "qualification_status",
        "qualification_limitations",
        "stage_order_receipt",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("EventNet source-dev calibration receipt fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_SOURCE_DEV_CALIBRATION_SCHEMA_VERSION
        or data["method_id"] != EVENTNET_SOURCE_DEV_CALIBRATION_METHOD_ID
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["calibration_split"] != "source_dev"
        or data["reference_parser_id"] != EVENTNET_SOURCE_DEV_REFERENCE_PARSER_ID
        or data["reference_mapping_id"] != EVENTNET_SOURCE_DEV_REFERENCE_MAPPING_ID
        or data["scope_receipt"] != _CALIBRATION_SCOPE
        or data["source_eval_use_authorized"] is not False
        or data["qualification_status"] != "research_only_not_accuracy_primary"
    ):
        raise ValueError(
            "EventNet source-dev calibration identity or permissions drifted"
        )
    for field in (
        "calibration_code_sha256",
        "decoder_code_sha256",
        "grid_bundle_receipt_sha256",
        "grid_definition_sha256",
        "raw_bundle_validation_receipt_sha256",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "recording_roster_sha256",
        "patient_alias_roster_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(data[field]):
            raise ValueError(f"EventNet source-dev calibration {field} is invalid")
    candidates = data["candidate_results"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("EventNet source-dev calibration has no candidates")
    selection = data["selection_definition"]
    if type(selection) is not dict or set(selection) != _SELECTION_FIELDS:
        raise ValueError("EventNet calibration selection definition drifted")
    candidate_ids: set[str] = set()
    feasible_ids: set[str] = set()
    pooled_floor = _finite(
        selection["minimum_pooled_event_sensitivity"],
        "calibration pooled sensitivity floor",
    )
    macro_floor = _finite(
        selection["minimum_patient_macro_event_sensitivity"],
        "calibration patient-macro sensitivity floor",
    )
    if pooled_floor > 1.0 or macro_floor > 1.0:
        raise ValueError("EventNet calibration sensitivity floors must lie in [0,1]")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        required_candidate = {
            "candidate_id",
            "policy_id",
            "policy_sha256",
            "decoder_policy",
            "prediction_row_roster_sha256",
            "metrics",
            "coverage_accounting",
            "high_recall_constraints_met",
        }
        if type(candidate) is not dict or set(candidate) != required_candidate:
            raise ValueError("EventNet calibration candidate fields drifted")
        candidate_id = _identifier(candidate["candidate_id"], "candidate ID")
        if candidate_id in candidate_ids:
            raise ValueError("EventNet calibration candidate IDs are not unique")
        candidate_ids.add(candidate_id)
        policy = validate_eventnet_native_decoder_policy(candidate["decoder_policy"])
        policy_sha256 = _canonical_sha256(policy)
        if (
            candidate["policy_sha256"] != policy_sha256
            or candidate["policy_id"] != "EVNPOL-" + policy_sha256[:20]
            or candidate_id != "EVNCALCAND-" + policy_sha256[:20]
        ):
            raise ValueError("EventNet calibration candidate policy binding drifted")
        metrics = candidate["metrics"]
        pooled = metrics.get("event_sensitivity")
        patient_macro = metrics.get("patient_macro")
        macro = (
            patient_macro.get("event_sensitivity_macro")
            if isinstance(patient_macro, Mapping)
            else None
        )
        replayed_feasible = (
            pooled is not None
            and float(pooled) >= pooled_floor - 1e-12
            and macro is not None
            and float(macro) >= macro_floor - 1e-12
        )
        if candidate["high_recall_constraints_met"] is not replayed_feasible:
            raise ValueError("EventNet candidate high-recall flag does not replay")
        if replayed_feasible:
            feasible_ids.add(candidate_id)
        if not _is_sha256(candidate["policy_sha256"]) or not _is_sha256(
            candidate["prediction_row_roster_sha256"]
        ):
            raise ValueError("EventNet calibration candidate hash drifted")
        candidate_by_id[candidate_id] = candidate
    selected = data["selected_operating_point"]
    if selected is None:
        if (
            feasible_ids
            or data["constraint_status"] != "not_met_no_native_operating_point_frozen"
            or data["future_source_eval_admission_eligible"] is not False
        ):
            raise ValueError("EventNet failed calibration selection drifted")
    else:
        if (
            type(selected) is not dict
            or selected.get("candidate_id") not in feasible_ids
            or data["constraint_status"] != "met_selected_one_native_operating_point"
        ):
            raise ValueError("EventNet selected native operating point is invalid")
        selected_candidate = candidate_by_id[selected["candidate_id"]]
        if (
            selected.get("policy_id") != selected_candidate["policy_id"]
            or selected.get("policy_sha256") != selected_candidate["policy_sha256"]
            or selected.get("decoder_policy") != selected_candidate["decoder_policy"]
            or selected.get("metrics") != selected_candidate["metrics"]
        ):
            raise ValueError(
                "EventNet selected operating point does not bind its candidate"
            )
        expected_future_eligibility = (
            data["complete_official_source_dev_inventory_verified"] is True
        )
        if (
            data["future_source_eval_admission_eligible"]
            is not expected_future_eligibility
        ):
            raise ValueError("EventNet future source-eval eligibility drifted")
    stage = data["stage_order_receipt"]
    if stage != {
        "decoder_grid_validation_completed_before_first_reference_open": True,
        "prediction_grid_mutated_after_reference_open": False,
        "reference_files_opened": data["reference_file_count"],
        "source_eval_reference_files_opened": 0,
    }:
        raise ValueError("EventNet calibration stage ordering drifted")
    digest = deepcopy(data)
    digest["calibration_id"] = "EVENTNET-SOURCE-DEV-CALIBRATION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["calibration_id"] != "EVNNATIVECAL-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet source-dev calibration ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet source-dev calibration receipt hash drifted")
    return data


def write_eventnet_source_dev_native_calibration_append_only(
    payload: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    receipt = validate_eventnet_source_dev_native_calibration_receipt(dict(payload))
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("EventNet calibration output must be a new path")
    output.mkdir(parents=True, exist_ok=False)
    path = output / EVENTNET_SOURCE_DEV_CALIBRATION_FILENAME
    content = (
        json.dumps(
            receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)
    return {
        "calibration_id": receipt["calibration_id"],
        "calibration_receipt_sha256": receipt["receipt_sha256"],
        "calibration_file_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "output_path": str(path.resolve(strict=True)),
        "append_only_new_directory": True,
    }


__all__ = [
    "EVENTNET_SOURCE_DEV_CALIBRATION_METHOD_ID",
    "EVENTNET_SOURCE_DEV_CALIBRATION_SCHEMA_VERSION",
    "calibrate_eventnet_native_decoder_grid_source_dev",
    "eventnet_native_calibration_code_sha256",
    "validate_eventnet_source_dev_native_calibration_receipt",
    "write_eventnet_source_dev_native_calibration_append_only",
]
