"""Source-development calibration for continuous seizure-event decoding.

The detector provider must materialize a dense, full-record EEG-only posterior
before this module is called.  Reference seizure intervals are introduced only
here, on the frozen ``source_dev`` split, to choose one model-neutral
hysteresis operating point.  The selected policy can then be applied unchanged
to ``source_eval``; this module never evaluates that split.

Selection is deliberately constrained rather than based on a blended score:

1. pooled and patient-macro event sensitivity must meet preregistered floors;
2. among feasible candidates, minimize false alarms per recording hour;
3. maximize reference-denominator onset hit rate at the chosen tolerance;
4. minimize matched-event absolute onset error;
5. use patient-macro false alarms/hour and a content hash as deterministic
   tie-breakers.

Every source-development recording remains in every candidate evaluation,
including seizure-free, zero-alarm and missed-seizure recordings.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .continuous_detection import decode_continuous_seizure_posterior
from .continuous_detection_benchmark import (
    DEFAULT_TOLERANCES_SECONDS,
    aggregate_continuous_detection_metrics,
    validate_continuous_benchmark_rows,
)


CONTINUOUS_CALIBRATION_SCHEMA_VERSION = (
    "continuous_detector_operating_point_calibration_v1"
)
CONTINUOUS_CALIBRATION_METHOD_ID = (
    "patient_level_constrained_hysteresis_grid_selection_v1"
)
DEFAULT_MINIMUM_EVENT_SENSITIVITY = 0.90
DEFAULT_ONSET_TIE_TOLERANCE_SECONDS = 5.0

_CALIBRATION_ROW_FIELDS = {
    "patient_id",
    "recording_id",
    "split",
    "duration_seconds",
    "source_signal_sha256",
    "posterior_artifact_id",
    "provider_receipt",
    "posterior_timeline",
    "reference_events",
}


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
    if len(value) > 512 or any(ord(character) < 32 for character in value):
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


def _unit_interval(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{context} must be within [0, 1]")
    return result


def _positive(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be positive")
    return result


def _normalized_roster(values: Iterable[str], context: str) -> list[str]:
    result = sorted({_identifier(value, context) for value in values})
    if not result:
        raise ValueError(f"{context} must not be empty")
    return result


def validate_continuous_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider_id: str,
) -> list[dict[str, Any]]:
    """Validate complete source-development posterior/reference rows."""

    expected_provider = _identifier(provider_id, "provider_id")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("calibration rows must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    posterior_ids: set[str] = set()
    projected: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if type(raw) is not dict or set(raw) != _CALIBRATION_ROW_FIELDS:
            raise ValueError(
                f"calibration row {index} has missing or unknown fields"
            )
        row = deepcopy(raw)
        patient_id = _identifier(row["patient_id"], "patient_id")
        recording_id = _identifier(row["recording_id"], "recording_id")
        if row["split"] != "source_dev":
            raise ValueError("operating-point calibration accepts source_dev only")
        posterior_id = _identifier(
            row["posterior_artifact_id"], "posterior_artifact_id"
        )
        if posterior_id in posterior_ids:
            raise ValueError("posterior artifact IDs must be unique")
        posterior_ids.add(posterior_id)
        source_sha256 = _sha256(row["source_signal_sha256"], "source signal")
        provider = row["provider_receipt"]
        if type(provider) is not dict or provider.get("provider_id") != expected_provider:
            raise ValueError("calibration row provider does not match provider_id")
        timeline = row["posterior_timeline"]
        if not isinstance(timeline, list) or not timeline:
            raise TypeError("posterior_timeline must be a non-empty array")
        projected.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "split": "source_dev",
                "duration_seconds": row["duration_seconds"],
                "reference_events": row["reference_events"],
                "predicted_events": [],
            }
        )
        normalized.append(
            {
                "patient_id": patient_id,
                "recording_id": recording_id,
                "split": "source_dev",
                "duration_seconds": row["duration_seconds"],
                "source_signal_sha256": source_sha256,
                "posterior_artifact_id": posterior_id,
                "provider_receipt": deepcopy(provider),
                "posterior_timeline": deepcopy(timeline),
                "reference_events": deepcopy(row["reference_events"]),
            }
        )

    # Reuse the frozen benchmark's interval, duration, identity and patient
    # split checks.  Predictions stay empty because labels have not yet been
    # used to decode any candidate policy.
    validated_projected = validate_continuous_benchmark_rows(projected)
    by_recording = {row["recording_id"]: row for row in validated_projected}
    for row in normalized:
        validated = by_recording[row["recording_id"]]
        row["duration_seconds"] = validated["duration_seconds"]
        row["reference_events"] = validated["reference_events"]
    normalized.sort(key=lambda row: (row["patient_id"], row["recording_id"]))
    return normalized


def _decode_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    benchmark_rows: list[dict[str, Any]] = []
    normalized_policy: dict[str, Any] | None = None
    zero_alarm_recordings = 0
    patients_with_alarm: set[str] = set()
    all_patients: set[str] = set()
    for row in rows:
        decoded = decode_continuous_seizure_posterior(
            recording_id=str(row["recording_id"]),
            source_signal_sha256=str(row["source_signal_sha256"]),
            recording_duration_seconds=float(row["duration_seconds"]),
            provider_receipt=row["provider_receipt"],
            posterior_timeline=row["posterior_timeline"],
            policy=policy,
        )
        if normalized_policy is None:
            normalized_policy = deepcopy(decoded["decoder_policy"])
        elif decoded["decoder_policy"] != normalized_policy:
            raise RuntimeError("one policy normalized differently across recordings")
        predicted = [
            {
                "start_seconds": float(event["start_offset_seconds"]),
                "stop_seconds": float(event["stop_offset_seconds"]),
            }
            for event in decoded["event_proposals"]
        ]
        patient_id = str(row["patient_id"])
        all_patients.add(patient_id)
        if predicted:
            patients_with_alarm.add(patient_id)
        else:
            zero_alarm_recordings += 1
        benchmark_rows.append(
            {
                "patient_id": patient_id,
                "recording_id": str(row["recording_id"]),
                "split": "source_dev",
                "duration_seconds": float(row["duration_seconds"]),
                "reference_events": deepcopy(row["reference_events"]),
                "predicted_events": predicted,
            }
        )
    if normalized_policy is None:
        raise RuntimeError("candidate policy received no source-development rows")
    return (
        normalized_policy,
        benchmark_rows,
        zero_alarm_recordings,
        len(all_patients.difference(patients_with_alarm)),
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _patient_macro_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    onset_tolerance_seconds: float,
) -> dict[str, Any]:
    patient_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        patient_rows.setdefault(str(row["patient_id"]), []).append(row)
    sensitivity_values: list[float] = []
    false_alarm_rates: list[float] = []
    onset_hit_values: list[float] = []
    for patient_id in sorted(patient_rows):
        metrics = aggregate_continuous_detection_metrics(patient_rows[patient_id])
        if metrics["event_sensitivity"] is not None:
            sensitivity_values.append(float(metrics["event_sensitivity"]))
        false_alarm_rates.append(
            float(metrics["alarm_false_alarms_per_recording_hour"])
        )
        key = f"{onset_tolerance_seconds:g}s"
        hit_rate = metrics["onset_absolute_hit_rate"][key]["rate"]
        if hit_rate is not None:
            onset_hit_values.append(float(hit_rate))
    return {
        "patient_count": len(patient_rows),
        "patients_with_reference_events": len(sensitivity_values),
        "event_sensitivity_macro": _mean(sensitivity_values),
        "alarm_false_alarms_per_recording_hour_macro": _mean(false_alarm_rates),
        "onset_absolute_hit_rate_macro": _mean(onset_hit_values),
        "zero_alarm_patients_are_included_in_false_alarm_macro": True,
    }


def _finite_or_infinity(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return math.inf


def select_continuous_detection_operating_point(
    *,
    provider_id: str,
    rows: Sequence[Mapping[str, Any]],
    candidate_policies: Sequence[Mapping[str, Any]],
    minimum_event_sensitivity: float = DEFAULT_MINIMUM_EVENT_SENSITIVITY,
    minimum_patient_macro_event_sensitivity: float | None = None,
    onset_tie_tolerance_seconds: float = DEFAULT_ONSET_TIE_TOLERANCE_SECONDS,
    expected_source_dev_recording_ids: Iterable[str] | None = None,
    evaluation_patient_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Select and content-bind one source-development decoder policy."""

    provider = _identifier(provider_id, "provider_id")
    minimum_pooled = _unit_interval(
        minimum_event_sensitivity, "minimum_event_sensitivity"
    )
    minimum_macro = _unit_interval(
        minimum_pooled
        if minimum_patient_macro_event_sensitivity is None
        else minimum_patient_macro_event_sensitivity,
        "minimum_patient_macro_event_sensitivity",
    )
    onset_tolerance = _positive(
        onset_tie_tolerance_seconds, "onset_tie_tolerance_seconds"
    )
    tolerances = tuple(float(value) for value in DEFAULT_TOLERANCES_SECONDS)
    if onset_tolerance not in tolerances:
        raise ValueError(
            "onset tie tolerance must be one of the benchmark tolerances"
        )
    validated_rows = validate_continuous_calibration_rows(rows, provider_id=provider)
    if sum(len(row["reference_events"]) for row in validated_rows) == 0:
        raise ValueError("source_dev has no reference seizure events for calibration")
    if (
        not isinstance(candidate_policies, Sequence)
        or isinstance(candidate_policies, (str, bytes))
        or not candidate_policies
    ):
        raise TypeError("candidate_policies must be a non-empty sequence")

    observed_recordings = sorted(str(row["recording_id"]) for row in validated_rows)
    inventory_status = "not_verified_no_expected_source_dev_roster"
    expected_roster_sha256: str | None = None
    if expected_source_dev_recording_ids is not None:
        expected_recordings = _normalized_roster(
            expected_source_dev_recording_ids, "source_dev recording ID"
        )
        if expected_recordings != observed_recordings:
            missing = sorted(set(expected_recordings).difference(observed_recordings))
            extra = sorted(set(observed_recordings).difference(expected_recordings))
            raise ValueError(
                "source_dev recording inventory mismatch: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        inventory_status = "verified_complete_expected_source_dev_inventory"
        expected_roster_sha256 = _canonical_sha256(expected_recordings)

    development_patients = sorted(
        {str(row["patient_id"]) for row in validated_rows}
    )
    evaluation_roster_sha256: str | None = None
    patient_isolation_status = "not_verified_no_evaluation_roster"
    if evaluation_patient_ids is not None:
        evaluation_patients = _normalized_roster(
            evaluation_patient_ids, "evaluation patient ID"
        )
        overlap = sorted(set(development_patients).intersection(evaluation_patients))
        if overlap:
            raise ValueError("source_dev/source_eval patient rosters overlap")
        evaluation_roster_sha256 = _canonical_sha256(evaluation_patients)
        patient_isolation_status = "verified_no_patient_overlap"

    candidate_results: list[dict[str, Any]] = []
    policy_hashes: set[str] = set()
    for raw_policy in candidate_policies:
        if type(raw_policy) is not dict:
            raise TypeError("each candidate policy must be an object")
        normalized_policy, benchmark_rows, zero_records, zero_patients = (
            _decode_candidate(validated_rows, policy=raw_policy)
        )
        policy_sha256 = _canonical_sha256(normalized_policy)
        if policy_sha256 in policy_hashes:
            raise ValueError("candidate policy grid contains duplicate policies")
        policy_hashes.add(policy_sha256)
        metrics = aggregate_continuous_detection_metrics(
            benchmark_rows, tolerances_seconds=tolerances
        )
        patient_macro = _patient_macro_metrics(
            benchmark_rows, onset_tolerance_seconds=onset_tolerance
        )
        pooled_sensitivity = metrics["event_sensitivity"]
        macro_sensitivity = patient_macro["event_sensitivity_macro"]
        feasible = (
            pooled_sensitivity is not None
            and float(pooled_sensitivity) >= minimum_pooled - 1e-12
            and macro_sensitivity is not None
            and float(macro_sensitivity) >= minimum_macro - 1e-12
        )
        candidate_results.append(
            {
                "candidate_id": "CONTCAND-" + policy_sha256[:20],
                "decoder_policy": normalized_policy,
                "decoder_policy_sha256": policy_sha256,
                "pooled_metrics": metrics,
                "patient_macro_metrics": patient_macro,
                "coverage_accounting": {
                    "recording_count": len(benchmark_rows),
                    "zero_alarm_recording_count": zero_records,
                    "zero_alarm_patient_count": zero_patients,
                    "zero_alarm_and_missed_records_retained": True,
                    "all_source_dev_rows_scored_for_this_candidate": True,
                },
                "high_recall_constraints_met": feasible,
            }
        )

    onset_key = f"{onset_tolerance:g}s"

    def ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = candidate["pooled_metrics"]
        macro = candidate["patient_macro_metrics"]
        onset_hit = metrics["onset_absolute_hit_rate"][onset_key]["rate"]
        return (
            _finite_or_infinity(
                metrics["alarm_false_alarms_per_recording_hour"]
            ),
            -float(onset_hit) if onset_hit is not None else math.inf,
            _finite_or_infinity(
                metrics["onset_latency_seconds"][
                    "absolute_mean_matched_only"
                ]
            ),
            _finite_or_infinity(
                macro["alarm_false_alarms_per_recording_hour_macro"]
            ),
            -float(metrics["event_sensitivity"]),
            str(candidate["decoder_policy_sha256"]),
        )

    feasible_candidates = [
        candidate
        for candidate in candidate_results
        if candidate["high_recall_constraints_met"] is True
    ]
    selected_candidate = (
        None if not feasible_candidates else min(feasible_candidates, key=ranking_key)
    )
    selected_operating_point: dict[str, Any] | None = None
    if selected_candidate is not None:
        selected_operating_point = {
            "operating_point_id": "CONTOP-"
            + _canonical_sha256(
                {
                    "provider_id": provider,
                    "source_dev_rows_sha256": _canonical_sha256(validated_rows),
                    "candidate_id": selected_candidate["candidate_id"],
                    "selection_method_id": CONTINUOUS_CALIBRATION_METHOD_ID,
                }
            )[:24],
            "candidate_id": selected_candidate["candidate_id"],
            "decoder_policy": deepcopy(selected_candidate["decoder_policy"]),
            "pooled_metrics": deepcopy(selected_candidate["pooled_metrics"]),
            "patient_macro_metrics": deepcopy(
                selected_candidate["patient_macro_metrics"]
            ),
        }

    limitations: list[str] = []
    if selected_candidate is None:
        limitations.append("no_candidate_met_preregistered_high_recall_constraints")
    if inventory_status != "verified_complete_expected_source_dev_inventory":
        limitations.append("complete_source_dev_recording_inventory_not_verified")
    if patient_isolation_status != "verified_no_patient_overlap":
        limitations.append("source_dev_source_eval_patient_isolation_not_verified")
    if not any(not row["reference_events"] for row in validated_rows):
        limitations.append("no_seizure_free_source_dev_records_for_false_alarm_transport")

    authorized = (
        selected_operating_point is not None
        and inventory_status == "verified_complete_expected_source_dev_inventory"
        and patient_isolation_status == "verified_no_patient_overlap"
    )
    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_CALIBRATION_SCHEMA_VERSION,
        "calibration_receipt_id": "CONTINUOUS-CALIBRATION-PENDING",
        "method_id": CONTINUOUS_CALIBRATION_METHOD_ID,
        "provider_id": provider,
        "calibration_split": "source_dev",
        "input_rows_sha256": _canonical_sha256(validated_rows),
        "posterior_artifact_roster_sha256": _canonical_sha256(
            sorted(str(row["posterior_artifact_id"]) for row in validated_rows)
        ),
        "development_patient_roster_sha256": _canonical_sha256(
            development_patients
        ),
        "evaluation_patient_roster_sha256": evaluation_roster_sha256,
        "patient_isolation_status": patient_isolation_status,
        "expected_source_dev_recording_roster_sha256": expected_roster_sha256,
        "source_dev_inventory_status": inventory_status,
        "selection_definition": {
            "minimum_pooled_event_sensitivity": minimum_pooled,
            "minimum_patient_macro_event_sensitivity": minimum_macro,
            "onset_tie_tolerance_seconds": onset_tolerance,
            "ordered_objective": [
                "satisfy_pooled_and_patient_macro_event_sensitivity_floors",
                "minimize_pooled_alarm_false_alarms_per_recording_hour",
                f"maximize_reference_denominator_onset_hit_rate_at_{onset_key}",
                "minimize_matched_event_absolute_onset_error_mean_seconds",
                "minimize_patient_macro_alarm_false_alarms_per_recording_hour",
                "maximize_pooled_event_sensitivity",
                "canonical_policy_hash_tie_break",
            ],
            "blended_accuracy_efficiency_score_used": False,
            "candidate_policy_count": len(candidate_results),
        },
        "candidate_results": candidate_results,
        "constraint_status": (
            "met_selected_one_operating_point"
            if selected_operating_point is not None
            else "not_met_no_operating_point_frozen"
        ),
        "selected_operating_point": selected_operating_point,
        "source_eval_use_authorized": authorized,
        "qualification_limitations": limitations,
        "scope_receipt": {
            "dense_posteriors_materialized_before_reference_scoring": True,
            "reference_labels_available_to_provider": False,
            "reference_labels_used_on_source_dev_for_calibration_only": True,
            "source_eval_labels_used": False,
            "edf_annotations_used": False,
            "excel_or_clinical_labels_used": False,
            "zero_alarm_records_retained": True,
            "posterior_used_for_candidate_timing_only": True,
            "detector_native_preprocessing_not_findings_fact_source": True,
        },
        "production_promotion_status": "research_calibration_only",
        "sota_claim_authorized": False,
    }
    body["calibration_receipt_id"] = "CONTCAL-" + _canonical_sha256(body)[:24]
    return validate_continuous_detection_calibration_receipt(body)


def validate_continuous_detection_calibration_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate content binding and the source-dev/source-eval firewall."""

    if type(payload) is not dict:
        raise TypeError("continuous calibration receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "calibration_receipt_id",
        "method_id",
        "provider_id",
        "calibration_split",
        "input_rows_sha256",
        "posterior_artifact_roster_sha256",
        "development_patient_roster_sha256",
        "evaluation_patient_roster_sha256",
        "patient_isolation_status",
        "expected_source_dev_recording_roster_sha256",
        "source_dev_inventory_status",
        "selection_definition",
        "candidate_results",
        "constraint_status",
        "selected_operating_point",
        "source_eval_use_authorized",
        "qualification_limitations",
        "scope_receipt",
        "production_promotion_status",
        "sota_claim_authorized",
    }
    if set(data) != required:
        raise ValueError("continuous calibration receipt has invalid fields")
    if (
        data["schema_version"] != CONTINUOUS_CALIBRATION_SCHEMA_VERSION
        or data["method_id"] != CONTINUOUS_CALIBRATION_METHOD_ID
        or data["calibration_split"] != "source_dev"
    ):
        raise ValueError("continuous calibration schema/method/split drifted")
    expected_scope = {
        "dense_posteriors_materialized_before_reference_scoring": True,
        "reference_labels_available_to_provider": False,
        "reference_labels_used_on_source_dev_for_calibration_only": True,
        "source_eval_labels_used": False,
        "edf_annotations_used": False,
        "excel_or_clinical_labels_used": False,
        "zero_alarm_records_retained": True,
        "posterior_used_for_candidate_timing_only": True,
        "detector_native_preprocessing_not_findings_fact_source": True,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("continuous calibration violated its label firewall")
    if data["production_promotion_status"] != "research_calibration_only" or data[
        "sota_claim_authorized"
    ] is not False:
        raise ValueError("calibration receipt cannot authorize promotion/SOTA claims")
    candidates = data["candidate_results"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("continuous calibration has no candidate results")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("continuous calibration candidate IDs are not unique")
    selected = data["selected_operating_point"]
    if selected is None:
        if data["constraint_status"] != "not_met_no_operating_point_frozen" or data[
            "source_eval_use_authorized"
        ] is not False:
            raise ValueError("failed calibration cannot authorize source_eval use")
    else:
        if (
            type(selected) is not dict
            or selected.get("candidate_id") not in candidate_ids
            or data["constraint_status"] != "met_selected_one_operating_point"
        ):
            raise ValueError("selected continuous operating point is invalid")
        if data["source_eval_use_authorized"] is True and (
            data["patient_isolation_status"] != "verified_no_patient_overlap"
            or data["source_dev_inventory_status"]
            != "verified_complete_expected_source_dev_inventory"
        ):
            raise ValueError("source_eval authorization lacks isolation/inventory proof")
    digest = deepcopy(data)
    digest["calibration_receipt_id"] = "CONTINUOUS-CALIBRATION-PENDING"
    expected_id = "CONTCAL-" + _canonical_sha256(digest)[:24]
    if data["calibration_receipt_id"] != expected_id:
        raise ValueError("continuous calibration receipt is not content-bound")
    return data


__all__ = [
    "CONTINUOUS_CALIBRATION_METHOD_ID",
    "CONTINUOUS_CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_MINIMUM_EVENT_SENSITIVITY",
    "DEFAULT_ONSET_TIE_TOLERANCE_SECONDS",
    "select_continuous_detection_operating_point",
    "validate_continuous_calibration_rows",
    "validate_continuous_detection_calibration_receipt",
]
