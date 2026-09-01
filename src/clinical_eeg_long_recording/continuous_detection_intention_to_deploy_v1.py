"""Attempt-aware benchmark wrapper for continuous long-EEG detection.

The legacy patient-level benchmark deliberately consumes only prediction and
reference intervals.  That is sufficient for interval metrics but cannot tell
an observed zero-alarm recording from an unmodelled tail or a technical
failure.  This append-only wrapper binds every benchmark row to the compact
projection of a validated full-record provider result and keeps failures in
the intention-to-deploy denominator.

The wrapper does not authorize a provider, operating point, SOTA claim, or
clinical use.  It also does not allow a detector tensor to become Findings
evidence; it only closes the long-record detection accounting contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .continuous_detection_benchmark import (
    _ordered_event_matching,
    _prediction_inventory,
    _reference_inventory,
    evaluate_patient_level_continuous_detection,
    validate_continuous_benchmark_rows,
    validate_continuous_detection_benchmark_receipt,
)
from .detector_provider_contract import (
    FULL_RECORD_FAILURE_STAGES,
    FULL_RECORD_PROVIDER_OUTCOMES,
    validate_full_record_provider_result,
)


PROVIDER_ATTEMPT_PROJECTION_SCHEMA_VERSION = (
    "continuous_detector_provider_attempt_projection_v1"
)
INTENTION_TO_DEPLOY_BENCHMARK_SCHEMA_VERSION = (
    "continuous_detection_intention_to_deploy_benchmark_v1"
)
INTENTION_TO_DEPLOY_BENCHMARK_METHOD_ID = (
    "provider_attempt_aware_patient_level_continuous_detection_v1"
)

_PROJECTION_FIELDS = {
    "schema_version",
    "attempt_id",
    "full_record_result_id",
    "full_record_result_sha256",
    "provider_id",
    "provider_execution_receipt_id",
    "recording_id",
    "source_signal_sha256",
    "recording_duration_seconds",
    "outcome_status",
    "decoder_outcome",
    "coverage_receipt",
    "technical_failure",
}

_COVERAGE_FIELDS = {
    "complete_recording_scan_attempted",
    "posterior_target_coverage_complete",
    "modeled_target_coverage_seconds",
    "unmodeled_target_coverage_seconds",
    "maximum_target_coverage_gap_seconds",
    "declared_partial_tail_seconds",
    "maximum_right_padding_seconds",
    "partial_tail_policy",
}

_DECODER_FIELDS = {
    "decoding_receipt_id",
    "decoder_policy_sha256",
    "event_proposal_count",
    "zero_candidates_is_valid",
}

_FAILURE_FIELDS = {
    "failure_code",
    "failure_stage",
    "retryable",
    "failure_detail_sha256",
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
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _validate_decoder(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _DECODER_FIELDS:
        raise ValueError("provider attempt decoder outcome has invalid fields")
    result = deepcopy(value)
    result["decoding_receipt_id"] = _identifier(
        result["decoding_receipt_id"], "decoding receipt ID"
    )
    result["decoder_policy_sha256"] = _sha256(
        result["decoder_policy_sha256"], "decoder policy SHA-256"
    )
    count = result["event_proposal_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError("event proposal count must be a non-negative integer")
    if result["zero_candidates_is_valid"] is not True:
        raise ValueError("zero-candidate decoder outcome is not valid")
    return result


def _validate_failure(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _FAILURE_FIELDS:
        raise ValueError("provider attempt technical failure has invalid fields")
    result = deepcopy(value)
    result["failure_code"] = _identifier(result["failure_code"], "failure code")
    if result["failure_stage"] not in FULL_RECORD_FAILURE_STAGES:
        raise ValueError("provider attempt failure stage is invalid")
    if type(result["retryable"]) is not bool:
        raise TypeError("provider attempt retryable flag must be boolean")
    result["failure_detail_sha256"] = _sha256(
        result["failure_detail_sha256"], "failure detail SHA-256"
    )
    return result


def _validate_coverage(value: object, *, duration: float) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _COVERAGE_FIELDS:
        raise ValueError("provider attempt coverage receipt has invalid fields")
    result = deepcopy(value)
    for name in (
        "complete_recording_scan_attempted",
        "posterior_target_coverage_complete",
    ):
        if type(result[name]) is not bool:
            raise TypeError(f"provider attempt {name} must be boolean")
    for name in (
        "modeled_target_coverage_seconds",
        "unmodeled_target_coverage_seconds",
        "maximum_target_coverage_gap_seconds",
        "declared_partial_tail_seconds",
        "maximum_right_padding_seconds",
    ):
        result[name] = _finite(result[name], name, minimum=0.0)
    if (
        abs(
            result["modeled_target_coverage_seconds"]
            + result["unmodeled_target_coverage_seconds"]
            - duration
        )
        > 1e-8
    ):
        raise ValueError("provider attempt modeled/unmodeled coverage is not closed")
    if result["maximum_target_coverage_gap_seconds"] > duration + 1e-8:
        raise ValueError("provider attempt maximum coverage gap exceeds duration")
    result["partial_tail_policy"] = _identifier(
        result["partial_tail_policy"], "partial tail policy"
    )
    return result


def materialize_provider_attempt_projection(
    full_record_provider_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a validated full provider result without retaining dense scores."""

    full = validate_full_record_provider_result(dict(full_record_provider_result))
    body: dict[str, Any] = {
        "schema_version": PROVIDER_ATTEMPT_PROJECTION_SCHEMA_VERSION,
        "attempt_id": "DETATTEMPT-PENDING",
        "full_record_result_id": full["result_id"],
        "full_record_result_sha256": _canonical_sha256(full),
        "provider_id": full["provider_id"],
        "provider_execution_receipt_id": full["provider_execution_receipt_id"],
        "recording_id": full["recording_id"],
        "source_signal_sha256": full["source_signal_sha256"],
        "recording_duration_seconds": full["recording_duration_seconds"],
        "outcome_status": full["outcome_status"],
        "decoder_outcome": deepcopy(full["decoder_outcome"]),
        "coverage_receipt": deepcopy(full["coverage_receipt"]),
        "technical_failure": deepcopy(full["technical_failure"]),
    }
    body["attempt_id"] = "DETATTEMPT-" + _canonical_sha256(body)[:24]
    return validate_provider_attempt_projection(body)


def validate_provider_attempt_projection(value: object) -> dict[str, Any]:
    """Validate compact terminal-outcome semantics and its self binding."""

    if type(value) is not dict or set(value) != _PROJECTION_FIELDS:
        raise ValueError("provider attempt projection has missing or unknown fields")
    data = deepcopy(value)
    if data["schema_version"] != PROVIDER_ATTEMPT_PROJECTION_SCHEMA_VERSION:
        raise ValueError("provider attempt projection schema drifted")
    for name in (
        "attempt_id",
        "full_record_result_id",
        "provider_id",
        "provider_execution_receipt_id",
        "recording_id",
    ):
        data[name] = _identifier(data[name], name)
    data["full_record_result_sha256"] = _sha256(
        data["full_record_result_sha256"], "full-record result SHA-256"
    )
    data["source_signal_sha256"] = _sha256(
        data["source_signal_sha256"], "source signal SHA-256"
    )
    duration = _finite(
        data["recording_duration_seconds"],
        "recording duration",
        minimum=0.0,
    )
    if duration <= 0:
        raise ValueError("provider attempt recording duration must be positive")
    data["recording_duration_seconds"] = duration
    if data["outcome_status"] not in FULL_RECORD_PROVIDER_OUTCOMES:
        raise ValueError("provider attempt outcome status is invalid")
    coverage = _validate_coverage(data["coverage_receipt"], duration=duration)
    outcome = data["outcome_status"]

    if outcome == "technical_failure":
        if data["decoder_outcome"] is not None:
            raise ValueError("technical failure cannot carry a decoder outcome")
        failure = _validate_failure(data["technical_failure"])
        if (
            coverage["posterior_target_coverage_complete"] is not False
            or coverage["modeled_target_coverage_seconds"] != 0.0
            or coverage["unmodeled_target_coverage_seconds"] != duration
            or coverage["partial_tail_policy"] != "not_applicable_technical_failure"
        ):
            raise ValueError("technical failure coverage semantics drifted")
        data["technical_failure"] = failure
    else:
        if data["technical_failure"] is not None:
            raise ValueError("non-failed provider attempt carries a failure")
        decoder = _validate_decoder(data["decoder_outcome"])
        count = decoder["event_proposal_count"]
        if coverage["complete_recording_scan_attempted"] is not True:
            raise ValueError("non-failed provider attempt did not scan the recording")
        if outcome == "partial_coverage":
            if (
                coverage["posterior_target_coverage_complete"] is not False
                or coverage["unmodeled_target_coverage_seconds"] <= 0
            ):
                raise ValueError("partial provider attempt claims complete coverage")
        else:
            if (
                coverage["posterior_target_coverage_complete"] is not True
                or coverage["unmodeled_target_coverage_seconds"] != 0.0
            ):
                raise ValueError("completed provider attempt lacks complete coverage")
            expected = "completed_zero_alarm" if count == 0 else "completed_with_alarms"
            if outcome != expected:
                raise ValueError("provider attempt outcome disagrees with alarm count")
        data["decoder_outcome"] = decoder
    data["coverage_receipt"] = coverage

    digest = deepcopy(data)
    digest["attempt_id"] = "DETATTEMPT-PENDING"
    if data["attempt_id"] != "DETATTEMPT-" + _canonical_sha256(digest)[:24]:
        raise ValueError("provider attempt ID does not bind its content")
    return data


def validate_intention_to_deploy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate one attempt-bound benchmark row for every recording."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("intention-to-deploy rows must be a non-empty sequence")
    base_rows: list[Mapping[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"benchmark_row", "provider_attempt"}:
            raise ValueError(
                f"intention-to-deploy row {index} has missing or unknown fields"
            )
        base_rows.append(row["benchmark_row"])
        attempts.append(validate_provider_attempt_projection(row["provider_attempt"]))
    validated_base = validate_continuous_benchmark_rows(base_rows)
    provider_ids = {attempt["provider_id"] for attempt in attempts}
    if len(provider_ids) != 1:
        raise ValueError("one intention-to-deploy batch must use exactly one provider")
    attempt_ids: set[str] = set()
    result_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, (base, attempt) in enumerate(zip(validated_base, attempts)):
        if (
            attempt["attempt_id"] in attempt_ids
            or attempt["full_record_result_id"] in result_ids
        ):
            raise ValueError("provider attempt/result was reused across benchmark rows")
        attempt_ids.add(attempt["attempt_id"])
        result_ids.add(attempt["full_record_result_id"])
        if attempt["recording_id"] != base["recording_id"]:
            raise ValueError(
                f"intention-to-deploy row {index} recording binding drifted"
            )
        if abs(attempt["recording_duration_seconds"] - base["duration_seconds"]) > 1e-8:
            raise ValueError(
                f"intention-to-deploy row {index} duration binding drifted"
            )
        predictions = base["predicted_events"]
        outcome = attempt["outcome_status"]
        if outcome == "technical_failure":
            if predictions:
                raise ValueError("technical failure cannot carry predicted events")
        else:
            count = attempt["decoder_outcome"]["event_proposal_count"]
            if len(predictions) != count:
                raise ValueError(
                    "predicted event count disagrees with provider decoder"
                )
            modeled_stop = attempt["coverage_receipt"][
                "modeled_target_coverage_seconds"
            ]
            if any(
                event["stop_seconds"] > modeled_stop + 1e-8 for event in predictions
            ):
                raise ValueError("prediction extends into unmodelled provider support")
        normalized.append({"benchmark_row": base, "provider_attempt": attempt})
    return normalized


def _attempt_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {status: 0 for status in FULL_RECORD_PROVIDER_OUTCOMES}
    failure_codes: dict[str, int] = {}
    reference_by_status = {status: 0 for status in FULL_RECORD_PROVIDER_OUTCOMES}
    modeled_seconds = 0.0
    unmodeled_seconds = 0.0
    modeled_background_seconds = 0.0
    fully_modeled_reference_count = 0
    fully_modeled_matched_count = 0

    for wrapped in rows:
        base = wrapped["benchmark_row"]
        attempt = wrapped["provider_attempt"]
        outcome = attempt["outcome_status"]
        statuses[outcome] += 1
        references = base["reference_events"]
        predictions = base["predicted_events"]
        reference_by_status[outcome] += len(references)
        coverage = attempt["coverage_receipt"]
        modeled_stop = float(coverage["modeled_target_coverage_seconds"])
        modeled_seconds += modeled_stop
        unmodeled_seconds += float(coverage["unmodeled_target_coverage_seconds"])
        modeled_seizure_seconds = sum(
            max(
                0.0,
                min(modeled_stop, float(reference["stop_seconds"]))
                - float(reference["start_seconds"]),
            )
            for reference in references
            if float(reference["start_seconds"]) < modeled_stop
        )
        modeled_background_seconds += max(0.0, modeled_stop - modeled_seizure_seconds)
        matches = _ordered_event_matching(references, predictions)
        matched_reference_indices = {
            reference_index for reference_index, _, _ in matches
        }
        for reference_index, reference in enumerate(references):
            if float(reference["stop_seconds"]) <= modeled_stop + 1e-8:
                fully_modeled_reference_count += 1
                if reference_index in matched_reference_indices:
                    fully_modeled_matched_count += 1
        if outcome == "technical_failure":
            code = str(attempt["technical_failure"]["failure_code"])
            failure_codes[code] = failure_codes.get(code, 0) + 1

    recording_count = len(rows)
    return {
        "recording_count": recording_count,
        "outcome_recording_counts": statuses,
        "complete_outcome_recording_count": (
            statuses["completed_with_alarms"] + statuses["completed_zero_alarm"]
        ),
        "complete_outcome_recording_rate": (
            statuses["completed_with_alarms"] + statuses["completed_zero_alarm"]
        )
        / recording_count,
        "technical_failure_recording_rate": statuses["technical_failure"]
        / recording_count,
        "partial_coverage_recording_rate": statuses["partial_coverage"]
        / recording_count,
        "reference_event_counts_by_outcome": reference_by_status,
        "technical_failure_code_counts": dict(sorted(failure_codes.items())),
        "modeled_eeg_hours": modeled_seconds / 3600.0,
        "unmodeled_eeg_hours": unmodeled_seconds / 3600.0,
        "modeled_background_hours": modeled_background_seconds / 3600.0,
        "fully_modeled_reference_event_count": fully_modeled_reference_count,
        "fully_modeled_matched_event_count": fully_modeled_matched_count,
        "fully_modeled_event_sensitivity": (
            None
            if fully_modeled_reference_count == 0
            else fully_modeled_matched_count / fully_modeled_reference_count
        ),
    }


def _reconstruct_wrapped_rows(
    benchmark_rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attempt_map = {str(row["recording_id"]): deepcopy(row) for row in attempts}
    wrapped: list[dict[str, Any]] = []
    for benchmark_row in benchmark_rows:
        recording_id = str(benchmark_row["recording_id"])
        if recording_id not in attempt_map:
            raise ValueError("attempt/core benchmark recording roster drifted")
        wrapped.append(
            {
                "benchmark_row": deepcopy(benchmark_row),
                "provider_attempt": attempt_map[recording_id],
            }
        )
    return validate_intention_to_deploy_rows(wrapped)


def evaluate_intention_to_deploy_continuous_detection(
    *,
    rows: Sequence[Mapping[str, Any]],
    **benchmark_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate interval metrics while retaining every provider terminal outcome."""

    validated = validate_intention_to_deploy_rows(rows)
    base_rows = [deepcopy(row["benchmark_row"]) for row in validated]
    provider_id = str(validated[0]["provider_attempt"]["provider_id"])
    supplied_provider_id = benchmark_kwargs.get("provider_id")
    if supplied_provider_id is not None and supplied_provider_id != provider_id:
        raise ValueError("benchmark provider ID disagrees with provider attempts")
    # ``None`` is semantically the same as an omitted admission.  In
    # development splits the core benchmark still requires the provider ID,
    # so an explicitly supplied ``source_eval_admission=None`` must not
    # suppress the attempt-derived provider binding.  On source-eval the core
    # evaluator will continue to fail closed because a replayed non-null
    # admission is mandatory there.
    if benchmark_kwargs.get("source_eval_admission") is None:
        benchmark_kwargs["provider_id"] = provider_id
    core = evaluate_patient_level_continuous_detection(
        rows=base_rows,
        **benchmark_kwargs,
    )
    if core["provider_id"] != provider_id:
        raise ValueError("source-eval admission provider disagrees with attempts")
    attempts = sorted(
        (deepcopy(row["provider_attempt"]) for row in validated),
        key=lambda row: str(row["recording_id"]),
    )
    metrics = _attempt_metrics(validated)
    benchmark_row_roster = sorted(
        base_rows,
        key=lambda row: (str(row["patient_id"]), str(row["recording_id"])),
    )
    core_metrics = core["metrics"]
    modeled_background_hours = metrics["modeled_background_hours"]
    metrics["background_only_false_alarms_per_modeled_background_hour"] = (
        None
        if modeled_background_hours <= 0
        else core_metrics["background_only_false_alarm_count"]
        / modeled_background_hours
    )
    metrics["unmatched_alarms_per_modeled_background_hour"] = (
        None
        if modeled_background_hours <= 0
        else core_metrics["false_alarm_count"] / modeled_background_hours
    )
    limitations: list[str] = []
    if metrics["outcome_recording_counts"]["technical_failure"]:
        limitations.append(
            "technical_failures_retained_in_intention_to_deploy_denominator"
        )
    if metrics["outcome_recording_counts"]["partial_coverage"]:
        limitations.append(
            "partial_coverage_retained_in_intention_to_deploy_denominator"
        )
    body: dict[str, Any] = {
        "schema_version": INTENTION_TO_DEPLOY_BENCHMARK_SCHEMA_VERSION,
        "method_id": INTENTION_TO_DEPLOY_BENCHMARK_METHOD_ID,
        "receipt_id": "DETITD-PENDING",
        "provider_id": provider_id,
        "core_benchmark_receipt": core,
        "benchmark_row_roster": benchmark_row_roster,
        "benchmark_row_roster_sha256": _canonical_sha256(benchmark_row_roster),
        "provider_attempt_roster": attempts,
        "provider_attempt_roster_sha256": _canonical_sha256(attempts),
        "provider_attempt_metrics": metrics,
        "qualification_limitations": limitations,
        "production_promotion_status": (
            "research_only_attempt_aware_metrics_not_independent_promotion_authority"
        ),
        "sota_claim_authorized": False,
    }
    body["receipt_id"] = "DETITD-" + _canonical_sha256(body)[:24]
    return validate_intention_to_deploy_benchmark_receipt(body)


def validate_intention_to_deploy_benchmark_receipt(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "receipt_id",
        "provider_id",
        "core_benchmark_receipt",
        "benchmark_row_roster",
        "benchmark_row_roster_sha256",
        "provider_attempt_roster",
        "provider_attempt_roster_sha256",
        "provider_attempt_metrics",
        "qualification_limitations",
        "production_promotion_status",
        "sota_claim_authorized",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("intention-to-deploy receipt has missing or unknown fields")
    data = deepcopy(value)
    if data["schema_version"] != INTENTION_TO_DEPLOY_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("intention-to-deploy receipt schema drifted")
    if data["method_id"] != INTENTION_TO_DEPLOY_BENCHMARK_METHOD_ID:
        raise ValueError("intention-to-deploy method drifted")
    provider_id = _identifier(data["provider_id"], "provider ID")
    core = validate_continuous_detection_benchmark_receipt(
        data["core_benchmark_receipt"]
    )
    if core["provider_id"] != provider_id:
        raise ValueError("intention-to-deploy provider/core binding drifted")
    if not isinstance(data["benchmark_row_roster"], list):
        raise TypeError("benchmark row roster must be an array")
    benchmark_rows = validate_continuous_benchmark_rows(data["benchmark_row_roster"])
    expected_benchmark_rows = sorted(
        benchmark_rows,
        key=lambda row: (str(row["patient_id"]), str(row["recording_id"])),
    )
    if benchmark_rows != expected_benchmark_rows:
        raise ValueError("benchmark row roster must be canonically sorted")
    if data["benchmark_row_roster_sha256"] != _canonical_sha256(benchmark_rows):
        raise ValueError("benchmark row roster hash drifted")
    if core["reference_inventory_sha256"] != _canonical_sha256(
        _reference_inventory(benchmark_rows)
    ):
        raise ValueError("benchmark row/core reference inventory drifted")
    if core["prediction_inventory_sha256"] != _canonical_sha256(
        _prediction_inventory(benchmark_rows)
    ):
        raise ValueError("benchmark row/core prediction inventory drifted")
    if not isinstance(data["provider_attempt_roster"], list):
        raise TypeError("provider attempt roster must be an array")
    attempts = [
        validate_provider_attempt_projection(row)
        for row in data["provider_attempt_roster"]
    ]
    expected_attempts = sorted(attempts, key=lambda row: str(row["recording_id"]))
    if attempts != expected_attempts:
        raise ValueError("provider attempt roster must be canonically sorted")
    if any(row["provider_id"] != provider_id for row in attempts):
        raise ValueError("provider attempt roster mixes providers")
    if data["provider_attempt_roster_sha256"] != _canonical_sha256(attempts):
        raise ValueError("provider attempt roster hash drifted")
    wrapped = _reconstruct_wrapped_rows(benchmark_rows, attempts)
    expected_metrics = _attempt_metrics(wrapped)
    modeled_background_hours = expected_metrics["modeled_background_hours"]
    expected_metrics["background_only_false_alarms_per_modeled_background_hour"] = (
        None
        if modeled_background_hours <= 0
        else core["metrics"]["background_only_false_alarm_count"]
        / modeled_background_hours
    )
    expected_metrics["unmatched_alarms_per_modeled_background_hour"] = (
        None
        if modeled_background_hours <= 0
        else core["metrics"]["false_alarm_count"] / modeled_background_hours
    )
    if data["provider_attempt_metrics"] != expected_metrics:
        raise ValueError("provider attempt metrics are not replayable")
    expected_limitations: list[str] = []
    if expected_metrics["outcome_recording_counts"]["technical_failure"]:
        expected_limitations.append(
            "technical_failures_retained_in_intention_to_deploy_denominator"
        )
    if expected_metrics["outcome_recording_counts"]["partial_coverage"]:
        expected_limitations.append(
            "partial_coverage_retained_in_intention_to_deploy_denominator"
        )
    if data["qualification_limitations"] != expected_limitations:
        raise ValueError("intention-to-deploy limitations are not replayable")
    if data["production_promotion_status"] != (
        "research_only_attempt_aware_metrics_not_independent_promotion_authority"
    ):
        raise ValueError("intention-to-deploy receipt cannot self-promote")
    if data["sota_claim_authorized"] is not False:
        raise ValueError("intention-to-deploy receipt cannot authorize SOTA")
    digest = deepcopy(data)
    digest["receipt_id"] = "DETITD-PENDING"
    if data["receipt_id"] != "DETITD-" + _canonical_sha256(digest)[:24]:
        raise ValueError("intention-to-deploy receipt ID does not bind its content")
    return data


__all__ = [
    "INTENTION_TO_DEPLOY_BENCHMARK_METHOD_ID",
    "INTENTION_TO_DEPLOY_BENCHMARK_SCHEMA_VERSION",
    "PROVIDER_ATTEMPT_PROJECTION_SCHEMA_VERSION",
    "evaluate_intention_to_deploy_continuous_detection",
    "materialize_provider_attempt_projection",
    "validate_intention_to_deploy_benchmark_receipt",
    "validate_intention_to_deploy_rows",
    "validate_provider_attempt_projection",
]
