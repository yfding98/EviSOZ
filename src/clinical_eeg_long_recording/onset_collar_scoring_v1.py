"""Independent one-to-one onset-collar scoring for continuous EEG alarms.

The strict continuous benchmark pairs events only when their intervals have
positive overlap.  Its ``onset_absolute_hit_rate`` consequently answers a
conditional question: among strict-overlap pairs, how many starts are close to
the reference start (with every reference retained in the denominator)?

This module implements the different question requested for alarm detection:
can one predicted alarm onset be assigned to one reference onset inside an
explicit early/late collar, irrespective of interval overlap?  The two tracks
must remain separately named and reported.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from .continuous_detection_benchmark import validate_continuous_benchmark_rows


ONSET_COLLAR_SCHEMA_VERSION = "continuous_onset_collar_metrics_v1"
ONSET_COLLAR_METHOD_ID = (
    "ordered_one_to_one_onset_matching_max_count_then_min_absolute_error_v1"
)


def _nonnegative_finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be a finite non-negative number")
    return result


def _start(event: Mapping[str, float]) -> float:
    return float(event["start_seconds"])


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _f1(precision: float | None, sensitivity: float | None) -> float | None:
    if precision is None or sensitivity is None:
        return None
    if precision + sensitivity == 0:
        return 0.0
    return 2.0 * precision * sensitivity / (precision + sensitivity)


def ordered_onset_collar_matching(
    references: Sequence[Mapping[str, float]],
    predictions: Sequence[Mapping[str, float]],
    *,
    early_seconds: float,
    late_seconds: float,
) -> list[tuple[int, int, float]]:
    """Return ordered one-to-one onset matches inside an asymmetric collar.

    A prediction is eligible when ``-early_seconds <= prediction - reference
    <= late_seconds``.  Dynamic programming first maximizes the number of
    matches and then minimizes total absolute onset error.  The signed error in
    each returned tuple is positive for a late alarm and negative for an early
    alarm.  Input order is therefore part of the contract; benchmark row
    validation supplies monotonically ordered, non-overlapping event lists.
    """

    early = _nonnegative_finite(early_seconds, context="early_seconds")
    late = _nonnegative_finite(late_seconds, context="late_seconds")
    n_reference = len(references)
    n_prediction = len(predictions)

    # Score: (number matched, negative total absolute onset error).
    scores: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(n_prediction + 1)]
        for _ in range(n_reference + 1)
    ]
    parents: list[list[tuple[int, int, str] | None]] = [
        [None for _ in range(n_prediction + 1)]
        for _ in range(n_reference + 1)
    ]
    for reference_index in range(1, n_reference + 1):
        parents[reference_index][0] = (
            reference_index - 1,
            0,
            "skip_reference",
        )
    for prediction_index in range(1, n_prediction + 1):
        parents[0][prediction_index] = (
            0,
            prediction_index - 1,
            "skip_prediction",
        )

    # Prefer a match at an exact score tie, then skip the later prediction.
    # This makes the backtrace deterministic without changing either objective.
    action_priority = {"skip_reference": 0, "skip_prediction": 1, "match": 2}
    for reference_index in range(1, n_reference + 1):
        for prediction_index in range(1, n_prediction + 1):
            candidates: list[
                tuple[tuple[int, float], tuple[int, int, str]]
            ] = [
                (
                    scores[reference_index - 1][prediction_index],
                    (reference_index - 1, prediction_index, "skip_reference"),
                ),
                (
                    scores[reference_index][prediction_index - 1],
                    (reference_index, prediction_index - 1, "skip_prediction"),
                ),
            ]
            signed_error = _start(predictions[prediction_index - 1]) - _start(
                references[reference_index - 1]
            )
            if -early - 1e-12 <= signed_error <= late + 1e-12:
                previous = scores[reference_index - 1][prediction_index - 1]
                candidates.append(
                    (
                        (previous[0] + 1, previous[1] - abs(signed_error)),
                        (reference_index - 1, prediction_index - 1, "match"),
                    )
                )
            best_score, best_parent = max(
                candidates,
                key=lambda item: (item[0], action_priority[item[1][2]]),
            )
            scores[reference_index][prediction_index] = best_score
            parents[reference_index][prediction_index] = best_parent

    matches: list[tuple[int, int, float]] = []
    reference_index = n_reference
    prediction_index = n_prediction
    while reference_index or prediction_index:
        parent = parents[reference_index][prediction_index]
        if parent is None:
            raise RuntimeError("onset-collar matching backtrace is incomplete")
        previous_reference, previous_prediction, action = parent
        if action == "match":
            signed_error = _start(predictions[prediction_index - 1]) - _start(
                references[reference_index - 1]
            )
            matches.append(
                (reference_index - 1, prediction_index - 1, signed_error)
            )
        reference_index = previous_reference
        prediction_index = previous_prediction
    matches.reverse()
    return matches


def _collar_key(early_seconds: float, late_seconds: float) -> str:
    if early_seconds == late_seconds:
        return f"plus_minus_{early_seconds:g}s"
    return f"early_{early_seconds:g}s_late_{late_seconds:g}s"


def aggregate_onset_collar_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    collars: Sequence[tuple[float, float]] = (
        (1.0, 1.0),
        (3.0, 3.0),
        (5.0, 5.0),
        (10.0, 10.0),
        (30.0, 60.0),
    ),
) -> dict[str, Any]:
    """Aggregate independent onset-collar metrics over validated recordings."""

    validated_rows = validate_continuous_benchmark_rows(rows)
    normalized_collars: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for index, raw in enumerate(collars):
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise TypeError(f"collars[{index}] must be an (early, late) pair")
        collar = (
            _nonnegative_finite(raw[0], context=f"collars[{index}].early"),
            _nonnegative_finite(raw[1], context=f"collars[{index}].late"),
        )
        if collar in seen:
            raise ValueError("onset collars must be unique")
        seen.add(collar)
        normalized_collars.append(collar)
    if not normalized_collars:
        raise ValueError("at least one onset collar is required")

    total_duration_seconds = sum(
        float(row["duration_seconds"]) for row in validated_rows
    )
    reference_count = sum(len(row["reference_events"]) for row in validated_rows)
    prediction_count = sum(len(row["predicted_events"]) for row in validated_rows)
    results: dict[str, Any] = {}
    for early, late in normalized_collars:
        signed_errors: list[float] = []
        matched_count = 0
        for row in validated_rows:
            matches = ordered_onset_collar_matching(
                row["reference_events"],
                row["predicted_events"],
                early_seconds=early,
                late_seconds=late,
            )
            matched_count += len(matches)
            signed_errors.extend(match[2] for match in matches)
        false_alarm_count = prediction_count - matched_count
        sensitivity = _safe_rate(matched_count, reference_count)
        precision = _safe_rate(matched_count, prediction_count)
        false_alarms_per_hour = _safe_rate(
            false_alarm_count, total_duration_seconds / 3600.0
        )
        absolute_errors = [abs(value) for value in signed_errors]
        results[_collar_key(early, late)] = {
            "early_seconds": early,
            "late_seconds": late,
            "matched_onset_count": matched_count,
            "reference_event_denominator": reference_count,
            "predicted_alarm_denominator": prediction_count,
            "unmatched_prediction_count": false_alarm_count,
            "sensitivity": sensitivity,
            "precision": precision,
            "f1": _f1(precision, sensitivity),
            "unmatched_predictions_per_recording_hour": false_alarms_per_hour,
            "unmatched_predictions_per_24h": (
                None
                if false_alarms_per_hour is None
                else 24.0 * false_alarms_per_hour
            ),
            "matched_signed_onset_error_mean_seconds": (
                None
                if not signed_errors
                else float(sum(signed_errors) / len(signed_errors))
            ),
            "matched_signed_onset_error_median_seconds": (
                None
                if not signed_errors
                else float(statistics.median(signed_errors))
            ),
            "matched_absolute_onset_error_mean_seconds": (
                None
                if not absolute_errors
                else float(sum(absolute_errors) / len(absolute_errors))
            ),
            "matched_absolute_onset_error_median_seconds": (
                None
                if not absolute_errors
                else float(statistics.median(absolute_errors))
            ),
        }

    return {
        "schema_version": ONSET_COLLAR_SCHEMA_VERSION,
        "method_id": ONSET_COLLAR_METHOD_ID,
        "semantics": (
            "prediction_start_in_reference_start_collar_independent_of_interval_"
            "overlap_ordered_one_to_one"
        ),
        "not_equivalent_to_strict_interval_overlap_metrics": True,
        "not_equivalent_to_SzCORE_event_overlap_tolerance": True,
        "recording_count": len(validated_rows),
        "total_recording_hours": total_duration_seconds / 3600.0,
        "reference_event_count": reference_count,
        "predicted_alarm_count": prediction_count,
        "collars": results,
    }


__all__ = [
    "ONSET_COLLAR_METHOD_ID",
    "ONSET_COLLAR_SCHEMA_VERSION",
    "aggregate_onset_collar_metrics",
    "ordered_onset_collar_matching",
]
