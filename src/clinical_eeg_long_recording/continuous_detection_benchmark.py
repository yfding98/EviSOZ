"""Patient-level evaluation for continuous long-EEG seizure alarms.

This benchmark is deliberately model-neutral.  It consumes frozen prediction
intervals and reference seizure intervals after inference has completed.  The
reference intervals must never be passed to the detector provider itself.

Matching is one-to-one and order preserving.  A single long alarm therefore
cannot claim multiple adjacent reference seizures.  Event sensitivity,
unmatched alarms/recording-hour, true background-only alarm rate, duplicate or
fragment alarms overlapping a reference, time-in-warning, event precision/F1,
signed and absolute onset latency with explicit coverage, onset hit rates,
event IoU and typed onset/offset boundary F1 are reported together.  Confidence
intervals resample patients, not recordings or events.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .continuous_detection import CONTINUOUS_DETECTION_METHOD_ID
from .continuous_detection_source_eval_admission import (
    ValidatedSourceEvalAdmission,
    authorize_source_eval_benchmark_rows,
    source_eval_admission_benchmark_binding,
)


CONTINUOUS_BENCHMARK_SCHEMA_VERSION = "patient_level_continuous_seizure_benchmark_v5"
CONTINUOUS_BENCHMARK_METHOD_ID = (
    "ordered_one_to_one_interval_matching_with_alarm_decomposition_v2"
)
DEFAULT_TOLERANCES_SECONDS = (1.0, 3.0, 5.0, 10.0)
REFERENCE_DURATION_STRATA_SECONDS = (8.0, 30.0)

_EXECUTION_RECEIPT_FIELDS = {
    "edf_io_seconds",
    "preprocessing_seconds",
    "inference_seconds",
    "postprocessing_seconds",
    "total_wall_seconds",
    "gpu_active_seconds",
    "gpu_measurement_status",
    "peak_gpu_memory_bytes",
    "peak_host_memory_bytes",
    "service_state",
    "device_type",
    "native_preprocessing_receipt_id",
    "native_preprocessing_receipt_sha256",
    "complete_recording_coverage",
}

_SOURCE_EVAL_ADMISSION_BINDING_FIELDS = {
    "admission_id",
    "admission_sha256",
    "provider_id",
    "operating_point_id",
    "calibration_receipt_id",
    "calibration_receipt_sha256",
    "provider_definition_sha256",
    "decoder_method_id",
    "decoder_code_sha256",
    "decoder_policy_sha256",
    "decoding_receipt_roster_sha256",
    "lockbox_access_ledger_sha256",
    "source_dev_patient_roster_sha256",
    "source_dev_recording_roster_sha256",
    "source_eval_patient_roster_sha256",
    "source_eval_recording_roster_sha256",
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


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 512 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _optional_nonnegative_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, context)
    if result < 0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _optional_nonnegative_int(value: object, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer or null")
    return int(value)


def _validate_execution_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _EXECUTION_RECEIPT_FIELDS:
        raise ValueError("execution receipt has missing or unknown fields")
    result = deepcopy(value)
    stage_names = (
        "edf_io_seconds",
        "preprocessing_seconds",
        "inference_seconds",
        "postprocessing_seconds",
        "total_wall_seconds",
    )
    for name in stage_names:
        result[name] = _optional_nonnegative_number(result[name], name)
        if result[name] is None:
            raise ValueError(f"execution receipt {name} must be measured")
    stage_sum = sum(float(result[name]) for name in stage_names[:-1])
    if float(result["total_wall_seconds"]) + 1e-9 < stage_sum:
        raise ValueError("total wall time is smaller than measured stage times")
    result["gpu_active_seconds"] = _optional_nonnegative_number(
        result["gpu_active_seconds"], "gpu_active_seconds"
    )
    result["peak_gpu_memory_bytes"] = _optional_nonnegative_int(
        result["peak_gpu_memory_bytes"], "peak_gpu_memory_bytes"
    )
    result["peak_host_memory_bytes"] = _optional_nonnegative_int(
        result["peak_host_memory_bytes"], "peak_host_memory_bytes"
    )
    if result["gpu_measurement_status"] not in {
        "measured",
        "not_applicable_cpu_only",
        "not_measured",
    }:
        raise ValueError("execution receipt GPU measurement status is invalid")
    if result["gpu_measurement_status"] == "measured":
        if (
            result["gpu_active_seconds"] is None
            or result["peak_gpu_memory_bytes"] is None
        ):
            raise ValueError("measured GPU receipt lacks time or peak memory")
    elif (
        result["gpu_active_seconds"] is not None
        or result["peak_gpu_memory_bytes"] is not None
    ):
        raise ValueError("unmeasured/not-applicable GPU receipt must use null values")
    if result["service_state"] not in {"cold", "warm"}:
        raise ValueError("execution receipt service_state must be cold or warm")
    result["device_type"] = _identifier(result["device_type"], "device_type")
    result["native_preprocessing_receipt_id"] = _identifier(
        result["native_preprocessing_receipt_id"],
        "native_preprocessing_receipt_id",
    )
    sha256 = result["native_preprocessing_receipt_sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("native preprocessing receipt SHA-256 is invalid")
    if result["complete_recording_coverage"] is not True:
        raise ValueError("execution receipt must cover the complete recording")
    return result


def _validate_tolerances(values: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("tolerances must be a sequence")
    result = tuple(_finite(value, "boundary tolerance") for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError("tolerances must be positive")
    if tuple(sorted(set(result))) != result:
        raise ValueError("tolerances must be strictly increasing and unique")
    return result


def _validate_events(
    value: object,
    *,
    duration_seconds: float,
    context: str,
) -> list[dict[str, float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    events: list[dict[str, float]] = []
    previous_stop = 0.0
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"start_seconds", "stop_seconds"}:
            raise ValueError(f"{context}[{index}] has invalid fields")
        start = _finite(raw["start_seconds"], f"{context}[{index}] start")
        stop = _finite(raw["stop_seconds"], f"{context}[{index}] stop")
        if start < 0 or stop <= start or stop > duration_seconds + 1e-9:
            raise ValueError(f"{context}[{index}] is outside the recording")
        if index and start < previous_stop - 1e-9:
            raise ValueError(f"{context} must be sorted and non-overlapping")
        events.append({"start_seconds": start, "stop_seconds": stop})
        previous_stop = stop
    return events


def validate_continuous_benchmark_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate one complete evaluation row per long recording."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("benchmark rows must be a non-empty sequence")
    required = {
        "patient_id",
        "recording_id",
        "split",
        "duration_seconds",
        "reference_events",
        "predicted_events",
    }
    result: list[dict[str, Any]] = []
    recording_ids: set[str] = set()
    patient_splits: dict[str, str] = {}
    for index, raw in enumerate(rows):
        if type(raw) is not dict or set(raw) not in {
            frozenset(required),
            frozenset(required | {"execution_receipt"}),
        }:
            raise ValueError(f"benchmark row {index} has missing or unknown fields")
        patient_id = _identifier(raw["patient_id"], "patient_id")
        recording_id = _identifier(raw["recording_id"], "recording_id")
        split = _identifier(raw["split"], "split")
        duration = _finite(raw["duration_seconds"], "duration_seconds")
        if duration <= 0:
            raise ValueError("recording duration must be positive")
        if recording_id in recording_ids:
            raise ValueError("benchmark recording IDs must be unique")
        recording_ids.add(recording_id)
        previous_split = patient_splits.setdefault(patient_id, split)
        if previous_split != split:
            raise ValueError("one patient occurs in multiple benchmark splits")
        normalized_row: dict[str, Any] = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "split": split,
            "duration_seconds": duration,
            "reference_events": _validate_events(
                raw["reference_events"],
                duration_seconds=duration,
                context="reference_events",
            ),
            "predicted_events": _validate_events(
                raw["predicted_events"],
                duration_seconds=duration,
                context="predicted_events",
            ),
        }
        if "execution_receipt" in raw:
            normalized_row["execution_receipt"] = _validate_execution_receipt(
                raw["execution_receipt"]
            )
        result.append(normalized_row)
    return result


def _intersection_over_union(
    reference: Mapping[str, float], prediction: Mapping[str, float]
) -> float:
    intersection = max(
        0.0,
        min(reference["stop_seconds"], prediction["stop_seconds"])
        - max(reference["start_seconds"], prediction["start_seconds"]),
    )
    if intersection <= 0:
        return 0.0
    union = max(reference["stop_seconds"], prediction["stop_seconds"]) - min(
        reference["start_seconds"], prediction["start_seconds"]
    )
    return intersection / union


def _ordered_event_matching(
    references: Sequence[Mapping[str, float]],
    predictions: Sequence[Mapping[str, float]],
) -> list[tuple[int, int, float]]:
    """Maximize match count, then IoU, then minimize absolute onset error."""

    n_reference = len(references)
    n_prediction = len(predictions)
    # Score is (number matched, total IoU, negative absolute onset error).
    scores: list[list[tuple[int, float, float]]] = [
        [(0, 0.0, 0.0) for _ in range(n_prediction + 1)] for _ in range(n_reference + 1)
    ]
    parents: list[list[tuple[int, int, str] | None]] = [
        [None for _ in range(n_prediction + 1)] for _ in range(n_reference + 1)
    ]
    for i in range(1, n_reference + 1):
        parents[i][0] = (i - 1, 0, "skip_reference")
    for j in range(1, n_prediction + 1):
        parents[0][j] = (0, j - 1, "skip_prediction")

    action_priority = {"skip_reference": 0, "skip_prediction": 1, "match": 2}
    for i in range(1, n_reference + 1):
        for j in range(1, n_prediction + 1):
            candidates: list[tuple[tuple[int, float, float], tuple[int, int, str]]] = [
                (scores[i - 1][j], (i - 1, j, "skip_reference")),
                (scores[i][j - 1], (i, j - 1, "skip_prediction")),
            ]
            iou = _intersection_over_union(references[i - 1], predictions[j - 1])
            if iou > 0:
                previous = scores[i - 1][j - 1]
                onset_error = abs(
                    prediction_start(predictions[j - 1])
                    - reference_start(references[i - 1])
                )
                candidates.append(
                    (
                        (
                            previous[0] + 1,
                            previous[1] + iou,
                            previous[2] - onset_error,
                        ),
                        (i - 1, j - 1, "match"),
                    )
                )
            best_score, best_parent = max(
                candidates,
                key=lambda item: (item[0], action_priority[item[1][2]]),
            )
            scores[i][j] = best_score
            parents[i][j] = best_parent

    matches: list[tuple[int, int, float]] = []
    i, j = n_reference, n_prediction
    while i or j:
        parent = parents[i][j]
        if parent is None:
            raise RuntimeError("event matching backtrace is incomplete")
        previous_i, previous_j, action = parent
        if action == "match":
            matches.append(
                (
                    i - 1,
                    j - 1,
                    _intersection_over_union(references[i - 1], predictions[j - 1]),
                )
            )
        i, j = previous_i, previous_j
    matches.reverse()
    return matches


def reference_start(event: Mapping[str, float]) -> float:
    return float(event["start_seconds"])


def prediction_start(event: Mapping[str, float]) -> float:
    return float(event["start_seconds"])


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("percentile probability is invalid")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _event_overlap_seconds(
    event: Mapping[str, float], references: Sequence[Mapping[str, float]]
) -> float:
    """Return event duration covered by the non-overlapping reference union."""

    start = float(event["start_seconds"])
    stop = float(event["stop_seconds"])
    return sum(
        max(
            0.0,
            min(stop, float(reference["stop_seconds"]))
            - max(start, float(reference["start_seconds"])),
        )
        for reference in references
    )


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    tolerances: Sequence[float],
) -> dict[str, Any]:
    reference_count = 0
    prediction_count = 0
    matched_count = 0
    total_duration_seconds = 0.0
    total_background_seconds = 0.0
    time_in_warning_seconds = 0.0
    false_positive_duration_seconds = 0.0
    background_only_false_alarm_count = 0
    reference_overlap_unmatched_alarm_count = 0
    latencies: list[float] = []
    offset_latencies: list[float] = []
    matched_ious: list[float] = []
    onset_hits = {tolerance: 0 for tolerance in tolerances}
    boundary_hits = {tolerance: 0 for tolerance in tolerances}
    duration_strata: dict[str, dict[str, Any]] = {
        "lt_8s": {"reference_count": 0, "matched_count": 0, "latencies": []},
        "8_to_30s": {"reference_count": 0, "matched_count": 0, "latencies": []},
        "gt_30s": {"reference_count": 0, "matched_count": 0, "latencies": []},
    }
    for values in duration_strata.values():
        values["onset_hits"] = {tolerance: 0 for tolerance in tolerances}

    for row in rows:
        references = row["reference_events"]
        predictions = row["predicted_events"]
        matches = _ordered_event_matching(references, predictions)
        matched_prediction_indices = {
            prediction_index for _, prediction_index, _ in matches
        }
        match_by_reference = {
            reference_index: prediction_index
            for reference_index, prediction_index, _ in matches
        }
        reference_count += len(references)
        prediction_count += len(predictions)
        matched_count += len(matches)
        duration = float(row["duration_seconds"])
        total_duration_seconds += duration
        seizure_seconds = sum(
            float(event["stop_seconds"] - event["start_seconds"])
            for event in references
        )
        total_background_seconds += max(0.0, duration - seizure_seconds)
        for prediction_index, prediction in enumerate(predictions):
            prediction_duration = float(
                prediction["stop_seconds"] - prediction["start_seconds"]
            )
            overlap_seconds = _event_overlap_seconds(prediction, references)
            time_in_warning_seconds += prediction_duration
            false_positive_duration_seconds += max(
                0.0, prediction_duration - overlap_seconds
            )
            if prediction_index not in matched_prediction_indices:
                if overlap_seconds <= 1e-12:
                    background_only_false_alarm_count += 1
                else:
                    reference_overlap_unmatched_alarm_count += 1
        for reference_index, reference in enumerate(references):
            reference_duration = float(
                reference["stop_seconds"] - reference["start_seconds"]
            )
            if reference_duration < REFERENCE_DURATION_STRATA_SECONDS[0]:
                stratum = "lt_8s"
            elif reference_duration <= REFERENCE_DURATION_STRATA_SECONDS[1]:
                stratum = "8_to_30s"
            else:
                stratum = "gt_30s"
            stratum_values = duration_strata[stratum]
            stratum_values["reference_count"] += 1
            prediction_index = match_by_reference.get(reference_index)
            if prediction_index is None:
                continue
            stratum_values["matched_count"] += 1
            onset_latency = prediction_start(
                predictions[prediction_index]
            ) - reference_start(reference)
            stratum_values["latencies"].append(onset_latency)
            for tolerance in tolerances:
                if abs(onset_latency) <= tolerance + 1e-12:
                    stratum_values["onset_hits"][tolerance] += 1
        for reference_index, prediction_index, iou in matches:
            reference = references[reference_index]
            prediction = predictions[prediction_index]
            onset_latency = prediction_start(prediction) - reference_start(reference)
            offset_latency = float(prediction["stop_seconds"]) - float(
                reference["stop_seconds"]
            )
            latencies.append(onset_latency)
            offset_latencies.append(offset_latency)
            matched_ious.append(iou)
            for tolerance in tolerances:
                if abs(onset_latency) <= tolerance + 1e-12:
                    onset_hits[tolerance] += 1
                    boundary_hits[tolerance] += 1
                if abs(offset_latency) <= tolerance + 1e-12:
                    boundary_hits[tolerance] += 1

    false_alarm_count = prediction_count - matched_count
    if (
        background_only_false_alarm_count
        + reference_overlap_unmatched_alarm_count
        != false_alarm_count
    ):
        raise RuntimeError("unmatched alarm decomposition is not exhaustive")
    absolute_latencies = [abs(value) for value in latencies]
    absolute_offset_latencies = [abs(value) for value in offset_latencies]
    q25 = _percentile(latencies, 0.25)
    q75 = _percentile(latencies, 0.75)
    absolute_q25 = _percentile(absolute_latencies, 0.25)
    absolute_q75 = _percentile(absolute_latencies, 0.75)
    event_precision = _safe_rate(matched_count, prediction_count)
    event_sensitivity = _safe_rate(matched_count, reference_count)
    false_alarms_per_recording_hour = _safe_rate(
        false_alarm_count, total_duration_seconds / 3600.0
    )
    metrics: dict[str, Any] = {
        "recording_count": len(rows),
        "seizure_free_recording_count": sum(
            not row["reference_events"] for row in rows
        ),
        "reference_event_count": reference_count,
        "predicted_alarm_count": prediction_count,
        "matched_event_count": matched_count,
        "false_alarm_count": false_alarm_count,
        "background_only_false_alarm_count": background_only_false_alarm_count,
        "reference_overlap_unmatched_alarm_count": (
            reference_overlap_unmatched_alarm_count
        ),
        "total_recording_hours": total_duration_seconds / 3600.0,
        "total_background_hours": total_background_seconds / 3600.0,
        "time_in_warning_seconds": time_in_warning_seconds,
        "time_in_warning_fraction_of_recording": _safe_rate(
            time_in_warning_seconds, total_duration_seconds
        ),
        "false_positive_duration_seconds": false_positive_duration_seconds,
        "background_time_in_warning_fraction": _safe_rate(
            false_positive_duration_seconds, total_background_seconds
        ),
        "event_sensitivity": event_sensitivity,
        "event_precision": event_precision,
        "event_f1": _f1(event_precision, event_sensitivity),
        "alarm_false_alarms_per_recording_hour": false_alarms_per_recording_hour,
        "alarm_false_alarms_per_24h": (
            None
            if false_alarms_per_recording_hour is None
            else 24.0 * false_alarms_per_recording_hour
        ),
        "unmatched_alarms_per_background_hour": _safe_rate(
            false_alarm_count, total_background_seconds / 3600.0
        ),
        "background_only_false_alarms_per_background_hour": _safe_rate(
            background_only_false_alarm_count,
            total_background_seconds / 3600.0,
        ),
        "onset_latency_seconds": {
            "matched_event_denominator": len(latencies),
            "reference_event_denominator": reference_count,
            "matched_reference_coverage": _safe_rate(len(latencies), reference_count),
            "signed_mean": (None if not latencies else sum(latencies) / len(latencies)),
            "median": None if not latencies else float(statistics.median(latencies)),
            "q25": q25,
            "q75": q75,
            "iqr": None if q25 is None or q75 is None else q75 - q25,
            "absolute_mean_matched_only": (
                None
                if not absolute_latencies
                else sum(absolute_latencies) / len(absolute_latencies)
            ),
            "absolute_median_matched_only": (
                None
                if not absolute_latencies
                else float(statistics.median(absolute_latencies))
            ),
            "absolute_q25_matched_only": absolute_q25,
            "absolute_q75_matched_only": absolute_q75,
            "absolute_iqr_matched_only": (
                None
                if absolute_q25 is None or absolute_q75 is None
                else absolute_q75 - absolute_q25
            ),
        },
        "offset_latency_seconds": {
            "matched_event_denominator": len(offset_latencies),
            "reference_event_denominator": reference_count,
            "matched_reference_coverage": _safe_rate(
                len(offset_latencies), reference_count
            ),
            "signed_mean": (
                None
                if not offset_latencies
                else sum(offset_latencies) / len(offset_latencies)
            ),
            "median": (
                None
                if not offset_latencies
                else float(statistics.median(offset_latencies))
            ),
            "absolute_mean_matched_only": (
                None
                if not absolute_offset_latencies
                else sum(absolute_offset_latencies) / len(absolute_offset_latencies)
            ),
            "absolute_median_matched_only": (
                None
                if not absolute_offset_latencies
                else float(statistics.median(absolute_offset_latencies))
            ),
        },
        "event_iou": {
            "mean_matched_only": (
                None if not matched_ious else sum(matched_ious) / len(matched_ious)
            ),
            "mean_reference_denominator_unmatched_zero": _safe_rate(
                sum(matched_ious), reference_count
            ),
        },
        "onset_absolute_hit_rate": {},
        "typed_boundary_precision_recall_f1": {},
        "reference_duration_strata": {},
    }
    for tolerance in tolerances:
        key = f"{tolerance:g}s"
        boundary_precision = _safe_rate(boundary_hits[tolerance], 2 * prediction_count)
        boundary_recall = _safe_rate(boundary_hits[tolerance], 2 * reference_count)
        metrics["onset_absolute_hit_rate"][key] = {
            "hit_count": onset_hits[tolerance],
            "reference_event_denominator": reference_count,
            "rate": _safe_rate(onset_hits[tolerance], reference_count),
        }
        metrics["typed_boundary_precision_recall_f1"][key] = {
            "true_positive_boundary_count": boundary_hits[tolerance],
            "predicted_boundary_denominator": 2 * prediction_count,
            "reference_boundary_denominator": 2 * reference_count,
            "precision": boundary_precision,
            "recall": boundary_recall,
            "f1": _f1(boundary_precision, boundary_recall),
        }
    for name, values in duration_strata.items():
        stratum_latencies = list(values["latencies"])
        reference_denominator = int(values["reference_count"])
        matched = int(values["matched_count"])
        metrics["reference_duration_strata"][name] = {
            "reference_event_count": reference_denominator,
            "matched_event_count": matched,
            "event_sensitivity": _safe_rate(matched, reference_denominator),
            "onset_absolute_error_median_matched_only_seconds": (
                None
                if not stratum_latencies
                else float(statistics.median(abs(value) for value in stratum_latencies))
            ),
            "onset_absolute_hit_rate": {
                f"{tolerance:g}s": {
                    "hit_count": int(values["onset_hits"][tolerance]),
                    "reference_event_denominator": reference_denominator,
                    "rate": _safe_rate(
                        int(values["onset_hits"][tolerance]),
                        reference_denominator,
                    ),
                }
                for tolerance in tolerances
            },
        }
    return metrics


def _patient_macro_metrics(
    rows: Sequence[Mapping[str, Any]], tolerances: Sequence[float]
) -> dict[str, Any]:
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)
    patient_metrics = [
        _aggregate_metrics(by_patient[patient_id], tolerances)
        for patient_id in sorted(by_patient)
    ]

    def macro(path: tuple[str, ...]) -> tuple[float | None, int]:
        values: list[float] = []
        for metrics in patient_metrics:
            current: Any = metrics
            for part in path:
                current = current[part]
            if current is not None:
                values.append(float(current))
        return (None if not values else sum(values) / len(values), len(values))

    sensitivity, sensitivity_count = macro(("event_sensitivity",))
    false_alarms, false_alarm_count = macro(("alarm_false_alarms_per_recording_hour",))
    background_only_false_alarms, background_only_false_alarm_count = macro(
        ("background_only_false_alarms_per_background_hour",)
    )
    time_in_warning, time_in_warning_count = macro(
        ("time_in_warning_fraction_of_recording",)
    )
    onset_hit: dict[str, Any] = {}
    for tolerance in tolerances:
        key = f"{tolerance:g}s"
        value, count = macro(("onset_absolute_hit_rate", key, "rate"))
        onset_hit[key] = {"patient_macro_rate": value, "evaluable_patient_count": count}
    return {
        "patient_count": len(by_patient),
        "event_sensitivity_macro": sensitivity,
        "event_sensitivity_evaluable_patient_count": sensitivity_count,
        "alarm_false_alarms_per_recording_hour_macro": false_alarms,
        "false_alarm_rate_evaluable_patient_count": false_alarm_count,
        "background_only_false_alarms_per_background_hour_macro": (
            background_only_false_alarms
        ),
        "background_only_false_alarm_rate_evaluable_patient_count": (
            background_only_false_alarm_count
        ),
        "time_in_warning_fraction_macro": time_in_warning,
        "time_in_warning_evaluable_patient_count": time_in_warning_count,
        "onset_absolute_hit_rate": onset_hit,
    }


def _reference_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "patient_id": str(row["patient_id"]),
                "recording_id": str(row["recording_id"]),
                "split": str(row["split"]),
                "duration_seconds": float(row["duration_seconds"]),
                "reference_events": deepcopy(row["reference_events"]),
            }
            for row in rows
        ],
        key=lambda row: (row["patient_id"], row["recording_id"]),
    )


def _prediction_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "recording_id": str(row["recording_id"]),
                "predicted_events": deepcopy(row["predicted_events"]),
            }
            for row in rows
        ],
        key=lambda row: row["recording_id"],
    )


def _aggregate_execution_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    with_receipts = [row for row in rows if "execution_receipt" in row]
    missing_count = len(rows) - len(with_receipts)
    duration_seconds = sum(float(row["duration_seconds"]) for row in with_receipts)
    duration_hours = duration_seconds / 3600.0
    stage_names = (
        "edf_io_seconds",
        "preprocessing_seconds",
        "inference_seconds",
        "postprocessing_seconds",
        "total_wall_seconds",
    )
    totals = {
        name: sum(float(row["execution_receipt"][name]) for row in with_receipts)
        for name in stage_names
    }
    by_service_state: dict[str, dict[str, float | int | None]] = {}
    for state in ("cold", "warm"):
        selected = [
            row
            for row in with_receipts
            if row["execution_receipt"]["service_state"] == state
        ]
        selected_duration = sum(float(row["duration_seconds"]) for row in selected)
        selected_wall = sum(
            float(row["execution_receipt"]["total_wall_seconds"]) for row in selected
        )
        by_service_state[state] = {
            "recording_count": len(selected),
            "eeg_duration_hours": selected_duration / 3600.0,
            "total_wall_seconds": selected_wall,
            "real_time_factor": _safe_rate(selected_wall, selected_duration),
            "wall_seconds_per_eeg_hour": _safe_rate(
                selected_wall, selected_duration / 3600.0
            ),
        }

    gpu_measured = [
        row
        for row in with_receipts
        if row["execution_receipt"]["gpu_measurement_status"] == "measured"
    ]
    gpu_duration_seconds = sum(float(row["duration_seconds"]) for row in gpu_measured)
    gpu_active_seconds = sum(
        float(row["execution_receipt"]["gpu_active_seconds"]) for row in gpu_measured
    )
    gpu_peaks = [
        int(row["execution_receipt"]["peak_gpu_memory_bytes"]) for row in gpu_measured
    ]
    host_peaks = [
        int(row["execution_receipt"]["peak_host_memory_bytes"])
        for row in with_receipts
        if row["execution_receipt"]["peak_host_memory_bytes"] is not None
    ]
    preprocessing_receipts = sorted(
        {
            (
                str(row["execution_receipt"]["native_preprocessing_receipt_id"]),
                str(row["execution_receipt"]["native_preprocessing_receipt_sha256"]),
            )
            for row in with_receipts
        }
    )
    gpu_status_counts = {
        status: sum(
            row["execution_receipt"]["gpu_measurement_status"] == status
            for row in with_receipts
        )
        for status in ("measured", "not_applicable_cpu_only", "not_measured")
    }
    return {
        "execution_receipt_recording_count": len(with_receipts),
        "execution_receipt_missing_recording_count": missing_count,
        "execution_receipt_coverage": _safe_rate(len(with_receipts), len(rows)),
        "receipted_eeg_duration_hours": duration_hours,
        "stage_total_seconds": totals,
        "total_real_time_factor": _safe_rate(
            totals["total_wall_seconds"], duration_seconds
        ),
        "inference_real_time_factor": _safe_rate(
            totals["inference_seconds"], duration_seconds
        ),
        "total_wall_seconds_per_eeg_hour": _safe_rate(
            totals["total_wall_seconds"], duration_hours
        ),
        "service_state_metrics": by_service_state,
        "gpu_measurement_status_recording_counts": gpu_status_counts,
        "gpu_measured_eeg_duration_hours": gpu_duration_seconds / 3600.0,
        "gpu_active_seconds_total": gpu_active_seconds if gpu_measured else None,
        "gpu_hours_total": (gpu_active_seconds / 3600.0 if gpu_measured else None),
        "gpu_active_seconds_per_eeg_hour": (
            _safe_rate(gpu_active_seconds, gpu_duration_seconds / 3600.0)
            if gpu_measured
            else None
        ),
        "peak_gpu_memory_bytes_max": max(gpu_peaks) if gpu_peaks else None,
        "peak_host_memory_bytes_max": max(host_peaks) if host_peaks else None,
        "device_types": sorted(
            {str(row["execution_receipt"]["device_type"]) for row in with_receipts}
        ),
        "native_preprocessing_receipt_count": len(preprocessing_receipts),
        "native_preprocessing_receipt_roster_sha256": (
            _canonical_sha256(preprocessing_receipts)
            if preprocessing_receipts
            else None
        ),
        "complete_recording_coverage_asserted_for_all_receipts": all(
            row["execution_receipt"]["complete_recording_coverage"] is True
            for row in with_receipts
        ),
    }


def _bootstrap_scalar_metrics(
    metrics: Mapping[str, Any], tolerances: Sequence[float]
) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "event_sensitivity": metrics["event_sensitivity"],
        "event_precision": metrics["event_precision"],
        "event_f1": metrics["event_f1"],
        "alarm_false_alarms_per_recording_hour": metrics[
            "alarm_false_alarms_per_recording_hour"
        ],
        "alarm_false_alarms_per_24h": metrics["alarm_false_alarms_per_24h"],
        "background_only_false_alarms_per_background_hour": metrics[
            "background_only_false_alarms_per_background_hour"
        ],
        "time_in_warning_fraction_of_recording": metrics[
            "time_in_warning_fraction_of_recording"
        ],
        "background_time_in_warning_fraction": metrics[
            "background_time_in_warning_fraction"
        ],
        "patient_macro_event_sensitivity": metrics["patient_macro"][
            "event_sensitivity_macro"
        ],
        "patient_macro_alarm_false_alarms_per_recording_hour": metrics[
            "patient_macro"
        ]["alarm_false_alarms_per_recording_hour_macro"],
        "patient_macro_background_only_false_alarms_per_background_hour": metrics[
            "patient_macro"
        ]["background_only_false_alarms_per_background_hour_macro"],
        "patient_macro_time_in_warning_fraction": metrics["patient_macro"][
            "time_in_warning_fraction_macro"
        ],
        "onset_matched_reference_coverage": metrics["onset_latency_seconds"][
            "matched_reference_coverage"
        ],
        "onset_latency_signed_mean_seconds": metrics["onset_latency_seconds"][
            "signed_mean"
        ],
        "onset_latency_median_seconds": metrics["onset_latency_seconds"]["median"],
        "onset_latency_iqr_seconds": metrics["onset_latency_seconds"]["iqr"],
        "onset_absolute_error_mean_matched_only_seconds": metrics[
            "onset_latency_seconds"
        ]["absolute_mean_matched_only"],
        "onset_absolute_error_median_matched_only_seconds": metrics[
            "onset_latency_seconds"
        ]["absolute_median_matched_only"],
        "mean_event_iou_reference_denominator_unmatched_zero": metrics["event_iou"][
            "mean_reference_denominator_unmatched_zero"
        ],
    }
    for tolerance in tolerances:
        key = f"{tolerance:g}s"
        values[f"onset_absolute_hit_rate_at_{key}"] = metrics[
            "onset_absolute_hit_rate"
        ][key]["rate"]
        values[f"typed_boundary_f1_at_{key}"] = metrics[
            "typed_boundary_precision_recall_f1"
        ][key]["f1"]
    return values


def aggregate_continuous_detection_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerances_seconds: Sequence[float] = DEFAULT_TOLERANCES_SECONDS,
) -> dict[str, Any]:
    """Return validated point metrics without issuing an evaluation receipt.

    This read-only API is intended for source-development operating-point
    selection.  It shares the exact validation and one-to-one matcher used by
    the frozen evaluation receipt while deliberately omitting any claim about
    an operating point having been frozen before evaluation.
    """

    tolerances = _validate_tolerances(tolerances_seconds)
    validated_rows = validate_continuous_benchmark_rows(rows)
    metrics = _aggregate_metrics(validated_rows, tolerances)
    metrics["patient_macro"] = _patient_macro_metrics(validated_rows, tolerances)
    return metrics


def _preflight_benchmark_split(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Read only the split field before any source-eval reference is parsed."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("benchmark rows must be a non-empty sequence")
    splits: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"benchmark row {index} must be an object")
        splits.add(_identifier(raw.get("split"), f"benchmark row {index} split"))
    if len(splits) != 1:
        raise ValueError("one benchmark receipt must contain exactly one frozen split")
    return next(iter(splits))


def _patient_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerances: Sequence[float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    patient_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        patient_rows.setdefault(str(row["patient_id"]), []).append(row)
    patients = sorted(patient_rows)
    random_state = random.Random(seed)
    distributions: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled_rows: list[Mapping[str, Any]] = []
        for draw_index in range(len(patients)):
            sampled_patient = patients[random_state.randrange(len(patients))]
            for row in patient_rows[sampled_patient]:
                bootstrap_row = deepcopy(row)
                # A patient sampled twice must contribute two macro units.  Reusing
                # its original ID would collapse both draws in _patient_macro_metrics.
                bootstrap_row["patient_id"] = f"BOOTSTRAP-DRAW-{draw_index:08d}"
                sampled_rows.append(bootstrap_row)
        sampled_metrics = _aggregate_metrics(sampled_rows, tolerances)
        sampled_metrics["patient_macro"] = _patient_macro_metrics(
            sampled_rows, tolerances
        )
        values = _bootstrap_scalar_metrics(
            sampled_metrics, tolerances
        )
        for name, value in values.items():
            if value is not None and math.isfinite(float(value)):
                distributions.setdefault(name, []).append(float(value))
    intervals: dict[str, Any] = {}
    observed_metrics = _aggregate_metrics(rows, tolerances)
    observed_metrics["patient_macro"] = _patient_macro_metrics(rows, tolerances)
    for name in sorted(_bootstrap_scalar_metrics(observed_metrics, tolerances)):
        values = distributions.get(name, [])
        intervals[name] = {
            "valid_replicates": len(values),
            "lower_2_5_percentile": _percentile(values, 0.025),
            "upper_97_5_percentile": _percentile(values, 0.975),
        }
    return {
        "unit": "patient",
        "method": "percentile_bootstrap",
        "seed": seed,
        "requested_replicates": replicates,
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def evaluate_patient_level_continuous_detection(
    *,
    rows: Sequence[Mapping[str, Any]],
    provider_id: str | None = None,
    operating_point_id: str | None = None,
    operating_point_frozen_before_evaluation: bool | None = None,
    source_eval_admission: ValidatedSourceEvalAdmission | None = None,
    development_patient_ids: Iterable[str] | None = None,
    expected_evaluation_recording_ids: Iterable[str] | None = None,
    tolerances_seconds: Sequence[float] = DEFAULT_TOLERANCES_SECONDS,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 20260820,
) -> dict[str, Any]:
    """Evaluate continuous predictions and return a content-bound receipt."""

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 0
    ):
        raise ValueError("bootstrap_replicates must be a non-negative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    tolerances = _validate_tolerances(tolerances_seconds)
    evaluation_split = _preflight_benchmark_split(rows)
    admission_binding: dict[str, Any] | None = None
    if evaluation_split == "source_eval":
        if source_eval_admission is None:
            raise ValueError(
                "source_eval requires a replayed source-eval admission artifact"
            )
        if (
            provider_id is not None
            or operating_point_id is not None
            or operating_point_frozen_before_evaluation is not None
        ):
            raise ValueError(
                "source_eval provider/operating-point/frozen authority must be "
                "derived from admission, not supplied by the caller"
            )
        # This projection reads prediction intervals and identity/split fields only.
        # It must succeed before validate_continuous_benchmark_rows parses the
        # source-eval reference intervals.
        admission_payload = authorize_source_eval_benchmark_rows(
            source_eval_admission, rows
        )
        provider_id = str(admission_payload["provider_id"])
        operating_point_id = str(admission_payload["operating_point_id"])
        operating_point_frozen_before_evaluation = True
        source_dev_roster = admission_payload["split_roster_receipt"][
            "split_rosters"
        ]["source_dev"]
        source_eval_roster = admission_payload["split_roster_receipt"][
            "split_rosters"
        ]["source_eval"]
        if development_patient_ids is not None and sorted(
            {
                _identifier(value, "development patient ID")
                for value in development_patient_ids
            }
        ) != source_dev_roster["patient_ids"]:
            raise ValueError(
                "caller development roster disagrees with source-eval admission"
            )
        if expected_evaluation_recording_ids is not None and sorted(
            {
                _identifier(value, "expected evaluation recording ID")
                for value in expected_evaluation_recording_ids
            }
        ) != source_eval_roster["recording_ids"]:
            raise ValueError(
                "caller evaluation roster disagrees with source-eval admission"
            )
        development_patient_ids = source_dev_roster["patient_ids"]
        expected_evaluation_recording_ids = source_eval_roster["recording_ids"]
        admission_binding = source_eval_admission_benchmark_binding(
            admission_payload
        )
    else:
        if source_eval_admission is not None:
            raise ValueError("source-eval admission cannot authorize another split")
        provider_id = _identifier(provider_id, "provider_id")
        operating_point_id = _identifier(
            operating_point_id, "operating_point_id"
        )
        if type(operating_point_frozen_before_evaluation) is not bool:
            raise TypeError("operating-point frozen flag must be boolean")
    validated_rows = validate_continuous_benchmark_rows(rows)
    provider_id = _identifier(provider_id, "provider_id")
    operating_point_id = _identifier(operating_point_id, "operating_point_id")
    if type(operating_point_frozen_before_evaluation) is not bool:
        raise TypeError("operating-point frozen flag must be boolean")
    evaluation_patients = sorted({str(row["patient_id"]) for row in validated_rows})
    observed_recordings = sorted(str(row["recording_id"]) for row in validated_rows)
    expected_recording_roster_sha256: str | None = None
    evaluation_inventory_status = "not_verified_no_expected_recording_roster"
    if expected_evaluation_recording_ids is not None:
        expected_recordings = sorted(
            {
                _identifier(value, "expected evaluation recording ID")
                for value in expected_evaluation_recording_ids
            }
        )
        if not expected_recordings:
            raise ValueError("expected evaluation recording roster must not be empty")
        if expected_recordings != observed_recordings:
            missing = sorted(set(expected_recordings).difference(observed_recordings))
            extra = sorted(set(observed_recordings).difference(expected_recordings))
            raise ValueError(
                "evaluation recording inventory mismatch: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        expected_recording_roster_sha256 = _canonical_sha256(expected_recordings)
        evaluation_inventory_status = "verified_complete_expected_recording_inventory"

    development_roster_sha256: str | None = None
    isolation_status = "not_verified_no_development_roster"
    if development_patient_ids is not None:
        development = sorted(
            {
                _identifier(value, "development patient ID")
                for value in development_patient_ids
            }
        )
        if not development:
            raise ValueError("development patient roster must not be empty")
        overlap = sorted(set(development).intersection(evaluation_patients))
        if overlap:
            raise ValueError("development/evaluation patient rosters overlap")
        development_roster_sha256 = _canonical_sha256(development)
        isolation_status = "verified_no_patient_overlap"

    metrics = _aggregate_metrics(validated_rows, tolerances)
    metrics["patient_macro"] = _patient_macro_metrics(validated_rows, tolerances)
    execution_metrics = _aggregate_execution_metrics(validated_rows)
    bootstrap = (
        None
        if bootstrap_replicates == 0
        else _patient_bootstrap(
            validated_rows,
            tolerances=tolerances,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    )
    limitations: list[str] = []
    if not operating_point_frozen_before_evaluation:
        limitations.append("operating_point_not_frozen_before_evaluation")
    if isolation_status != "verified_no_patient_overlap":
        limitations.append("development_evaluation_patient_isolation_not_verified")
    if evaluation_inventory_status != "verified_complete_expected_recording_inventory":
        limitations.append("complete_evaluation_recording_inventory_not_verified")
    if metrics["seizure_free_recording_count"] == 0:
        limitations.append("no_seizure_free_recordings_for_false_alarm_transport")
    if bootstrap is None:
        limitations.append("patient_bootstrap_confidence_intervals_not_computed")
    if execution_metrics["execution_receipt_missing_recording_count"] > 0:
        limitations.append("execution_receipts_missing_for_one_or_more_recordings")
    if execution_metrics["service_state_metrics"]["warm"]["recording_count"] == 0:
        limitations.append("no_warm_service_execution_receipts")

    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_BENCHMARK_SCHEMA_VERSION,
        "benchmark_receipt_id": "CONTINUOUS-BENCHMARK-PENDING",
        "method_id": CONTINUOUS_BENCHMARK_METHOD_ID,
        "provider_id": provider_id,
        "operating_point_id": operating_point_id,
        "evaluation_split": evaluation_split,
        "input_rows_sha256": _canonical_sha256(validated_rows),
        "reference_inventory_sha256": _canonical_sha256(
            _reference_inventory(validated_rows)
        ),
        "prediction_inventory_sha256": _canonical_sha256(
            _prediction_inventory(validated_rows)
        ),
        "recording_roster_sha256": _canonical_sha256(observed_recordings),
        "expected_recording_roster_sha256": expected_recording_roster_sha256,
        "evaluation_inventory_status": evaluation_inventory_status,
        "patient_count": len(evaluation_patients),
        "evaluation_patient_roster_sha256": _canonical_sha256(evaluation_patients),
        "development_patient_roster_sha256": development_roster_sha256,
        "patient_isolation_status": isolation_status,
        "operating_point_frozen_before_evaluation": (
            operating_point_frozen_before_evaluation
        ),
        "source_eval_admission_binding": admission_binding,
        "reference_labels_used_for_scoring_only_after_inference": True,
        "reference_labels_available_to_provider": False,
        "tolerances_seconds": list(tolerances),
        "metric_definitions": {
            "event_match": "ordered_one_to_one_positive_interval_overlap",
            "event_sensitivity": "matched_reference_events/all_reference_events",
            "event_precision": "matched_predicted_intervals/all_predicted_intervals",
            "event_f1": "harmonic_mean_of_event_precision_and_event_sensitivity",
            "alarm_false_alarms_per_recording_hour": (
                "unmatched_predicted_intervals/total_recording_hours"
            ),
            "alarm_false_alarms_per_24h": (
                "24_times_unmatched_predicted_intervals/total_recording_hours"
            ),
            "alarm_decomposition": (
                "every_unmatched_alarm_is_partitioned_into_background_only_or_"
                "reference_overlapping_duplicate_or_fragment; background_only_"
                "rate_uses_background_hours; unmatched_per_background_hour_is_"
                "retained_under_an_explicit_non_background_only_name"
            ),
            "time_in_warning": (
                "union_duration_of_nonoverlapping_predicted_intervals/recording_"
                "duration; background_time_in_warning_uses_predicted_duration_"
                "outside_the_reference_interval_union/background_duration"
            ),
            "onset_latency_seconds": (
                "predicted_start-reference_start_for_matched_events; absolute_error_"
                "summaries_are_matched_only_and_coverage_is_reported_separately"
            ),
            "onset_absolute_hit_rate": (
                "matched_events_with_absolute_onset_latency_within_tolerance/"
                "all_reference_events"
            ),
            "event_iou": "intersection_over_union_for_one_to_one_matches",
            "reference_duration_strata": (
                "reference_event_duration_lt_8s_8_to_30s_inclusive_and_gt_30s; "
                "unmatched_reference_events_remain_in_each_denominator"
            ),
            "patient_macro": (
                "arithmetic_mean_of_patient_level_metrics_over_patients_with_"
                "an_evaluable_denominator; evaluable_patient_counts_are_reported"
            ),
            "typed_boundary_f1": (
                "onset_compared_to_onset_and_offset_compared_to_offset; "
                "all_predicted_and_reference_boundaries_are_denominators"
            ),
            "execution_metrics": (
                "per_record_edf_io_preprocessing_inference_postprocessing_and_"
                "total_wall_receipts; RTF=seconds_wall/seconds_EEG; cold_and_warm_"
                "service_states_are_not_pooled_silently"
            ),
        },
        "metrics": metrics,
        "execution_metrics": execution_metrics,
        "patient_bootstrap": bootstrap,
        "qualification_limitations": limitations,
        "production_promotion_status": (
            "metrics_only_not_a_production_promotion_receipt"
        ),
        "sota_claim_authorized": False,
    }
    body["benchmark_receipt_id"] = "CONTBE-" + _canonical_sha256(body)[:24]
    return validate_continuous_detection_benchmark_receipt(body)


def validate_continuous_detection_benchmark_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate a v3 metrics receipt and its comparable reference inventory."""

    if type(payload) is not dict:
        raise TypeError("continuous benchmark receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "benchmark_receipt_id",
        "method_id",
        "provider_id",
        "operating_point_id",
        "evaluation_split",
        "input_rows_sha256",
        "reference_inventory_sha256",
        "prediction_inventory_sha256",
        "recording_roster_sha256",
        "expected_recording_roster_sha256",
        "evaluation_inventory_status",
        "patient_count",
        "evaluation_patient_roster_sha256",
        "development_patient_roster_sha256",
        "patient_isolation_status",
        "operating_point_frozen_before_evaluation",
        "source_eval_admission_binding",
        "reference_labels_used_for_scoring_only_after_inference",
        "reference_labels_available_to_provider",
        "tolerances_seconds",
        "metric_definitions",
        "metrics",
        "execution_metrics",
        "patient_bootstrap",
        "qualification_limitations",
        "production_promotion_status",
        "sota_claim_authorized",
    }
    if set(data) != required:
        raise ValueError("continuous benchmark receipt has missing or unknown fields")
    if data["schema_version"] != CONTINUOUS_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("continuous benchmark schema drifted")
    if data["method_id"] != CONTINUOUS_BENCHMARK_METHOD_ID:
        raise ValueError("continuous benchmark method drifted")
    _identifier(data["provider_id"], "provider_id")
    _identifier(data["operating_point_id"], "operating_point_id")
    evaluation_split = _identifier(data["evaluation_split"], "evaluation_split")
    if type(data["operating_point_frozen_before_evaluation"]) is not bool:
        raise TypeError("continuous benchmark frozen flag must be boolean")
    admission_binding = data["source_eval_admission_binding"]
    if evaluation_split == "source_eval":
        if (
            type(admission_binding) is not dict
            or set(admission_binding) != _SOURCE_EVAL_ADMISSION_BINDING_FIELDS
        ):
            raise ValueError(
                "source_eval benchmark lacks an exact source-eval admission binding"
            )
        for name in (
            "admission_id",
            "provider_id",
            "operating_point_id",
            "calibration_receipt_id",
            "decoder_method_id",
        ):
            _identifier(admission_binding[name], name)
        for name in _SOURCE_EVAL_ADMISSION_BINDING_FIELDS.difference(
            {
                "admission_id",
                "provider_id",
                "operating_point_id",
                "calibration_receipt_id",
                "decoder_method_id",
            }
        ):
            _sha256(admission_binding[name], name)
        if admission_binding["decoder_method_id"] != CONTINUOUS_DETECTION_METHOD_ID:
            raise ValueError("source_eval benchmark decoder method disagrees with admission")
        if (
            admission_binding["provider_id"] != data["provider_id"]
            or admission_binding["operating_point_id"]
            != data["operating_point_id"]
        ):
            raise ValueError(
                "source_eval benchmark provider/operating point disagrees with admission"
            )
        if (
            admission_binding["source_eval_recording_roster_sha256"]
            != data["recording_roster_sha256"]
            or admission_binding["source_eval_recording_roster_sha256"]
            != data["expected_recording_roster_sha256"]
            or admission_binding["source_eval_patient_roster_sha256"]
            != data["evaluation_patient_roster_sha256"]
            or admission_binding["source_dev_patient_roster_sha256"]
            != data["development_patient_roster_sha256"]
        ):
            raise ValueError("source_eval benchmark roster disagrees with admission")
        if data["operating_point_frozen_before_evaluation"] is not True:
            raise ValueError("source_eval benchmark operating point is not frozen")
    elif admission_binding is not None:
        raise ValueError(
            "non-source_eval benchmark cannot carry a source-eval admission binding"
        )
    for name in (
        "input_rows_sha256",
        "reference_inventory_sha256",
        "prediction_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
    ):
        _sha256(data[name], name)
    _sha256(
        data["expected_recording_roster_sha256"],
        "expected_recording_roster_sha256",
        nullable=True,
    )
    _sha256(
        data["development_patient_roster_sha256"],
        "development_patient_roster_sha256",
        nullable=True,
    )
    if data["evaluation_inventory_status"] not in {
        "not_verified_no_expected_recording_roster",
        "verified_complete_expected_recording_inventory",
    }:
        raise ValueError("continuous benchmark inventory status is invalid")
    inventory_verified = (
        data["evaluation_inventory_status"]
        == "verified_complete_expected_recording_inventory"
    )
    if inventory_verified != (data["expected_recording_roster_sha256"] is not None):
        raise ValueError("continuous benchmark inventory proof is inconsistent")
    if data["patient_isolation_status"] not in {
        "not_verified_no_development_roster",
        "verified_no_patient_overlap",
    }:
        raise ValueError("continuous benchmark patient isolation status is invalid")
    isolation_verified = (
        data["patient_isolation_status"] == "verified_no_patient_overlap"
    )
    if isolation_verified != (data["development_patient_roster_sha256"] is not None):
        raise ValueError("continuous benchmark patient-isolation proof is inconsistent")
    if evaluation_split == "source_eval" and not (
        inventory_verified and isolation_verified
    ):
        raise ValueError(
            "source_eval benchmark lacks admitted inventory/isolation proof"
        )
    if data["reference_labels_used_for_scoring_only_after_inference"] is not True:
        raise ValueError("continuous benchmark reference timing firewall was weakened")
    if data["reference_labels_available_to_provider"] is not False:
        raise ValueError("continuous benchmark leaked references to the provider")
    if data["production_promotion_status"] != (
        "metrics_only_not_a_production_promotion_receipt"
    ):
        raise ValueError("continuous benchmark cannot self-promote a detector")
    if data["sota_claim_authorized"] is not False:
        raise ValueError("continuous benchmark cannot self-authorize a SOTA claim")
    if not isinstance(data["qualification_limitations"], list):
        raise TypeError("continuous benchmark limitations must be an array")
    inventory_limitation = "complete_evaluation_recording_inventory_not_verified"
    if inventory_verified == (
        inventory_limitation in data["qualification_limitations"]
    ):
        raise ValueError("continuous benchmark inventory limitation is inconsistent")
    digest = deepcopy(data)
    digest["benchmark_receipt_id"] = "CONTINUOUS-BENCHMARK-PENDING"
    expected_id = "CONTBE-" + _canonical_sha256(digest)[:24]
    if data["benchmark_receipt_id"] != expected_id:
        raise ValueError("continuous benchmark receipt is not content-bound")
    return data


__all__ = [
    "CONTINUOUS_BENCHMARK_METHOD_ID",
    "CONTINUOUS_BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_TOLERANCES_SECONDS",
    "REFERENCE_DURATION_STRATA_SECONDS",
    "aggregate_continuous_detection_metrics",
    "evaluate_patient_level_continuous_detection",
    "validate_continuous_detection_benchmark_receipt",
    "validate_continuous_benchmark_rows",
]
