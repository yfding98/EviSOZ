"""Reference-gated dual operating-point diagnostics for long-EEG detectors.

This module is deliberately provider neutral and performs no filesystem I/O.
It has two explicit stages:

1. :func:`freeze_detector_prediction_inventory_v1` validates and content-binds
   a complete, reference-free record-by-policy prediction inventory.
2. An upper layer may then join source-development references to the exact
   frozen prediction rows and call :func:`score_detector_dual_op_v1`.

``OP-BALANCED`` reports strict-overlap event sensitivity at 1/3/6/12
all-unmatched alarms per 24 processed EEG hours and a sensitivity--FA partial
AUC. ``OP-NAVIGATION`` reports reference-denominator recall under ranked
candidate and queried-EEG-second budgets, together with onset-envelope recall
and anchor hit rates at 1/3/5/10 seconds.

Technical failures and unmodelled partial tails remain misses in every
reference denominator.  They are excluded from the false-alarm and query-rate
opportunity denominator so that missing computation cannot dilute burden.
Records are combined within patient before patient-macro aggregation.

The returned object is a calibration diagnostic only.  It cannot promote a
provider, authorize clinical use, or select a descriptive research primary.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .continuous_detection_benchmark import _ordered_event_matching


DETECTOR_DUAL_OP_PREDICTION_SCHEMA_VERSION = (
    "detector_dual_op_reference_free_prediction_inventory_v1"
)
DETECTOR_DUAL_OP_PREDICTION_METHOD_ID = (
    "complete_record_policy_cross_product_reference_free_freeze_v1"
)
DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION = (
    "detector_dual_operating_point_calibration_diagnostic_v1"
)
DETECTOR_DUAL_OP_DIAGNOSTIC_METHOD_ID = (
    "patient_cluster_balanced_and_navigation_curve_scorer_v1"
)

DEFAULT_FALSE_ALARM_BUDGETS_PER_24H = (1.0, 3.0, 6.0, 12.0)
DEFAULT_CANDIDATE_BUDGETS_PER_HOUR = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_QUERY_BUDGETS_SECONDS_PER_HOUR = (60.0, 120.0, 300.0, 600.0)
DEFAULT_ONSET_TOLERANCES_SECONDS = (1.0, 3.0, 5.0, 10.0)

PROCESSING_STATUSES = frozenset(
    {"completed", "partial_coverage", "technical_failure"}
)

_PREDICTION_ROW_FIELDS = {
    "provider_id",
    "patient_id",
    "recording_id",
    "split",
    "duration_seconds",
    "policy_id",
    "processing_status",
    "modeled_duration_seconds",
    "failure_code",
    "candidates",
}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "start_seconds",
    "stop_seconds",
    "anchor_seconds",
    "ranking_score",
    "search_envelope_start_seconds",
    "search_envelope_stop_seconds",
    "query_start_seconds",
    "query_stop_seconds",
}
_REFERENCE_ROW_FIELDS = {
    "patient_id",
    "recording_id",
    "split",
    "duration_seconds",
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
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
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
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _normalized_unique_identifiers(
    values: Iterable[str], context: str
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be an iterable of identifiers")
    rows = [_identifier(value, context) for value in values]
    if not rows:
        raise ValueError(f"{context} must not be empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{context} contains duplicates")
    return sorted(rows)


def _strict_positive_grid(values: Sequence[float], context: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence")
    result = tuple(_finite(value, context, minimum=0.0) for value in values)
    if not result or any(value <= 0.0 for value in result):
        raise ValueError(f"{context} values must be positive")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{context} must be strictly increasing and unique")
    return result


def _validate_interval_events(
    value: object,
    *,
    duration_seconds: float,
    context: str,
) -> list[dict[str, float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result: list[dict[str, float]] = []
    previous_stop = 0.0
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"start_seconds", "stop_seconds"}:
            raise ValueError(f"{context}[{index}] fields drifted")
        start = _finite(raw["start_seconds"], f"{context}[{index}] start")
        stop = _finite(raw["stop_seconds"], f"{context}[{index}] stop")
        if start < 0.0 or stop <= start or stop > duration_seconds + 1e-9:
            raise ValueError(f"{context}[{index}] lies outside the recording")
        if index and start < previous_stop - 1e-9:
            raise ValueError(f"{context} must be sorted and non-overlapping")
        result.append({"start_seconds": start, "stop_seconds": stop})
        previous_stop = stop
    return result


def _validate_candidates(
    value: object,
    *,
    modeled_duration_seconds: float,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    result: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    previous_stop = 0.0
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != _CANDIDATE_FIELDS:
            raise ValueError(f"{context}[{index}] fields drifted")
        candidate_id = _identifier(raw["candidate_id"], "candidate ID")
        if candidate_id in candidate_ids:
            raise ValueError(f"{context} candidate IDs must be unique")
        candidate_ids.add(candidate_id)
        start = _finite(raw["start_seconds"], "candidate start")
        stop = _finite(raw["stop_seconds"], "candidate stop")
        anchor = _finite(raw["anchor_seconds"], "candidate anchor")
        score = _finite(raw["ranking_score"], "candidate ranking score")
        envelope_start = _finite(
            raw["search_envelope_start_seconds"], "search envelope start"
        )
        envelope_stop = _finite(
            raw["search_envelope_stop_seconds"], "search envelope stop"
        )
        query_start = _finite(raw["query_start_seconds"], "query start")
        query_stop = _finite(raw["query_stop_seconds"], "query stop")
        if (
            start < 0.0
            or stop <= start
            or stop > modeled_duration_seconds + 1e-9
            or (index and start < previous_stop - 1e-9)
        ):
            raise ValueError(
                f"{context} candidate events must be chronological, "
                "non-overlapping and inside modeled support"
            )
        if not start - 1e-9 <= anchor <= stop + 1e-9:
            raise ValueError("candidate anchor must lie inside its event interval")
        if not (
            0.0 <= query_start + 1e-9
            and query_start <= envelope_start + 1e-9
            and envelope_start <= anchor + 1e-9
            and anchor <= envelope_stop + 1e-9
            and envelope_stop <= query_stop + 1e-9
            and query_stop <= modeled_duration_seconds + 1e-9
            and query_stop > query_start
        ):
            raise ValueError(
                "query support must contain a non-reversed onset search envelope "
                "and its anchor"
            )
        result.append(
            {
                "candidate_id": candidate_id,
                "start_seconds": start,
                "stop_seconds": stop,
                "anchor_seconds": anchor,
                "ranking_score": score,
                "search_envelope_start_seconds": envelope_start,
                "search_envelope_stop_seconds": envelope_stop,
                "query_start_seconds": query_start,
                "query_stop_seconds": query_stop,
            }
        )
        previous_stop = stop
    return result


def _validate_prediction_row(value: object, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PREDICTION_ROW_FIELDS:
        raise ValueError(f"prediction row {index} has missing or unknown fields")
    row = deepcopy(value)
    for field in ("provider_id", "patient_id", "recording_id", "policy_id"):
        row[field] = _identifier(row[field], f"prediction row {field}")
    if row["split"] != "source_dev":
        raise ValueError("dual-OP calibration prediction rows accept source_dev only")
    duration = _finite(row["duration_seconds"], "recording duration", minimum=0.0)
    if duration <= 0.0:
        raise ValueError("recording duration must be positive")
    modeled = _finite(
        row["modeled_duration_seconds"], "modeled duration", minimum=0.0
    )
    if modeled > duration + 1e-9:
        raise ValueError("modeled duration exceeds recording duration")
    status = row["processing_status"]
    if status not in PROCESSING_STATUSES:
        raise ValueError("prediction processing status is invalid")
    if status == "technical_failure":
        if modeled != 0.0 or row["candidates"] != []:
            raise ValueError(
                "technical failure must have zero modeled opportunity and no candidates"
            )
        row["failure_code"] = _identifier(row["failure_code"], "failure code")
    else:
        if row["failure_code"] is not None:
            raise ValueError("non-failed prediction row carries a failure code")
        if status == "completed" and not math.isclose(
            modeled, duration, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("completed prediction row lacks full modeled coverage")
        if status == "partial_coverage" and not 0.0 < modeled < duration - 1e-9:
            raise ValueError("partial prediction row must have partial opportunity")
    row["duration_seconds"] = duration
    row["modeled_duration_seconds"] = modeled
    row["candidates"] = _validate_candidates(
        row["candidates"],
        modeled_duration_seconds=modeled,
        context=f"prediction row {index} candidates",
    )
    return row


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider_id: str,
    expected_recording_ids: Sequence[str],
    expected_policy_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("prediction rows must be a non-empty sequence")
    normalized = [_validate_prediction_row(row, index) for index, row in enumerate(rows)]
    if any(row["provider_id"] != provider_id for row in normalized):
        raise ValueError("prediction inventory mixes providers")
    observed_pairs = [
        (str(row["recording_id"]), str(row["policy_id"])) for row in normalized
    ]
    if len(set(observed_pairs)) != len(observed_pairs):
        raise ValueError("prediction record-policy pairs must be unique")
    expected_pairs = {
        (recording_id, policy_id)
        for recording_id in expected_recording_ids
        for policy_id in expected_policy_ids
    }
    if set(observed_pairs) != expected_pairs:
        raise ValueError(
            "prediction rows do not close the expected recording-policy cross product"
        )
    metadata: dict[str, tuple[Any, ...]] = {}
    patient_splits: dict[str, str] = {}
    for row in normalized:
        recording_id = str(row["recording_id"])
        record_metadata = (
            row["patient_id"],
            row["split"],
            row["duration_seconds"],
            row["processing_status"],
            row["modeled_duration_seconds"],
            row["failure_code"],
        )
        prior = metadata.setdefault(recording_id, record_metadata)
        if prior != record_metadata:
            raise ValueError(
                "record processing identity or terminal outcome differs across policies"
            )
        prior_split = patient_splits.setdefault(str(row["patient_id"]), row["split"])
        if prior_split != row["split"]:
            raise ValueError("one patient occurs in multiple splits")
    normalized.sort(key=lambda row: (row["recording_id"], row["policy_id"]))
    return normalized


def _prediction_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[str],
) -> dict[str, Any]:
    first_policy = policy_ids[0]
    records = [row for row in rows if row["policy_id"] == first_policy]
    planned_seconds = sum(float(row["duration_seconds"]) for row in records)
    processed_seconds = sum(
        float(row["modeled_duration_seconds"]) for row in records
    )
    failures = [row for row in records if row["processing_status"] == "technical_failure"]
    partial = [row for row in records if row["processing_status"] == "partial_coverage"]
    technical_failure_seconds = sum(float(row["duration_seconds"]) for row in failures)
    partial_planned_seconds = sum(float(row["duration_seconds"]) for row in partial)
    partial_modeled_seconds = sum(
        float(row["modeled_duration_seconds"]) for row in partial
    )
    completed = [row for row in records if row["processing_status"] == "completed"]
    return {
        "recording_count": len(records),
        "patient_count": len({str(row["patient_id"]) for row in records}),
        "policy_count": len(policy_ids),
        "planned_total_hours": planned_seconds / 3600.0,
        "processed_evaluable_hours": processed_seconds / 3600.0,
        "processed_evaluable_hour_coverage": (
            processed_seconds / planned_seconds if planned_seconds > 0.0 else None
        ),
        "completed_recording_count": len(completed),
        "partial_coverage_recording_count": len(partial),
        "partial_coverage_planned_hours": partial_planned_seconds / 3600.0,
        "partial_coverage_modeled_hours": partial_modeled_seconds / 3600.0,
        "partial_coverage_unmodeled_hours": (
            partial_planned_seconds - partial_modeled_seconds
        )
        / 3600.0,
        "technical_failure_count": len(failures),
        "technical_failure_hours": technical_failure_seconds / 3600.0,
        "technical_failure_recording_fraction": (
            len(failures) / len(records) if records else None
        ),
        "technical_failure_planned_hour_fraction": (
            technical_failure_seconds / planned_seconds
            if planned_seconds > 0.0
            else None
        ),
        "technical_failure_not_counted_as_zero_alarm": True,
        "rate_denominator_semantics": (
            "processed_evaluable_modeled_eeg_hours_excluding_technical_failure_"
            "and_unmodeled_partial_tail"
        ),
        "reference_denominator_semantics": (
            "all_joined_reference_events_including_technical_failure_and_"
            "unmodeled_partial_tail"
        ),
        "qualification_eligible": not failures and not partial,
    }


def freeze_detector_prediction_inventory_v1(
    *,
    provider_id: str,
    rows: Sequence[Mapping[str, Any]],
    expected_recording_ids: Iterable[str],
    expected_policy_ids: Iterable[str],
) -> dict[str, Any]:
    """Freeze a complete reference-free prediction inventory in memory."""

    provider = _identifier(provider_id, "provider ID")
    recording_ids = _normalized_unique_identifiers(
        expected_recording_ids, "expected recording ID"
    )
    policy_ids = _normalized_unique_identifiers(expected_policy_ids, "expected policy ID")
    normalized = _validate_prediction_rows(
        rows,
        provider_id=provider,
        expected_recording_ids=recording_ids,
        expected_policy_ids=policy_ids,
    )
    body: dict[str, Any] = {
        "schema_version": DETECTOR_DUAL_OP_PREDICTION_SCHEMA_VERSION,
        "method_id": DETECTOR_DUAL_OP_PREDICTION_METHOD_ID,
        "provider_id": provider,
        "split": "source_dev",
        "expected_recording_ids": recording_ids,
        "expected_policy_ids": policy_ids,
        "prediction_rows": normalized,
        "prediction_row_roster_sha256": _canonical_sha256(normalized),
        "coverage_accounting": _prediction_coverage(
            normalized, policy_ids=policy_ids
        ),
        "reference_fields_present": False,
        "reference_accessed": False,
        "provider_promotion_authorized": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_detector_prediction_inventory_v1(body)


def validate_detector_prediction_inventory_v1(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "provider_id",
        "split",
        "expected_recording_ids",
        "expected_policy_ids",
        "prediction_rows",
        "prediction_row_roster_sha256",
        "coverage_accounting",
        "reference_fields_present",
        "reference_accessed",
        "provider_promotion_authorized",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("dual-OP prediction inventory fields drifted")
    data = deepcopy(value)
    if (
        data["schema_version"] != DETECTOR_DUAL_OP_PREDICTION_SCHEMA_VERSION
        or data["method_id"] != DETECTOR_DUAL_OP_PREDICTION_METHOD_ID
        or data["split"] != "source_dev"
        or data["reference_fields_present"] is not False
        or data["reference_accessed"] is not False
        or data["provider_promotion_authorized"] is not False
    ):
        raise ValueError("dual-OP prediction inventory identity or firewall drifted")
    provider = _identifier(data["provider_id"], "provider ID")
    recording_ids = _normalized_unique_identifiers(
        data["expected_recording_ids"], "expected recording ID"
    )
    policy_ids = _normalized_unique_identifiers(
        data["expected_policy_ids"], "expected policy ID"
    )
    if recording_ids != data["expected_recording_ids"] or policy_ids != data[
        "expected_policy_ids"
    ]:
        raise ValueError("dual-OP expected rosters must be canonically sorted")
    rows = _validate_prediction_rows(
        data["prediction_rows"],
        provider_id=provider,
        expected_recording_ids=recording_ids,
        expected_policy_ids=policy_ids,
    )
    if rows != data["prediction_rows"]:
        raise ValueError("dual-OP prediction rows must be canonically normalized")
    if data["prediction_row_roster_sha256"] != _canonical_sha256(rows):
        raise ValueError("dual-OP prediction row roster hash drifted")
    expected_coverage = _prediction_coverage(rows, policy_ids=policy_ids)
    if data["coverage_accounting"] != expected_coverage:
        raise ValueError("dual-OP prediction coverage accounting drifted")
    receipt = _sha256(data["receipt_sha256"], "prediction inventory receipt")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt != _canonical_sha256(digest):
        raise ValueError("dual-OP prediction inventory receipt hash drifted")
    return data


def _validate_reference_row(value: object, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _REFERENCE_ROW_FIELDS:
        raise ValueError(f"reference row {index} has missing or unknown fields")
    row = deepcopy(value)
    row["patient_id"] = _identifier(row["patient_id"], "reference patient ID")
    row["recording_id"] = _identifier(
        row["recording_id"], "reference recording ID"
    )
    if row["split"] != "source_dev":
        raise ValueError("dual-OP reference rows accept source_dev only")
    duration = _finite(row["duration_seconds"], "reference duration", minimum=0.0)
    if duration <= 0.0:
        raise ValueError("reference duration must be positive")
    row["duration_seconds"] = duration
    row["reference_events"] = _validate_interval_events(
        row["reference_events"],
        duration_seconds=duration,
        context=f"reference row {index} events",
    )
    return row


def validate_detector_dual_op_joined_rows_v1(
    *,
    frozen_prediction_inventory: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate an upper-layer reference join against the exact frozen rows."""

    frozen = validate_detector_prediction_inventory_v1(
        dict(frozen_prediction_inventory)
    )
    if (
        not isinstance(joined_rows, Sequence)
        or isinstance(joined_rows, (str, bytes))
        or not joined_rows
    ):
        raise TypeError("joined dual-OP rows must be a non-empty sequence")
    frozen_by_pair = {
        (str(row["recording_id"]), str(row["policy_id"])): row
        for row in frozen["prediction_rows"]
    }
    normalized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    reference_by_recording: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(joined_rows):
        if type(raw) is not dict or set(raw) != {"prediction_row", "reference_row"}:
            raise ValueError(f"joined row {index} fields drifted")
        prediction = _validate_prediction_row(raw["prediction_row"], index)
        pair = (str(prediction["recording_id"]), str(prediction["policy_id"]))
        if pair in seen_pairs or pair not in frozen_by_pair:
            raise ValueError("joined prediction pair is duplicate or outside frozen roster")
        seen_pairs.add(pair)
        if prediction != frozen_by_pair[pair]:
            raise ValueError("joined prediction row differs from frozen inventory")
        # The reference object is not inspected until the prediction receipt and
        # exact prediction row have both replayed successfully.
        reference = _validate_reference_row(raw["reference_row"], index)
        if (
            reference["recording_id"] != prediction["recording_id"]
            or reference["patient_id"] != prediction["patient_id"]
            or reference["split"] != prediction["split"]
            or not math.isclose(
                reference["duration_seconds"],
                prediction["duration_seconds"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("joined prediction/reference record identity drifted")
        recording_id = str(reference["recording_id"])
        prior = reference_by_recording.setdefault(recording_id, reference)
        if prior != reference:
            raise ValueError("one recording carries different references across policies")
        normalized.append(
            {"prediction_row": prediction, "reference_row": reference}
        )
    if seen_pairs != set(frozen_by_pair):
        raise ValueError("joined rows dropped a frozen prediction row")
    normalized.sort(
        key=lambda row: (
            row["prediction_row"]["recording_id"],
            row["prediction_row"]["policy_id"],
        )
    )
    return normalized


def _union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(start), float(stop)) for start, stop in intervals)
    total = 0.0
    active_start, active_stop = ordered[0]
    for start, stop in ordered[1:]:
        if start <= active_stop + 1e-12:
            active_stop = max(active_stop, stop)
        else:
            total += active_stop - active_start
            active_start, active_stop = start, stop
    return total + active_stop - active_start


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0.0 else float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _ordered_onset_envelope_matching(
    references: Sequence[Mapping[str, float]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """One-to-one chronological matching when onset lies in search envelope."""

    n_reference = len(references)
    n_candidate = len(candidates)
    # Maximize count, then minimize absolute anchor error.
    scores: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(n_candidate + 1)]
        for _ in range(n_reference + 1)
    ]
    parents: list[list[tuple[int, int, str] | None]] = [
        [None for _ in range(n_candidate + 1)]
        for _ in range(n_reference + 1)
    ]
    for reference_index in range(1, n_reference + 1):
        parents[reference_index][0] = (
            reference_index - 1,
            0,
            "skip_reference",
        )
    for candidate_index in range(1, n_candidate + 1):
        parents[0][candidate_index] = (0, candidate_index - 1, "skip_candidate")
    priority = {"skip_reference": 0, "skip_candidate": 1, "match": 2}
    for reference_index in range(1, n_reference + 1):
        onset = float(references[reference_index - 1]["start_seconds"])
        for candidate_index in range(1, n_candidate + 1):
            candidate = candidates[candidate_index - 1]
            options: list[
                tuple[tuple[int, float], tuple[int, int, str]]
            ] = [
                (
                    scores[reference_index - 1][candidate_index],
                    (reference_index - 1, candidate_index, "skip_reference"),
                ),
                (
                    scores[reference_index][candidate_index - 1],
                    (reference_index, candidate_index - 1, "skip_candidate"),
                ),
            ]
            if (
                float(candidate["search_envelope_start_seconds"]) - 1e-12
                <= onset
                <= float(candidate["search_envelope_stop_seconds"]) + 1e-12
            ):
                previous = scores[reference_index - 1][candidate_index - 1]
                options.append(
                    (
                        (
                            previous[0] + 1,
                            previous[1]
                            - abs(float(candidate["anchor_seconds"]) - onset),
                        ),
                        (reference_index - 1, candidate_index - 1, "match"),
                    )
                )
            best_score, best_parent = max(
                options,
                key=lambda item: (item[0], priority[item[1][2]]),
            )
            scores[reference_index][candidate_index] = best_score
            parents[reference_index][candidate_index] = best_parent
    result: list[tuple[int, int]] = []
    reference_index, candidate_index = n_reference, n_candidate
    while reference_index or candidate_index:
        parent = parents[reference_index][candidate_index]
        if parent is None:
            raise RuntimeError("onset-envelope matching backtrace is incomplete")
        prior_reference, prior_candidate, action = parent
        if action == "match":
            result.append((reference_index - 1, candidate_index - 1))
        reference_index, candidate_index = prior_reference, prior_candidate
    result.reverse()
    return result


def _policy_coverage_with_references(
    policy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _prediction_coverage(
        [row["prediction_row"] for row in policy_rows],
        policy_ids=[str(policy_rows[0]["prediction_row"]["policy_id"])],
    )
    complete_zero = 0
    partial_zero = 0
    seizure_free = 0
    seizure_bearing = 0
    reference_by_status = {status: 0 for status in sorted(PROCESSING_STATUSES)}
    for joined in policy_rows:
        prediction = joined["prediction_row"]
        references = joined["reference_row"]["reference_events"]
        status = str(prediction["processing_status"])
        reference_by_status[status] += len(references)
        seizure_free += int(not references)
        seizure_bearing += int(bool(references))
        if status == "completed" and not prediction["candidates"]:
            complete_zero += 1
        if status == "partial_coverage" and not prediction["candidates"]:
            partial_zero += 1
    base.update(
        {
            "completed_zero_candidate_recording_count": complete_zero,
            "partial_zero_candidate_recording_count": partial_zero,
            "seizure_free_recording_count": seizure_free,
            "seizure_bearing_recording_count": seizure_bearing,
            "reference_event_counts_by_processing_status": reference_by_status,
            "technical_failure_not_counted_as_zero_candidate": True,
        }
    )
    return base


def _evaluate_selected_candidates(
    policy_rows: Sequence[Mapping[str, Any]],
    selected_ids_by_recording: Mapping[str, set[str]],
    *,
    onset_tolerances_seconds: Sequence[float],
) -> dict[str, Any]:
    reference_count = 0
    strict_match_count = 0
    envelope_match_count = 0
    candidate_count = 0
    false_alarm_count = 0
    warning_seconds = 0.0
    query_seconds = 0.0
    seizure_bearing_records = 0
    strict_hit_records = 0
    envelope_hit_records = 0
    onset_hits = {float(value): 0 for value in onset_tolerances_seconds}
    absolute_anchor_errors: list[float] = []
    patient_values: dict[str, dict[str, Any]] = {}
    zero_selected_nonfailure_records = 0

    for joined in policy_rows:
        prediction = joined["prediction_row"]
        reference = joined["reference_row"]
        recording_id = str(prediction["recording_id"])
        selected_ids = selected_ids_by_recording.get(recording_id, set())
        candidates = [
            candidate
            for candidate in prediction["candidates"]
            if candidate["candidate_id"] in selected_ids
        ]
        candidates.sort(key=lambda row: (row["start_seconds"], row["candidate_id"]))
        references = reference["reference_events"]
        predicted_intervals = [
            {
                "start_seconds": float(candidate["start_seconds"]),
                "stop_seconds": float(candidate["stop_seconds"]),
            }
            for candidate in candidates
        ]
        strict_matches = _ordered_event_matching(references, predicted_intervals)
        envelope_matches = _ordered_onset_envelope_matching(references, candidates)
        matched_candidate_indices = {
            candidate_index for _, candidate_index, _ in strict_matches
        }
        reference_count += len(references)
        strict_match_count += len(strict_matches)
        envelope_match_count += len(envelope_matches)
        candidate_count += len(candidates)
        false_alarm_count += len(candidates) - len(strict_matches)
        record_warning = _union_seconds(
            [
                (float(candidate["start_seconds"]), float(candidate["stop_seconds"]))
                for candidate in candidates
            ]
        )
        record_query = _union_seconds(
            [
                (
                    float(candidate["query_start_seconds"]),
                    float(candidate["query_stop_seconds"]),
                )
                for candidate in candidates
            ]
        )
        warning_seconds += record_warning
        query_seconds += record_query
        if references:
            seizure_bearing_records += 1
            strict_hit_records += int(bool(strict_matches))
            envelope_hit_records += int(bool(envelope_matches))
        if (
            prediction["processing_status"] != "technical_failure"
            and not candidates
        ):
            zero_selected_nonfailure_records += 1

        patient = patient_values.setdefault(
            str(prediction["patient_id"]),
            {
                "reference_count": 0,
                "strict_match_count": 0,
                "envelope_match_count": 0,
                "false_alarm_count": 0,
                "candidate_count": 0,
                "modeled_seconds": 0.0,
                "query_seconds": 0.0,
                "warning_seconds": 0.0,
                "onset_hits": {
                    float(value): 0 for value in onset_tolerances_seconds
                },
            },
        )
        patient["reference_count"] += len(references)
        patient["strict_match_count"] += len(strict_matches)
        patient["envelope_match_count"] += len(envelope_matches)
        patient["false_alarm_count"] += len(candidates) - len(strict_matches)
        patient["candidate_count"] += len(candidates)
        patient["modeled_seconds"] += float(prediction["modeled_duration_seconds"])
        patient["query_seconds"] += record_query
        patient["warning_seconds"] += record_warning
        for reference_index, candidate_index, _ in strict_matches:
            error = abs(
                float(candidates[candidate_index]["anchor_seconds"])
                - float(references[reference_index]["start_seconds"])
            )
            absolute_anchor_errors.append(error)
            for tolerance in onset_tolerances_seconds:
                if error <= float(tolerance) + 1e-12:
                    onset_hits[float(tolerance)] += 1
                    patient["onset_hits"][float(tolerance)] += 1

    coverage = _policy_coverage_with_references(policy_rows)
    processed_seconds = float(coverage["processed_evaluable_hours"]) * 3600.0
    planned_seconds = float(coverage["planned_total_hours"]) * 3600.0

    sensitivity_values: list[float] = []
    envelope_values: list[float] = []
    false_alarm_rate_values: list[float] = []
    candidate_rate_values: list[float] = []
    query_rate_values: list[float] = []
    warning_fraction_values: list[float] = []
    patient_onset_values = {
        float(tolerance): [] for tolerance in onset_tolerances_seconds
    }
    for patient in patient_values.values():
        patient_reference_count = int(patient["reference_count"])
        if patient_reference_count > 0:
            sensitivity_values.append(
                float(patient["strict_match_count"]) / patient_reference_count
            )
            envelope_values.append(
                float(patient["envelope_match_count"]) / patient_reference_count
            )
            for tolerance in onset_tolerances_seconds:
                patient_onset_values[float(tolerance)].append(
                    float(patient["onset_hits"][float(tolerance)])
                    / patient_reference_count
                )
        patient_hours = float(patient["modeled_seconds"]) / 3600.0
        if patient_hours > 0.0:
            false_alarm_rate_values.append(
                24.0 * float(patient["false_alarm_count"]) / patient_hours
            )
            candidate_rate_values.append(
                float(patient["candidate_count"]) / patient_hours
            )
            query_rate_values.append(
                float(patient["query_seconds"]) / patient_hours
            )
            warning_fraction_values.append(
                float(patient["warning_seconds"])
                / float(patient["modeled_seconds"])
            )

    onset_hit_payload = {
        f"{float(tolerance):g}s": {
            "hit_count": onset_hits[float(tolerance)],
            "reference_event_denominator": reference_count,
            "rate": _safe_rate(onset_hits[float(tolerance)], reference_count),
        }
        for tolerance in onset_tolerances_seconds
    }
    patient_onset_payload = {
        f"{float(tolerance):g}s": {
            "patient_macro_rate": _mean(patient_onset_values[float(tolerance)]),
            "evaluable_patient_count": len(patient_onset_values[float(tolerance)]),
        }
        for tolerance in onset_tolerances_seconds
    }
    return {
        "recording_count": len(policy_rows),
        "patient_count": len(patient_values),
        "reference_event_count": reference_count,
        "candidate_count": candidate_count,
        "strict_matched_event_count": strict_match_count,
        "onset_search_envelope_matched_event_count": envelope_match_count,
        "false_alarm_count": false_alarm_count,
        "event_sensitivity": _safe_rate(strict_match_count, reference_count),
        "onset_search_envelope_recall": _safe_rate(
            envelope_match_count, reference_count
        ),
        "onset_absolute_hit_rate": onset_hit_payload,
        "anchor_absolute_error_matched_only_seconds": {
            "matched_denominator": len(absolute_anchor_errors),
            "mean": _mean(absolute_anchor_errors),
            "maximum": (
                max(absolute_anchor_errors) if absolute_anchor_errors else None
            ),
        },
        "record_level": {
            "seizure_bearing_recording_count": seizure_bearing_records,
            "strict_event_hit_recording_count": strict_hit_records,
            "strict_event_recall": _safe_rate(
                strict_hit_records, seizure_bearing_records
            ),
            "onset_envelope_hit_recording_count": envelope_hit_records,
            "onset_envelope_recall": _safe_rate(
                envelope_hit_records, seizure_bearing_records
            ),
            "zero_selected_candidate_nonfailure_recording_count": (
                zero_selected_nonfailure_records
            ),
            "technical_failure_not_counted_as_zero_candidate": True,
        },
        "burden": {
            "planned_total_hours": planned_seconds / 3600.0,
            "processed_evaluable_hours": processed_seconds / 3600.0,
            "false_alarm_rate_denominator_hours": processed_seconds / 3600.0,
            "query_rate_denominator_hours": processed_seconds / 3600.0,
            "candidate_rate_denominator_hours": processed_seconds / 3600.0,
            "all_unmatched_alarms_per_24_processed_evaluable_hours": (
                None
                if processed_seconds <= 0.0
                else 24.0 * false_alarm_count / (processed_seconds / 3600.0)
            ),
            "candidates_per_processed_evaluable_hour": (
                None
                if processed_seconds <= 0.0
                else candidate_count / (processed_seconds / 3600.0)
            ),
            "queried_eeg_seconds": query_seconds,
            "queried_eeg_seconds_per_processed_evaluable_hour": (
                None
                if processed_seconds <= 0.0
                else query_seconds / (processed_seconds / 3600.0)
            ),
            "time_in_warning_seconds": warning_seconds,
            "time_in_warning_fraction_of_processed_evaluable_eeg": (
                None if processed_seconds <= 0.0 else warning_seconds / processed_seconds
            ),
            "rate_denominator_semantics": coverage["rate_denominator_semantics"],
            "planned_hours_diluted_false_alarm_rate_diagnostic_not_for_budgeting": (
                None
                if planned_seconds <= 0.0
                else 24.0 * false_alarm_count / (planned_seconds / 3600.0)
            ),
            "technical_failures_are_not_zero_alarm_opportunities": True,
        },
        "patient_macro": {
            "event_sensitivity": _mean(sensitivity_values),
            "event_sensitivity_evaluable_patient_count": len(sensitivity_values),
            "onset_search_envelope_recall": _mean(envelope_values),
            "onset_search_envelope_evaluable_patient_count": len(envelope_values),
            "onset_absolute_hit_rate": patient_onset_payload,
            "all_unmatched_alarms_per_24_processed_evaluable_hours": _mean(
                false_alarm_rate_values
            ),
            "false_alarm_rate_evaluable_patient_count": len(
                false_alarm_rate_values
            ),
            "candidates_per_processed_evaluable_hour": _mean(
                candidate_rate_values
            ),
            "queried_eeg_seconds_per_processed_evaluable_hour": _mean(
                query_rate_values
            ),
            "time_in_warning_fraction_of_processed_evaluable_eeg": _mean(
                warning_fraction_values
            ),
            "aggregation_semantics": (
                "combine_all_recordings_within_patient_then_equal_weight_patients"
            ),
        },
        "coverage_accounting": coverage,
    }


def _all_candidate_ids_by_recording(
    policy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    return {
        str(row["prediction_row"]["recording_id"]): {
            str(candidate["candidate_id"])
            for candidate in row["prediction_row"]["candidates"]
        }
        for row in policy_rows
    }


def _ranked_candidates(
    policy_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for row in policy_rows:
        recording_id = str(row["prediction_row"]["recording_id"])
        for candidate in row["prediction_row"]["candidates"]:
            result.append((recording_id, candidate))
    result.sort(
        key=lambda item: (
            -float(item[1]["ranking_score"]),
            item[0],
            float(item[1]["anchor_seconds"]),
            str(item[1]["candidate_id"]),
        )
    )
    return result


def _candidate_budget_selection(
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    budget_per_hour: float,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    coverage = _policy_coverage_with_references(policy_rows)
    hours = float(coverage["processed_evaluable_hours"])
    cap = int(math.floor(float(budget_per_hour) * hours + 1e-12))
    ranked = _ranked_candidates(policy_rows)
    selected = ranked[:cap]
    by_recording: dict[str, set[str]] = {}
    for recording_id, candidate in selected:
        by_recording.setdefault(recording_id, set()).add(str(candidate["candidate_id"]))
    return by_recording, {
        "candidate_budget_per_processed_evaluable_hour": float(budget_per_hour),
        "processed_evaluable_hours": hours,
        "candidate_count_cap": cap,
        "available_candidate_count": len(ranked),
        "selected_candidate_count": len(selected),
        "ranking_semantics": (
            "global_reference_free_descending_score_then_stable_identity"
        ),
    }


def _insert_interval_into_union(
    intervals: Sequence[tuple[float, float]],
    interval: tuple[float, float],
) -> tuple[list[tuple[float, float]], float]:
    """Insert one interval and return the disjoint union plus added seconds."""

    start, stop = interval
    old_seconds = sum(right - left for left, right in intervals)
    result: list[tuple[float, float]] = []
    inserted = False
    for left, right in intervals:
        if right < start - 1e-12:
            result.append((left, right))
        elif stop < left - 1e-12:
            if not inserted:
                result.append((start, stop))
                inserted = True
            result.append((left, right))
        else:
            start = min(start, left)
            stop = max(stop, right)
    if not inserted:
        result.append((start, stop))
    new_seconds = sum(right - left for left, right in result)
    return result, max(0.0, new_seconds - old_seconds)


def _query_budget_selection(
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    budget_seconds_per_hour: float,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    coverage = _policy_coverage_with_references(policy_rows)
    hours = float(coverage["processed_evaluable_hours"])
    allowance = float(budget_seconds_per_hour) * hours
    ranked = _ranked_candidates(policy_rows)
    selected: list[tuple[str, Mapping[str, Any]]] = []
    query_union_by_recording: dict[str, list[tuple[float, float]]] = {}
    selected_query_seconds = 0.0
    for item in ranked:
        recording_id, candidate = item
        updated_union, incremental_seconds = _insert_interval_into_union(
            query_union_by_recording.get(recording_id, []),
            (
                float(candidate["query_start_seconds"]),
                float(candidate["query_stop_seconds"]),
            ),
        )
        if selected_query_seconds + incremental_seconds <= allowance + 1e-9:
            selected.append(item)
            query_union_by_recording[recording_id] = updated_union
            selected_query_seconds += incremental_seconds
        else:
            # Preserve a score-threshold/prefix interpretation.  Skipping an
            # expensive high-score item to admit lower-score items would create
            # a label-free knapsack policy different from the frozen ranking.
            break
    by_recording: dict[str, set[str]] = {}
    for recording_id, candidate in selected:
        by_recording.setdefault(recording_id, set()).add(str(candidate["candidate_id"]))
    actual = selected_query_seconds
    return by_recording, {
        "query_budget_seconds_per_processed_evaluable_hour": float(
            budget_seconds_per_hour
        ),
        "processed_evaluable_hours": hours,
        "query_seconds_allowance": allowance,
        "available_candidate_count": len(ranked),
        "selected_candidate_count": len(selected),
        "selected_query_union_seconds": actual,
        "selection_semantics": (
            "descending_score_prefix_with_per_record_query_interval_union_cost"
        ),
    }


def _metric_value(metrics: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = metrics
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _descending_or_worst(value: float | None) -> float:
    return math.inf if value is None else -float(value)


def _ascending_or_worst(value: float | None) -> float:
    return math.inf if value is None else float(value)


def _balanced_ranking(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        _descending_or_worst(
            _metric_value(metrics, ("patient_macro", "event_sensitivity"))
        ),
        _descending_or_worst(_metric_value(metrics, ("event_sensitivity",))),
        _descending_or_worst(
            _metric_value(
                metrics, ("patient_macro", "onset_search_envelope_recall")
            )
        ),
        _ascending_or_worst(
            _metric_value(
                metrics,
                (
                    "burden",
                    "time_in_warning_fraction_of_processed_evaluable_eeg",
                ),
            )
        ),
        _ascending_or_worst(
            _metric_value(
                metrics,
                ("burden", "queried_eeg_seconds_per_processed_evaluable_hour"),
            )
        ),
        str(row["policy_id"]),
    )


def _navigation_ranking(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        _descending_or_worst(
            _metric_value(
                metrics, ("patient_macro", "onset_search_envelope_recall")
            )
        ),
        _descending_or_worst(
            _metric_value(metrics, ("onset_search_envelope_recall",))
        ),
        _descending_or_worst(
            _metric_value(metrics, ("patient_macro", "event_sensitivity"))
        ),
        _descending_or_worst(_metric_value(metrics, ("event_sensitivity",))),
        _ascending_or_worst(
            _metric_value(
                metrics, ("burden", "candidates_per_processed_evaluable_hour")
            )
        ),
        _ascending_or_worst(
            _metric_value(
                metrics,
                ("burden", "queried_eeg_seconds_per_processed_evaluable_hour"),
            )
        ),
        str(row["policy_id"]),
    )


def _partial_auc(
    policy_diagnostics: Sequence[Mapping[str, Any]],
    *,
    metric_path: Sequence[str],
    maximum_budget_per_24h: float,
) -> dict[str, Any]:
    raw: list[tuple[float, float]] = [(0.0, 0.0)]
    for row in policy_diagnostics:
        false_alarm_rate = _metric_value(
            row["metrics"],
            (
                "burden",
                "all_unmatched_alarms_per_24_processed_evaluable_hours",
            ),
        )
        sensitivity = _metric_value(row["metrics"], metric_path)
        if false_alarm_rate is None or sensitivity is None:
            continue
        if false_alarm_rate <= maximum_budget_per_24h + 1e-12:
            raw.append((max(0.0, false_alarm_rate), sensitivity))
    if len(raw) == 1:
        return {
            "maximum_false_alarm_budget_per_24h": maximum_budget_per_24h,
            "upper_envelope_points": [],
            "partial_auc": None,
            "standardized_partial_auc": None,
            "method": "trapezoidal_monotone_upper_envelope_with_null_origin",
        }
    x_values = sorted({point[0] for point in raw} | {maximum_budget_per_24h})
    envelope: list[dict[str, float]] = []
    running = 0.0
    for x_value in x_values:
        running = max(
            running,
            max(
                sensitivity
                for false_alarm_rate, sensitivity in raw
                if false_alarm_rate <= x_value + 1e-12
            ),
        )
        envelope.append(
            {
                "false_alarms_per_24_processed_evaluable_hours": x_value,
                "sensitivity_upper_envelope": running,
            }
        )
    area = 0.0
    for left, right in zip(envelope, envelope[1:]):
        width = (
            right["false_alarms_per_24_processed_evaluable_hours"]
            - left["false_alarms_per_24_processed_evaluable_hours"]
        )
        area += width * (
            left["sensitivity_upper_envelope"]
            + right["sensitivity_upper_envelope"]
        ) / 2.0
    return {
        "maximum_false_alarm_budget_per_24h": maximum_budget_per_24h,
        "upper_envelope_points": envelope,
        "partial_auc": area,
        "standardized_partial_auc": area / maximum_budget_per_24h,
        "method": "trapezoidal_monotone_upper_envelope_with_null_origin",
    }


def _selected_point(
    selected: Mapping[str, Any] | None,
    *,
    budget_name: str,
    budget_value: float,
    inventory_qualification_eligible: bool,
) -> dict[str, Any]:
    if selected is None:
        return {
            budget_name: float(budget_value),
            "status": "no_policy_with_processed_opportunity_within_budget",
            "selected_policy_id": None,
            "selection_accounting": None,
            "metrics": None,
            "qualification_eligible": False,
            "qualification_granted": False,
        }
    return {
        budget_name: float(budget_value),
        "status": "selected_calibration_diagnostic_only",
        "selected_policy_id": selected["policy_id"],
        "selection_accounting": deepcopy(selected.get("selection_accounting")),
        "metrics": deepcopy(selected["metrics"]),
        "qualification_eligible": bool(inventory_qualification_eligible),
        "qualification_granted": False,
    }


def score_detector_dual_op_v1(
    *,
    frozen_prediction_inventory: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    false_alarm_budgets_per_24h: Sequence[float] = DEFAULT_FALSE_ALARM_BUDGETS_PER_24H,
    candidate_budgets_per_hour: Sequence[float] = DEFAULT_CANDIDATE_BUDGETS_PER_HOUR,
    query_budgets_seconds_per_hour: Sequence[float] = DEFAULT_QUERY_BUDGETS_SECONDS_PER_HOUR,
    onset_tolerances_seconds: Sequence[float] = DEFAULT_ONSET_TOLERANCES_SECONDS,
) -> dict[str, Any]:
    """Score frozen source-development predictions without promotion authority."""

    # This validation is intentionally the first operation: references are not
    # inspected until the prediction-only receipt has closed and replayed.
    frozen = validate_detector_prediction_inventory_v1(
        dict(frozen_prediction_inventory)
    )
    alarm_budgets = _strict_positive_grid(
        false_alarm_budgets_per_24h, "false alarm budget"
    )
    candidate_budgets = _strict_positive_grid(
        candidate_budgets_per_hour, "candidate budget"
    )
    query_budgets = _strict_positive_grid(
        query_budgets_seconds_per_hour, "query budget"
    )
    tolerances = _strict_positive_grid(
        onset_tolerances_seconds, "onset tolerance"
    )
    joined = validate_detector_dual_op_joined_rows_v1(
        frozen_prediction_inventory=frozen,
        joined_rows=joined_rows,
    )
    by_policy: dict[str, list[dict[str, Any]]] = {
        policy_id: [] for policy_id in frozen["expected_policy_ids"]
    }
    reference_inventory: dict[str, dict[str, Any]] = {}
    for row in joined:
        policy_id = str(row["prediction_row"]["policy_id"])
        by_policy[policy_id].append(row)
        reference = row["reference_row"]
        reference_inventory.setdefault(str(reference["recording_id"]), reference)

    policy_diagnostics: list[dict[str, Any]] = []
    for policy_id in frozen["expected_policy_ids"]:
        policy_rows = sorted(
            by_policy[policy_id],
            key=lambda row: str(row["prediction_row"]["recording_id"]),
        )
        metrics = _evaluate_selected_candidates(
            policy_rows,
            _all_candidate_ids_by_recording(policy_rows),
            onset_tolerances_seconds=tolerances,
        )
        policy_diagnostics.append(
            {
                "policy_id": policy_id,
                "metrics": metrics,
                "qualification_eligible": bool(
                    metrics["coverage_accounting"]["qualification_eligible"]
                ),
                "qualification_granted": False,
            }
        )

    inventory_eligible = bool(
        frozen["coverage_accounting"]["qualification_eligible"]
    )
    balanced_curve: list[dict[str, Any]] = []
    for budget in alarm_budgets:
        within = [
            row
            for row in policy_diagnostics
            if (
                _metric_value(
                    row["metrics"],
                    (
                        "burden",
                        "all_unmatched_alarms_per_24_processed_evaluable_hours",
                    ),
                )
                is not None
                and float(
                    _metric_value(
                        row["metrics"],
                        (
                            "burden",
                            "all_unmatched_alarms_per_24_processed_evaluable_hours",
                        ),
                    )
                )
                <= budget + 1e-12
            )
        ]
        selected = min(within, key=_balanced_ranking) if within else None
        balanced_curve.append(
            _selected_point(
                selected,
                budget_name="false_alarm_budget_per_24_processed_evaluable_hours",
                budget_value=budget,
                inventory_qualification_eligible=inventory_eligible,
            )
        )

    navigation_candidate_curve: list[dict[str, Any]] = []
    for budget in candidate_budgets:
        evaluated: list[dict[str, Any]] = []
        for policy_id in frozen["expected_policy_ids"]:
            policy_rows = by_policy[policy_id]
            selected_ids, accounting = _candidate_budget_selection(
                policy_rows, budget_per_hour=budget
            )
            metrics = _evaluate_selected_candidates(
                policy_rows,
                selected_ids,
                onset_tolerances_seconds=tolerances,
            )
            if metrics["burden"]["processed_evaluable_hours"] > 0.0:
                evaluated.append(
                    {
                        "policy_id": policy_id,
                        "selection_accounting": accounting,
                        "metrics": metrics,
                    }
                )
        selected = min(evaluated, key=_navigation_ranking) if evaluated else None
        navigation_candidate_curve.append(
            _selected_point(
                selected,
                budget_name="candidate_budget_per_processed_evaluable_hour",
                budget_value=budget,
                inventory_qualification_eligible=inventory_eligible,
            )
        )

    navigation_query_curve: list[dict[str, Any]] = []
    for budget in query_budgets:
        evaluated = []
        for policy_id in frozen["expected_policy_ids"]:
            policy_rows = by_policy[policy_id]
            selected_ids, accounting = _query_budget_selection(
                policy_rows, budget_seconds_per_hour=budget
            )
            metrics = _evaluate_selected_candidates(
                policy_rows,
                selected_ids,
                onset_tolerances_seconds=tolerances,
            )
            if metrics["burden"]["processed_evaluable_hours"] > 0.0:
                evaluated.append(
                    {
                        "policy_id": policy_id,
                        "selection_accounting": accounting,
                        "metrics": metrics,
                    }
                )
        selected = min(evaluated, key=_navigation_ranking) if evaluated else None
        navigation_query_curve.append(
            _selected_point(
                selected,
                budget_name=(
                    "query_budget_eeg_seconds_per_processed_evaluable_hour"
                ),
                budget_value=budget,
                inventory_qualification_eligible=inventory_eligible,
            )
        )

    coverage = deepcopy(frozen["coverage_accounting"])
    reference_rows = sorted(
        reference_inventory.values(), key=lambda row: str(row["recording_id"])
    )
    body: dict[str, Any] = {
        "schema_version": DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION,
        "method_id": DETECTOR_DUAL_OP_DIAGNOSTIC_METHOD_ID,
        "provider_id": frozen["provider_id"],
        "split": "source_dev",
        "prediction_inventory_receipt_sha256": frozen["receipt_sha256"],
        "prediction_row_roster_sha256": frozen["prediction_row_roster_sha256"],
        "reference_inventory_sha256": _canonical_sha256(reference_rows),
        "stage_order_receipt": {
            "prediction_inventory_validated_before_reference_join_validation": True,
            "joined_prediction_rows_exactly_equal_frozen_rows": True,
            "prediction_inventory_mutated_after_reference_join": False,
            "reference_rows_opened_by_this_algorithm": 0,
            "reference_rows_received_from_upper_layer_after_freeze": len(
                reference_rows
            ),
        },
        "budget_definition": {
            "false_alarm_budgets_per_24h": list(alarm_budgets),
            "candidate_budgets_per_hour": list(candidate_budgets),
            "query_budgets_eeg_seconds_per_hour": list(query_budgets),
            "onset_tolerances_seconds": list(tolerances),
            "rate_denominator_semantics": coverage["rate_denominator_semantics"],
            "reference_denominator_semantics": coverage[
                "reference_denominator_semantics"
            ],
        },
        "coverage_accounting": coverage,
        "policy_diagnostics": policy_diagnostics,
        "op_balanced": {
            "selection_priority": (
                "patient_macro_sensitivity_then_pooled_sensitivity_then_onset_"
                "envelope_recall_then_lower_warning_query_cost_then_policy_id"
            ),
            "false_alarm_budget_curve": balanced_curve,
            "pooled_sensitivity_false_alarm_partial_auc": _partial_auc(
                policy_diagnostics,
                metric_path=("event_sensitivity",),
                maximum_budget_per_24h=max(alarm_budgets),
            ),
            "patient_macro_sensitivity_false_alarm_partial_auc": _partial_auc(
                policy_diagnostics,
                metric_path=("patient_macro", "event_sensitivity"),
                maximum_budget_per_24h=max(alarm_budgets),
            ),
        },
        "op_navigation": {
            "selection_priority": (
                "patient_macro_onset_envelope_recall_then_pooled_onset_envelope_"
                "recall_then_event_recall_then_lower_cost_then_policy_id"
            ),
            "candidate_budget_curve": navigation_candidate_curve,
            "query_budget_curve": navigation_query_curve,
            "onset_hit_denominator": (
                "all_reference_events_after_strict_event_pairing_unmatched_zero"
            ),
        },
        "qualification_eligible": inventory_eligible,
        "qualification_granted": False,
        "descriptive_research_primary_selected": False,
        "provider_promotion_authorized": False,
        "clinical_or_production_permission": False,
        "sota_claim_authorized": False,
        "status": "source_dev_calibration_diagnostic_only_no_promotion_authority",
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_detector_dual_op_diagnostic_v1(body)


def validate_detector_dual_op_diagnostic_v1(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "provider_id",
        "split",
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "reference_inventory_sha256",
        "stage_order_receipt",
        "budget_definition",
        "coverage_accounting",
        "policy_diagnostics",
        "op_balanced",
        "op_navigation",
        "qualification_eligible",
        "qualification_granted",
        "descriptive_research_primary_selected",
        "provider_promotion_authorized",
        "clinical_or_production_permission",
        "sota_claim_authorized",
        "status",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("dual-OP diagnostic fields drifted")
    data = deepcopy(value)
    if (
        data["schema_version"] != DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION
        or data["method_id"] != DETECTOR_DUAL_OP_DIAGNOSTIC_METHOD_ID
        or data["split"] != "source_dev"
        or data["qualification_granted"] is not False
        or data["descriptive_research_primary_selected"] is not False
        or data["provider_promotion_authorized"] is not False
        or data["clinical_or_production_permission"] is not False
        or data["sota_claim_authorized"] is not False
        or data["status"]
        != "source_dev_calibration_diagnostic_only_no_promotion_authority"
    ):
        raise ValueError("dual-OP diagnostic identity or permissions drifted")
    _identifier(data["provider_id"], "provider ID")
    for field in (
        "prediction_inventory_receipt_sha256",
        "prediction_row_roster_sha256",
        "reference_inventory_sha256",
        "receipt_sha256",
    ):
        _sha256(data[field], field)
    coverage = data["coverage_accounting"]
    if not isinstance(coverage, Mapping):
        raise TypeError("dual-OP coverage accounting must be an object")
    expected_eligible = (
        int(coverage.get("technical_failure_count", -1)) == 0
        and int(coverage.get("partial_coverage_recording_count", -1)) == 0
    )
    if data["qualification_eligible"] is not expected_eligible:
        raise ValueError("dual-OP qualification eligibility ignores incomplete rows")
    if coverage.get("technical_failure_not_counted_as_zero_alarm") is not True:
        raise ValueError("dual-OP diagnostic collapses failure into zero alarm")
    stage = data["stage_order_receipt"]
    if not isinstance(stage, Mapping) or (
        stage.get("prediction_inventory_validated_before_reference_join_validation")
        is not True
        or stage.get("joined_prediction_rows_exactly_equal_frozen_rows") is not True
        or stage.get("prediction_inventory_mutated_after_reference_join") is not False
        or stage.get("reference_rows_opened_by_this_algorithm") != 0
    ):
        raise ValueError("dual-OP prediction/reference stage ordering drifted")
    if not isinstance(data["policy_diagnostics"], list) or not data[
        "policy_diagnostics"
    ]:
        raise ValueError("dual-OP diagnostic lacks policy results")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("dual-OP diagnostic receipt hash drifted")
    return data


__all__ = [
    "DEFAULT_CANDIDATE_BUDGETS_PER_HOUR",
    "DEFAULT_FALSE_ALARM_BUDGETS_PER_24H",
    "DEFAULT_ONSET_TOLERANCES_SECONDS",
    "DEFAULT_QUERY_BUDGETS_SECONDS_PER_HOUR",
    "DETECTOR_DUAL_OP_DIAGNOSTIC_METHOD_ID",
    "DETECTOR_DUAL_OP_DIAGNOSTIC_SCHEMA_VERSION",
    "DETECTOR_DUAL_OP_PREDICTION_METHOD_ID",
    "DETECTOR_DUAL_OP_PREDICTION_SCHEMA_VERSION",
    "PROCESSING_STATUSES",
    "freeze_detector_prediction_inventory_v1",
    "score_detector_dual_op_v1",
    "validate_detector_dual_op_diagnostic_v1",
    "validate_detector_dual_op_joined_rows_v1",
    "validate_detector_prediction_inventory_v1",
]
