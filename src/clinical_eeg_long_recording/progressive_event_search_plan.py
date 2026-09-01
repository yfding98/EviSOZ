"""Plan progressive per-event EEG search envelopes without reading EEG.

The input is a completed continuous-posterior decode receipt whose adjacent
alarms have already been merged into event proposals.  This planner creates
three nested search levels (60, 120 and 300 seconds) around each decoded event
interval.  Physical recording boundaries clip a level; neighbouring events
are marked as overlaps but never midpoint-clipped or silently split.

This is a planning contract only.  It does not inspect EEG, determine whether
background/ictal/post-event states close, refine onset/offset, or claim that a
decoded proposal is a seizure.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Sequence

from .continuous_detection import validate_continuous_seizure_decoding


PROGRESSIVE_EVENT_SEARCH_PLAN_SCHEMA_VERSION = "progressive_event_search_plan_v1"
PROGRESSIVE_EVENT_SEARCH_PLAN_METHOD_ID = (
    "decoded_interval_nested_60_120_300_search_planner_v1"
)
DEFAULT_PROGRESSIVE_SEARCH_LEVEL_SECONDS = (60.0, 120.0, 300.0)


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


def _levels(value: Sequence[float] | None) -> tuple[float, ...]:
    raw = DEFAULT_PROGRESSIVE_SEARCH_LEVEL_SECONDS if value is None else value
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("progressive search levels must be a sequence")
    levels = tuple(_finite(item, "progressive search level") for item in raw)
    if (
        len(levels) < 2
        or any(item <= 0 for item in levels)
        or any(right <= left for left, right in zip(levels, levels[1:]))
    ):
        raise ValueError("progressive search levels must be positive and increasing")
    if levels != DEFAULT_PROGRESSIVE_SEARCH_LEVEL_SECONDS:
        raise ValueError("progressive_event_search_plan_v1 requires 60/120/300 seconds")
    return levels


def _event_interval(event: dict[str, Any], duration: float) -> tuple[float, float, float]:
    required = {
        "start_offset_seconds",
        "stop_offset_seconds",
        "anchor_offset_seconds",
        "peak_probability",
        "mean_probability",
        "right_censored",
        "support_window_ids",
        "candidate_id",
        "candidate_semantics",
    }
    if set(event) != required:
        raise ValueError("continuous decoded event has missing or unknown fields")
    start = _finite(event["start_offset_seconds"], "decoded event start")
    stop = _finite(event["stop_offset_seconds"], "decoded event stop")
    anchor = _finite(event["anchor_offset_seconds"], "decoded event anchor")
    if not 0 <= start <= anchor < stop <= duration + 1e-6:
        raise ValueError("continuous decoded event interval is invalid")
    if type(event["right_censored"]) is not bool:
        raise TypeError("decoded event right_censored must be boolean")
    if not isinstance(event["candidate_id"], str) or not event["candidate_id"]:
        raise ValueError("decoded event candidate_id is invalid")
    if event["candidate_semantics"] != (
        "model_detected_event_proposal_not_confirmed_seizure_or_onset"
    ):
        raise ValueError("decoded event candidate semantics drifted")
    return start, stop, anchor


def _overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) < min(
        float(left[1]), float(right[1])
    ) - 1e-9


def _next_level_decision(
    *,
    current_level: float,
    next_level: float | None,
    recording_fully_exhausted: bool,
) -> dict[str, Any]:
    if recording_fully_exhausted:
        unresolved_action = "stop_recording_boundary_censored"
        next_target = None
    elif next_level is None:
        unresolved_action = "stop_search_cap_censored"
        next_target = None
    else:
        unresolved_action = "expand_to_next_level"
        next_target = next_level
    return {
        "decision_status": "pending_signal_state_closure_evaluation",
        "state_segmentation_already_completed": False,
        "if_onset_and_termination_state_closed": (
            "stop_and_derive_variable_event_window"
        ),
        "if_state_unresolved": unresolved_action,
        "current_target_span_seconds": current_level,
        "next_target_span_seconds": next_target,
    }


def build_progressive_event_search_plan(
    continuous_decoding_receipt: object,
    *,
    search_level_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build nested event-specific search intervals from decoded event spans."""

    decoded = validate_continuous_seizure_decoding(continuous_decoding_receipt)
    levels = _levels(search_level_seconds)
    duration = _finite(decoded["recording_duration_seconds"], "recording duration")
    source_events = [deepcopy(item) for item in decoded["event_proposals"]]
    parsed: list[tuple[float, float, float]] = [
        _event_interval(event, duration) for event in source_events
    ]
    candidate_ids = [str(event["candidate_id"]) for event in source_events]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("continuous decoded candidate IDs are not unique")
    if parsed != sorted(parsed, key=lambda item: (item[0], item[2], item[1])):
        raise ValueError("continuous decoded events are not in recording order")

    planned_intervals: list[list[list[float]]] = []
    for (decoded_start, decoded_stop, anchor) in parsed:
        event_levels: list[list[float]] = []
        for level in levels:
            half = level / 2.0
            requested = [
                min(decoded_start, anchor - half),
                max(decoded_stop, anchor + half),
            ]
            event_levels.append(
                [max(0.0, requested[0]), min(duration, requested[1])]
            )
        planned_intervals.append(event_levels)

    event_rows: list[dict[str, Any]] = []
    for event_index, (event, parsed_interval) in enumerate(
        zip(source_events, parsed), start=1
    ):
        decoded_start, decoded_stop, anchor = parsed_interval
        level_rows: list[dict[str, Any]] = []
        for level_index, level in enumerate(levels):
            half = level / 2.0
            requested = [
                min(decoded_start, anchor - half),
                max(decoded_stop, anchor + half),
            ]
            effective = planned_intervals[event_index - 1][level_index]
            recording_start = requested[0] < 0.0
            recording_stop = requested[1] > duration
            overlapping_search_ids = [
                candidate_ids[other_index]
                for other_index in range(len(source_events))
                if other_index != event_index - 1
                and _overlap(effective, planned_intervals[other_index][level_index])
            ]
            overlapping_decoded_ids = [
                candidate_ids[other_index]
                for other_index, other_interval in enumerate(parsed)
                if other_index != event_index - 1
                and _overlap(effective, other_interval[:2])
            ]
            fully_exhausted = (
                effective[0] <= 1e-9 and effective[1] >= duration - 1e-9
            )
            next_level = levels[level_index + 1] if level_index + 1 < len(levels) else None
            level_rows.append(
                {
                    "level_index": level_index + 1,
                    "target_span_seconds": level,
                    "requested_interval_recording_seconds": requested,
                    "effective_interval_recording_seconds": list(effective),
                    "decoded_event_interval_fully_retained": (
                        effective[0] <= decoded_start + 1e-9
                        and effective[1] >= decoded_stop - 1e-9
                    ),
                    "boundary_and_overlap_markers": {
                        "recording_start": recording_start,
                        "recording_stop": recording_stop,
                        "recording_fully_exhausted": fully_exhausted,
                        "decoded_event_right_censored": bool(
                            event["right_censored"]
                        ),
                        "neighbor_event_overlap": bool(overlapping_search_ids),
                        "neighbor_search_interval_overlap_candidate_ids": (
                            overlapping_search_ids
                        ),
                        "neighbor_decoded_interval_candidate_ids": (
                            overlapping_decoded_ids
                        ),
                        "search_cap_if_unresolved": (
                            next_level is None and not fully_exhausted
                        ),
                    },
                    "neighbor_handling": {
                        "midpoint_clipped": False,
                        "interval_split": False,
                        "overlap_is_marker_only": True,
                    },
                    "next_level_decision": _next_level_decision(
                        current_level=level,
                        next_level=next_level,
                        recording_fully_exhausted=fully_exhausted,
                    ),
                }
            )
        event_rows.append(
            {
                "event_index": event_index,
                "candidate_id": event["candidate_id"],
                "decoded_event_interval_recording_seconds": [
                    decoded_start,
                    decoded_stop,
                ],
                "navigation_anchor_recording_seconds": anchor,
                "decoded_event_right_censored": bool(event["right_censored"]),
                "candidate_semantics": event["candidate_semantics"],
                "levels": level_rows,
            }
        )

    body: dict[str, Any] = {
        "schema_version": PROGRESSIVE_EVENT_SEARCH_PLAN_SCHEMA_VERSION,
        "plan_id": "CONTENT-ADDRESS-PENDING",
        "method_id": PROGRESSIVE_EVENT_SEARCH_PLAN_METHOD_ID,
        "source_decoding_receipt_id": decoded["decoding_receipt_id"],
        "source_decoding_receipt_sha256": _canonical_sha256(decoded),
        "recording_id": decoded["recording_id"],
        "source_signal_sha256": decoded["source_signal_sha256"],
        "recording_duration_seconds": duration,
        "search_level_seconds": list(levels),
        "event_count": len(event_rows),
        "events": event_rows,
        "execution_status": "planned_not_signal_segmented",
        "policy": {
            "input_unit": "merged_continuous_decoded_event_interval",
            "requested_interval_rule": (
                "union_of_decoded_interval_and_anchor_centered_target_span"
            ),
            "physical_recording_boundary_clipping_only": True,
            "neighbor_midpoint_clipping": False,
            "neighbor_overlap_action": "mark_do_not_clip_or_split",
            "state_closure_required_before_stopping_early": True,
            "final_unresolved_level_semantics": "search_cap_censored",
            "signal_state_segmentation_performed_by_planner": False,
        },
        "scope_receipt": {
            "continuous_decoding_receipt_only": True,
            "eeg_samples_read": False,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "decoded_event_is_confirmed_seizure": False,
            "navigation_anchor_is_confirmed_onset": False,
        },
    }
    body["plan_id"] = "PROGRESSIVE-PLAN-" + _canonical_sha256(body)[:20]
    return validate_progressive_event_search_plan(body)


def validate_progressive_event_search_plan(payload: object) -> dict[str, Any]:
    """Strictly validate progressive nesting, markers and non-leakage."""

    if type(payload) is not dict:
        raise TypeError("progressive event search plan must be an object")
    required = {
        "schema_version",
        "plan_id",
        "method_id",
        "source_decoding_receipt_id",
        "source_decoding_receipt_sha256",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "search_level_seconds",
        "event_count",
        "events",
        "execution_status",
        "policy",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("progressive event search plan has missing or unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != PROGRESSIVE_EVENT_SEARCH_PLAN_SCHEMA_VERSION
        or data["method_id"] != PROGRESSIVE_EVENT_SEARCH_PLAN_METHOD_ID
        or data["execution_status"] != "planned_not_signal_segmented"
    ):
        raise ValueError("progressive event search plan schema/status drifted")
    for field in ("source_decoding_receipt_sha256", "source_signal_sha256"):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"progressive event search {field} is invalid")
    for field in ("plan_id", "source_decoding_receipt_id", "recording_id"):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"progressive event search {field} is invalid")
    duration = _finite(data["recording_duration_seconds"], "recording duration")
    if duration <= 0:
        raise ValueError("progressive event search duration is invalid")
    levels = _levels(data["search_level_seconds"])
    events = data["events"]
    if not isinstance(events, list) or data["event_count"] != len(events):
        raise ValueError("progressive event search event count drifted")
    ids: list[str] = []
    intervals_by_level: list[list[list[float]]] = [
        [] for _ in range(len(levels))
    ]
    decoded_intervals: list[list[float]] = []
    for event_index, event in enumerate(events, start=1):
        if type(event) is not dict or set(event) != {
            "event_index",
            "candidate_id",
            "decoded_event_interval_recording_seconds",
            "navigation_anchor_recording_seconds",
            "decoded_event_right_censored",
            "candidate_semantics",
            "levels",
        }:
            raise ValueError("progressive event search event is malformed")
        if event["event_index"] != event_index:
            raise ValueError("progressive event search event order drifted")
        candidate_id = event["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in ids:
            raise ValueError("progressive event search candidate ID is invalid")
        ids.append(candidate_id)
        decoded_interval = event["decoded_event_interval_recording_seconds"]
        if not isinstance(decoded_interval, list) or len(decoded_interval) != 2:
            raise TypeError("progressive decoded interval must be a pair")
        decoded_interval = [
            _finite(item, "progressive decoded interval") for item in decoded_interval
        ]
        anchor = _finite(
            event["navigation_anchor_recording_seconds"], "navigation anchor"
        )
        if not 0 <= decoded_interval[0] <= anchor < decoded_interval[1] <= duration + 1e-6:
            raise ValueError("progressive decoded interval is invalid")
        decoded_intervals.append(decoded_interval)
        if type(event["decoded_event_right_censored"]) is not bool:
            raise TypeError("progressive decoded censoring flag is invalid")
        if event["candidate_semantics"] != (
            "model_detected_event_proposal_not_confirmed_seizure_or_onset"
        ):
            raise ValueError("progressive candidate semantics drifted")
        level_rows = event["levels"]
        if not isinstance(level_rows, list) or len(level_rows) != len(levels):
            raise ValueError("progressive event level count drifted")
        previous_effective: list[float] | None = None
        for level_index, (level, row) in enumerate(zip(levels, level_rows), start=1):
            if type(row) is not dict or set(row) != {
                "level_index",
                "target_span_seconds",
                "requested_interval_recording_seconds",
                "effective_interval_recording_seconds",
                "decoded_event_interval_fully_retained",
                "boundary_and_overlap_markers",
                "neighbor_handling",
                "next_level_decision",
            }:
                raise ValueError("progressive event search level is malformed")
            if row["level_index"] != level_index or abs(
                _finite(row["target_span_seconds"], "target span") - level
            ) > 1e-9:
                raise ValueError("progressive event search level identity drifted")
            half = level / 2.0
            expected_requested = [
                min(decoded_interval[0], anchor - half),
                max(decoded_interval[1], anchor + half),
            ]
            requested = row["requested_interval_recording_seconds"]
            effective = row["effective_interval_recording_seconds"]
            if not isinstance(requested, list) or not isinstance(effective, list) or len(requested) != 2 or len(effective) != 2:
                raise TypeError("progressive event search intervals must be pairs")
            requested = [_finite(item, "requested interval") for item in requested]
            effective = [_finite(item, "effective interval") for item in effective]
            expected_effective = [
                max(0.0, expected_requested[0]),
                min(duration, expected_requested[1]),
            ]
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(requested, expected_requested)
            ) or any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(effective, expected_effective)
            ):
                raise ValueError("progressive event interval policy drifted")
            if previous_effective is not None and (
                effective[0] > previous_effective[0] + 1e-6
                or effective[1] < previous_effective[1] - 1e-6
            ):
                raise ValueError("progressive event levels are not nested")
            previous_effective = effective
            intervals_by_level[level_index - 1].append(effective)
            retained = (
                effective[0] <= decoded_interval[0] + 1e-9
                and effective[1] >= decoded_interval[1] - 1e-9
            )
            if row["decoded_event_interval_fully_retained"] is not retained or not retained:
                raise ValueError("progressive search dropped decoded event signal")
            neighbor = row["neighbor_handling"]
            if neighbor != {
                "midpoint_clipped": False,
                "interval_split": False,
                "overlap_is_marker_only": True,
            }:
                raise ValueError("progressive search midpoint-clipped a neighbor")

    for event_index, event in enumerate(events):
        for level_index, row in enumerate(event["levels"]):
            effective = intervals_by_level[level_index][event_index]
            expected_search_neighbors = [
                ids[other]
                for other in range(len(events))
                if other != event_index
                and _overlap(effective, intervals_by_level[level_index][other])
            ]
            expected_decoded_neighbors = [
                ids[other]
                for other in range(len(events))
                if other != event_index
                and _overlap(effective, decoded_intervals[other])
            ]
            markers = row["boundary_and_overlap_markers"]
            if type(markers) is not dict or markers != {
                "recording_start": row["requested_interval_recording_seconds"][0] < 0,
                "recording_stop": row["requested_interval_recording_seconds"][1] > duration,
                "recording_fully_exhausted": (
                    row["effective_interval_recording_seconds"][0] <= 1e-9
                    and row["effective_interval_recording_seconds"][1]
                    >= duration - 1e-9
                ),
                "decoded_event_right_censored": event["decoded_event_right_censored"],
                "neighbor_event_overlap": bool(expected_search_neighbors),
                "neighbor_search_interval_overlap_candidate_ids": expected_search_neighbors,
                "neighbor_decoded_interval_candidate_ids": expected_decoded_neighbors,
                "search_cap_if_unresolved": (
                    level_index == len(levels) - 1
                    and not (
                        row["effective_interval_recording_seconds"][0] <= 1e-9
                        and row["effective_interval_recording_seconds"][1]
                        >= duration - 1e-9
                    )
                ),
            }:
                raise ValueError("progressive boundary/overlap markers drifted")
            effective_interval = row["effective_interval_recording_seconds"]
            fully_exhausted = (
                effective_interval[0] <= 1e-9
                and effective_interval[1] >= duration - 1e-9
            )
            next_level = levels[level_index + 1] if level_index + 1 < len(levels) else None
            expected_decision = _next_level_decision(
                current_level=levels[level_index],
                next_level=next_level,
                recording_fully_exhausted=fully_exhausted,
            )
            if row["next_level_decision"] != expected_decision:
                raise ValueError("progressive next-level decision drifted")

    expected_policy = {
        "input_unit": "merged_continuous_decoded_event_interval",
        "requested_interval_rule": (
            "union_of_decoded_interval_and_anchor_centered_target_span"
        ),
        "physical_recording_boundary_clipping_only": True,
        "neighbor_midpoint_clipping": False,
        "neighbor_overlap_action": "mark_do_not_clip_or_split",
        "state_closure_required_before_stopping_early": True,
        "final_unresolved_level_semantics": "search_cap_censored",
        "signal_state_segmentation_performed_by_planner": False,
    }
    if data["policy"] != expected_policy:
        raise ValueError("progressive event search policy drifted")
    expected_scope = {
        "continuous_decoding_receipt_only": True,
        "eeg_samples_read": False,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "decoded_event_is_confirmed_seizure": False,
        "navigation_anchor_is_confirmed_onset": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("progressive event search plan violated EEG-only scope")
    digest = deepcopy(data)
    digest["plan_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "PROGRESSIVE-PLAN-" + _canonical_sha256(digest)[:20]
    if data["plan_id"] != expected_id:
        raise ValueError("progressive event search plan ID does not bind content")
    return data


__all__ = [
    "DEFAULT_PROGRESSIVE_SEARCH_LEVEL_SECONDS",
    "PROGRESSIVE_EVENT_SEARCH_PLAN_METHOD_ID",
    "PROGRESSIVE_EVENT_SEARCH_PLAN_SCHEMA_VERSION",
    "build_progressive_event_search_plan",
    "validate_progressive_event_search_plan",
]
