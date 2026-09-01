"""Execute progressive 60/120/300-second EEG state searches per event.

The executor is deliberately independent from the default report batch.  It
binds a continuous decoding receipt, its deterministic progressive plan and a
complete EDF signal hash, then reads each event level in order.  A level stops
early only when the existing adaptive transition analyzer returns
``qualified_complete``.  Otherwise it expands until the physical recording is
fully exhausted or the 300-second search cap is reached.

EDF annotations, spreadsheets, labels and clinical context have no input
route.  Neighbour overlap remains a marker and never causes midpoint clipping.
The output describes an algorithmic scalp-EEG transition search; it does not
claim a confirmed seizure, cortical SOZ, epileptogenic zone, or completed joint
background/ictal/post-event state segmentation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .adaptive_search import (
    ADAPTIVE_SEARCH_METHOD_ID,
    ADAPTIVE_SEARCH_POLICY,
    ADAPTIVE_SEARCH_POLICY_SHA256,
    analyze_adaptive_eeg_envelope,
    generalized_signal_tensor_sha256,
    validate_adaptive_search_receipt,
)
from .adaptive_search_materialization import (
    AdaptiveEnvelopeLoader,
    LoadedAdaptiveEnvelope,
    load_standard19_adaptive_envelope,
    validate_adaptive_preprocessing_receipt,
)
from .continuous_detection import validate_continuous_seizure_decoding
from .progressive_event_search_plan import (
    build_progressive_event_search_plan,
    validate_progressive_event_search_plan,
)


PROGRESSIVE_STATE_SEARCH_EXECUTION_SCHEMA_VERSION = (
    "progressive_eeg_state_search_execution_v1"
)
PROGRESSIVE_STATE_SEARCH_EXECUTION_METHOD_ID = (
    "progressive_60_120_300_adaptive_transition_executor_v1"
)

AdaptiveAnalyzer = Callable[..., dict[str, Any]]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_edf(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("progressive state-search EDF must not be a symlink")
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".edf":
        raise ValueError("progressive state-search source must be a regular EDF")
    return source


def _atomic_json(path: Path, value: object) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, target)
        os.chmod(target, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _same_pair(left: object, right: object) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != 2 or len(right) != 2:
        return False
    return all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(left, right))


def _unresolved_reasons(status: str) -> list[str]:
    return {
        "partial_left_boundary": ["onset_left_censored"],
        "partial_right_boundary": ["termination_right_censored"],
        "abstained_insufficient_baseline": ["baseline_unavailable"],
        "abstained_no_onset_transition": ["onset_unresolved"],
        "abstained_no_termination_transition": ["termination_unresolved"],
        "abstained_artifact_dominated": ["artifact_dominated"],
        "abstained_low_confidence": ["joint_confidence_unresolved"],
        "abstained_envelope_unavailable": ["envelope_unavailable"],
    }.get(status, ["adaptive_state_unresolved"])


def _terminal_censoring(
    *,
    search_status: str,
    markers: dict[str, Any],
    terminal_decision: str,
) -> dict[str, Any]:
    if search_status == "qualified_complete":
        return {"is_censored": False, "reasons": []}
    reasons = _unresolved_reasons(search_status)
    if markers["recording_start"]:
        reasons.append("recording_start")
    if markers["recording_stop"]:
        reasons.append("recording_stop")
    if terminal_decision == "stop_search_cap_censored":
        reasons.append("search_cap_300_seconds")
    return {"is_censored": True, "reasons": list(dict.fromkeys(reasons))}


def execute_progressive_eeg_state_search(
    *,
    continuous_decoding_receipt: object,
    progressive_search_plan: object,
    edf_path: Path,
    output_path: Path | None = None,
    envelope_loader: AdaptiveEnvelopeLoader = load_standard19_adaptive_envelope,
    adaptive_analyzer: AdaptiveAnalyzer = analyze_adaptive_eeg_envelope,
) -> dict[str, Any]:
    """Execute each planned level until closure, physical boundary, or cap."""

    decoded = validate_continuous_seizure_decoding(continuous_decoding_receipt)
    plan = validate_progressive_event_search_plan(progressive_search_plan)
    expected_plan = build_progressive_event_search_plan(decoded)
    if plan != expected_plan:
        raise ValueError("progressive plan does not exactly bind the decoding receipt")
    if (
        plan["source_decoding_receipt_id"] != decoded["decoding_receipt_id"]
        or plan["source_decoding_receipt_sha256"] != _canonical_sha256(decoded)
    ):
        raise ValueError("progressive plan source decoding binding drifted")
    source = _source_edf(edf_path)
    if _file_sha256(source) != decoded["source_signal_sha256"]:
        raise ValueError("progressive state-search EDF hash differs from decoding")
    if not callable(envelope_loader) or not callable(adaptive_analyzer):
        raise TypeError("progressive envelope loader/analyzer must be callable")

    duration = float(decoded["recording_duration_seconds"])
    event_rows: list[dict[str, Any]] = []
    for event in plan["events"]:
        executed_levels: list[dict[str, Any]] = []
        terminal_decision: str | None = None
        terminal_search_status: str | None = None
        terminal_markers: dict[str, Any] | None = None
        for planned_level in event["levels"]:
            start, stop = map(
                float, planned_level["effective_interval_recording_seconds"]
            )
            loaded = envelope_loader(
                source,
                start_recording_seconds=start,
                stop_recording_seconds=stop,
                source_signal_sha256=decoded["source_signal_sha256"],
            )
            if not isinstance(loaded, LoadedAdaptiveEnvelope):
                raise TypeError("progressive envelope loader returned an invalid object")
            preprocessing = validate_adaptive_preprocessing_receipt(
                loaded.preprocessing_receipt
            )
            if (
                preprocessing["source_signal_sha256"]
                != decoded["source_signal_sha256"]
                or not _same_pair(
                    preprocessing["requested_interval_recording_seconds"],
                    [start, stop],
                )
            ):
                raise ValueError("progressive preprocessing binding drifted")
            signal_hash = generalized_signal_tensor_sha256(loaded.signal)
            if signal_hash != preprocessing["processed_envelope_sha256"]:
                raise ValueError("progressive preprocessing receipt does not bind EEG")
            search = adaptive_analyzer(
                loaded.signal,
                sampling_rate_hz=preprocessing["output_sampling_rate_hz"],
                envelope_start_recording_seconds=start,
                candidate_anchor_recording_seconds=event[
                    "navigation_anchor_recording_seconds"
                ],
                recording_duration_seconds=duration,
                processed_envelope_sha256=signal_hash,
                preprocessing_receipt_sha256=preprocessing["receipt_sha256"],
            )
            search = validate_adaptive_search_receipt(search)
            if (
                search["processed_envelope_sha256"] != signal_hash
                or search["preprocessing_receipt_sha256"]
                != preprocessing["receipt_sha256"]
                or not _same_pair(
                    search["envelope_interval_recording_seconds"], [start, stop]
                )
                or abs(
                    float(search["coarse_anchor_recording_seconds"])
                    - float(event["navigation_anchor_recording_seconds"])
                )
                > 1e-6
            ):
                raise ValueError("progressive adaptive-search binding drifted")

            markers = deepcopy(
                planned_level["boundary_and_overlap_markers"]
            )
            status = str(search["status"])
            if status == "qualified_complete":
                decision = "stop_qualified_complete"
            elif markers["recording_fully_exhausted"]:
                decision = "stop_recording_boundary_censored"
            elif planned_level["level_index"] == len(event["levels"]):
                decision = "stop_search_cap_censored"
            else:
                decision = "expand_to_next_level"
            executed_levels.append(
                {
                    "level_index": planned_level["level_index"],
                    "target_span_seconds": planned_level["target_span_seconds"],
                    "planned_level": deepcopy(planned_level),
                    "preprocessing_receipt": preprocessing,
                    "adaptive_search_receipt": search,
                    "search_status": status,
                    "execution_decision": decision,
                }
            )
            if decision != "expand_to_next_level":
                terminal_decision = decision
                terminal_search_status = status
                terminal_markers = markers
                break
        if terminal_decision is None or terminal_search_status is None or terminal_markers is None:
            raise RuntimeError("progressive event search ended without a terminal decision")
        censoring = _terminal_censoring(
            search_status=terminal_search_status,
            markers=terminal_markers,
            terminal_decision=terminal_decision,
        )
        event_rows.append(
            {
                "event_index": event["event_index"],
                "candidate_id": event["candidate_id"],
                "decoded_event_interval_recording_seconds": event[
                    "decoded_event_interval_recording_seconds"
                ],
                "navigation_anchor_recording_seconds": event[
                    "navigation_anchor_recording_seconds"
                ],
                "decoded_event_right_censored": event[
                    "decoded_event_right_censored"
                ],
                "executed_level_count": len(executed_levels),
                "executed_levels": executed_levels,
                "terminal_status": (
                    "completed_qualified_state_closed"
                    if terminal_decision == "stop_qualified_complete"
                    else "completed_censored_state_unresolved"
                ),
                "terminal_search_status": terminal_search_status,
                "terminal_decision": terminal_decision,
                "censoring": censoring,
            }
        )

    body: dict[str, Any] = {
        "schema_version": PROGRESSIVE_STATE_SEARCH_EXECUTION_SCHEMA_VERSION,
        "execution_id": "CONTENT-ADDRESS-PENDING",
        "method_id": PROGRESSIVE_STATE_SEARCH_EXECUTION_METHOD_ID,
        "adaptive_analyzer_method_id": ADAPTIVE_SEARCH_METHOD_ID,
        "analyzer_search_contract": {
            "policy_sha256": ADAPTIVE_SEARCH_POLICY_SHA256,
            "progressive_envelope_level_seconds": [60.0, 120.0, 300.0],
            "onset_search_horizon_relative_to_navigation_anchor_seconds": [
                -float(ADAPTIVE_SEARCH_POLICY[
                    "maximum_onset_seconds_before_anchor"
                ]),
                float(ADAPTIVE_SEARCH_POLICY[
                    "maximum_onset_seconds_after_anchor"
                ]),
            ],
            "progressive_envelope_expands_internal_onset_horizon": False,
            "termination_search_can_use_remaining_envelope": True,
            "joint_state_segmentation_performed": False,
        },
        "source_decoding_receipt_id": decoded["decoding_receipt_id"],
        "source_decoding_receipt_sha256": _canonical_sha256(decoded),
        "source_progressive_plan_id": plan["plan_id"],
        "source_progressive_plan_sha256": _canonical_sha256(plan),
        "recording_id": decoded["recording_id"],
        "source_signal_sha256": decoded["source_signal_sha256"],
        "recording_duration_seconds": duration,
        "event_count": len(event_rows),
        "events": event_rows,
        "summary": {
            "qualified_complete_event_count": sum(
                event["terminal_status"] == "completed_qualified_state_closed"
                for event in event_rows
            ),
            "censored_unresolved_event_count": sum(
                event["terminal_status"] == "completed_censored_state_unresolved"
                for event in event_rows
            ),
            "executed_level_count": sum(
                event["executed_level_count"] for event in event_rows
            ),
        },
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotation_api_called": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "neighbor_midpoint_clipping_used": False,
            "silent_padding_used": False,
            "default_batch_integration_enabled": False,
            "decoded_event_is_confirmed_seizure": False,
            "navigation_anchor_is_confirmed_onset": False,
            "joint_state_segmentation_claimed_complete": False,
        },
    }
    body["execution_id"] = "PROGRESSIVE-EXEC-" + _canonical_sha256(body)[:20]
    receipt = validate_progressive_eeg_state_search_execution(body)
    if output_path is not None:
        _atomic_json(output_path, receipt)
    return receipt


def validate_progressive_eeg_state_search_execution(
    payload: object,
) -> dict[str, Any]:
    """Validate bindings, sequential decisions, censoring and EEG-only scope."""

    if type(payload) is not dict:
        raise TypeError("progressive state-search execution must be an object")
    required = {
        "schema_version",
        "execution_id",
        "method_id",
        "adaptive_analyzer_method_id",
        "analyzer_search_contract",
        "source_decoding_receipt_id",
        "source_decoding_receipt_sha256",
        "source_progressive_plan_id",
        "source_progressive_plan_sha256",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "event_count",
        "events",
        "summary",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("progressive state-search execution has invalid fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != PROGRESSIVE_STATE_SEARCH_EXECUTION_SCHEMA_VERSION
        or data["method_id"] != PROGRESSIVE_STATE_SEARCH_EXECUTION_METHOD_ID
        or data["adaptive_analyzer_method_id"] != ADAPTIVE_SEARCH_METHOD_ID
    ):
        raise ValueError("progressive state-search schema/method drifted")
    expected_analyzer_contract = {
        "policy_sha256": ADAPTIVE_SEARCH_POLICY_SHA256,
        "progressive_envelope_level_seconds": [60.0, 120.0, 300.0],
        "onset_search_horizon_relative_to_navigation_anchor_seconds": [
            -float(ADAPTIVE_SEARCH_POLICY["maximum_onset_seconds_before_anchor"]),
            float(ADAPTIVE_SEARCH_POLICY["maximum_onset_seconds_after_anchor"]),
        ],
        "progressive_envelope_expands_internal_onset_horizon": False,
        "termination_search_can_use_remaining_envelope": True,
        "joint_state_segmentation_performed": False,
    }
    if data["analyzer_search_contract"] != expected_analyzer_contract:
        raise ValueError("progressive adaptive-analyzer search contract drifted")
    for field in (
        "source_decoding_receipt_sha256",
        "source_progressive_plan_sha256",
        "source_signal_sha256",
    ):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"progressive state-search {field} is invalid")
    for field in (
        "execution_id",
        "source_decoding_receipt_id",
        "source_progressive_plan_id",
        "recording_id",
    ):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"progressive state-search {field} is invalid")
    duration = float(data["recording_duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("progressive state-search duration is invalid")
    events = data["events"]
    if not isinstance(events, list) or data["event_count"] != len(events):
        raise ValueError("progressive state-search event count drifted")
    seen: set[str] = set()
    for event_index, event in enumerate(events, start=1):
        if type(event) is not dict or set(event) != {
            "event_index",
            "candidate_id",
            "decoded_event_interval_recording_seconds",
            "navigation_anchor_recording_seconds",
            "decoded_event_right_censored",
            "executed_level_count",
            "executed_levels",
            "terminal_status",
            "terminal_search_status",
            "terminal_decision",
            "censoring",
        }:
            raise ValueError("progressive state-search event is malformed")
        candidate_id = event["candidate_id"]
        if (
            event["event_index"] != event_index
            or not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
        ):
            raise ValueError("progressive state-search event identity drifted")
        seen.add(candidate_id)
        if type(event["decoded_event_right_censored"]) is not bool:
            raise TypeError("progressive decoded censoring marker is invalid")
        anchor = float(event["navigation_anchor_recording_seconds"])
        if not math.isfinite(anchor) or not 0 <= anchor <= duration:
            raise ValueError("progressive navigation anchor is invalid")
        decoded_interval = event["decoded_event_interval_recording_seconds"]
        if (
            not isinstance(decoded_interval, list)
            or len(decoded_interval) != 2
            or not 0
            <= float(decoded_interval[0])
            <= anchor
            < float(decoded_interval[1])
            <= duration
        ):
            raise ValueError("progressive decoded event interval is invalid")
        levels = event["executed_levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or event["executed_level_count"] != len(levels)
            or len(levels) > 3
        ):
            raise ValueError("progressive executed level count is invalid")
        for level_index, level in enumerate(levels, start=1):
            if type(level) is not dict or set(level) != {
                "level_index",
                "target_span_seconds",
                "planned_level",
                "preprocessing_receipt",
                "adaptive_search_receipt",
                "search_status",
                "execution_decision",
            }:
                raise ValueError("progressive executed level is malformed")
            planned = level["planned_level"]
            if type(planned) is not dict or set(planned) != {
                "level_index",
                "target_span_seconds",
                "requested_interval_recording_seconds",
                "effective_interval_recording_seconds",
                "decoded_event_interval_fully_retained",
                "boundary_and_overlap_markers",
                "neighbor_handling",
                "next_level_decision",
            }:
                raise ValueError("progressive embedded plan level is malformed")
            expected_target = (60.0, 120.0, 300.0)[level_index - 1]
            if (
                level["level_index"] != level_index
                or level["level_index"] != planned.get("level_index")
                or float(level["target_span_seconds"])
                != float(planned.get("target_span_seconds"))
                or float(level["target_span_seconds"]) != expected_target
                or planned.get("neighbor_handling")
                != {
                    "midpoint_clipped": False,
                    "interval_split": False,
                    "overlap_is_marker_only": True,
                }
            ):
                raise ValueError("progressive executed plan level drifted")
            expected_requested = [
                min(float(decoded_interval[0]), anchor - expected_target / 2.0),
                max(float(decoded_interval[1]), anchor + expected_target / 2.0),
            ]
            expected_effective = [
                max(0.0, expected_requested[0]),
                min(duration, expected_requested[1]),
            ]
            if (
                not _same_pair(
                    planned["requested_interval_recording_seconds"],
                    expected_requested,
                )
                or not _same_pair(
                    planned["effective_interval_recording_seconds"],
                    expected_effective,
                )
                or planned["decoded_event_interval_fully_retained"] is not True
            ):
                raise ValueError("progressive embedded plan interval drifted")
            markers = planned["boundary_and_overlap_markers"]
            if type(markers) is not dict or set(markers) != {
                "recording_start",
                "recording_stop",
                "recording_fully_exhausted",
                "decoded_event_right_censored",
                "neighbor_event_overlap",
                "neighbor_search_interval_overlap_candidate_ids",
                "neighbor_decoded_interval_candidate_ids",
                "search_cap_if_unresolved",
            }:
                raise ValueError("progressive embedded boundary markers are malformed")
            if (
                markers["recording_start"] is not (expected_requested[0] < 0)
                or markers["recording_stop"] is not (expected_requested[1] > duration)
                or markers["recording_fully_exhausted"]
                is not (
                    expected_effective[0] <= 1e-9
                    and expected_effective[1] >= duration - 1e-9
                )
                or markers["decoded_event_right_censored"]
                is not event["decoded_event_right_censored"]
                or type(markers["neighbor_event_overlap"]) is not bool
                or markers["neighbor_event_overlap"]
                is not bool(
                    markers["neighbor_search_interval_overlap_candidate_ids"]
                )
                or not isinstance(
                    markers["neighbor_search_interval_overlap_candidate_ids"], list
                )
                or not isinstance(
                    markers["neighbor_decoded_interval_candidate_ids"], list
                )
                or type(markers["search_cap_if_unresolved"]) is not bool
                or markers["search_cap_if_unresolved"]
                is not (
                    level_index == 3
                    and not markers["recording_fully_exhausted"]
                )
            ):
                raise ValueError("progressive embedded boundary markers drifted")
            next_target = (
                None if level_index == 3 else (60.0, 120.0, 300.0)[level_index]
            )
            if markers["recording_fully_exhausted"]:
                unresolved_action = "stop_recording_boundary_censored"
                next_target = None
            elif level_index == 3:
                unresolved_action = "stop_search_cap_censored"
            else:
                unresolved_action = "expand_to_next_level"
            if planned["next_level_decision"] != {
                "decision_status": "pending_signal_state_closure_evaluation",
                "state_segmentation_already_completed": False,
                "if_onset_and_termination_state_closed": (
                    "stop_and_derive_variable_event_window"
                ),
                "if_state_unresolved": unresolved_action,
                "current_target_span_seconds": expected_target,
                "next_target_span_seconds": next_target,
            }:
                raise ValueError("progressive embedded next-level decision drifted")
            preprocessing = validate_adaptive_preprocessing_receipt(
                level["preprocessing_receipt"]
            )
            search = validate_adaptive_search_receipt(
                level["adaptive_search_receipt"]
            )
            interval = planned["effective_interval_recording_seconds"]
            if (
                not _same_pair(
                    preprocessing["requested_interval_recording_seconds"], interval
                )
                or not _same_pair(
                    search["envelope_interval_recording_seconds"], interval
                )
                or search["processed_envelope_sha256"]
                != preprocessing["processed_envelope_sha256"]
                or search["preprocessing_receipt_sha256"]
                != preprocessing["receipt_sha256"]
                or level["search_status"] != search["status"]
                or preprocessing["source_signal_sha256"]
                != data["source_signal_sha256"]
                or abs(
                    float(search["coarse_anchor_recording_seconds"]) - anchor
                )
                > 1e-6
            ):
                raise ValueError("progressive executed signal/search binding drifted")
            decision = level["execution_decision"]
            last = level_index == len(levels)
            if search["status"] == "qualified_complete":
                expected_decision = "stop_qualified_complete"
            elif markers["recording_fully_exhausted"]:
                expected_decision = "stop_recording_boundary_censored"
            elif level["level_index"] == 3:
                expected_decision = "stop_search_cap_censored"
            else:
                expected_decision = "expand_to_next_level"
            if decision != expected_decision:
                raise ValueError("progressive execution decision drifted")
            if not last and decision != "expand_to_next_level":
                raise ValueError("progressive execution continued after a stop")
            if last and decision == "expand_to_next_level":
                raise ValueError("progressive execution lacks a terminal decision")
        final = levels[-1]
        if (
            event["terminal_search_status"] != final["search_status"]
            or event["terminal_decision"] != final["execution_decision"]
        ):
            raise ValueError("progressive terminal event binding drifted")
        expected_terminal = (
            "completed_qualified_state_closed"
            if event["terminal_decision"] == "stop_qualified_complete"
            else "completed_censored_state_unresolved"
        )
        if event["terminal_status"] != expected_terminal:
            raise ValueError("progressive terminal status drifted")
        expected_censoring = _terminal_censoring(
            search_status=event["terminal_search_status"],
            markers=final["planned_level"]["boundary_and_overlap_markers"],
            terminal_decision=event["terminal_decision"],
        )
        if event["censoring"] != expected_censoring:
            raise ValueError("progressive event censoring drifted")

    expected_summary = {
        "qualified_complete_event_count": sum(
            event["terminal_status"] == "completed_qualified_state_closed"
            for event in events
        ),
        "censored_unresolved_event_count": sum(
            event["terminal_status"] == "completed_censored_state_unresolved"
            for event in events
        ),
        "executed_level_count": sum(event["executed_level_count"] for event in events),
    }
    if data["summary"] != expected_summary:
        raise ValueError("progressive state-search summary drifted")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotation_api_called": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "neighbor_midpoint_clipping_used": False,
        "silent_padding_used": False,
        "default_batch_integration_enabled": False,
        "decoded_event_is_confirmed_seizure": False,
        "navigation_anchor_is_confirmed_onset": False,
        "joint_state_segmentation_claimed_complete": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("progressive state-search violated EEG-only scope")
    digest = deepcopy(data)
    digest["execution_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "PROGRESSIVE-EXEC-" + _canonical_sha256(digest)[:20]
    if data["execution_id"] != expected_id:
        raise ValueError("progressive state-search execution ID does not bind content")
    return data


__all__ = [
    "PROGRESSIVE_STATE_SEARCH_EXECUTION_METHOD_ID",
    "PROGRESSIVE_STATE_SEARCH_EXECUTION_SCHEMA_VERSION",
    "execute_progressive_eeg_state_search",
    "validate_progressive_eeg_state_search_execution",
]
