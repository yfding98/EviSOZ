"""Derive a per-event variable analysis window from adaptive EEG search.

The long-recording detector supplies only a navigation anchor.  The adaptive
search receipt may then identify an earlier sustained scalp transition and,
when visible, a later return transition.  This module turns those signal-only
coordinates into the window used for event Findings and waveform rendering.

The window is deliberately *not* the legacy ``[-12,+48]`` carrier.  It keeps
an event-specific pre-transition baseline, the complete observed evolution,
and a bounded recovery tail.  Missing recording context is represented as
censoring and never silently padded.  The output remains an algorithmic scalp
EEG analysis interval, not a confirmed seizure or cortical SOZ boundary.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from .adaptive_search import validate_adaptive_search_receipt


ADAPTIVE_EVENT_WINDOW_SCHEMA_VERSION = "adaptive_eeg_event_analysis_window_v2"
ADAPTIVE_EVENT_WINDOW_METHOD_ID = "adaptive_transition_bounded_variable_window_v2"

_WINDOW_STATUSES = {
    "complete_variable_window",
    "right_censored_variable_window",
    "left_censored_nonlocalizing_window",
    "transition_unavailable",
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


def _clip_pair(
    pair: list[float], *, lower: float, upper: float
) -> list[float] | None:
    start = max(lower, float(pair[0]))
    stop = min(upper, float(pair[1]))
    return [start, stop] if stop > start else None


def derive_adaptive_event_analysis_window(
    adaptive_search_receipt: object,
    *,
    minimum_pre_onset_seconds: float = 8.0,
    maximum_pre_onset_seconds: float = 60.0,
    recovery_seconds: float = 10.0,
    maximum_window_seconds: float = 240.0,
) -> dict[str, Any]:
    """Create one EEG-only, event-specific analysis-window receipt.

    A qualified or right-censored transition retains its refined onset.  A
    left-censored transition may still be displayed and described, but is
    explicitly ineligible for onset localization.  Search failures do not
    invent an event interval.
    """

    search = validate_adaptive_search_receipt(adaptive_search_receipt)
    minimum_pre = _finite(minimum_pre_onset_seconds, "minimum_pre_onset_seconds")
    maximum_pre = _finite(maximum_pre_onset_seconds, "maximum_pre_onset_seconds")
    recovery = _finite(recovery_seconds, "recovery_seconds")
    maximum_window = _finite(maximum_window_seconds, "maximum_window_seconds")
    if (
        minimum_pre <= 0
        or maximum_pre < minimum_pre
        or recovery < 0
        or maximum_window <= maximum_pre
    ):
        raise ValueError("adaptive variable-window policy is invalid")

    envelope_start, envelope_stop = map(
        float, search["envelope_interval_recording_seconds"]
    )
    anchor = float(search["coarse_anchor_recording_seconds"])
    baseline_relative = search["baseline_interval_relative_to_anchor_seconds"]
    baseline_recording = (
        [anchor + float(baseline_relative[0]), anchor + float(baseline_relative[1])]
        if baseline_relative is not None
        else None
    )
    critical = search["critical_transition"]
    search_status = str(search["status"])

    analysis_interval: list[float] | None = None
    analysis_relative: list[float] | None = None
    core_interval: list[float | None] | None = None
    baseline_used: list[float] | None = None
    recovery_interval: list[float] | None = None
    refined_onset: float | None = None
    observed_stop: float | None = None
    left_censored = False
    right_censored = False
    onset_localization_eligible = False
    findings_eligible = False

    if search_status == "partial_left_boundary":
        # The transition touches the left search boundary.  Preserve the
        # available signal for review but never manufacture an onset time.
        status = "left_censored_nonlocalizing_window"
        analysis_interval = [envelope_start, envelope_stop]
        if analysis_interval[1] - analysis_interval[0] > maximum_window:
            analysis_interval[0] = analysis_interval[1] - maximum_window
        left_censored = True
        findings_eligible = True
    elif critical is not None and critical.get(
        "start_offset_seconds_relative_to_anchor"
    ) is not None:
        refined_onset = anchor + float(
            critical["start_offset_seconds_relative_to_anchor"]
        )
        raw_stop = critical.get("stop_offset_seconds_relative_to_anchor")
        observed_stop = anchor + float(raw_stop) if raw_stop is not None else None

        baseline_floor = (
            float(baseline_recording[0])
            if baseline_recording is not None
            else refined_onset - maximum_pre
        )
        window_start = max(
            envelope_start,
            refined_onset - maximum_pre,
            baseline_floor,
        )
        # Retain at least the minimum immediate pre-onset context whenever it
        # exists, even if the far-baseline interval ended earlier.
        window_start = min(window_start, refined_onset - minimum_pre)
        window_start = max(envelope_start, window_start)

        if observed_stop is None:
            window_stop = envelope_stop
            right_censored = True
        else:
            requested_stop = observed_stop + recovery
            window_stop = min(envelope_stop, requested_stop)
            right_censored = window_stop < requested_stop - 1e-6
            if window_stop > observed_stop:
                recovery_interval = [observed_stop, window_stop]

        if window_stop - window_start > maximum_window:
            # Preserve the refined onset and post-onset evolution.  Long
            # pre-onset context is the first part to be clipped.
            window_start = max(window_start, window_stop - maximum_window)
        analysis_interval = [window_start, window_stop]
        analysis_relative = [
            window_start - refined_onset,
            window_stop - refined_onset,
        ]
        core_interval = [refined_onset, observed_stop]
        if baseline_recording is not None:
            baseline_used = _clip_pair(
                [float(baseline_recording[0]), float(baseline_recording[1])],
                lower=window_start,
                upper=min(window_stop, refined_onset),
            )
        available_pre = refined_onset - window_start
        onset_localization_eligible = available_pre >= minimum_pre - 1e-6
        left_censored = not onset_localization_eligible
        findings_eligible = True
        if left_censored:
            status = "left_censored_nonlocalizing_window"
        elif right_censored:
            status = "right_censored_variable_window"
        else:
            status = "complete_variable_window"
    else:
        status = "transition_unavailable"

    body: dict[str, Any] = {
        "schema_version": ADAPTIVE_EVENT_WINDOW_SCHEMA_VERSION,
        "window_receipt_id": "CONTENT-ADDRESS-PENDING",
        "method_id": ADAPTIVE_EVENT_WINDOW_METHOD_ID,
        "source_search_receipt_id": search["search_receipt_id"],
        "source_search_receipt_sha256": _canonical_sha256(search),
        "status": status,
        "analysis_interval_recording_seconds": analysis_interval,
        "analysis_interval_relative_to_refined_onset_seconds": analysis_relative,
        "analysis_core_recording_seconds": core_interval,
        "baseline_context_recording_seconds": baseline_used,
        "recovery_context_recording_seconds": recovery_interval,
        "duration_seconds": (
            analysis_interval[1] - analysis_interval[0]
            if analysis_interval is not None
            else None
        ),
        "censoring": {
            "left": left_censored,
            "right": right_censored,
            "termination_observed": observed_stop is not None,
        },
        "eligibility": {
            "signal_findings": findings_eligible,
            "onset_localization": onset_localization_eligible,
            "research_channel_ranking": onset_localization_eligible,
        },
        "policy": {
            "minimum_pre_onset_seconds": minimum_pre,
            "maximum_pre_onset_seconds": maximum_pre,
            "recovery_seconds": recovery,
            "maximum_window_seconds": maximum_window,
            "legacy_fixed_minus12_plus48_used": False,
            "silent_padding_used": False,
        },
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used": False,
            "coarse_anchor_is_confirmed_onset": False,
            "refined_transition_is_confirmed_seizure": False,
        },
    }
    body["window_receipt_id"] = (
        "ADAPT-WINDOW-" + _canonical_sha256(body)[:20]
    )
    return validate_adaptive_event_analysis_window(body)


def validate_adaptive_event_analysis_window(payload: object) -> dict[str, Any]:
    """Strictly validate a variable event-analysis-window receipt."""

    if type(payload) is not dict:
        raise TypeError("adaptive event window must be an object")
    required = {
        "schema_version",
        "window_receipt_id",
        "method_id",
        "source_search_receipt_id",
        "source_search_receipt_sha256",
        "status",
        "analysis_interval_recording_seconds",
        "analysis_interval_relative_to_refined_onset_seconds",
        "analysis_core_recording_seconds",
        "baseline_context_recording_seconds",
        "recovery_context_recording_seconds",
        "duration_seconds",
        "censoring",
        "eligibility",
        "policy",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("adaptive event window has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_EVENT_WINDOW_SCHEMA_VERSION:
        raise ValueError("adaptive event window schema drifted")
    if data["method_id"] != ADAPTIVE_EVENT_WINDOW_METHOD_ID:
        raise ValueError("adaptive event window method drifted")
    if data["status"] not in _WINDOW_STATUSES:
        raise ValueError("adaptive event window status is unsupported")
    source_hash = data["source_search_receipt_sha256"]
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("adaptive event window source hash is invalid")

    def pair_or_none(value: object, context: str) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError(f"{context} must be null or a two-number array")
        if value[1] is None and context == "analysis core":
            start = _finite(value[0], context)
            return [start, None]  # type: ignore[list-item]
        pair = [_finite(item, context) for item in value]
        if pair[1] <= pair[0]:
            raise ValueError(f"{context} must have positive duration")
        return pair

    interval = pair_or_none(data["analysis_interval_recording_seconds"], "analysis interval")
    relative = pair_or_none(
        data["analysis_interval_relative_to_refined_onset_seconds"],
        "relative analysis interval",
    )
    core = pair_or_none(data["analysis_core_recording_seconds"], "analysis core")
    baseline = pair_or_none(data["baseline_context_recording_seconds"], "baseline context")
    recovery = pair_or_none(data["recovery_context_recording_seconds"], "recovery context")
    duration = data["duration_seconds"]
    if interval is None:
        if data["status"] != "transition_unavailable" or any(
            value is not None for value in (relative, core, baseline, recovery, duration)
        ):
            raise ValueError("unavailable transition must not claim a window")
    else:
        measured = interval[1] - interval[0]
        if abs(_finite(duration, "duration_seconds") - measured) > 1e-6:
            raise ValueError("adaptive event window duration does not close")
        if core is not None and relative is not None:
            onset = float(core[0])
            if (
                abs(relative[0] - (interval[0] - onset)) > 1e-6
                or abs(relative[1] - (interval[1] - onset)) > 1e-6
            ):
                raise ValueError("adaptive event window relative timebase drifted")
        for context_pair in (baseline, recovery):
            if context_pair is not None and (
                context_pair[0] < interval[0] - 1e-6
                or context_pair[1] > interval[1] + 1e-6
            ):
                raise ValueError("adaptive event context lies outside its window")

    censoring = data["censoring"]
    if type(censoring) is not dict or set(censoring) != {
        "left",
        "right",
        "termination_observed",
    } or any(type(censoring[key]) is not bool for key in censoring):
        raise ValueError("adaptive event censoring receipt is invalid")
    eligibility = data["eligibility"]
    if type(eligibility) is not dict or set(eligibility) != {
        "signal_findings",
        "onset_localization",
        "research_channel_ranking",
    } or any(type(eligibility[key]) is not bool for key in eligibility):
        raise ValueError("adaptive event eligibility receipt is invalid")
    if eligibility["research_channel_ranking"] and not eligibility["onset_localization"]:
        raise ValueError("channel ranking cannot exceed onset eligibility")
    if data["status"] == "left_censored_nonlocalizing_window" and (
        not censoring["left"] or eligibility["onset_localization"]
    ):
        raise ValueError("left-censored window cannot localize onset")
    if data["status"] == "right_censored_variable_window" and not censoring["right"]:
        raise ValueError("right-censored status requires right censoring")

    policy = data["policy"]
    expected_policy_keys = {
        "minimum_pre_onset_seconds",
        "maximum_pre_onset_seconds",
        "recovery_seconds",
        "maximum_window_seconds",
        "legacy_fixed_minus12_plus48_used",
        "silent_padding_used",
    }
    if type(policy) is not dict or set(policy) != expected_policy_keys:
        raise ValueError("adaptive event window policy is invalid")
    if policy["legacy_fixed_minus12_plus48_used"] is not False or policy[
        "silent_padding_used"
    ] is not False:
        raise ValueError("adaptive event window reverted to fixed/padded semantics")
    if interval is not None and interval[1] - interval[0] > float(
        policy["maximum_window_seconds"]
    ) + 1e-6:
        raise ValueError("adaptive event window exceeds its maximum")

    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used": False,
        "coarse_anchor_is_confirmed_onset": False,
        "refined_transition_is_confirmed_seizure": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("adaptive event window violates the EEG-only boundary")
    digest_source = deepcopy(data)
    digest_source["window_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "ADAPT-WINDOW-" + _canonical_sha256(digest_source)[:20]
    if data["window_receipt_id"] != expected_id:
        raise ValueError("adaptive event window ID does not bind its content")
    return data


__all__ = [
    "ADAPTIVE_EVENT_WINDOW_METHOD_ID",
    "ADAPTIVE_EVENT_WINDOW_SCHEMA_VERSION",
    "derive_adaptive_event_analysis_window",
    "validate_adaptive_event_analysis_window",
]
