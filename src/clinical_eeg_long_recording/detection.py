"""Provider-neutral construction of a de-identified detector manifest.

This module does not implement or bless a particular seizure detector.  A
provider supplies frozen alarm support intervals and a strict detector receipt;
the code applies only the frozen threshold, deterministic merging, boundary
abstention and receipt construction required by the recording pipeline.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .schema import (
    CANDIDATE_SEMANTICS,
    DETECTION_MANIFEST_SCHEMA_VERSION,
    FIXED_EVENT_WINDOW_SECONDS,
    MINIMUM_ANALYZABLE_ANCHOR_SECONDS,
    canonical_payload_sha256,
    validate_long_term_seizure_detection_manifest,
)


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _identifier_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise TypeError(f"{context} values must be non-empty identifiers")
        if item in result:
            raise ValueError(f"{context} contains duplicates")
        result.append(item)
    return result


def _canonical_id(prefix: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _passes(score: float, threshold: float, direction: str) -> bool:
    if direction == "greater_or_equal":
        return score >= threshold
    if direction == "less_or_equal":
        return score <= threshold
    raise ValueError("detector score direction is unsupported")


def build_long_term_detection_manifest(
    *,
    recording_id: str,
    patient_pseudonym: str,
    source_signal_sha256: str,
    recording_duration_seconds: float,
    detector_receipt: Mapping[str, Any],
    raw_alarm_observations: Sequence[Mapping[str, Any]],
    merge_gap_seconds: float,
    max_selected_candidates: int | None = None,
) -> dict[str, Any]:
    """Threshold and merge provider alarms into strict review candidates.

    Every raw observation must contain exactly:

    ``start_offset_seconds``, ``stop_offset_seconds``,
    ``anchor_offset_seconds``, ``score``,
    ``decision_available_offset_seconds`` and ``support_window_ids``.

    The latter are de-identified scan-window receipt IDs, not EDF annotation or
    event-label IDs.  The selected output remains a review queue and never a
    seizure diagnosis.
    """

    duration = _number(recording_duration_seconds, "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    gap = _number(merge_gap_seconds, "merge gap")
    if gap < 0:
        raise ValueError("merge gap must be non-negative")
    if max_selected_candidates is not None and (
        isinstance(max_selected_candidates, bool)
        or not isinstance(max_selected_candidates, int)
        or max_selected_candidates < 1
    ):
        raise ValueError("max_selected_candidates must be a positive integer")
    receipt = deepcopy(dict(_object(detector_receipt, "detector receipt")))
    operating = _object(receipt.get("operating_point"), "detector operating point")
    threshold = _number(operating.get("threshold"), "detector threshold")
    direction = operating.get("score_direction")
    if direction not in {"greater_or_equal", "less_or_equal"}:
        raise ValueError("detector score direction is unsupported")

    exact_keys = {
        "start_offset_seconds",
        "stop_offset_seconds",
        "anchor_offset_seconds",
        "score",
        "decision_available_offset_seconds",
        "support_window_ids",
    }
    alarms: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_alarm_observations, start=1):
        item = _object(raw, f"raw alarm observation {ordinal}")
        if set(item) != exact_keys:
            raise ValueError(
                f"raw alarm observation {ordinal} has missing or unknown fields"
            )
        start = _number(item["start_offset_seconds"], "alarm support start")
        stop = _number(item["stop_offset_seconds"], "alarm support stop")
        anchor = _number(item["anchor_offset_seconds"], "alarm anchor")
        score = _number(item["score"], "alarm score")
        available_offset = _number(
            item["decision_available_offset_seconds"], "alarm decision time"
        )
        support_ids = _identifier_list(
            item["support_window_ids"], "alarm support window IDs"
        )
        if start < 0 or stop > duration or stop <= start:
            raise ValueError("raw alarm support interval is outside the recording")
        if anchor < start or anchor > stop:
            raise ValueError("raw alarm anchor is outside its support interval")
        if available_offset < stop or available_offset > duration:
            raise ValueError("raw alarm decision time is outside its causal range")
        identity = {
            "recording_id": recording_id,
            "ordinal": ordinal,
            "start": start,
            "stop": stop,
            "anchor": anchor,
            "support_window_ids": support_ids,
        }
        retained = _passes(score, threshold, str(direction))
        alarms.append(
            {
                "alarm_id": _canonical_id("ALARM", identity),
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "anchor_offset_seconds": anchor,
                "score": score,
                "support_count": len(support_ids),
                "decision_available": True,
                "decision_available_offset_seconds": available_offset,
                "decision": (
                    "retained_for_merge" if retained else "rejected_below_threshold"
                ),
                "member_ids": support_ids,
                "semantics": CANDIDATE_SEMANTICS,
            }
        )
    alarms.sort(
        key=lambda item: (
            item["start_offset_seconds"],
            item["anchor_offset_seconds"],
            item["alarm_id"],
        )
    )

    retained = [item for item in alarms if item["decision"] == "retained_for_merge"]
    groups: list[list[dict[str, Any]]] = []
    for alarm in retained:
        if not groups or float(alarm["start_offset_seconds"]) > max(
            float(item["stop_offset_seconds"]) for item in groups[-1]
        ) + gap:
            groups.append([alarm])
        else:
            groups[-1].append(alarm)

    provisional: list[dict[str, Any]] = []
    for group in groups:
        best = (
            max(group, key=lambda item: (float(item["score"]), -float(item["anchor_offset_seconds"])))
            if direction == "greater_or_equal"
            else min(group, key=lambda item: (float(item["score"]), float(item["anchor_offset_seconds"])))
        )
        start = min(float(item["start_offset_seconds"]) for item in group)
        stop = max(float(item["stop_offset_seconds"]) for item in group)
        anchor = float(best["anchor_offset_seconds"])
        members = [str(item["alarm_id"]) for item in group]
        available_offset = max(
            float(item["decision_available_offset_seconds"]) for item in group
        )
        identity = {
            "recording_id": recording_id,
            "member_ids": members,
            "anchor": anchor,
        }
        boundary_ok = (
            anchor >= MINIMUM_ANALYZABLE_ANCHOR_SECONDS
            and duration - anchor >= FIXED_EVENT_WINDOW_SECONDS[1]
            and start - anchor >= FIXED_EVENT_WINDOW_SECONDS[0]
            and stop - anchor <= FIXED_EVENT_WINDOW_SECONDS[1]
        )
        provisional.append(
            {
                "candidate_id": _canonical_id("CAND", identity),
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "anchor_offset_seconds": anchor,
                "score": float(best["score"]),
                "support_count": len(members),
                "decision_available": True,
                "decision_available_offset_seconds": available_offset,
                "decision": (
                    "selected_for_event_analysis"
                    if boundary_ok
                    else "rejected_insufficient_fixed_window"
                ),
                "member_ids": members,
                "semantics": CANDIDATE_SEMANTICS,
            }
        )
    eligible = [
        item for item in provisional if item["decision"] == "selected_for_event_analysis"
    ]
    if max_selected_candidates is not None and len(eligible) > max_selected_candidates:
        priority = sorted(
            eligible,
            key=(
                (lambda item: (-float(item["score"]), float(item["anchor_offset_seconds"])))
                if direction == "greater_or_equal"
                else (lambda item: (float(item["score"]), float(item["anchor_offset_seconds"])))
            ),
        )
        selected_ids = {
            item["candidate_id"] for item in priority[:max_selected_candidates]
        }
        for item in provisional:
            if (
                item["decision"] == "selected_for_event_analysis"
                and item["candidate_id"] not in selected_ids
            ):
                item["decision"] = "review_only_not_segmented"

    provisional.sort(
        key=lambda item: (
            item["start_offset_seconds"],
            item["anchor_offset_seconds"],
            item["candidate_id"],
        )
    )
    body = {
        "schema_version": DETECTION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "MANIFEST-PENDING",
        "recording_id": recording_id,
        "patient_pseudonym": patient_pseudonym,
        "source_signal_sha256": source_signal_sha256,
        "recording_duration_seconds": duration,
        "scan_coverage_intervals": [
            {"start_offset_seconds": 0.0, "stop_offset_seconds": duration}
        ],
        "detector_receipt": receipt,
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "raw_alarms": alarms,
        "merge_candidates": provisional,
    }
    body["manifest_id"] = f"LTDET-{canonical_payload_sha256(body)[:24]}"
    return validate_long_term_seizure_detection_manifest(body)


__all__ = ["build_long_term_detection_manifest"]
