"""Continuous long-EEG seizure-posterior decoding.

This module is the model-agnostic boundary between a trained seizure detector
and the long-recording report pipeline.  A detector must scan the complete
recording and emit a dense posterior timeline.  Hysteresis converts that
timeline into event proposals without forcing a minimum number of alarms.

The decoded anchor is only a navigation coordinate for the later adaptive
EEG transition search.  It is not a confirmed electrographic onset.  The
decoder accepts neither EDF annotations nor spreadsheet/clinical labels.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CONTINUOUS_DETECTION_SCHEMA_VERSION = "continuous_seizure_posterior_decoding_v1"
CONTINUOUS_DETECTION_METHOD_ID = "dense_posterior_hysteresis_event_decoder_v1"

DEFAULT_DECODER_POLICY: dict[str, Any] = {
    "on_threshold": 0.70,
    "off_threshold": 0.30,
    "minimum_on_windows": 2,
    "minimum_off_windows": 3,
    "merge_gap_seconds": 5.0,
    "maximum_coverage_gap_seconds": 2.01,
    "minimum_event_seconds": 2.0,
    "force_minimum_candidate_count": False,
    "anchor_semantics": "first_persistent_on_threshold_navigation_coordinate",
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


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{context} must be a positive integer")
    return int(value)


def _validate_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(dict(DEFAULT_DECODER_POLICY if policy is None else policy))
    if set(result) != set(DEFAULT_DECODER_POLICY):
        raise ValueError("continuous detector policy has missing or unknown fields")
    on = _finite(result["on_threshold"], "on_threshold")
    off = _finite(result["off_threshold"], "off_threshold")
    if not 0 <= off < on <= 1:
        raise ValueError("continuous detector thresholds must satisfy 0<=off<on<=1")
    _positive_int(result["minimum_on_windows"], "minimum_on_windows")
    _positive_int(result["minimum_off_windows"], "minimum_off_windows")
    for name in (
        "merge_gap_seconds",
        "maximum_coverage_gap_seconds",
        "minimum_event_seconds",
    ):
        if _finite(result[name], name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if result["force_minimum_candidate_count"] is not False:
        raise ValueError("continuous event decoder must not force candidates")
    if result["anchor_semantics"] != (
        "first_persistent_on_threshold_navigation_coordinate"
    ):
        raise ValueError("continuous detector anchor semantics drifted")
    return result


def _validate_timeline(
    timeline: Sequence[Mapping[str, Any]],
    *,
    recording_duration_seconds: float,
    maximum_gap_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(timeline, Sequence) or isinstance(timeline, (str, bytes)):
        raise TypeError("posterior timeline must be a sequence")
    if not timeline:
        raise ValueError("posterior timeline must not be empty")
    rows: list[dict[str, Any]] = []
    previous_start = -math.inf
    previous_stop = 0.0
    window_ids: set[str] = set()
    required = {
        "window_id",
        "start_offset_seconds",
        "stop_offset_seconds",
        "seizure_probability",
        "signal_usable",
    }
    for index, raw in enumerate(timeline):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"posterior timeline row {index} has invalid fields")
        window_id = raw["window_id"]
        if not isinstance(window_id, str) or not window_id or window_id in window_ids:
            raise ValueError("posterior timeline window IDs must be unique strings")
        window_ids.add(window_id)
        start = _finite(raw["start_offset_seconds"], "timeline start")
        stop = _finite(raw["stop_offset_seconds"], "timeline stop")
        probability = _finite(raw["seizure_probability"], "seizure probability")
        if (
            not 0 <= start < stop <= recording_duration_seconds + 1e-6
            or not 0 <= probability <= 1
            or start < previous_start - 1e-9
        ):
            raise ValueError("posterior timeline time/probability is invalid")
        if index and start - previous_stop > maximum_gap_seconds + 1e-9:
            raise ValueError("posterior timeline has an unaudited coverage gap")
        if type(raw["signal_usable"]) is not bool:
            raise TypeError("timeline signal_usable must be boolean")
        rows.append(
            {
                "window_id": window_id,
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "seizure_probability": probability,
                "signal_usable": raw["signal_usable"],
            }
        )
        previous_start = start
        previous_stop = max(previous_stop, stop)
    if rows[0]["start_offset_seconds"] > maximum_gap_seconds + 1e-9:
        raise ValueError("posterior timeline does not cover recording start")
    if recording_duration_seconds - previous_stop > maximum_gap_seconds + 1e-9:
        raise ValueError("posterior timeline does not cover recording stop")
    return rows


def _candidate_from_rows(
    active_rows: list[dict[str, Any]],
    *,
    right_censored: bool,
) -> dict[str, Any]:
    probabilities = [float(row["seizure_probability"]) for row in active_rows]
    first_on = active_rows[0]
    return {
        "start_offset_seconds": float(active_rows[0]["start_offset_seconds"]),
        "stop_offset_seconds": float(active_rows[-1]["stop_offset_seconds"]),
        "anchor_offset_seconds": float(first_on["start_offset_seconds"]),
        "peak_probability": max(probabilities),
        "mean_probability": sum(probabilities) / len(probabilities),
        "right_censored": bool(right_censored),
        "support_window_ids": [str(row["window_id"]) for row in active_rows],
    }


def decode_continuous_seizure_posterior(
    *,
    recording_id: str,
    source_signal_sha256: str,
    recording_duration_seconds: float,
    provider_receipt: Mapping[str, Any],
    posterior_timeline: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a full-record posterior timeline into coarse event proposals."""

    duration = _finite(recording_duration_seconds, "recording_duration_seconds")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("recording_id must be a non-empty string")
    if (
        not isinstance(source_signal_sha256, str)
        or len(source_signal_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_signal_sha256)
    ):
        raise ValueError("source signal SHA-256 is invalid")
    provider = deepcopy(dict(provider_receipt))
    required_provider = {
        "provider_id",
        "model_family",
        "checkpoint_sha256",
        "code_sha256",
        "training_corpus",
        "posterior_calibration_status",
        "deployment_qualification_status",
        "annotations_used_for_current_recording",
        "labels_used_for_current_recording",
    }
    if set(provider) != required_provider:
        raise ValueError("continuous detector provider receipt is invalid")
    if provider["annotations_used_for_current_recording"] is not False or provider[
        "labels_used_for_current_recording"
    ] is not False:
        raise ValueError("continuous detector violated EEG-only inference")
    decoder_policy = _validate_policy(policy)
    rows = _validate_timeline(
        posterior_timeline,
        recording_duration_seconds=duration,
        maximum_gap_seconds=float(decoder_policy["maximum_coverage_gap_seconds"]),
    )

    on_threshold = float(decoder_policy["on_threshold"])
    off_threshold = float(decoder_policy["off_threshold"])
    minimum_on = int(decoder_policy["minimum_on_windows"])
    minimum_off = int(decoder_policy["minimum_off_windows"])

    candidates: list[dict[str, Any]] = []
    active = False
    pending_on: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    pending_off: list[dict[str, Any]] = []
    for row in rows:
        usable = bool(row["signal_usable"])
        probability = float(row["seizure_probability"])
        if not active:
            if usable and probability >= on_threshold:
                pending_on.append(row)
                if len(pending_on) >= minimum_on:
                    active = True
                    active_rows = list(pending_on)
                    pending_on = []
            else:
                pending_on = []
            continue

        active_rows.append(row)
        if (not usable) or probability <= off_threshold:
            pending_off.append(row)
        else:
            pending_off = []
        if len(pending_off) >= minimum_off:
            # Off-threshold rows establish the end but are not themselves part
            # of the posterior-positive support interval.
            active_rows = active_rows[: -len(pending_off)]
            if active_rows:
                candidates.append(
                    _candidate_from_rows(active_rows, right_censored=False)
                )
            active = False
            active_rows = []
            pending_off = []
    if active and active_rows:
        candidates.append(_candidate_from_rows(active_rows, right_censored=True))

    minimum_event = float(decoder_policy["minimum_event_seconds"])
    candidates = [
        item
        for item in candidates
        if item["stop_offset_seconds"] - item["start_offset_seconds"]
        >= minimum_event - 1e-9
    ]
    merged: list[dict[str, Any]] = []
    merge_gap = float(decoder_policy["merge_gap_seconds"])
    for candidate in candidates:
        if not merged or (
            candidate["start_offset_seconds"] - merged[-1]["stop_offset_seconds"]
            > merge_gap
        ):
            merged.append(deepcopy(candidate))
            continue
        previous = merged[-1]
        previous["stop_offset_seconds"] = max(
            previous["stop_offset_seconds"], candidate["stop_offset_seconds"]
        )
        previous["peak_probability"] = max(
            previous["peak_probability"], candidate["peak_probability"]
        )
        all_ids = previous["support_window_ids"] + candidate["support_window_ids"]
        previous["support_window_ids"] = list(dict.fromkeys(all_ids))
        previous["mean_probability"] = (
            previous["mean_probability"] + candidate["mean_probability"]
        ) / 2.0
        previous["right_censored"] = bool(
            previous["right_censored"] or candidate["right_censored"]
        )

    for index, candidate in enumerate(merged, start=1):
        candidate["candidate_id"] = f"MODEL-EVENT-{index:04d}"
        candidate["candidate_semantics"] = (
            "model_detected_event_proposal_not_confirmed_seizure_or_onset"
        )

    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_DETECTION_SCHEMA_VERSION,
        "decoding_receipt_id": "CONTENT-ADDRESS-PENDING",
        "method_id": CONTINUOUS_DETECTION_METHOD_ID,
        "recording_id": recording_id,
        "source_signal_sha256": source_signal_sha256,
        "recording_duration_seconds": duration,
        "provider_receipt": provider,
        "decoder_policy": decoder_policy,
        "timeline_window_count": len(rows),
        "usable_timeline_window_count": sum(row["signal_usable"] for row in rows),
        "event_proposal_count": len(merged),
        "event_proposals": merged,
        "coverage_receipt": {
            "complete_recording_scanned": True,
            "maximum_observed_gap_seconds": max(
                [rows[0]["start_offset_seconds"]]
                + [
                    max(
                        0.0,
                        rows[index]["start_offset_seconds"]
                        - rows[index - 1]["stop_offset_seconds"],
                    )
                    for index in range(1, len(rows))
                ]
                + [max(0.0, duration - rows[-1]["stop_offset_seconds"])]
            ),
            "forced_candidate_count": False,
            "zero_candidates_is_valid": True,
        },
        "scope_receipt": {
            "eeg_signal_only": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_context_used": False,
            "labels_or_ground_truth_used_for_current_recording": False,
            "event_anchor_is_confirmed_onset": False,
        },
    }
    body["decoding_receipt_id"] = "CONT-DETECT-" + _canonical_sha256(body)[:20]
    return validate_continuous_seizure_decoding(body)


def validate_continuous_seizure_decoding(payload: object) -> dict[str, Any]:
    """Validate the content binding and EEG-only boundary of a decode receipt."""

    if type(payload) is not dict:
        raise TypeError("continuous decoding receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "decoding_receipt_id",
        "method_id",
        "recording_id",
        "source_signal_sha256",
        "recording_duration_seconds",
        "provider_receipt",
        "decoder_policy",
        "timeline_window_count",
        "usable_timeline_window_count",
        "event_proposal_count",
        "event_proposals",
        "coverage_receipt",
        "scope_receipt",
    }
    if set(data) != required:
        raise ValueError("continuous decoding receipt has missing or unknown fields")
    if data["schema_version"] != CONTINUOUS_DETECTION_SCHEMA_VERSION or data[
        "method_id"
    ] != CONTINUOUS_DETECTION_METHOD_ID:
        raise ValueError("continuous decoding schema/method drifted")
    _validate_policy(data["decoder_policy"])
    if data["event_proposal_count"] != len(data["event_proposals"]):
        raise ValueError("continuous event proposal count drifted")
    if data["timeline_window_count"] < data["usable_timeline_window_count"]:
        raise ValueError("usable posterior count exceeds timeline count")
    if data["coverage_receipt"].get("forced_candidate_count") is not False or data[
        "coverage_receipt"
    ].get("zero_candidates_is_valid") is not True:
        raise ValueError("continuous decoder reintroduced forced candidates")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "clinical_context_used": False,
        "labels_or_ground_truth_used_for_current_recording": False,
        "event_anchor_is_confirmed_onset": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("continuous decoding violates the EEG-only boundary")
    digest = deepcopy(data)
    digest["decoding_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "CONT-DETECT-" + _canonical_sha256(digest)[:20]
    if data["decoding_receipt_id"] != expected_id:
        raise ValueError("continuous decoding ID does not bind its content")
    return data


__all__ = [
    "CONTINUOUS_DETECTION_METHOD_ID",
    "CONTINUOUS_DETECTION_SCHEMA_VERSION",
    "DEFAULT_DECODER_POLICY",
    "decode_continuous_seizure_posterior",
    "validate_continuous_seizure_decoding",
]
