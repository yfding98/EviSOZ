"""EEG-only adaptive transition search around a coarse long-recording alarm.

The coarse detector anchor is a navigation coordinate, not a known onset.  This
module therefore accepts an arbitrary-length, already preprocessed standard-19
EEG envelope and searches for a bounded critical-change interval without using
EDF annotations, spreadsheets, clinical metadata, or ground-truth event times.

The result deliberately remains an *algorithmic scalp-EEG change candidate*.
It is not a seizure diagnosis, cortical SOZ estimate, or calibrated
probability.  Boundary contact, insufficient baseline, weak evolution,
artifact dominance, and missing termination all fail closed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.soz.geometry import STANDARD_19, TCP_20_EDGES

from .canonical_adaptive_binding import (
    validate_canonical_adaptive_signal_binding,
)


ENVELOPE_PLAN_SCHEMA_VERSION = "adaptive_eeg_search_envelope_plan_v1"
ADAPTIVE_SEARCH_SCHEMA_VERSION = "adaptive_eeg_transition_search_v2"
ADAPTIVE_SEARCH_METHOD_ID = (
    "canonical_missing_aware_multiscale_bipolar_change_point_v2"
)
ADAPTIVE_SEARCH_SCOPE = "eeg_signal_only_no_annotation_excel_or_ground_truth"

DEFAULT_PRE_SEARCH_SECONDS = 90.0
DEFAULT_POST_SEARCH_SECONDS = 120.0
DEFAULT_NEIGHBOR_GUARD_SECONDS = 2.0
DEFAULT_CAUSAL_WARMUP_SECONDS = 30.0

# The v1 detection manifest predates adaptive search and labels candidates near
# a recording boundary as ``rejected_insufficient_fixed_window`` solely because
# a 60-second v29 carrier cannot be centred on the *coarse* alarm.  Those alarms
# have nevertheless passed the detector threshold and merge policy.  Adaptive
# EEG search must retain them: only the refined anchor decides whether the
# downstream fixed carrier is available.
_ADAPTIVE_INPUT_DECISIONS = (
    "selected_for_event_analysis",
    "rejected_insufficient_fixed_window",
)

_STATUS = (
    "qualified_complete",
    "partial_left_boundary",
    "partial_right_boundary",
    "abstained_insufficient_baseline",
    "abstained_no_onset_transition",
    "abstained_no_termination_transition",
    "abstained_artifact_dominated",
    "abstained_low_confidence",
    "abstained_envelope_unavailable",
)
_FAIL_REASON = (
    "none",
    "left_boundary_contact",
    "right_boundary_contact",
    "insufficient_clean_baseline",
    "onset_transition_below_threshold",
    "termination_transition_below_threshold",
    "artifact_dominates_transition",
    "joint_confidence_below_threshold",
    "envelope_signal_unavailable",
)

ADAPTIVE_SEARCH_POLICY: dict[str, Any] = {
    "grid_step_seconds": 0.5,
    "comparison_scales_seconds": [1.0, 2.0, 4.0],
    "baseline_left_margin_seconds": 4.0,
    "baseline_exclusion_before_anchor_seconds": 8.0,
    "maximum_baseline_seconds": 30.0,
    "minimum_baseline_seconds": 8.0,
    "maximum_onset_seconds_before_anchor": 60.0,
    "maximum_onset_seconds_after_anchor": 30.0,
    "minimum_candidate_duration_seconds": 4.0,
    "onset_hysteresis_fraction": 0.82,
    "minimum_onset_run_points": 2,
    "channel_activation_z_threshold": 2.0,
    "minimum_connected_derivations": 2,
    "boundary_guard_seconds": 2.0,
    "onset_score_threshold": 0.52,
    "termination_score_threshold": 0.50,
    "persistence_fraction_threshold": 0.52,
    "joint_confidence_threshold": 0.58,
    "artifact_abstention_threshold": 0.72,
    "v29_pre_seconds": 12.0,
    "v29_post_seconds": 48.0,
    "v29_causal_warmup_seconds": 30.0,
    "feature_family": [
        "log_rms",
        "log_line_length",
        "delta_fraction",
        "theta_fraction",
        "alpha_fraction",
        "beta_fraction",
        "low_gamma_fraction",
        "spectral_entropy",
        "spectral_centroid",
        "rhythmicity",
    ],
    "bipolar_geometry": [f"{left}-{right}" for left, right in TCP_20_EDGES],
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ADAPTIVE_SEARCH_POLICY_SHA256 = _canonical_sha256(ADAPTIVE_SEARCH_POLICY)


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty identifier")
    if len(value) > 128 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} is not a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def generalized_signal_tensor_sha256(signal: torch.Tensor) -> str:
    """Hash an arbitrary-length finite standard-19 float32 envelope."""

    if not isinstance(signal, torch.Tensor):
        raise TypeError("adaptive search signal must be a torch.Tensor")
    tensor = signal.detach().cpu().to(torch.float32).contiguous()
    if tensor.ndim != 2 or tensor.shape[0] != len(STANDARD_19):
        raise ValueError("adaptive search signal must have shape [19,T]")
    if tensor.shape[1] < 2 or not torch.isfinite(tensor).all():
        raise ValueError("adaptive search signal must be non-empty and finite")
    prefix = (
        "adaptive-standard19-float32-le:"
        f"{tensor.shape[0]}x{tensor.shape[1]}:"
    ).encode("ascii")
    payload = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(prefix + payload).hexdigest()


def plan_adaptive_search_envelopes(
    detection_manifest: object,
    event_id_by_candidate: Mapping[str, str] | None = None,
    *,
    pre_search_seconds: float = DEFAULT_PRE_SEARCH_SECONDS,
    post_search_seconds: float = DEFAULT_POST_SEARCH_SECONDS,
    neighbor_guard_seconds: float = DEFAULT_NEIGHBOR_GUARD_SECONDS,
    causal_warmup_seconds: float = DEFAULT_CAUSAL_WARMUP_SECONDS,
) -> list[dict[str, Any]]:
    """Plan non-overlapping, boundary-aware envelopes for selected candidates.

    Adjacent candidates are separated at their midpoint with a small guard.
    The adaptive layer is offline and may inspect signal from recording offset
    zero.  The 30-second causal warmup belongs only to the later v29 carrier
    eligibility check; it must not erase early EEG evidence here.  This may
    yield a partial envelope, and later signal analysis decides whether enough
    clean pre-anchor baseline remains.
    """

    # Local import avoids making schema-only consumers import NumPy/Torch code.
    from .schema import validate_long_term_seizure_detection_manifest

    manifest = validate_long_term_seizure_detection_manifest(detection_manifest)
    pre = _finite(pre_search_seconds, "pre_search_seconds")
    post = _finite(post_search_seconds, "post_search_seconds")
    guard = _finite(neighbor_guard_seconds, "neighbor_guard_seconds")
    warmup = _finite(causal_warmup_seconds, "causal_warmup_seconds")
    if pre <= 0 or post <= 0 or guard < 0 or warmup < 0:
        raise ValueError("adaptive envelope durations must be positive")
    selected = [
        item
        for item in manifest["merge_candidates"]
        if item["decision_available"] is True
        and item["decision"] in _ADAPTIVE_INPUT_DECISIONS
    ]
    selected.sort(key=lambda item: (item["anchor_offset_seconds"], item["candidate_id"]))
    mapping = dict(event_id_by_candidate or {})
    if mapping and set(mapping) != {item["candidate_id"] for item in selected}:
        raise ValueError("event_id_by_candidate must exactly cover selected candidates")
    duration = float(manifest["recording_duration_seconds"])
    plans: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        anchor = float(candidate["anchor_offset_seconds"])
        requested_start = anchor - pre
        requested_stop = anchor + post
        previous = selected[index - 1] if index else None
        following = selected[index + 1] if index + 1 < len(selected) else None
        previous_midpoint = (
            (float(previous["anchor_offset_seconds"]) + anchor) / 2.0
            if previous is not None
            else None
        )
        next_midpoint = (
            (anchor + float(following["anchor_offset_seconds"])) / 2.0
            if following is not None
            else None
        )
        effective_start = max(0.0, requested_start)
        effective_stop = min(duration, requested_stop)
        reasons: list[str] = []
        if requested_start < 0:
            reasons.append("recording_start")
        if requested_stop > duration:
            reasons.append("recording_stop")
        if previous_midpoint is not None:
            neighbor_floor = previous_midpoint + guard
            if neighbor_floor > effective_start:
                effective_start = neighbor_floor
                reasons.append("previous_candidate")
        if next_midpoint is not None:
            neighbor_ceiling = next_midpoint - guard
            if neighbor_ceiling < effective_stop:
                effective_stop = neighbor_ceiling
                reasons.append("next_candidate")
        if effective_start >= anchor or effective_stop <= anchor:
            reasons.append("anchor_context_unavailable")
        plan_core = {
            "schema_version": ENVELOPE_PLAN_SCHEMA_VERSION,
            "recording_id": manifest["recording_id"],
            "patient_pseudonym": manifest["patient_pseudonym"],
            "source_signal_sha256": manifest["source_signal_sha256"],
            "recording_duration_seconds": duration,
            "candidate_id": candidate["candidate_id"],
            "source_candidate_decision": candidate["decision"],
            "analysis_route": "adaptive_eeg_search_before_fixed_v29_carrier",
            "eeg_event_id": mapping.get(
                candidate["candidate_id"], f"UNASSIGNED-{index + 1:04d}"
            ),
            "coarse_anchor_recording_seconds": anchor,
            "requested_interval_recording_seconds": [requested_start, requested_stop],
            "effective_interval_recording_seconds": [effective_start, effective_stop],
            "effective_interval_relative_to_anchor_seconds": [
                effective_start - anchor,
                effective_stop - anchor,
            ],
            "boundary_truncation_reasons": sorted(set(reasons)),
            "neighbor_constraints": {
                "previous_candidate_id": (
                    previous["candidate_id"] if previous is not None else None
                ),
                "previous_midpoint_recording_seconds": previous_midpoint,
                "next_candidate_id": (
                    following["candidate_id"] if following is not None else None
                ),
                "next_midpoint_recording_seconds": next_midpoint,
                "guard_seconds": guard,
            },
            "v29_causal_warmup_seconds": warmup,
            "scope_receipt": {
                "eeg_signal_only": True,
                "edf_annotations_read": False,
                "excel_read": False,
                "labels_or_ground_truth_read": False,
            },
        }
        plan_core["plan_id"] = f"ADAPT-PLAN-{_canonical_sha256(plan_core)[:20]}"
        plans.append(validate_adaptive_search_envelope_plan(plan_core))
    return plans


def validate_adaptive_search_envelope_plan(payload: object) -> dict[str, Any]:
    """Validate one boundary-aware, EEG-only adaptive envelope plan."""

    if type(payload) is not dict:
        raise TypeError("adaptive envelope plan must be an object")
    required = {
        "schema_version",
        "plan_id",
        "recording_id",
        "patient_pseudonym",
        "source_signal_sha256",
        "recording_duration_seconds",
        "candidate_id",
        "source_candidate_decision",
        "analysis_route",
        "eeg_event_id",
        "coarse_anchor_recording_seconds",
        "requested_interval_recording_seconds",
        "effective_interval_recording_seconds",
        "effective_interval_relative_to_anchor_seconds",
        "boundary_truncation_reasons",
        "neighbor_constraints",
        "v29_causal_warmup_seconds",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("adaptive envelope plan has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != ENVELOPE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive envelope plan schema")
    for field in (
        "plan_id",
        "recording_id",
        "patient_pseudonym",
        "candidate_id",
        "eeg_event_id",
    ):
        _identifier(data[field], f"adaptive plan {field}")
    _sha256(data["source_signal_sha256"], "adaptive plan source signal hash")
    duration = _finite(data["recording_duration_seconds"], "recording duration")
    anchor = _finite(
        data["coarse_anchor_recording_seconds"], "coarse detector anchor"
    )
    if duration <= 0 or not 0 <= anchor <= duration:
        raise ValueError("adaptive plan recording clock is invalid")

    def pair(value: object, context: str) -> list[float]:
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError(f"{context} must be a two-number array")
        return [_finite(item, context) for item in value]

    requested = pair(
        data["requested_interval_recording_seconds"], "requested interval"
    )
    effective = pair(
        data["effective_interval_recording_seconds"], "effective interval"
    )
    relative = pair(
        data["effective_interval_relative_to_anchor_seconds"],
        "relative effective interval",
    )
    if (
        requested[1] <= requested[0]
        or effective[1] <= effective[0]
        or effective[0] < 0
        or effective[1] > duration
        or effective[0] < requested[0] - 1e-6
        or effective[1] > requested[1] + 1e-6
        or abs(relative[0] - (effective[0] - anchor)) > 1e-6
        or abs(relative[1] - (effective[1] - anchor)) > 1e-6
    ):
        raise ValueError("adaptive plan intervals do not close")
    reasons = data["boundary_truncation_reasons"]
    allowed_reasons = {
        "recording_start",
        "recording_stop",
        "previous_candidate",
        "next_candidate",
        "anchor_context_unavailable",
    }
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or not set(reasons) <= allowed_reasons
        or "causal_warmup" in reasons
    ):
        raise ValueError("adaptive plan boundary reasons are invalid")
    if data["source_candidate_decision"] not in _ADAPTIVE_INPUT_DECISIONS:
        raise ValueError("adaptive plan source candidate was not detector-retained")
    if data["analysis_route"] != "adaptive_eeg_search_before_fixed_v29_carrier":
        raise ValueError("adaptive plan analysis route drifted")
    warmup = _finite(data["v29_causal_warmup_seconds"], "v29 causal warmup")
    if warmup < 0:
        raise ValueError("v29 causal warmup must be non-negative")
    neighbors = data["neighbor_constraints"]
    if type(neighbors) is not dict or set(neighbors) != {
        "previous_candidate_id",
        "previous_midpoint_recording_seconds",
        "next_candidate_id",
        "next_midpoint_recording_seconds",
        "guard_seconds",
    }:
        raise ValueError("adaptive neighbor constraints are invalid")
    for id_key, midpoint_key in (
        ("previous_candidate_id", "previous_midpoint_recording_seconds"),
        ("next_candidate_id", "next_midpoint_recording_seconds"),
    ):
        identifier = neighbors[id_key]
        midpoint = neighbors[midpoint_key]
        if (identifier is None) != (midpoint is None):
            raise ValueError("adaptive neighbor ID and midpoint must co-occur")
        if identifier is not None:
            _identifier(identifier, f"adaptive neighbor {id_key}")
            midpoint_value = _finite(midpoint, f"adaptive neighbor {midpoint_key}")
            if not 0 <= midpoint_value <= duration:
                raise ValueError("adaptive neighbor midpoint is outside the recording")
    guard = _finite(neighbors["guard_seconds"], "adaptive neighbor guard")
    if guard < 0:
        raise ValueError("adaptive neighbor guard must be non-negative")
    expected_scope = {
        "eeg_signal_only": True,
        "edf_annotations_read": False,
        "excel_read": False,
        "labels_or_ground_truth_read": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("adaptive envelope plan violates the EEG-only scope")
    digest_source = deepcopy(data)
    digest_source.pop("plan_id")
    # The creator hashes the body before ``plan_id`` exists.  Reconstruct that
    # exact content rather than accepting a merely well-formed identifier.
    expected_id = f"ADAPT-PLAN-{_canonical_sha256(digest_source)[:20]}"
    if data["plan_id"] != expected_id:
        raise ValueError("adaptive envelope plan ID does not bind its content")
    return data


def _sigmoid(value: float, *, center: float, scale: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (value - center) / scale)))))


def _feature_matrix(
    bipolar: np.ndarray,
    *,
    sfreq: float,
    start_seconds: float,
    stop_seconds: float,
    envelope_start_seconds: float,
) -> np.ndarray:
    start = max(0, int(round((start_seconds - envelope_start_seconds) * sfreq)))
    stop = min(
        bipolar.shape[1], int(round((stop_seconds - envelope_start_seconds) * sfreq))
    )
    if stop - start < max(16, int(round(0.5 * sfreq))):
        raise ValueError("feature interval is too short")
    values = bipolar[:, start:stop].astype(np.float64, copy=True)
    values -= np.median(values, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(values * values, axis=1))
    line = np.mean(np.abs(np.diff(values, axis=1)), axis=1)
    taper = np.hanning(values.shape[1])[None, :]
    power = np.abs(np.fft.rfft(values * taper, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sfreq)
    keep = (frequencies >= 0.5) & (frequencies <= min(45.0, 0.95 * sfreq / 2.0))
    kept = power[:, keep]
    kept_frequencies = frequencies[keep]
    totals = np.maximum(np.sum(kept, axis=1), 1e-24)
    probabilities = kept / totals[:, None]
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-24)), axis=1
    ) / max(math.log(max(2, kept.shape[1])), 1e-12)
    centroid = np.sum(kept * kept_frequencies[None, :], axis=1) / totals
    rhythmicity = np.max(kept, axis=1) / totals
    band_edges = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 45.1))
    bands = []
    for low, high in band_edges:
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.sum(power[:, mask], axis=1) / totals)
    return np.column_stack(
        (
            np.log(np.maximum(rms, 1e-12)),
            np.log(np.maximum(line, 1e-12)),
            *bands,
            entropy,
            centroid,
            rhythmicity,
        )
    )


def _baseline_distribution(
    bipolar: np.ndarray,
    *,
    sfreq: float,
    envelope_start: float,
    baseline_start: float,
    baseline_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    cursor = baseline_start
    while cursor + 1.0 <= baseline_stop + 1e-9:
        frames.append(
            _feature_matrix(
                bipolar,
                sfreq=sfreq,
                start_seconds=cursor,
                stop_seconds=cursor + 1.0,
                envelope_start_seconds=envelope_start,
            )
        )
        cursor += 0.5
    if len(frames) < 6:
        raise ValueError("insufficient clean baseline frames")
    stack = np.stack(frames, axis=1)
    median = np.median(stack, axis=1)
    mad = np.median(np.abs(stack - median[:, None, :]), axis=1)
    floors = np.asarray(
        [0.12, 0.12, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.8, 0.04],
        dtype=np.float64,
    )
    scale = np.maximum(1.4826 * mad, floors[None, :])
    return median, scale


def _distance_from_baseline(
    features: np.ndarray, baseline: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = np.clip(np.abs(features - baseline) / scale, 0.0, 20.0)
    ordered = np.sort(z, axis=1)
    return np.mean(ordered[:, -3:], axis=1)


def _top_mean(values: np.ndarray, count: int = 3) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    count = min(max(1, count), finite.size)
    return float(np.mean(np.partition(finite, finite.size - count)[-count:]))


def _connected_derivation_components(
    indices: Sequence[int],
    geometry: Sequence[tuple[str, str]],
) -> list[list[int]]:
    """Return TCP derivation groups connected through a shared electrode."""

    remaining = {int(index) for index in indices}
    components: list[list[int]] = []
    while remaining:
        stack = [remaining.pop()]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            electrodes = set(geometry[current])
            neighbors = [
                other
                for other in remaining
                if electrodes.intersection(geometry[other])
            ]
            for other in neighbors:
                remaining.remove(other)
                stack.append(other)
        components.append(sorted(component))
    return components


def _dominant_connected_support(
    distances: np.ndarray,
    geometry: Sequence[tuple[str, str]],
) -> list[int]:
    values = np.asarray(distances, dtype=np.float64)
    active = np.flatnonzero(
        values >= ADAPTIVE_SEARCH_POLICY["channel_activation_z_threshold"]
    )
    components = _connected_derivation_components(active.tolist(), geometry)
    if not components:
        return []
    return max(
        components,
        key=lambda component: (
            len(component),
            _top_mean(values[np.asarray(component, dtype=int)]),
            -min(component),
        ),
    )


def _select_onset_hysteresis_run(
    rows: Sequence[
        tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray]
    ],
    geometry: Sequence[tuple[str, str]],
) -> tuple[
    tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray],
    tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray],
    list[int],
] | None:
    """Find the earliest persistent connected transition and its local peak.

    A single high global score is insufficient.  Each time point must also
    contain a connected TCP component, the transition must persist over
    adjacent grid points, and the run must contain at least one full-threshold
    point.  The first credible run is selected; the coarse detector anchor is
    only a navigation coordinate and cannot force the onset to time zero.
    """

    full_threshold = float(ADAPTIVE_SEARCH_POLICY["onset_score_threshold"])
    low_threshold = full_threshold * float(
        ADAPTIVE_SEARCH_POLICY["onset_hysteresis_fraction"]
    )
    minimum_connected = int(
        ADAPTIVE_SEARCH_POLICY["minimum_connected_derivations"]
    )
    grid = float(ADAPTIVE_SEARCH_POLICY["grid_step_seconds"])
    qualified: list[tuple[int, list[int]]] = []
    for index, row in enumerate(rows):
        component = _dominant_connected_support(row[4], geometry)
        shortest_scale_score = min(
            row[2], key=lambda item: float(item["scale_seconds"])
        )["score"]
        if (
            row[0] >= low_threshold
            and shortest_scale_score >= low_threshold
            and len(component) >= minimum_connected
        ):
            qualified.append((index, component))

    runs: list[list[tuple[int, list[int]]]] = []
    for item in qualified:
        if (
            not runs
            or rows[item[0]][1] - rows[runs[-1][-1][0]][1] > 1.5 * grid
        ):
            runs.append([item])
        else:
            runs[-1].append(item)
    credible = [
        run
        for run in runs
        if len(run) >= int(ADAPTIVE_SEARCH_POLICY["minimum_onset_run_points"])
        and max(rows[index][0] for index, _ in run) >= full_threshold
    ]
    if not credible:
        return None
    run = min(credible, key=lambda value: (rows[value[0][0]][1], value[0][0]))
    peak_index, peak_component = max(
        run,
        key=lambda item: (
            rows[item[0]][0],
            -abs(rows[item[0]][1]),
            -rows[item[0]][1],
        ),
    )
    onset_index = run[0][0]
    support_union = sorted(
        set(peak_component).union(
            *(set(component) for _, component in run)
        )
    )
    return rows[onset_index], rows[peak_index], support_union


def _select_termination_run(
    rows: Sequence[
        tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray]
    ],
    geometry: Sequence[tuple[str, str]],
) -> tuple[
    tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray],
    bool,
]:
    """Return the earliest persistent return transition, or the best miss."""

    threshold = float(ADAPTIVE_SEARCH_POLICY["termination_score_threshold"])
    grid = float(ADAPTIVE_SEARCH_POLICY["grid_step_seconds"])
    qualifying: list[int] = []
    for index, row in enumerate(rows):
        active_before = _dominant_connected_support(row[3], geometry)
        if (
            row[0] >= threshold
            and len(active_before)
            >= int(ADAPTIVE_SEARCH_POLICY["minimum_connected_derivations"])
        ):
            qualifying.append(index)
    runs: list[list[int]] = []
    for index in qualifying:
        if not runs or rows[index][1] - rows[runs[-1][-1]][1] > 1.5 * grid:
            runs.append([index])
        else:
            runs[-1].append(index)
    persistent = [
        run
        for run in runs
        if len(run) >= int(ADAPTIVE_SEARCH_POLICY["minimum_onset_run_points"])
    ]
    if persistent:
        run = min(persistent, key=lambda value: rows[value[0]][1])
        peak_index = max(run, key=lambda index: (rows[index][0], -rows[index][1]))
        return rows[peak_index], True
    return max(rows, key=lambda row: (row[0], -row[1])), False


def _transition_scores(
    bipolar: np.ndarray,
    *,
    sfreq: float,
    envelope_start: float,
    baseline: np.ndarray,
    baseline_scale: np.ndarray,
    time_seconds: float,
    direction: str,
) -> tuple[float, list[dict[str, float]], np.ndarray, np.ndarray]:
    scale_rows: list[dict[str, float]] = []
    best_pre = best_post = np.zeros(bipolar.shape[0], dtype=np.float64)
    best_score = -1.0
    for window_seconds in ADAPTIVE_SEARCH_POLICY["comparison_scales_seconds"]:
        pre = _feature_matrix(
            bipolar,
            sfreq=sfreq,
            start_seconds=time_seconds - window_seconds,
            stop_seconds=time_seconds,
            envelope_start_seconds=envelope_start,
        )
        post = _feature_matrix(
            bipolar,
            sfreq=sfreq,
            start_seconds=time_seconds,
            stop_seconds=time_seconds + window_seconds,
            envelope_start_seconds=envelope_start,
        )
        pre_distance = _distance_from_baseline(pre, baseline, baseline_scale)
        post_distance = _distance_from_baseline(post, baseline, baseline_scale)
        pre_level = _top_mean(pre_distance)
        post_level = _top_mean(post_distance)
        if direction == "onset":
            contrast = post_level - pre_level
            active_level = post_level
            return_level = pre_level
        elif direction == "termination":
            contrast = pre_level - post_level
            active_level = pre_level
            return_level = post_level
        else:  # pragma: no cover - internal invariant
            raise AssertionError(direction)
        contrast_score = _sigmoid(contrast, center=0.9, scale=0.65)
        active_score = _sigmoid(active_level, center=2.2, scale=0.8)
        return_score = _sigmoid(2.5 - return_level, center=0.0, scale=0.9)
        score = contrast_score * active_score
        if direction == "termination":
            score *= 0.55 + 0.45 * return_score
        scale_rows.append(
            {
                "scale_seconds": float(window_seconds),
                "score": float(np.clip(score, 0.0, 1.0)),
                "pre_activity": pre_level,
                "post_activity": post_level,
            }
        )
        if score > best_score:
            best_score = score
            best_pre, best_post = pre_distance, post_distance
    combined = float(
        np.clip(
            0.55 * max(row["score"] for row in scale_rows)
            + 0.45 * np.median([row["score"] for row in scale_rows]),
            0.0,
            1.0,
        )
    )
    return combined, scale_rows, best_pre, best_post


def _artifact_score(
    bipolar: np.ndarray,
    *,
    sfreq: float,
    envelope_start: float,
    start: float,
    stop: float,
    baseline_start: float,
    baseline_stop: float,
) -> float:
    def peak_to_peak(left: float, right: float) -> np.ndarray:
        s0 = max(0, int(round((left - envelope_start) * sfreq)))
        s1 = min(bipolar.shape[1], int(round((right - envelope_start) * sfreq)))
        values = bipolar[:, s0:s1]
        if values.shape[1] < 2:
            return np.zeros(values.shape[0], dtype=np.float64)
        return np.ptp(values, axis=1)

    baseline_p2p = peak_to_peak(baseline_start, baseline_stop)
    event_p2p = peak_to_peak(start, stop)
    extreme = event_p2p > np.maximum(1e-3, np.maximum(baseline_p2p, 1e-8) * 10.0)
    extreme_fraction = float(np.mean(extreme))
    broad_ratio = event_p2p / np.maximum(baseline_p2p, 1e-8)
    synchronous_fraction = float(np.mean(broad_ratio > 6.0))
    return float(
        np.clip(
            max(
                _sigmoid(extreme_fraction, center=0.18, scale=0.06),
                _sigmoid(synchronous_fraction, center=0.70, scale=0.08),
            ),
            0.0,
            1.0,
        )
    )


def _empty_stage_evidence() -> dict[str, Any]:
    return {
        "onset": {"offset_seconds_relative_to_anchor": None, "score": 0.0, "scale_evidence": []},
        "evolution": {
            "score": 0.0,
            "persistence_fraction": 0.0,
            "change_dimensions": [],
            "supporting_derivations": [],
        },
        "termination": {"offset_seconds_relative_to_anchor": None, "score": 0.0, "scale_evidence": []},
        "artifact": {"score": 0.0, "dominant": False},
    }


def build_unavailable_adaptive_search_receipt(
    *,
    processed_envelope_sha256: str,
    preprocessing_receipt_sha256: str,
    sampling_rate_hz: float,
    envelope_interval_recording_seconds: Sequence[float],
    coarse_anchor_recording_seconds: float,
    canonical_signal_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a fail-closed no-signal receipt for an unavailable envelope."""

    return _finalize_receipt(
        processed_envelope_sha256=processed_envelope_sha256,
        preprocessing_receipt_sha256=preprocessing_receipt_sha256,
        sampling_rate_hz=sampling_rate_hz,
        envelope_interval_recording_seconds=envelope_interval_recording_seconds,
        coarse_anchor_recording_seconds=coarse_anchor_recording_seconds,
        canonical_signal_binding=canonical_signal_binding,
        baseline_interval_relative_to_anchor_seconds=None,
        onset_search_interval_relative_to_anchor_seconds=None,
        termination_search_interval_relative_to_anchor_seconds=None,
        status="abstained_envelope_unavailable",
        fail_reason="envelope_signal_unavailable",
        critical_transition=None,
        stage_evidence=_empty_stage_evidence(),
        confidence_score=0.0,
        recording_duration_seconds=None,
    )


def analyze_adaptive_eeg_envelope(
    signal: torch.Tensor,
    *,
    sampling_rate_hz: float,
    envelope_start_recording_seconds: float,
    candidate_anchor_recording_seconds: float,
    recording_duration_seconds: float,
    processed_envelope_sha256: str | None = None,
    preprocessing_receipt_sha256: str,
    canonical_signal_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Search an arbitrary-length EEG envelope for a complete change interval."""

    sfreq = _finite(sampling_rate_hz, "sampling_rate_hz")
    envelope_start = _finite(
        envelope_start_recording_seconds, "envelope_start_recording_seconds"
    )
    anchor = _finite(
        candidate_anchor_recording_seconds, "candidate_anchor_recording_seconds"
    )
    recording_duration = _finite(
        recording_duration_seconds, "recording_duration_seconds"
    )
    if sfreq <= 0 or envelope_start < 0 or recording_duration <= 0:
        raise ValueError("adaptive search clocks must be positive")
    tensor = signal.detach().cpu().to(torch.float32).contiguous()
    actual_hash = generalized_signal_tensor_sha256(tensor)
    if processed_envelope_sha256 is not None and _sha256(
        processed_envelope_sha256, "processed_envelope_sha256"
    ) != actual_hash:
        raise ValueError("processed envelope hash does not match signal")
    preprocess_hash = _sha256(
        preprocessing_receipt_sha256, "preprocessing_receipt_sha256"
    )
    canonical_binding = (
        None
        if canonical_signal_binding is None
        else validate_canonical_adaptive_signal_binding(
            canonical_signal_binding
        )
    )
    duration = tensor.shape[1] / sfreq
    envelope_stop = envelope_start + duration
    if envelope_stop > recording_duration + 1.0 / sfreq + 1e-6:
        raise ValueError("adaptive envelope extends beyond recording duration")
    if not envelope_start <= anchor <= envelope_stop:
        raise ValueError("coarse anchor is outside adaptive envelope")
    relative_start = envelope_start - anchor
    relative_stop = envelope_stop - anchor

    channel_index = {name: index for index, name in enumerate(STANDARD_19)}
    observed = (
        set(STANDARD_19)
        if canonical_binding is None
        else set(canonical_binding["observed_channel_ids"])
    )
    geometry = tuple(
        (left, right)
        for left, right in TCP_20_EDGES
        if left in observed and right in observed
    )
    if len(geometry) < int(
        ADAPTIVE_SEARCH_POLICY["minimum_connected_derivations"]
    ):
        return _finalize_receipt(
            processed_envelope_sha256=actual_hash,
            preprocessing_receipt_sha256=preprocess_hash,
            sampling_rate_hz=sfreq,
            envelope_interval_recording_seconds=[
                envelope_start,
                envelope_stop,
            ],
            coarse_anchor_recording_seconds=anchor,
            baseline_interval_relative_to_anchor_seconds=None,
            onset_search_interval_relative_to_anchor_seconds=None,
            termination_search_interval_relative_to_anchor_seconds=None,
            status="abstained_envelope_unavailable",
            fail_reason="envelope_signal_unavailable",
            critical_transition=None,
            stage_evidence=_empty_stage_evidence(),
            confidence_score=0.0,
            recording_duration_seconds=recording_duration,
            canonical_signal_binding=canonical_binding,
        )
    values = tensor.numpy().astype(np.float64, copy=False)
    bipolar = np.stack(
        [
            values[channel_index[left]] - values[channel_index[right]]
            for left, right in geometry
        ]
    )
    max_scale = max(ADAPTIVE_SEARCH_POLICY["comparison_scales_seconds"])
    baseline_start = relative_start + ADAPTIVE_SEARCH_POLICY[
        "baseline_left_margin_seconds"
    ]
    baseline_stop = min(
        -ADAPTIVE_SEARCH_POLICY["baseline_exclusion_before_anchor_seconds"],
        baseline_start + ADAPTIVE_SEARCH_POLICY["maximum_baseline_seconds"],
    )
    minimum_baseline = ADAPTIVE_SEARCH_POLICY["minimum_baseline_seconds"]
    common_kwargs = {
        "processed_envelope_sha256": actual_hash,
        "preprocessing_receipt_sha256": preprocess_hash,
        "sampling_rate_hz": sfreq,
        "envelope_interval_recording_seconds": [envelope_start, envelope_stop],
        "coarse_anchor_recording_seconds": anchor,
        "recording_duration_seconds": recording_duration,
        "canonical_signal_binding": canonical_binding,
    }
    if baseline_stop - baseline_start < minimum_baseline:
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=None,
            termination_search_interval_relative_to_anchor_seconds=None,
            status="abstained_insufficient_baseline",
            fail_reason="insufficient_clean_baseline",
            critical_transition=None,
            stage_evidence=_empty_stage_evidence(),
            confidence_score=0.0,
        )
    try:
        baseline, baseline_scale = _baseline_distribution(
            bipolar,
            sfreq=sfreq,
            envelope_start=relative_start,
            baseline_start=baseline_start,
            baseline_stop=baseline_stop,
        )
    except ValueError:
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=None,
            termination_search_interval_relative_to_anchor_seconds=None,
            status="abstained_insufficient_baseline",
            fail_reason="insufficient_clean_baseline",
            critical_transition=None,
            stage_evidence=_empty_stage_evidence(),
            confidence_score=0.0,
        )

    onset_left = max(
        baseline_stop,
        -ADAPTIVE_SEARCH_POLICY["maximum_onset_seconds_before_anchor"],
        relative_start + max_scale,
    )
    onset_right = min(
        ADAPTIVE_SEARCH_POLICY["maximum_onset_seconds_after_anchor"],
        relative_stop - max_scale,
    )
    if onset_right <= onset_left:
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=[onset_left, onset_right],
            termination_search_interval_relative_to_anchor_seconds=None,
            status="abstained_insufficient_baseline",
            fail_reason="insufficient_clean_baseline",
            critical_transition=None,
            stage_evidence=_empty_stage_evidence(),
            confidence_score=0.0,
        )

    onset_candidates: list[tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray]] = []
    for time_seconds in np.arange(
        onset_left,
        onset_right + 0.25 * ADAPTIVE_SEARCH_POLICY["grid_step_seconds"],
        ADAPTIVE_SEARCH_POLICY["grid_step_seconds"],
    ):
        score, scales, pre_distance, post_distance = _transition_scores(
            bipolar,
            sfreq=sfreq,
            envelope_start=relative_start,
            baseline=baseline,
            baseline_scale=baseline_scale,
            time_seconds=float(time_seconds),
            direction="onset",
        )
        onset_candidates.append(
            (score, float(time_seconds), scales, pre_distance, post_distance)
        )
    onset_selection = _select_onset_hysteresis_run(onset_candidates, geometry)
    stage = _empty_stage_evidence()
    if onset_selection is None:
        onset_peak = max(
            onset_candidates, key=lambda row: (row[0], -abs(row[1]))
        )
        stage["onset"] = {
            "offset_seconds_relative_to_anchor": None,
            "peak_offset_seconds_relative_to_anchor": onset_peak[1],
            "score": onset_peak[0],
            "scale_evidence": onset_peak[2],
            "selection_method": "persistent_connected_hysteresis_not_met",
            "supporting_derivations": [],
        }
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=[onset_left, onset_right],
            termination_search_interval_relative_to_anchor_seconds=None,
            status="abstained_no_onset_transition",
            fail_reason="onset_transition_below_threshold",
            critical_transition=None,
            stage_evidence=stage,
            confidence_score=0.30 * onset_peak[0],
        )
    onset_row, onset_peak, onset_component = onset_selection
    onset_score, onset_time, onset_scales, _, onset_post = onset_row
    peak_score, peak_time, peak_scales, _, peak_post = onset_peak
    support_strength = np.maximum(onset_post, peak_post)
    support_component = _dominant_connected_support(support_strength, geometry)
    if len(support_component) < ADAPTIVE_SEARCH_POLICY[
        "minimum_connected_derivations"
    ]:
        support_component = onset_component
    names = [f"{left}-{right}" for left, right in geometry]
    support_order = sorted(
        support_component,
        key=lambda index: (-float(support_strength[index]), names[index]),
    )
    onset_supporting = [names[int(index)] for index in support_order[:4]]
    stage["onset"] = {
        "offset_seconds_relative_to_anchor": onset_time,
        "peak_offset_seconds_relative_to_anchor": peak_time,
        "score": peak_score,
        "scale_evidence": peak_scales,
        "selection_method": "earliest_persistent_connected_hysteresis_run",
        "supporting_derivations": onset_supporting,
    }
    boundary_guard = ADAPTIVE_SEARCH_POLICY["boundary_guard_seconds"]
    if onset_time <= onset_left + boundary_guard:
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=[onset_left, onset_right],
            termination_search_interval_relative_to_anchor_seconds=None,
            status="partial_left_boundary",
            fail_reason="left_boundary_contact",
            critical_transition={
                "start_offset_seconds_relative_to_anchor": None,
                "stop_offset_seconds_relative_to_anchor": None,
                "duration_seconds": None,
                "refined_anchor_recording_seconds": None,
                "supporting_derivations": onset_supporting,
            },
            stage_evidence=stage,
            confidence_score=0.30 * peak_score,
        )

    termination_left = onset_time + ADAPTIVE_SEARCH_POLICY[
        "minimum_candidate_duration_seconds"
    ]
    termination_right = relative_stop - max_scale
    termination_candidates: list[
        tuple[float, float, list[dict[str, float]], np.ndarray, np.ndarray]
    ] = []
    if termination_right > termination_left:
        for time_seconds in np.arange(
            termination_left,
            termination_right
            + 0.25 * ADAPTIVE_SEARCH_POLICY["grid_step_seconds"],
            ADAPTIVE_SEARCH_POLICY["grid_step_seconds"],
        ):
            score, scales, pre_distance, post_distance = _transition_scores(
                bipolar,
                sfreq=sfreq,
                envelope_start=relative_start,
                baseline=baseline,
                baseline_scale=baseline_scale,
                time_seconds=float(time_seconds),
                direction="termination",
            )
            termination_candidates.append(
                (score, float(time_seconds), scales, pre_distance, post_distance)
            )
    if not termination_candidates:
        stage["termination"] = {
            "offset_seconds_relative_to_anchor": None,
            "score": 0.0,
            "scale_evidence": [],
        }
        return _finalize_receipt(
            **common_kwargs,
            baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
            onset_search_interval_relative_to_anchor_seconds=[onset_left, onset_right],
            termination_search_interval_relative_to_anchor_seconds=[termination_left, termination_right],
            status="partial_right_boundary",
            fail_reason="right_boundary_contact",
            critical_transition={
                "start_offset_seconds_relative_to_anchor": onset_time,
                "stop_offset_seconds_relative_to_anchor": None,
                "duration_seconds": None,
                "refined_anchor_recording_seconds": anchor + onset_time,
                "supporting_derivations": onset_supporting,
            },
            stage_evidence=stage,
            confidence_score=0.30 * peak_score,
        )
    termination_best, termination_run_qualified = _select_termination_run(
        termination_candidates,
        geometry,
    )
    termination_score, termination_time, termination_scales, termination_pre, _ = (
        termination_best
    )
    stage["termination"] = {
        "offset_seconds_relative_to_anchor": termination_time,
        "score": termination_score,
        "scale_evidence": termination_scales,
        "selection_method": (
            "earliest_persistent_return_transition"
            if termination_run_qualified
            else "best_transition_below_persistence_gate"
        ),
    }

    support_strength = np.maximum(support_strength, termination_pre)
    support_order = np.argsort(support_strength)[::-1]
    onset_component_set = set(onset_component).union(support_component)
    ranked_supporting = [
        names[int(index)]
        for index in support_order
        if int(index) in onset_component_set
    ]
    supporting = list(onset_supporting)
    supporting.extend(
        item for item in ranked_supporting if item not in supporting
    )
    supporting = supporting[:4]

    activity_times = np.arange(
        onset_time + 1.0,
        termination_time - 0.5,
        1.0,
    )
    activity_levels: list[float] = []
    activity_features: list[np.ndarray] = []
    for center in activity_times:
        feature = _feature_matrix(
            bipolar,
            sfreq=sfreq,
            start_seconds=float(center - 0.5),
            stop_seconds=float(center + 0.5),
            envelope_start_seconds=relative_start,
        )
        activity_features.append(feature)
        activity_levels.append(
            _top_mean(_distance_from_baseline(feature, baseline, baseline_scale))
        )
    persistence = (
        float(np.mean(np.asarray(activity_levels) >= 2.0))
        if activity_levels
        else 0.0
    )
    dimension_scores: dict[str, float] = {
        "amplitude": 0.0,
        "frequency": 0.0,
        "spectral_shape": 0.0,
        "spatial_distribution": 0.0,
    }
    if len(activity_features) >= 4:
        early = np.median(np.stack(activity_features[:2]), axis=0)
        late = np.median(np.stack(activity_features[-2:]), axis=0)
        dimension_scores["amplitude"] = float(
            np.clip(_top_mean(np.abs(late[:, :2] - early[:, :2])) / 0.7, 0.0, 1.0)
        )
        dimension_scores["frequency"] = float(
            np.clip(_top_mean(np.abs(late[:, 8] - early[:, 8])) / 4.0, 0.0, 1.0)
        )
        dimension_scores["spectral_shape"] = float(
            np.clip(
                _top_mean(np.max(np.abs(late[:, 2:8] - early[:, 2:8]), axis=1))
                / 0.18,
                0.0,
                1.0,
            )
        )
        early_distance = _distance_from_baseline(early, baseline, baseline_scale)
        late_distance = _distance_from_baseline(late, baseline, baseline_scale)
        early_top = set(np.argsort(early_distance)[-4:].tolist())
        late_top = set(np.argsort(late_distance)[-4:].tolist())
        dimension_scores["spatial_distribution"] = 1.0 - len(
            early_top & late_top
        ) / max(1, len(early_top | late_top))
    evolution_dimensions = [
        name for name, score in dimension_scores.items() if score >= 0.25
    ]
    evolution_score = float(
        np.clip(
            0.55 * persistence + 0.45 * max(dimension_scores.values()), 0.0, 1.0
        )
    )
    stage["evolution"] = {
        "score": evolution_score,
        "persistence_fraction": persistence,
        "change_dimensions": evolution_dimensions,
        "supporting_derivations": supporting,
    }
    artifact = _artifact_score(
        bipolar,
        sfreq=sfreq,
        envelope_start=relative_start,
        start=onset_time,
        stop=termination_time,
        baseline_start=baseline_start,
        baseline_stop=baseline_stop,
    )
    stage["artifact"] = {
        "score": artifact,
        "dominant": artifact >= ADAPTIVE_SEARCH_POLICY["artifact_abstention_threshold"],
    }
    confidence = float(
        np.clip(
            (
                0.30 * peak_score
                + 0.25 * persistence
                + 0.20 * evolution_score
                + 0.25 * termination_score
            )
            * (1.0 - 0.45 * artifact),
            0.0,
            1.0,
        )
    )
    critical = {
        "start_offset_seconds_relative_to_anchor": onset_time,
        "stop_offset_seconds_relative_to_anchor": termination_time,
        "duration_seconds": termination_time - onset_time,
        "refined_anchor_recording_seconds": anchor + onset_time,
        "supporting_derivations": supporting,
    }
    if not termination_run_qualified:
        status = (
            "partial_right_boundary"
            if termination_time >= termination_right - boundary_guard
            else "abstained_no_termination_transition"
        )
        reason = (
            "right_boundary_contact"
            if status == "partial_right_boundary"
            else "termination_transition_below_threshold"
        )
        critical["stop_offset_seconds_relative_to_anchor"] = None
        critical["duration_seconds"] = None
    elif artifact >= ADAPTIVE_SEARCH_POLICY["artifact_abstention_threshold"]:
        status = "abstained_artifact_dominated"
        reason = "artifact_dominates_transition"
        critical = None
    elif (
        persistence < ADAPTIVE_SEARCH_POLICY["persistence_fraction_threshold"]
        or confidence < ADAPTIVE_SEARCH_POLICY["joint_confidence_threshold"]
    ):
        status = "abstained_low_confidence"
        reason = "joint_confidence_below_threshold"
        critical = None
    else:
        status = "qualified_complete"
        reason = "none"

    return _finalize_receipt(
        **common_kwargs,
        baseline_interval_relative_to_anchor_seconds=[baseline_start, baseline_stop],
        onset_search_interval_relative_to_anchor_seconds=[onset_left, onset_right],
        termination_search_interval_relative_to_anchor_seconds=[
            termination_left,
            termination_right,
        ],
        status=status,
        fail_reason=reason,
        critical_transition=critical,
        stage_evidence=stage,
        confidence_score=confidence,
    )


def _finalize_receipt(
    *,
    processed_envelope_sha256: str,
    preprocessing_receipt_sha256: str,
    sampling_rate_hz: float,
    envelope_interval_recording_seconds: Sequence[float],
    coarse_anchor_recording_seconds: float,
    baseline_interval_relative_to_anchor_seconds: Sequence[float] | None,
    onset_search_interval_relative_to_anchor_seconds: Sequence[float] | None,
    termination_search_interval_relative_to_anchor_seconds: Sequence[float] | None,
    status: str,
    fail_reason: str,
    critical_transition: Mapping[str, Any] | None,
    stage_evidence: Mapping[str, Any],
    confidence_score: float,
    recording_duration_seconds: float | None,
    canonical_signal_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    envelope = [float(value) for value in envelope_interval_recording_seconds]
    anchor = float(coarse_anchor_recording_seconds)
    v29_projection: dict[str, Any] = {
        "decision": "not_eligible_search_abstained",
        "refined_anchor_recording_seconds": None,
        "fixed_window_recording_seconds": None,
        "reason": "adaptive_search_not_complete",
    }
    onset_refined_statuses = {
        "qualified_complete",
        "partial_right_boundary",
        "abstained_no_termination_transition",
    }
    if status in onset_refined_statuses and critical_transition is not None:
        raw_refined = critical_transition.get("refined_anchor_recording_seconds")
        if raw_refined is None:  # pragma: no cover - guarded by validator
            raise ValueError("onset-refined status requires a refined anchor")
        refined = float(raw_refined)
        v29_projection["refined_anchor_recording_seconds"] = refined
        if (
            recording_duration_seconds is not None
            and refined
            >= ADAPTIVE_SEARCH_POLICY["v29_pre_seconds"]
            + ADAPTIVE_SEARCH_POLICY["v29_causal_warmup_seconds"]
            and float(recording_duration_seconds) - refined
            >= ADAPTIVE_SEARCH_POLICY["v29_post_seconds"]
        ):
            v29_projection = {
                "decision": "eligible_after_refinement",
                "refined_anchor_recording_seconds": refined,
                "fixed_window_recording_seconds": [refined - 12.0, refined + 48.0],
                "reason": "complete_fixed_window_available",
            }
        else:
            v29_projection = {
                "decision": "abstain_insufficient_fixed_window",
                "refined_anchor_recording_seconds": refined,
                "fixed_window_recording_seconds": None,
                "reason": "refined_anchor_lacks_warmup_or_post_context",
            }
    confidence = float(np.clip(confidence_score, 0.0, 1.0))
    body = {
        "schema_version": ADAPTIVE_SEARCH_SCHEMA_VERSION,
        "search_receipt_id": "CONTENT-ADDRESS-PENDING",
        "method_id": ADAPTIVE_SEARCH_METHOD_ID,
        "policy_sha256": ADAPTIVE_SEARCH_POLICY_SHA256,
        "processed_envelope_sha256": processed_envelope_sha256,
        "preprocessing_receipt_sha256": preprocessing_receipt_sha256,
        "canonical_signal_binding": (
            validate_canonical_adaptive_signal_binding(
                canonical_signal_binding
            )
            if canonical_signal_binding is not None
            else None
        ),
        "sampling_rate_hz": float(sampling_rate_hz),
        "channel_order": list(STANDARD_19),
        "recording_duration_seconds": (
            float(recording_duration_seconds)
            if recording_duration_seconds is not None
            else None
        ),
        "envelope_interval_recording_seconds": envelope,
        "coarse_anchor_recording_seconds": anchor,
        "envelope_interval_relative_to_anchor_seconds": [
            envelope[0] - anchor,
            envelope[1] - anchor,
        ],
        "baseline_interval_relative_to_anchor_seconds": (
            list(baseline_interval_relative_to_anchor_seconds)
            if baseline_interval_relative_to_anchor_seconds is not None
            else None
        ),
        "onset_search_interval_relative_to_anchor_seconds": (
            list(onset_search_interval_relative_to_anchor_seconds)
            if onset_search_interval_relative_to_anchor_seconds is not None
            else None
        ),
        "termination_search_interval_relative_to_anchor_seconds": (
            list(termination_search_interval_relative_to_anchor_seconds)
            if termination_search_interval_relative_to_anchor_seconds is not None
            else None
        ),
        "status": status,
        "fail_closed_reason": fail_reason,
        "critical_transition": deepcopy(dict(critical_transition))
        if critical_transition is not None
        else None,
        "stage_evidence": deepcopy(dict(stage_evidence)),
        "confidence_receipt": {
            "score": confidence,
            "band": (
                "high"
                if confidence >= 0.75
                else "moderate"
                if confidence >= ADAPTIVE_SEARCH_POLICY["joint_confidence_threshold"]
                else "low"
            ),
            "qualification_threshold": ADAPTIVE_SEARCH_POLICY[
                "joint_confidence_threshold"
            ],
            "calibrated_probability": False,
        },
        "v29_projection": v29_projection,
        "scope_receipt": {
            "scope": ADAPTIVE_SEARCH_SCOPE,
            "eeg_signal_used": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "labels_or_ground_truth_used": False,
            "detector_anchor_used_for_navigation_only": True,
            "candidate_is_confirmed_seizure": False,
            "confidence_is_calibrated_probability": False,
        },
    }
    body["search_receipt_id"] = f"ADAPT-SEARCH-{_canonical_sha256(body)[:20]}"
    return validate_adaptive_search_receipt(body)


def validate_adaptive_search_receipt(payload: object) -> dict[str, Any]:
    """Strictly validate an adaptive-search receipt and its non-leakage flags."""

    if type(payload) is not dict:
        raise TypeError("adaptive search receipt must be an object")
    required = {
        "schema_version",
        "search_receipt_id",
        "method_id",
        "policy_sha256",
        "processed_envelope_sha256",
        "preprocessing_receipt_sha256",
        "canonical_signal_binding",
        "sampling_rate_hz",
        "channel_order",
        "recording_duration_seconds",
        "envelope_interval_recording_seconds",
        "coarse_anchor_recording_seconds",
        "envelope_interval_relative_to_anchor_seconds",
        "baseline_interval_relative_to_anchor_seconds",
        "onset_search_interval_relative_to_anchor_seconds",
        "termination_search_interval_relative_to_anchor_seconds",
        "status",
        "fail_closed_reason",
        "critical_transition",
        "stage_evidence",
        "confidence_receipt",
        "v29_projection",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("adaptive search receipt has missing or unknown fields")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_SEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive search schema")
    if data["method_id"] != ADAPTIVE_SEARCH_METHOD_ID:
        raise ValueError("adaptive search method drifted")
    if data["policy_sha256"] != ADAPTIVE_SEARCH_POLICY_SHA256:
        raise ValueError("adaptive search policy hash drifted")
    _identifier(data["search_receipt_id"], "search_receipt_id")
    _sha256(data["processed_envelope_sha256"], "processed_envelope_sha256")
    _sha256(data["preprocessing_receipt_sha256"], "preprocessing_receipt_sha256")
    canonical_binding = (
        None
        if data["canonical_signal_binding"] is None
        else validate_canonical_adaptive_signal_binding(
            data["canonical_signal_binding"]
        )
    )
    data["canonical_signal_binding"] = canonical_binding
    sfreq = _finite(data["sampling_rate_hz"], "sampling_rate_hz")
    if sfreq <= 0 or data["channel_order"] != list(STANDARD_19):
        raise ValueError("adaptive search channel/sampling contract drifted")
    envelope = data["envelope_interval_recording_seconds"]
    relative = data["envelope_interval_relative_to_anchor_seconds"]
    if not isinstance(envelope, list) or len(envelope) != 2:
        raise TypeError("envelope interval must be a pair")
    if not isinstance(relative, list) or len(relative) != 2:
        raise TypeError("relative envelope interval must be a pair")
    envelope = [_finite(value, "envelope interval") for value in envelope]
    anchor = _finite(data["coarse_anchor_recording_seconds"], "coarse anchor")
    raw_recording_duration = data["recording_duration_seconds"]
    recording_duration = (
        None
        if raw_recording_duration is None
        else _finite(raw_recording_duration, "recording duration")
    )
    if canonical_binding is not None and (
        recording_duration is None
        or abs(
            float(canonical_binding["recording_duration_seconds"])
            - recording_duration
        )
        > 1e-6
    ):
        raise ValueError(
            "adaptive search clock disagrees with its canonical signal binding"
        )
    if recording_duration is not None and (
        recording_duration <= 0
        or envelope[0] < 0
        or envelope[1] > recording_duration + 1e-6
    ):
        raise ValueError("adaptive envelope lies outside the recording")
    if envelope[1] <= envelope[0] or not envelope[0] <= anchor <= envelope[1]:
        raise ValueError("adaptive envelope clock is invalid")
    if any(
        abs(float(actual) - expected) > 1e-6
        for actual, expected in zip(relative, (envelope[0] - anchor, envelope[1] - anchor))
    ):
        raise ValueError("adaptive envelope timebases do not close")
    status = data["status"]
    reason = data["fail_closed_reason"]
    if status not in _STATUS or reason not in _FAIL_REASON:
        raise ValueError("adaptive search status/reason is unsupported")
    expected_reason = {
        "qualified_complete": "none",
        "partial_left_boundary": "left_boundary_contact",
        "partial_right_boundary": "right_boundary_contact",
        "abstained_insufficient_baseline": "insufficient_clean_baseline",
        "abstained_no_onset_transition": "onset_transition_below_threshold",
        "abstained_no_termination_transition": (
            "termination_transition_below_threshold"
        ),
        "abstained_artifact_dominated": "artifact_dominates_transition",
        "abstained_low_confidence": "joint_confidence_below_threshold",
        "abstained_envelope_unavailable": "envelope_signal_unavailable",
    }[status]
    if reason != expected_reason:
        raise ValueError("adaptive search status/reason combination is invalid")
    confidence = data["confidence_receipt"]
    if type(confidence) is not dict or set(confidence) != {
        "score",
        "band",
        "qualification_threshold",
        "calibrated_probability",
    }:
        raise ValueError("adaptive confidence receipt is invalid")
    score = _finite(confidence["score"], "adaptive confidence score")
    if not 0 <= score <= 1 or confidence["calibrated_probability"] is not False:
        raise ValueError("adaptive confidence must remain uncalibrated in [0,1]")
    if abs(
        float(confidence["qualification_threshold"])
        - ADAPTIVE_SEARCH_POLICY["joint_confidence_threshold"]
    ) > 1e-12:
        raise ValueError("adaptive confidence threshold drifted")
    expected_band = (
        "high"
        if score >= 0.75
        else "moderate"
        if score >= ADAPTIVE_SEARCH_POLICY["joint_confidence_threshold"]
        else "low"
    )
    if confidence["band"] != expected_band:
        raise ValueError("adaptive confidence band does not match its score")
    def interval_or_none(value: object, context: str) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError(f"{context} must be null or a two-number array")
        pair = [_finite(item, context) for item in value]
        if pair[1] <= pair[0]:
            raise ValueError(f"{context} must have positive duration")
        if pair[0] < relative[0] - 1e-6 or pair[1] > relative[1] + 1e-6:
            raise ValueError(f"{context} lies outside the adaptive envelope")
        return pair

    baseline_interval = interval_or_none(
        data["baseline_interval_relative_to_anchor_seconds"], "baseline interval"
    )
    onset_interval = interval_or_none(
        data["onset_search_interval_relative_to_anchor_seconds"],
        "onset search interval",
    )
    termination_interval = interval_or_none(
        data["termination_search_interval_relative_to_anchor_seconds"],
        "termination search interval",
    )
    if baseline_interval is not None and baseline_interval[1] > 0:
        raise ValueError("adaptive baseline must precede the coarse anchor")
    if onset_interval is not None and baseline_interval is None:
        raise ValueError("onset search requires an explicit baseline interval")
    if termination_interval is not None and onset_interval is None:
        raise ValueError("termination search requires an onset search interval")

    observed_channels = (
        set(STANDARD_19)
        if canonical_binding is None
        else set(canonical_binding["observed_channel_ids"])
    )
    allowed_derivations = {
        f"{left}-{right}"
        for left, right in TCP_20_EDGES
        if left in observed_channels and right in observed_channels
    }

    def derivation_list(value: object, context: str, *, allow_empty: bool) -> list[str]:
        if not isinstance(value, list) or (not allow_empty and not value):
            raise TypeError(f"{context} must be a {'possibly empty ' if allow_empty else ''}list")
        if (
            any(not isinstance(item, str) for item in value)
            or len(value) != len(set(value))
            or len(value) > 4
            or not set(value) <= allowed_derivations
        ):
            raise ValueError(f"{context} contains invalid derivations")
        return list(value)

    critical = data["critical_transition"]
    critical_statuses = {
        "qualified_complete",
        "partial_left_boundary",
        "partial_right_boundary",
        "abstained_no_termination_transition",
    }
    start: float | None = None
    stop: float | None = None
    critical_derivations: list[str] = []
    if status in critical_statuses:
        if type(critical) is not dict or set(critical) != {
            "start_offset_seconds_relative_to_anchor",
            "stop_offset_seconds_relative_to_anchor",
            "duration_seconds",
            "refined_anchor_recording_seconds",
            "supporting_derivations",
        }:
            raise TypeError("adaptive boundary status requires a strict critical transition")
        critical_derivations = derivation_list(
            critical["supporting_derivations"],
            "critical transition derivations",
            allow_empty=False,
        )
        if status == "partial_left_boundary":
            if any(
                critical[key] is not None
                for key in (
                    "start_offset_seconds_relative_to_anchor",
                    "stop_offset_seconds_relative_to_anchor",
                    "duration_seconds",
                    "refined_anchor_recording_seconds",
                )
            ):
                raise ValueError("left-censored transition cannot claim a boundary or refined anchor")
        else:
            start = _finite(
                critical["start_offset_seconds_relative_to_anchor"],
                "adaptive start",
            )
            refined = _finite(
                critical["refined_anchor_recording_seconds"],
                "refined adaptive anchor",
            )
            if not relative[0] <= start <= relative[1] or abs(refined - (anchor + start)) > 1e-6:
                raise ValueError("adaptive refined anchor does not close its timebase")
            if status == "qualified_complete":
                stop = _finite(
                    critical["stop_offset_seconds_relative_to_anchor"],
                    "adaptive stop",
                )
                duration_value = _finite(
                    critical["duration_seconds"], "adaptive duration"
                )
                if (
                    stop > relative[1]
                    or stop - start
                    < ADAPTIVE_SEARCH_POLICY["minimum_candidate_duration_seconds"]
                    or abs(duration_value - (stop - start)) > 1e-6
                ):
                    raise ValueError("qualified adaptive interval is invalid")
                if score < ADAPTIVE_SEARCH_POLICY["joint_confidence_threshold"]:
                    raise ValueError("qualified adaptive transition is below threshold")
            elif (
                critical["stop_offset_seconds_relative_to_anchor"] is not None
                or critical["duration_seconds"] is not None
            ):
                raise ValueError("right-censored or unterminated transition cannot claim a stop")
    elif critical is not None:
        raise ValueError("abstained adaptive search must not retain a critical transition")

    stage = data["stage_evidence"]
    if type(stage) is not dict or set(stage) != {
        "onset",
        "evolution",
        "termination",
        "artifact",
    }:
        raise ValueError("adaptive stage evidence is invalid")
    onset_stage = stage["onset"]
    onset_allowed = {
        "offset_seconds_relative_to_anchor",
        "peak_offset_seconds_relative_to_anchor",
        "score",
        "scale_evidence",
        "selection_method",
        "supporting_derivations",
    }
    if type(onset_stage) is not dict or not set(onset_stage) <= onset_allowed or not {
        "offset_seconds_relative_to_anchor",
        "score",
        "scale_evidence",
    } <= set(onset_stage):
        raise ValueError("adaptive onset evidence is invalid")
    onset_stage_score = _finite(onset_stage["score"], "onset score")
    if not 0 <= onset_stage_score <= 1 or not isinstance(
        onset_stage["scale_evidence"], list
    ):
        raise ValueError("adaptive onset evidence score is invalid")
    if "supporting_derivations" in onset_stage:
        onset_derivations = derivation_list(
            onset_stage["supporting_derivations"],
            "onset supporting derivations",
            allow_empty=True,
        )
        if critical_derivations and not set(onset_derivations) <= set(
            critical_derivations
        ):
            raise ValueError("onset derivations drift from the critical transition")
    onset_offset = onset_stage["offset_seconds_relative_to_anchor"]
    if onset_offset is not None:
        onset_offset = _finite(onset_offset, "onset offset")
        if status not in {"partial_left_boundary"} and start is not None and abs(onset_offset - start) > 1e-6:
            raise ValueError("onset evidence and critical start disagree")

    evolution_stage = stage["evolution"]
    if type(evolution_stage) is not dict or set(evolution_stage) != {
        "score",
        "persistence_fraction",
        "change_dimensions",
        "supporting_derivations",
    }:
        raise ValueError("adaptive evolution evidence is invalid")
    evolution_score = _finite(evolution_stage["score"], "evolution score")
    persistence = _finite(
        evolution_stage["persistence_fraction"], "evolution persistence"
    )
    if not 0 <= evolution_score <= 1 or not 0 <= persistence <= 1:
        raise ValueError("adaptive evolution scores must be in [0,1]")
    dimensions = evolution_stage["change_dimensions"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != len(set(dimensions))
        or not set(dimensions)
        <= {"amplitude", "frequency", "spectral_shape", "spatial_distribution"}
    ):
        raise ValueError("adaptive evolution dimensions are invalid")
    derivation_list(
        evolution_stage["supporting_derivations"],
        "evolution supporting derivations",
        allow_empty=True,
    )

    termination_stage = stage["termination"]
    if type(termination_stage) is not dict or not set(termination_stage) <= {
        "offset_seconds_relative_to_anchor",
        "score",
        "scale_evidence",
        "selection_method",
    } or not {
        "offset_seconds_relative_to_anchor",
        "score",
        "scale_evidence",
    } <= set(termination_stage):
        raise ValueError("adaptive termination evidence is invalid")
    termination_stage_score = _finite(
        termination_stage["score"], "termination score"
    )
    if not 0 <= termination_stage_score <= 1 or not isinstance(
        termination_stage["scale_evidence"], list
    ):
        raise ValueError("adaptive termination evidence score is invalid")
    termination_offset = termination_stage["offset_seconds_relative_to_anchor"]
    if termination_offset is not None:
        termination_offset = _finite(termination_offset, "termination offset")
        if stop is not None and abs(termination_offset - stop) > 1e-6:
            raise ValueError("termination evidence and critical stop disagree")

    artifact_stage = stage["artifact"]
    if type(artifact_stage) is not dict or set(artifact_stage) != {
        "score",
        "dominant",
    }:
        raise ValueError("adaptive artifact evidence is invalid")
    artifact_score = _finite(artifact_stage["score"], "artifact score")
    if not 0 <= artifact_score <= 1 or type(artifact_stage["dominant"]) is not bool:
        raise ValueError("adaptive artifact evidence score is invalid")
    if artifact_stage["dominant"] != (
        artifact_score >= ADAPTIVE_SEARCH_POLICY["artifact_abstention_threshold"]
    ):
        raise ValueError("adaptive artifact dominance flag drifted")

    projection = data["v29_projection"]
    if type(projection) is not dict or set(projection) != {
        "decision",
        "refined_anchor_recording_seconds",
        "fixed_window_recording_seconds",
        "reason",
    }:
        raise ValueError("adaptive v29 projection is invalid")
    if projection["decision"] == "eligible_after_refinement":
        refined = _finite(
            projection["refined_anchor_recording_seconds"],
            "v29 refined anchor",
        )
        fixed = projection["fixed_window_recording_seconds"]
        if not isinstance(fixed, list) or len(fixed) != 2:
            raise TypeError("eligible v29 projection requires a fixed window")
        fixed = [_finite(item, "v29 fixed window") for item in fixed]
        if (
            abs(fixed[0] - (refined - ADAPTIVE_SEARCH_POLICY["v29_pre_seconds"]))
            > 1e-6
            or abs(fixed[1] - (refined + ADAPTIVE_SEARCH_POLICY["v29_post_seconds"]))
            > 1e-6
            or refined
            < ADAPTIVE_SEARCH_POLICY["v29_pre_seconds"]
            + ADAPTIVE_SEARCH_POLICY["v29_causal_warmup_seconds"]
            or recording_duration is None
            or fixed[1] > recording_duration + 1e-6
            or projection["reason"] != "complete_fixed_window_available"
        ):
            raise ValueError("eligible v29 projection violates the fixed carrier contract")
    elif projection["decision"] == "abstain_insufficient_fixed_window":
        if (
            projection["refined_anchor_recording_seconds"] is None
            or projection["fixed_window_recording_seconds"] is not None
            or projection["reason"]
            != "refined_anchor_lacks_warmup_or_post_context"
        ):
            raise ValueError("v29 fixed-window abstention is malformed")
    elif projection["decision"] == "not_eligible_search_abstained":
        if (
            projection["refined_anchor_recording_seconds"] is not None
            or projection["fixed_window_recording_seconds"] is not None
            or projection["reason"] != "adaptive_search_not_complete"
        ):
            raise ValueError("v29 search abstention is malformed")
    else:
        raise ValueError("adaptive v29 projection decision is unsupported")
    scope = data["scope_receipt"]
    expected_scope = {
        "scope": ADAPTIVE_SEARCH_SCOPE,
        "eeg_signal_used": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "labels_or_ground_truth_used": False,
        "detector_anchor_used_for_navigation_only": True,
        "candidate_is_confirmed_seizure": False,
        "confidence_is_calibrated_probability": False,
    }
    if scope != expected_scope:
        raise ValueError("adaptive search scope receipt violates EEG-only policy")
    digest_source = deepcopy(data)
    digest_source["search_receipt_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = f"ADAPT-SEARCH-{_canonical_sha256(digest_source)[:20]}"
    if data["search_receipt_id"] != expected_id:
        raise ValueError("adaptive search receipt ID does not bind its content")
    return data


__all__ = [
    "ADAPTIVE_SEARCH_METHOD_ID",
    "ADAPTIVE_SEARCH_POLICY",
    "ADAPTIVE_SEARCH_POLICY_SHA256",
    "ADAPTIVE_SEARCH_SCHEMA_VERSION",
    "ENVELOPE_PLAN_SCHEMA_VERSION",
    "analyze_adaptive_eeg_envelope",
    "build_unavailable_adaptive_search_receipt",
    "generalized_signal_tensor_sha256",
    "plan_adaptive_search_envelopes",
    "validate_adaptive_search_envelope_plan",
    "validate_adaptive_search_receipt",
]
