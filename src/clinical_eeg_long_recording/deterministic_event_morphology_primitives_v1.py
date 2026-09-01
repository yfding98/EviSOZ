"""Replayable numerical waveform-morphology primitives for one EEG event.

This module closes a deliberately narrow gap in ``event_eeg_findings_v3``:
the wire can carry a morphology candidate, but the deterministic producer has
no signal-replayable supervision interface for the physical waveform
primitives underneath that candidate.

The materializer measures caller-supplied ``(view, unit, physical interval)``
queries on an immutable native-morphology view.  It emits numerical targets
only: excursions, RMS, line length, slopes, curvature, crossing/turning-point
counts, and dominant-excursion timing.  It does **not** detect or name a spike,
sharp wave, IED, seizure, onset pattern, SOZ, EZ, pathology, or diagnosis.

Every requested row remains in the receipt.  Family ineligibility, missing or
imputed units, padding/filter edges, QC overlap, too few samples, and degenerate
geometry are represented by per-target opportunity masks and typed reason
codes; a masked zero is never a negative label.  The receipt binds physical
recording time, view samples, canonical raw samples, units, reference, QC,
transform lineage, policy, values and masks with content hashes.  Exact replay
requires the same host-supplied canonical/view receipts and tensors.  There is
no annotation, spreadsheet, clinical-text, private-label, or Qwen API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from .ba_ieg_numerical_kernel import (
    BA_IEG_BASE_NUMERICAL_KERNEL_ID,
    BA_IEG_BASE_MEASUREMENT_NAMES,
    BAIEGBaseNumericalPolicy,
    measure_ba_ieg_base_numerical_features,
)
from .canonical_signal_views import (
    recording_seconds_to_canonical_sample_index,
    recording_seconds_to_view_tensor_index,
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
    view_tensor_index_to_recording_seconds,
)
from .deterministic_event_findings import deterministic_view_tensor_sha256


EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_event_morphology_primitive_supervision_v1"
EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID: Final[
    str
] = "DETERMINISTIC-EVENT-MORPHOLOGY-PRIMITIVES-V1"
EVENT_MORPHOLOGY_PRIMITIVE_POLICY_ID: Final[
    str
] = "DETERMINISTIC-EVENT-MORPHOLOGY-PRIMITIVE-POLICY-V1"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOL = 1e-8
_QUERY_AUTHORITIES = frozenset(
    {
        "deterministic_signal_proposal",
        "frozen_model_proposal",
        "synthetic_signal_injection",
    }
)

# These are numerical atoms and unit declarations, not clinical terminology.
# ``family`` controls only which signal-view opportunity gate is required.
EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("support_duration_seconds", "s", "waveform"),
    ("rms_uv", "uV", "amplitude"),
    ("peak_to_peak_uv", "uV", "amplitude"),
    ("positive_excursion_uv", "uV", "amplitude"),
    ("negative_excursion_uv", "uV", "amplitude"),
    ("line_length_uv", "uV", "amplitude"),
    ("max_rise_slope_uv_per_s", "uV_per_s", "morphology"),
    ("max_fall_slope_uv_per_s", "uV_per_s", "morphology"),
    ("max_abs_curvature_uv_per_s2", "uV_per_s2", "morphology"),
    ("median_crossing_count", "count", "morphology"),
    ("turning_point_count", "count", "morphology"),
    ("dominant_excursion_latency_seconds", "s", "morphology"),
    ("dominant_excursion_half_height_width_seconds", "s", "morphology"),
    ("dominant_excursion_rise_half_height_seconds", "s", "morphology"),
    ("dominant_excursion_fall_half_height_seconds", "s", "morphology"),
    ("dominant_excursion_asymmetry_ratio", "ratio", "morphology"),
)

EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES: Final[tuple[str, ...]] = tuple(
    row[0] for row in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
)
_TARGET_INDEX = {
    name: index for index, name in enumerate(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
}
_TARGET_FAMILY = tuple(row[2] for row in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS)

_FIREWALL = {
    "eeg_samples_used": True,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_used": False,
    "sleep_or_activation_labels_used": False,
    "qwen_or_other_llm_used": False,
}

_AUTHORIZATION = {
    "supervision_scope": "numerical_waveform_morphology_primitives_only",
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "event_qualification_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "allowed_query_authorities": sorted(_QUERY_AUTHORITIES),
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_TARGET_REGISTRY_SHA256 = _canonical_sha256(
    [
        {"target_name": name, "unit_id": unit, "opportunity_family": family}
        for name, unit, family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
    ]
)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be an event-contract compatible ID")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start = _finite(value[0], f"{name}[0]", minimum=0.0)
    stop = _finite(value[1], f"{name}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _sorted_reasons(values: Sequence[str]) -> list[str]:
    result = sorted(set(str(item) for item in values))
    if any(not item or item != item.strip() for item in result):
        raise ValueError("reason codes must be non-empty trimmed strings")
    return result


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


@dataclass(frozen=True)
class EventMorphologyPrimitivePolicy:
    """Frozen numerical policy; values are engineering rules, not norms."""

    minimum_samples: int = 5
    centering: str = "segment_median"
    crossing_interpolation: str = "piecewise_linear"
    tie_breaking: str = "earliest_sample"

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples < 5
        ):
            raise ValueError("minimum_samples must be an integer >= 5")
        if self.centering != "segment_median":
            raise ValueError("v1 freezes segment-median centering")
        if self.crossing_interpolation != "piecewise_linear":
            raise ValueError("v1 freezes piecewise-linear crossing interpolation")
        if self.tie_breaking != "earliest_sample":
            raise ValueError("v1 freezes earliest-sample tie breaking")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "policy_id": EVENT_MORPHOLOGY_PRIMITIVE_POLICY_ID,
            "method_id": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
            "target_registry_sha256": _TARGET_REGISTRY_SHA256,
            "clinical_thresholds_defined": False,
            "shared_amplitude_kernel_id": BA_IEG_BASE_NUMERICAL_KERNEL_ID,
            "shared_amplitude_targets": [
                "rms_uv",
                "peak_to_peak_uv",
                "line_length_uv_per_sample",
            ],
            "line_length_projection": (
                "shared_line_length_uv_per_sample_times_sample_differences_v1"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY = EventMorphologyPrimitivePolicy()


@dataclass(frozen=True)
class EventMorphologyPrimitiveViewInput:
    """One host-supplied native-morphology tensor and its immutable receipt."""

    view_receipt: object
    tensor: torch.Tensor


@dataclass(frozen=True)
class EventMorphologyPrimitiveQuery:
    """One requested physical interval; it carries no clinical target label."""

    view_id: str
    unit_id: str
    recording_interval_seconds: tuple[float, float]
    query_authority: str

    def __post_init__(self) -> None:
        _identifier(self.view_id, "view_id")
        _identifier(self.unit_id, "unit_id")
        if self.query_authority not in _QUERY_AUTHORITIES:
            raise ValueError(
                "query_authority must be signal-derived or synthetic; external "
                "annotations/labels are forbidden"
            )
        object.__setattr__(
            self,
            "recording_interval_seconds",
            _interval(
                self.recording_interval_seconds,
                "recording_interval_seconds",
            ),
        )


@dataclass(frozen=True)
class _PreparedView:
    receipt: dict[str, Any]
    tensor: np.ndarray
    sampling_rate_hz: float
    unit_index: Mapping[str, int]


def _prepare_views(
    canonical: Mapping[str, Any],
    inputs: Sequence[EventMorphologyPrimitiveViewInput],
    *,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, _PreparedView]:
    if not inputs:
        raise ValueError("at least one native morphology view is required")
    result: dict[str, _PreparedView] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, EventMorphologyPrimitiveViewInput):
            raise TypeError(f"views[{index}] must be EventMorphologyPrimitiveViewInput")
        receipt = validate_signal_view_receipt(
            item.view_receipt,
            canonical,
            trusted_parent_views=trusted_parent_views,
        )
        if receipt["task_role"] != "findings_native_morphology":
            raise ValueError(
                "morphology primitive supervision requires a "
                "findings_native_morphology view"
            )
        temporal = receipt["temporal_evidence"]
        if (
            temporal["future_sample_access"] is not False
            or temporal["dependency_policy"] != "instantaneous"
            or temporal["raw_support_end_policy"]
            != "at_or_before_unshifted_evidence_sample_v1"
        ):
            raise ValueError(
                "native morphology view must retain instantaneous physical support"
            )
        transform = receipt["transform_spec"]
        if (
            transform["filter"]["phase_policy"] != "none"
            or transform["normalization"]["preserves_physical_amplitude"] is not True
            or transform["clipping"]["applied"] is not False
        ):
            raise ValueError(
                "native morphology supervision requires unclipped physical amplitude "
                "and no phase-transforming filter"
            )
        view_id = _identifier(receipt["view_id"], "view_id")
        if view_id in result:
            raise ValueError("morphology view IDs must be unique")
        unit_ids = tuple(str(row["unit_id"]) for row in receipt["output_units"])
        if any(row["physical_unit"] != "V" for row in receipt["output_units"]):
            raise ValueError("native morphology view must be expressed in volts")
        tensor = item.tensor.detach().cpu().to(torch.float32).contiguous()
        expected = (
            len(unit_ids),
            int(receipt["tensor_layout"]["tensor_sample_count"]),
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"view tensor shape {tuple(tensor.shape)} != receipt {expected}"
            )
        actual_hash = deterministic_view_tensor_sha256(tensor, unit_ids=unit_ids)
        if actual_hash != receipt["processed_view_sha256"]:
            raise ValueError("processed view tensor hash does not match its receipt")
        clock = transform["output_clock"]
        sampling_rate = float(clock["sampling_rate_numerator"]) / float(
            clock["sampling_rate_denominator"]
        )
        result[view_id] = _PreparedView(
            receipt=receipt,
            tensor=tensor.numpy().astype(np.float64, copy=False),
            sampling_rate_hz=sampling_rate,
            unit_index={unit_id: i for i, unit_id in enumerate(unit_ids)},
        )
    return result


def _family_eligibility(
    view: _PreparedView,
    *,
    local_unit_index: int,
    tensor_interval: tuple[int, int],
    family: str,
) -> list[str]:
    unit = view.receipt["output_units"][local_unit_index]
    row = next(
        value for value in unit["evidence_eligibility"] if value["family"] == family
    )
    reasons = list(str(item) for item in row["reason_codes"])
    if not unit["observed"]:
        reasons.append("unit_unobserved")
    if unit["imputed"]:
        reasons.append("unit_imputed")
    for start, stop in view.receipt["masks"]["padding_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_padding_overlap")
    for start, stop in view.receipt["masks"]["edge_invalid_intervals"]:
        if _overlaps(tensor_interval, (int(start), int(stop))):
            reasons.append("view_filter_edge_overlap")
    for quality in view.receipt["masks"]["quality_invalid_intervals"]:
        if str(quality["unit_id"]) != str(unit["unit_id"]):
            continue
        quality_interval = tuple(
            int(value) for value in quality["tensor_sample_interval"]
        )
        if (
            _overlaps(tensor_interval, quality_interval)
            and family in quality["disabled_evidence_families"]
        ):
            reasons.append(f"quality_severity:{quality['severity']}")
            reasons.extend(str(item) for item in quality["reason_codes"])
    return _sorted_reasons(reasons)


def _all_quality_reasons(
    view: _PreparedView,
    *,
    local_unit_index: int,
    tensor_interval: tuple[int, int],
) -> list[str]:
    result: list[str] = []
    for family in ("waveform", "amplitude", "morphology"):
        result.extend(
            _family_eligibility(
                view,
                local_unit_index=local_unit_index,
                tensor_interval=tensor_interval,
                family=family,
            )
        )
    return _sorted_reasons(result)


def _map_query(
    view: _PreparedView,
    requested: tuple[float, float],
    *,
    minimum_samples: int,
) -> tuple[tuple[int, int], tuple[float, float]]:
    selected = tuple(
        float(value)
        for value in view.receipt["coordinates"]["selected_recording_seconds"]
    )
    if requested[0] < selected[0] - _TOL or requested[1] > selected[1] + _TOL:
        raise ValueError("morphology query lies outside its supplied view")
    start = recording_seconds_to_view_tensor_index(
        view.receipt,
        recording_seconds=requested[0],
        rounding="ceil",
    )
    stop = recording_seconds_to_view_tensor_index(
        view.receipt,
        recording_seconds=requested[1],
        rounding="floor",
    )
    if stop <= start:
        raise ValueError("morphology query contains no complete view samples")
    actual_start = view_tensor_index_to_recording_seconds(
        view.receipt,
        tensor_sample_index=start,
    )
    actual_stop = view_tensor_index_to_recording_seconds(
        view.receipt,
        tensor_sample_index=stop,
    )
    tolerance = 1.0 / view.sampling_rate_hz + _TOL
    if (
        actual_start < requested[0] - _TOL
        or actual_stop > requested[1] + _TOL
        or actual_start - requested[0] > tolerance
        or requested[1] - actual_stop > tolerance
    ):
        raise ValueError("morphology query cannot be mapped inward on the view clock")
    # Too-short support is a real opportunity outcome, not a construction
    # error.  It is retained and masked below.
    _ = minimum_samples
    return (int(start), int(stop)), (float(actual_start), float(actual_stop))


def _compress_signs(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values)
    return signs[signs != 0.0]


def _crossing_time(
    left_time: float,
    right_time: float,
    left_value: float,
    right_value: float,
    level: float,
) -> float:
    delta = right_value - left_value
    if abs(delta) <= np.finfo(np.float64).eps:
        return float((left_time + right_time) / 2.0)
    fraction = min(1.0, max(0.0, (level - left_value) / delta))
    return float(left_time + fraction * (right_time - left_time))


def _measure_primitives(
    segment_volts: np.ndarray,
    *,
    sampling_rate_hz: float,
    effective_bandwidth_hz: Sequence[float],
    actual_duration_seconds: float,
    base_masks: Sequence[bool],
    base_reasons: Sequence[Sequence[str]],
    minimum_samples: int,
) -> tuple[list[float], list[bool], list[list[str]]]:
    count = len(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
    values = np.zeros(count, dtype=np.float64)
    masks = np.asarray(base_masks, dtype=bool).copy()
    reasons = [list(_sorted_reasons(row)) for row in base_reasons]

    def mask(name: str, reason: str) -> None:
        index = _TARGET_INDEX[name]
        if reason not in reasons[index]:
            reasons[index].append(reason)
        masks[index] = False
        values[index] = 0.0

    duration_index = _TARGET_INDEX["support_duration_seconds"]
    if masks[duration_index]:
        values[duration_index] = actual_duration_seconds

    if segment_volts.size < minimum_samples:
        for name in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES[1:]:
            if masks[_TARGET_INDEX[name]]:
                mask(name, "measurement_window_too_short")
    elif not np.isfinite(segment_volts).all():
        for name in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES[1:]:
            if masks[_TARGET_INDEX[name]]:
                mask(name, "nonfinite_signal_samples")
    else:
        centered_uv = (segment_volts - np.median(segment_volts)) * 1.0e6
        dt = 1.0 / sampling_rate_hz
        derivative = np.diff(centered_uv) / dt
        curvature = np.diff(centered_uv, n=2) / (dt * dt)

        # Reuse the exact amplitude kernel already shared by BA-IEG P0 and the
        # dense sidecar.  This prevents a third, silently divergent RMS/P2P/
        # line-length implementation.  The morphology receipt projects the
        # kernel's per-sample line length to total line length on this ragged
        # query support; that projection is policy-hash bound above.
        shared = measure_ba_ieg_base_numerical_features(
            segment_volts,
            sampling_rate_hz=sampling_rate_hz,
            effective_bandwidth_hz=effective_bandwidth_hz,
            policy=BAIEGBaseNumericalPolicy(),
            amplitude_reason_codes=base_reasons[_TARGET_INDEX["rms_uv"]],
            spectral_reason_codes=("not_requested_by_morphology_primitive_sidecar",),
        )
        shared_index = {
            name: index for index, name in enumerate(BA_IEG_BASE_MEASUREMENT_NAMES)
        }
        for target_name, shared_name, multiplier in (
            ("rms_uv", "rms_uv", 1.0),
            ("peak_to_peak_uv", "peak_to_peak_uv", 1.0),
            (
                "line_length_uv",
                "line_length_uv_per_sample",
                float(segment_volts.size - 1),
            ),
        ):
            target_index = _TARGET_INDEX[target_name]
            source_index = shared_index[shared_name]
            masks[target_index] = shared.value_mask[source_index]
            reasons[target_index] = list(shared.reason_codes[source_index])
            values[target_index] = shared.values[source_index] * multiplier

        direct_values = {
            "positive_excursion_uv": float(max(0.0, np.max(centered_uv))),
            "negative_excursion_uv": float(max(0.0, -np.min(centered_uv))),
            "max_rise_slope_uv_per_s": float(max(0.0, np.max(derivative))),
            "max_fall_slope_uv_per_s": float(max(0.0, -np.min(derivative))),
            "max_abs_curvature_uv_per_s2": float(np.max(np.abs(curvature))),
        }
        median_signs = _compress_signs(centered_uv)
        derivative_signs = _compress_signs(derivative)
        direct_values["median_crossing_count"] = float(
            np.count_nonzero(median_signs[1:] != median_signs[:-1])
            if median_signs.size > 1
            else 0
        )
        direct_values["turning_point_count"] = float(
            np.count_nonzero(derivative_signs[1:] != derivative_signs[:-1])
            if derivative_signs.size > 1
            else 0
        )
        for name, value in direct_values.items():
            if masks[_TARGET_INDEX[name]]:
                values[_TARGET_INDEX[name]] = value

        peak_index = int(np.argmax(np.abs(centered_uv)))
        peak_value = float(centered_uv[peak_index])
        numerical_floor = np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(centered_uv)))
        )
        geometry_names = (
            "dominant_excursion_latency_seconds",
            "dominant_excursion_half_height_width_seconds",
            "dominant_excursion_rise_half_height_seconds",
            "dominant_excursion_fall_half_height_seconds",
            "dominant_excursion_asymmetry_ratio",
        )
        if abs(peak_value) <= numerical_floor:
            for name in geometry_names:
                if masks[_TARGET_INDEX[name]]:
                    mask(name, "dominant_excursion_degenerate")
        else:
            polarity = 1.0 if peak_value > 0.0 else -1.0
            oriented = centered_uv * polarity
            peak_height = float(oriented[peak_index])
            half_height = 0.5 * peak_height
            peak_time = peak_index * dt
            latency_name = "dominant_excursion_latency_seconds"
            if masks[_TARGET_INDEX[latency_name]]:
                values[_TARGET_INDEX[latency_name]] = peak_time

            left_candidates = np.flatnonzero(oriented[:peak_index] <= half_height)
            right_relative = np.flatnonzero(oriented[peak_index + 1 :] <= half_height)
            if left_candidates.size == 0 or right_relative.size == 0:
                for name in geometry_names[1:]:
                    if masks[_TARGET_INDEX[name]]:
                        mask(name, "dominant_excursion_half_height_censored")
            else:
                left_index = int(left_candidates[-1])
                right_index = int(peak_index + 1 + right_relative[0])
                left_cross = _crossing_time(
                    left_index * dt,
                    (left_index + 1) * dt,
                    float(oriented[left_index]),
                    float(oriented[left_index + 1]),
                    half_height,
                )
                right_cross = _crossing_time(
                    (right_index - 1) * dt,
                    right_index * dt,
                    float(oriented[right_index - 1]),
                    float(oriented[right_index]),
                    half_height,
                )
                rise = max(0.0, peak_time - left_cross)
                fall = max(0.0, right_cross - peak_time)
                width = rise + fall
                geometry_values = {
                    "dominant_excursion_half_height_width_seconds": width,
                    "dominant_excursion_rise_half_height_seconds": rise,
                    "dominant_excursion_fall_half_height_seconds": fall,
                    "dominant_excursion_asymmetry_ratio": (
                        (fall - rise) / width if width > 0.0 else 0.0
                    ),
                }
                for name, value in geometry_values.items():
                    if masks[_TARGET_INDEX[name]]:
                        values[_TARGET_INDEX[name]] = value

    for index in range(count):
        reasons[index] = _sorted_reasons(reasons[index])
        if masks[index] and reasons[index]:
            raise ValueError("available morphology target carries a reason code")
        if not masks[index] and not reasons[index]:
            reasons[index] = ["measurement_unavailable"]
        if not masks[index]:
            values[index] = 0.0
        if not math.isfinite(float(values[index])):
            raise ValueError("morphology primitive measurement is non-finite")
    return (
        [float(value) for value in values],
        [bool(value) for value in masks],
        reasons,
    )


def measure_native_morphology_primitives(
    segment_volts: np.ndarray,
    *,
    sampling_rate_hz: float,
    effective_bandwidth_hz: Sequence[float] = (0.5, 45.0),
) -> dict[str, object]:
    """Measure the established numerical morphology roster on one 1-D trace.

    The public wrapper is intentionally terminology-free: it exposes physical
    waveform primitives, masks, units and reason codes, but it cannot label a
    spike/sharp wave, seizure, onset, SOZ or diagnosis.  Adaptive native
    evidence code uses this entry point instead of duplicating the shared
    amplitude, slope, curvature and half-height geometry kernels.
    """

    values = np.asarray(segment_volts, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("native morphology measurement requires one 1-D trace")
    if not np.isfinite(values).all():
        raise ValueError("native morphology measurement requires finite samples")
    rate = _finite(sampling_rate_hz, "sampling_rate_hz", minimum=1.0)
    if len(effective_bandwidth_hz) != 2:
        raise ValueError("effective_bandwidth_hz must contain low/high cutoffs")
    low = _finite(
        effective_bandwidth_hz[0], "effective_bandwidth_hz[0]", minimum=0.0
    )
    high = _finite(
        effective_bandwidth_hz[1], "effective_bandwidth_hz[1]", minimum=0.0
    )
    if high <= low or high >= rate / 2.0 + _TOL:
        raise ValueError("effective bandwidth must be ordered and below Nyquist")
    masks = [True] * len(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
    reasons = [[] for _ in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES]
    measured, available, reason_rows = _measure_primitives(
        values,
        sampling_rate_hz=rate,
        effective_bandwidth_hz=(low, high),
        actual_duration_seconds=values.size / rate,
        base_masks=masks,
        base_reasons=reasons,
        minimum_samples=DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY.minimum_samples,
    )
    units = {
        name: unit
        for name, unit, _family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
    }
    return {
        "values": {
            name: float(measured[index])
            for index, name in enumerate(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
        },
        "available": {
            name: bool(available[index])
            for index, name in enumerate(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
        },
        "reason_codes": {
            name: list(reason_rows[index])
            for index, name in enumerate(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
        },
        "units": units,
        "method_id": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
        "policy_sha256": DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY.sha256,
        "clinical_terminology_authorized": False,
    }


def _raw_support(
    canonical: Mapping[str, Any],
    view: _PreparedView,
    *,
    local_unit_index: int,
    actual_interval: tuple[float, float],
) -> list[dict[str, object]]:
    unit = view.receipt["output_units"][local_unit_index]
    channel_catalog = {str(row["channel_id"]): row for row in canonical["channels"]}
    result: list[dict[str, object]] = []
    for channel_id in sorted(
        str(item) for item in unit["canonical_source_channel_ids"]
    ):
        if channel_id not in channel_catalog:
            raise ValueError("morphology view names an unknown canonical channel")
        channel = channel_catalog[channel_id]
        start = recording_seconds_to_canonical_sample_index(
            canonical,
            channel_id=channel_id,
            recording_seconds=actual_interval[0],
            rounding="floor",
        )
        stop = recording_seconds_to_canonical_sample_index(
            canonical,
            channel_id=channel_id,
            recording_seconds=actual_interval[1],
            rounding="ceil",
        )
        result.append(
            {
                "channel_id": channel_id,
                "sample_rate_numerator": int(channel["sample_rate_numerator"]),
                "sample_rate_denominator": int(channel["sample_rate_denominator"]),
                "raw_start_sample": int(start),
                "raw_stop_sample_exclusive": int(stop),
                "channel_sample_count": int(channel["sample_count"]),
            }
        )
    if not result:
        raise ValueError("morphology query has no canonical raw support")
    return result


def _view_binding(view: _PreparedView) -> dict[str, object]:
    receipt = view.receipt
    transform = receipt["transform_spec"]
    return {
        "view_id": str(receipt["view_id"]),
        "task_role": str(receipt["task_role"]),
        "view_receipt_id": str(receipt["view_receipt_id"]),
        "view_receipt_sha256": str(receipt["receipt_sha256"]),
        "transform_spec_sha256": str(transform["transform_spec_sha256"]),
        "processed_view_sha256": str(receipt["processed_view_sha256"]),
        "quality_mask_sha256": str(receipt["masks"]["mask_sha256"]),
        "reference_type": str(transform["reference"]["reference_type"]),
        "reference_matrix_sha256": str(transform["reference"]["matrix_sha256"]),
        "sampling_rate_hz": float(view.sampling_rate_hz),
        "output_unit_ids": sorted(view.unit_index),
        "future_sample_access": False,
        "dependency_policy": "instantaneous",
    }


def _row_status(mask: Sequence[bool]) -> str:
    count = sum(bool(item) for item in mask)
    if count == 0:
        return "not_evaluable"
    if count == len(mask):
        return "measured"
    return "partially_measured"


def materialize_event_morphology_primitive_supervision_v1(
    *,
    event_id: str,
    canonical_receipt: object,
    views: Sequence[EventMorphologyPrimitiveViewInput],
    analysis_interval_seconds: Sequence[float],
    queries: Sequence[EventMorphologyPrimitiveQuery],
    policy: EventMorphologyPrimitivePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize a content-addressed numerical morphology supervision bank."""

    _identifier(event_id, "event_id")
    if not isinstance(policy, EventMorphologyPrimitivePolicy):
        raise TypeError("policy must be EventMorphologyPrimitivePolicy")
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    analysis = _interval(analysis_interval_seconds, "analysis_interval_seconds")
    if analysis[1] > float(canonical["recording_duration_seconds"]) + _TOL:
        raise ValueError("analysis interval exceeds the canonical recording")
    if not queries:
        raise ValueError("morphology supervision requires at least one query")
    prepared = _prepare_views(
        canonical,
        views,
        trusted_parent_views=trusted_parent_views,
    )
    normalized_queries: list[EventMorphologyPrimitiveQuery] = []
    seen: set[tuple[object, ...]] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, EventMorphologyPrimitiveQuery):
            raise TypeError(f"queries[{index}] must be EventMorphologyPrimitiveQuery")
        key = (
            query.view_id,
            query.unit_id,
            *query.recording_interval_seconds,
        )
        if key in seen:
            raise ValueError("duplicate morphology query")
        seen.add(key)
        if query.view_id not in prepared:
            raise ValueError("morphology query references an unknown view")
        view = prepared[query.view_id]
        if query.unit_id not in view.unit_index:
            raise ValueError("morphology query references an unknown view unit")
        if (
            query.recording_interval_seconds[0] < analysis[0] - _TOL
            or query.recording_interval_seconds[1] > analysis[1] + _TOL
        ):
            raise ValueError(
                "morphology query lies outside the event analysis interval"
            )
        normalized_queries.append(query)
    normalized_queries.sort(
        key=lambda row: (
            row.view_id,
            row.unit_id,
            row.recording_interval_seconds[0],
            row.recording_interval_seconds[1],
        )
    )

    query_roster_body = [
        {
            "view_id": row.view_id,
            "unit_id": row.unit_id,
            "recording_interval_seconds": list(row.recording_interval_seconds),
            "query_authority": row.query_authority,
        }
        for row in normalized_queries
    ]
    query_roster_sha256 = _canonical_sha256(query_roster_body)
    rows: list[dict[str, Any]] = []
    training_values: list[list[float]] = []
    training_masks: list[list[bool]] = []
    training_row_ids: list[str] = []

    for query in normalized_queries:
        view = prepared[query.view_id]
        local_unit_index = view.unit_index[query.unit_id]
        tensor_interval, actual_interval = _map_query(
            view,
            query.recording_interval_seconds,
            minimum_samples=policy.minimum_samples,
        )
        unit = view.receipt["output_units"][local_unit_index]
        family_reasons = {
            family: _family_eligibility(
                view,
                local_unit_index=local_unit_index,
                tensor_interval=tensor_interval,
                family=family,
            )
            for family in ("waveform", "amplitude", "morphology")
        }
        base_reasons = [family_reasons[family] for family in _TARGET_FAMILY]
        base_masks = [not reason for reason in base_reasons]
        start, stop = tensor_interval
        values, value_mask, target_reasons = _measure_primitives(
            view.tensor[local_unit_index, start:stop],
            sampling_rate_hz=view.sampling_rate_hz,
            effective_bandwidth_hz=unit["effective_bandwidth_hz"],
            actual_duration_seconds=actual_interval[1] - actual_interval[0],
            base_masks=base_masks,
            base_reasons=base_reasons,
            minimum_samples=policy.minimum_samples,
        )
        query_id = (
            "MORPHQ-"
            + _canonical_sha256(
                {
                    "event_id": event_id,
                    "query": {
                        "view_id": query.view_id,
                        "unit_id": query.unit_id,
                        "recording_interval_seconds": list(
                            query.recording_interval_seconds
                        ),
                        "query_authority": query.query_authority,
                    },
                }
            )[:24]
        )
        source_binding = {
            "canonical_signal_id": str(canonical["canonical_signal_id"]),
            "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
            "source_signal_sha256": str(canonical["source_signal_sha256"]),
            "view_id": query.view_id,
            "view_receipt_id": str(view.receipt["view_receipt_id"]),
            "view_receipt_sha256": str(view.receipt["receipt_sha256"]),
            "transform_spec_sha256": str(
                view.receipt["transform_spec"]["transform_spec_sha256"]
            ),
            "processed_view_sha256": str(view.receipt["processed_view_sha256"]),
            "quality_mask_sha256": str(view.receipt["masks"]["mask_sha256"]),
            "reference_type": str(
                view.receipt["transform_spec"]["reference"]["reference_type"]
            ),
            "reference_matrix_sha256": str(
                view.receipt["transform_spec"]["reference"]["matrix_sha256"]
            ),
            "unit_id": query.unit_id,
            "query_authority": query.query_authority,
            "unit_type": str(unit["unit_type"]),
            "physical_unit": str(unit["physical_unit"]),
            "effective_bandwidth_hz": [
                float(value) for value in unit["effective_bandwidth_hz"]
            ],
            "requested_recording_interval_seconds": list(
                query.recording_interval_seconds
            ),
            "recording_interval_seconds": list(actual_interval),
            "tensor_sample_interval": list(tensor_interval),
            "raw_sample_intervals": _raw_support(
                canonical,
                view,
                local_unit_index=local_unit_index,
                actual_interval=actual_interval,
            ),
            "future_sample_access": False,
            "dependency_policy": "instantaneous",
            "policy_sha256": policy.sha256,
        }
        source_binding_sha256 = _canonical_sha256(source_binding)
        row = {
            "row_id": "MORPHROW-"
            + _canonical_sha256(
                {
                    "query_id": query_id,
                    "source_binding_sha256": source_binding_sha256,
                    "target_registry_sha256": _TARGET_REGISTRY_SHA256,
                }
            )[:24],
            "query_id": query_id,
            "assertion_level": "measured",
            "clinical_term_authorized": False,
            "source_binding": source_binding,
            "source_binding_sha256": source_binding_sha256,
            "opportunity": {
                "status": _row_status(value_mask),
                "target_value_mask": value_mask,
                "target_reason_codes": target_reasons,
                "aggregate_opportunity_reason_codes": _all_quality_reasons(
                    view,
                    local_unit_index=local_unit_index,
                    tensor_interval=tensor_interval,
                ),
            },
            "values": values,
        }
        row["row_binding_sha256"] = _self_hash(row, "row_binding_sha256")
        rows.append(row)
        if any(value_mask):
            training_values.append(values)
            training_masks.append(value_mask)
            training_row_ids.append(str(row["row_id"]))

    view_bindings = [_view_binding(prepared[view_id]) for view_id in sorted(prepared)]
    source_binding_sha256 = _canonical_sha256(
        {
            "schema_version": EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION,
            "method_id": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
            "event_id": event_id,
            "recording_id": canonical["recording_id"],
            "canonical_signal_id": canonical["canonical_signal_id"],
            "canonical_receipt_sha256": canonical["receipt_sha256"],
            "source_signal_sha256": canonical["source_signal_sha256"],
            "analysis_interval_seconds": list(analysis),
            "policy_sha256": policy.sha256,
            "target_registry_sha256": _TARGET_REGISTRY_SHA256,
            "query_roster_sha256": query_roster_sha256,
            "view_bindings": view_bindings,
            "row_binding_sha256s": [row["row_binding_sha256"] for row in rows],
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION,
        "method_id": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
        "event_id": event_id,
        "recording_id": str(canonical["recording_id"]),
        "canonical_signal_id": str(canonical["canonical_signal_id"]),
        "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
        "source_signal_sha256": str(canonical["source_signal_sha256"]),
        "analysis_interval_seconds": list(analysis),
        "coordinate_system": "recording_relative_seconds",
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "target_registry": [
            {
                "target_name": name,
                "unit_id": unit,
                "opportunity_family": family,
                "semantic_level": "numerical_measurement_only",
            }
            for name, unit, family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
        ],
        "target_registry_sha256": _TARGET_REGISTRY_SHA256,
        "query_roster": query_roster_body,
        "query_roster_sha256": query_roster_sha256,
        "view_bindings": view_bindings,
        "rows": rows,
        "training_targets": {
            "values": training_values,
            "value_mask": training_masks,
            "row_ids": training_row_ids,
            "masked_zero_is_negative_label": False,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "source_binding_sha256": source_binding_sha256,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return validate_event_morphology_primitive_supervision_v1(receipt)


def validate_event_morphology_primitive_supervision_v1(
    value: object,
) -> dict[str, Any]:
    """Validate content hashes and opportunity semantics without signal I/O."""

    if type(value) is not dict:
        raise TypeError("morphology primitive receipt must be an object")
    receipt = deepcopy(value)
    required = {
        "schema_version",
        "method_id",
        "event_id",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "analysis_interval_seconds",
        "coordinate_system",
        "policy",
        "policy_sha256",
        "target_registry",
        "target_registry_sha256",
        "query_roster",
        "query_roster_sha256",
        "view_bindings",
        "rows",
        "training_targets",
        "firewall",
        "authorization",
        "source_binding_sha256",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("morphology primitive receipt keys drifted")
    if receipt["schema_version"] != EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION:
        raise ValueError("morphology primitive schema version drifted")
    if receipt["method_id"] != EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID:
        raise ValueError("morphology primitive method drifted")
    _identifier(receipt["event_id"], "event_id")
    _identifier(receipt["recording_id"], "recording_id")
    _identifier(receipt["canonical_signal_id"], "canonical_signal_id")
    _sha256(receipt["canonical_receipt_sha256"], "canonical_receipt_sha256")
    _sha256(receipt["source_signal_sha256"], "source_signal_sha256")
    _sha256(receipt["policy_sha256"], "policy_sha256")
    _sha256(receipt["target_registry_sha256"], "target_registry_sha256")
    _sha256(receipt["query_roster_sha256"], "query_roster_sha256")
    _sha256(receipt["source_binding_sha256"], "source_binding_sha256")
    _sha256(receipt["receipt_sha256"], "receipt_sha256")
    _interval(receipt["analysis_interval_seconds"], "analysis_interval_seconds")
    if receipt["coordinate_system"] != "recording_relative_seconds":
        raise ValueError("morphology primitive coordinate system drifted")
    if receipt["firewall"] != _FIREWALL:
        raise ValueError("morphology primitive EEG-only firewall drifted")
    if receipt["authorization"] != _AUTHORIZATION:
        raise ValueError("morphology primitive authorization drifted")
    if receipt["policy_sha256"] != _canonical_sha256(receipt["policy"]):
        raise ValueError("morphology primitive policy hash mismatch")
    policy_body = receipt["policy"]
    if type(policy_body) is not dict:
        raise TypeError("morphology primitive policy must be an object")
    expected_policy_keys = {
        "minimum_samples",
        "centering",
        "crossing_interpolation",
        "tie_breaking",
        "policy_id",
        "method_id",
        "target_registry_sha256",
        "clinical_thresholds_defined",
        "shared_amplitude_kernel_id",
        "shared_amplitude_targets",
        "line_length_projection",
    }
    if set(policy_body) != expected_policy_keys:
        raise ValueError("morphology primitive policy keys drifted")
    replayed_policy = EventMorphologyPrimitivePolicy(
        minimum_samples=policy_body["minimum_samples"],
        centering=policy_body["centering"],
        crossing_interpolation=policy_body["crossing_interpolation"],
        tie_breaking=policy_body["tie_breaking"],
    )
    if policy_body != replayed_policy.to_dict():
        raise ValueError("morphology primitive policy semantics drifted")
    if receipt["policy_sha256"] != replayed_policy.sha256:
        raise ValueError("morphology primitive replayed policy hash mismatch")
    expected_registry = [
        {
            "target_name": name,
            "unit_id": unit,
            "opportunity_family": family,
            "semantic_level": "numerical_measurement_only",
        }
        for name, unit, family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
    ]
    if receipt["target_registry"] != expected_registry:
        raise ValueError("morphology primitive target registry drifted")
    registry_hash = _canonical_sha256(
        [
            {
                "target_name": name,
                "unit_id": unit,
                "opportunity_family": family,
            }
            for name, unit, family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
        ]
    )
    if receipt["target_registry_sha256"] != registry_hash:
        raise ValueError("morphology primitive target registry hash mismatch")
    if receipt["query_roster_sha256"] != _canonical_sha256(receipt["query_roster"]):
        raise ValueError("morphology primitive query roster hash mismatch")
    view_bindings = receipt["view_bindings"]
    if not isinstance(view_bindings, list) or not view_bindings:
        raise ValueError("morphology primitive view bindings must be non-empty")
    view_ids: list[str] = []
    for index, binding in enumerate(view_bindings):
        if type(binding) is not dict:
            raise TypeError(f"view_bindings[{index}] must be an object")
        view_id = _identifier(binding.get("view_id"), "view binding view_id")
        view_ids.append(view_id)
        if binding.get("task_role") != "findings_native_morphology":
            raise ValueError("morphology primitive view role drifted")
        for field in (
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_matrix_sha256",
        ):
            _sha256(binding.get(field), f"view_bindings[{index}].{field}")
        if (
            binding.get("future_sample_access") is not False
            or binding.get("dependency_policy") != "instantaneous"
        ):
            raise ValueError("morphology primitive view dependency drifted")
    if view_ids != sorted(view_ids) or len(view_ids) != len(set(view_ids)):
        raise ValueError("morphology primitive view bindings are not canonical")

    query_roster = receipt["query_roster"]
    if not isinstance(query_roster, list) or not query_roster:
        raise ValueError("morphology primitive query roster must be non-empty")
    normalized_query_keys: list[tuple[str, str, float, float]] = []
    for index, query in enumerate(query_roster):
        if type(query) is not dict or set(query) != {
            "view_id",
            "unit_id",
            "recording_interval_seconds",
            "query_authority",
        }:
            raise ValueError(f"query_roster[{index}] keys drifted")
        query_view = _identifier(query["view_id"], "query view_id")
        query_unit = _identifier(query["unit_id"], "query unit_id")
        if query["query_authority"] not in _QUERY_AUTHORITIES:
            raise ValueError("morphology primitive query authority is forbidden")
        query_interval = _interval(
            query["recording_interval_seconds"],
            f"query_roster[{index}].recording_interval_seconds",
        )
        normalized_query_keys.append(
            (query_view, query_unit, query_interval[0], query_interval[1])
        )
    if normalized_query_keys != sorted(normalized_query_keys) or len(
        normalized_query_keys
    ) != len(set(normalized_query_keys)):
        raise ValueError("morphology primitive query roster is not canonical")
    rows = receipt["rows"]
    if not isinstance(rows, list) or len(rows) != len(receipt["query_roster"]):
        raise ValueError("morphology primitive rows and query roster do not align")
    target_count = len(EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES)
    included_values: list[list[float]] = []
    included_masks: list[list[bool]] = []
    included_ids: list[str] = []
    seen_row_ids: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise TypeError(f"rows[{index}] must be an object")
        if set(row) != {
            "row_id",
            "query_id",
            "assertion_level",
            "clinical_term_authorized",
            "source_binding",
            "source_binding_sha256",
            "opportunity",
            "values",
            "row_binding_sha256",
        }:
            raise ValueError("morphology primitive row keys drifted")
        if row.get("row_binding_sha256") != _self_hash(row, "row_binding_sha256"):
            raise ValueError("morphology primitive row binding hash mismatch")
        row_id = _identifier(row.get("row_id"), f"rows[{index}].row_id")
        if row_id in seen_row_ids:
            raise ValueError("morphology primitive row IDs must be unique")
        seen_row_ids.add(row_id)
        _identifier(row.get("query_id"), f"rows[{index}].query_id")
        if row.get("assertion_level") != "measured":
            raise ValueError("morphology primitive assertion level drifted")
        if row.get("clinical_term_authorized") is not False:
            raise ValueError("morphology primitive row authorized a clinical term")
        source = row.get("source_binding")
        if type(source) is not dict:
            raise TypeError("morphology primitive source binding must be an object")
        if row.get("source_binding_sha256") != _canonical_sha256(source):
            raise ValueError("morphology primitive source binding hash mismatch")
        query = query_roster[index]
        if (
            source.get("view_id") != query["view_id"]
            or source.get("unit_id") != query["unit_id"]
            or source.get("query_authority") != query["query_authority"]
            or source.get("requested_recording_interval_seconds")
            != query["recording_interval_seconds"]
        ):
            raise ValueError("morphology primitive row/query binding drifted")
        expected_query_id = (
            "MORPHQ-"
            + _canonical_sha256(
                {
                    "event_id": receipt["event_id"],
                    "query": query,
                }
            )[:24]
        )
        if row["query_id"] != expected_query_id:
            raise ValueError("morphology primitive query ID drifted")
        expected_row_id = (
            "MORPHROW-"
            + _canonical_sha256(
                {
                    "query_id": expected_query_id,
                    "source_binding_sha256": row["source_binding_sha256"],
                    "target_registry_sha256": receipt["target_registry_sha256"],
                }
            )[:24]
        )
        if row_id != expected_row_id:
            raise ValueError("morphology primitive row ID drifted")
        if source.get("physical_unit") != "V":
            raise ValueError("morphology primitive source lost physical volts")
        if (
            source.get("future_sample_access") is not False
            or source.get("dependency_policy") != "instantaneous"
        ):
            raise ValueError("morphology primitive raw dependency drifted")
        values = row.get("values")
        opportunity = row.get("opportunity")
        if not isinstance(values, list) or len(values) != target_count:
            raise ValueError("morphology primitive value vocabulary drifted")
        if type(opportunity) is not dict:
            raise TypeError("morphology primitive opportunity must be an object")
        if set(opportunity) != {
            "status",
            "target_value_mask",
            "target_reason_codes",
            "aggregate_opportunity_reason_codes",
        }:
            raise ValueError("morphology primitive opportunity keys drifted")
        masks = opportunity.get("target_value_mask")
        reasons = opportunity.get("target_reason_codes")
        if not isinstance(masks, list) or len(masks) != target_count:
            raise ValueError("morphology primitive target mask vocabulary drifted")
        if not isinstance(reasons, list) or len(reasons) != target_count:
            raise ValueError("morphology primitive target reasons drifted")
        normalized_values: list[float] = []
        normalized_masks: list[bool] = []
        for target_index, raw_value in enumerate(values):
            numeric = _finite(
                raw_value,
                f"rows[{index}].values[{target_index}]",
            )
            available = masks[target_index]
            if not isinstance(available, bool):
                raise TypeError("morphology primitive masks must be booleans")
            normalized_reason = _sorted_reasons(reasons[target_index])
            if available and normalized_reason:
                raise ValueError("available morphology target carries reasons")
            if not available and not normalized_reason:
                raise ValueError("masked morphology target lacks a reason")
            if not available and numeric != 0.0:
                raise ValueError("masked morphology target must serialize zero")
            normalized_values.append(numeric)
            normalized_masks.append(available)
        expected_status = _row_status(normalized_masks)
        if opportunity.get("status") != expected_status:
            raise ValueError("morphology primitive opportunity status drifted")
        if any(normalized_masks):
            included_values.append(normalized_values)
            included_masks.append(normalized_masks)
            included_ids.append(row_id)
    expected_training = {
        "values": included_values,
        "value_mask": included_masks,
        "row_ids": included_ids,
        "masked_zero_is_negative_label": False,
    }
    if receipt["training_targets"] != expected_training:
        raise ValueError("morphology primitive compact training targets drifted")
    aggregate = _canonical_sha256(
        {
            "schema_version": EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION,
            "method_id": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
            "event_id": receipt["event_id"],
            "recording_id": receipt["recording_id"],
            "canonical_signal_id": receipt["canonical_signal_id"],
            "canonical_receipt_sha256": receipt["canonical_receipt_sha256"],
            "source_signal_sha256": receipt["source_signal_sha256"],
            "analysis_interval_seconds": receipt["analysis_interval_seconds"],
            "policy_sha256": receipt["policy_sha256"],
            "target_registry_sha256": receipt["target_registry_sha256"],
            "query_roster_sha256": receipt["query_roster_sha256"],
            "view_bindings": receipt["view_bindings"],
            "row_binding_sha256s": [row["row_binding_sha256"] for row in rows],
        }
    )
    if receipt["source_binding_sha256"] != aggregate:
        raise ValueError("morphology primitive aggregate source binding mismatch")
    if receipt["receipt_sha256"] != _self_hash(receipt, "receipt_sha256"):
        raise ValueError("morphology primitive receipt hash mismatch")
    return receipt


def replay_event_morphology_primitive_supervision_v1(
    expected_receipt: object,
    *,
    event_id: str,
    canonical_receipt: object,
    views: Sequence[EventMorphologyPrimitiveViewInput],
    analysis_interval_seconds: Sequence[float],
    queries: Sequence[EventMorphologyPrimitiveQuery],
    policy: EventMorphologyPrimitivePolicy = (
        DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Recompute every target and fail if the serialized receipt differs."""

    expected = validate_event_morphology_primitive_supervision_v1(expected_receipt)
    replayed = materialize_event_morphology_primitive_supervision_v1(
        event_id=event_id,
        canonical_receipt=canonical_receipt,
        views=views,
        analysis_interval_seconds=analysis_interval_seconds,
        queries=queries,
        policy=policy,
        trusted_parent_views=trusted_parent_views,
    )
    if replayed != expected:
        raise ValueError("morphology primitive receipt does not replay")
    return replayed


__all__ = [
    "DEFAULT_EVENT_MORPHOLOGY_PRIMITIVE_POLICY",
    "EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID",
    "EVENT_MORPHOLOGY_PRIMITIVE_POLICY_ID",
    "EVENT_MORPHOLOGY_PRIMITIVE_SCHEMA_VERSION",
    "EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES",
    "EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS",
    "EventMorphologyPrimitivePolicy",
    "EventMorphologyPrimitiveQuery",
    "EventMorphologyPrimitiveViewInput",
    "materialize_event_morphology_primitive_supervision_v1",
    "measure_native_morphology_primitives",
    "replay_event_morphology_primitive_supervision_v1",
    "validate_event_morphology_primitive_supervision_v1",
]
