"""Replayable signal-only producer for the S04 physical-amplitude slot.

The producer is deliberately narrower than a clinical EEG interpretation.  It
measures calibrated RMS and peak-to-peak amplitude in microvolts, constructs a
time-ordered amplitude trajectory, and computes a dimensionless RMS ratio to
an explicitly signal-selected comparison context.  It never turns a low ratio
into ``attenuation``, ``electrodecrement``, ictal evolution, onset, or SOZ.

All numerical values are replayed through the shared BA-IEG numerical kernel
and therefore require an unclipped physical-volts view, immutable signal/view
hashes, physical time, and amplitude-family opportunity.  A spatial-reference
child is allowed only when its single trusted parent is the native morphology
view.  Such an output remains one atomic unit: a bipolar derivation is emitted
as one whole ``lead`` and its voltage difference is never assigned to either
endpoint.  The producer is additive and does not alter or depend on the frozen
morphology-primitive receipt.

The query roster is signal-only.  EDF annotations, spreadsheets, doctor text,
clinical metadata, video/behaviour and LLM output have no input field.
Comparison contexts are candidates selected by an upstream signal-only method;
the receipt does not claim that they are clinically normal background.
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
    BA_IEG_BASE_MEASUREMENT_NAMES,
    BA_IEG_BASE_NUMERICAL_KERNEL_ID,
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
from .deterministic_event_findings import (
    deterministic_view_tensor_sha256,
)


EVENT_PHYSICAL_AMPLITUDE_FINDINGS_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_event_physical_amplitude_findings_v1"
EVENT_PHYSICAL_AMPLITUDE_FINDINGS_METHOD_ID: Final[
    str
] = "DETERMINISTIC-EVENT-PHYSICAL-AMPLITUDE-FINDINGS-V1"
EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY_ID: Final[
    str
] = "DETERMINISTIC-EVENT-PHYSICAL-AMPLITUDE-POLICY-V1"
EVENT_PHYSICAL_AMPLITUDE_SOURCE_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_event_physical_amplitude_source_v1"
EVENT_PHYSICAL_AMPLITUDE_SOURCE_METHOD_ID: Final[
    str
] = "DETERMINISTIC-EVENT-PHYSICAL-AMPLITUDE-SOURCE-V1"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-9
_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1.0e-3, "uV": 1.0e-6}
_QUERY_AUTHORITIES = frozenset(
    {
        "deterministic_signal_proposal",
        "frozen_model_proposal",
        "synthetic_signal_injection",
    }
)
_MEASUREMENT_ROLES = frozenset({"signal_selected_comparison_context", "event_course"})
_BASE_TARGET_INDEX = {
    name: index for index, name in enumerate(BA_IEG_BASE_MEASUREMENT_NAMES)
}
_AMPLITUDE_SOURCE_TARGETS: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
)
_AMPLITUDE_SOURCE_TARGET_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(_AMPLITUDE_SOURCE_TARGETS)
}

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "event_card_slot_id": "S04_PHYSICAL_AMPLITUDE",
    "projection_scope": "physical_measurements_and_relative_ratios_only",
    "clinical_attenuation_term_authorized": False,
    "electrodecrement_term_authorized": False,
    "amplitude_change_as_evolution_authorized": False,
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
}

_SOURCE_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "measurement_scope": "calibrated_rms_and_peak_to_peak_only",
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a contract-compatible identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum - _TOL:
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


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] - _TOL and right[0] < left[1] - _TOL


def _sorted_reasons(values: Sequence[str]) -> list[str]:
    result = sorted(set(str(item) for item in values))
    if any(not item or item != item.strip() for item in result):
        raise ValueError("reason codes must be non-empty trimmed strings")
    return result


@dataclass(frozen=True)
class EventPhysicalAmplitudeFindingsPolicy:
    """Engineering measurement policy; none of its values is clinical."""

    minimum_sample_rate_hz: float = 200.0
    required_bandwidth_low_hz: float = 0.5
    required_bandwidth_high_hz: float = 45.0
    minimum_course_points: int = 2
    ratio_denominator_floor_uv: float = 1.0e-6
    context_aggregation: str = "median_of_measured_context_windows"
    transition_policy: str = "adjacent_scheduled_course_points_only"

    def __post_init__(self) -> None:
        _finite(
            self.minimum_sample_rate_hz,
            "minimum_sample_rate_hz",
            minimum=_TOL,
        )
        low = _finite(
            self.required_bandwidth_low_hz,
            "required_bandwidth_low_hz",
            minimum=0.0,
        )
        high = _finite(
            self.required_bandwidth_high_hz,
            "required_bandwidth_high_hz",
            minimum=_TOL,
        )
        if high <= low + _TOL:
            raise ValueError("required physical-amplitude bandwidth is empty")
        if (
            isinstance(self.minimum_course_points, bool)
            or not isinstance(self.minimum_course_points, int)
            or self.minimum_course_points < 2
        ):
            raise ValueError("minimum_course_points must be an integer >= 2")
        _finite(
            self.ratio_denominator_floor_uv,
            "ratio_denominator_floor_uv",
            minimum=np.finfo(np.float64).tiny,
        )
        if self.context_aggregation != "median_of_measured_context_windows":
            raise ValueError("v1 freezes median context aggregation")
        if self.transition_policy != "adjacent_scheduled_course_points_only":
            raise ValueError("v1 freezes adjacent-only trajectory transitions")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "policy_id": EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY_ID,
            "method_id": EVENT_PHYSICAL_AMPLITUDE_FINDINGS_METHOD_ID,
            "measurement_units": {
                "rms": "uV",
                "peak_to_peak": "uV",
                "attenuation_ratio": "dimensionless_event_rms_over_context_rms",
            },
            "clinical_thresholds_defined": False,
            "amplitude_alone_can_qualify_evolution": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY = (
    EventPhysicalAmplitudeFindingsPolicy()
)


@dataclass(frozen=True)
class EventPhysicalAmplitudeViewInput:
    """Host-supplied physical signal view and the exact tensor it hashes."""

    view_receipt: object
    tensor: torch.Tensor


@dataclass(frozen=True)
class EventPhysicalAmplitudeQuery:
    """One signal-selected physical interval with no clinical target label."""

    view_id: str
    unit_id: str
    recording_interval_seconds: tuple[float, float]
    measurement_role: str
    comparison_set_id: str
    ordinal: int
    query_authority: str
    selection_receipt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.view_id, "view_id")
        _identifier(self.unit_id, "unit_id")
        _identifier(self.comparison_set_id, "comparison_set_id")
        if self.measurement_role not in _MEASUREMENT_ROLES:
            raise ValueError("measurement_role is unsupported")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 1
        ):
            raise ValueError("ordinal must be a positive integer")
        if self.query_authority not in _QUERY_AUTHORITIES:
            raise ValueError(
                "query_authority must be signal-derived or synthetic; annotations, "
                "labels and clinical text are forbidden"
            )
        _sha(self.selection_receipt_sha256, "selection_receipt_sha256")
        object.__setattr__(
            self,
            "recording_interval_seconds",
            _interval(
                self.recording_interval_seconds,
                "recording_interval_seconds",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "unit_id": self.unit_id,
            "recording_interval_seconds": list(self.recording_interval_seconds),
            "measurement_role": self.measurement_role,
            "comparison_set_id": self.comparison_set_id,
            "ordinal": self.ordinal,
            "query_authority": self.query_authority,
            "selection_receipt_sha256": self.selection_receipt_sha256,
        }


def _query_sort_key(query: EventPhysicalAmplitudeQuery) -> tuple[object, ...]:
    role_order = (
        0 if query.measurement_role == "signal_selected_comparison_context" else 1
    )
    return (
        query.view_id,
        query.unit_id,
        query.comparison_set_id,
        role_order,
        query.ordinal,
        query.recording_interval_seconds[0],
        query.recording_interval_seconds[1],
    )


def _normalize_queries(
    queries: Sequence[EventPhysicalAmplitudeQuery],
    *,
    analysis_interval: tuple[float, float],
) -> list[EventPhysicalAmplitudeQuery]:
    if not queries:
        raise ValueError("physical-amplitude producer requires at least one query")
    result: list[EventPhysicalAmplitudeQuery] = []
    interval_keys: set[tuple[str, str, float, float]] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, EventPhysicalAmplitudeQuery):
            raise TypeError(f"queries[{index}] must be EventPhysicalAmplitudeQuery")
        start, stop = query.recording_interval_seconds
        if start < analysis_interval[0] - _TOL or stop > analysis_interval[1] + _TOL:
            raise ValueError("physical-amplitude query lies outside event analysis")
        interval_key = (query.view_id, query.unit_id, start, stop)
        if interval_key in interval_keys:
            raise ValueError(
                "one physical view/unit interval cannot carry multiple semantic roles"
            )
        interval_keys.add(interval_key)
        result.append(query)
    result.sort(key=_query_sort_key)

    grouped: dict[tuple[str, str, str], list[EventPhysicalAmplitudeQuery]] = {}
    for query in result:
        grouped.setdefault(
            (query.view_id, query.unit_id, query.comparison_set_id), []
        ).append(query)
    if not any(query.measurement_role == "event_course" for query in result):
        raise ValueError("physical-amplitude producer requires event-course queries")
    for key, rows in grouped.items():
        by_role = {
            role: [row for row in rows if row.measurement_role == role]
            for role in _MEASUREMENT_ROLES
        }
        for role, role_rows in by_role.items():
            if not role_rows:
                continue
            ordered = sorted(role_rows, key=lambda row: row.ordinal)
            if [row.ordinal for row in ordered] != list(range(1, len(ordered) + 1)):
                raise ValueError(f"{role} ordinals must be contiguous from one")
            intervals = [row.recording_interval_seconds for row in ordered]
            if intervals != sorted(intervals):
                raise ValueError(f"{role} ordinals must follow physical time")
            if any(
                _overlaps(left, right) for left, right in zip(intervals, intervals[1:])
            ):
                raise ValueError(f"{role} measurement intervals must not overlap")
        contexts = by_role["signal_selected_comparison_context"]
        course = by_role["event_course"]
        if not course:
            raise ValueError(
                f"physical-amplitude comparison group {key} has no event course"
            )
        if course and any(
            _overlaps(
                context.recording_interval_seconds, point.recording_interval_seconds
            )
            for context in contexts
            for point in course
        ):
            raise ValueError(
                f"comparison context overlaps event course for group {key}"
            )
    return result


def build_event_physical_amplitude_queries_v1(
    *,
    view_id: str,
    unit_ids: Sequence[str],
    event_course_interval_seconds: tuple[float, float],
    comparison_context_intervals_seconds: Sequence[tuple[float, float]],
    comparison_set_id: str,
    selection_receipt_sha256: str,
    window_seconds: float = 1.0,
    step_seconds: float = 1.0,
    query_authority: str = "frozen_model_proposal",
) -> tuple[EventPhysicalAmplitudeQuery, ...]:
    """Build a deterministic non-overlapping query roster.

    The caller owns event/context selection and must bind it with a signal-only
    selection receipt.  This helper only tiles the supplied physical intervals;
    it does not inspect annotations or infer a clinical baseline.
    """

    _identifier(view_id, "view_id")
    units = tuple(_identifier(item, "unit_ids") for item in unit_ids)
    if not units or len(units) != len(set(units)):
        raise ValueError("unit_ids must be non-empty and unique")
    _identifier(comparison_set_id, "comparison_set_id")
    _sha(selection_receipt_sha256, "selection_receipt_sha256")
    if query_authority not in _QUERY_AUTHORITIES:
        raise ValueError("query_authority is not signal-only")
    window = _finite(window_seconds, "window_seconds", minimum=_TOL)
    step = _finite(step_seconds, "step_seconds", minimum=window)
    event_interval = _interval(
        event_course_interval_seconds, "event_course_interval_seconds"
    )
    contexts = [
        _interval(value, f"comparison_context_intervals_seconds[{index}]")
        for index, value in enumerate(comparison_context_intervals_seconds)
    ]
    if any(
        _overlaps(left, right)
        for index, left in enumerate(contexts)
        for right in contexts[index + 1 :]
    ):
        raise ValueError("comparison context intervals overlap each other")
    if any(_overlaps(context, event_interval) for context in contexts):
        raise ValueError("comparison context intervals overlap the event course")

    def tile(interval: tuple[float, float]) -> list[tuple[float, float]]:
        rows: list[tuple[float, float]] = []
        start = interval[0]
        while start + window <= interval[1] + _TOL:
            stop = min(interval[1], start + window)
            rows.append((float(start), float(stop)))
            start += step
        return rows

    context_windows = [window for interval in contexts for window in tile(interval)]
    course_windows = tile(event_interval)
    if not course_windows:
        raise ValueError("event course contains no complete amplitude window")
    result: list[EventPhysicalAmplitudeQuery] = []
    for unit_id in units:
        for ordinal, interval in enumerate(context_windows, start=1):
            result.append(
                EventPhysicalAmplitudeQuery(
                    view_id=view_id,
                    unit_id=unit_id,
                    recording_interval_seconds=interval,
                    measurement_role="signal_selected_comparison_context",
                    comparison_set_id=comparison_set_id,
                    ordinal=ordinal,
                    query_authority=query_authority,
                    selection_receipt_sha256=selection_receipt_sha256,
                )
            )
        for ordinal, interval in enumerate(course_windows, start=1):
            result.append(
                EventPhysicalAmplitudeQuery(
                    view_id=view_id,
                    unit_id=unit_id,
                    recording_interval_seconds=interval,
                    measurement_role="event_course",
                    comparison_set_id=comparison_set_id,
                    ordinal=ordinal,
                    query_authority=query_authority,
                    selection_receipt_sha256=selection_receipt_sha256,
                )
            )
    return tuple(sorted(result, key=_query_sort_key))


def _validated_view_catalog(
    canonical: Mapping[str, Any],
    views: Sequence[EventPhysicalAmplitudeViewInput],
    *,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, "_PreparedAmplitudeView"]]:
    if not views:
        raise ValueError("at least one physical-amplitude view is required")
    catalog: dict[str, dict[str, Any]] = {}
    prepared: dict[str, _PreparedAmplitudeView] = {}
    trusted = {} if trusted_parent_views is None else trusted_parent_views
    for index, item in enumerate(views):
        if not isinstance(item, EventPhysicalAmplitudeViewInput):
            raise TypeError(f"views[{index}] must be EventPhysicalAmplitudeViewInput")
        receipt = validate_signal_view_receipt(
            item.view_receipt,
            canonical,
            trusted_parent_views=trusted_parent_views,
        )
        view_id = _identifier(receipt["view_id"], "view_id")
        if view_id in catalog:
            raise ValueError("physical-amplitude view IDs must be unique")
        task_role = str(receipt["task_role"])
        if task_role not in {"findings_native_morphology", "spatial_reference"}:
            raise ValueError(
                "physical amplitude requires a native morphology view or its "
                "instantaneous spatial-reference child"
            )
        temporal = receipt["temporal_evidence"]
        if (
            temporal["future_sample_access"] is not False
            or temporal["dependency_policy"] != "instantaneous"
            or temporal["raw_support_end_policy"]
            != "at_or_before_unshifted_evidence_sample_v1"
        ):
            raise ValueError(
                "physical-amplitude view must retain instantaneous physical support"
            )
        transform = receipt["transform_spec"]
        if (
            transform["filter"]["family"] != "none"
            or transform["filter"]["phase_policy"] != "none"
            or int(transform["resampler"]["up"]) != 1
            or int(transform["resampler"]["down"]) != 1
            or transform["normalization"]["preserves_physical_amplitude"] is not True
            or transform["clipping"]["applied"] is not False
        ):
            raise ValueError(
                "physical-amplitude source requires unfiltered, unresampled, "
                "unclipped physical amplitude"
            )
        unit_ids = tuple(str(row["unit_id"]) for row in receipt["output_units"])
        if any(row["physical_unit"] != "V" for row in receipt["output_units"]):
            raise ValueError("physical-amplitude view must be expressed in volts")

        native_parent_binding: dict[str, object] | None = None
        parent_bindings = receipt["parent_view_bindings"]
        if task_role == "findings_native_morphology":
            if parent_bindings:
                raise ValueError("native morphology amplitude view must be canonical")
        else:
            if len(parent_bindings) != 1:
                raise ValueError(
                    "spatial-reference amplitude view requires one native parent"
                )
            parent_id = str(parent_bindings[0]["view_id"])
            if parent_id not in trusted:
                raise ValueError(
                    "spatial-reference amplitude view requires its host-supplied "
                    "native parent"
                )
            parent = validate_signal_view_receipt(trusted[parent_id], canonical)
            if parent["task_role"] != "findings_native_morphology":
                raise ValueError(
                    "spatial-reference amplitude parent must be native morphology"
                )
            if str(parent["receipt_sha256"]) != str(
                parent_bindings[0]["receipt_sha256"]
            ):
                raise ValueError("trusted native parent binding hash drifted")
            native_parent_binding = {
                "view_id": str(parent["view_id"]),
                "view_receipt_sha256": str(parent["receipt_sha256"]),
                "processed_view_sha256": str(parent["processed_view_sha256"]),
                "task_role": "findings_native_morphology",
            }

        tensor = item.tensor.detach().cpu().to(torch.float32).contiguous()
        expected_shape = (
            len(unit_ids),
            int(receipt["tensor_layout"]["tensor_sample_count"]),
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"view tensor shape {tuple(tensor.shape)} != receipt {expected_shape}"
            )
        actual_hash = deterministic_view_tensor_sha256(tensor, unit_ids=unit_ids)
        if actual_hash != receipt["processed_view_sha256"]:
            raise ValueError("processed view tensor hash does not match its receipt")
        clock = transform["output_clock"]
        sampling_rate = float(clock["sampling_rate_numerator"]) / float(
            clock["sampling_rate_denominator"]
        )
        catalog[view_id] = receipt
        prepared[view_id] = _PreparedAmplitudeView(
            receipt=receipt,
            tensor=tensor.numpy().astype(np.float64, copy=False),
            sampling_rate_hz=sampling_rate,
            unit_index={unit_id: offset for offset, unit_id in enumerate(unit_ids)},
            native_parent_binding=native_parent_binding,
        )
    return catalog, prepared


@dataclass(frozen=True)
class _PreparedAmplitudeView:
    receipt: dict[str, Any]
    tensor: np.ndarray
    sampling_rate_hz: float
    unit_index: Mapping[str, int]
    native_parent_binding: Mapping[str, object] | None


def _amplitude_family_reasons(
    view: _PreparedAmplitudeView,
    *,
    local_unit_index: int,
    tensor_interval: tuple[int, int],
) -> list[str]:
    unit = view.receipt["output_units"][local_unit_index]
    eligibility = next(
        row for row in unit["evidence_eligibility"] if row["family"] == "amplitude"
    )
    reasons = list(str(item) for item in eligibility["reason_codes"])
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
            and "amplitude" in quality["disabled_evidence_families"]
        ):
            reasons.append(f"quality_severity:{quality['severity']}")
            reasons.extend(str(item) for item in quality["reason_codes"])
    return _sorted_reasons(reasons)


def _map_amplitude_query(
    view: _PreparedAmplitudeView,
    requested: tuple[float, float],
) -> tuple[tuple[int, int], tuple[float, float]]:
    selected = tuple(
        float(value)
        for value in view.receipt["coordinates"]["selected_recording_seconds"]
    )
    if requested[0] < selected[0] - _TOL or requested[1] > selected[1] + _TOL:
        raise ValueError("physical-amplitude query lies outside its supplied view")
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
        raise ValueError("physical-amplitude query contains no complete view samples")
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
        raise ValueError(
            "physical-amplitude query cannot be mapped inward on the view clock"
        )
    return (int(start), int(stop)), (float(actual_start), float(actual_stop))


def _raw_amplitude_support(
    canonical: Mapping[str, Any],
    view: _PreparedAmplitudeView,
    *,
    local_unit_index: int,
    actual_interval: tuple[float, float],
) -> list[dict[str, object]]:
    unit = view.receipt["output_units"][local_unit_index]
    channel_catalog = {str(row["channel_id"]): row for row in canonical["channels"]}
    result: list[dict[str, object]] = []
    for channel_id in sorted(str(item) for item in unit["canonical_source_channel_ids"]):
        channel = channel_catalog[channel_id]
        if not channel["observed"] or int(channel["sample_count"]) == 0:
            # Preserve the missing carrier in the lineage without inventing a
            # physical sample interval.  Opportunity logic already marks any
            # unit depending on it as not evaluable.
            start = 0
            stop = 0
        else:
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
        raise ValueError("physical-amplitude query has no canonical raw support")
    return result


def _calibration_ledger(
    canonical: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    queries: Sequence[EventPhysicalAmplitudeQuery],
) -> list[dict[str, Any]]:
    channel_by_id = {str(row["channel_id"]): row for row in canonical["channels"]}
    requested = sorted({(row.view_id, row.unit_id) for row in queries})
    ledger: list[dict[str, Any]] = []
    for view_id, unit_id in requested:
        if view_id not in views:
            raise ValueError("physical-amplitude query references an unknown view")
        view = views[view_id]
        units = {str(row["unit_id"]): row for row in view["output_units"]}
        if unit_id not in units:
            raise ValueError("physical-amplitude query references an unknown unit")
        unit = units[unit_id]
        transform = view["transform_spec"]
        output_ids = list(transform["output_unit_ids"])
        row_index = output_ids.index(unit_id)
        coefficients = [
            {"input_unit_id": input_id, "coefficient": float(coefficient)}
            for input_id, coefficient in zip(
                transform["input_unit_ids"],
                transform["reference"]["matrix"][row_index],
            )
            if abs(float(coefficient)) > _TOL
        ]
        source_channels = []
        for channel_id in unit["canonical_source_channel_ids"]:
            channel = channel_by_id[str(channel_id)]
            source_channels.append(
                {
                    "channel_id": str(channel["channel_id"]),
                    "source_physical_unit": str(channel["source_physical_unit"]),
                    "scale_to_volts": float(channel["scale_to_volts"]),
                    "observed": bool(channel["observed"]),
                    "imputed": bool(channel["imputed"]),
                    "acquisition_highpass_hz": channel["acquisition_highpass_hz"],
                    "acquisition_lowpass_hz": channel["acquisition_lowpass_hz"],
                    "reference_label": str(channel["reference_label"]),
                }
            )
        amplitude = next(
            row for row in unit["evidence_eligibility"] if row["family"] == "amplitude"
        )
        clock = transform["output_clock"]
        body: dict[str, Any] = {
            "view_id": view_id,
            "view_receipt_id": str(view["view_receipt_id"]),
            "view_receipt_sha256": str(view["receipt_sha256"]),
            "transform_spec_sha256": str(transform["transform_spec_sha256"]),
            "processed_view_sha256": str(view["processed_view_sha256"]),
            "quality_mask_sha256": str(view["masks"]["mask_sha256"]),
            "unit_id": unit_id,
            "unit_type": str(unit["unit_type"]),
            "whole_output_unit_identity_preserved": True,
            "bipolar_endpoint_fact_projection_authorized": False,
            "source_channels_are_calibration_lineage_not_evidence_units": True,
            "output_physical_unit": str(unit["physical_unit"]),
            "emitted_amplitude_unit": "uV",
            "sampling_rate_numerator": int(clock["sampling_rate_numerator"]),
            "sampling_rate_denominator": int(clock["sampling_rate_denominator"]),
            "effective_bandwidth_hz": [
                float(value) for value in unit["effective_bandwidth_hz"]
            ],
            "physical_amplitude_preserved": bool(
                transform["normalization"]["preserves_physical_amplitude"]
            )
            and transform["clipping"]["applied"] is False,
            "amplitude_evidence_eligible": bool(amplitude["eligible"]),
            "amplitude_ineligibility_reason_codes": _sorted_reasons(
                amplitude["reason_codes"]
            ),
            "reference_type": str(transform["reference"]["reference_type"]),
            "reference_matrix_sha256": str(transform["reference"]["matrix_sha256"]),
            "reference_input_coefficients": coefficients,
            "canonical_source_channels": source_channels,
            "canonical_signal_id": str(canonical["canonical_signal_id"]),
            "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
            "source_signal_sha256": str(canonical["source_signal_sha256"]),
        }
        body["calibration_binding_sha256"] = _self_hash(
            body, "calibration_binding_sha256"
        )
        ledger.append(body)
    return ledger


def _validate_calibration_ledger(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("physical-amplitude calibration ledger must be non-empty")
    required = {
        "view_id",
        "view_receipt_id",
        "view_receipt_sha256",
        "transform_spec_sha256",
        "processed_view_sha256",
        "quality_mask_sha256",
        "unit_id",
        "unit_type",
        "whole_output_unit_identity_preserved",
        "bipolar_endpoint_fact_projection_authorized",
        "source_channels_are_calibration_lineage_not_evidence_units",
        "output_physical_unit",
        "emitted_amplitude_unit",
        "sampling_rate_numerator",
        "sampling_rate_denominator",
        "effective_bandwidth_hz",
        "physical_amplitude_preserved",
        "amplitude_evidence_eligible",
        "amplitude_ineligibility_reason_codes",
        "reference_type",
        "reference_matrix_sha256",
        "reference_input_coefficients",
        "canonical_source_channels",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "calibration_binding_sha256",
    }
    result: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"calibration_ledger[{index}] keys drifted")
        row = deepcopy(raw)
        for field in ("view_id", "view_receipt_id", "unit_id", "canonical_signal_id"):
            _identifier(row[field], f"calibration_ledger[{index}].{field}")
        for field in (
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_matrix_sha256",
            "canonical_receipt_sha256",
            "source_signal_sha256",
            "calibration_binding_sha256",
        ):
            _sha(row[field], f"calibration_ledger[{index}].{field}")
        if row["unit_type"] not in {"electrode", "lead", "virtual"}:
            raise ValueError("calibration ledger unit_type is unsupported")
        if (
            row["whole_output_unit_identity_preserved"] is not True
            or row["bipolar_endpoint_fact_projection_authorized"] is not False
            or row["source_channels_are_calibration_lineage_not_evidence_units"]
            is not True
        ):
            raise ValueError("physical-amplitude unit identity policy drifted")
        if (
            row["output_physical_unit"] != "V"
            or row["emitted_amplitude_unit"] != "uV"
            or row["physical_amplitude_preserved"] is not True
        ):
            raise ValueError("physical-amplitude calibration is unavailable")
        numerator = row["sampling_rate_numerator"]
        denominator = row["sampling_rate_denominator"]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator < 1
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator < 1
            or math.gcd(numerator, denominator) != 1
        ):
            raise ValueError("calibration ledger sampling rate is invalid")
        _interval(row["effective_bandwidth_hz"], "effective_bandwidth_hz")
        if type(row["amplitude_evidence_eligible"]) is not bool:
            raise TypeError("amplitude_evidence_eligible must be boolean")
        reasons = _sorted_reasons(row["amplitude_ineligibility_reason_codes"])
        if row["amplitude_evidence_eligible"] == bool(reasons):
            raise ValueError("amplitude eligibility and reason codes disagree")
        if not isinstance(row["reference_type"], str) or not row["reference_type"]:
            raise ValueError("reference_type must be non-empty")
        coefficients = row["reference_input_coefficients"]
        if not isinstance(coefficients, list) or not coefficients:
            raise ValueError("reference coefficient ledger must be non-empty")
        coefficient_ids: list[str] = []
        for coefficient in coefficients:
            if type(coefficient) is not dict or set(coefficient) != {
                "input_unit_id",
                "coefficient",
            }:
                raise ValueError("reference coefficient keys drifted")
            coefficient_ids.append(
                _identifier(coefficient["input_unit_id"], "reference input unit")
            )
            if (
                abs(_finite(coefficient["coefficient"], "reference coefficient"))
                <= _TOL
            ):
                raise ValueError("zero reference coefficients must be omitted")
        if len(coefficient_ids) != len(set(coefficient_ids)):
            raise ValueError("reference coefficient inputs contain duplicates")
        channels = row["canonical_source_channels"]
        if not isinstance(channels, list) or not channels:
            raise ValueError("calibration source-channel ledger must be non-empty")
        channel_ids: list[str] = []
        for channel in channels:
            if type(channel) is not dict or set(channel) != {
                "channel_id",
                "source_physical_unit",
                "scale_to_volts",
                "observed",
                "imputed",
                "acquisition_highpass_hz",
                "acquisition_lowpass_hz",
                "reference_label",
            }:
                raise ValueError("calibration source-channel keys drifted")
            channel_ids.append(_identifier(channel["channel_id"], "channel_id"))
            unit = channel["source_physical_unit"]
            if unit not in _UNIT_TO_VOLTS or not math.isclose(
                _finite(channel["scale_to_volts"], "scale_to_volts"),
                _UNIT_TO_VOLTS[str(unit)],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("source physical dimension/scale is uncalibrated")
            if (
                type(channel["observed"]) is not bool
                or type(channel["imputed"]) is not bool
            ):
                raise TypeError(
                    "calibration source observed/imputed flags must be boolean"
                )
            if channel["observed"] and channel["imputed"]:
                raise ValueError("a calibration source cannot be observed and imputed")
            for bandwidth_field in (
                "acquisition_highpass_hz",
                "acquisition_lowpass_hz",
            ):
                if channel[bandwidth_field] is not None:
                    _finite(channel[bandwidth_field], bandwidth_field, minimum=0.0)
            if (
                not isinstance(channel["reference_label"], str)
                or not channel["reference_label"]
            ):
                raise ValueError("source reference label is unavailable")
        if channel_ids != sorted(channel_ids) or len(channel_ids) != len(
            set(channel_ids)
        ):
            raise ValueError("calibration source channels are not canonical")
        if row["amplitude_evidence_eligible"] and any(
            not channel["observed"] or channel["imputed"] for channel in channels
        ):
            raise ValueError(
                "amplitude-eligible calibration depends on unobserved/imputed source"
            )
        if row["calibration_binding_sha256"] != _self_hash(
            row, "calibration_binding_sha256"
        ):
            raise ValueError("calibration binding hash mismatch")
        keys.append((str(row["view_id"]), str(row["unit_id"])))
        result.append(row)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("calibration ledger order/identity drifted")
    return result


def _source_view_binding(view: _PreparedAmplitudeView) -> dict[str, Any]:
    receipt = view.receipt
    transform = receipt["transform_spec"]
    clock = transform["output_clock"]
    body: dict[str, Any] = {
        "view_id": str(receipt["view_id"]),
        "task_role": str(receipt["task_role"]),
        "view_receipt_id": str(receipt["view_receipt_id"]),
        "view_receipt_sha256": str(receipt["receipt_sha256"]),
        "transform_spec_sha256": str(transform["transform_spec_sha256"]),
        "processed_view_sha256": str(receipt["processed_view_sha256"]),
        "quality_mask_sha256": str(receipt["masks"]["mask_sha256"]),
        "reference_type": str(transform["reference"]["reference_type"]),
        "reference_matrix_sha256": str(
            transform["reference"]["matrix_sha256"]
        ),
        "sampling_rate_numerator": int(clock["sampling_rate_numerator"]),
        "sampling_rate_denominator": int(clock["sampling_rate_denominator"]),
        "output_unit_ids": sorted(view.unit_index),
        "native_parent_binding": (
            None
            if view.native_parent_binding is None
            else deepcopy(dict(view.native_parent_binding))
        ),
        "future_sample_access": False,
        "dependency_policy": "instantaneous",
        "physical_amplitude_preserved": True,
    }
    body["view_binding_sha256"] = _self_hash(body, "view_binding_sha256")
    return body


def _row_status(mask: Sequence[bool]) -> str:
    count = sum(bool(value) for value in mask)
    if count == 0:
        return "not_evaluable"
    if count == len(mask):
        return "measured"
    return "partially_measured"


def _materialize_physical_amplitude_source_receipt(
    *,
    event_id: str,
    canonical: Mapping[str, Any],
    prepared_views: Mapping[str, _PreparedAmplitudeView],
    analysis_interval: tuple[float, float],
    queries: Sequence[EventPhysicalAmplitudeQuery],
) -> dict[str, Any]:
    kernel_policy = BAIEGBaseNumericalPolicy()
    kernel_policy_body = kernel_policy.to_dict()
    kernel_policy_sha256 = _canonical_sha256(kernel_policy_body)
    query_roster = [query.to_dict() for query in queries]
    rows: list[dict[str, Any]] = []
    for query in queries:
        if query.view_id not in prepared_views:
            raise ValueError("physical-amplitude query references an unknown view")
        view = prepared_views[query.view_id]
        if query.unit_id not in view.unit_index:
            raise ValueError("physical-amplitude query references an unknown unit")
        local_unit_index = view.unit_index[query.unit_id]
        tensor_interval, actual_interval = _map_amplitude_query(
            view, query.recording_interval_seconds
        )
        unit = view.receipt["output_units"][local_unit_index]
        amplitude_reasons = _amplitude_family_reasons(
            view,
            local_unit_index=local_unit_index,
            tensor_interval=tensor_interval,
        )
        start, stop = tensor_interval
        measured = measure_ba_ieg_base_numerical_features(
            view.tensor[local_unit_index, start:stop],
            sampling_rate_hz=view.sampling_rate_hz,
            effective_bandwidth_hz=unit["effective_bandwidth_hz"],
            policy=kernel_policy,
            amplitude_reason_codes=amplitude_reasons,
            spectral_reason_codes=("not_requested_by_s04_amplitude_source",),
        )
        values = [
            float(measured.values[_BASE_TARGET_INDEX[name]])
            for name in _AMPLITUDE_SOURCE_TARGETS
        ]
        value_mask = [
            bool(measured.value_mask[_BASE_TARGET_INDEX[name]])
            for name in _AMPLITUDE_SOURCE_TARGETS
        ]
        reason_codes = [
            list(measured.reason_codes[_BASE_TARGET_INDEX[name]])
            for name in _AMPLITUDE_SOURCE_TARGETS
        ]
        source_binding: dict[str, Any] = {
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
            "raw_sample_intervals": _raw_amplitude_support(
                canonical,
                view,
                local_unit_index=local_unit_index,
                actual_interval=actual_interval,
            ),
            "query_authority": query.query_authority,
            "query_binding_sha256": _canonical_sha256(query.to_dict()),
            "kernel_policy_sha256": kernel_policy_sha256,
            "future_sample_access": False,
            "dependency_policy": "instantaneous",
        }
        source_binding_sha256 = _canonical_sha256(source_binding)
        row: dict[str, Any] = {
            "row_id": "S04SRCROW-"
            + _canonical_sha256(
                {
                    "event_id": event_id,
                    "source_binding_sha256": source_binding_sha256,
                    "targets": list(_AMPLITUDE_SOURCE_TARGETS),
                }
            )[:24],
            "assertion_level": "measured",
            "clinical_term_authorized": False,
            "source_binding": source_binding,
            "source_binding_sha256": source_binding_sha256,
            "opportunity": {
                "status": _row_status(value_mask),
                "target_value_mask": value_mask,
                "target_reason_codes": reason_codes,
                "aggregate_opportunity_reason_codes": _sorted_reasons(
                    [reason for target in reason_codes for reason in target]
                ),
            },
            "values": values,
        }
        row["row_binding_sha256"] = _self_hash(row, "row_binding_sha256")
        rows.append(row)

    view_bindings = [
        _source_view_binding(prepared_views[view_id])
        for view_id in sorted(prepared_views)
    ]
    source_binding_sha256 = _canonical_sha256(
        {
            "schema_version": EVENT_PHYSICAL_AMPLITUDE_SOURCE_SCHEMA_VERSION,
            "method_id": EVENT_PHYSICAL_AMPLITUDE_SOURCE_METHOD_ID,
            "event_id": event_id,
            "recording_id": canonical["recording_id"],
            "canonical_signal_id": canonical["canonical_signal_id"],
            "canonical_receipt_sha256": canonical["receipt_sha256"],
            "source_signal_sha256": canonical["source_signal_sha256"],
            "analysis_interval_seconds": list(analysis_interval),
            "kernel_policy_sha256": kernel_policy_sha256,
            "query_roster_sha256": _canonical_sha256(query_roster),
            "view_binding_sha256s": [
                row["view_binding_sha256"] for row in view_bindings
            ],
            "row_binding_sha256s": [row["row_binding_sha256"] for row in rows],
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": EVENT_PHYSICAL_AMPLITUDE_SOURCE_SCHEMA_VERSION,
        "method_id": EVENT_PHYSICAL_AMPLITUDE_SOURCE_METHOD_ID,
        "event_id": event_id,
        "recording_id": str(canonical["recording_id"]),
        "canonical_signal_id": str(canonical["canonical_signal_id"]),
        "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
        "source_signal_sha256": str(canonical["source_signal_sha256"]),
        "analysis_interval_seconds": list(analysis_interval),
        "coordinate_system": "recording_relative_seconds",
        "kernel_policy": kernel_policy_body,
        "kernel_policy_sha256": kernel_policy_sha256,
        "target_registry": [
            {
                "target_name": name,
                "unit_id": "uV",
                "semantic_level": "physical_measurement_only",
            }
            for name in _AMPLITUDE_SOURCE_TARGETS
        ],
        "query_roster": query_roster,
        "query_roster_sha256": _canonical_sha256(query_roster),
        "view_bindings": view_bindings,
        "rows": rows,
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_SOURCE_AUTHORIZATION),
        "source_binding_sha256": source_binding_sha256,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return _validate_physical_amplitude_source_receipt(receipt)


def _validate_physical_amplitude_source_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("physical-amplitude source receipt must be an object")
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
        "kernel_policy",
        "kernel_policy_sha256",
        "target_registry",
        "query_roster",
        "query_roster_sha256",
        "view_bindings",
        "rows",
        "firewall",
        "authorization",
        "source_binding_sha256",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("physical-amplitude source receipt keys drifted")
    if (
        receipt["schema_version"] != EVENT_PHYSICAL_AMPLITUDE_SOURCE_SCHEMA_VERSION
        or receipt["method_id"] != EVENT_PHYSICAL_AMPLITUDE_SOURCE_METHOD_ID
    ):
        raise ValueError("physical-amplitude source receipt identity drifted")
    for field in ("event_id", "recording_id", "canonical_signal_id"):
        _identifier(receipt[field], field)
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "kernel_policy_sha256",
        "query_roster_sha256",
        "source_binding_sha256",
        "receipt_sha256",
    ):
        _sha(receipt[field], field)
    analysis = _interval(
        receipt["analysis_interval_seconds"], "analysis_interval_seconds"
    )
    if receipt["coordinate_system"] != "recording_relative_seconds":
        raise ValueError("physical-amplitude source coordinate system drifted")
    if (
        receipt["kernel_policy"] != BAIEGBaseNumericalPolicy().to_dict()
        or receipt["kernel_policy_sha256"]
        != _canonical_sha256(receipt["kernel_policy"])
    ):
        raise ValueError("physical-amplitude source numerical policy drifted")
    expected_registry = [
        {
            "target_name": name,
            "unit_id": "uV",
            "semantic_level": "physical_measurement_only",
        }
        for name in _AMPLITUDE_SOURCE_TARGETS
    ]
    if receipt["target_registry"] != expected_registry:
        raise ValueError("physical-amplitude source target registry drifted")
    if (
        not isinstance(receipt["query_roster"], list)
        or not receipt["query_roster"]
        or receipt["query_roster_sha256"]
        != _canonical_sha256(receipt["query_roster"])
    ):
        raise ValueError("physical-amplitude source query roster drifted")
    if receipt["firewall"] != _FIREWALL:
        raise ValueError("physical-amplitude source firewall drifted")
    if receipt["authorization"] != _SOURCE_AUTHORIZATION:
        raise ValueError("physical-amplitude source authorization drifted")

    view_bindings = receipt["view_bindings"]
    if not isinstance(view_bindings, list) or not view_bindings:
        raise ValueError("physical-amplitude source view bindings are empty")
    view_ids: list[str] = []
    view_binding_hashes: list[str] = []
    for index, row in enumerate(view_bindings):
        if type(row) is not dict:
            raise TypeError(f"view_bindings[{index}] must be an object")
        expected_keys = {
            "view_id",
            "task_role",
            "view_receipt_id",
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_type",
            "reference_matrix_sha256",
            "sampling_rate_numerator",
            "sampling_rate_denominator",
            "output_unit_ids",
            "native_parent_binding",
            "future_sample_access",
            "dependency_policy",
            "physical_amplitude_preserved",
            "view_binding_sha256",
        }
        if set(row) != expected_keys:
            raise ValueError(f"view_bindings[{index}] keys drifted")
        view_ids.append(_identifier(row["view_id"], "view binding view_id"))
        if row["task_role"] not in {
            "findings_native_morphology",
            "spatial_reference",
        }:
            raise ValueError("physical-amplitude source view role drifted")
        for field in (
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_matrix_sha256",
            "view_binding_sha256",
        ):
            _sha(row[field], f"view_bindings[{index}].{field}")
        if (
            row["future_sample_access"] is not False
            or row["dependency_policy"] != "instantaneous"
            or row["physical_amplitude_preserved"] is not True
        ):
            raise ValueError("physical-amplitude source view semantics drifted")
        parent = row["native_parent_binding"]
        if row["task_role"] == "findings_native_morphology":
            if parent is not None:
                raise ValueError("native amplitude view cannot claim a parent")
        else:
            if type(parent) is not dict or set(parent) != {
                "view_id",
                "view_receipt_sha256",
                "processed_view_sha256",
                "task_role",
            }:
                raise ValueError("spatial amplitude view native parent drifted")
            if parent["task_role"] != "findings_native_morphology":
                raise ValueError("spatial amplitude parent role drifted")
            _identifier(parent["view_id"], "native parent view_id")
            _sha(parent["view_receipt_sha256"], "native parent receipt hash")
            _sha(parent["processed_view_sha256"], "native parent tensor hash")
        if row["view_binding_sha256"] != _self_hash(
            row, "view_binding_sha256"
        ):
            raise ValueError("physical-amplitude source view binding hash mismatch")
        view_binding_hashes.append(str(row["view_binding_sha256"]))
    if view_ids != sorted(view_ids) or len(view_ids) != len(set(view_ids)):
        raise ValueError("physical-amplitude source view order drifted")

    rows = receipt["rows"]
    if not isinstance(rows, list) or len(rows) != len(receipt["query_roster"]):
        raise ValueError("physical-amplitude source row roster drifted")
    row_hashes: list[str] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "row_id",
            "assertion_level",
            "clinical_term_authorized",
            "source_binding",
            "source_binding_sha256",
            "opportunity",
            "values",
            "row_binding_sha256",
        }:
            raise ValueError(f"source rows[{index}] keys drifted")
        _identifier(row["row_id"], "physical-amplitude source row_id")
        _sha(row["source_binding_sha256"], "source row binding hash")
        _sha(row["row_binding_sha256"], "source row hash")
        if (
            row["assertion_level"] != "measured"
            or row["clinical_term_authorized"] is not False
        ):
            raise ValueError("physical-amplitude source row semantics drifted")
        binding = row["source_binding"]
        if type(binding) is not dict:
            raise TypeError("physical-amplitude source row binding must be an object")
        if binding.get("canonical_signal_id") != receipt["canonical_signal_id"]:
            raise ValueError("source row canonical identity drifted")
        if binding.get("canonical_receipt_sha256") != receipt[
            "canonical_receipt_sha256"
        ] or binding.get("source_signal_sha256") != receipt["source_signal_sha256"]:
            raise ValueError("source row canonical hash drifted")
        for field in (
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
            "reference_matrix_sha256",
            "query_binding_sha256",
            "kernel_policy_sha256",
        ):
            _sha(binding.get(field), f"source rows[{index}].{field}")
        if (
            binding.get("physical_unit") != "V"
            or binding.get("future_sample_access") is not False
            or binding.get("dependency_policy") != "instantaneous"
            or binding.get("kernel_policy_sha256")
            != receipt["kernel_policy_sha256"]
        ):
            raise ValueError("physical-amplitude source binding semantics drifted")
        requested = _interval(
            binding.get("requested_recording_interval_seconds"),
            "source requested interval",
        )
        actual = _interval(
            binding.get("recording_interval_seconds"), "source actual interval"
        )
        if requested[0] < analysis[0] - _TOL or requested[1] > analysis[1] + _TOL:
            raise ValueError("physical-amplitude source row exceeds analysis")
        if actual[0] < requested[0] - _TOL or actual[1] > requested[1] + _TOL:
            raise ValueError("physical-amplitude source row expands query support")
        tensor_interval = binding.get("tensor_sample_interval")
        if (
            not isinstance(tensor_interval, list)
            or len(tensor_interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in tensor_interval)
            or tensor_interval[1] <= tensor_interval[0]
        ):
            raise ValueError("source tensor interval is invalid")
        raw_support = binding.get("raw_sample_intervals")
        if not isinstance(raw_support, list) or not raw_support:
            raise ValueError("source raw support is empty")
        raw_ids: list[str] = []
        for support in raw_support:
            if type(support) is not dict or set(support) != {
                "channel_id",
                "sample_rate_numerator",
                "sample_rate_denominator",
                "raw_start_sample",
                "raw_stop_sample_exclusive",
                "channel_sample_count",
            }:
                raise ValueError("source raw-support keys drifted")
            raw_ids.append(_identifier(support["channel_id"], "raw channel_id"))
            for field in (
                "raw_start_sample",
                "raw_stop_sample_exclusive",
                "channel_sample_count",
            ):
                if isinstance(support[field], bool) or not isinstance(
                    support[field], int
                ) or support[field] < 0:
                    raise ValueError("source raw-support sample index is invalid")
            if support["raw_stop_sample_exclusive"] < support["raw_start_sample"]:
                raise ValueError("source raw-support interval is reversed")
            if support["raw_stop_sample_exclusive"] > support["channel_sample_count"]:
                raise ValueError("source raw-support interval exceeds its carrier")
        if raw_ids != sorted(raw_ids) or len(raw_ids) != len(set(raw_ids)):
            raise ValueError("source raw-support order drifted")
        if row["source_binding_sha256"] != _canonical_sha256(binding):
            raise ValueError("physical-amplitude source binding hash mismatch")
        opportunity = row["opportunity"]
        if type(opportunity) is not dict or set(opportunity) != {
            "status",
            "target_value_mask",
            "target_reason_codes",
            "aggregate_opportunity_reason_codes",
        }:
            raise ValueError("physical-amplitude source opportunity drifted")
        masks = opportunity["target_value_mask"]
        reasons = opportunity["target_reason_codes"]
        values = row["values"]
        if not (
            isinstance(masks, list)
            and isinstance(reasons, list)
            and isinstance(values, list)
            and len(masks) == len(reasons) == len(values) == 2
        ):
            raise ValueError("physical-amplitude source target width drifted")
        for target_index in range(2):
            if type(masks[target_index]) is not bool:
                raise TypeError("source target mask must be boolean")
            normalized_reasons = _sorted_reasons(reasons[target_index])
            if masks[target_index] == bool(normalized_reasons):
                raise ValueError("source target mask/reasons disagree")
            measured_value = _finite(values[target_index], "source target value")
            if not masks[target_index] and measured_value != 0.0:
                raise ValueError("masked physical-amplitude source value must be zero")
        aggregate = _sorted_reasons(
            [reason for target in reasons for reason in target]
        )
        if opportunity["aggregate_opportunity_reason_codes"] != aggregate:
            raise ValueError("source aggregate opportunity reasons drifted")
        if opportunity["status"] != _row_status(masks):
            raise ValueError("source opportunity status drifted")
        if row["row_binding_sha256"] != _self_hash(row, "row_binding_sha256"):
            raise ValueError("physical-amplitude source row hash mismatch")
        row_hashes.append(str(row["row_binding_sha256"]))

    expected_source_binding = _canonical_sha256(
        {
            "schema_version": receipt["schema_version"],
            "method_id": receipt["method_id"],
            "event_id": receipt["event_id"],
            "recording_id": receipt["recording_id"],
            "canonical_signal_id": receipt["canonical_signal_id"],
            "canonical_receipt_sha256": receipt["canonical_receipt_sha256"],
            "source_signal_sha256": receipt["source_signal_sha256"],
            "analysis_interval_seconds": receipt["analysis_interval_seconds"],
            "kernel_policy_sha256": receipt["kernel_policy_sha256"],
            "query_roster_sha256": receipt["query_roster_sha256"],
            "view_binding_sha256s": view_binding_hashes,
            "row_binding_sha256s": row_hashes,
        }
    )
    if receipt["source_binding_sha256"] != expected_source_binding:
        raise ValueError("physical-amplitude source aggregate binding drifted")
    if receipt["receipt_sha256"] != _self_hash(receipt, "receipt_sha256"):
        raise ValueError("physical-amplitude source receipt hash mismatch")
    return receipt


def _measurement_rows(
    source_receipt: Mapping[str, Any],
    queries: Sequence[EventPhysicalAmplitudeQuery],
    calibration: Sequence[Mapping[str, Any]],
    *,
    policy: EventPhysicalAmplitudeFindingsPolicy,
) -> list[dict[str, Any]]:
    source_row_by_key = {
        (
            str(row["source_binding"]["view_id"]),
            str(row["source_binding"]["unit_id"]),
            float(row["source_binding"]["requested_recording_interval_seconds"][0]),
            float(row["source_binding"]["requested_recording_interval_seconds"][1]),
        ): row
        for row in source_receipt["rows"]
    }
    calibration_by_key = {
        (str(row["view_id"]), str(row["unit_id"])): row for row in calibration
    }
    rows: list[dict[str, Any]] = []
    for query in queries:
        key = (
            query.view_id,
            query.unit_id,
            query.recording_interval_seconds[0],
            query.recording_interval_seconds[1],
        )
        if key not in source_row_by_key:
            raise ValueError("physical-amplitude query has no source measurement row")
        source_row = source_row_by_key[key]
        calibration_row = calibration_by_key[(query.view_id, query.unit_id)]
        source = source_row["source_binding"]
        binding_checks = {
            "view_receipt_sha256": calibration_row["view_receipt_sha256"],
            "transform_spec_sha256": calibration_row["transform_spec_sha256"],
            "processed_view_sha256": calibration_row["processed_view_sha256"],
            "quality_mask_sha256": calibration_row["quality_mask_sha256"],
            "reference_matrix_sha256": calibration_row["reference_matrix_sha256"],
            "reference_type": calibration_row["reference_type"],
            "unit_type": calibration_row["unit_type"],
            "physical_unit": calibration_row["output_physical_unit"],
        }
        if any(source[field] != expected for field, expected in binding_checks.items()):
            raise ValueError("primitive and amplitude calibration bindings disagree")
        if [float(value) for value in source["effective_bandwidth_hz"]] != [
            float(value) for value in calibration_row["effective_bandwidth_hz"]
        ]:
            raise ValueError("primitive and calibration bandwidth disagree")
        raw_channels = sorted(
            str(row["channel_id"]) for row in source["raw_sample_intervals"]
        )
        calibration_channels = [
            str(row["channel_id"])
            for row in calibration_row["canonical_source_channels"]
        ]
        if raw_channels != calibration_channels:
            raise ValueError("primitive raw support and calibration lineage disagree")

        rms_index = _AMPLITUDE_SOURCE_TARGET_INDEX["rms_uv"]
        p2p_index = _AMPLITUDE_SOURCE_TARGET_INDEX["peak_to_peak_uv"]
        masks = source_row["opportunity"]["target_value_mask"]
        target_reasons = source_row["opportunity"]["target_reason_codes"]
        reasons = [*target_reasons[rms_index], *target_reasons[p2p_index]]
        sampling_rate = float(calibration_row["sampling_rate_numerator"]) / float(
            calibration_row["sampling_rate_denominator"]
        )
        if sampling_rate + _TOL < policy.minimum_sample_rate_hz:
            reasons.append("sample_rate_below_s04_minimum")
        bandwidth = calibration_row["effective_bandwidth_hz"]
        if (
            float(bandwidth[0]) > policy.required_bandwidth_low_hz + _TOL
            or float(bandwidth[1]) + _TOL < policy.required_bandwidth_high_hz
        ):
            reasons.append("effective_bandwidth_below_s04_requirement")
        if not calibration_row["amplitude_evidence_eligible"]:
            reasons.extend(calibration_row["amplitude_ineligibility_reason_codes"])
        available = bool(masks[rms_index]) and bool(masks[p2p_index]) and not reasons
        reasons = _sorted_reasons(reasons)
        values = source_row["values"]
        row: dict[str, Any] = {
            "measurement_id": "S04AMP-"
            + _canonical_sha256(
                {
                    "event_id": source_receipt["event_id"],
                    "query": query.to_dict(),
                    "source_row_binding_sha256": source_row["row_binding_sha256"],
                    "calibration_binding_sha256": calibration_row[
                        "calibration_binding_sha256"
                    ],
                }
            )[:24],
            "view_id": query.view_id,
            "unit_id": query.unit_id,
            "unit_type": str(source["unit_type"]),
            "whole_output_unit_identity_preserved": True,
            "bipolar_endpoint_fact_projection_authorized": False,
            "measurement_role": query.measurement_role,
            "comparison_set_id": query.comparison_set_id,
            "ordinal": query.ordinal,
            "requested_recording_interval_seconds": list(
                query.recording_interval_seconds
            ),
            "recording_interval_seconds": list(source["recording_interval_seconds"]),
            "recording_time_seconds": float(
                (
                    source["recording_interval_seconds"][0]
                    + source["recording_interval_seconds"][1]
                )
                / 2.0
            ),
            "status": "measured" if available else "not_evaluable",
            "assertion_level": "measured",
            "rms_uv": float(values[rms_index]) if available else None,
            "peak_to_peak_uv": float(values[p2p_index]) if available else None,
            "amplitude_unit": "uV",
            "opportunity": {
                "status": "sufficient" if available else "not_evaluable",
                "reason_codes": reasons,
                "not_evaluable_is_negative": False,
            },
            "effective_bandwidth_hz": [float(value) for value in bandwidth],
            "sampling_rate_hz": sampling_rate,
            "reference_type": str(source["reference_type"]),
            "reference_matrix_sha256": str(source["reference_matrix_sha256"]),
            "selection_receipt_sha256": query.selection_receipt_sha256,
            "source_amplitude_row_id": str(source_row["row_id"]),
            "source_amplitude_row_binding_sha256": str(
                source_row["row_binding_sha256"]
            ),
            "calibration_binding_sha256": str(
                calibration_row["calibration_binding_sha256"]
            ),
            "clinical_attenuation_or_electrodecrement_authorized": False,
            "amplitude_change_as_evolution_authorized": False,
        }
        row["measurement_sha256"] = _self_hash(row, "measurement_sha256")
        rows.append(row)
    return rows


def _group_measurements(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault(
            (str(row["view_id"]), str(row["unit_id"]), str(row["comparison_set_id"])),
            [],
        ).append(row)
    return result


def _attenuation_ratios(
    measurements: Sequence[Mapping[str, Any]],
    *,
    policy: EventPhysicalAmplitudeFindingsPolicy,
) -> list[dict[str, Any]]:
    ratios: list[dict[str, Any]] = []
    for (view_id, unit_id, set_id), rows in sorted(
        _group_measurements(measurements).items()
    ):
        contexts = sorted(
            (
                row
                for row in rows
                if row["measurement_role"] == "signal_selected_comparison_context"
            ),
            key=lambda row: int(row["ordinal"]),
        )
        course = sorted(
            (row for row in rows if row["measurement_role"] == "event_course"),
            key=lambda row: int(row["ordinal"]),
        )
        measured_context = [row for row in contexts if row["status"] == "measured"]
        reference_rms = (
            float(np.median([float(row["rms_uv"]) for row in measured_context]))
            if measured_context
            else None
        )
        reference_p2p = (
            float(
                np.median([float(row["peak_to_peak_uv"]) for row in measured_context])
            )
            if measured_context
            else None
        )
        for point in course:
            reasons: list[str] = []
            if point["status"] != "measured":
                reasons.append("event_course_amplitude_not_evaluable")
            if not contexts:
                reasons.append("comparison_context_not_requested")
            elif not measured_context:
                reasons.append("comparison_context_amplitude_not_evaluable")
            elif len(measured_context) != len(contexts):
                reasons.append("one_or_more_comparison_context_points_not_evaluable")
            if (
                reference_rms is not None
                and reference_rms <= policy.ratio_denominator_floor_uv
            ):
                reasons.append("comparison_context_rms_at_or_below_ratio_floor")
            if (
                reference_p2p is not None
                and reference_p2p <= policy.ratio_denominator_floor_uv
            ):
                reasons.append(
                    "comparison_context_peak_to_peak_at_or_below_ratio_floor"
                )
            rms_ratio_available = (
                point["status"] == "measured"
                and reference_rms is not None
                and reference_rms > policy.ratio_denominator_floor_uv
            )
            p2p_ratio_available = (
                point["status"] == "measured"
                and reference_p2p is not None
                and reference_p2p > policy.ratio_denominator_floor_uv
            )
            if not rms_ratio_available:
                reasons.append("rms_attenuation_ratio_not_evaluable")
            if not p2p_ratio_available:
                reasons.append("peak_to_peak_context_ratio_not_evaluable")
            opportunity = (
                "sufficient"
                if rms_ratio_available
                and p2p_ratio_available
                and len(measured_context) == len(contexts)
                else "limited"
                if rms_ratio_available or p2p_ratio_available
                else "not_evaluable"
            )
            row: dict[str, Any] = {
                "ratio_id": "S04RATIO-"
                + _canonical_sha256(
                    {
                        "event_measurement_id": point["measurement_id"],
                        "context_measurement_ids": [
                            item["measurement_id"] for item in measured_context
                        ],
                        "policy_sha256": policy.sha256,
                    }
                )[:24],
                "view_id": view_id,
                "unit_id": unit_id,
                "unit_type": str(point["unit_type"]),
                "whole_output_unit_identity_preserved": True,
                "comparison_set_id": set_id,
                "event_course_ordinal": int(point["ordinal"]),
                "event_recording_interval_seconds": list(
                    point["recording_interval_seconds"]
                ),
                "status": (
                    "measured"
                    if rms_ratio_available and p2p_ratio_available
                    else "partially_measured"
                    if rms_ratio_available or p2p_ratio_available
                    else "not_evaluable"
                ),
                "opportunity": {
                    "status": opportunity,
                    "reason_codes": _sorted_reasons(reasons),
                    "not_evaluable_is_negative": False,
                },
                "comparison_context_aggregation": policy.context_aggregation,
                "comparison_context_measurement_ids": [
                    item["measurement_id"] for item in measured_context
                ],
                "comparison_context_requested_count": len(contexts),
                "comparison_context_measured_count": len(measured_context),
                "comparison_context_rms_uv": reference_rms,
                "comparison_context_peak_to_peak_uv": reference_p2p,
                "event_rms_uv": point["rms_uv"],
                "event_peak_to_peak_uv": point["peak_to_peak_uv"],
                "attenuation_ratio": (
                    float(point["rms_uv"]) / reference_rms
                    if rms_ratio_available
                    else None
                ),
                "attenuation_ratio_definition": (
                    "event_course_rms_uv_over_median_signal_selected_"
                    "comparison_context_rms_uv"
                ),
                "peak_to_peak_ratio_to_comparison_context": (
                    float(point["peak_to_peak_uv"]) / reference_p2p
                    if p2p_ratio_available
                    else None
                ),
                "ratio_unit": "dimensionless",
                "ratio_below_one_only_means_lower_measured_amplitude": True,
                "clinical_attenuation_or_electrodecrement_authorized": False,
                "amplitude_change_as_evolution_authorized": False,
                "source_event_measurement_id": str(point["measurement_id"]),
                "source_event_measurement_sha256": str(point["measurement_sha256"]),
            }
            row["ratio_sha256"] = _self_hash(row, "ratio_sha256")
            ratios.append(row)
    return ratios


def _trajectories(
    measurements: Sequence[Mapping[str, Any]],
    *,
    policy: EventPhysicalAmplitudeFindingsPolicy,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for (view_id, unit_id, set_id), rows in sorted(
        _group_measurements(measurements).items()
    ):
        course = sorted(
            (row for row in rows if row["measurement_role"] == "event_course"),
            key=lambda row: int(row["ordinal"]),
        )
        if not course:
            continue
        points = [
            {
                "ordinal": int(row["ordinal"]),
                "measurement_id": str(row["measurement_id"]),
                "measurement_sha256": str(row["measurement_sha256"]),
                "recording_interval_seconds": list(row["recording_interval_seconds"]),
                "recording_time_seconds": float(row["recording_time_seconds"]),
                "status": str(row["status"]),
                "rms_uv": row["rms_uv"],
                "peak_to_peak_uv": row["peak_to_peak_uv"],
                "opportunity_reason_codes": list(row["opportunity"]["reason_codes"]),
            }
            for row in course
        ]
        transitions: list[dict[str, Any]] = []
        for previous, current in zip(points, points[1:]):
            if previous["status"] != "measured" or current["status"] != "measured":
                continue
            delta_time = float(current["recording_time_seconds"]) - float(
                previous["recording_time_seconds"]
            )
            if delta_time <= _TOL:
                raise ValueError("amplitude course time is not strictly increasing")
            previous_rms = float(previous["rms_uv"])
            previous_p2p = float(previous["peak_to_peak_uv"])
            ratio_reasons: list[str] = []
            if previous_rms <= policy.ratio_denominator_floor_uv:
                ratio_reasons.append("previous_rms_at_or_below_ratio_floor")
            if previous_p2p <= policy.ratio_denominator_floor_uv:
                ratio_reasons.append("previous_peak_to_peak_at_or_below_ratio_floor")
            transition = {
                "ordinal": len(transitions) + 1,
                "from_point_ordinal": int(previous["ordinal"]),
                "to_point_ordinal": int(current["ordinal"]),
                "recording_time_interval_seconds": [
                    float(previous["recording_time_seconds"]),
                    float(current["recording_time_seconds"]),
                ],
                "delta_rms_uv": float(current["rms_uv"]) - previous_rms,
                "rms_slope_uv_per_second": (float(current["rms_uv"]) - previous_rms)
                / delta_time,
                "rms_ratio_to_previous": (
                    float(current["rms_uv"]) / previous_rms
                    if previous_rms > policy.ratio_denominator_floor_uv
                    else None
                ),
                "delta_peak_to_peak_uv": float(current["peak_to_peak_uv"])
                - previous_p2p,
                "peak_to_peak_slope_uv_per_second": (
                    float(current["peak_to_peak_uv"]) - previous_p2p
                )
                / delta_time,
                "peak_to_peak_ratio_to_previous": (
                    float(current["peak_to_peak_uv"]) / previous_p2p
                    if previous_p2p > policy.ratio_denominator_floor_uv
                    else None
                ),
                "ratio_reason_codes": _sorted_reasons(ratio_reasons),
            }
            transition["transition_sha256"] = _self_hash(
                transition, "transition_sha256"
            )
            transitions.append(transition)
        measured_count = sum(point["status"] == "measured" for point in points)
        complete_transition_count = sum(
            current["ordinal"] == previous["ordinal"] + 1
            for previous, current in zip(points, points[1:])
            if previous["status"] == "measured" and current["status"] == "measured"
        )
        evaluable = measured_count >= policy.minimum_course_points and bool(transitions)
        complete = (
            evaluable
            and measured_count == len(points)
            and complete_transition_count == max(0, len(points) - 1)
        )
        reasons: list[str] = []
        if measured_count < policy.minimum_course_points:
            reasons.append("fewer_than_minimum_measured_course_points")
        if any(point["status"] != "measured" for point in points):
            reasons.append("one_or_more_course_points_not_evaluable")
        if measured_count >= policy.minimum_course_points and not transitions:
            reasons.append("no_adjacent_measured_course_transition")
        body: dict[str, Any] = {
            "trajectory_id": "S04COURSE-"
            + _canonical_sha256(
                {
                    "view_id": view_id,
                    "unit_id": unit_id,
                    "comparison_set_id": set_id,
                    "point_measurement_sha256s": [
                        point["measurement_sha256"] for point in points
                    ],
                    "policy_sha256": policy.sha256,
                }
            )[:24],
            "view_id": view_id,
            "unit_id": unit_id,
            "unit_type": str(course[0]["unit_type"]),
            "whole_output_unit_identity_preserved": True,
            "bipolar_endpoint_fact_projection_authorized": False,
            "comparison_set_id": set_id,
            "coordinate_system": "recording_relative_seconds",
            "status": (
                "measured"
                if complete
                else "partially_measured"
                if evaluable
                else "not_evaluable"
            ),
            "opportunity": {
                "status": "sufficient"
                if complete
                else "limited"
                if evaluable
                else "not_evaluable",
                "reason_codes": _sorted_reasons(reasons),
                "not_evaluable_is_negative": False,
            },
            "trajectory_source": "adjacent_nonoverlapping_physical_amplitude_windows",
            "points": points,
            "transition_intervals": transitions,
            "amplitude_change_alone_is_not_ictal_evolution": True,
            "clinical_attenuation_or_electrodecrement_authorized": False,
            "amplitude_change_as_evolution_authorized": False,
        }
        body["trajectory_sha256"] = _self_hash(body, "trajectory_sha256")
        result.append(body)
    return result


def _atom_results(
    measurements: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    measured = [row for row in measurements if row["status"] == "measured"]
    measurable_courses = [
        row
        for row in trajectories
        if row["status"] in {"measured", "partially_measured"}
    ]
    return [
        {
            "atom_id": "c1_physical_amplitude_profile",
            "term_query_id": "TQ-PHYSICAL-AMPLITUDE-PROFILE",
            "term_id": "deterministic_event_physical_amplitude_profile",
            "status": "present" if measured else "not_evaluable",
            "assertion_level": "measured",
            "measurement_ids": [row["measurement_id"] for row in measured],
            "report_promotion_authorized": False,
            "onset_support_eligible": False,
            "soz_support_eligible": False,
        },
        {
            "atom_id": "c3_amplitude_course",
            "term_query_id": "TQ-EVENT-AMPLITUDE-COURSE",
            "term_id": "event_amplitude_course_profile",
            "status": "present" if measurable_courses else "not_evaluable",
            "assertion_level": "measured",
            "trajectory_ids": [row["trajectory_id"] for row in measurable_courses],
            "report_promotion_authorized": False,
            "onset_support_eligible": False,
            "soz_support_eligible": False,
        },
    ]


def _body_from_sources(
    *,
    physical_source: Mapping[str, Any],
    analysis_interval: tuple[float, float],
    queries: Sequence[EventPhysicalAmplitudeQuery],
    calibration: Sequence[Mapping[str, Any]],
    policy: EventPhysicalAmplitudeFindingsPolicy,
) -> dict[str, Any]:
    measurements = _measurement_rows(
        physical_source, queries, calibration, policy=policy
    )
    ratios = _attenuation_ratios(measurements, policy=policy)
    trajectories = _trajectories(measurements, policy=policy)
    query_roster = [row.to_dict() for row in queries]
    calibration_rows = [deepcopy(dict(row)) for row in calibration]
    source_summary = {
        "canonical_signal_id": str(physical_source["canonical_signal_id"]),
        "canonical_receipt_sha256": str(
            physical_source["canonical_receipt_sha256"]
        ),
        "source_signal_sha256": str(physical_source["source_signal_sha256"]),
        "physical_amplitude_source_receipt_sha256": str(
            physical_source["receipt_sha256"]
        ),
        "view_receipt_sha256s": sorted(
            {str(row["view_receipt_sha256"]) for row in calibration_rows}
        ),
        "transform_spec_sha256s": sorted(
            {str(row["transform_spec_sha256"]) for row in calibration_rows}
        ),
        "processed_view_sha256s": sorted(
            {str(row["processed_view_sha256"]) for row in calibration_rows}
        ),
        "quality_mask_sha256s": sorted(
            {str(row["quality_mask_sha256"]) for row in calibration_rows}
        ),
        "selection_receipt_sha256s": sorted(
            {row.selection_receipt_sha256 for row in queries}
        ),
        "raw_eeg_used": True,
        "edf_annotations_used": False,
        "excel_used": False,
        "doctor_labels_used": False,
        "clinical_text_used": False,
        "research_ranking_used": False,
    }
    source_summary["source_receipt_sha256"] = _self_hash(
        source_summary, "source_receipt_sha256"
    )
    return {
        "schema_version": EVENT_PHYSICAL_AMPLITUDE_FINDINGS_SCHEMA_VERSION,
        "method_id": EVENT_PHYSICAL_AMPLITUDE_FINDINGS_METHOD_ID,
        "event_id": str(physical_source["event_id"]),
        "recording_id": str(physical_source["recording_id"]),
        "canonical_signal_id": str(physical_source["canonical_signal_id"]),
        "canonical_receipt_sha256": str(
            physical_source["canonical_receipt_sha256"]
        ),
        "source_signal_sha256": str(physical_source["source_signal_sha256"]),
        "analysis_interval_seconds": list(analysis_interval),
        "coordinate_system": "recording_relative_seconds",
        "event_card_slot_id": "S04_PHYSICAL_AMPLITUDE",
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "query_roster": query_roster,
        "query_roster_sha256": _canonical_sha256(query_roster),
        "calibration_ledger": calibration_rows,
        "calibration_ledger_sha256": _canonical_sha256(calibration_rows),
        "source_physical_amplitude_receipt": deepcopy(dict(physical_source)),
        "source_physical_amplitude_receipt_sha256": str(
            physical_source["receipt_sha256"]
        ),
        "measurements": measurements,
        "attenuation_ratios": ratios,
        "amplitude_trajectories": trajectories,
        "atom_results": _atom_results(measurements, trajectories),
        "source_receipt": source_summary,
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_event_physical_amplitude_findings_v1(
    *,
    event_id: str,
    canonical_receipt: object,
    views: Sequence[EventPhysicalAmplitudeViewInput],
    analysis_interval_seconds: Sequence[float],
    queries: Sequence[EventPhysicalAmplitudeQuery],
    policy: EventPhysicalAmplitudeFindingsPolicy = (
        DEFAULT_EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Measure and content-bind one event's S04 signal evidence."""

    _identifier(event_id, "event_id")
    if not isinstance(policy, EventPhysicalAmplitudeFindingsPolicy):
        raise TypeError("policy must be EventPhysicalAmplitudeFindingsPolicy")
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    analysis = _interval(analysis_interval_seconds, "analysis_interval_seconds")
    if analysis[1] > float(canonical["recording_duration_seconds"]) + _TOL:
        raise ValueError("physical-amplitude analysis exceeds the recording")
    normalized_queries = _normalize_queries(queries, analysis_interval=analysis)
    view_catalog, prepared_views = _validated_view_catalog(
        canonical,
        views,
        trusted_parent_views=trusted_parent_views,
    )
    calibration = _calibration_ledger(canonical, view_catalog, normalized_queries)
    calibration = _validate_calibration_ledger(calibration)
    physical_source = _materialize_physical_amplitude_source_receipt(
        event_id=event_id,
        canonical=canonical,
        prepared_views=prepared_views,
        analysis_interval=analysis,
        queries=normalized_queries,
    )
    body = _body_from_sources(
        physical_source=physical_source,
        analysis_interval=analysis,
        queries=normalized_queries,
        calibration=calibration,
        policy=policy,
    )
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return validate_event_physical_amplitude_findings_v1(body)


def validate_event_physical_amplitude_findings_v1(value: object) -> dict[str, Any]:
    """Validate source replay projection, opportunity semantics and hashes."""

    if type(value) is not dict:
        raise TypeError("event physical-amplitude findings must be an object")
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
        "event_card_slot_id",
        "policy",
        "policy_sha256",
        "query_roster",
        "query_roster_sha256",
        "calibration_ledger",
        "calibration_ledger_sha256",
        "source_physical_amplitude_receipt",
        "source_physical_amplitude_receipt_sha256",
        "measurements",
        "attenuation_ratios",
        "amplitude_trajectories",
        "atom_results",
        "source_receipt",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("event physical-amplitude findings keys drifted")
    if (
        receipt["schema_version"] != EVENT_PHYSICAL_AMPLITUDE_FINDINGS_SCHEMA_VERSION
        or receipt["method_id"] != EVENT_PHYSICAL_AMPLITUDE_FINDINGS_METHOD_ID
        or receipt["event_card_slot_id"] != "S04_PHYSICAL_AMPLITUDE"
    ):
        raise ValueError("event physical-amplitude findings identity drifted")
    for field in ("event_id", "recording_id", "canonical_signal_id"):
        _identifier(receipt[field], field)
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "policy_sha256",
        "query_roster_sha256",
        "calibration_ledger_sha256",
        "source_physical_amplitude_receipt_sha256",
        "receipt_sha256",
    ):
        _sha(receipt[field], field)
    analysis = _interval(
        receipt["analysis_interval_seconds"], "analysis_interval_seconds"
    )
    if receipt["coordinate_system"] != "recording_relative_seconds":
        raise ValueError("physical-amplitude coordinate system drifted")
    if receipt["firewall"] != _FIREWALL or receipt["authorization"] != _AUTHORIZATION:
        raise ValueError("physical-amplitude firewall/authorization drifted")
    if (
        any(
            bool(value)
            for key, value in receipt["firewall"].items()
            if key != "eeg_samples_used"
        )
        or receipt["firewall"]["eeg_samples_used"] is not True
    ):
        raise ValueError("physical-amplitude producer violates the EEG-only firewall")

    policy_body = receipt["policy"]
    if type(policy_body) is not dict:
        raise TypeError("physical-amplitude policy must be an object")
    defaults = asdict(DEFAULT_EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY)
    policy = EventPhysicalAmplitudeFindingsPolicy(
        **{name: policy_body[name] for name in defaults}
    )
    if policy_body != policy.to_dict() or receipt["policy_sha256"] != policy.sha256:
        raise ValueError("physical-amplitude policy content/hash drifted")

    raw_queries = receipt["query_roster"]
    if not isinstance(raw_queries, list):
        raise TypeError("physical-amplitude query roster must be an array")
    query_keys = set(EventPhysicalAmplitudeQuery.__dataclass_fields__)
    queries: list[EventPhysicalAmplitudeQuery] = []
    for index, row in enumerate(raw_queries):
        if type(row) is not dict or set(row) != query_keys:
            raise ValueError(f"query_roster[{index}] keys drifted")
        queries.append(
            EventPhysicalAmplitudeQuery(
                view_id=row["view_id"],
                unit_id=row["unit_id"],
                recording_interval_seconds=tuple(row["recording_interval_seconds"]),
                measurement_role=row["measurement_role"],
                comparison_set_id=row["comparison_set_id"],
                ordinal=row["ordinal"],
                query_authority=row["query_authority"],
                selection_receipt_sha256=row["selection_receipt_sha256"],
            )
        )
    queries = _normalize_queries(queries, analysis_interval=analysis)
    canonical_query_roster = [row.to_dict() for row in queries]
    if raw_queries != canonical_query_roster or receipt[
        "query_roster_sha256"
    ] != _canonical_sha256(canonical_query_roster):
        raise ValueError("physical-amplitude query roster order/hash drifted")
    calibration = _validate_calibration_ledger(receipt["calibration_ledger"])
    if receipt["calibration_ledger_sha256"] != _canonical_sha256(calibration):
        raise ValueError("physical-amplitude calibration ledger hash drifted")

    physical_source = _validate_physical_amplitude_source_receipt(
        receipt["source_physical_amplitude_receipt"]
    )
    if (
        receipt["source_physical_amplitude_receipt_sha256"]
        != physical_source["receipt_sha256"]
    ):
        raise ValueError("physical-amplitude source hash drifted")
    identity = {
        "event_id": physical_source["event_id"],
        "recording_id": physical_source["recording_id"],
        "canonical_signal_id": physical_source["canonical_signal_id"],
        "canonical_receipt_sha256": physical_source[
            "canonical_receipt_sha256"
        ],
        "source_signal_sha256": physical_source["source_signal_sha256"],
        "analysis_interval_seconds": physical_source[
            "analysis_interval_seconds"
        ],
    }
    if any(receipt[field] != expected for field, expected in identity.items()):
        raise ValueError("physical-amplitude source identity drifted")
    expected = _body_from_sources(
        physical_source=physical_source,
        analysis_interval=analysis,
        queries=queries,
        calibration=calibration,
        policy=policy,
    )
    actual = deepcopy(receipt)
    actual.pop("receipt_sha256")
    if actual != expected:
        raise ValueError("physical-amplitude findings are not replayable from sources")
    if receipt["receipt_sha256"] != _self_hash(receipt, "receipt_sha256"):
        raise ValueError("physical-amplitude findings receipt hash mismatch")
    return receipt


def replay_event_physical_amplitude_findings_v1(
    expected_receipt: object,
    *,
    event_id: str,
    canonical_receipt: object,
    views: Sequence[EventPhysicalAmplitudeViewInput],
    analysis_interval_seconds: Sequence[float],
    queries: Sequence[EventPhysicalAmplitudeQuery],
    policy: EventPhysicalAmplitudeFindingsPolicy = (
        DEFAULT_EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY
    ),
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Recompute the complete S04 receipt from host-supplied signal artifacts."""

    expected = validate_event_physical_amplitude_findings_v1(expected_receipt)
    replayed = materialize_event_physical_amplitude_findings_v1(
        event_id=event_id,
        canonical_receipt=canonical_receipt,
        views=views,
        analysis_interval_seconds=analysis_interval_seconds,
        queries=queries,
        policy=policy,
        trusted_parent_views=trusted_parent_views,
    )
    if replayed != expected:
        raise ValueError("physical-amplitude findings receipt does not replay")
    return replayed


__all__ = [
    "DEFAULT_EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY",
    "EVENT_PHYSICAL_AMPLITUDE_FINDINGS_METHOD_ID",
    "EVENT_PHYSICAL_AMPLITUDE_FINDINGS_POLICY_ID",
    "EVENT_PHYSICAL_AMPLITUDE_FINDINGS_SCHEMA_VERSION",
    "EventPhysicalAmplitudeFindingsPolicy",
    "EventPhysicalAmplitudeQuery",
    "EventPhysicalAmplitudeViewInput",
    "build_event_physical_amplitude_queries_v1",
    "materialize_event_physical_amplitude_findings_v1",
    "replay_event_physical_amplitude_findings_v1",
    "validate_event_physical_amplitude_findings_v1",
]
