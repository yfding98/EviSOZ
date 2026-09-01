"""Exact, EEG-only S01 native signal-quality and opportunity Findings.

This module turns one materialized canonical EDF bundle into a closed S01
technical ledger for an event support interval.  It deliberately does not
classify clinical artefact types.  The only signal-quality names emitted by
the producer are physical/engineering observations that can be recomputed
from the native samples and the EDF signal calibration fields:

* missing typed carriers and their sampling clocks;
* non-finite samples (the canonical loader fails closed before admission);
* repeated-value/constant runs;
* ADC-rail plateaus when EDF calibration rails are available;
* an explicitly unqualified repeated-value fallback when rails are absent;
* canonical/view quality masks, filter edges and record-edge censoring; and
* exact usable-sample and usable-interval denominators for every evidence
  family in every admitted task view.

Electrodes and longitudinal bipolar derivations are both enumerated.  A
bipolar derivation is always a ``whole_bipolar_lead`` and is never projected
onto either endpoint.  ``not_evaluable`` is an opportunity state, never a
negative clinical assertion.

The public API accepts no annotation, spreadsheet, doctor label/report,
clinical text, video/behaviour, sleep/activation, ECG/EMG/EOG or LLM input.
An exact replay must independently reopen the EDF, rebuild the canonical
bundle and call :func:`replay_event_native_signal_quality_findings_v1`.
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

from src.soz.geometry import STANDARD_19, TCP_20_EDGES

from .canonical_edf_materialization import (
    CanonicalEDFViewBundle,
    validate_canonical_edf_materialization,
)
from .canonical_signal_views import (
    EVIDENCE_FAMILIES,
    recording_seconds_to_view_tensor_index,
    view_tensor_index_to_recording_seconds,
)


EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_event_native_signal_quality_findings_v1"
EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_METHOD_ID: Final[
    str
] = "EEG-ONLY-EXACT-NATIVE-S01-QC-OPPORTUNITY-V1"
EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_POLICY_ID: Final[
    str
] = "S01-NATIVE-QC-OPPORTUNITY-DENOMINATOR-POLICY-V1"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-8
_REFERENCE_KINDS: Final[tuple[str, ...]] = ("referential", "tcp_bipolar")
_VIEW_ROLES: Final[tuple[str, ...]] = (
    "findings_native_morphology",
    "onset_causal",
    "context_offline",
)
_STATUS_VALUES = frozenset({"measured", "not_evaluable"})
_ORDERED_TCP_CARRIERS: Final[dict[str, tuple[str, str]]] = {
    f"{left}-{right}": (left, right) for left, right in TCP_20_EDGES
}

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_patient_or_recording_header_used": False,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_information_used": False,
    "ecg_emg_eog_used": False,
    "llm_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str | list[object]]] = {
    "event_card_slot_id": "S01_SIGNAL_QUALITY_AND_OPPORTUNITY",
    "technical_measurement_only": True,
    "clinical_artifact_classification_authorized": False,
    "muscle_eye_movement_or_other_clinical_artifact_terms_authorized": False,
    "digital_saturation_or_clipping_clinical_term_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "not_evaluable_is_negative": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "bipolar_endpoint_fact_projection_authorized": False,
    "whole_bipolar_lead_identity_required": True,
    "report_eligible_automated_term_allowlist": [],
    "report_text_authorized": False,
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


def _self_hash(value: Mapping[str, object]) -> str:
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a contract-compatible identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _requested_interval(
    value: Sequence[float],
    *,
    duration_seconds: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("requested analysis interval must contain two numbers")
    start = _finite(value[0], "requested_analysis_interval_seconds[0]")
    stop = _finite(value[1], "requested_analysis_interval_seconds[1]")
    if stop <= start + _TOL:
        raise ValueError("requested analysis interval must be non-empty")
    bounded = (max(0.0, start), min(float(duration_seconds), stop))
    if bounded[1] <= bounded[0] + _TOL:
        raise ValueError("requested analysis interval does not overlap the recording")
    return (start, stop), bounded


def _true_runs(mask: np.ndarray, *, minimum_length: int = 1) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("quality run mask must be one-dimensional")
    if minimum_length < 1:
        raise ValueError("minimum quality run length must be positive")
    padded = np.concatenate([np.asarray([False]), values, np.asarray([False])]).astype(
        np.int8
    )
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop) - int(start) >= minimum_length
    ]


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(
        (int(start), int(stop)) for start, stop in intervals if int(stop) > int(start)
    )
    result: list[tuple[int, int]] = []
    for start, stop in ordered:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
        else:
            result.append((start, stop))
    return result


def _intersect_intervals(
    intervals: Sequence[tuple[int, int]],
    carrier: tuple[int, int],
) -> list[tuple[int, int]]:
    return _merge_intervals(
        [
            (max(start, carrier[0]), min(stop, carrier[1]))
            for start, stop in intervals
            if start < carrier[1] and stop > carrier[0]
        ]
    )


def _interval_sample_count(intervals: Sequence[tuple[int, int]]) -> int:
    return int(sum(stop - start for start, stop in _merge_intervals(intervals)))


def _mark_intervals(
    mask: np.ndarray,
    intervals: Sequence[tuple[int, int]],
) -> None:
    for start, stop in intervals:
        left = max(0, int(start))
        right = min(int(mask.size), int(stop))
        if right > left:
            mask[left:right] = True


def _segment_sha256(
    values: torch.Tensor,
    *,
    unit_id: str,
    tensor_interval: tuple[int, int],
) -> str:
    tensor = values.detach().cpu().to(torch.float32).contiguous()
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError("native S01 tensor segment must be finite and one-dimensional")
    header = {
        "domain": "event-native-s01-physical-float32-le-v1",
        "unit_id": unit_id,
        "tensor_sample_interval": list(tensor_interval),
        "dtype": "float32-le",
        "sample_count": int(tensor.numel()),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _view_rate(view_receipt: Mapping[str, object]) -> tuple[int, int, float]:
    clock = view_receipt["transform_spec"]["output_clock"]  # type: ignore[index]
    numerator = int(clock["sampling_rate_numerator"])
    denominator = int(clock["sampling_rate_denominator"])
    return numerator, denominator, numerator / denominator


def _view_interval(
    view_receipt: Mapping[str, object],
    bounded_interval_seconds: tuple[float, float],
) -> tuple[int, int]:
    start = recording_seconds_to_view_tensor_index(
        dict(view_receipt),
        recording_seconds=bounded_interval_seconds[0],
        rounding="ceil",
    )
    stop = recording_seconds_to_view_tensor_index(
        dict(view_receipt),
        recording_seconds=bounded_interval_seconds[1],
        rounding="floor",
    )
    return int(start), int(stop)


def _interval_rows(
    intervals: Sequence[tuple[int, int]],
    *,
    view_receipt: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start, stop in _merge_intervals(intervals):
        rows.append(
            {
                "tensor_sample_interval": [start, stop],
                "recording_interval_seconds": [
                    view_tensor_index_to_recording_seconds(
                        dict(view_receipt), tensor_sample_index=start
                    ),
                    view_tensor_index_to_recording_seconds(
                        dict(view_receipt), tensor_sample_index=stop
                    ),
                ],
                "sample_count": stop - start,
            }
        )
    return rows


@dataclass(frozen=True)
class EventNativeSignalQualityPolicy:
    """Frozen semantic policy; thresholds come from canonical preprocessing."""

    reference_kinds: tuple[str, ...] = _REFERENCE_KINDS
    opportunity_view_roles: tuple[str, ...] = _VIEW_ROLES
    usable_denominator: str = (
        "observed_and_finite_minus_padding_edge_view_qc_and_direct_native_qc_union"
    )
    interval_quantization: str = "complete_sample_edges_ceil_start_floor_stop"
    flat_run_semantics: str = "technical_repeated_value_run_not_clinical_artifact"
    adc_rail_semantics: str = (
        "header_calibration_rail_plateau_not_clinical_saturation_diagnosis"
    )
    missing_rail_fallback_semantics: str = (
        "local_repeated_value_plateau_model_candidate_not_clipping"
    )
    gap_semantics: str = (
        "materialized_clock_holes_measured_acquisition_discontinuity_not_evaluable"
    )

    def __post_init__(self) -> None:
        if self.reference_kinds != _REFERENCE_KINDS:
            raise ValueError("S01 v1 freezes referential and whole TCP bipolar units")
        if self.opportunity_view_roles != _VIEW_ROLES:
            raise ValueError("S01 v1 freezes native, causal and offline opportunities")

    def to_dict(self) -> dict[str, object]:
        fields = asdict(self)
        fields["reference_kinds"] = list(self.reference_kinds)
        fields["opportunity_view_roles"] = list(self.opportunity_view_roles)
        body = {
            **fields,
            "policy_id": EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_POLICY_ID,
            "method_id": EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_METHOD_ID,
            "status_vocabulary": sorted(_STATUS_VALUES),
            "evidence_family_order": list(EVIDENCE_FAMILIES),
            "clinical_thresholds_defined": False,
            "negative_clinical_assertions_defined": False,
            "policy_sha256": "CONTENT-ADDRESS-PENDING",
        }
        body["policy_sha256"] = _canonical_sha256(body)
        return body


DEFAULT_EVENT_NATIVE_SIGNAL_QUALITY_POLICY = EventNativeSignalQualityPolicy()


def _source_channel_ledger(
    bundle: CanonicalEDFViewBundle,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    canonical = bundle.canonical_record.canonical_receipt
    header_by_id = {
        str(row["channel_id"]): row
        for row in bundle.canonical_record.source_header_receipt[
            "channel_signal_headers"
        ]
    }
    result: list[dict[str, object]] = []
    catalog: dict[str, dict[str, object]] = {}
    for channel in canonical["channels"]:
        channel_id = str(channel["channel_id"])
        header = header_by_id.get(channel_id)
        observed = bool(channel["observed"])
        calibration_status = (
            "measured"
            if observed
            and header is not None
            and header["clipping_qc_source"] == "edf_signal_header_calibration_rails_v1"
            else "not_evaluable"
        )
        calibration_reasons: list[str] = []
        if not observed:
            calibration_reasons.append("canonical_source_channel_unobserved")
        elif header is None:
            calibration_reasons.append("source_signal_header_row_unavailable")
        elif calibration_status == "not_evaluable":
            calibration_reasons.append("edf_header_calibration_rails_unavailable")
        row: dict[str, object] = {
            "channel_id": channel_id,
            "observed": observed,
            "imputed": bool(channel["imputed"]),
            "source_physical_unit": channel["source_physical_unit"],
            "scale_to_volts": float(channel["scale_to_volts"]),
            "sampling_rate_numerator": int(channel["sample_rate_numerator"]),
            "sampling_rate_denominator": int(channel["sample_rate_denominator"]),
            "sample_count": int(channel["sample_count"]),
            "acquisition_highpass_hz": channel["acquisition_highpass_hz"],
            "acquisition_lowpass_hz": channel["acquisition_lowpass_hz"],
            "acquisition_bandwidth_metadata_complete": bool(
                observed
                and channel["acquisition_highpass_hz"] is not None
                and channel["acquisition_lowpass_hz"] is not None
            ),
            "reference_label": channel["reference_label"],
            "adc_rail_detection_opportunity": {
                "status": calibration_status,
                "source_policy": (
                    None if header is None else header["clipping_qc_source"]
                ),
                "adc_lsb_volts": (
                    None
                    if calibration_status == "not_evaluable"
                    else float(header["adc_lsb_volts"])
                ),
                "reason_codes": calibration_reasons,
            },
        }
        result.append(row)
        catalog[channel_id] = row
    return result, catalog


def _source_rail_and_plateau_intervals(
    bundle: CanonicalEDFViewBundle,
    *,
    flat_samples: int,
    plateau_samples: int,
    tolerance_volts: float,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]],]:
    native = bundle.task_reference_views["findings_native_morphology"]["referential"]
    index_by_id = {
        str(row["unit_id"]): index
        for index, row in enumerate(native.receipt["output_units"])
    }
    header_by_id = {
        str(row["channel_id"]): row
        for row in bundle.canonical_record.source_header_receipt[
            "channel_signal_headers"
        ]
    }
    rail: dict[str, list[tuple[int, int]]] = {item: [] for item in STANDARD_19}
    fallback: dict[str, list[tuple[int, int]]] = {item: [] for item in STANDARD_19}
    for channel_id in STANDARD_19:
        header = header_by_id.get(channel_id)
        if header is None:
            continue
        signal = (
            native.tensor[index_by_id[channel_id]]
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
            .astype(np.float64, copy=False)
        )
        if header["clipping_qc_source"] == "edf_signal_header_calibration_rails_v1":
            rail_tolerance = max(
                float(tolerance_volts), 0.51 * float(header["adc_lsb_volts"])
            )
            at_rail = np.isclose(
                signal,
                float(header["adc_rail_minimum_volts"]),
                rtol=0.0,
                atol=rail_tolerance,
            ) | np.isclose(
                signal,
                float(header["adc_rail_maximum_volts"]),
                rtol=0.0,
                atol=rail_tolerance,
            )
            rail[channel_id] = _true_runs(at_rail, minimum_length=plateau_samples)
        else:
            differences = np.abs(np.diff(signal)) <= float(tolerance_volts)
            candidates = [
                (start, stop + 1)
                for start, stop in _true_runs(
                    differences, minimum_length=max(1, plateau_samples - 1)
                )
            ]
            fallback[channel_id] = [
                interval
                for interval in candidates
                if interval[1] - interval[0] < flat_samples
            ]
    return rail, fallback


def _direct_unit_intervals(
    values: torch.Tensor,
    *,
    observed: bool,
    flat_samples: int,
    tolerance_volts: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not observed:
        return [], []
    signal = (
        values.detach().cpu().to(torch.float32).numpy().astype(np.float64, copy=False)
    )
    finite = np.isfinite(signal)
    nonfinite = _true_runs(~finite)
    safe = np.where(finite, signal, 0.0)
    flat_differences = (
        (np.abs(np.diff(safe)) <= float(tolerance_volts)) & finite[:-1] & finite[1:]
    )
    flat = [
        (start, stop + 1)
        for start, stop in _true_runs(
            flat_differences, minimum_length=max(1, flat_samples - 1)
        )
    ]
    return flat, nonfinite


def _native_detection_rows(
    *,
    intervals: Sequence[tuple[int, int]],
    selected: tuple[int, int],
    view_receipt: Mapping[str, object],
    kind: str,
    assertion_level: str,
    source_channel_ids: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detected_start, detected_stop in _merge_intervals(intervals):
        overlap_start = max(detected_start, selected[0])
        overlap_stop = min(detected_stop, selected[1])
        if overlap_stop <= overlap_start:
            continue
        rows.append(
            {
                "kind": kind,
                "assertion_level": assertion_level,
                "source_channel_ids": list(source_channel_ids),
                "detected_global_native_sample_interval": [
                    detected_start,
                    detected_stop,
                ],
                "event_overlap_tensor_sample_interval": [
                    overlap_start,
                    overlap_stop,
                ],
                "event_overlap_recording_interval_seconds": [
                    view_tensor_index_to_recording_seconds(
                        dict(view_receipt), tensor_sample_index=overlap_start
                    ),
                    view_tensor_index_to_recording_seconds(
                        dict(view_receipt), tensor_sample_index=overlap_stop
                    ),
                ],
                "event_overlap_sample_count": overlap_stop - overlap_start,
            }
        )
    return rows


def _project_recording_intervals_to_view(
    rows: Sequence[Mapping[str, object]],
    *,
    view_receipt: Mapping[str, object],
    selected: tuple[int, int],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for row in rows:
        start_seconds, stop_seconds = (
            float(item) for item in row["event_overlap_recording_interval_seconds"]
        )
        start = recording_seconds_to_view_tensor_index(
            dict(view_receipt), recording_seconds=start_seconds, rounding="floor"
        )
        stop = recording_seconds_to_view_tensor_index(
            dict(view_receipt), recording_seconds=stop_seconds, rounding="ceil"
        )
        result.append((max(selected[0], start), min(selected[1], stop)))
    return _merge_intervals(result)


def _quality_kind_sample_counts(
    view_receipt: Mapping[str, object],
    *,
    unit_id: str,
    selected: tuple[int, int],
) -> dict[str, int]:
    intervals_by_kind: dict[str, list[tuple[int, int]]] = {}
    for row in view_receipt["masks"]["quality_invalid_intervals"]:  # type: ignore[index]
        if str(row["unit_id"]) != unit_id:
            continue
        kinds = [
            str(reason).split(":", 1)[1]
            for reason in row["reason_codes"]
            if str(reason).startswith("canonical_quality:")
        ]
        kind = kinds[0] if len(kinds) == 1 else "inherited_or_additional_quality"
        interval = tuple(int(item) for item in row["tensor_sample_interval"])
        intervals_by_kind.setdefault(kind, []).append(interval)
    return {
        kind: _interval_sample_count(_intersect_intervals(intervals, selected))
        for kind, intervals in sorted(intervals_by_kind.items())
    }


def _view_opportunity(
    *,
    view: object,
    opportunity_role: str,
    unit_id: str,
    bounded_interval_seconds: tuple[float, float],
    record_censoring: Mapping[str, object],
    direct_quality_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    receipt = view.receipt
    unit_index = next(
        index
        for index, row in enumerate(receipt["output_units"])
        if str(row["unit_id"]) == unit_id
    )
    unit = receipt["output_units"][unit_index]
    selected = _view_interval(receipt, bounded_interval_seconds)
    total = max(0, selected[1] - selected[0])
    values = view.tensor[unit_index].detach().cpu().to(torch.float32).contiguous()
    if values.ndim != 1 or int(values.shape[0]) != int(
        receipt["tensor_layout"]["tensor_sample_count"]
    ):
        raise ValueError("S01 view tensor shape drifted from its receipt")

    padding = [
        tuple(int(item) for item in row)
        for row in receipt["masks"]["padding_intervals"]
    ]
    edges = [
        tuple(int(item) for item in row)
        for row in receipt["masks"]["edge_invalid_intervals"]
    ]
    quality_all = [
        tuple(int(item) for item in row["tensor_sample_interval"])
        for row in receipt["masks"]["quality_invalid_intervals"]
        if str(row["unit_id"]) == unit_id
    ]
    direct = _project_recording_intervals_to_view(
        direct_quality_rows,
        view_receipt=receipt,
        selected=selected,
    )
    nonfinite_local: list[tuple[int, int]] = []
    if total:
        finite = torch.isfinite(values[selected[0] : selected[1]]).numpy()
        nonfinite_local = [
            (selected[0] + start, selected[0] + stop)
            for start, stop in _true_runs(~finite)
        ]

    invalid = np.zeros(total, dtype=bool)
    if not bool(unit["observed"]) or bool(unit["imputed"]):
        invalid[:] = True
    for collection in (padding, edges, quality_all, direct, nonfinite_local):
        overlaps = _intersect_intervals(collection, selected)
        _mark_intervals(
            invalid,
            [(start - selected[0], stop - selected[0]) for start, stop in overlaps],
        )
    usable = ~invalid
    invalid_count = int(np.count_nonzero(invalid))
    usable_count = int(np.count_nonzero(usable))
    if total != invalid_count + usable_count:
        raise AssertionError("S01 usable-sample denominator does not conserve")
    usable_intervals = [
        (selected[0] + start, selected[0] + stop) for start, stop in _true_runs(usable)
    ]
    invalid_intervals = [
        (selected[0] + start, selected[0] + stop) for start, stop in _true_runs(invalid)
    ]
    leading = int(np.argmax(~invalid)) if total and usable_count else total
    trailing = int(np.argmax((~invalid)[::-1])) if total and usable_count else total

    family_rows: list[dict[str, object]] = []
    family_catalog = {str(row["family"]): row for row in unit["evidence_eligibility"]}
    for family in EVIDENCE_FAMILIES:
        family_invalid = np.zeros(total, dtype=bool)
        capability = bool(family_catalog[family]["eligible"])
        if not capability or not bool(unit["observed"]) or bool(unit["imputed"]):
            family_invalid[:] = True
        for collection in (padding, edges, direct, nonfinite_local):
            overlaps = _intersect_intervals(collection, selected)
            _mark_intervals(
                family_invalid,
                [(start - selected[0], stop - selected[0]) for start, stop in overlaps],
            )
        family_quality = [
            tuple(int(item) for item in row["tensor_sample_interval"])
            for row in receipt["masks"]["quality_invalid_intervals"]
            if str(row["unit_id"]) == unit_id
            and family in row["disabled_evidence_families"]
        ]
        overlaps = _intersect_intervals(family_quality, selected)
        _mark_intervals(
            family_invalid,
            [(start - selected[0], stop - selected[0]) for start, stop in overlaps],
        )
        family_usable = total - int(np.count_nonzero(family_invalid))
        family_rows.append(
            {
                "family": family,
                "status": "measured" if family_usable > 0 else "not_evaluable",
                "capability_eligible": capability,
                "total_sample_count": total,
                "usable_sample_count": family_usable,
                "invalid_sample_count": total - family_usable,
                "usable_fraction": (
                    None if total == 0 else float(family_usable / total)
                ),
                "capability_reason_codes": list(family_catalog[family]["reason_codes"]),
            }
        )

    transform = receipt["transform_spec"]
    numerator, denominator, sfreq = _view_rate(receipt)
    status = "measured" if usable_count > 0 else "not_evaluable"
    censor_reasons: list[str] = []
    if bool(record_censoring["left"]):
        censor_reasons.append("requested_support_crosses_record_left_edge")
    if bool(record_censoring["right"]):
        censor_reasons.append("requested_support_crosses_record_right_edge")
    if not bool(unit["observed"]) or bool(unit["imputed"]):
        censor_reasons.append("typed_unit_unobserved_or_imputed")
    if leading:
        censor_reasons.append("left_boundary_has_unusable_samples")
    if trailing:
        censor_reasons.append("right_boundary_has_unusable_samples")
    quality_kind_counts = _quality_kind_sample_counts(
        receipt, unit_id=unit_id, selected=selected
    )
    return {
        "view_role": opportunity_role,
        "view_task_role": str(receipt["task_role"]),
        "view_id": str(receipt["view_id"]),
        "view_receipt_sha256": str(receipt["receipt_sha256"]),
        "view_tensor_sha256": str(receipt["processed_view_sha256"]),
        "view_mask_sha256": str(receipt["masks"]["mask_sha256"]),
        "transform_spec_sha256": str(transform["transform_spec_sha256"]),
        "reference_type": str(transform["reference"]["reference_type"]),
        "reference_matrix_sha256": str(transform["reference"]["matrix_sha256"]),
        "canonical_source_channel_ids": list(unit["canonical_source_channel_ids"]),
        "sampling_clock": {
            "sampling_rate_numerator": numerator,
            "sampling_rate_denominator": denominator,
            "sampling_rate_hz": sfreq,
        },
        "effective_bandwidth_hz": [
            float(item) for item in unit["effective_bandwidth_hz"]
        ],
        "effective_bandwidth_semantics": (
            "view_contract_capability_not_proof_of_unknown_acquisition_filter"
        ),
        "temporal_transform": {
            "filter_family": str(transform["filter"]["family"]),
            "filter_order": transform["filter"]["order"],
            "filter_highpass_hz": transform["filter"]["highpass_hz"],
            "filter_lowpass_hz": transform["filter"]["lowpass_hz"],
            "phase_policy": str(transform["filter"]["phase_policy"]),
            "resampler_implementation": str(transform["resampler"]["implementation"]),
            "edge_policy": str(transform["edge_handling"]["policy"]),
            "declared_left_invalid_samples": int(
                transform["edge_handling"]["left_invalid_samples"]
            ),
            "declared_right_invalid_samples": int(
                transform["edge_handling"]["right_invalid_samples"]
            ),
            "future_sample_access": bool(
                receipt["temporal_evidence"]["future_sample_access"]
            ),
            "onset_evidence_authorized_by_view": bool(
                receipt["temporal_evidence"]["onset_evidence_authorized"]
            ),
            "reference_operation_temporal_support": "instantaneous",
        },
        "status": status,
        "coverage_status": (
            "not_evaluable"
            if usable_count == 0
            else "complete"
            if usable_count == total
            else "limited"
        ),
        "sample_support": {
            "tensor_sample_interval": list(selected),
            "recording_interval_seconds": (
                [
                    view_tensor_index_to_recording_seconds(
                        receipt, tensor_sample_index=selected[0]
                    ),
                    view_tensor_index_to_recording_seconds(
                        receipt, tensor_sample_index=selected[1]
                    ),
                ]
                if total
                else None
            ),
            "total_sample_count": total,
        },
        "denominators": {
            "total_sample_count": total,
            "observed_sample_count": (
                total if bool(unit["observed"]) and not bool(unit["imputed"]) else 0
            ),
            "usable_sample_count": usable_count,
            "invalid_sample_count": invalid_count,
            "usable_fraction": None if total == 0 else float(usable_count / total),
            "padding_invalid_sample_count": _interval_sample_count(
                _intersect_intervals(padding, selected)
            ),
            "edge_invalid_sample_count": _interval_sample_count(
                _intersect_intervals(edges, selected)
            ),
            "view_quality_invalid_sample_count": _interval_sample_count(
                _intersect_intervals(quality_all, selected)
            ),
            "direct_native_quality_invalid_sample_count": _interval_sample_count(
                _intersect_intervals(direct, selected)
            ),
            "nonfinite_sample_count": _interval_sample_count(nonfinite_local),
            "union_semantics": (
                "invalid_categories_may_overlap_only_union_enters_denominator"
            ),
            "conservation_holds": total == usable_count + invalid_count,
        },
        "usable_intervals": _interval_rows(usable_intervals, view_receipt=receipt),
        "invalid_intervals": _interval_rows(invalid_intervals, view_receipt=receipt),
        "view_quality_kind_sample_counts": quality_kind_counts,
        "family_opportunities": family_rows,
        "boundary_censoring": {
            "left": bool(record_censoring["left"]) or leading > 0,
            "right": bool(record_censoring["right"]) or trailing > 0,
            "leading_unusable_sample_count": leading,
            "trailing_unusable_sample_count": trailing,
            "reason_codes": sorted(set(censor_reasons)),
        },
    }


def _typed_unit_rows(
    bundle: CanonicalEDFViewBundle,
    *,
    bounded_interval_seconds: tuple[float, float],
    record_censoring: Mapping[str, object],
    source_catalog: Mapping[str, Mapping[str, object]],
    flat_samples: int,
    plateau_samples: int,
    tolerance_volts: float,
) -> list[dict[str, object]]:
    source_rail, source_fallback = _source_rail_and_plateau_intervals(
        bundle,
        flat_samples=flat_samples,
        plateau_samples=plateau_samples,
        tolerance_volts=tolerance_volts,
    )
    result: list[dict[str, object]] = []
    for reference_kind in _REFERENCE_KINDS:
        native = bundle.task_reference_views["findings_native_morphology"][
            reference_kind
        ]
        native_selected = _view_interval(native.receipt, bounded_interval_seconds)
        for unit_index, unit in enumerate(native.receipt["output_units"]):
            unit_id = str(unit["unit_id"])
            source_ids = tuple(
                str(item) for item in unit["canonical_source_channel_ids"]
            )
            typed_unit_type = (
                "electrode" if reference_kind == "referential" else "whole_bipolar_lead"
            )
            observed = bool(unit["observed"]) and not bool(unit["imputed"])
            direct_flat, direct_nonfinite = _direct_unit_intervals(
                native.tensor[unit_index],
                observed=observed,
                flat_samples=flat_samples,
                tolerance_volts=tolerance_volts,
            )
            rail_complete = observed and all(
                source_catalog[source_id]["adc_rail_detection_opportunity"]["status"]
                == "measured"
                for source_id in source_ids
            )
            rail_intervals = _merge_intervals(
                [
                    interval
                    for source_id in source_ids
                    for interval in source_rail[source_id]
                ]
            )
            fallback_intervals = _merge_intervals(
                [
                    interval
                    for source_id in source_ids
                    for interval in source_fallback[source_id]
                ]
            )
            flat_rows = _native_detection_rows(
                intervals=direct_flat,
                selected=native_selected,
                view_receipt=native.receipt,
                kind="flat_or_constant_repeated_value_run",
                assertion_level="measured",
                source_channel_ids=source_ids,
            )
            nonfinite_rows = _native_detection_rows(
                intervals=direct_nonfinite,
                selected=native_selected,
                view_receipt=native.receipt,
                kind="nonfinite_native_sample_run",
                assertion_level="measured",
                source_channel_ids=source_ids,
            )
            rail_rows = _native_detection_rows(
                intervals=rail_intervals,
                selected=native_selected,
                view_receipt=native.receipt,
                kind="source_adc_calibration_rail_plateau",
                assertion_level="measured",
                source_channel_ids=source_ids,
            )
            fallback_rows = _native_detection_rows(
                intervals=fallback_intervals,
                selected=native_selected,
                view_receipt=native.receipt,
                kind="source_local_repeated_value_plateau_candidate",
                assertion_level="model_candidate",
                source_channel_ids=source_ids,
            )
            direct_rows = sorted(
                [*flat_rows, *nonfinite_rows, *rail_rows, *fallback_rows],
                key=lambda row: (
                    float(row["event_overlap_recording_interval_seconds"][0]),
                    float(row["event_overlap_recording_interval_seconds"][1]),
                    str(row["kind"]),
                ),
            )
            segment = native.tensor[unit_index, native_selected[0] : native_selected[1]]
            native_status = (
                "measured" if observed and int(segment.numel()) else "not_evaluable"
            )
            reason_codes: list[str] = []
            if not observed:
                reason_codes.append("typed_unit_unobserved_or_imputed")
            if int(segment.numel()) == 0:
                reason_codes.append("no_complete_native_samples_in_support")
            source_bandwidth_complete = observed and all(
                bool(
                    source_catalog[source_id]["acquisition_bandwidth_metadata_complete"]
                )
                for source_id in source_ids
            )
            view_opportunities = [
                _view_opportunity(
                    view=bundle.task_reference_views[role][reference_kind],
                    opportunity_role=role,
                    unit_id=unit_id,
                    bounded_interval_seconds=bounded_interval_seconds,
                    record_censoring=record_censoring,
                    direct_quality_rows=direct_rows,
                )
                for role in _VIEW_ROLES
            ]
            result.append(
                {
                    "unit_id": unit_id,
                    "typed_unit_type": typed_unit_type,
                    "reference_kind": reference_kind,
                    "canonical_source_channel_ids": list(source_ids),
                    "ordered_bipolar_electrode_ids": (
                        list(_ORDERED_TCP_CARRIERS[unit_id])
                        if typed_unit_type == "whole_bipolar_lead"
                        else None
                    ),
                    "whole_bipolar_lead_identity_preserved": (
                        typed_unit_type == "whole_bipolar_lead"
                    ),
                    "bipolar_endpoint_fact_projection_authorized": False,
                    "observed": observed,
                    "imputed": bool(unit["imputed"]),
                    "source_acquisition_bandwidth_metadata_complete": (
                        source_bandwidth_complete
                    ),
                    "direct_native_qc": {
                        "status": native_status,
                        "native_view_id": str(native.receipt["view_id"]),
                        "native_view_receipt_sha256": str(
                            native.receipt["receipt_sha256"]
                        ),
                        "native_view_tensor_sha256": str(
                            native.receipt["processed_view_sha256"]
                        ),
                        "native_view_mask_sha256": str(
                            native.receipt["masks"]["mask_sha256"]
                        ),
                        "tensor_sample_interval": list(native_selected),
                        "recording_interval_seconds": (
                            [
                                view_tensor_index_to_recording_seconds(
                                    native.receipt,
                                    tensor_sample_index=native_selected[0],
                                ),
                                view_tensor_index_to_recording_seconds(
                                    native.receipt,
                                    tensor_sample_index=native_selected[1],
                                ),
                            ]
                            if int(segment.numel())
                            else None
                        ),
                        "sample_count": int(segment.numel()),
                        "native_segment_sha256": _segment_sha256(
                            segment,
                            unit_id=unit_id,
                            tensor_interval=native_selected,
                        ),
                        "nonfinite_assessment": {
                            "status": native_status,
                            "sample_count": (
                                None
                                if native_status == "not_evaluable"
                                else _interval_sample_count(
                                    _intersect_intervals(
                                        direct_nonfinite, native_selected
                                    )
                                )
                            ),
                            "canonical_loader_rejects_nonfinite_source": True,
                            "negative_clinical_assertion_authorized": False,
                        },
                        "flat_or_constant_assessment": {
                            "status": native_status,
                            "flagged_sample_count": (
                                None
                                if native_status == "not_evaluable"
                                else _interval_sample_count(
                                    _intersect_intervals(direct_flat, native_selected)
                                )
                            ),
                            "run_count": None
                            if native_status == "not_evaluable"
                            else len(flat_rows),
                            "technical_only": True,
                            "negative_clinical_assertion_authorized": False,
                        },
                        "adc_rail_plateau_assessment": {
                            "status": "measured" if rail_complete else "not_evaluable",
                            "qualified_carrier_count": sum(
                                source_catalog[source_id][
                                    "adc_rail_detection_opportunity"
                                ]["status"]
                                == "measured"
                                for source_id in source_ids
                            ),
                            "required_carrier_count": len(source_ids),
                            "flagged_sample_count": (
                                _interval_sample_count(
                                    _intersect_intervals(
                                        rail_intervals, native_selected
                                    )
                                )
                                if rail_complete
                                else None
                            ),
                            "run_count": len(rail_rows) if rail_complete else None,
                            "clinical_clipping_or_saturation_term_authorized": False,
                            "reason_codes": (
                                []
                                if rail_complete
                                else [
                                    "complete_source_adc_rail_opportunity_unavailable"
                                ]
                            ),
                        },
                        "unqualified_local_plateau_fallback": {
                            "status": (
                                "measured"
                                if observed and not rail_complete
                                else "not_evaluable"
                            ),
                            "assertion_level": "model_candidate",
                            "flagged_sample_count": (
                                _interval_sample_count(
                                    _intersect_intervals(
                                        fallback_intervals, native_selected
                                    )
                                )
                                if observed and not rail_complete
                                else None
                            ),
                            "run_count": (
                                len(fallback_rows)
                                if observed and not rail_complete
                                else None
                            ),
                            "may_be_called_clipping_or_saturation": False,
                            "reason_codes": (
                                ["header_rails_available_fallback_not_used"]
                                if rail_complete
                                else []
                                if observed
                                else ["typed_unit_unobserved_or_imputed"]
                            ),
                        },
                        "quality_interval_ledger": direct_rows,
                        "reason_codes": reason_codes,
                    },
                    "view_opportunities": view_opportunities,
                }
            )
    return result


def materialize_event_native_signal_quality_findings_v1(
    *,
    event_id: str,
    bundle: CanonicalEDFViewBundle,
    requested_analysis_interval_seconds: Sequence[float],
    policy: EventNativeSignalQualityPolicy = DEFAULT_EVENT_NATIVE_SIGNAL_QUALITY_POLICY,
) -> dict[str, Any]:
    """Measure one exact S01 event ledger from a canonical EDF bundle."""

    _identifier(event_id, "event_id")
    if not isinstance(bundle, CanonicalEDFViewBundle):
        raise TypeError("bundle must be a CanonicalEDFViewBundle")
    if not isinstance(policy, EventNativeSignalQualityPolicy):
        raise TypeError("policy must be EventNativeSignalQualityPolicy")
    materialization = validate_canonical_edf_materialization(bundle)
    canonical = bundle.canonical_record.canonical_receipt
    duration = float(canonical["recording_duration_seconds"])
    requested, bounded = _requested_interval(
        requested_analysis_interval_seconds,
        duration_seconds=duration,
    )
    record_censoring: dict[str, object] = {
        "left": requested[0] < -_TOL,
        "right": requested[1] > duration + _TOL,
        "left_censored_seconds": max(0.0, -requested[0]),
        "right_censored_seconds": max(0.0, requested[1] - duration),
        "reason_codes": [
            *(
                ["requested_support_crosses_record_left_edge"]
                if requested[0] < -_TOL
                else []
            ),
            *(
                ["requested_support_crosses_record_right_edge"]
                if requested[1] > duration + _TOL
                else []
            ),
        ],
    }
    preprocessing = materialization["preprocessing_config"]
    source_rate = bundle.canonical_record.source_header_receipt[
        "channel_signal_headers"
    ][0]
    sfreq = int(source_rate["sampling_rate_numerator"]) / int(
        source_rate["sampling_rate_denominator"]
    )
    flat_samples = max(
        2, int(math.ceil(float(preprocessing["flatline_run_seconds"]) * sfreq))
    )
    plateau_samples = max(
        2, int(math.ceil(float(preprocessing["clipping_run_seconds"]) * sfreq))
    )
    tolerance_volts = float(preprocessing["qc_tolerance_volts"])
    source_ledger, source_catalog = _source_channel_ledger(bundle)
    typed_units = _typed_unit_rows(
        bundle,
        bounded_interval_seconds=bounded,
        record_censoring=record_censoring,
        source_catalog=source_catalog,
        flat_samples=flat_samples,
        plateau_samples=plateau_samples,
        tolerance_volts=tolerance_volts,
    )
    native_opportunities = [row["view_opportunities"][0] for row in typed_units]
    native_total = sum(
        int(row["denominators"]["total_sample_count"]) for row in native_opportunities
    )
    native_usable = sum(
        int(row["denominators"]["usable_sample_count"]) for row in native_opportunities
    )
    event_denominators = {
        "typed_unit_count": len(typed_units),
        "electrode_unit_count": sum(
            row["typed_unit_type"] == "electrode" for row in typed_units
        ),
        "whole_bipolar_lead_unit_count": sum(
            row["typed_unit_type"] == "whole_bipolar_lead" for row in typed_units
        ),
        "observed_typed_unit_count": sum(bool(row["observed"]) for row in typed_units),
        "not_evaluable_native_typed_unit_count": sum(
            row["view_opportunities"][0]["status"] == "not_evaluable"
            for row in typed_units
        ),
        "view_opportunity_count": sum(
            len(row["view_opportunities"]) for row in typed_units
        ),
        "native_total_sample_opportunity_count": native_total,
        "native_usable_sample_count": native_usable,
        "native_invalid_sample_count": native_total - native_usable,
        "native_usable_fraction": (
            None if native_total == 0 else float(native_usable / native_total)
        ),
        "aggregation_semantics": (
            "typed_electrode_and_whole_lead_sample_opportunities_not_independent_clinical_observations"
        ),
    }
    body: dict[str, Any] = {
        "schema_version": EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_SCHEMA_VERSION,
        "method_id": EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_METHOD_ID,
        "event_id": event_id,
        "recording_id": canonical["recording_id"],
        "canonical_signal_id": canonical["canonical_signal_id"],
        "canonical_receipt_sha256": canonical["receipt_sha256"],
        "source_signal_sha256": canonical["source_signal_sha256"],
        "source_header_receipt_sha256": bundle.canonical_record.source_header_receipt[
            "receipt_sha256"
        ],
        "canonical_materialization_receipt_sha256": materialization["receipt_sha256"],
        "requested_analysis_interval_seconds": list(requested),
        "materialized_analysis_interval_seconds": list(bounded),
        "recording_duration_seconds": duration,
        "record_censoring": record_censoring,
        "measurement_parameters": {
            "source": "canonical_materialization_preprocessing_config",
            "source_receipt_sha256": materialization["receipt_sha256"],
            "source_sampling_rate_hz": sfreq,
            "flatline_run_seconds": float(preprocessing["flatline_run_seconds"]),
            "flatline_minimum_samples": flat_samples,
            "adc_rail_or_plateau_run_seconds": float(
                preprocessing["clipping_run_seconds"]
            ),
            "adc_rail_or_plateau_minimum_samples": plateau_samples,
            "repeated_value_tolerance_volts": tolerance_volts,
            "full_record_extrema_used_as_adc_rails": False,
        },
        "policy": policy.to_dict(),
        "source_channel_ledger": source_ledger,
        "typed_unit_roster": typed_units,
        "event_denominators": event_denominators,
        "gap_assessment": {
            "materialized_clock_status": "measured",
            "structural_sample_hole_count": 0,
            "structural_sample_hole_semantics": (
                "no_holes_in_admitted_fixed_rate_materialized_eeg_tensor"
            ),
            "acquisition_discontinuity_status": "not_evaluable",
            "acquisition_discontinuity_count": None,
            "reason_codes": [
                "edf_discontinuity_tal_not_opened_under_eeg_only_annotation_firewall"
            ],
            "absence_of_materialized_holes_is_not_absence_of_acquisition_gaps": True,
        },
        "input_firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _self_hash(body)
    return validate_event_native_signal_quality_findings_v1(body)


def _require_exact_keys(
    value: object,
    keys: set[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    if set(value) != keys:
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def validate_event_native_signal_quality_findings_v1(
    value: object,
) -> dict[str, Any]:
    """Strictly validate S01 structure and denominator conservation.

    This receipt-only validator proves internal accounting.  Signal identity
    and numeric replay require the host-supplied bundle accepted by
    :func:`replay_event_native_signal_quality_findings_v1`.
    """

    top_keys = {
        "schema_version",
        "method_id",
        "event_id",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_header_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "requested_analysis_interval_seconds",
        "materialized_analysis_interval_seconds",
        "recording_duration_seconds",
        "record_censoring",
        "measurement_parameters",
        "policy",
        "source_channel_ledger",
        "typed_unit_roster",
        "event_denominators",
        "gap_assessment",
        "input_firewall",
        "authorization",
        "receipt_sha256",
    }
    data = deepcopy(_require_exact_keys(value, top_keys, "S01 Findings"))
    if data["schema_version"] != EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_SCHEMA_VERSION:
        raise ValueError("unsupported S01 Findings schema")
    if data["method_id"] != EVENT_NATIVE_SIGNAL_QUALITY_FINDINGS_METHOD_ID:
        raise ValueError("S01 Findings method_id drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "source_header_receipt_sha256",
        "canonical_materialization_receipt_sha256",
        "receipt_sha256",
    ):
        _sha(data[field], field)
    if data["input_firewall"] != _FIREWALL:
        raise ValueError("S01 Findings violates the EEG-only firewall")
    if data["authorization"] != _AUTHORIZATION:
        raise ValueError("S01 Findings authorization boundary drifted")
    if data["policy"] != DEFAULT_EVENT_NATIVE_SIGNAL_QUALITY_POLICY.to_dict():
        raise ValueError("S01 Findings policy drifted")
    duration = _finite(data["recording_duration_seconds"], "recording duration")
    requested, bounded = _requested_interval(
        data["requested_analysis_interval_seconds"], duration_seconds=duration
    )
    if data["materialized_analysis_interval_seconds"] != list(bounded):
        raise ValueError("S01 bounded support does not replay from record edges")
    expected_censor = {
        "left": requested[0] < -_TOL,
        "right": requested[1] > duration + _TOL,
        "left_censored_seconds": max(0.0, -requested[0]),
        "right_censored_seconds": max(0.0, requested[1] - duration),
        "reason_codes": [
            *(
                ["requested_support_crosses_record_left_edge"]
                if requested[0] < -_TOL
                else []
            ),
            *(
                ["requested_support_crosses_record_right_edge"]
                if requested[1] > duration + _TOL
                else []
            ),
        ],
    }
    if data["record_censoring"] != expected_censor:
        raise ValueError("S01 record censoring does not replay")

    if not isinstance(data["source_channel_ledger"], list):
        raise TypeError("S01 source channel ledger must be an array")
    if [row.get("channel_id") for row in data["source_channel_ledger"]] != list(
        STANDARD_19
    ):
        raise ValueError("S01 source channel ledger order or coverage drifted")
    for row in data["source_channel_ledger"]:
        if row["imputed"] is not False:
            raise ValueError("S01 canonical source channels may never be imputed")
        opportunity = row["adc_rail_detection_opportunity"]
        if opportunity["status"] not in _STATUS_VALUES:
            raise ValueError("S01 ADC-rail opportunity status is invalid")
        if opportunity["status"] == "measured" and opportunity["reason_codes"]:
            raise ValueError(
                "measured ADC-rail opportunity cannot carry failure reasons"
            )

    expected_units = [
        *STANDARD_19,
        *(f"{left}-{right}" for left, right in TCP_20_EDGES),
    ]
    rows = data["typed_unit_roster"]
    if (
        not isinstance(rows, list)
        or [row.get("unit_id") for row in rows] != expected_units
    ):
        raise ValueError("S01 typed-unit roster is incomplete or out of order")
    native_total = 0
    native_usable = 0
    observed_count = 0
    not_evaluable_count = 0
    for index, row in enumerate(rows):
        electrode = index < len(STANDARD_19)
        expected_type = "electrode" if electrode else "whole_bipolar_lead"
        expected_reference = "referential" if electrode else "tcp_bipolar"
        if (
            row["typed_unit_type"] != expected_type
            or row["reference_kind"] != expected_reference
        ):
            raise ValueError("S01 typed-unit identity/reference semantics drifted")
        if electrode:
            expected_sources = [str(row["unit_id"])]
        else:
            edge = TCP_20_EDGES[index - len(STANDARD_19)]
            expected_sources = sorted(edge)
        if sorted(row["canonical_source_channel_ids"]) != expected_sources:
            raise ValueError("S01 whole-lead carrier identity drifted")
        expected_ordered = (
            None if electrode else list(TCP_20_EDGES[index - len(STANDARD_19)])
        )
        if row["ordered_bipolar_electrode_ids"] != expected_ordered:
            raise ValueError("S01 bipolar orientation/carrier order drifted")
        if row["bipolar_endpoint_fact_projection_authorized"] is not False:
            raise ValueError("S01 illegally projected a whole lead to its endpoints")
        if bool(row["whole_bipolar_lead_identity_preserved"]) is not (not electrode):
            raise ValueError("S01 whole-lead identity flag drifted")
        observed_count += int(bool(row["observed"]))
        direct = row["direct_native_qc"]
        if direct["status"] not in _STATUS_VALUES:
            raise ValueError("S01 direct native QC status is invalid")
        _sha(direct["native_segment_sha256"], "native_segment_sha256")
        for ledger_row in direct["quality_interval_ledger"]:
            if ledger_row["kind"] not in {
                "flat_or_constant_repeated_value_run",
                "nonfinite_native_sample_run",
                "source_adc_calibration_rail_plateau",
                "source_local_repeated_value_plateau_candidate",
            }:
                raise ValueError("S01 emitted an unauthorized quality category")
            expected_assertion = (
                "model_candidate"
                if ledger_row["kind"] == "source_local_repeated_value_plateau_candidate"
                else "measured"
            )
            if ledger_row["assertion_level"] != expected_assertion:
                raise ValueError("S01 quality assertion level drifted")
        opportunities = row["view_opportunities"]
        if [item.get("view_role") for item in opportunities] != list(_VIEW_ROLES):
            raise ValueError("S01 typed unit lacks its three view opportunities")
        for opportunity in opportunities:
            if opportunity["status"] not in _STATUS_VALUES:
                raise ValueError("S01 view opportunity status is invalid")
            denominators = opportunity["denominators"]
            total = int(denominators["total_sample_count"])
            usable = int(denominators["usable_sample_count"])
            invalid = int(denominators["invalid_sample_count"])
            if min(total, usable, invalid) < 0 or total != usable + invalid:
                raise ValueError("S01 view denominator does not conserve")
            if denominators["conservation_holds"] is not True:
                raise ValueError("S01 denominator conservation flag is false")
            expected_status = "measured" if usable > 0 else "not_evaluable"
            if opportunity["status"] != expected_status:
                raise ValueError(
                    "S01 not_evaluable/measured status contradicts opportunity"
                )
            families = opportunity["family_opportunities"]
            if [item.get("family") for item in families] != list(EVIDENCE_FAMILIES):
                raise ValueError("S01 family denominator roster drifted")
            for family in families:
                family_total = int(family["total_sample_count"])
                family_usable = int(family["usable_sample_count"])
                family_invalid = int(family["invalid_sample_count"])
                if (
                    family_total != total
                    or family_total != family_usable + family_invalid
                ):
                    raise ValueError("S01 family denominator does not conserve")
                family_status = "measured" if family_usable > 0 else "not_evaluable"
                if family["status"] != family_status:
                    raise ValueError(
                        "S01 family opportunity status contradicts samples"
                    )
        native = opportunities[0]
        native_total += int(native["denominators"]["total_sample_count"])
        native_usable += int(native["denominators"]["usable_sample_count"])
        not_evaluable_count += int(native["status"] == "not_evaluable")

    denominators = data["event_denominators"]
    expected_aggregate = {
        "typed_unit_count": len(rows),
        "electrode_unit_count": len(STANDARD_19),
        "whole_bipolar_lead_unit_count": len(TCP_20_EDGES),
        "observed_typed_unit_count": observed_count,
        "not_evaluable_native_typed_unit_count": not_evaluable_count,
        "view_opportunity_count": len(rows) * len(_VIEW_ROLES),
        "native_total_sample_opportunity_count": native_total,
        "native_usable_sample_count": native_usable,
        "native_invalid_sample_count": native_total - native_usable,
        "native_usable_fraction": (
            None if native_total == 0 else float(native_usable / native_total)
        ),
        "aggregation_semantics": (
            "typed_electrode_and_whole_lead_sample_opportunities_not_independent_clinical_observations"
        ),
    }
    if denominators != expected_aggregate:
        raise ValueError("S01 event denominator aggregate does not replay")
    expected_gap = {
        "materialized_clock_status": "measured",
        "structural_sample_hole_count": 0,
        "structural_sample_hole_semantics": (
            "no_holes_in_admitted_fixed_rate_materialized_eeg_tensor"
        ),
        "acquisition_discontinuity_status": "not_evaluable",
        "acquisition_discontinuity_count": None,
        "reason_codes": [
            "edf_discontinuity_tal_not_opened_under_eeg_only_annotation_firewall"
        ],
        "absence_of_materialized_holes_is_not_absence_of_acquisition_gaps": True,
    }
    if data["gap_assessment"] != expected_gap:
        raise ValueError("S01 gap-assessment boundary drifted")
    if data["receipt_sha256"] != _self_hash(data):
        raise ValueError("S01 receipt_sha256 does not bind its content")
    return data


def replay_event_native_signal_quality_findings_v1(
    value: object,
    *,
    event_id: str,
    fresh_bundle: CanonicalEDFViewBundle,
    requested_analysis_interval_seconds: Sequence[float],
    policy: EventNativeSignalQualityPolicy = DEFAULT_EVENT_NATIVE_SIGNAL_QUALITY_POLICY,
) -> dict[str, Any]:
    """Recompute S01 from a freshly reopened EDF bundle and require identity."""

    observed = validate_event_native_signal_quality_findings_v1(value)
    expected = materialize_event_native_signal_quality_findings_v1(
        event_id=event_id,
        bundle=fresh_bundle,
        requested_analysis_interval_seconds=requested_analysis_interval_seconds,
        policy=policy,
    )
    if observed != expected:
        raise ValueError("S01 Findings do not replay exactly from fresh native EEG")
    return expected
