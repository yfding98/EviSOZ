"""Deterministic EEG-only Findings for one canonical adaptive event window.

This is the first executable Stage-C baseline.  It consumes only an adaptive
search/window receipt, an immutable canonical EEG receipt, and host-supplied
task-view tensors whose hashes match their view receipts.  It never accepts
annotations, spreadsheets, clinical text, labels, or physician conclusions.

The producer emits quantitative observations, not clinical terminology:
quality coverage, frequency/rhythm/amplitude measurements, robust change
points, reference-specific field extrema, and onset/later-involvement
candidates.  Spike/IED labels, ACNS evolution, a confirmed seizure, cortical
SOZ, epileptogenic zone, and causal propagation are deliberately out of scope.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .adaptive_event_window import validate_adaptive_event_analysis_window
from .adaptive_search import validate_adaptive_search_receipt
from .canonical_adaptive_binding import (
    validate_canonical_adaptive_binding_against_receipt,
)
from .canonical_signal_views import (
    CANONICAL_EDF_ONSET_TRANSFORM_NAME,
    ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY,
    ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,
    ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY,
    ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,
    recording_seconds_to_view_tensor_index,
    validate_canonical_signal_receipt,
    validate_signal_view_receipt,
    view_tensor_index_to_recording_seconds,
)
from .event_findings_validation import validate_event_eeg_findings_payload


DETERMINISTIC_EVENT_FINDINGS_METHOD_ID = "DETERMINISTIC-EVENT-FINDINGS-V1"
DETERMINISTIC_VIEW_TENSOR_HASH_DOMAIN = "clinical-eeg-view-float32-le-v1"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)
_ALLOWED_VIEW_ROLES = {
    "findings_native",
    "findings_clinical",
    "findings_native_morphology",
    "onset_causal",
    "context_offline",
    "spatial_reference",
}
_ROLE_PRIORITY = {
    "findings_clinical": 0,
    "context_offline": 0,
    "findings_native": 1,
    "findings_native_morphology": 1,
    "onset_causal": 2,
    "spatial_reference": 3,
}

# Internal segmentation is deliberately non-clinical.  The v1 wire contract
# predates this producer and still names these four relative signal segments
# ``baseline/early_ictal/evolved_ictal/recovery``.  Keep the compatibility
# mapping at the serialization edge; S1/S2 membership is never an event-
# qualification decision and must not be read as a confirmed ictal state.
_SEGMENT_TO_V1_PHASE = {
    "S0": "baseline",
    "S1": "early_ictal",
    "S2": "evolved_ictal",
    "S3": "recovery",
}
_LEFT_ELECTRODES = {
    "FP1",
    "F3",
    "F7",
    "C3",
    "T7",
    "P3",
    "P7",
    "O1",
}
_RIGHT_ELECTRODES = {
    "FP2",
    "F4",
    "F8",
    "C4",
    "T8",
    "P4",
    "P8",
    "O2",
}
_MIDLINE_ELECTRODES = {"FZ", "CZ", "PZ"}


@dataclass(frozen=True)
class DeterministicEventFindingsPolicy:
    window_seconds: float = 1.0
    step_seconds: float = 0.5
    minimum_baseline_windows: int = 4
    change_score_threshold: float = 3.5
    sustained_change_windows: int = 2
    onset_tolerance_seconds: float = 1.5
    near_synchronous_seconds: float = 0.75
    later_involvement_delay_seconds: float = 2.0
    maximum_ranked_candidates: int = 3
    maximum_descriptive_units: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY = DeterministicEventFindingsPolicy()


@dataclass(frozen=True)
class DeterministicViewInput:
    view_receipt: object
    tensor: torch.Tensor
    onset_fir_response_qualification: object | None = None
    onset_fir_clinical_admission_qualification: object | None = None


@dataclass(frozen=True)
class _PreparedView:
    receipt: dict[str, Any]
    tensor: np.ndarray
    unit_ids: tuple[str, ...]
    unit_types: tuple[str, ...]
    sampling_rate_hz: float
    final_tensor_interval: tuple[int, int]


@dataclass(frozen=True)
class _FeatureGrid:
    tensor_starts: np.ndarray
    tensor_stops: np.ndarray
    recording_starts: np.ndarray
    recording_stops: np.ndarray
    rms_uv: np.ndarray
    peak_to_peak_uv: np.ndarray
    line_length_uv: np.ndarray
    dominant_frequency_hz: np.ndarray
    spectral_concentration: np.ndarray
    spectral_entropy: np.ndarray
    rhythmicity_index: np.ndarray
    band_ratio: np.ndarray
    amplitude_valid: np.ndarray
    spectral_valid: np.ndarray
    spatial_valid: np.ndarray


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{context} is not event-contract ID compatible")
    return value


def deterministic_view_tensor_sha256(
    tensor: torch.Tensor,
    *,
    unit_ids: Sequence[str],
) -> str:
    """Hash one ``[unit,time]`` view tensor using a frozen serialization."""

    values = tensor.detach().cpu().to(torch.float32).contiguous()
    if values.ndim != 2 or values.shape[0] != len(unit_ids):
        raise ValueError("view tensor shape and unit order are inconsistent")
    if not torch.isfinite(values).all():
        raise ValueError("view tensor must be finite")
    header = {
        "domain": DETERMINISTIC_VIEW_TENSOR_HASH_DOMAIN,
        "dtype": "float32-le",
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "unit_ids": [str(item) for item in unit_ids],
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
    digest.update(values.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _view_sampling_rate(view: Mapping[str, Any]) -> float:
    clock = view["transform_spec"]["output_clock"]
    return float(clock["sampling_rate_numerator"]) / float(
        clock["sampling_rate_denominator"]
    )


def _canonical_edf_onset_transform(
    *,
    receipt: Mapping[str, Any],
    canonical: Mapping[str, Any],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = [receipt]
    trusted = {} if trusted_parent_views is None else trusted_parent_views
    for binding in receipt["parent_view_bindings"]:
        parent_id = str(binding["view_id"])
        if parent_id not in trusted:
            continue
        candidates.append(
            validate_signal_view_receipt(
                trusted[parent_id],
                canonical,
            )
        )
    for candidate in candidates:
        transform = candidate["transform_spec"]
        if transform["transform_name"] == CANONICAL_EDF_ONSET_TRANSFORM_NAME:
            return transform
    return None


def _validate_canonical_edf_onset_admission(
    *,
    item: DeterministicViewInput,
    receipt: Mapping[str, Any],
    canonical: Mapping[str, Any],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> None:
    """Bind canonical EDF onset admission to both replayed qualification layers.

    Generic/legacy causal views retain their existing contract.  A view from
    the host canonical EDF materializer is stricter: the transform name makes
    the qualification key and full replayable receipt mandatory, including
    for an instantaneous spatial-reference child.
    """

    onset_transform = _canonical_edf_onset_transform(
        receipt=receipt,
        canonical=canonical,
        trusted_parent_views=trusted_parent_views,
    )
    supplied_response = item.onset_fir_response_qualification
    supplied_clinical = item.onset_fir_clinical_admission_qualification
    if onset_transform is None:
        if supplied_response is not None or supplied_clinical is not None:
            raise ValueError(
                "onset FIR qualification was supplied for a noncanonical view"
            )
        return
    if supplied_response is None:
        raise ValueError(
            "canonical EDF onset Findings admission requires a replayable FIR "
            "response qualification"
        )

    # Local import avoids making the metadata-only signal-view module depend
    # on scipy while still replaying the exact host qualification at the
    # Findings boundary.
    from .canonical_edf_materialization import (
        validate_onset_causal_fir_clinical_admission_qualification,
        validate_onset_causal_fir_response_qualification,
    )

    qualification = validate_onset_causal_fir_response_qualification(supplied_response)
    design = qualification["design"]
    measurement = qualification["measurement"]
    software = onset_transform["software_versions"]
    output_clock = onset_transform["output_clock"]
    output_rate = float(output_clock["sampling_rate_numerator"]) / float(
        output_clock["sampling_rate_denominator"]
    )
    if not math.isclose(float(design["sampling_rate_hz"]), output_rate, abs_tol=1e-12):
        raise ValueError("onset FIR qualification uses a different sampling clock")
    if int(design["numtaps"]) != int(onset_transform["filter"]["order"]) + 1:
        raise ValueError("onset FIR qualification uses a different filter order")
    try:
        target_band = tuple(
            float(value)
            for value in str(software["fir_design_target_band_hz"]).split(",")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "canonical onset transform lacks its design target band"
        ) from exc
    if len(target_band) != 2 or target_band != (
        float(design["target_highpass_hz"]),
        float(design["target_lowpass_hz"]),
    ):
        raise ValueError("onset FIR transform and qualification target bands differ")
    if (
        software.get("fir_response_qualification_sha256")
        != qualification["receipt_sha256"]
    ):
        raise ValueError("onset FIR transform and qualification hashes differ")
    measured_band = measurement["measured_minus_3db_bandwidth_hz"]
    if float(onset_transform["filter"]["highpass_hz"]) != float(
        measured_band[0]
    ) or float(onset_transform["filter"]["lowpass_hz"]) != float(measured_band[1]):
        raise ValueError("onset FIR transform does not expose measured bandwidth")

    response_authorized = bool(qualification["target_band_claim_authorized"])
    expected_marker = "true" if response_authorized else "false"
    if software.get(ONSET_FIR_RESPONSE_AUTHORIZATION_SOFTWARE_KEY) != expected_marker:
        raise ValueError(
            "canonical onset transform authorization marker disagrees with "
            "replayed FIR qualification"
        )

    clinical_marker = software.get(
        ONSET_FIR_CLINICAL_ADMISSION_AUTHORIZATION_SOFTWARE_KEY
    )
    clinical_hash = software.get("fir_clinical_admission_qualification_sha256")
    if clinical_marker is None and clinical_hash is None:
        if supplied_clinical is not None:
            raise ValueError(
                "clinical admission qualification was supplied for a legacy FIR view"
            )
        if response_authorized:
            raise ValueError(
                "legacy response-qualified onset view lacks clinical admission qualification"
            )
        clinical_authorized = False
    elif clinical_marker is None or clinical_hash is None:
        raise ValueError("canonical onset transform has a partial clinical gate")
    else:
        if supplied_clinical is None:
            raise ValueError(
                "canonical EDF onset Findings admission requires a replayable "
                "clinical admission qualification"
            )
        clinical = validate_onset_causal_fir_clinical_admission_qualification(
            supplied_clinical
        )
        if clinical["input_receipts"]["fir_response_qualification"] != qualification:
            raise ValueError(
                "clinical admission qualification binds a different FIR response"
            )
        selection = clinical["input_receipts"]["fir_design_selection"]
        if software.get("fir_design_selection_sha256") != selection["receipt_sha256"]:
            raise ValueError(
                "clinical admission qualification binds a different FIR selection"
            )
        if clinical_hash != clinical["receipt_sha256"]:
            raise ValueError(
                "canonical onset transform and clinical qualification hashes differ"
            )
        clinical_authorized = bool(clinical["clinical_onset_support_authorized"])
        expected_clinical_marker = "true" if clinical_authorized else "false"
        if clinical_marker != expected_clinical_marker:
            raise ValueError(
                "canonical onset clinical authorization marker disagrees with "
                "replayed clinical qualification"
            )

    temporal = receipt["temporal_evidence"]
    if bool(temporal["onset_evidence_authorized"]) is not clinical_authorized:
        raise ValueError(
            "canonical onset temporal permission disagrees with replayed clinical "
            "qualification"
        )
    if clinical_authorized:
        expected_reasons = []
    elif not response_authorized:
        expected_reasons = [ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE]
    else:
        expected_reasons = [ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE]
    if temporal["authorization_reason_codes"] != expected_reasons:
        raise ValueError(
            "canonical onset temporal denial reason disagrees with layered qualification"
        )


def _prepare_views(
    *,
    canonical: Mapping[str, Any],
    views: Sequence[DeterministicViewInput],
    final_interval: tuple[float, float],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None,
) -> list[_PreparedView]:
    if not views:
        raise ValueError("deterministic Findings require at least one signal view")
    prepared: list[_PreparedView] = []
    for index, item in enumerate(views):
        if not isinstance(item, DeterministicViewInput):
            raise TypeError(f"views[{index}] must be DeterministicViewInput")
        receipt = validate_signal_view_receipt(
            item.view_receipt,
            canonical,
            trusted_parent_views=trusted_parent_views,
        )
        _validate_canonical_edf_onset_admission(
            item=item,
            receipt=receipt,
            canonical=canonical,
            trusted_parent_views=trusted_parent_views,
        )
        if receipt["task_role"] not in _ALLOWED_VIEW_ROLES:
            raise ValueError(
                "detector/boundary/display views cannot support deterministic Findings"
            )
        unit_ids = tuple(str(row["unit_id"]) for row in receipt["output_units"])
        unit_types = tuple(str(row["unit_type"]) for row in receipt["output_units"])
        for unit_id in unit_ids:
            _identifier(unit_id, "view output unit_id")
        values = item.tensor.detach().cpu().to(torch.float32).contiguous()
        expected_shape = (
            len(unit_ids),
            int(receipt["tensor_layout"]["tensor_sample_count"]),
        )
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"view tensor shape {tuple(values.shape)} != receipt {expected_shape}"
            )
        actual_hash = deterministic_view_tensor_sha256(values, unit_ids=unit_ids)
        if actual_hash != receipt["processed_view_sha256"]:
            raise ValueError("processed view tensor hash does not match its receipt")
        selected_start, selected_stop = map(
            float, receipt["coordinates"]["selected_recording_seconds"]
        )
        if (
            final_interval[0] < selected_start - 1e-8
            or final_interval[1] > selected_stop + 1e-8
        ):
            raise ValueError("adaptive event interval lies outside a signal view")
        start_index = recording_seconds_to_view_tensor_index(
            receipt,
            recording_seconds=final_interval[0],
            rounding="ceil",
        )
        stop_index = recording_seconds_to_view_tensor_index(
            receipt,
            recording_seconds=final_interval[1],
            rounding="floor",
        )
        if stop_index <= start_index:
            raise ValueError("adaptive event contains no view samples")
        prepared.append(
            _PreparedView(
                receipt=receipt,
                tensor=values.numpy().astype(np.float64, copy=False),
                unit_ids=unit_ids,
                unit_types=unit_types,
                sampling_rate_hz=_view_sampling_rate(receipt),
                final_tensor_interval=(start_index, stop_index),
            )
        )
    prepared.sort(
        key=lambda item: (
            _ROLE_PRIORITY[str(item.receipt["task_role"])],
            str(item.receipt["view_id"]),
        )
    )
    primary_units = (prepared[0].unit_ids, prepared[0].unit_types)
    if any((item.unit_ids, item.unit_types) != primary_units for item in prepared[1:]):
        raise ValueError(
            "deterministic v1 requires identical output unit IDs/types across views"
        )
    if any(row["physical_unit"] != "V" for row in prepared[0].receipt["output_units"]):
        raise ValueError("primary Findings view must preserve physical volts")
    return prepared


def _family_eligible(unit: Mapping[str, Any], family: str) -> bool:
    return bool(
        next(row for row in unit["evidence_eligibility"] if row["family"] == family)[
            "eligible"
        ]
    )


def _invalid_samples(
    view: _PreparedView,
    *,
    family: str,
) -> np.ndarray:
    sample_count = view.tensor.shape[1]
    invalid = np.zeros((len(view.unit_ids), sample_count), dtype=bool)
    for unit_index, unit in enumerate(view.receipt["output_units"]):
        if not _family_eligible(unit, family):
            invalid[unit_index, :] = True
    for start, stop in view.receipt["masks"]["padding_intervals"]:
        invalid[:, int(start) : int(stop)] = True
    for start, stop in view.receipt["masks"]["edge_invalid_intervals"]:
        invalid[:, int(start) : int(stop)] = True
    unit_index = {unit_id: index for index, unit_id in enumerate(view.unit_ids)}
    for row in view.receipt["masks"]["quality_invalid_intervals"]:
        if family not in row["disabled_evidence_families"]:
            continue
        start, stop = (int(item) for item in row["tensor_sample_interval"])
        invalid[unit_index[str(row["unit_id"])], start:stop] = True
    return invalid


def _quality_invalid_samples(view: _PreparedView) -> np.ndarray:
    """Return acquisition/transform quality invalidity, not family capability.

    A view may legitimately be ineligible for morphology or HFO evidence while
    remaining perfectly usable for amplitude, spectrum and spatial change.
    Treating one unsupported family as global sample corruption would erase
    every deterministic feature window.
    """

    invalid = np.zeros((len(view.unit_ids), view.tensor.shape[1]), dtype=bool)
    for unit_index, unit in enumerate(view.receipt["output_units"]):
        if not unit["observed"] or unit["imputed"]:
            invalid[unit_index, :] = True
    for start, stop in view.receipt["masks"]["padding_intervals"]:
        invalid[:, int(start) : int(stop)] = True
    for start, stop in view.receipt["masks"]["edge_invalid_intervals"]:
        invalid[:, int(start) : int(stop)] = True
    unit_index = {unit_id: index for index, unit_id in enumerate(view.unit_ids)}
    for row in view.receipt["masks"]["quality_invalid_intervals"]:
        start, stop = (int(item) for item in row["tensor_sample_interval"])
        invalid[unit_index[str(row["unit_id"])], start:stop] = True
    return invalid


def _feature_grid(
    view: _PreparedView,
    *,
    policy: DeterministicEventFindingsPolicy,
) -> _FeatureGrid:
    window_samples = max(4, int(round(policy.window_seconds * view.sampling_rate_hz)))
    step_samples = max(1, int(round(policy.step_seconds * view.sampling_rate_hz)))
    start, stop = view.final_tensor_interval
    if stop - start < window_samples:
        raise ValueError("adaptive event is shorter than one Findings window")
    tensor_starts = np.arange(
        start,
        stop - window_samples + 1,
        step_samples,
        dtype=np.int64,
    )
    tensor_stops = tensor_starts + window_samples
    recording_starts = np.asarray(
        [
            view_tensor_index_to_recording_seconds(
                view.receipt, tensor_sample_index=int(index)
            )
            for index in tensor_starts
        ],
        dtype=np.float64,
    )
    recording_stops = np.asarray(
        [
            view_tensor_index_to_recording_seconds(
                view.receipt, tensor_sample_index=int(index)
            )
            for index in tensor_stops
        ],
        dtype=np.float64,
    )
    shape = (tensor_starts.size, len(view.unit_ids))
    rms = np.full(shape, np.nan, dtype=np.float64)
    p2p = np.full(shape, np.nan, dtype=np.float64)
    line = np.full(shape, np.nan, dtype=np.float64)
    dominant = np.full(shape, np.nan, dtype=np.float64)
    concentration = np.full(shape, np.nan, dtype=np.float64)
    entropy = np.full(shape, np.nan, dtype=np.float64)
    rhythmicity = np.full(shape, np.nan, dtype=np.float64)
    band_ratio = np.full((shape[0], shape[1], len(_BANDS)), np.nan, dtype=np.float64)

    amplitude_invalid = _invalid_samples(view, family="amplitude")
    spectral_invalid = _invalid_samples(view, family="spectral")
    spatial_invalid = _invalid_samples(view, family="spatial_field")
    relation_invalid = amplitude_invalid | spectral_invalid | spatial_invalid
    amplitude_valid = np.zeros(shape, dtype=bool)
    spectral_valid = np.zeros(shape, dtype=bool)
    spatial_valid = np.zeros(shape, dtype=bool)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / view.sampling_rate_hz)
    taper = np.hanning(window_samples)

    for window_index, (window_start, window_stop) in enumerate(
        zip(tensor_starts, tensor_stops)
    ):
        segment_volts = view.tensor[:, int(window_start) : int(window_stop)]
        centered_volts = segment_volts - np.median(segment_volts, axis=1, keepdims=True)
        centered_uv = centered_volts * 1e6
        for unit_index, unit in enumerate(view.receipt["output_units"]):
            amplitude_ok = not amplitude_invalid[
                unit_index, int(window_start) : int(window_stop)
            ].any()
            spectral_ok = not spectral_invalid[
                unit_index, int(window_start) : int(window_stop)
            ].any()
            spatial_ok = not spatial_invalid[
                unit_index, int(window_start) : int(window_stop)
            ].any()
            clean_for_relation = not relation_invalid[
                unit_index, int(window_start) : int(window_stop)
            ].any()
            amplitude_valid[window_index, unit_index] = amplitude_ok
            spectral_valid[window_index, unit_index] = spectral_ok
            spatial_valid[window_index, unit_index] = spatial_ok and clean_for_relation
            if amplitude_ok:
                values = centered_uv[unit_index]
                rms[window_index, unit_index] = math.sqrt(float(np.mean(values**2)))
                p2p[window_index, unit_index] = float(np.ptp(values))
                line[window_index, unit_index] = float(np.mean(np.abs(np.diff(values))))
            if not spectral_ok:
                continue
            low, high = (float(item) for item in unit["effective_bandwidth_hz"])
            frequency_mask = (frequencies >= max(0.5, low)) & (
                frequencies <= min(45.0, high)
            )
            if np.count_nonzero(frequency_mask) < 3:
                continue
            spectrum = np.abs(np.fft.rfft(centered_uv[unit_index] * taper)) ** 2
            total = float(np.sum(spectrum[frequency_mask])) + 1e-12
            local_frequencies = frequencies[frequency_mask]
            local_power = spectrum[frequency_mask]
            peak_index = int(np.argmax(local_power))
            peak_frequency = float(local_frequencies[peak_index])
            dominant[window_index, unit_index] = peak_frequency
            concentration[window_index, unit_index] = float(
                local_power[peak_index] / total
            )
            probabilities = local_power / total
            entropy[window_index, unit_index] = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
                / math.log(max(2, probabilities.size))
            )
            lag = int(round(view.sampling_rate_hz / max(peak_frequency, 0.5)))
            if 1 <= lag < window_samples // 2:
                left = centered_uv[unit_index, :-lag]
                right = centered_uv[unit_index, lag:]
                denominator = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
                rhythmicity[window_index, unit_index] = (
                    float(np.sum(left * right) / denominator)
                    if denominator > 1e-12
                    else 0.0
                )
            else:
                rhythmicity[window_index, unit_index] = 0.0
            for band_index, (_, band_low, band_high) in enumerate(_BANDS):
                band_mask = (frequencies >= max(band_low, low)) & (
                    frequencies < min(band_high, high + 1e-12)
                )
                band_ratio[window_index, unit_index, band_index] = (
                    float(np.sum(spectrum[band_mask]) / total)
                    if np.any(band_mask)
                    else 0.0
                )

    return _FeatureGrid(
        tensor_starts=tensor_starts,
        tensor_stops=tensor_stops,
        recording_starts=recording_starts,
        recording_stops=recording_stops,
        rms_uv=rms,
        peak_to_peak_uv=p2p,
        line_length_uv=line,
        dominant_frequency_hz=dominant,
        spectral_concentration=concentration,
        spectral_entropy=entropy,
        rhythmicity_index=rhythmicity,
        band_ratio=band_ratio,
        amplitude_valid=amplitude_valid,
        spectral_valid=spectral_valid,
        spatial_valid=spatial_valid,
    )


def _finite_median_axis0(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1:], np.nan, dtype=np.float64)
    for trailing_index in np.ndindex(values.shape[1:]):
        column = values[(slice(None),) + trailing_index]
        finite = column[np.isfinite(column)]
        if finite.size:
            result[trailing_index] = float(np.median(finite))
    return result


def _robust_z(values: np.ndarray, baseline: np.ndarray, floor: float) -> np.ndarray:
    center = _finite_median_axis0(baseline)
    mad = _finite_median_axis0(np.abs(baseline - center))
    scale = np.maximum(1.4826 * mad, floor)
    return np.abs(values - center) / scale


def _change_scores(
    grid: _FeatureGrid,
    *,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    shape = grid.rms_uv.shape
    result = np.full(shape, np.nan, dtype=np.float64)
    if np.count_nonzero(baseline_mask) < 2:
        return result
    log_rms = np.log(grid.rms_uv + 1e-6)
    log_line = np.log(grid.line_length_uv + 1e-6)
    rms_z = _robust_z(log_rms, log_rms[baseline_mask], 0.15)
    line_z = _robust_z(log_line, log_line[baseline_mask], 0.15)
    per_band_z = _robust_z(
        grid.band_ratio,
        grid.band_ratio[baseline_mask],
        0.04,
    )
    band_finite = np.isfinite(per_band_z)
    band_z = np.max(np.where(band_finite, per_band_z, -np.inf), axis=2)
    band_z[~np.any(band_finite, axis=2)] = np.nan
    entropy_z = _robust_z(
        grid.spectral_entropy,
        grid.spectral_entropy[baseline_mask],
        0.08,
    )
    components = np.stack((rms_z, line_z, band_z, entropy_z), axis=2)
    ranked_components = np.sort(
        np.where(np.isfinite(components), components, -np.inf),
        axis=2,
    )
    top_components = ranked_components[:, :, -2:]
    top_finite = np.isfinite(top_components)
    top_count = np.sum(top_finite, axis=2)
    top_sum = np.sum(np.where(top_finite, top_components, 0.0), axis=2)
    result = np.divide(
        top_sum,
        top_count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=top_count > 0,
    )
    result[~(grid.amplitude_valid & grid.spectral_valid & grid.spatial_valid)] = np.nan
    return result


def _first_sustained_changes(
    scores: np.ndarray,
    grid: _FeatureGrid,
    *,
    candidate_mask: np.ndarray,
    policy: DeterministicEventFindingsPolicy,
) -> dict[int, int]:
    first: dict[int, int] = {}
    required = int(policy.sustained_change_windows)
    for unit_index in range(scores.shape[1]):
        active = (
            candidate_mask
            & np.isfinite(scores[:, unit_index])
            & (scores[:, unit_index] >= policy.change_score_threshold)
        )
        indices = np.flatnonzero(active)
        for index in indices:
            block = np.arange(index, index + required)
            if block[-1] >= active.size or not np.all(active[block]):
                continue
            first[unit_index] = int(index)
            break
    return first


def _electrode_laterality(name: str) -> str:
    normalized = name.upper()
    if normalized in _LEFT_ELECTRODES:
        return "left"
    if normalized in _RIGHT_ELECTRODES:
        return "right"
    if normalized in _MIDLINE_ELECTRODES:
        return "midline"
    return "indeterminate"


def _lead_endpoints(unit_id: str) -> tuple[str, str]:
    parts = unit_id.upper().split("-")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"lead unit {unit_id!r} must encode exactly two electrode endpoints"
        )
    return parts[0], parts[1]


def _unit_laterality(unit_id: str, unit_type: str) -> str:
    if unit_type == "electrode":
        return _electrode_laterality(unit_id)
    left, right = _lead_endpoints(unit_id)
    values = {_electrode_laterality(left), _electrode_laterality(right)}
    values.discard("indeterminate")
    if len(values) == 1:
        return next(iter(values))
    if values == {"left", "right"}:
        return "bilateral"
    return "indeterminate"


def _electrode_region(electrode: str) -> str:
    name = electrode.upper()
    if name in {"F7", "T7", "P7"}:
        return "temporal"
    if name in {"F8", "T8", "P8"}:
        return "temporal"
    if name.startswith("FP") or name.startswith("F"):
        return "frontal"
    if name.startswith("C"):
        return "central"
    if name.startswith("P"):
        return "parietal"
    if name.startswith("O"):
        return "occipital"
    return "other"


def _unit_region(unit_id: str, unit_type: str, laterality: str) -> str:
    electrodes = (
        (unit_id.upper(),) if unit_type == "electrode" else _lead_endpoints(unit_id)
    )
    regions = [_electrode_region(item) for item in electrodes]
    base = regions[0] if len(set(regions)) == 1 else "multiregional"
    prefix = laterality if laterality in {"left", "right", "midline"} else "bilateral"
    return f"{prefix}_{base}"


def _analysis_reference(view: _PreparedView) -> str:
    raw = str(view.receipt["transform_spec"]["reference"]["reference_type"]).lower()
    if all(unit_type == "lead" for unit_type in view.unit_types) or "bipolar" in raw:
        return "bipolar"
    if "laplac" in raw:
        return "laplacian"
    if "common" in raw or raw == "car" or "average" in raw:
        return "common_average"
    if "linked" in raw or "ear" in raw:
        return "linked_ears"
    return "other"


def _montage(
    prepared: Sequence[_PreparedView],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    primary = prepared[0]
    input_units: list[dict[str, Any]] = []
    electrode_ids: set[str] = set()
    lead_definitions: list[dict[str, str]] = []
    laterality_by_unit: dict[str, str] = {}
    region_by_unit: dict[str, str] = {}
    for unit_id, unit_type, output in zip(
        primary.unit_ids,
        primary.unit_types,
        primary.receipt["output_units"],
    ):
        if unit_type not in {"lead", "electrode"}:
            raise ValueError(
                "deterministic v1 cannot map virtual units into event montage"
            )
        laterality = _unit_laterality(unit_id, unit_type)
        region = _unit_region(unit_id, unit_type, laterality)
        laterality_by_unit[unit_id] = laterality
        region_by_unit[unit_id] = region
        if unit_type == "lead":
            anode, cathode = _lead_endpoints(unit_id)
            electrode_ids.update((anode, cathode))
            lead_definitions.append(
                {"lead_id": unit_id, "anode": anode, "cathode": cathode}
            )
            event_unit_type = "bipolar_lead"
        else:
            electrode_ids.add(unit_id)
            event_unit_type = "electrode"
        input_units.append(
            {
                "unit_id": unit_id,
                "unit_type": event_unit_type,
                "canonical_name": unit_id,
                "source_name": f"{primary.receipt['view_id']}:{unit_id}",
                "available": bool(output["observed"]) and not bool(output["imputed"]),
                "region": region,
                "laterality": laterality,
            }
        )
    perturbations = []
    for view in prepared:
        reference = _analysis_reference(view)
        if (
            reference in {"common_average", "bipolar", "laplacian"}
            and reference not in perturbations
        ):
            perturbations.append(reference)
    return (
        {
            "analysis_reference": _analysis_reference(primary),
            "input_units": input_units,
            "electrode_ids": sorted(electrode_ids),
            "lead_definitions": lead_definitions,
            "reference_perturbations_evaluated": perturbations,
        },
        laterality_by_unit,
        region_by_unit,
    )


def _phase_membership(
    interval: tuple[float, float],
    segment_spans: Mapping[str, tuple[float, float] | None],
) -> dict[str, float]:
    overlaps: dict[str, float] = {}
    for name, span in segment_spans.items():
        overlaps[name] = (
            0.0
            if span is None
            else max(0.0, min(interval[1], span[1]) - max(interval[0], span[0]))
        )
    total = sum(overlaps.values())
    if total <= 1e-9:
        # The adaptive partition is descriptive.  A narrow boundary atom that
        # only touches the S1 edge is serialized into the legacy v1 S1 slot;
        # this does not qualify it as an ictal event.
        return {
            "baseline": 0.0,
            "early_ictal": 1.0,
            "evolved_ictal": 0.0,
            "recovery": 0.0,
        }
    return {
        wire_name: overlaps[segment_name] / total
        for segment_name, wire_name in _SEGMENT_TO_V1_PHASE.items()
    }


def _window_contract(
    adaptive_window: Mapping[str, Any],
    *,
    temporal_resolution: float,
) -> tuple[
    dict[str, Any], dict[str, tuple[float, float] | None], tuple[float, float] | None
]:
    final = tuple(
        float(item) for item in adaptive_window["analysis_interval_recording_seconds"]
    )
    core = adaptive_window["analysis_core_recording_seconds"]
    baseline = adaptive_window["baseline_context_recording_seconds"]
    recovery = adaptive_window["recovery_context_recording_seconds"]
    onset_interval: dict[str, Any]
    onset_bounds: tuple[float, float] | None
    if core is None or not adaptive_window["eligibility"]["onset_localization"]:
        onset_interval = {
            "interval": None,
            "status": (
                "censored" if adaptive_window["censoring"]["left"] else "not_observed"
            ),
        }
        onset_bounds = None
        early_span = None
        evolution_span = final
    else:
        onset = float(core[0])
        lower = max(final[0], onset - temporal_resolution / 2.0)
        upper = min(final[1], onset + temporal_resolution / 2.0)
        onset_bounds = (lower, upper)
        onset_interval = {
            "interval": {
                "lower": lower,
                "median": onset,
                "upper": upper,
                "resolution_seconds": temporal_resolution,
                "calibration_status": "uncalibrated",
            },
            "status": "interval_estimate",
        }
        event_stop = (
            float(core[1]) if len(core) == 2 and core[1] is not None else final[1]
        )
        early_stop = min(
            event_stop, onset + max(2.0, min(10.0, 0.25 * (event_stop - onset)))
        )
        if early_stop <= onset + 1e-9:
            early_stop = min(final[1], onset + temporal_resolution)
        early_span = (onset, early_stop) if early_stop > onset else None
        evolution_span = (
            (early_stop, event_stop) if event_stop > early_stop + 1e-9 else None
        )
    baseline_span = tuple(map(float, baseline)) if baseline is not None else None
    recovery_span = tuple(map(float, recovery)) if recovery is not None else None
    segment_spans = {
        "S0": baseline_span,
        "S1": early_span,
        "S2": evolution_span,
        "S3": recovery_span,
    }

    def phase(
        span: tuple[float, float] | None, status: str = "observed"
    ) -> dict[str, Any]:
        return {
            "interval": (
                None
                if span is None
                else {
                    "start": span[0],
                    "stop": span[1],
                    "resolution_seconds": temporal_resolution,
                }
            ),
            "status": "not_observed" if span is None else status,
        }

    offset: dict[str, Any]
    if core is not None and len(core) == 2 and core[1] is not None:
        value = float(core[1])
        offset = {
            "interval": {
                "lower": value,
                "median": value,
                "upper": value,
                "resolution_seconds": temporal_resolution,
                "calibration_status": "uncalibrated",
            },
            "status": "uncertain",
        }
    else:
        offset = {"interval": None, "status": "not_observed"}
    right_censored = (
        bool(adaptive_window["censoring"]["right"])
        or offset["interval"] is None
        or recovery_span is None
    )
    contract = {
        "baseline": phase(baseline_span),
        "onset_interval": onset_interval,
        "early_ictal": phase(early_span, "uncertain"),
        "evolution": phase(evolution_span, "uncertain"),
        "offset_interval": offset,
        "recovery": phase(recovery_span, "uncertain"),
        "left_censored": bool(adaptive_window["censoring"]["left"]),
        "right_censored": right_censored,
        "search_cap_censored": bool(
            adaptive_window["censoring"]["left"]
            or adaptive_window["censoring"]["right"]
        ),
        "merge_split_status": "single_event",
    }
    if contract["search_cap_censored"] and not (
        contract["left_censored"] or contract["right_censored"]
    ):
        raise AssertionError("adaptive censoring projection drifted")
    return contract, segment_spans, onset_bounds


def _binding_bandwidth(
    view: _PreparedView,
    unit_ids: Sequence[str],
) -> list[float]:
    catalog = {str(row["unit_id"]): row for row in view.receipt["output_units"]}
    rows = [catalog[item]["effective_bandwidth_hz"] for item in unit_ids]
    lower = max(float(row[0]) for row in rows)
    upper = min(float(row[1]) for row in rows)
    if upper <= lower:
        raise ValueError("measurement units have no common effective bandwidth")
    return [lower, upper]


def _source_binding(
    *,
    view: _PreparedView,
    unit_ids: Sequence[str],
    recording_interval: tuple[float, float],
    evidence_family: str,
    background_reference_ids: Sequence[str],
    policy: DeterministicEventFindingsPolicy,
) -> dict[str, Any]:
    start = recording_seconds_to_view_tensor_index(
        view.receipt, recording_seconds=recording_interval[0], rounding="ceil"
    )
    stop = recording_seconds_to_view_tensor_index(
        view.receipt, recording_seconds=recording_interval[1], rounding="floor"
    )
    if stop <= start:
        raise ValueError("measurement interval has no complete view samples")
    transform = view.receipt["transform_spec"]
    return {
        "source_view_id": _identifier(view.receipt["view_id"], "source_view_id"),
        "view_receipt_id": _identifier(
            view.receipt["view_receipt_id"], "view_receipt_id"
        ),
        "view_receipt_sha256": view.receipt["receipt_sha256"],
        "transform_spec_sha256": transform["transform_spec_sha256"],
        "processed_view_sha256": view.receipt["processed_view_sha256"],
        "source_unit_ids": list(unit_ids),
        "recording_interval": [recording_interval[0], recording_interval[1]],
        "tensor_sample_interval": [start, stop],
        "effective_bandwidth_hz": _binding_bandwidth(view, unit_ids),
        "reference_type": str(transform["reference"]["reference_type"]),
        "evidence_family": evidence_family,
        "quality_mask_sha256": view.receipt["masks"]["mask_sha256"],
        "background_reference_ids": list(background_reference_ids),
        "method_id": DETERMINISTIC_EVENT_FINDINGS_METHOD_ID,
        "policy_sha256": policy.sha256,
    }


def _measurement(
    *,
    name: str,
    value: float,
    unit: str,
    view: _PreparedView,
    unit_ids: Sequence[str],
    interval: tuple[float, float],
    evidence_family: str,
    background_reference_ids: Sequence[str],
    policy: DeterministicEventFindingsPolicy,
    baseline_delta: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "value": float(value),
        "unit": unit,
        "producer_type": "deterministic_signal_measurement",
        "source_binding": _source_binding(
            view=view,
            unit_ids=unit_ids,
            recording_interval=interval,
            evidence_family=evidence_family,
            background_reference_ids=background_reference_ids,
            policy=policy,
        ),
    }
    if baseline_delta is not None:
        row["baseline_delta"] = float(baseline_delta)
    return row


def _uncertainty(
    *,
    adaptive_window: Mapping[str, Any],
    usable_fraction: float,
    background_available: bool,
) -> dict[str, Any]:
    return {
        "boundary": 0.65 if adaptive_window["censoring"]["left"] else 0.30,
        "quality": max(0.0, min(1.0, 1.0 - usable_fraction)),
        "background": 0.15 if background_available else 1.0,
        "model": 0.0,
        # Merely supplying more views is not evidence of cross-reference
        # stability.  V1 emits reference-specific measurements only; a later
        # qualified module must compare field geometry/polarity/ranks before
        # this component may be reduced.
        "reference_stability": 1.0,
        "semantics": "componentwise_descriptive_not_individual_correctness_probability",
    }


def _artifact_type(reason_codes: Sequence[str]) -> str:
    text = " ".join(str(item).lower() for item in reason_codes)
    for name in ("flat", "clipping", "step", "line_noise"):
        if name in text:
            return name
    return "other"


def _quality_contract(
    primary: _PreparedView,
    *,
    final_interval: tuple[float, float],
    has_baseline: bool,
    has_onset: bool,
) -> tuple[dict[str, Any], float]:
    start, stop = primary.final_tensor_interval
    total = stop - start
    invalid = _quality_invalid_samples(primary)
    per_unit: list[dict[str, Any]] = []
    fractions: list[float] = []
    for index, (unit_id, output) in enumerate(
        zip(primary.unit_ids, primary.receipt["output_units"])
    ):
        if not output["observed"] or output["imputed"]:
            fraction = 0.0
        else:
            fraction = 1.0 - float(np.mean(invalid[index, start:stop]))
        fraction = max(0.0, min(1.0, fraction))
        fractions.append(fraction)
        per_unit.append(
            {
                "unit_id": unit_id,
                "usable_fraction": fraction,
                "status": (
                    "usable"
                    if fraction >= 1.0 - 1e-9
                    else "limited"
                    if fraction > 0.0
                    else "unusable"
                ),
            }
        )
    usable_fraction = float(sum(fractions) / len(fractions))
    artifacts: list[dict[str, Any]] = []
    for row in primary.receipt["masks"]["quality_invalid_intervals"]:
        tensor_start, tensor_stop = (
            int(item) for item in row["tensor_sample_interval"]
        )
        recording_start = view_tensor_index_to_recording_seconds(
            primary.receipt, tensor_sample_index=tensor_start
        )
        recording_stop = view_tensor_index_to_recording_seconds(
            primary.receipt, tensor_sample_index=tensor_stop
        )
        clipped = (
            max(final_interval[0], recording_start),
            min(final_interval[1], recording_stop),
        )
        if clipped[1] <= clipped[0]:
            continue
        artifacts.append(
            {
                "interval": [clipped[0], clipped[1]],
                "artifact_type": _artifact_type(row["reason_codes"]),
                "affected_unit_ids": [str(row["unit_id"])],
                "assertion_level": "measured",
            }
        )
    for tensor_start, tensor_stop in primary.receipt["masks"]["edge_invalid_intervals"]:
        recording_start = view_tensor_index_to_recording_seconds(
            primary.receipt, tensor_sample_index=int(tensor_start)
        )
        recording_stop = view_tensor_index_to_recording_seconds(
            primary.receipt, tensor_sample_index=int(tensor_stop)
        )
        clipped = (
            max(final_interval[0], recording_start),
            min(final_interval[1], recording_stop),
        )
        if clipped[1] > clipped[0]:
            artifacts.append(
                {
                    "interval": [clipped[0], clipped[1]],
                    "artifact_type": "other",
                    "affected_unit_ids": list(primary.unit_ids),
                    "assertion_level": "measured",
                }
            )
    spectral_status = "available" if has_baseline else "limited"
    spectral_reasons = [] if has_baseline else ["background_unavailable"]
    spatial_available = any(
        _family_eligible(row, "spatial_field")
        for row in primary.receipt["output_units"]
    )
    feature_availability = [
        {
            "family": "spectral",
            "status": spectral_status,
            "reason_codes": spectral_reasons,
        },
        {
            "family": "rhythm",
            "status": spectral_status,
            "reason_codes": spectral_reasons,
        },
        {
            "family": "morphology",
            "status": "not_evaluable",
            "reason_codes": ["deterministic_v1_morphology_not_qualified"],
        },
        {
            "family": "evolution",
            "status": spectral_status,
            "reason_codes": spectral_reasons,
        },
        {
            "family": "spatial_field",
            "status": "available" if spatial_available else "not_evaluable",
            "reason_codes": [] if spatial_available else ["spatial_view_ineligible"],
        },
        {
            "family": "recruitment",
            "status": "available" if has_onset and spatial_available else "limited",
            "reason_codes": []
            if has_onset and spatial_available
            else ["onset_not_localizable"],
        },
        {
            "family": "termination_recovery",
            "status": "limited",
            "reason_codes": ["return_candidate_only_not_clinical_termination"],
        },
        {
            "family": "high_frequency",
            "status": "not_evaluable",
            "reason_codes": ["deterministic_v1_hfo_not_qualified"],
        },
    ]
    return (
        {
            "usable_fraction": usable_fraction,
            "per_unit": per_unit,
            "artifact_intervals": artifacts,
            "feature_availability": feature_availability,
        },
        usable_fraction,
    )


def produce_deterministic_event_eeg_findings(
    *,
    event_id: str,
    adaptive_search_receipt: object,
    adaptive_window_receipt: object,
    canonical_receipt: object,
    views: Sequence[DeterministicViewInput],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
    policy: DeterministicEventFindingsPolicy = DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY,
) -> dict[str, Any]:
    """Produce one replayable ``event_eeg_findings_v1`` evidence bundle."""

    _identifier(event_id, "event_id")
    if not isinstance(policy, DeterministicEventFindingsPolicy):
        raise TypeError("policy must be DeterministicEventFindingsPolicy")
    search = validate_adaptive_search_receipt(adaptive_search_receipt)
    adaptive_window = validate_adaptive_event_analysis_window(adaptive_window_receipt)
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    if search["canonical_signal_binding"] is None:
        raise ValueError(
            "adaptive search lacks canonical signal binding; deterministic "
            "Findings cannot prove same-record EEG identity"
        )
    validate_canonical_adaptive_binding_against_receipt(
        search["canonical_signal_binding"],
        canonical,
    )
    if adaptive_window["source_search_receipt_id"] != search["search_receipt_id"]:
        raise ValueError("adaptive window belongs to a different search receipt")
    if adaptive_window["source_search_receipt_sha256"] != _canonical_sha256(search):
        raise ValueError("adaptive window source-search hash drifted")
    if search["recording_duration_seconds"] is not None and not math.isclose(
        float(search["recording_duration_seconds"]),
        float(canonical["recording_duration_seconds"]),
        abs_tol=1e-6,
    ):
        raise ValueError("adaptive search and canonical recording durations differ")
    if not adaptive_window["eligibility"]["signal_findings"]:
        raise ValueError(
            "adaptive window is explicitly ineligible for signal Findings; "
            "do not invent an event bundle"
        )
    final_interval = tuple(
        float(item) for item in adaptive_window["analysis_interval_recording_seconds"]
    )
    prepared = _prepare_views(
        canonical=canonical,
        views=views,
        final_interval=final_interval,
        trusted_parent_views=trusted_parent_views,
    )
    primary = prepared[0]
    temporal_resolution = max(
        policy.step_seconds,
        1.0 / primary.sampling_rate_hz,
    )
    window_core, segment_spans, onset_bounds = _window_contract(
        adaptive_window,
        temporal_resolution=temporal_resolution,
    )
    window_core["search_interval"] = [
        float(search["envelope_interval_recording_seconds"][0]),
        float(search["envelope_interval_recording_seconds"][1]),
    ]
    window_core["final_interval"] = [final_interval[0], final_interval[1]]
    montage, laterality_by_unit, region_by_unit = _montage(prepared)
    grid = _feature_grid(primary, policy=policy)
    baseline_span = segment_spans["S0"]
    baseline_mask = (
        np.zeros(grid.recording_starts.shape, dtype=bool)
        if baseline_span is None
        else (
            (grid.recording_starts >= baseline_span[0] - 1e-9)
            & (grid.recording_stops <= baseline_span[1] + 1e-9)
        )
    )
    has_baseline = (
        int(np.count_nonzero(baseline_mask)) >= policy.minimum_baseline_windows
    )
    scores = (
        _change_scores(grid, baseline_mask=baseline_mask)
        if has_baseline
        else np.full(grid.rms_uv.shape, np.nan, dtype=np.float64)
    )
    candidate_start = onset_bounds[0] if onset_bounds is not None else final_interval[0]
    candidate_mask = grid.recording_stops > candidate_start + 1e-9
    first_changes = _first_sustained_changes(
        scores,
        grid,
        candidate_mask=candidate_mask,
        policy=policy,
    )
    onset_units: list[int] = []
    later_units: list[int] = []
    if onset_bounds is not None and first_changes:
        earliest = min(grid.recording_starts[index] for index in first_changes.values())
        # A single referential electrode maximum is reference dependent and
        # is not a qualified scalp field.  Until explicit multi-reference
        # geometry/polarity/rank stability is implemented, only directly
        # observed bipolar leads may enter onset Top-k.  Electrode views remain
        # useful for descriptive, reference-specific measurements.
        near = [
            unit_index
            for unit_index, window_index in first_changes.items()
            if primary.unit_types[unit_index] == "lead"
            if grid.recording_starts[window_index]
            <= earliest + policy.near_synchronous_seconds + 1e-9
            and grid.recording_starts[window_index]
            <= onset_bounds[1] + policy.onset_tolerance_seconds
            and grid.recording_stops[window_index] >= onset_bounds[0] - 1e-9
        ]
        onset_units = sorted(
            near,
            key=lambda index: (
                -float(scores[first_changes[index], index]),
                primary.unit_ids[index],
            ),
        )
        if onset_units:
            source_time = min(
                grid.recording_starts[first_changes[index]] for index in onset_units
            )
            later_units = sorted(
                [
                    index
                    for index, window_index in first_changes.items()
                    if index not in onset_units
                    and grid.recording_starts[window_index] - source_time
                    >= policy.later_involvement_delay_seconds - 1e-9
                ],
                key=lambda index: (
                    grid.recording_starts[first_changes[index]],
                    primary.unit_ids[index],
                ),
            )
    background_available = (
        has_baseline and baseline_span is not None and onset_bounds is not None
    )
    quality, usable_fraction = _quality_contract(
        primary,
        final_interval=final_interval,
        has_baseline=background_available,
        has_onset=bool(onset_units),
    )
    background_bank_id = (
        f"BGBANK-{_canonical_sha256([event_id, baseline_span, canonical['receipt_sha256']])[:20]}"
        if background_available
        else None
    )
    selection_receipt_id = (
        f"BGSELECT-{_canonical_sha256([event_id, search['search_receipt_id']])[:20]}"
        if background_available
        else None
    )
    background_reference_ids = (
        [background_bank_id] if background_bank_id is not None else []
    )
    context = {
        "queried_intervals": [
            [
                float(search["envelope_interval_recording_seconds"][0]),
                float(search["envelope_interval_recording_seconds"][1]),
            ]
        ],
        "local_background_intervals": (
            [[baseline_span[0], baseline_span[1]]]
            if background_available and baseline_span is not None
            else []
        ),
        "distant_background_intervals": [],
        "background_status": "available" if background_available else "unavailable",
        "background_bank_id": background_bank_id,
        "selection_receipt_id": selection_receipt_id,
        "selection_scope": "eeg_detector_quality_only",
        "contamination_risk": 0.15 if background_available else 1.0,
    }
    waveform_rows: list[dict[str, Any]] = []
    waveform_ids: dict[tuple[str, tuple[str, ...], float, float], str] = {}

    def add_waveform(
        view: _PreparedView,
        unit_ids: Sequence[str],
        interval: tuple[float, float],
    ) -> str:
        key = (
            str(view.receipt["view_receipt_id"]),
            tuple(unit_ids),
            float(interval[0]),
            float(interval[1]),
        )
        if key in waveform_ids:
            return waveform_ids[key]
        identifier = "WAVE-" + _canonical_sha256(key)[:20]
        waveform_ids[key] = identifier
        waveform_rows.append(
            {
                "waveform_evidence_id": identifier,
                "interval": [interval[0], interval[1]],
                "unit_ids": list(unit_ids),
                "render_policy": "CANONICAL-VIEW-REPLAY-V1",
                "signal_sha256": canonical["source_signal_sha256"],
            }
        )
        return identifier

    uncertainty = _uncertainty(
        adaptive_window=adaptive_window,
        usable_fraction=usable_fraction,
        background_available=background_available,
    )
    findings: list[dict[str, Any]] = []
    quality_wave = add_waveform(primary, primary.unit_ids, final_interval)
    findings.append(
        {
            "evidence_id": "E-QUALITY-COVERAGE",
            "family": "quality",
            "term": "deterministic_signal_usable_fraction",
            "evidence_role": "context_only",
            "assertion_level": "measured",
            "status": "present",
            "phase_membership": _phase_membership(final_interval, segment_spans),
            "time_interval": {
                "start": final_interval[0],
                "stop": final_interval[1],
                "resolution_seconds": temporal_resolution,
            },
            "spatial_support": [],
            "measurements": [
                _measurement(
                    name="usable_sample_fraction",
                    value=usable_fraction,
                    unit="ratio",
                    view=primary,
                    unit_ids=primary.unit_ids,
                    interval=final_interval,
                    evidence_family="waveform",
                    background_reference_ids=[],
                    policy=policy,
                )
            ],
            "uncertainty": deepcopy(uncertainty),
            "qualification_receipt_id": None,
            "term_decision_receipt_id": None,
            "waveform_evidence_ids": [quality_wave],
        }
    )

    descriptive_indices = list(onset_units[: policy.maximum_descriptive_units])
    if not descriptive_indices:
        finite_rms = _finite_median_axis0(grid.rms_uv)
        descriptive_indices = [
            int(index)
            for index in np.argsort(np.nan_to_num(finite_rms, nan=-np.inf))[::-1]
            if np.isfinite(finite_rms[index])
        ][: policy.maximum_descriptive_units]
    descriptive_start = (
        onset_bounds[0] if onset_bounds is not None else final_interval[0]
    )
    descriptive_interval = (descriptive_start, final_interval[1])
    descriptive_windows = (grid.recording_starts >= descriptive_interval[0] - 1e-9) & (
        grid.recording_stops <= descriptive_interval[1] + 1e-9
    )
    for ordinal, unit_index in enumerate(descriptive_indices, start=1):
        valid = descriptive_windows & grid.spectral_valid[:, unit_index]
        amplitude_ok = descriptive_windows & grid.amplitude_valid[:, unit_index]
        if not np.any(valid) or not np.any(amplitude_ok):
            continue
        unit_id = primary.unit_ids[unit_index]
        interval = descriptive_interval
        support_type = (
            "lead" if primary.unit_types[unit_index] == "lead" else "electrode"
        )
        measurements = [
            _measurement(
                name=f"dominant_frequency_hz_{ordinal}",
                value=float(
                    np.nanmedian(grid.dominant_frequency_hz[valid, unit_index])
                ),
                unit="Hz",
                view=primary,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="spectral",
                background_reference_ids=background_reference_ids,
                policy=policy,
            ),
            _measurement(
                name=f"spectral_concentration_{ordinal}",
                value=float(
                    np.nanmedian(grid.spectral_concentration[valid, unit_index])
                ),
                unit="ratio",
                view=primary,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="spectral",
                background_reference_ids=background_reference_ids,
                policy=policy,
            ),
            _measurement(
                name=f"rhythmicity_autocorrelation_{ordinal}",
                value=float(np.nanmedian(grid.rhythmicity_index[valid, unit_index])),
                unit="unitless",
                view=primary,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="spectral",
                background_reference_ids=background_reference_ids,
                policy=policy,
            ),
            _measurement(
                name=f"rms_amplitude_uv_{ordinal}",
                value=float(np.nanmedian(grid.rms_uv[amplitude_ok, unit_index])),
                unit="uV",
                view=primary,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="amplitude",
                background_reference_ids=background_reference_ids,
                policy=policy,
            ),
            _measurement(
                name=f"peak_to_peak_amplitude_uv_{ordinal}",
                value=float(
                    np.nanmedian(grid.peak_to_peak_uv[amplitude_ok, unit_index])
                ),
                unit="uV",
                view=primary,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="amplitude",
                background_reference_ids=background_reference_ids,
                policy=policy,
            ),
        ]
        wave = add_waveform(primary, [unit_id], interval)
        findings.append(
            {
                "evidence_id": f"E-DESCRIPTIVE-{ordinal}",
                "family": "spectral",
                "term": "deterministic_frequency_rhythm_amplitude_profile",
                "evidence_role": "context_only",
                "assertion_level": "measured",
                "status": "present",
                "phase_membership": _phase_membership(interval, segment_spans),
                "time_interval": {
                    "start": interval[0],
                    "stop": interval[1],
                    "resolution_seconds": temporal_resolution,
                },
                "spatial_support": [
                    {
                        "unit_type": support_type,
                        "id": unit_id,
                        "mapping_status": "direct",
                    }
                ],
                "measurements": measurements,
                "uncertainty": deepcopy(uncertainty),
                "qualification_receipt_id": None,
                "term_decision_receipt_id": None,
                "waveform_evidence_ids": [wave],
            }
        )

    per_unit_intervals: list[dict[str, Any]] = []
    for unit_index, (unit_id, unit_type) in enumerate(
        zip(primary.unit_ids, primary.unit_types)
    ):
        event_type = "lead" if unit_type == "lead" else "electrode"
        if unit_index not in first_changes or onset_bounds is None:
            per_unit_intervals.append(
                {
                    "unit_type": event_type,
                    "unit_id": unit_id,
                    "interval": None,
                    "status": "not_evaluable",
                }
            )
            continue
        window_index = first_changes[unit_index]
        lower = float(grid.recording_starts[window_index])
        upper = float(grid.recording_stops[window_index])
        per_unit_intervals.append(
            {
                "unit_type": event_type,
                "unit_id": unit_id,
                "interval": {
                    "lower": lower,
                    "median": (lower + upper) / 2.0,
                    "upper": upper,
                    "resolution_seconds": temporal_resolution,
                    "calibration_status": "uncalibrated",
                },
                "status": "observed",
            }
        )

    onset_evidence_by_unit: dict[int, list[str]] = {}
    primary_field_id: str | None = None
    if onset_units:
        onset_window_indices = [first_changes[index] for index in onset_units]
        interval = (
            min(float(grid.recording_starts[index]) for index in onset_window_indices),
            max(float(grid.recording_stops[index]) for index in onset_window_indices),
        )
        field_support = []
        for unit_index in onset_units:
            unit_id = primary.unit_ids[unit_index]
            field_support.append(
                {
                    "unit_type": "lead"
                    if primary.unit_types[unit_index] == "lead"
                    else "electrode",
                    "id": unit_id,
                    "mapping_status": "direct",
                    "support_score": float(
                        scores[first_changes[unit_index], unit_index]
                    ),
                }
            )
        lateralities = sorted(
            {laterality_by_unit[primary.unit_ids[index]] for index in onset_units}
        )
        for laterality in lateralities:
            field_support.append(
                {
                    "unit_type": "laterality",
                    "id": laterality,
                    "mapping_status": "derived",
                }
            )
        field_measurements = [
            _measurement(
                name=f"reference_specific_field_change_score_{ordinal}",
                value=float(scores[first_changes[unit_index], unit_index]),
                unit="robust_z",
                view=primary,
                unit_ids=[primary.unit_ids[unit_index]],
                interval=interval,
                evidence_family="spatial_field",
                background_reference_ids=background_reference_ids,
                policy=policy,
                baseline_delta=float(scores[first_changes[unit_index], unit_index]),
            )
            for ordinal, unit_index in enumerate(onset_units, start=1)
        ]
        primary_field_id = "E-REFERENCE-FIELD-PRIMARY"
        wave = add_waveform(
            primary,
            [primary.unit_ids[index] for index in onset_units],
            interval,
        )
        findings.append(
            {
                "evidence_id": primary_field_id,
                "family": "spatial_field",
                "term": "reference_specific_spatial_field_change_candidate",
                "evidence_role": "onset_support",
                "assertion_level": "measured",
                "status": "present",
                "phase_membership": _phase_membership(interval, segment_spans),
                "time_interval": {
                    "start": interval[0],
                    "stop": interval[1],
                    "resolution_seconds": temporal_resolution,
                },
                "spatial_support": field_support,
                "measurements": field_measurements,
                "uncertainty": deepcopy(uncertainty),
                "qualification_receipt_id": None,
                "term_decision_receipt_id": None,
                "waveform_evidence_ids": [wave],
            }
        )
        for unit_index in onset_units:
            window_index = first_changes[unit_index]
            unit_id = primary.unit_ids[unit_index]
            change_interval = (
                float(grid.recording_starts[window_index]),
                float(grid.recording_stops[window_index]),
            )
            evidence_id = f"E-CHANGE-{len(onset_evidence_by_unit) + 1}"
            onset_evidence_by_unit[unit_index] = [primary_field_id, evidence_id]
            wave = add_waveform(primary, [unit_id], change_interval)
            findings.append(
                {
                    "evidence_id": evidence_id,
                    "family": "evolution",
                    "term": "deterministic_multivariate_change_point_candidate",
                    "evidence_role": "onset_support",
                    "assertion_level": "measured",
                    "status": "present",
                    "phase_membership": _phase_membership(
                        change_interval, segment_spans
                    ),
                    "time_interval": {
                        "start": change_interval[0],
                        "stop": change_interval[1],
                        "resolution_seconds": temporal_resolution,
                    },
                    "spatial_support": [
                        {
                            "unit_type": "lead"
                            if primary.unit_types[unit_index] == "lead"
                            else "electrode",
                            "id": unit_id,
                            "mapping_status": "direct",
                            "support_score": float(scores[window_index, unit_index]),
                        }
                    ],
                    "measurements": [
                        _measurement(
                            name="robust_multifeature_change_score",
                            value=float(scores[window_index, unit_index]),
                            unit="robust_z",
                            view=primary,
                            unit_ids=[unit_id],
                            interval=change_interval,
                            evidence_family="spectral",
                            background_reference_ids=background_reference_ids,
                            policy=policy,
                            baseline_delta=float(scores[window_index, unit_index]),
                        )
                    ],
                    "uncertainty": deepcopy(uncertainty),
                    "qualification_receipt_id": None,
                    "term_decision_receipt_id": None,
                    "waveform_evidence_ids": [wave],
                }
            )

    for reference_index, view in enumerate(prepared[1:], start=2):
        if not onset_units:
            break
        indices = [first_changes[index] for index in onset_units]
        interval = (
            min(float(grid.recording_starts[index]) for index in indices),
            max(float(grid.recording_stops[index]) for index in indices),
        )
        local_grid = _feature_grid(view, policy=policy)
        local_mask = (local_grid.recording_starts >= interval[0] - 1e-9) & (
            local_grid.recording_stops <= interval[1] + 1e-9
        )
        measurements = []
        support = []
        for ordinal, unit_index in enumerate(onset_units, start=1):
            valid = local_mask & local_grid.spatial_valid[:, unit_index]
            amplitude_valid = local_mask & local_grid.amplitude_valid[:, unit_index]
            if not np.any(valid) or not np.any(amplitude_valid):
                continue
            unit_id = view.unit_ids[unit_index]
            measurements.append(
                _measurement(
                    name=f"reference_specific_rms_uv_{ordinal}",
                    value=float(
                        np.nanmedian(local_grid.rms_uv[amplitude_valid, unit_index])
                    ),
                    unit="uV",
                    view=view,
                    unit_ids=[unit_id],
                    interval=interval,
                    evidence_family="spatial_field",
                    background_reference_ids=background_reference_ids,
                    policy=policy,
                )
            )
            support.append(
                {
                    "unit_type": "lead"
                    if view.unit_types[unit_index] == "lead"
                    else "electrode",
                    "id": unit_id,
                    "mapping_status": "direct",
                }
            )
        if not measurements:
            continue
        wave = add_waveform(view, [item["id"] for item in support], interval)
        findings.append(
            {
                "evidence_id": f"E-REFERENCE-FIELD-{reference_index}",
                "family": "spatial_field",
                "term": "reference_specific_spatial_field_measurement",
                "evidence_role": "context_only",
                "assertion_level": "measured",
                "status": "present",
                "phase_membership": _phase_membership(interval, segment_spans),
                "time_interval": {
                    "start": interval[0],
                    "stop": interval[1],
                    "resolution_seconds": temporal_resolution,
                },
                "spatial_support": support,
                "measurements": measurements,
                "uncertainty": deepcopy(uncertainty),
                "qualification_receipt_id": None,
                "term_decision_receipt_id": None,
                "waveform_evidence_ids": [wave],
            }
        )

    recruitment_order: list[dict[str, Any]] = []
    if onset_units:
        source_index = onset_units[0]
        source_window = first_changes[source_index]
        source_interval = (
            float(grid.recording_starts[source_window]),
            float(grid.recording_stops[source_window]),
        )
        source_id = primary.unit_ids[source_index]
        source_type = (
            "lead" if primary.unit_types[source_index] == "lead" else "electrode"
        )
        for ordinal, target_index in enumerate(later_units, start=1):
            target_window = first_changes[target_index]
            target_interval = (
                float(grid.recording_starts[target_window]),
                float(grid.recording_stops[target_window]),
            )
            target_id = primary.unit_ids[target_index]
            target_type = (
                "lead" if primary.unit_types[target_index] == "lead" else "electrode"
            )
            evidence_id = f"E-LATER-INVOLVEMENT-{ordinal}"
            wave = add_waveform(primary, [source_id, target_id], target_interval)
            findings.append(
                {
                    "evidence_id": evidence_id,
                    "family": "spatial_recruitment",
                    "term": "deterministic_later_involvement_candidate",
                    "evidence_role": "spread_support",
                    "assertion_level": "measured",
                    "status": "present",
                    "phase_membership": _phase_membership(
                        target_interval, segment_spans
                    ),
                    "time_interval": {
                        "start": target_interval[0],
                        "stop": target_interval[1],
                        "resolution_seconds": temporal_resolution,
                    },
                    "spatial_support": [
                        {
                            "unit_type": source_type,
                            "id": source_id,
                            "mapping_status": "direct",
                        },
                        {
                            "unit_type": target_type,
                            "id": target_id,
                            "mapping_status": "direct",
                        },
                    ],
                    "measurements": [
                        _measurement(
                            name="later_involvement_delay_seconds",
                            value=target_interval[0] - source_interval[0],
                            unit="s",
                            view=primary,
                            unit_ids=[source_id, target_id],
                            interval=target_interval,
                            evidence_family="spatial_field",
                            background_reference_ids=background_reference_ids,
                            policy=policy,
                        )
                    ],
                    "uncertainty": deepcopy(uncertainty),
                    "qualification_receipt_id": None,
                    "term_decision_receipt_id": None,
                    "waveform_evidence_ids": [wave],
                }
            )
            delay = {
                "lower": target_interval[0] - source_interval[1],
                "median": (
                    (target_interval[0] + target_interval[1]) / 2.0
                    - (source_interval[0] + source_interval[1]) / 2.0
                ),
                "upper": target_interval[1] - source_interval[0],
                "resolution_seconds": temporal_resolution,
                "calibration_status": "uncalibrated",
            }
            relation_status = (
                "precedes"
                if delay["lower"] > temporal_resolution + 1e-9
                else "near_synchronous"
                if delay["lower"] >= -temporal_resolution - 1e-9
                and delay["upper"] <= temporal_resolution + 1e-9
                else "order_unresolved"
            )
            recruitment_order.append(
                {
                    "from_type": source_type,
                    "from_id": source_id,
                    "to_type": target_type,
                    "to_id": target_id,
                    "delay_interval": delay,
                    "relation_status": relation_status,
                    "evidence_ids": [evidence_id],
                }
            )

    if onset_units and primary_field_id is not None:
        ranked = onset_units[: policy.maximum_ranked_candidates]
        candidate_type = (
            "lead" if primary.unit_types[ranked[0]] == "lead" else "electrode"
        )
        ranked_scores = [float(scores[first_changes[index], index]) for index in ranked]
        top_k = [
            {
                "rank": rank,
                "candidate_type": candidate_type,
                "candidate_id": primary.unit_ids[index],
                "score": ranked_scores[rank - 1],
                "score_semantics": "uncalibrated_ranking_score",
                "supporting_evidence_ids": onset_evidence_by_unit[index],
            }
            for rank, index in enumerate(ranked, start=1)
        ]
        top_unit = primary.unit_ids[ranked[0]]
        top_laterality = laterality_by_unit[top_unit]
        top_region = region_by_unit[top_unit]
        supporting_ids = sorted(
            {
                evidence_id
                for row in top_k
                for evidence_id in row["supporting_evidence_ids"]
            }
        )
        spatial_onset = {
            "allowed_resolution": candidate_type,
            "localization_status": "ranked_candidates",
            "phenotype_scores": [{"name": "focal", "score": 1.0}],
            "laterality_scores": [{"name": top_laterality, "score": 1.0}],
            "region_scores": [{"name": top_region, "score": 1.0}],
            "per_unit_intervals": per_unit_intervals,
            "recruitment_order": recruitment_order,
            "top_k": top_k,
            "supporting_evidence_ids": supporting_ids,
            "contradictory_evidence_ids": [],
        }
    else:
        for row in per_unit_intervals:
            row["interval"] = None
            row["status"] = "not_evaluable"
        spatial_onset = {
            "allowed_resolution": "none",
            "localization_status": "nonlocalizable",
            "phenotype_scores": [{"name": "scalp_onset_nonlocalizable", "score": 1.0}],
            "laterality_scores": [{"name": "indeterminate", "score": 1.0}],
            "region_scores": [],
            "per_unit_intervals": per_unit_intervals,
            "recruitment_order": [],
            "top_k": [],
            "supporting_evidence_ids": [],
            "contradictory_evidence_ids": [],
        }

    limitations = [
        {
            "code": "scalp_eeg_only",
            "scope": "clinical_claim",
            "text_zh": "仅输出可重放的头皮 EEG 定量候选，不等同于皮层 SOZ、致痫区或临床诊断。",
        },
        {
            "code": "deterministic_terms_only",
            "scope": "finding",
            "text_zh": "本版本不自动生成 spike、IED、ACNS evolution 或已确认发作等临床术语。",
        },
        {
            "code": "reference_stability_not_evaluated",
            "scope": "spatial",
            "text_zh": "本版本仅输出逐参考测量，尚未实算跨参考场形、极性及排序稳定性；单参考电极极值不进入起始 Top-k。",
        },
    ]
    if not background_available:
        limitations.append(
            {
                "code": "background_unavailable",
                "scope": "signal",
                "text_zh": "未获得合格的发作前背景，未生成相对背景的起始定位证据。",
            }
        )
    if onset_bounds is None:
        limitations.append(
            {
                "code": "onset_not_observed",
                "scope": "boundary",
                "text_zh": "起始边界未观察或被删失，检测器导航锚点未被替代为起始。",
            }
        )
    native_rates = [
        float(row["sample_rate_numerator"]) / float(row["sample_rate_denominator"])
        for row in canonical["channels"]
        if row["observed"]
    ]
    payload = {
        "schema_version": "event_eeg_findings_v1",
        "event_id": event_id,
        "provenance": {
            "record_id": _identifier(canonical["recording_id"], "recording_id"),
            "signal_sha256": canonical["source_signal_sha256"],
            "preprocess_receipt_id": _identifier(
                canonical["canonical_signal_id"], "canonical_signal_id"
            ),
            "model_ids": [
                DETERMINISTIC_EVENT_FINDINGS_METHOD_ID,
                str(adaptive_window["method_id"]),
            ],
            "policy_sha256": policy.sha256,
            "inference_exclusions": {
                "edf_annotations_used": False,
                "excel_used": False,
                "doctor_labels_used": False,
                "clinical_text_used": False,
            },
        },
        "coordinates": {
            "system": "recording_relative_seconds",
            "recording_duration_seconds": float(
                canonical["recording_duration_seconds"]
            ),
            "model_sample_rate_hz": primary.sampling_rate_hz,
            "native_sample_rate_hz": max(native_rates),
        },
        "montage": montage,
        "window": window_core,
        "context": context,
        "quality": quality,
        "qualification_receipts": [],
        "term_decision_receipts": [],
        "findings": findings,
        "spatial_onset": spatial_onset,
        "waveform_evidence": waveform_rows,
        "limitations": limitations,
    }
    return validate_event_eeg_findings_payload(payload)


__all__ = [
    "DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY",
    "DETERMINISTIC_EVENT_FINDINGS_METHOD_ID",
    "DETERMINISTIC_VIEW_TENSOR_HASH_DOMAIN",
    "DeterministicEventFindingsPolicy",
    "DeterministicViewInput",
    "deterministic_view_tensor_sha256",
    "produce_deterministic_event_eeg_findings",
]
