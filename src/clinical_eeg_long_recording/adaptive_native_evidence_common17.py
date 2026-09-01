"""Incremental common-17 native evidence acquisition for one EEG event.

The detector coordinate accepted here is only a navigation anchor.  The core
starts with a minimal q0 support and then *actually reads* incremental left and
right EEG intervals to 8, 16 and 32 seconds independently.  Every reveal is
followed by full native remeasurement.  This differs materially from loading a
90/120-second envelope first or repeatedly analysing nested 60/120/300-second
crops.

The output is an event-level numerical evidence receipt.  It contains robust
local-background matching, physical/spectral/morphology primitives,
per-channel change evidence, an earliest scalp field, spatial connectivity,
reference-stability audits, typed censoring and a replay-oriented query trace.
It cannot certify that the candidate is a seizure and cannot promote a scalp
channel ranking to cortical SOZ, EZ, pathology or a diagnosis.

Only EEG samples, acquisition parameters and an optional sample-level EEG-QC
mask have an input route.  EDF annotations, spreadsheets, doctor text, labels,
video/behaviour, sleep/activation labels and LLM output are absent by design.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from src.soz.geometry import STANDARD_19, TCP_20_EDGES

from .deterministic_event_morphology_primitives_v1 import (
    EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
    measure_native_morphology_primitives,
)
from .signal_findings import (
    SIGNAL_FINDINGS_PRODUCER_ID,
    measure_native_signal_window_features,
)


COMMON17_CHANNELS: Final[tuple[str, ...]] = tuple(
    channel for channel in STANDARD_19 if channel not in {"FZ", "PZ"}
)
COMMON17_INDEX: Final[dict[str, int]] = {
    channel: index for index, channel in enumerate(COMMON17_CHANNELS)
}
COMMON17_TCP_EDGES: Final[tuple[tuple[str, str], ...]] = tuple(
    edge
    for edge in TCP_20_EDGES
    if edge[0] in COMMON17_INDEX and edge[1] in COMMON17_INDEX
)

ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION: Final[
    str
] = "clinical_eeg_common17_adaptive_native_event_evidence_v1"
ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID: Final[
    str
] = "COMMON17-Q0-LR4-8-16-32-NATIVE-EVIDENCE-V1"

_BAND_NAMES: Final[tuple[str, ...]] = (
    "delta_relative_power",
    "theta_relative_power",
    "alpha_relative_power",
    "beta_relative_power",
    "gamma_relative_power",
)
_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
    "teager_energy_uv2",
    *_BAND_NAMES,
    "dominant_frequency_hz",
    "spectral_entropy",
    "spectral_flux",
    "rhythmicity",
    "max_abs_slope_uv_per_s",
    "max_abs_curvature_uv_per_s2",
    "dominant_half_height_width_seconds",
    "sharpness_index",
)
_FEATURE_UNITS: Final[dict[str, str]] = {
    "rms_uv": "uV",
    "peak_to_peak_uv": "uV",
    "line_length_uv_per_sample": "uV_per_sample",
    "teager_energy_uv2": "uV2",
    **{name: "relative_power" for name in _BAND_NAMES},
    "dominant_frequency_hz": "Hz",
    "spectral_entropy": "normalized_ratio",
    "spectral_flux": "normalized_l2",
    "rhythmicity": "spectral_concentration_ratio",
    "max_abs_slope_uv_per_s": "uV_per_s",
    "max_abs_curvature_uv_per_s2": "uV_per_s2",
    "dominant_half_height_width_seconds": "s",
    "sharpness_index": "dimensionless_engineering_index",
}
_POSITIVE_CHANGE_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "rms_uv",
        "peak_to_peak_uv",
        "line_length_uv_per_sample",
        "teager_energy_uv2",
        "spectral_flux",
        "rhythmicity",
        "max_abs_slope_uv_per_s",
        "max_abs_curvature_uv_per_s2",
        "sharpness_index",
    }
)
_LOG_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "rms_uv",
        "peak_to_peak_uv",
        "line_length_uv_per_sample",
        "teager_energy_uv2",
        "spectral_flux",
        "max_abs_slope_uv_per_s",
        "max_abs_curvature_uv_per_s2",
        "sharpness_index",
    }
)
_ROBUST_SCALE_FLOORS: Final[dict[str, float]] = {
    "rms_uv": 0.12,
    "peak_to_peak_uv": 0.12,
    "line_length_uv_per_sample": 0.12,
    "teager_energy_uv2": 0.18,
    **{name: 0.035 for name in _BAND_NAMES},
    "dominant_frequency_hz": 0.75,
    "spectral_entropy": 0.04,
    "spectral_flux": 0.08,
    "rhythmicity": 0.025,
    "max_abs_slope_uv_per_s": 0.18,
    "max_abs_curvature_uv_per_s2": 0.22,
    "dominant_half_height_width_seconds": 0.015,
    "sharpness_index": 0.18,
}
_FAMILY_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "amplitude_and_line_length": (
        "rms_uv",
        "peak_to_peak_uv",
        "line_length_uv_per_sample",
    ),
    "rhythm_and_spectrum": (
        *_BAND_NAMES,
        "dominant_frequency_hz",
        "spectral_entropy",
        "spectral_flux",
        "rhythmicity",
    ),
    "waveform_shape": (
        "teager_energy_uv2",
        "max_abs_slope_uv_per_s",
        "max_abs_curvature_uv_per_s2",
        "dominant_half_height_width_seconds",
        "sharpness_index",
    ),
}
_ALLOWED_CENSORS: Final[frozenset[str]] = frozenset(
    {
        "recording_start",
        "recording_stop",
        "search_cap_32s",
        "impassable_qc_gap",
    }
)
_SCOPE_RECEIPT: Final[dict[str, object]] = {
    "eeg_samples_used": True,
    "acquisition_parameters_used": True,
    "eeg_derived_qc_used_if_supplied": True,
    "detector_anchor_used_for_navigation_only": True,
    "edf_annotation_api_called": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_or_behaviour_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}
_AUTHORIZATION: Final[dict[str, object]] = {
    "output_scope": "event_level_scalp_eeg_numerical_evidence_candidate",
    "candidate_is_confirmed_seizure": False,
    "change_posterior_is_calibrated_probability": False,
    "spike_or_sharp_wave_term_authorized": False,
    "clinical_pathology_term_authorized": False,
    "cortical_soz_or_ez_claim_authorized": False,
    "diagnosis_or_report_claim_authorized": False,
    "downstream_native_remeasurement_required_before_claim": True,
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


def _array_sha256(array: np.ndarray, *, prefix: str) -> str:
    values = np.ascontiguousarray(array)
    header = f"{prefix}:{values.dtype.str}:{'x'.join(map(str, values.shape))}:".encode(
        "ascii"
    )
    return hashlib.sha256(header + values.tobytes(order="C")).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    if len(value) > 160 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{name} is not a safe identifier")
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


def _round(value: float) -> float:
    return round(float(value), 6)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        return _round(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return _round(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class AdaptiveNativeEvidencePolicy:
    """Frozen engineering policy; thresholds are not clinical norms."""

    q0_extent_seconds_each_side: float = 4.0
    expansion_extents_seconds_each_side: tuple[float, ...] = (8.0, 16.0, 32.0)
    window_seconds: float = 1.0
    step_seconds: float = 0.5
    baseline_guard_before_anchor_seconds: float = 2.0
    baseline_early_fraction: float = 0.5
    minimum_baseline_windows: int = 4
    maximum_baseline_windows: int = 12
    baseline_trim_quantile: float = 0.75
    minimum_qc_valid_fraction_per_window: float = 0.90
    minimum_evaluable_channel_fraction: float = 0.70
    change_score_threshold: float = 3.0
    return_score_threshold: float = 1.6
    minimum_onset_run_windows: int = 2
    minimum_active_channels: int = 2
    earliest_field_tolerance_seconds: float = 1.0
    left_boundary_guard_seconds: float = 1.5
    recovery_duration_seconds: float = 3.0
    minimum_evolution_seconds: float = 6.0
    saturation_context_seconds: float = 4.0
    saturation_similarity_threshold: float = 0.90
    posterior_temperature: float = 1.5

    def __post_init__(self) -> None:
        extents = tuple(float(item) for item in self.expansion_extents_seconds_each_side)
        if extents != (8.0, 16.0, 32.0):
            raise ValueError("v1 freezes independent expansion extents to 8/16/32 s")
        if float(self.q0_extent_seconds_each_side) != 4.0:
            raise ValueError("v1 freezes q0 to four seconds on each side")
        if float(self.window_seconds) != 1.0 or float(self.step_seconds) != 0.5:
            raise ValueError("v1 freezes one-second windows with 0.5-second steps")
        for name in (
            "window_seconds",
            "step_seconds",
            "baseline_guard_before_anchor_seconds",
            "left_boundary_guard_seconds",
            "recovery_duration_seconds",
            "minimum_evolution_seconds",
            "saturation_context_seconds",
            "posterior_temperature",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "baseline_early_fraction",
            "baseline_trim_quantile",
            "minimum_qc_valid_fraction_per_window",
            "minimum_evaluable_channel_fraction",
            "saturation_similarity_threshold",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0,1]")
        if self.step_seconds > self.window_seconds:
            raise ValueError("step_seconds cannot exceed window_seconds")
        if (
            isinstance(self.minimum_baseline_windows, bool)
            or self.minimum_baseline_windows < 3
            or self.maximum_baseline_windows < self.minimum_baseline_windows
        ):
            raise ValueError("baseline window counts are invalid")
        if (
            isinstance(self.minimum_onset_run_windows, bool)
            or self.minimum_onset_run_windows < 2
            or isinstance(self.minimum_active_channels, bool)
            or self.minimum_active_channels < 1
        ):
            raise ValueError("onset persistence policy is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "channel_contract": "STANDARD_19_minus_FZ_PZ_no_imputation_v1",
            "feature_kernel_bindings": {
                "window_kernel": SIGNAL_FINDINGS_PRODUCER_ID,
                "morphology_kernel": EVENT_MORPHOLOGY_PRIMITIVE_METHOD_ID,
                "new_native_primitives": [
                    "teager_energy_uv2",
                    "spectral_flux",
                ],
            },
            "posterior_semantics": (
                "algorithmic_logistic_change_evidence_and_within_event_spatial_"
                "softmax_not_clinically_calibrated"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY = AdaptiveNativeEvidencePolicy()


@dataclass(frozen=True)
class NativeEEGQueryChunk:
    """The only allowed host response: common-17 volts plus EEG-derived QC."""

    signal_volts: np.ndarray
    valid_sample_mask: np.ndarray | None = None


NativeEEGQueryReader = Callable[[int, int], NativeEEGQueryChunk | np.ndarray]


@dataclass(frozen=True)
class _AcquiredChunk:
    start_sample: int
    stop_sample: int
    signal_volts: np.ndarray
    valid_sample_mask: np.ndarray
    signal_sha256: str
    qc_sha256: str


@dataclass
class _EvidenceSnapshot:
    serializable: dict[str, Any]
    baseline_status: str
    onset_sample: int | None
    onset_relative_seconds: float | None
    earliest_channel_samples: dict[str, int | None]
    recovery_sample: int | None
    support_start_sample: int
    support_stop_sample: int
    onset_left_margin_seconds: float | None
    posterior_saturation_similarity: float | None
    reference_status: str


def _normalize_query_result(
    result: NativeEEGQueryChunk | np.ndarray,
    *,
    expected_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(result, NativeEEGQueryChunk):
        signal_raw = result.signal_volts
        mask_raw = result.valid_sample_mask
    elif isinstance(result, np.ndarray):
        signal_raw = result
        mask_raw = None
    else:
        raise TypeError(
            "query reader must return NativeEEGQueryChunk or a common-17 ndarray"
        )
    signal = np.asarray(signal_raw, dtype=np.float64)
    if signal.shape != (len(COMMON17_CHANNELS), expected_samples):
        raise ValueError(
            "query reader must return exact common-17 [17, requested_samples] data"
        )
    if not np.isfinite(signal).all():
        raise ValueError("queried EEG must be finite")
    if mask_raw is None:
        mask = np.ones(signal.shape, dtype=bool)
    else:
        supplied = np.asarray(mask_raw)
        if supplied.shape == (expected_samples,):
            supplied = np.broadcast_to(supplied[None, :], signal.shape)
        if supplied.shape != signal.shape or supplied.dtype != np.bool_:
            raise ValueError("EEG-QC mask must be boolean [T] or [17,T]")
        mask = np.array(supplied, dtype=bool, copy=True)
    return np.ascontiguousarray(signal), np.ascontiguousarray(mask)


def _assemble(chunks: Sequence[_AcquiredChunk]) -> tuple[int, int, np.ndarray, np.ndarray]:
    ordered = sorted(chunks, key=lambda item: (item.start_sample, item.stop_sample))
    if not ordered:
        raise ValueError("no EEG has been acquired")
    cursor = ordered[0].start_sample
    signal_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for chunk in ordered:
        if chunk.start_sample != cursor:
            raise ValueError("adaptive native support must remain contiguous")
        signal_rows.append(chunk.signal_volts)
        mask_rows.append(chunk.valid_sample_mask)
        cursor = chunk.stop_sample
    return (
        ordered[0].start_sample,
        ordered[-1].stop_sample,
        np.concatenate(signal_rows, axis=1),
        np.concatenate(mask_rows, axis=1),
    )


def _window_starts(sample_count: int, *, rate: float, policy: AdaptiveNativeEvidencePolicy) -> np.ndarray:
    window = int(round(policy.window_seconds * rate))
    step = int(round(policy.step_seconds * rate))
    if window < 5 or step < 1 or sample_count < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, sample_count - window + 1, step, dtype=np.int64)


def _teager_and_flux(
    signal_uv: np.ndarray,
    starts: np.ndarray,
    *,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    teager = np.zeros((starts.size, signal_uv.shape[0]), dtype=np.float64)
    spectra: list[np.ndarray] = []
    taper = np.hanning(window_samples)[None, :]
    for row_index, start in enumerate(starts):
        window = signal_uv[:, int(start) : int(start) + window_samples]
        centered = window - np.median(window, axis=1, keepdims=True)
        if centered.shape[1] >= 3:
            energy = centered[:, 1:-1] ** 2 - centered[:, :-2] * centered[:, 2:]
            teager[row_index] = np.mean(np.abs(energy), axis=1)
        spectrum = np.abs(np.fft.rfft(centered * taper, axis=1))
        spectrum /= np.maximum(np.linalg.norm(spectrum, axis=1, keepdims=True), 1e-12)
        spectra.append(spectrum)
    flux = np.zeros_like(teager)
    for index in range(1, len(spectra)):
        flux[index] = np.sqrt(np.sum((spectra[index] - spectra[index - 1]) ** 2, axis=1))
    if flux.shape[0] > 1:
        flux[0] = flux[1]
    return teager, flux


def _morphology_features(
    signal_volts: np.ndarray,
    starts: np.ndarray,
    *,
    global_start_sample: int,
    rate: float,
    window_samples: int,
    cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (starts.size, signal_volts.shape[0])
    slope = np.zeros(shape, dtype=np.float64)
    curvature = np.zeros(shape, dtype=np.float64)
    width = np.zeros(shape, dtype=np.float64)
    sharpness = np.zeros(shape, dtype=np.float64)
    high = min(45.0, 0.45 * rate)
    for window_index, local_start in enumerate(starts):
        absolute_start = global_start_sample + int(local_start)
        for channel_index in range(signal_volts.shape[0]):
            key = (absolute_start, channel_index)
            cached = cache.get(key)
            if cached is None:
                measured = measure_native_morphology_primitives(
                    signal_volts[
                        channel_index,
                        int(local_start) : int(local_start) + window_samples,
                    ],
                    sampling_rate_hz=rate,
                    effective_bandwidth_hz=(0.5, high),
                )
                values = measured["values"]
                available = measured["available"]
                maximum_slope = max(
                    float(values["max_rise_slope_uv_per_s"]),
                    float(values["max_fall_slope_uv_per_s"]),
                )
                maximum_curvature = float(values["max_abs_curvature_uv_per_s2"])
                half_width = (
                    float(values["dominant_excursion_half_height_width_seconds"])
                    if bool(available["dominant_excursion_half_height_width_seconds"])
                    else policy_safe_width(rate)
                )
                engineering_sharpness = math.log1p(maximum_curvature) / max(
                    half_width, policy_safe_width(rate)
                )
                cached = (
                    maximum_slope,
                    maximum_curvature,
                    half_width,
                    engineering_sharpness,
                )
                cache[key] = cached
            slope[window_index, channel_index] = cached[0]
            curvature[window_index, channel_index] = cached[1]
            width[window_index, channel_index] = cached[2]
            sharpness[window_index, channel_index] = cached[3]
    return slope, curvature, width, sharpness


def policy_safe_width(rate: float) -> float:
    """Numerical half-width floor; this is not a spike-duration threshold."""

    return 1.0 / float(rate)


def _native_feature_tensor(
    signal_volts: np.ndarray,
    starts: np.ndarray,
    *,
    global_start_sample: int,
    rate: float,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> np.ndarray:
    starts_seconds = starts.astype(np.float64) / rate
    base = measure_native_signal_window_features(
        signal_volts * 1.0e6,
        starts_seconds,
        sampling_rate_hz=rate,
        window_seconds=(int(round(rate)) / rate),
    )
    window_samples = int(round(rate))
    teager, flux = _teager_and_flux(
        signal_volts * 1.0e6,
        starts,
        window_samples=window_samples,
    )
    slope, curvature, width, sharpness = _morphology_features(
        signal_volts,
        starts,
        global_start_sample=global_start_sample,
        rate=rate,
        window_samples=window_samples,
        cache=morphology_cache,
    )
    return np.stack(
        (
            base["rms"],
            base["amplitude"],
            base["line_length"],
            teager,
            *[base["band_ratio"][:, :, index] for index in range(5)],
            base["peak_frequency"],
            base["spectral_entropy"],
            flux,
            base["spectral_concentration"],
            slope,
            curvature,
            width,
            sharpness,
        ),
        axis=2,
    )


def _transform_feature_tensor(features: np.ndarray) -> np.ndarray:
    result = np.array(features, dtype=np.float64, copy=True)
    for index, name in enumerate(_FEATURE_NAMES):
        if name in _LOG_FEATURES:
            result[:, :, index] = np.log(np.maximum(result[:, :, index], 1e-12))
    return result


def _window_qc_mask(
    valid_samples: np.ndarray,
    starts: np.ndarray,
    *,
    window_samples: int,
    threshold: float,
) -> np.ndarray:
    rows = [
        np.mean(valid_samples[:, int(start) : int(start) + window_samples], axis=1)
        >= threshold
        for start in starts
    ]
    return np.stack(rows, axis=0) if rows else np.zeros((0, valid_samples.shape[0]), dtype=bool)


def _baseline_selection(
    transformed: np.ndarray,
    window_qc: np.ndarray,
    starts_absolute: np.ndarray,
    *,
    support_start_sample: int,
    anchor_sample: int,
    window_samples: int,
    rate: float,
    policy: AdaptiveNativeEvidencePolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = anchor_sample - int(round(policy.baseline_guard_before_anchor_seconds * rate))
    early_stop = support_start_sample + int(
        round((cutoff - support_start_sample) * policy.baseline_early_fraction)
    )
    candidate = (
        (starts_absolute + window_samples <= early_stop)
        & (
            np.mean(window_qc, axis=1)
            >= policy.minimum_evaluable_channel_fraction
        )
    )
    candidate_indices = np.flatnonzero(candidate)
    if candidate_indices.size < policy.minimum_baseline_windows:
        return np.zeros(starts_absolute.size, dtype=bool), {
            "status": "insufficient_clean_matched_baseline",
            "pool_interval_recording_seconds": [
                _round(support_start_sample / rate),
                _round(max(support_start_sample, early_stop) / rate),
            ],
            "candidate_window_count": int(candidate_indices.size),
            "selected_window_count": 0,
            "selection_rule": "early_local_clean_robust_multivariate_trim_v1",
            "selected_window_intervals_recording_seconds": [],
        }
    signature_indices = [
        _FEATURE_NAMES.index("rms_uv"),
        _FEATURE_NAMES.index("line_length_uv_per_sample"),
        _FEATURE_NAMES.index("dominant_frequency_hz"),
        _FEATURE_NAMES.index("spectral_entropy"),
        _FEATURE_NAMES.index("rhythmicity"),
    ]
    signatures = np.nanmedian(
        np.where(window_qc[:, :, None], transformed, np.nan), axis=1
    )[:, signature_indices]
    pool = signatures[candidate_indices]
    center = np.nanmedian(pool, axis=0)
    mad = np.nanmedian(np.abs(pool - center), axis=0)
    scale = np.maximum(1.4826 * mad, np.asarray((0.12, 0.12, 0.75, 0.04, 0.025)))
    distance = np.sqrt(np.nanmean(((pool - center) / scale) ** 2, axis=1))
    threshold = float(np.quantile(distance, policy.baseline_trim_quantile))
    retained = candidate_indices[distance <= threshold + 1e-12]
    if retained.size < policy.minimum_baseline_windows:
        retained = candidate_indices[
            np.argsort(distance, kind="stable")[: policy.minimum_baseline_windows]
        ]
    if retained.size > policy.maximum_baseline_windows:
        distance_by_index = {int(index): float(value) for index, value in zip(candidate_indices, distance)}
        retained = np.asarray(
            sorted(
                retained.tolist(),
                key=lambda index: (distance_by_index[int(index)], int(index)),
            )[: policy.maximum_baseline_windows],
            dtype=np.int64,
        )
    retained.sort()
    baseline_mask = np.zeros(starts_absolute.size, dtype=bool)
    baseline_mask[retained] = True
    intervals = [
        [
            _round(starts_absolute[index] / rate),
            _round((starts_absolute[index] + window_samples) / rate),
        ]
        for index in retained
    ]
    return baseline_mask, {
        "status": "qualified_robust_matched_baseline",
        "pool_interval_recording_seconds": [
            _round(support_start_sample / rate),
            _round(early_stop / rate),
        ],
        "candidate_window_count": int(candidate_indices.size),
        "selected_window_count": int(retained.size),
        "selection_rule": "early_local_clean_robust_multivariate_trim_v1",
        "trim_quantile": policy.baseline_trim_quantile,
        "selected_window_intervals_recording_seconds": intervals,
        "same_channel_sampling_reference_contract": True,
    }


def _robust_change_scores(
    transformed: np.ndarray,
    window_qc: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    policy: AdaptiveNativeEvidencePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows, channels, feature_count = transformed.shape
    z = np.zeros_like(transformed)
    centers = np.zeros((channels, feature_count), dtype=np.float64)
    scales = np.zeros_like(centers)
    evaluable = np.zeros(channels, dtype=bool)
    for channel in range(channels):
        valid_baseline = baseline_mask & window_qc[:, channel]
        if int(np.count_nonzero(valid_baseline)) < policy.minimum_baseline_windows:
            continue
        evaluable[channel] = True
        baseline = transformed[valid_baseline, channel]
        centers[channel] = np.median(baseline, axis=0)
        mad = np.median(np.abs(baseline - centers[channel]), axis=0)
        floors = np.asarray([_ROBUST_SCALE_FLOORS[name] for name in _FEATURE_NAMES])
        scales[channel] = np.maximum(1.4826 * mad, floors)
        raw_z = (transformed[:, channel] - centers[channel]) / scales[channel]
        for feature_index, name in enumerate(_FEATURE_NAMES):
            if name in _POSITIVE_CHANGE_FEATURES:
                z[:, channel, feature_index] = np.maximum(0.0, raw_z[:, feature_index])
            else:
                z[:, channel, feature_index] = np.abs(raw_z[:, feature_index])
    z = np.clip(z, 0.0, 20.0)
    z[~window_qc] = 0.0
    family_scores: list[np.ndarray] = []
    for names in _FAMILY_FEATURES.values():
        indices = [_FEATURE_NAMES.index(name) for name in names]
        family_values = np.sort(z[:, :, indices], axis=2)
        take = min(2, len(indices))
        family_scores.append(np.mean(family_values[:, :, -take:], axis=2))
    families = np.stack(family_scores, axis=2)
    ranked_families = np.sort(families, axis=2)
    scores = np.mean(ranked_families[:, :, -2:], axis=2)
    scores[:, ~evaluable] = 0.0
    change_probability = 1.0 / (
        1.0 + np.exp(-np.clip((scores - policy.change_score_threshold) / 0.75, -30.0, 30.0))
    )
    logits = scores / policy.posterior_temperature
    logits[:, ~evaluable] = -np.inf
    spatial = np.zeros_like(scores)
    if np.any(evaluable):
        maxima = np.max(logits[:, evaluable], axis=1, keepdims=True)
        exponent = np.exp(logits[:, evaluable] - maxima)
        spatial[:, evaluable] = exponent / np.maximum(
            np.sum(exponent, axis=1, keepdims=True), 1e-12
        )
    return scores, change_probability, spatial, centers, scales


def _global_score(scores: np.ndarray, evaluable: np.ndarray) -> np.ndarray:
    if not np.any(evaluable):
        return np.zeros(scores.shape[0], dtype=np.float64)
    values = np.sort(scores[:, evaluable], axis=1)
    take = min(3, values.shape[1])
    return np.mean(values[:, -take:], axis=1)


def _persistent_start(
    mask: np.ndarray,
    *,
    minimum_length: int,
    start_index: int = 0,
) -> int | None:
    run = 0
    for index in range(max(0, start_index), mask.size):
        run = run + 1 if bool(mask[index]) else 0
        if run >= minimum_length:
            return index - run + 1
    return None


def _channel_onsets(
    scores: np.ndarray,
    starts_absolute: np.ndarray,
    search_mask: np.ndarray,
    *,
    policy: AdaptiveNativeEvidencePolicy,
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for channel_index, channel in enumerate(COMMON17_CHANNELS):
        active = search_mask & (scores[:, channel_index] >= policy.change_score_threshold)
        index = _persistent_start(
            active,
            minimum_length=policy.minimum_onset_run_windows,
        )
        result[channel] = int(starts_absolute[index]) if index is not None else None
    return result


def _components(channels: Sequence[str]) -> list[list[str]]:
    remaining = set(channels)
    adjacency = {name: set() for name in COMMON17_CHANNELS}
    for left, right in COMMON17_TCP_EDGES:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component or node not in remaining:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node]))
        remaining.difference_update(component)
        result.append(sorted(component, key=COMMON17_CHANNELS.index))
    result.sort(key=lambda row: (-len(row), [COMMON17_CHANNELS.index(item) for item in row]))
    return result


def _js_similarity(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / max(float(np.sum(p)), 1e-12)
    q = q / max(float(np.sum(q)), 1e-12)
    middle = 0.5 * (p + q)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / middle[mask])))

    return float(np.clip(1.0 - 0.5 * (divergence(p) + divergence(q)), 0.0, 1.0))


def _reference_view_distribution(
    data_uv: np.ndarray,
    starts_seconds: np.ndarray,
    baseline_mask: np.ndarray,
    opportunity: np.ndarray,
    onset_indices: np.ndarray,
    *,
    rate: float,
) -> np.ndarray:
    features = measure_native_signal_window_features(
        data_uv,
        starts_seconds,
        sampling_rate_hz=rate,
        window_seconds=1.0,
    )
    rows = (
        np.log(np.maximum(features["rms"], 1e-12)),
        np.log(np.maximum(features["line_length"], 1e-12)),
        features["band_ratio"],
        features["peak_frequency"][:, :, None],
        features["spectral_entropy"][:, :, None],
        features["spectral_concentration"][:, :, None],
    )
    tensor = np.concatenate(
        (rows[0][:, :, None], rows[1][:, :, None], *rows[2:]), axis=2
    )
    units = tensor.shape[1]
    scores = np.zeros((tensor.shape[0], units), dtype=np.float64)
    for unit in range(units):
        valid = baseline_mask & opportunity[:, unit]
        if np.count_nonzero(valid) < 3:
            continue
        baseline = tensor[valid, unit]
        center = np.median(baseline, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(baseline - center), axis=0), 0.05)
        z = np.abs((tensor[:, unit] - center) / scale)
        z[:, -1] = np.maximum(0.0, (tensor[:, unit, -1] - center[-1]) / scale[-1])
        ranked = np.sort(np.clip(z, 0.0, 20.0), axis=1)
        scores[:, unit] = np.mean(ranked[:, -2:], axis=1)
    vector = np.mean(scores[onset_indices], axis=0) if onset_indices.size else np.max(scores, axis=0)
    vector = np.exp(vector / 1.5 - np.max(vector / 1.5))
    return vector / max(float(np.sum(vector)), 1e-12)


def _reference_stability(
    signal_volts: np.ndarray,
    starts: np.ndarray,
    window_qc: np.ndarray,
    baseline_mask: np.ndarray,
    onset_indices: np.ndarray,
    native_distribution: np.ndarray,
    *,
    rate: float,
) -> dict[str, Any]:
    if not np.any(baseline_mask):
        return {
            "status": "not_evaluable_baseline_unavailable",
            "view_rankings": {},
            "pairwise_js_similarity": {},
            "minimum_similarity": None,
            "top3_consensus_channels": [],
        }
    starts_seconds = starts.astype(np.float64) / rate
    referential = _reference_view_distribution(
        signal_volts * 1e6,
        starts_seconds,
        baseline_mask,
        window_qc,
        onset_indices,
        rate=rate,
    )
    common_average = signal_volts - np.mean(signal_volts, axis=0, keepdims=True)
    car = _reference_view_distribution(
        common_average * 1e6,
        starts_seconds,
        baseline_mask,
        window_qc,
        onset_indices,
        rate=rate,
    )
    bipolar = np.stack(
        [
            signal_volts[COMMON17_INDEX[left]] - signal_volts[COMMON17_INDEX[right]]
            for left, right in COMMON17_TCP_EDGES
        ]
    )
    edge_qc = np.stack(
        [
            window_qc[:, COMMON17_INDEX[left]] & window_qc[:, COMMON17_INDEX[right]]
            for left, right in COMMON17_TCP_EDGES
        ],
        axis=1,
    )
    edge_distribution = _reference_view_distribution(
        bipolar * 1e6,
        starts_seconds,
        baseline_mask,
        edge_qc,
        onset_indices,
        rate=rate,
    )
    projected = np.zeros(len(COMMON17_CHANNELS), dtype=np.float64)
    for weight, (left, right) in zip(edge_distribution, COMMON17_TCP_EDGES):
        projected[COMMON17_INDEX[left]] += 0.5 * weight
        projected[COMMON17_INDEX[right]] += 0.5 * weight
    projected /= max(float(np.sum(projected)), 1e-12)
    views = {
        "native_referential": referential,
        "common_average": car,
        "tcp_bipolar_endpoint_projection": projected,
    }
    similarities = {
        "native_vs_common_average": _js_similarity(referential, car),
        "native_vs_tcp_projection": _js_similarity(referential, projected),
        "common_average_vs_tcp_projection": _js_similarity(car, projected),
    }
    rankings = {
        name: [
            COMMON17_CHANNELS[index]
            for index in sorted(
                range(len(COMMON17_CHANNELS)),
                key=lambda index: (-float(values[index]), COMMON17_CHANNELS[index]),
            )
        ]
        for name, values in views.items()
    }
    top_sets = [set(row[:3]) for row in rankings.values()]
    consensus = sorted(set.intersection(*top_sets), key=COMMON17_CHANNELS.index)
    minimum = min(similarities.values())
    return {
        "status": "stable" if minimum >= 0.75 else "reference_sensitive",
        "view_rankings": rankings,
        "view_channel_mass": {
            name: {
                channel: _round(values[index])
                for index, channel in enumerate(COMMON17_CHANNELS)
            }
            for name, values in views.items()
        },
        "pairwise_js_similarity": {name: _round(value) for name, value in similarities.items()},
        "minimum_similarity": _round(minimum),
        "top3_consensus_channels": consensus,
        "native_distribution_binding_matches_primary": bool(
            np.allclose(native_distribution, referential, atol=0.15)
        ),
        "clinical_probability_calibrated": False,
    }


def _summarize_primitives(
    raw_features: np.ndarray,
    transformed: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    baseline_mask: np.ndarray,
    window_qc: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, Any]:
    channel_rows: dict[str, Any] = {}
    for channel_index, channel in enumerate(COMMON17_CHANNELS):
        valid_candidate = candidate_indices[window_qc[candidate_indices, channel_index]]
        feature_rows: dict[str, Any] = {}
        for feature_index, name in enumerate(_FEATURE_NAMES):
            available = valid_candidate.size > 0 and np.count_nonzero(
                baseline_mask & window_qc[:, channel_index]
            ) > 0
            if not available:
                feature_rows[name] = {
                    "unit": _FEATURE_UNITS[name],
                    "available": False,
                    "baseline_median": None,
                    "candidate_peak": None,
                    "maximum_robust_change_z": None,
                }
                continue
            baseline_indices = np.flatnonzero(baseline_mask & window_qc[:, channel_index])
            baseline_median = float(np.median(raw_features[baseline_indices, channel_index, feature_index]))
            candidate_values = raw_features[valid_candidate, channel_index, feature_index]
            if name in _POSITIVE_CHANGE_FEATURES:
                raw_z = np.maximum(
                    0.0,
                    (transformed[valid_candidate, channel_index, feature_index] - centers[channel_index, feature_index])
                    / max(scales[channel_index, feature_index], 1e-12),
                )
                peak_local = int(np.argmax(raw_z))
            else:
                raw_z = np.abs(
                    (transformed[valid_candidate, channel_index, feature_index] - centers[channel_index, feature_index])
                    / max(scales[channel_index, feature_index], 1e-12)
                )
                peak_local = int(np.argmax(raw_z))
            feature_rows[name] = {
                "unit": _FEATURE_UNITS[name],
                "available": True,
                "baseline_median": _round(baseline_median),
                "candidate_peak": _round(candidate_values[peak_local]),
                "maximum_robust_change_z": _round(min(20.0, float(raw_z[peak_local]))),
            }
        channel_rows[channel] = feature_rows
    return {
        "feature_order": list(_FEATURE_NAMES),
        "feature_units": deepcopy(_FEATURE_UNITS),
        "channel_feature_summaries": channel_rows,
        "clinical_term_qualification_authorized": False,
        "teager_definition": "mean_abs_x[n]^2_minus_x[n-1]x[n+1]",
        "spectral_flux_definition": "l2_successive_unit_norm_magnitude_spectra",
        "sharpness_definition": (
            "log1p_max_abs_curvature_divided_by_dominant_half_height_width_floor"
        ),
    }


def _evaluate_support(
    chunks: Sequence[_AcquiredChunk],
    *,
    anchor_sample: int,
    rate: float,
    policy: AdaptiveNativeEvidencePolicy,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> _EvidenceSnapshot:
    support_start, support_stop, signal, qc = _assemble(chunks)
    starts_local = _window_starts(signal.shape[1], rate=rate, policy=policy)
    if starts_local.size == 0:
        serializable = {
            "status": "insufficient_window_support",
            "robust_matched_baseline": {
                "status": "insufficient_clean_matched_baseline",
                "candidate_window_count": 0,
                "selected_window_count": 0,
                "selected_window_intervals_recording_seconds": [],
            },
            "onset_candidate": None,
            "per_channel_evidence": [],
            "earliest_field": None,
            "spatial_connectivity": None,
            "reference_stability": {
                "status": "not_evaluable_baseline_unavailable"
            },
            "evolution": None,
            "native_primitives": None,
        }
        return _EvidenceSnapshot(
            serializable=serializable,
            baseline_status="insufficient_clean_matched_baseline",
            onset_sample=None,
            onset_relative_seconds=None,
            earliest_channel_samples={channel: None for channel in COMMON17_CHANNELS},
            recovery_sample=None,
            support_start_sample=support_start,
            support_stop_sample=support_stop,
            onset_left_margin_seconds=None,
            posterior_saturation_similarity=None,
            reference_status="not_evaluable_baseline_unavailable",
        )
    starts_absolute = support_start + starts_local
    window_samples = int(round(policy.window_seconds * rate))
    window_qc = _window_qc_mask(
        qc,
        starts_local,
        window_samples=window_samples,
        threshold=policy.minimum_qc_valid_fraction_per_window,
    )
    raw = _native_feature_tensor(
        signal,
        starts_local,
        global_start_sample=support_start,
        rate=rate,
        morphology_cache=morphology_cache,
    )
    transformed = _transform_feature_tensor(raw)
    baseline_mask, baseline_receipt = _baseline_selection(
        transformed,
        window_qc,
        starts_absolute,
        support_start_sample=support_start,
        anchor_sample=anchor_sample,
        window_samples=window_samples,
        rate=rate,
        policy=policy,
    )
    if not np.any(baseline_mask):
        serializable = {
            "status": "insufficient_clean_matched_baseline",
            "window_count": int(starts_local.size),
            "robust_matched_baseline": baseline_receipt,
            "onset_candidate": None,
            "per_channel_evidence": [],
            "earliest_field": None,
            "spatial_connectivity": None,
            "reference_stability": {
                "status": "not_evaluable_baseline_unavailable",
                "view_rankings": {},
            },
            "evolution": None,
            "native_primitives": {
                "feature_order": list(_FEATURE_NAMES),
                "measurement_completed": True,
                "qualification_withheld_reason": "baseline_unavailable",
            },
        }
        return _EvidenceSnapshot(
            serializable=serializable,
            baseline_status=str(baseline_receipt["status"]),
            onset_sample=None,
            onset_relative_seconds=None,
            earliest_channel_samples={channel: None for channel in COMMON17_CHANNELS},
            recovery_sample=None,
            support_start_sample=support_start,
            support_stop_sample=support_stop,
            onset_left_margin_seconds=None,
            posterior_saturation_similarity=None,
            reference_status="not_evaluable_baseline_unavailable",
        )
    scores, probability, spatial, centers, scales = _robust_change_scores(
        transformed,
        window_qc,
        baseline_mask,
        policy=policy,
    )
    evaluable = scales[:, 0] > 0.0
    global_scores = _global_score(scores, evaluable)
    pool_stop_seconds = float(baseline_receipt["pool_interval_recording_seconds"][1])
    search_start_sample = max(
        support_start,
        int(round(pool_stop_seconds * rate)),
    )
    search_mask = starts_absolute >= search_start_sample
    active_channels = np.sum(scores >= policy.change_score_threshold, axis=1)
    sustained = (
        search_mask
        & (global_scores >= policy.change_score_threshold)
        & (active_channels >= policy.minimum_active_channels)
    )
    onset_index = _persistent_start(
        sustained,
        minimum_length=policy.minimum_onset_run_windows,
    )
    channel_onsets = _channel_onsets(
        scores,
        starts_absolute,
        search_mask,
        policy=policy,
    )
    onset_sample = int(starts_absolute[onset_index]) if onset_index is not None else None
    onset_relative = (
        (onset_sample - anchor_sample) / rate if onset_sample is not None else None
    )
    onset_indices = (
        np.flatnonzero(
            (starts_absolute >= onset_sample)
            & (starts_absolute <= onset_sample + int(round(2.0 * rate)))
        )
        if onset_sample is not None
        else np.zeros(0, dtype=np.int64)
    )
    if onset_indices.size:
        onset_distribution = np.mean(spatial[onset_indices], axis=0)
        onset_distribution /= max(float(np.sum(onset_distribution)), 1e-12)
    elif np.any(evaluable):
        peak = int(np.argmax(global_scores))
        onset_distribution = spatial[peak]
    else:
        onset_distribution = np.zeros(len(COMMON17_CHANNELS), dtype=np.float64)
    per_channel: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(COMMON17_CHANNELS):
        peak_index = int(np.argmax(scores[:, channel_index]))
        channel_onset = channel_onsets[channel]
        per_channel.append(
            {
                "channel": channel,
                "evaluable": bool(evaluable[channel_index]),
                "earliest_change_recording_seconds": (
                    _round(channel_onset / rate) if channel_onset is not None else None
                ),
                "earliest_change_relative_to_anchor_seconds": (
                    _round((channel_onset - anchor_sample) / rate)
                    if channel_onset is not None
                    else None
                ),
                "peak_change_score": _round(scores[peak_index, channel_index]),
                "peak_algorithmic_change_posterior": _round(
                    probability[peak_index, channel_index]
                ),
                "onset_spatial_posterior_mass": _round(onset_distribution[channel_index]),
                "peak_time_relative_to_anchor_seconds": _round(
                    (starts_absolute[peak_index] - anchor_sample) / rate
                ),
            }
        )
    per_channel.sort(
        key=lambda row: (
            -float(row["onset_spatial_posterior_mass"]),
            -float(row["peak_change_score"]),
            COMMON17_CHANNELS.index(str(row["channel"])),
        )
    )
    earliest_channels: list[str] = []
    earliest_time: int | None = None
    if onset_sample is not None:
        observed = [value for value in channel_onsets.values() if value is not None]
        if observed:
            earliest_time = min(observed)
            tolerance = int(round(policy.earliest_field_tolerance_seconds * rate))
            earliest_channels = [
                channel
                for channel in COMMON17_CHANNELS
                if channel_onsets[channel] is not None
                and int(channel_onsets[channel]) <= earliest_time + tolerance
            ]
    components = _components(earliest_channels)
    dominant = components[0] if components else []
    graph_edges = [
        [left, right]
        for left, right in COMMON17_TCP_EDGES
        if left in earliest_channels and right in earliest_channels
    ]
    connectivity = (
        len(dominant) / len(earliest_channels) if earliest_channels else 0.0
    )
    recovery_index: int | None = None
    if onset_index is not None:
        needed = max(1, int(math.ceil(policy.recovery_duration_seconds / policy.step_seconds)))
        below = global_scores < policy.return_score_threshold
        recovery_index = _persistent_start(
            below,
            minimum_length=needed,
            start_index=onset_index + policy.minimum_onset_run_windows,
        )
    recovery_sample = (
        int(starts_absolute[recovery_index]) if recovery_index is not None else None
    )
    saturation: float | None = None
    if onset_index is not None:
        context_windows = max(1, int(round(policy.saturation_context_seconds / policy.step_seconds)))
        active_indices = np.flatnonzero(starts_absolute >= onset_sample)
        if active_indices.size >= 2 * context_windows:
            previous = np.mean(spatial[active_indices[-2 * context_windows : -context_windows]], axis=0)
            current = np.mean(spatial[active_indices[-context_windows:]], axis=0)
            saturation = _js_similarity(previous, current)
    reference = _reference_stability(
        signal,
        starts_local,
        window_qc,
        baseline_mask,
        onset_indices,
        onset_distribution,
        rate=rate,
    )
    candidate_indices = np.flatnonzero(search_mask)
    primitive_summary = _summarize_primitives(
        raw,
        transformed,
        centers,
        scales,
        baseline_mask,
        window_qc,
        candidate_indices,
    )
    trajectory_rows = [
        {
            "window_start_recording_seconds": _round(starts_absolute[index] / rate),
            "window_start_relative_to_anchor_seconds": _round(
                (starts_absolute[index] - anchor_sample) / rate
            ),
            "global_change_score": _round(global_scores[index]),
            "active_channel_count": int(active_channels[index]),
            "top_channels": [
                COMMON17_CHANNELS[channel_index]
                for channel_index in sorted(
                    range(len(COMMON17_CHANNELS)),
                    key=lambda channel_index: (
                        -float(scores[index, channel_index]),
                        COMMON17_CHANNELS[channel_index],
                    ),
                )[:4]
            ],
        }
        for index in range(starts_absolute.size)
    ]
    status = "qualified_change_candidate" if onset_sample is not None else "no_sustained_change_candidate"
    serializable = {
        "status": status,
        "window_count": int(starts_absolute.size),
        "window_seconds": policy.window_seconds,
        "step_seconds": policy.step_seconds,
        "robust_matched_baseline": baseline_receipt,
        "onset_candidate": (
            {
                "recording_seconds": _round(onset_sample / rate),
                "relative_to_navigation_anchor_seconds": _round(onset_relative),
                "selection_rule": "earliest_persistent_multichannel_multifamily_change_v1",
                "clinical_onset_claim_authorized": False,
            }
            if onset_sample is not None
            else None
        ),
        "per_channel_evidence": per_channel,
        "earliest_field": (
            {
                "recording_seconds": _round(earliest_time / rate),
                "relative_to_navigation_anchor_seconds": _round(
                    (earliest_time - anchor_sample) / rate
                ),
                "channels": earliest_channels,
                "dominant_connected_component": dominant,
                "clinical_soz_claim_authorized": False,
            }
            if earliest_time is not None
            else None
        ),
        "spatial_connectivity": {
            "graph": "common17_induced_TCP20_electrode_graph",
            "components": components,
            "within_field_edges": graph_edges,
            "dominant_component_fraction": _round(connectivity),
            "connected_field": bool(earliest_channels and len(components) == 1),
        },
        "reference_stability": reference,
        "evolution": {
            "candidate_return_to_baseline_recording_seconds": (
                _round(recovery_sample / rate) if recovery_sample is not None else None
            ),
            "candidate_return_relative_to_onset_seconds": (
                _round((recovery_sample - onset_sample) / rate)
                if recovery_sample is not None and onset_sample is not None
                else None
            ),
            "posterior_saturation_similarity": (
                _round(saturation) if saturation is not None else None
            ),
            "later_channel_delays_seconds": [
                {
                    "channel": channel,
                    "delay_seconds": _round((sample - earliest_time) / rate),
                }
                for channel, sample in channel_onsets.items()
                if sample is not None
                and earliest_time is not None
                and sample > earliest_time
            ],
            "trajectory_is_acns_evolution": False,
        },
        "native_primitives": primitive_summary,
        "change_trajectory": trajectory_rows,
    }
    return _EvidenceSnapshot(
        serializable=serializable,
        baseline_status=str(baseline_receipt["status"]),
        onset_sample=onset_sample,
        onset_relative_seconds=onset_relative,
        earliest_channel_samples=channel_onsets,
        recovery_sample=recovery_sample,
        support_start_sample=support_start,
        support_stop_sample=support_stop,
        onset_left_margin_seconds=(
            (onset_sample - support_start) / rate if onset_sample is not None else None
        ),
        posterior_saturation_similarity=saturation,
        reference_status=str(reference["status"]),
    )


def _side_decisions(
    snapshot: _EvidenceSnapshot,
    *,
    anchor_sample: int,
    recording_sample_count: int,
    current_extent_seconds: float,
    rate: float,
    policy: AdaptiveNativeEvidencePolicy,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    states = {"left": "open", "right": "open"}
    reasons = {"left": [], "right": []}
    if snapshot.support_start_sample == 0:
        states["left"] = "typed_censored"
        reasons["left"].append("recording_start")
    if snapshot.support_stop_sample == recording_sample_count:
        states["right"] = "typed_censored"
        reasons["right"].append("recording_stop")
    if snapshot.baseline_status != "qualified_robust_matched_baseline":
        reasons["left"].append("clean_matched_baseline_open")
    elif snapshot.onset_sample is None:
        reasons["left"].append("onset_transition_open")
    elif (
        snapshot.onset_left_margin_seconds is not None
        and snapshot.onset_left_margin_seconds <= policy.left_boundary_guard_seconds
    ):
        reasons["left"].append("onset_touches_left_support")
    else:
        if states["left"] == "open":
            states["left"] = "normal_closed"
        reasons["left"].append("baseline_and_preonset_margin_closed")

    if snapshot.onset_sample is None:
        reasons["right"].append("onset_transition_open")
    elif snapshot.recovery_sample is not None:
        if states["right"] == "open":
            states["right"] = "normal_closed"
        reasons["right"].append("postchange_recovery_closed")
    else:
        elapsed = (snapshot.support_stop_sample - snapshot.onset_sample) / rate
        saturated = (
            elapsed >= policy.minimum_evolution_seconds
            and snapshot.posterior_saturation_similarity is not None
            and snapshot.posterior_saturation_similarity
            >= policy.saturation_similarity_threshold
            and snapshot.reference_status == "stable"
            and current_extent_seconds >= 16.0
        )
        if saturated:
            if states["right"] == "open":
                states["right"] = "normal_closed"
            reasons["right"].append("stable_spatial_course_closed")
        else:
            reasons["right"].append("course_evidence_open")
    return states, reasons


def _merge_terminal_state(
    previous: Mapping[str, str],
    proposed: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for side in ("left", "right"):
        result[side] = previous[side] if previous[side] != "open" else proposed[side]
    return result


def materialize_common17_adaptive_native_event_evidence(
    *,
    event_id: str,
    recording_id: str,
    navigation_anchor_recording_seconds: float,
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader: NativeEEGQueryReader,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: AdaptiveNativeEvidencePolicy = DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY,
) -> dict[str, Any]:
    """Acquire and measure one event through actual q0/left/right EEG queries."""

    event = _identifier(event_id, "event_id")
    recording = _identifier(recording_id, "recording_id")
    rate = _finite(sampling_rate_hz, "sampling_rate_hz", minimum=10.0)
    if tuple(channel_order) != COMMON17_CHANNELS:
        raise ValueError(
            "adaptive native evidence requires exact common-17 order and never "
            "zero-fills/interpolates FZ/PZ"
        )
    if (
        isinstance(recording_sample_count, bool)
        or not isinstance(recording_sample_count, int)
        or recording_sample_count < int(round(2 * policy.window_seconds * rate))
    ):
        raise ValueError("recording_sample_count is invalid")
    if not callable(query_reader):
        raise TypeError("query_reader must be callable")
    anchor_seconds = _finite(
        navigation_anchor_recording_seconds,
        "navigation_anchor_recording_seconds",
        minimum=0.0,
    )
    anchor_sample = int(round(anchor_seconds * rate))
    if anchor_sample < 0 or anchor_sample > recording_sample_count:
        raise ValueError("navigation anchor lies outside the recording")

    chunks: list[_AcquiredChunk] = []
    trace: list[dict[str, Any]] = []
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    side_states = {"left": "open", "right": "open"}
    side_reasons = {"left": [], "right": []}
    current_extent = policy.q0_extent_seconds_each_side

    def execute_query(
        start_sample: int,
        stop_sample: int,
        *,
        side: str,
        target_extent_seconds: float,
    ) -> _EvidenceSnapshot:
        if not 0 <= start_sample < stop_sample <= recording_sample_count:
            raise ValueError("adaptive query interval is outside the recording")
        before = (
            []
            if not chunks
            else [
                _round(_assemble(chunks)[0] / rate),
                _round(_assemble(chunks)[1] / rate),
            ]
        )
        raw_result = query_reader(start_sample, stop_sample)
        signal, qc = _normalize_query_result(
            raw_result,
            expected_samples=stop_sample - start_sample,
        )
        chunk = _AcquiredChunk(
            start_sample=start_sample,
            stop_sample=stop_sample,
            signal_volts=signal,
            valid_sample_mask=qc,
            signal_sha256=_array_sha256(signal.astype("<f8", copy=False), prefix="common17-volts"),
            qc_sha256=_array_sha256(qc.astype(np.uint8), prefix="common17-eeg-qc"),
        )
        chunks.append(chunk)
        snapshot = _evaluate_support(
            chunks,
            anchor_sample=anchor_sample,
            rate=rate,
            policy=policy,
            morphology_cache=morphology_cache,
        )
        after = [
            _round(snapshot.support_start_sample / rate),
            _round(snapshot.support_stop_sample / rate),
        ]
        proposed_states, proposed_reasons = _side_decisions(
            snapshot,
            anchor_sample=anchor_sample,
            recording_sample_count=recording_sample_count,
            current_extent_seconds=target_extent_seconds,
            rate=rate,
            policy=policy,
        )
        nonlocal side_states, side_reasons
        side_states = _merge_terminal_state(side_states, proposed_states)
        for item in ("left", "right"):
            side_reasons[item] = sorted(
                set(side_reasons[item]).union(proposed_reasons[item])
            )
        snapshot_body = _jsonable(snapshot.serializable)
        trace.append(
            {
                "query_index": len(trace),
                "action": {
                    "side": side,
                    "target_extent_seconds_from_anchor": target_extent_seconds,
                    "interval_samples": [start_sample, stop_sample],
                    "interval_recording_seconds": [
                        _round(start_sample / rate),
                        _round(stop_sample / rate),
                    ],
                },
                "support_before_recording_seconds": before,
                "support_after_recording_seconds": after,
                "native_samples_revealed": int(signal.size),
                "physical_seconds_revealed": _round((stop_sample - start_sample) / rate),
                "raw_eeg_chunk_sha256": chunk.signal_sha256,
                "eeg_qc_chunk_sha256": chunk.qc_sha256,
                "full_native_remeasurement_used": True,
                "evidence_snapshot_sha256": _canonical_sha256(snapshot_body),
                "evidence_status_after": snapshot.serializable["status"],
                "side_states_after": deepcopy(side_states),
                "decision": (
                    "continue" if "open" in side_states.values() else "stop"
                ),
            }
        )
        return snapshot

    q0_radius = int(round(policy.q0_extent_seconds_each_side * rate))
    q0_start = max(0, anchor_sample - q0_radius)
    q0_stop = min(recording_sample_count, anchor_sample + q0_radius)
    if q0_stop <= q0_start:
        raise ValueError("q0 support is empty")
    snapshot = execute_query(
        q0_start,
        q0_stop,
        side="q0_bilateral",
        target_extent_seconds=policy.q0_extent_seconds_each_side,
    )

    for extent in policy.expansion_extents_seconds_each_side:
        current_extent = extent
        target_radius = int(round(extent * rate))
        if side_states["left"] == "open":
            support_start = _assemble(chunks)[0]
            target_start = max(0, anchor_sample - target_radius)
            if target_start < support_start:
                snapshot = execute_query(
                    target_start,
                    support_start,
                    side="left",
                    target_extent_seconds=extent,
                )
            elif support_start == 0:
                side_states["left"] = "typed_censored"
                side_reasons["left"] = sorted(
                    set(side_reasons["left"]).union({"recording_start"})
                )
        if side_states["right"] == "open":
            support_stop = _assemble(chunks)[1]
            target_stop = min(recording_sample_count, anchor_sample + target_radius)
            if target_stop > support_stop:
                snapshot = execute_query(
                    support_stop,
                    target_stop,
                    side="right",
                    target_extent_seconds=extent,
                )
            elif support_stop == recording_sample_count:
                side_states["right"] = "typed_censored"
                side_reasons["right"] = sorted(
                    set(side_reasons["right"]).union({"recording_stop"})
                )
        if "open" not in side_states.values():
            break

    for side in ("left", "right"):
        if side_states[side] == "open":
            side_states[side] = "typed_censored"
            side_reasons[side] = sorted(
                set(side_reasons[side]).union({"search_cap_32s"})
            )
    trace[-1]["side_states_after"] = deepcopy(side_states)
    trace[-1]["decision"] = "stop"
    final_start, final_stop, final_signal, final_qc = _assemble(chunks)
    final_snapshot = _evaluate_support(
        chunks,
        anchor_sample=anchor_sample,
        rate=rate,
        policy=policy,
        morphology_cache=morphology_cache,
    )
    censored = {
        side: {
            "state": side_states[side],
            "reason_codes": [
                reason for reason in side_reasons[side] if reason in _ALLOWED_CENSORS
            ],
            "closure_evidence_codes": [
                reason for reason in side_reasons[side] if reason not in _ALLOWED_CENSORS
            ],
        }
        for side in ("left", "right")
    }
    if final_snapshot.onset_sample is not None:
        status = "qualified_scalp_change_candidate"
    elif final_snapshot.baseline_status != "qualified_robust_matched_baseline":
        status = "unresolved_baseline_censored"
    else:
        status = "unresolved_no_sustained_change"
    body: dict[str, Any] = {
        "schema_version": ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID,
        "policy": _jsonable(policy.to_dict()),
        "policy_sha256": policy.sha256,
        "event_id": event,
        "recording_id": recording,
        "status": status,
        "acquisition": {
            "sampling_rate_hz": rate,
            "recording_sample_count": recording_sample_count,
            "recording_duration_seconds": _round(recording_sample_count / rate),
            "channel_order": list(COMMON17_CHANNELS),
            "signal_unit": "V",
            "removed_channels": ["FZ", "PZ"],
            "missing_channel_imputation_used": False,
        },
        "navigation_anchor_recording_seconds": _round(anchor_sample / rate),
        "query_trace": trace,
        "final_variable_support": {
            "interval_samples": [final_start, final_stop],
            "interval_recording_seconds": [
                _round(final_start / rate),
                _round(final_stop / rate),
            ],
            "interval_relative_to_anchor_seconds": [
                _round((final_start - anchor_sample) / rate),
                _round((final_stop - anchor_sample) / rate),
            ],
            "left_extent_seconds": _round((anchor_sample - final_start) / rate),
            "right_extent_seconds": _round((final_stop - anchor_sample) / rate),
            "side_closure": censored,
            "unique_physical_samples_per_channel": int(final_stop - final_start),
            "query_count": len(trace),
            "full_recording_preloaded": False,
        },
        "final_evidence": _jsonable(final_snapshot.serializable),
        "source_bindings": {
            "final_acquired_eeg_sha256": _array_sha256(
                final_signal.astype("<f8", copy=False), prefix="common17-final-volts"
            ),
            "final_eeg_qc_sha256": _array_sha256(
                final_qc.astype(np.uint8), prefix="common17-final-eeg-qc"
            ),
            "query_chunk_sha256s": [row["raw_eeg_chunk_sha256"] for row in trace],
        },
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    body = _jsonable(body)  # type: ignore[assignment]
    assert isinstance(body, dict)
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_common17_adaptive_native_event_evidence(body)


def validate_common17_adaptive_native_event_evidence(payload: object) -> dict[str, Any]:
    """Validate structural, query-continuity, scope and content-hash invariants."""

    if type(payload) is not dict:
        raise TypeError("adaptive native event evidence must be an object")
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "policy",
        "policy_sha256",
        "event_id",
        "recording_id",
        "status",
        "acquisition",
        "navigation_anchor_recording_seconds",
        "query_trace",
        "final_variable_support",
        "final_evidence",
        "source_bindings",
        "scope_receipt",
        "authorization",
    }
    if set(payload) != required:
        raise ValueError("adaptive native event evidence fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("adaptive native evidence schema drifted")
    if data["method_id"] != ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID:
        raise ValueError("adaptive native evidence method drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    if data["policy_sha256"] != _canonical_sha256(data["policy"]):
        raise ValueError("adaptive native evidence policy hash mismatch")
    acquisition = data["acquisition"]
    if not isinstance(acquisition, dict) or acquisition.get("channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("adaptive native evidence is not exact common-17")
    if acquisition.get("removed_channels") != ["FZ", "PZ"] or acquisition.get(
        "missing_channel_imputation_used"
    ) is not False:
        raise ValueError("adaptive native evidence silently imputed removed channels")
    rate = _finite(acquisition.get("sampling_rate_hz"), "sampling_rate_hz", minimum=10.0)
    samples = acquisition.get("recording_sample_count")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("recording sample count is invalid")
    trace = data["query_trace"]
    if not isinstance(trace, list) or not trace:
        raise ValueError("adaptive native evidence lacks a query trace")
    previous_after: list[float] | None = None
    for index, row in enumerate(trace):
        if not isinstance(row, dict) or row.get("query_index") != index:
            raise ValueError("adaptive query indices are not contiguous")
        action = row.get("action")
        if not isinstance(action, dict) or action.get("side") not in {
            "q0_bilateral",
            "left",
            "right",
        }:
            raise ValueError("adaptive query action is invalid")
        if index == 0 and action["side"] != "q0_bilateral":
            raise ValueError("adaptive query trace does not begin with q0")
        if index > 0 and action["side"] == "q0_bilateral":
            raise ValueError("q0 appears more than once")
        extent = float(action.get("target_extent_seconds_from_anchor"))
        if extent not in {4.0, 8.0, 16.0, 32.0}:
            raise ValueError("adaptive query extent escaped q0/8/16/32")
        interval = action.get("interval_samples")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in interval)
            or not 0 <= interval[0] < interval[1] <= samples
        ):
            raise ValueError("adaptive query sample interval is invalid")
        before = row.get("support_before_recording_seconds")
        after = row.get("support_after_recording_seconds")
        if index == 0:
            if before != []:
                raise ValueError("q0 support-before must be empty")
        elif before != previous_after:
            raise ValueError("adaptive query support is not trajectory-contiguous")
        if not isinstance(after, list) or len(after) != 2 or after[1] <= after[0]:
            raise ValueError("adaptive support-after interval is invalid")
        if row.get("full_native_remeasurement_used") is not True:
            raise ValueError("adaptive query skipped full native remeasurement")
        for field in ("raw_eeg_chunk_sha256", "eeg_qc_chunk_sha256", "evidence_snapshot_sha256"):
            digest = row.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"adaptive query {field} is invalid")
        previous_after = after
    if trace[-1].get("decision") != "stop":
        raise ValueError("adaptive query trace lacks a terminal stop")
    support = data["final_variable_support"]
    if not isinstance(support, dict) or support.get("interval_recording_seconds") != previous_after:
        raise ValueError("final variable support differs from the query trajectory")
    if support.get("full_recording_preloaded") is not False:
        raise ValueError("adaptive native evidence claims incremental retrieval incorrectly")
    closures = support.get("side_closure")
    if not isinstance(closures, dict) or set(closures) != {"left", "right"}:
        raise ValueError("adaptive side closure is invalid")
    for side, closure in closures.items():
        if closure.get("state") not in {"normal_closed", "typed_censored"}:
            raise ValueError(f"adaptive {side} side did not close")
        censor_reasons = closure.get("reason_codes")
        if not isinstance(censor_reasons, list) or not set(censor_reasons) <= _ALLOWED_CENSORS:
            raise ValueError(f"adaptive {side} censor reasons are invalid")
        if closure["state"] == "typed_censored" and not censor_reasons:
            raise ValueError(f"adaptive {side} typed censor lacks a reason")
        if closure["state"] == "normal_closed" and censor_reasons:
            raise ValueError(f"adaptive {side} normal closure carries censor reasons")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("adaptive native evidence violates EEG-only scope")
    if data["authorization"] != _AUTHORIZATION:
        raise ValueError("adaptive native evidence authorization drifted")
    final_evidence = data["final_evidence"]
    if not isinstance(final_evidence, dict):
        raise ValueError("adaptive native final evidence is invalid")
    channel_rows = final_evidence.get("per_channel_evidence", [])
    if channel_rows:
        observed = {row.get("channel") for row in channel_rows if isinstance(row, dict)}
        if observed != set(COMMON17_CHANNELS) or len(channel_rows) != len(COMMON17_CHANNELS):
            raise ValueError("per-channel evidence does not exactly cover common-17")
        mass = sum(float(row["onset_spatial_posterior_mass"]) for row in channel_rows)
        if abs(mass - 1.0) > 2e-4:
            raise ValueError("within-event channel posterior mass does not close")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("adaptive native event evidence content hash mismatch")
    return data


__all__ = [
    "ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID",
    "ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION",
    "COMMON17_CHANNELS",
    "COMMON17_TCP_EDGES",
    "DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY",
    "AdaptiveNativeEvidencePolicy",
    "NativeEEGQueryChunk",
    "NativeEEGQueryReader",
    "materialize_common17_adaptive_native_event_evidence",
    "validate_common17_adaptive_native_event_evidence",
]
