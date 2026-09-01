"""Shared pure numerical kernel for BA-IEG signal measurements.

This module owns the twelve pointwise measurements common to the P0 token
input and the dense deterministic supervision sidecar.  It has no EDF, event
selection, annotation, spreadsheet, label, clinical-term, or SOZ API.

The thirteenth dense target (background-referenced robust change) and the P0
previous-tile change input are deliberately excluded: they have different
reference populations and therefore are not the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

import numpy as np


BA_IEG_BASE_NUMERICAL_KERNEL_ID: Final[str] = (
    "ba_ieg_shared_base_numerical_kernel_v1"
)

BA_IEG_BASE_MEASUREMENT_NAMES: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
    "dominant_frequency_hz",
    "spectral_concentration",
    "spectral_entropy",
    "rhythmicity_index",
    "delta_power_ratio",
    "theta_power_ratio",
    "alpha_power_ratio",
    "beta_power_ratio",
    "low_gamma_power_ratio",
)

BA_IEG_BASE_BAND_TARGETS: Final[
    tuple[tuple[str, float, float], ...]
] = (
    ("delta_power_ratio", 0.5, 4.0),
    ("theta_power_ratio", 4.0, 8.0),
    ("alpha_power_ratio", 8.0, 13.0),
    ("beta_power_ratio", 13.0, 30.0),
    ("low_gamma_power_ratio", 30.0, 45.0),
)

_INDEX = {
    name: index for index, name in enumerate(BA_IEG_BASE_MEASUREMENT_NAMES)
}
_AMPLITUDE_INDICES = tuple(range(0, 3))
_SPECTRAL_INDICES = tuple(range(3, len(BA_IEG_BASE_MEASUREMENT_NAMES)))
_TOL = 1e-8


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _reasons(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted(set(str(item) for item in values)))
    if any(not item or item != item.strip() for item in result):
        raise ValueError("numerical-kernel reason codes must be non-empty")
    return result


@dataclass(frozen=True)
class BAIEGBaseNumericalPolicy:
    """Policy inputs needed by the shared twelve-measurement kernel."""

    analysis_low_hz: float = 0.5
    analysis_high_hz: float = 45.0
    minimum_spectral_bins: int = 3
    spectral_power_floor_uv2: float = 1e-12

    def __post_init__(self) -> None:
        low = _finite(self.analysis_low_hz, "analysis_low_hz", minimum=0.0)
        high = _finite(
            self.analysis_high_hz, "analysis_high_hz", minimum=_TOL
        )
        floor = _finite(
            self.spectral_power_floor_uv2,
            "spectral_power_floor_uv2",
            minimum=0.0,
        )
        if high <= low + _TOL:
            raise ValueError("shared numerical analysis band is empty")
        if (
            isinstance(self.minimum_spectral_bins, bool)
            or not isinstance(self.minimum_spectral_bins, int)
            or self.minimum_spectral_bins < 3
        ):
            raise ValueError("minimum_spectral_bins must be an integer >= 3")
        if floor <= 0.0:
            raise ValueError("spectral_power_floor_uv2 must be positive")
        object.__setattr__(self, "analysis_low_hz", low)
        object.__setattr__(self, "analysis_high_hz", high)
        object.__setattr__(self, "spectral_power_floor_uv2", floor)

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": BA_IEG_BASE_NUMERICAL_KERNEL_ID,
            "analysis_low_hz": self.analysis_low_hz,
            "analysis_high_hz": self.analysis_high_hz,
            "minimum_spectral_bins": self.minimum_spectral_bins,
            "spectral_power_floor_uv2": self.spectral_power_floor_uv2,
            "centering": "median",
            "taper": "numpy_hanning_symmetric_v1",
            "measurement_names": list(BA_IEG_BASE_MEASUREMENT_NAMES),
        }


@dataclass(frozen=True)
class BAIEGBaseNumericalResult:
    """Immutable values, per-value opportunity masks and reason codes."""

    values: tuple[float, ...]
    value_mask: tuple[bool, ...]
    reason_codes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        size = len(BA_IEG_BASE_MEASUREMENT_NAMES)
        if not (
            len(self.values) == len(self.value_mask) == len(self.reason_codes) == size
        ):
            raise ValueError("shared numerical result vocabulary drifted")
        normalized_values: list[float] = []
        normalized_reasons: list[tuple[str, ...]] = []
        for index, raw_value in enumerate(self.values):
            value = _finite(raw_value, f"values[{index}]")
            reasons = _reasons(self.reason_codes[index])
            available = bool(self.value_mask[index])
            if available == bool(reasons):
                raise ValueError(
                    "available measurements need no reason; masked values need one"
                )
            if not available and value != 0.0:
                raise ValueError("masked shared numerical values must be zero")
            normalized_values.append(value)
            normalized_reasons.append(reasons)
        object.__setattr__(self, "values", tuple(normalized_values))
        object.__setattr__(self, "value_mask", tuple(bool(v) for v in self.value_mask))
        object.__setattr__(self, "reason_codes", tuple(normalized_reasons))


def measure_ba_ieg_base_numerical_features(
    signal_volts: Sequence[float] | np.ndarray,
    *,
    sampling_rate_hz: float,
    effective_bandwidth_hz: Sequence[float],
    policy: BAIEGBaseNumericalPolicy,
    amplitude_reason_codes: Sequence[str] = (),
    spectral_reason_codes: Sequence[str] = (),
) -> BAIEGBaseNumericalResult:
    """Measure one physical signal interval without clinical interpretation."""

    if not isinstance(policy, BAIEGBaseNumericalPolicy):
        raise TypeError("policy must be BAIEGBaseNumericalPolicy")
    sfreq = _finite(sampling_rate_hz, "sampling_rate_hz", minimum=_TOL)
    if len(effective_bandwidth_hz) != 2:
        raise ValueError("effective_bandwidth_hz must contain two values")
    effective_low = _finite(
        effective_bandwidth_hz[0], "effective_bandwidth_hz[0]", minimum=0.0
    )
    effective_high = _finite(
        effective_bandwidth_hz[1], "effective_bandwidth_hz[1]", minimum=_TOL
    )
    if effective_high <= effective_low + _TOL or effective_high > 0.5 * sfreq + _TOL:
        raise ValueError("effective bandwidth is invalid on the supplied clock")
    values = np.asarray(signal_volts, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("shared numerical signal must be finite")

    output = np.zeros(len(BA_IEG_BASE_MEASUREMENT_NAMES), dtype=np.float64)
    masks = np.zeros(len(BA_IEG_BASE_MEASUREMENT_NAMES), dtype=bool)
    reasons: list[list[str]] = [
        [] for _ in BA_IEG_BASE_MEASUREMENT_NAMES
    ]
    amplitude_reasons = list(_reasons(amplitude_reason_codes))
    spectral_reasons = list(_reasons(spectral_reason_codes))
    for index in _AMPLITUDE_INDICES:
        reasons[index].extend(amplitude_reasons)
        masks[index] = not reasons[index]
    for index in _SPECTRAL_INDICES:
        reasons[index].extend(spectral_reasons)
        masks[index] = not reasons[index]

    def mask(index: int, reason: str) -> None:
        if reason not in reasons[index]:
            reasons[index].append(reason)
        masks[index] = False
        output[index] = 0.0

    if values.size < 4:
        for index in _AMPLITUDE_INDICES + _SPECTRAL_INDICES:
            if masks[index]:
                mask(index, "measurement_window_too_short")
    else:
        centered_uv = (values - np.median(values)) * 1.0e6
        if masks[_INDEX["rms_uv"]]:
            output[_INDEX["rms_uv"]] = math.sqrt(
                float(np.mean(centered_uv**2))
            )
        if masks[_INDEX["peak_to_peak_uv"]]:
            output[_INDEX["peak_to_peak_uv"]] = float(np.ptp(centered_uv))
        if masks[_INDEX["line_length_uv_per_sample"]]:
            output[_INDEX["line_length_uv_per_sample"]] = float(
                np.mean(np.abs(np.diff(centered_uv)))
            )

        spectral_candidates = [
            index for index in _SPECTRAL_INDICES if masks[index]
        ]
        if spectral_candidates:
            analysis_low = max(policy.analysis_low_hz, effective_low)
            analysis_high = min(policy.analysis_high_hz, effective_high)
            frequencies = np.fft.rfftfreq(centered_uv.size, d=1.0 / sfreq)
            analysis_mask = (frequencies >= analysis_low - _TOL) & (
                frequencies <= analysis_high + _TOL
            )
            if (
                analysis_high <= analysis_low + _TOL
                or np.count_nonzero(analysis_mask)
                < policy.minimum_spectral_bins
            ):
                for index in spectral_candidates:
                    mask(index, "insufficient_effective_spectral_bandwidth")
            else:
                taper = np.hanning(centered_uv.size)
                power = np.abs(np.fft.rfft(centered_uv * taper)) ** 2
                local_power = power[analysis_mask]
                local_frequencies = frequencies[analysis_mask]
                total_power = float(np.sum(local_power))
                if total_power <= policy.spectral_power_floor_uv2:
                    for index in spectral_candidates:
                        mask(index, "spectral_energy_below_floor")
                else:
                    peak_index = int(np.argmax(local_power))
                    dominant = float(local_frequencies[peak_index])
                    probabilities = local_power / total_power
                    output[_INDEX["dominant_frequency_hz"]] = dominant
                    output[_INDEX["spectral_concentration"]] = float(
                        local_power[peak_index] / total_power
                    )
                    output[_INDEX["spectral_entropy"]] = float(
                        -np.sum(
                            probabilities * np.log(probabilities + 1e-12)
                        )
                        / math.log(max(2, probabilities.size))
                    )
                    lag = int(round(sfreq / max(dominant, _TOL)))
                    rhythmicity_index = _INDEX["rhythmicity_index"]
                    if 1 <= lag < centered_uv.size // 2:
                        left = centered_uv[:-lag]
                        right = centered_uv[lag:]
                        denominator = math.sqrt(
                            float(np.sum(left**2) * np.sum(right**2))
                        )
                        if denominator > policy.spectral_power_floor_uv2:
                            output[rhythmicity_index] = float(
                                np.clip(
                                    np.sum(left * right) / denominator,
                                    -1.0,
                                    1.0,
                                )
                            )
                        else:
                            mask(
                                rhythmicity_index,
                                "rhythmicity_denominator_below_floor",
                            )
                    else:
                        mask(
                            rhythmicity_index,
                            "dominant_period_not_resolved",
                        )

                    for target_name, band_low, band_high in BA_IEG_BASE_BAND_TARGETS:
                        target_index = _INDEX[target_name]
                        if not masks[target_index]:
                            continue
                        if (
                            effective_low > band_low + _TOL
                            or effective_high < band_high - _TOL
                        ):
                            mask(
                                target_index,
                                "target_band_not_fully_observable",
                            )
                            continue
                        band_mask = (frequencies >= band_low - _TOL) & (
                            frequencies < band_high - _TOL
                        )
                        if target_name == "low_gamma_power_ratio":
                            band_mask = (frequencies >= band_low - _TOL) & (
                                frequencies <= band_high + _TOL
                            )
                        if not np.any(band_mask):
                            mask(target_index, "target_band_has_no_fft_bins")
                            continue
                        output[target_index] = float(
                            np.sum(power[band_mask]) / total_power
                        )

    for index in range(len(output)):
        reasons[index] = list(_reasons(reasons[index]))
        if masks[index] and reasons[index]:
            raise ValueError("available shared value carries a reason")
        if not masks[index] and not reasons[index]:
            reasons[index].append("measurement_unavailable")
        if not masks[index]:
            output[index] = 0.0
    return BAIEGBaseNumericalResult(
        values=tuple(float(value) for value in output),
        value_mask=tuple(bool(value) for value in masks),
        reason_codes=tuple(tuple(items) for items in reasons),
    )


__all__ = [
    "BA_IEG_BASE_BAND_TARGETS",
    "BA_IEG_BASE_MEASUREMENT_NAMES",
    "BA_IEG_BASE_NUMERICAL_KERNEL_ID",
    "BAIEGBaseNumericalPolicy",
    "BAIEGBaseNumericalResult",
    "measure_ba_ieg_base_numerical_features",
]
