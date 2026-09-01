"""Target-free signal-quality port for the score-preserving MRSC layer.

The detector operates only on an already materialized standard-19 signal in
physical volts.  It does not repair the signal, alter a localization score,
or emit a clinical artifact diagnosis.  Its sole deployment role is to mark
candidate inputs as quality-valid/invalid and to provide a conservative
uncertainty value for review or abstention.

Thresholds are frozen from ``configs/preprocess_qc.yaml``.  Both 50 and 60 Hz
are checked because the source corpus and future deployment sites need not
share a mains frequency.  The synthetic corruption helpers are deliberately
kept in this module so the qualification runner and unit tests exercise the
same fixed recipes; they are never called by clinical inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping

import numpy as np

from src.preprocessing.artifact_scoring import score_high, score_low

from .geometry import STANDARD_19
from .v11_reasoner import V11_CANDIDATE_INDICES


MRSC_SIGNAL_QUALITY_SCHEMA: Final[str] = "soz_mrsc_signal_quality_port_v1"
MRSC_SIGNAL_QUALITY_USE_POLICY: Final[str] = (
    "quality_review_or_abstention_only_never_soz_scoring_or_artifact_diagnosis"
)
MRSC_QUALITY_STRESS_KINDS: Final[tuple[str, ...]] = (
    "flatline",
    "clipping",
    "line_noise",
    "emg_like_broadband",
    "high_amplitude_transient",
)
MRSC_QUALITY_COMPONENTS: Final[tuple[str, ...]] = (
    "flatline",
    "saturation_or_clipping",
    "line_noise",
    "high_frequency_burden",
    "high_amplitude_transient",
)
MRSC_QUALITY_CANDIDATE_CHANNELS: Final[tuple[str, ...]] = tuple(
    STANDARD_19[index] for index in V11_CANDIDATE_INDICES
)


def _require_thresholds(config: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("quality config must be a mapping")
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("quality config requires a thresholds mapping")
    required = {
        "flatline_std",
        "flatline_std_severe",
        "flatline_ptp",
        "flatline_ptp_severe",
        "flatline_fraction",
        "flatline_fraction_severe",
        "saturation_amplitude",
        "saturation_amplitude_severe",
        "line_noise_ratio",
        "line_noise_ratio_severe",
        "muscle_hf_ratio",
        "muscle_hf_ratio_severe",
        "electrode_pop_threshold",
        "electrode_pop_threshold_severe",
        "electrode_pop_abs_jump",
        "electrode_pop_abs_jump_severe",
        "component_unusable_score",
    }
    if not required.issubset(thresholds):
        missing = sorted(required - set(thresholds))
        raise ValueError(f"quality config is missing frozen thresholds: {missing}")
    return thresholds


def _validate_signal(signal_volts: np.ndarray, sfreq_hz: float) -> np.ndarray:
    values = np.asarray(signal_volts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(STANDARD_19):
        raise ValueError("signal_volts must have shape [19,T]")
    if values.shape[1] < 2:
        raise ValueError("signal_volts must contain at least two samples")
    if not np.isfinite(values).all():
        raise ValueError("signal_volts must be finite")
    if (
        isinstance(sfreq_hz, bool)
        or not isinstance(sfreq_hz, (int, float))
        or not math.isfinite(float(sfreq_hz))
        or float(sfreq_hz) < 100.0
    ):
        raise ValueError("sfreq_hz must be finite and at least 100 Hz")
    return np.ascontiguousarray(values)


def _max_flat_fraction(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 1.0
    differences = np.abs(np.diff(samples))
    scale = max(float(np.percentile(np.abs(samples), 95)), 1e-12)
    flat = differences <= scale * 1e-6
    padded = np.concatenate(([False], flat, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    longest = int(np.max(stops - starts)) if starts.size else 0
    return float(longest / max(samples.size - 1, 1))


def _band_ratio(
    samples: np.ndarray,
    *,
    sfreq_hz: float,
    low_hz: float,
    high_hz: float,
    total_low_hz: float = 1.0,
    total_high_hz: float = 90.0,
) -> float:
    centered = samples - float(np.mean(samples))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sfreq_hz)
    power = np.abs(np.fft.rfft(centered)) ** 2
    nyquist = sfreq_hz / 2.0
    total = (frequencies >= total_low_hz) & (
        frequencies <= min(total_high_hz, nyquist)
    )
    selected = (frequencies >= max(low_hz, 0.0)) & (
        frequencies <= min(high_hz, nyquist)
    )
    denominator = float(np.sum(power[total])) + 1e-24
    return float(np.clip(float(np.sum(power[selected])) / denominator, 0.0, 1.0))


def _window_slices(n_samples: int, sfreq_hz: float) -> tuple[slice, ...]:
    width = max(2, int(round(sfreq_hz)))
    slices: list[slice] = []
    for start in range(0, n_samples, width):
        stop = min(n_samples, start + width)
        if stop - start >= 2:
            slices.append(slice(start, stop))
    return tuple(slices)


@dataclass(frozen=True)
class MRSCSignalQualityAssessment:
    """Per-channel quality facts with a fixed candidate-validity projection."""

    component_scores: tuple[tuple[float, ...], ...]
    channel_uncertainty: tuple[float, ...]
    channel_valid: tuple[bool, ...]
    candidate_valid: tuple[bool, ...]
    signal_quality_uncertainty: float
    hard_invalid: bool
    reason_codes: tuple[str, ...]
    target_labels_used: bool = False
    private_data_used: bool = False
    localization_scores_used: bool = False
    training_performed: bool = False
    use_policy: str = MRSC_SIGNAL_QUALITY_USE_POLICY
    schema_version: str = MRSC_SIGNAL_QUALITY_SCHEMA

    def __post_init__(self) -> None:
        if len(self.component_scores) != len(STANDARD_19) or any(
            len(row) != len(MRSC_QUALITY_COMPONENTS)
            for row in self.component_scores
        ):
            raise ValueError("component_scores must have shape [19,5]")
        if len(self.channel_uncertainty) != len(STANDARD_19):
            raise ValueError("channel_uncertainty must have length 19")
        if len(self.channel_valid) != len(STANDARD_19):
            raise ValueError("channel_valid must have length 19")
        if len(self.candidate_valid) != len(MRSC_QUALITY_CANDIDATE_CHANNELS):
            raise ValueError("candidate_valid must have length 18")
        flat = tuple(value for row in self.component_scores for value in row)
        for value in (*flat, *self.channel_uncertainty, self.signal_quality_uncertainty):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("quality scores must lie in [0,1]")
        if any(type(value) is not bool for value in self.channel_valid):
            raise TypeError("channel_valid values must be bool")
        if any(type(value) is not bool for value in self.candidate_valid):
            raise TypeError("candidate_valid values must be bool")
        expected_candidates = tuple(
            self.channel_valid[index] for index in V11_CANDIDATE_INDICES
        )
        if self.candidate_valid != expected_candidates:
            raise ValueError("candidate_valid disagrees with standard-19 projection")
        expected_uncertainty = max(self.channel_uncertainty)
        if not math.isclose(
            float(self.signal_quality_uncertainty),
            float(expected_uncertainty),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("signal quality uncertainty must be the worst channel")
        if type(self.hard_invalid) is not bool:
            raise TypeError("hard_invalid must be bool")
        if self.hard_invalid != (not all(self.candidate_valid)):
            raise ValueError("hard_invalid must replay from candidate validity")
        if (
            not isinstance(self.reason_codes, tuple)
            or len(set(self.reason_codes)) != len(self.reason_codes)
        ):
            raise ValueError("reason_codes must be a unique tuple")
        if self.hard_invalid != bool(self.reason_codes):
            raise ValueError("hard-invalid quality needs explicit reason codes")
        for name in (
            "target_labels_used",
            "private_data_used",
            "localization_scores_used",
            "training_performed",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
            if value:
                raise ValueError("MRSC signal quality must remain target/private/score free")
        if self.use_policy != MRSC_SIGNAL_QUALITY_USE_POLICY:
            raise ValueError("Unsupported MRSC signal-quality use policy")
        if self.schema_version != MRSC_SIGNAL_QUALITY_SCHEMA:
            raise ValueError("Unsupported MRSC signal-quality schema")


def assess_mrsc_signal_quality(
    signal_volts: np.ndarray,
    sfreq_hz: float,
    config: Mapping[str, object],
) -> MRSCSignalQualityAssessment:
    """Assess five observable quality components without changing the signal."""

    values = _validate_signal(signal_volts, sfreq_hz)
    thresholds = _require_thresholds(config)
    windows = _window_slices(values.shape[1], float(sfreq_hz))
    rows: list[tuple[float, ...]] = []
    for channel in values:
        per_window: list[tuple[float, ...]] = []
        for window in windows:
            samples = channel[window]
            differences = np.diff(samples)
            difference_median = float(np.median(differences))
            difference_mad = max(
                float(np.median(np.abs(differences - difference_median))), 1e-12
            )
            flatline = float(
                np.max(
                    np.maximum.reduce(
                        (
                            score_low(
                                np.std(samples),
                                float(thresholds["flatline_std"]),
                                float(thresholds["flatline_std_severe"]),
                            ),
                            score_low(
                                np.ptp(samples),
                                float(thresholds["flatline_ptp"]),
                                float(thresholds["flatline_ptp_severe"]),
                            ),
                            score_high(
                                _max_flat_fraction(samples),
                                float(thresholds["flatline_fraction"]),
                                float(thresholds["flatline_fraction_severe"]),
                            ),
                        )
                    )
                )
            )
            saturation = float(
                score_high(
                    np.max(np.abs(samples)),
                    float(thresholds["saturation_amplitude"]),
                    float(thresholds["saturation_amplitude_severe"]),
                )
            )
            line_ratio = max(
                _band_ratio(
                    samples,
                    sfreq_hz=float(sfreq_hz),
                    low_hz=line_frequency - 1.0,
                    high_hz=line_frequency + 1.0,
                )
                for line_frequency in (50.0, 60.0)
                if line_frequency - 1.0 < float(sfreq_hz) / 2.0
            )
            line_noise = float(
                score_high(
                    line_ratio,
                    float(thresholds["line_noise_ratio"]),
                    float(thresholds["line_noise_ratio_severe"]),
                )
            )
            high_frequency = float(
                score_high(
                    _band_ratio(
                        samples,
                        sfreq_hz=float(sfreq_hz),
                        low_hz=30.0,
                        high_hz=80.0,
                    ),
                    float(thresholds["muscle_hf_ratio"]),
                    float(thresholds["muscle_hf_ratio_severe"]),
                )
            )
            derivative_z = float(
                np.max(
                    np.abs(
                        (differences - difference_median)
                        / (1.4826 * difference_mad)
                    )
                )
            )
            transient = float(
                max(
                    score_high(
                        derivative_z,
                        float(thresholds["electrode_pop_threshold"]),
                        float(thresholds["electrode_pop_threshold_severe"]),
                    ),
                    score_high(
                        np.max(np.abs(differences)),
                        float(thresholds["electrode_pop_abs_jump"]),
                        float(thresholds["electrode_pop_abs_jump_severe"]),
                    ),
                )
            )
            per_window.append(
                (flatline, saturation, line_noise, high_frequency, transient)
            )
        rows.append(
            tuple(float(max(values_)) for values_ in zip(*per_window, strict=True))
        )
    uncertainty = tuple(max(row) for row in rows)
    cutoff = float(thresholds["component_unusable_score"])
    valid = tuple(value < cutoff for value in uncertainty)
    candidates = tuple(valid[index] for index in V11_CANDIDATE_INDICES)
    invalid_channels = tuple(
        channel
        for channel, is_valid in zip(MRSC_QUALITY_CANDIDATE_CHANNELS, candidates)
        if not is_valid
    )
    reasons = (
        ("candidate_signal_quality_invalid",) if invalid_channels else ()
    )
    return MRSCSignalQualityAssessment(
        component_scores=tuple(rows),
        channel_uncertainty=uncertainty,
        channel_valid=valid,
        candidate_valid=candidates,
        signal_quality_uncertainty=max(uncertainty),
        hard_invalid=bool(invalid_channels),
        reason_codes=reasons,
    )


def inject_mrsc_quality_stress(
    clean_signal_volts: np.ndarray,
    sfreq_hz: float,
    *,
    channel_index: int,
    corruption: str,
) -> np.ndarray:
    """Apply one frozen source-only stress recipe to a detached signal copy."""

    values = _validate_signal(clean_signal_volts, sfreq_hz).copy()
    if (
        isinstance(channel_index, bool)
        or not isinstance(channel_index, int)
        or not 0 <= channel_index < len(STANDARD_19)
    ):
        raise ValueError("channel_index is outside standard-19")
    if corruption not in MRSC_QUALITY_STRESS_KINDS:
        raise ValueError("Unsupported MRSC quality corruption")
    samples = values[channel_index]
    times = np.arange(samples.size, dtype=np.float64) / float(sfreq_hz)
    if corruption == "flatline":
        samples[:] = float(np.median(samples))
    elif corruption == "clipping":
        width = min(samples.size, max(2, int(round(0.75 * sfreq_hz))))
        start = max(0, (samples.size - width) // 2)
        samples[start : start + width] = 0.012
    elif corruption == "line_noise":
        samples += 0.002 * np.sin(2.0 * math.pi * 50.0 * times)
    elif corruption == "emg_like_broadband":
        # Fixed multi-sine broadband burden avoids a random-seed degree of
        # freedom while spanning the 30--80 Hz observable band.
        for frequency, phase in zip(
            (33.0, 41.0, 53.0, 67.0, 77.0),
            (0.1, 0.7, 1.3, 1.9, 2.5),
            strict=True,
        ):
            samples += 0.0008 * np.sin(2.0 * math.pi * frequency * times + phase)
    elif corruption == "high_amplitude_transient":
        position = samples.size // 2
        samples[position] += 0.012
    if not np.isfinite(values).all():
        raise RuntimeError("Synthetic quality stress produced a non-finite signal")
    return values


__all__ = [
    "MRSC_QUALITY_CANDIDATE_CHANNELS",
    "MRSC_QUALITY_COMPONENTS",
    "MRSC_QUALITY_STRESS_KINDS",
    "MRSC_SIGNAL_QUALITY_SCHEMA",
    "MRSC_SIGNAL_QUALITY_USE_POLICY",
    "MRSCSignalQualityAssessment",
    "assess_mrsc_signal_quality",
    "inject_mrsc_quality_stress",
]
