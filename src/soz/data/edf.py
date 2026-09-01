"""Direct physical-EDF loading with causal, receipt-bearing preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, firwin, sosfilt, sosfreqz, upfirdn
import torch

from ..geometry import STANDARD_19, normalize_electrode_name
from ..models.labram import (
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    bind_labram_record_positions,
)
from ..signal import (
    EventEEGWindow,
    SignalProcessingReceipt,
    crop_event_window,
    select_standard19_physical,
)


EDF_PREPROCESS_SCHEMA = "standard19_causal_edf_event_v2"
CAUSAL_IIR_INITIAL_STATE_POLICY = (
    "zero_sos_state_reset_at_each_event_read_start_no_cross_event_carry_v1"
)
CAUSAL_IIR_PHASE_POLICY = (
    "frequency_dependent_group_delay_recorded_no_scalar_delay_correction_v1"
)
CAUSAL_IIR_GROUP_DELAY_ESTIMATOR = (
    "sosfreqz_unwrapped_phase_gradient_32769_dense_passband_grid_v1"
)
_IIR_GROUP_DELAY_GRID_SIZE = 17
_UNIT_TO_VOLTS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
_GAP_WORDS = ("boundary", "discont", "gap")
_EDF_EVENT_ELIGIBILITY_CODES = frozenset(
    {
        "ambiguous_standard19",
        "invalid_sfreq",
        "mixed_sfreq",
        "sample_count_mismatch",
        "insufficient_warmup",
        "insufficient_post",
        "payload_shape",
        "signal_qc",
        "reference_or_signal_contract",
    }
)


class EDFEventEligibilityError(ValueError):
    """Expected, auditable reason that one EDF event cannot enter training.

    Only failures caused by the source signal or the event's available time
    support use this exception.  Invalid configuration, missing files,
    unexpected reader failures, and internal invariant failures deliberately
    retain their original exception types so callers cannot silently omit
    them as if they were ordinary data attrition.
    """

    allowed_codes = _EDF_EVENT_ELIGIBILITY_CODES

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = str(code).strip()
        if normalized not in self.allowed_codes:
            raise ValueError(f"Unknown EDF event eligibility code: {code!r}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("EDF event eligibility message must be non-empty")
        if details is None:
            normalized_details: dict[str, Any] = {}
        elif type(details) is dict:
            normalized_details = dict(details)
        else:
            raise TypeError("EDF event eligibility details must be an object")
        self.code = normalized
        # Structured details are deliberately separate from exception prose so
        # callers can persist closed, signal-only reason receipts without ever
        # serializing a raw reader/path message.
        self.details = normalized_details
        super().__init__(message)


def _ineligible(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> EDFEventEligibilityError:
    return EDFEventEligibilityError(code, message, details=details)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Sample coordinate must be finite")
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _unit_scale(unit: object) -> float:
    normalized = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported EDF physical unit: {unit!r}") from exc


def _maximum_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _max_flatline_run_samples(signal_volts: np.ndarray, tolerance_volts: float) -> int:
    if signal_volts.size < 2:
        return int(signal_volts.size)
    unchanged = np.abs(np.diff(signal_volts)) <= float(tolerance_volts)
    return _maximum_true_run(unchanged) + 1 if unchanged.any() else 1


def _max_extreme_run_samples(signal_volts: np.ndarray, tolerance_volts: float) -> int:
    if signal_volts.size < 1 or not np.isfinite(signal_volts).all():
        return int(signal_volts.size)
    low = float(np.min(signal_volts))
    high = float(np.max(signal_volts))
    at_extreme = np.isclose(signal_volts, low, rtol=0.0, atol=tolerance_volts) | np.isclose(
        signal_volts, high, rtol=0.0, atol=tolerance_volts
    )
    return _maximum_true_run(at_extreme)


@dataclass(frozen=True)
class CausalEDFConfig:
    """Frozen primary preprocessing policy for public EDF event windows."""

    output_sfreq_hz: float = 200.0
    highpass_hz: float = 0.5
    lowpass_hz: float = 45.0
    butterworth_order: int = 4
    warmup_sec: float = 30.0
    pre_onset_sec: float = 12.0
    post_onset_sec: float = 48.0
    fir_half_length_per_rate: int = 10
    flatline_run_sec: float = 2.0
    clipping_run_sec: float = 0.5
    qc_tolerance_volts: float = 1e-12
    reference_policy: str = "primary_ref"
    sensitivity_reference: str | None = None
    apply_car19: bool = True

    def __post_init__(self) -> None:
        numeric_positive = (
            self.output_sfreq_hz,
            self.highpass_hz,
            self.lowpass_hz,
            self.warmup_sec,
            self.pre_onset_sec,
            self.post_onset_sec,
            self.flatline_run_sec,
            self.clipping_run_sec,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in numeric_positive):
            raise ValueError("EDF preprocessing frequencies/durations must be positive")
        if self.highpass_hz >= self.lowpass_hz:
            raise ValueError("highpass_hz must be below lowpass_hz")
        if self.butterworth_order < 1 or self.fir_half_length_per_rate < 1:
            raise ValueError("Filter orders must be positive")
        if self.qc_tolerance_volts < 0:
            raise ValueError("qc_tolerance_volts must be non-negative")


@dataclass(frozen=True)
class EDFLoadReceipt:
    """Source identity, alignment, filter, resample, and raw-payload QC."""

    schema_version: str
    edf_sha256: str
    semantic_channels: tuple[str, ...]
    raw_channel_names: tuple[str, ...]
    raw_units: tuple[str, ...]
    source_sfreq_hz: float
    output_sfreq_hz: float
    read_start_sample: int
    read_stop_sample: int
    requested_onset_sec: float
    source_aligned_onset_sec: float
    source_alignment_error_sec: float
    resample_up: int
    resample_down: int
    resample_fir_taps: int
    resample_latency_sec: float
    highpass_hz: float
    lowpass_hz: float
    butterworth_order: int
    iir_initial_state_policy: str
    iir_state_reset_sample: int
    iir_warmup_samples: int
    iir_phase_policy: str
    iir_group_delay_estimator: str
    iir_group_delay_frequency_hz: tuple[float, ...]
    iir_group_delay_seconds: tuple[float, ...]
    iir_scalar_delay_correction_applied: bool
    labram_position_binding_policy: str
    labram_position_names: tuple[str, ...]
    labram_position_ids: tuple[int, ...]
    warmup_sec: float
    max_flatline_run_sec: tuple[float, ...]
    max_extreme_run_sec: tuple[float, ...]
    overlapping_gap_annotations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EDF_PREPROCESS_SCHEMA:
            raise ValueError("Unsupported EDF preprocessing receipt schema")
        if len(self.edf_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.edf_sha256
        ):
            raise ValueError("edf_sha256 must be a lowercase SHA256 digest")
        if self.semantic_channels != STANDARD_19:
            raise ValueError("EDF receipt must use the frozen standard-19 order")
        fields: Sequence[Sequence[object]] = (
            self.raw_channel_names,
            self.raw_units,
            self.max_flatline_run_sec,
            self.max_extreme_run_sec,
        )
        if any(len(values) != 19 for values in fields):
            raise ValueError("EDF receipt per-channel fields must contain 19 values")
        if self.read_start_sample < 0 or self.read_stop_sample <= self.read_start_sample:
            raise ValueError("EDF receipt has an invalid source interval")
        if self.resample_up < 1 or self.resample_down < 1 or self.resample_fir_taps < 3:
            raise ValueError("EDF receipt has invalid rational-resampling metadata")
        if not math.isfinite(self.source_sfreq_hz) or self.source_sfreq_hz <= 0:
            raise ValueError("EDF receipt source sampling rate must be positive")
        if self.iir_initial_state_policy != CAUSAL_IIR_INITIAL_STATE_POLICY:
            raise ValueError("EDF receipt has an unsupported causal-IIR state policy")
        if self.iir_state_reset_sample != self.read_start_sample:
            raise ValueError("Causal-IIR state reset must coincide with read_start_sample")
        expected_warmup_samples = _half_up(self.warmup_sec * self.source_sfreq_hz)
        if self.iir_warmup_samples != expected_warmup_samples:
            raise ValueError("Causal-IIR warmup samples disagree with the receipt clock")
        if self.iir_phase_policy != CAUSAL_IIR_PHASE_POLICY:
            raise ValueError("EDF receipt has an unsupported causal-IIR phase policy")
        if self.iir_group_delay_estimator != CAUSAL_IIR_GROUP_DELAY_ESTIMATOR:
            raise ValueError("EDF receipt has an unsupported IIR delay estimator")
        frequencies = self.iir_group_delay_frequency_hz
        delays = self.iir_group_delay_seconds
        if (
            len(frequencies) != _IIR_GROUP_DELAY_GRID_SIZE
            or len(delays) != _IIR_GROUP_DELAY_GRID_SIZE
            or any(not math.isfinite(value) for value in (*frequencies, *delays))
            or any(left >= right for left, right in zip(frequencies, frequencies[1:]))
            or frequencies[0] < self.highpass_hz - 1e-12
            or frequencies[-1] > self.lowpass_hz + 1e-12
        ):
            raise ValueError("EDF receipt has an invalid frequency-dependent IIR delay grid")
        if self.iir_scalar_delay_correction_applied is not False:
            raise ValueError("A scalar IIR delay correction is forbidden by this policy")
        binding = bind_labram_record_positions(
            self.raw_channel_names,
            semantic_channels=self.semantic_channels,
        )
        if (
            self.labram_position_binding_policy
            != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
            or self.labram_position_binding_policy != binding.policy
            or self.labram_position_names != binding.position_names
            or self.labram_position_ids != binding.position_ids
        ):
            raise ValueError("EDF receipt LaBraM position binding drifted from raw headers")


@dataclass(frozen=True)
class LoadedEDFEvent:
    """A model-ready 60-second event plus both signal-processing receipts."""

    window: EventEEGWindow
    signal_receipt: SignalProcessingReceipt
    edf_receipt: EDFLoadReceipt

    def __post_init__(self) -> None:
        expected_samples = _half_up(60.0 * self.edf_receipt.output_sfreq_hz)
        if tuple(self.window.data.shape) != (19, expected_samples):
            raise ValueError("Loaded EDF event must contain exactly 19 x 60 seconds")
        if self.signal_receipt.output_sfreq_hz != self.window.sfreq_hz:
            raise ValueError("EDF event and signal receipt sampling rates disagree")


def _rational_resampling(source_sfreq: float, output_sfreq: float) -> tuple[int, int]:
    ratio = Fraction(str(float(output_sfreq))) / Fraction(str(float(source_sfreq)))
    ratio = ratio.limit_denominator(10_000)
    reconstructed = float(source_sfreq) * ratio.numerator / ratio.denominator
    if abs(reconstructed - float(output_sfreq)) > 1e-9:
        raise ValueError("Sampling-rate ratio cannot be represented reproducibly")
    return ratio.numerator, ratio.denominator


def _causal_bandpass_sos(
    source_sfreq_hz: float,
    config: CausalEDFConfig,
) -> np.ndarray:
    nyquist = 0.5 * float(source_sfreq_hz)
    if not 0 < config.highpass_hz < config.lowpass_hz < nyquist:
        raise ValueError("Bandpass frequencies are invalid for source sampling rate")
    return butter(
        config.butterworth_order,
        [config.highpass_hz / nyquist, config.lowpass_hz / nyquist],
        btype="bandpass",
        output="sos",
    )


def causal_iir_group_delay_receipt(
    *,
    source_sfreq_hz: float,
    config: CausalEDFConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a diagnostic frequency grid and IIR group delay in seconds.

    The values document the frequency-dependent phase response; they are not
    used to shift the signal or to claim a single correctable onset latency.
    """

    sos = _causal_bandpass_sos(source_sfreq_hz, config)
    frequencies = np.geomspace(
        float(config.highpass_hz),
        float(config.lowpass_hz),
        num=_IIR_GROUP_DELAY_GRID_SIZE,
        dtype=np.float64,
    )
    dense_frequency = np.linspace(
        max(float(config.highpass_hz) * 0.5, np.finfo(np.float64).eps),
        min(
            float(config.lowpass_hz) * 1.25,
            0.5 * float(source_sfreq_hz) * (1.0 - 1e-9),
        ),
        num=32_769,
        dtype=np.float64,
    )
    dense_angular = 2.0 * np.pi * dense_frequency / float(source_sfreq_hz)
    _, response = sosfreqz(sos, worN=dense_angular)
    unwrapped_phase = np.unwrap(np.angle(response))
    dense_delay_samples = -np.gradient(unwrapped_phase, dense_angular)
    delay_samples = np.interp(frequencies, dense_frequency, dense_delay_samples)
    delay_seconds = np.asarray(delay_samples, dtype=np.float64) / float(
        source_sfreq_hz
    )
    if not np.isfinite(delay_seconds).all():
        raise RuntimeError("Causal-IIR group-delay receipt contains non-finite values")
    return (
        tuple(float(value) for value in frequencies),
        tuple(float(value) for value in delay_seconds),
    )


def causal_bandpass_resample_channel_field(
    data: np.ndarray,
    *,
    source_sfreq_hz: float,
    config: CausalEDFConfig,
) -> tuple[np.ndarray, int, int, int, float]:
    """Filter/resample one finite physical channel field without rereferencing.

    The transformation is channel-separable.  Keeping this primitive generic
    lets EviSOZ process ``Standard19 + A1 + A2`` on the exact same causal
    clock before deriving CAR19 and signed TCP22.  Task-specific callers must
    still enforce their own channel identity and row-count contracts.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(
            "Causal channel-field preprocessing expects non-empty [C,T] data"
        )
    if not np.isfinite(values).all():
        raise ValueError("EDF payload contains non-finite samples")
    sos = _causal_bandpass_sos(source_sfreq_hz, config)
    filtered = sosfilt(sos, values, axis=-1)

    up, down = _rational_resampling(source_sfreq_hz, config.output_sfreq_hz)
    max_rate = max(up, down)
    half_length = config.fir_half_length_per_rate * max_rate
    taps = firwin(
        2 * half_length + 1,
        1.0 / max_rate,
        window=("kaiser", 5.0),
    )
    taps *= up
    resampled = upfirdn(taps, filtered, up=up, down=down, axis=-1)
    latency_sec = half_length / (up * float(source_sfreq_hz))
    return resampled, up, down, len(taps), latency_sec


def causal_bandpass_resample(
    data: np.ndarray,
    *,
    source_sfreq_hz: float,
    config: CausalEDFConfig,
) -> tuple[np.ndarray, int, int, int, float]:
    """Frozen Standard19 wrapper around the channel-field primitive."""

    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] != 19 or values.shape[1] < 2:
        raise ValueError("Causal preprocessing expects [19,T] non-empty data")
    return causal_bandpass_resample_channel_field(
        values,
        source_sfreq_hz=source_sfreq_hz,
        config=config,
    )


def _annotation_gaps(
    reader: object,
    *,
    start_sec: float,
    stop_sec: float,
) -> tuple[str, ...]:
    if not hasattr(reader, "readAnnotations"):
        return ()
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    except NotImplementedError:
        return ()
    overlapping: list[str] = []
    for onset, duration, description in zip(onsets, durations, descriptions):
        text = str(description).strip()
        lowered = text.lower()
        if not any(word in lowered for word in _GAP_WORDS):
            continue
        annotation_start = float(onset)
        annotation_stop = annotation_start + max(float(duration), 0.0)
        if annotation_start < stop_sec and annotation_stop >= start_sec:
            overlapping.append(text)
    return tuple(overlapping)


def _default_reader_factory(path: str) -> object:
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise RuntimeError("pyedflib is required for physical EDF loading") from exc
    return pyedflib.EdfReader(path)


def load_standard19_edf_event(
    edf_path: str | Path,
    onset_sec: float,
    *,
    config: CausalEDFConfig = CausalEDFConfig(),
    reader_factory: Callable[[str], object] | None = None,
    use_edf_gap_annotations_for_signal_qc: bool = True,
) -> LoadedEDFEvent:
    """Load one known-onset event without reconstructing physical channels.

    A 30-second pre-window warmup is required by default. The causal rational
    FIR latency is retained and applied equally to the event-alignment clock;
    no zero-phase filtering or delay-compensating future samples are used.
    """

    if type(use_edf_gap_annotations_for_signal_qc) is not bool:
        raise TypeError("use_edf_gap_annotations_for_signal_qc must be boolean")

    path = Path(edf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    requested_onset = float(onset_sec)
    if not math.isfinite(requested_onset) or requested_onset < 0:
        raise ValueError("onset_sec must be finite and non-negative")
    factory = _default_reader_factory if reader_factory is None else reader_factory
    reader = factory(str(path))
    try:
        labels = tuple(str(value).strip() for value in reader.getSignalLabels())
        candidates: dict[str, list[int]] = {channel: [] for channel in STANDARD_19}
        for index, label in enumerate(labels):
            canonical = normalize_electrode_name(label)
            if canonical in candidates:
                candidates[canonical].append(index)
        missing = [channel for channel, indices in candidates.items() if not indices]
        duplicates = {
            channel: tuple(labels[index] for index in indices)
            for channel, indices in candidates.items()
            if len(indices) > 1
        }
        if missing or duplicates:
            raise _ineligible(
                "ambiguous_standard19",
                "EDF lacks an unambiguous direct standard-19 montage; "
                f"missing={missing}, duplicates={duplicates}",
            )
        indices = tuple(candidates[channel][0] for channel in STANDARD_19)
        selected_names = tuple(labels[index] for index in indices)
        labram_binding = bind_labram_record_positions(
            selected_names,
            semantic_channels=STANDARD_19,
        )
        raw_sampling_rates = tuple(
            reader.getSampleFrequency(index) for index in indices
        )
        try:
            sampling_rates = tuple(float(value) for value in raw_sampling_rates)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _ineligible(
                "invalid_sfreq",
                "EDF contains an invalid physical-channel sampling rate",
            ) from exc
        if any(not math.isfinite(value) or value <= 0 for value in sampling_rates):
            raise _ineligible(
                "invalid_sfreq",
                "EDF contains an invalid physical-channel sampling rate",
            )
        source_sfreq = sampling_rates[0]
        if any(abs(value - source_sfreq) > 1e-9 for value in sampling_rates):
            raise _ineligible(
                "mixed_sfreq",
                "All selected physical channels must share one sampling rate",
            )
        raw_units = tuple(str(reader.getPhysicalDimension(index)).strip() for index in indices)
        try:
            scales = np.asarray(
                [_unit_scale(unit) for unit in raw_units], dtype=np.float64
            )
        except ValueError as exc:
            raise _ineligible(
                "reference_or_signal_contract", str(exc)
            ) from exc
        raw_sample_counts = reader.getNSamples()
        try:
            sample_counts = tuple(int(raw_sample_counts[index]) for index in indices)
        except (IndexError, TypeError, ValueError, OverflowError) as exc:
            raise _ineligible(
                "sample_count_mismatch",
                "Selected physical channels have invalid sample counts",
            ) from exc
        if any(value <= 0 for value in sample_counts) or len(set(sample_counts)) != 1:
            raise _ineligible(
                "sample_count_mismatch",
                "All selected physical channels must share one sample count",
            )

        nyquist = 0.5 * source_sfreq
        if config.lowpass_hz >= nyquist:
            raise _ineligible(
                "invalid_sfreq",
                "Bandpass frequencies are invalid for source sampling rate",
            )

        up, down = _rational_resampling(source_sfreq, config.output_sfreq_hz)
        half_length = config.fir_half_length_per_rate * max(up, down)
        latency_sec = half_length / (up * source_sfreq)
        onset_sample = _half_up(requested_onset * source_sfreq)
        source_aligned_onset = onset_sample / source_sfreq
        warmup_and_pre = _half_up(
            (config.warmup_sec + config.pre_onset_sec) * source_sfreq
        )
        post_and_latency = _half_up(
            (config.post_onset_sec + latency_sec + 1.0 / source_sfreq)
            * source_sfreq
        )
        read_start = onset_sample - warmup_and_pre
        read_stop = onset_sample + post_and_latency
        if read_start < 0:
            raise _ineligible(
                "insufficient_warmup",
                "Event lacks the required causal warmup and pre-onset interval"
            )
        if read_stop > sample_counts[0]:
            raise _ineligible(
                "insufficient_post",
                "Event lacks the complete delayed post-onset interval",
            )
        n_read = read_stop - read_start
        reader_payloads = tuple(
            reader.readSignal(index, read_start, n_read) for index in indices
        )
        try:
            raw = np.stack(
                [
                    np.asarray(payload, dtype=np.float64)
                    for payload in reader_payloads
                ]
            )
        except (TypeError, ValueError) as exc:
            raise _ineligible(
                "payload_shape",
                "EDF reader returned an unexpected selected payload shape",
            ) from exc
        if tuple(raw.shape) != (19, n_read):
            raise _ineligible(
                "payload_shape",
                "EDF reader returned an unexpected selected payload shape",
            )
        raw_volts = raw * scales[:, None]
        max_flatline_samples = tuple(
            _max_flatline_run_samples(channel, config.qc_tolerance_volts)
            for channel in raw_volts
        )
        max_extreme_samples = tuple(
            _max_extreme_run_samples(channel, config.qc_tolerance_volts)
            for channel in raw_volts
        )
        flatline_limit = _half_up(config.flatline_run_sec * source_sfreq)
        clipping_limit = _half_up(config.clipping_run_sec * source_sfreq)
        bad_flatline = [
            STANDARD_19[index]
            for index, run in enumerate(max_flatline_samples)
            if run >= flatline_limit
        ]
        bad_clipping = [
            STANDARD_19[index]
            for index, run in enumerate(max_extreme_samples)
            if run >= clipping_limit
        ]
        read_start_sec = read_start / source_sfreq
        read_stop_sec = read_stop / source_sfreq
        gaps = (
            _annotation_gaps(reader, start_sec=read_start_sec, stop_sec=read_stop_sec)
            if use_edf_gap_annotations_for_signal_qc
            else ()
        )
        if bad_flatline or bad_clipping or gaps:
            raise _ineligible(
                "signal_qc",
                "EDF payload failed pre-filter QC; "
                f"flatline={bad_flatline}, clipping={bad_clipping}, gaps={gaps}",
                details={
                    "qc_stage": "pre_filter_raw_physical",
                    "failed_checks": [
                        *(["flatline_run"] if bad_flatline else []),
                        *(["extreme_value_run"] if bad_clipping else []),
                    ],
                    "flatline_channels": list(bad_flatline),
                    "clipping_channels": list(bad_clipping),
                    "flatline_run_threshold_seconds": float(
                        config.flatline_run_sec
                    ),
                    "clipping_run_threshold_seconds": float(
                        config.clipping_run_sec
                    ),
                    "qc_tolerance_volts": float(config.qc_tolerance_volts),
                    "edf_gap_annotations_used": bool(
                        use_edf_gap_annotations_for_signal_qc
                    ),
                },
            )

        processed, actual_up, actual_down, n_taps, actual_latency = (
            causal_bandpass_resample(
                raw,
                source_sfreq_hz=source_sfreq,
                config=config,
            )
        )
        if (actual_up, actual_down) != (up, down) or abs(actual_latency - latency_sec) > 1e-12:
            raise RuntimeError("Causal resampling metadata drifted during preprocessing")
        iir_delay_frequencies, iir_delay_seconds = causal_iir_group_delay_receipt(
            source_sfreq_hz=source_sfreq,
            config=config,
        )
        try:
            physical = select_standard19_physical(
                torch.from_numpy(processed),
                selected_names,
                sfreq_hz=config.output_sfreq_hz,
                source_sfreq_hz=source_sfreq,
                input_unit=raw_units,
                filter_version=(
                    f"causal_sos_butter_bandpass_v1_order{config.butterworth_order}_"
                    f"{config.highpass_hz:g}-{config.lowpass_hz:g}Hz"
                ),
                resample_version=(
                    f"causal_upfirdn_kaiser5_v1_{up}up_{down}down_{n_taps}taps_"
                    "delay_retained"
                ),
                channel_gap_detected=(False,) * 19,
                channel_clipping_detected=(False,) * 19,
                apply_car19=config.apply_car19,
                expected_sfreq_hz=config.output_sfreq_hz,
                reference_policy=config.reference_policy,
                sensitivity_reference=config.sensitivity_reference,
                flatline_tolerance_volts=config.qc_tolerance_volts,
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("Source reference mismatch"):
                raise _ineligible(
                    "reference_or_signal_contract", message
                ) from exc
            if message.startswith("Selected physical EEG failed channel QC"):
                raise _ineligible(
                    "signal_qc",
                    message,
                    details={
                        "qc_stage": "post_preprocessing_physical_contract",
                        "failed_checks": [
                            "downstream_physical_signal_contract"
                        ],
                        "flatline_channels": [],
                        "clipping_channels": [],
                        "flatline_run_threshold_seconds": float(
                            config.flatline_run_sec
                        ),
                        "clipping_run_threshold_seconds": float(
                            config.clipping_run_sec
                        ),
                        "qc_tolerance_volts": float(
                            config.qc_tolerance_volts
                        ),
                        "edf_gap_annotations_used": False,
                    },
                ) from exc
            raise
        onset_in_processed = (onset_sample - read_start) / source_sfreq + latency_sec
        window = crop_event_window(
            physical,
            onset_in_processed,
            pre_onset_sec=config.pre_onset_sec,
            post_onset_sec=config.post_onset_sec,
        )
        receipt = EDFLoadReceipt(
            schema_version=EDF_PREPROCESS_SCHEMA,
            edf_sha256=_file_sha256(path),
            semantic_channels=STANDARD_19,
            raw_channel_names=selected_names,
            raw_units=raw_units,
            source_sfreq_hz=source_sfreq,
            output_sfreq_hz=config.output_sfreq_hz,
            read_start_sample=read_start,
            read_stop_sample=read_stop,
            requested_onset_sec=requested_onset,
            source_aligned_onset_sec=source_aligned_onset,
            source_alignment_error_sec=source_aligned_onset - requested_onset,
            resample_up=up,
            resample_down=down,
            resample_fir_taps=n_taps,
            resample_latency_sec=latency_sec,
            highpass_hz=config.highpass_hz,
            lowpass_hz=config.lowpass_hz,
            butterworth_order=config.butterworth_order,
            iir_initial_state_policy=CAUSAL_IIR_INITIAL_STATE_POLICY,
            iir_state_reset_sample=read_start,
            iir_warmup_samples=_half_up(config.warmup_sec * source_sfreq),
            iir_phase_policy=CAUSAL_IIR_PHASE_POLICY,
            iir_group_delay_estimator=CAUSAL_IIR_GROUP_DELAY_ESTIMATOR,
            iir_group_delay_frequency_hz=iir_delay_frequencies,
            iir_group_delay_seconds=iir_delay_seconds,
            iir_scalar_delay_correction_applied=False,
            labram_position_binding_policy=labram_binding.policy,
            labram_position_names=labram_binding.position_names,
            labram_position_ids=labram_binding.position_ids,
            warmup_sec=config.warmup_sec,
            max_flatline_run_sec=tuple(
                run / source_sfreq for run in max_flatline_samples
            ),
            max_extreme_run_sec=tuple(
                run / source_sfreq for run in max_extreme_samples
            ),
            overlapping_gap_annotations=gaps,
        )
        return LoadedEDFEvent(
            window=window,
            signal_receipt=physical.receipt,
            edf_receipt=receipt,
        )
    finally:
        if hasattr(reader, "close"):
            reader.close()


__all__ = [
    "CausalEDFConfig",
    "CAUSAL_IIR_GROUP_DELAY_ESTIMATOR",
    "CAUSAL_IIR_INITIAL_STATE_POLICY",
    "CAUSAL_IIR_PHASE_POLICY",
    "EDFEventEligibilityError",
    "EDFLoadReceipt",
    "EDF_PREPROCESS_SCHEMA",
    "LoadedEDFEvent",
    "causal_bandpass_resample",
    "causal_bandpass_resample_channel_field",
    "causal_iir_group_delay_receipt",
    "load_standard19_edf_event",
]
