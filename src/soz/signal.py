"""Fail-closed standard-19 physical EEG signal contract."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math
import re
from typing import Sequence

import torch

from .geometry import N_STANDARD_CHANNELS, STANDARD_19, normalize_electrode_name


_UNIT_TO_VOLTS = {
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
}
_UNSPECIFIED_COMMON_REFERENCE = "UNSPECIFIED_COMMON"
_REFERENCE_SUFFIXES = frozenset(
    {"REF", "LE", "AR", "AVG", "AV", "CAR", _UNSPECIFIED_COMMON_REFERENCE}
)
_REFERENCE_PATTERN = re.compile(r"-(REF|LE|AR|AVG|AV|CAR)$", re.IGNORECASE)
_PRIMARY_REFERENCE_POLICY = "primary_ref"
_SENSITIVITY_REFERENCE_POLICY = "sensitivity_uniform"
_ONSET_ROUNDING = "decimal_half_up_to_nearest_sample"


def _unit_scale(unit: str) -> float:
    normalized = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    try:
        return _UNIT_TO_VOLTS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported EEG unit {unit!r}; provide V, mV, uV, µV, or μV explicitly"
        ) from exc


def _reference_suffix(raw_name: object) -> str | None:
    """Return the explicitly encoded source reference without discarding it."""

    text = str(raw_name).strip().upper().replace("_", "-")
    match = _REFERENCE_PATTERN.search(text)
    return None if match is None else match.group(1).upper()


def _as_per_channel_units(
    input_unit: str | Sequence[str], n_channels: int
) -> tuple[str, ...]:
    if isinstance(input_unit, str):
        units = (input_unit,) * n_channels
    else:
        units = tuple(str(unit) for unit in input_unit)
        if len(units) != n_channels:
            raise ValueError("Per-channel input units must align with channel_names")
    for unit in units:
        _unit_scale(unit)
    return units


def _as_qc_flags(
    values: Sequence[bool], *, n_channels: int, name: str
) -> tuple[bool, ...]:
    flags = tuple(values)
    if len(flags) != n_channels:
        raise ValueError(f"{name} must align with channel_names")
    if any(not isinstance(value, bool) for value in flags):
        raise TypeError(f"{name} must contain explicit boolean values")
    return flags


def _require_version(value: str, *, name: str) -> str:
    version = str(value).strip()
    if not version or version.lower() in {"none", "unknown", "unspecified"}:
        raise ValueError(f"{name} must be an explicit non-empty processing version")
    return version


def _expected_source_reference(
    *, reference_policy: str, sensitivity_reference: str | None
) -> str:
    policy = str(reference_policy).strip().lower()
    if policy == _PRIMARY_REFERENCE_POLICY:
        if sensitivity_reference is not None:
            raise ValueError(
                "sensitivity_reference is forbidden under the primary REF policy"
            )
        return "REF"
    if policy == _SENSITIVITY_REFERENCE_POLICY:
        if sensitivity_reference is None:
            raise ValueError(
                "sensitivity_uniform requires an explicit sensitivity_reference"
            )
        expected = str(sensitivity_reference).strip().upper()
        if expected not in _REFERENCE_SUFFIXES:
            raise ValueError(
                "sensitivity_reference must be one of "
                f"{sorted(_REFERENCE_SUFFIXES)}"
            )
        return expected
    if policy == "unlabeled_common_car19":
        if sensitivity_reference is not None:
            raise ValueError(
                "sensitivity_reference is forbidden under unlabeled_common_car19"
            )
        return _UNSPECIFIED_COMMON_REFERENCE
    raise ValueError(
        "reference_policy must be 'primary_ref', 'sensitivity_uniform', or "
        "'unlabeled_common_car19'"
    )


def _round_half_up_samples(seconds: float, sfreq_hz: float) -> int:
    """Freeze decimal half-up rounding from seconds to a non-negative sample."""

    if not math.isfinite(float(seconds)) or float(seconds) < 0:
        raise ValueError("Time values must be finite and non-negative")
    exact_samples = Decimal(str(float(seconds))) * Decimal(str(float(sfreq_hz)))
    return int(exact_samples.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class ChannelSignalQC:
    """QC receipt for one selected physical channel before CAR."""

    semantic_name: str
    raw_name: str
    source_unit: str
    source_reference: str
    n_samples: int
    finite_pass: bool
    gap_pass: bool
    flatline_pass: bool
    clipping_pass: bool
    peak_to_peak_volts: float
    peak_abs_volts: float

    def __post_init__(self) -> None:
        if self.semantic_name not in STANDARD_19:
            raise ValueError("Channel QC semantic_name must belong to standard-19")
        if self.source_reference not in _REFERENCE_SUFFIXES:
            raise ValueError("Channel QC requires an explicit recognized reference")
        if self.n_samples < 1:
            raise ValueError("Channel QC requires at least one sample")
        if not all(
            (
                self.finite_pass,
                self.gap_pass,
                self.flatline_pass,
                self.clipping_pass,
            )
        ):
            raise ValueError("A stored primary signal receipt may contain only passed QC")
        if not math.isfinite(self.peak_to_peak_volts) or not math.isfinite(
            self.peak_abs_volts
        ):
            raise ValueError("Channel QC amplitude summaries must be finite")


@dataclass(frozen=True)
class SignalProcessingReceipt:
    """Immutable lineage and QC receipt for the selected standard-19 signal."""

    semantic_channels: tuple[str, ...]
    selected_raw_names: tuple[str, ...]
    source_units: tuple[str, ...]
    source_references: tuple[str, ...]
    source_sfreq_hz: float
    output_sfreq_hz: float
    reference_policy: str
    sensitivity_reference: str | None
    output_reference: str
    filter_version: str
    resample_version: str
    flatline_tolerance_volts: float
    channel_qc: tuple[ChannelSignalQC, ...]

    def __post_init__(self) -> None:
        if self.semantic_channels != STANDARD_19:
            raise ValueError("Signal receipt semantic channels must use frozen standard-19")
        fields = (
            self.selected_raw_names,
            self.source_units,
            self.source_references,
            self.channel_qc,
        )
        if any(len(values) != N_STANDARD_CHANNELS for values in fields):
            raise ValueError("Signal receipt channel fields must contain exactly 19 entries")
        if self.source_sfreq_hz <= 0 or self.output_sfreq_hz <= 0:
            raise ValueError("Signal receipt sampling rates must be positive")
        if self.flatline_tolerance_volts < 0:
            raise ValueError("flatline_tolerance_volts must be non-negative")
        _require_version(self.filter_version, name="filter_version")
        _require_version(self.resample_version, name="resample_version")
        expected = _expected_source_reference(
            reference_policy=self.reference_policy,
            sensitivity_reference=self.sensitivity_reference,
        )
        if any(reference != expected for reference in self.source_references):
            raise ValueError("Signal receipt contains a reference-policy mismatch")


@dataclass(frozen=True)
class PhysicalEEG:
    """Continuous complete standard-19 physical EEG in volts."""

    data: torch.Tensor
    sfreq_hz: float
    reference: str
    receipt: SignalProcessingReceipt

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.shape[0] != N_STANDARD_CHANNELS:
            raise ValueError("PhysicalEEG data must have shape [19,T]")
        if self.data.shape[1] < 1:
            raise ValueError("PhysicalEEG cannot be empty")
        if not self.data.is_floating_point() or not torch.isfinite(self.data).all():
            raise ValueError("PhysicalEEG must contain finite floating-point values")
        if self.sfreq_hz <= 0:
            raise ValueError("sfreq_hz must be positive")
        if abs(self.sfreq_hz - self.receipt.output_sfreq_hz) > 1e-9:
            raise ValueError("PhysicalEEG sampling rate must match its receipt")
        if self.reference != self.receipt.output_reference:
            raise ValueError("PhysicalEEG reference must match its receipt")
        if any(qc.n_samples != self.data.shape[1] for qc in self.receipt.channel_qc):
            raise ValueError("Channel QC sample counts must match PhysicalEEG")

    @property
    def selected_raw_names(self) -> tuple[str, ...]:
        """Backward-compatible access to the receipt's raw names."""

        return self.receipt.selected_raw_names

    @property
    def source_unit(self) -> str:
        """Return the uniform source unit, or ``mixed`` for per-channel units."""

        unique = set(self.receipt.source_units)
        return next(iter(unique)) if len(unique) == 1 else "mixed"


@dataclass(frozen=True)
class EventEEGWindow:
    """A fixed event window with a recorded onset-to-sample alignment receipt."""

    data: torch.Tensor
    sfreq_hz: float
    start_sec: float
    stop_sec: float
    onset_index: int
    onset_sample_in_record: int
    requested_onset_sec: float
    aligned_onset_sec: float
    alignment_error_sec: float
    onset_rounding: str = _ONSET_ROUNDING

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.shape[0] != N_STANDARD_CHANNELS:
            raise ValueError("EventEEGWindow data must have shape [19,T]")
        if not torch.isfinite(self.data).all():
            raise ValueError("EventEEGWindow must be finite")
        if not 0 <= self.onset_index < self.data.shape[1]:
            raise ValueError("onset_index must lie inside the event window")
        if self.onset_sample_in_record < 0:
            raise ValueError("onset_sample_in_record must be non-negative")
        if self.onset_rounding != _ONSET_ROUNDING:
            raise ValueError("Event onset rounding policy is not the frozen policy")
        expected_aligned = self.onset_sample_in_record / self.sfreq_hz
        if abs(self.aligned_onset_sec - expected_aligned) > 1e-12:
            raise ValueError("aligned_onset_sec does not match the onset sample")
        expected_error = self.aligned_onset_sec - self.requested_onset_sec
        if abs(self.alignment_error_sec - expected_error) > 1e-12:
            raise ValueError("alignment_error_sec is inconsistent")
        if abs(self.alignment_error_sec) > 0.5 / self.sfreq_hz + 1e-12:
            raise ValueError("Onset alignment error exceeds half a sample")


def select_standard19_physical(
    data: torch.Tensor,
    channel_names: Sequence[object],
    *,
    sfreq_hz: float,
    source_sfreq_hz: float,
    input_unit: str | Sequence[str],
    filter_version: str,
    resample_version: str,
    channel_gap_detected: Sequence[bool],
    channel_clipping_detected: Sequence[bool],
    apply_car19: bool = True,
    expected_sfreq_hz: float = 200.0,
    reference_policy: str = _PRIMARY_REFERENCE_POLICY,
    sensitivity_reference: str | None = None,
    flatline_tolerance_volts: float = 1e-12,
) -> PhysicalEEG:
    """Select direct physical channels and enforce reference/QC provenance.

    The primary policy accepts exactly 19 uniformly ``-REF``-encoded physical
    channels. A different, still uniform, source reference is available only
    through the explicit ``sensitivity_uniform`` policy. The private-transfer
    policy ``unlabeled_common_car19`` accepts only uniformly suffix-free direct
    physical channels and requires CAR19. It records an assumption that all
    channels shared one unrecorded source reference; it does not rename that
    source as REF. Bipolar inputs are never inverted into physical electrodes.
    """

    tensor = torch.as_tensor(data)
    if tensor.ndim != 2:
        raise ValueError(f"Raw EEG data must have shape [C,T], got {tuple(tensor.shape)}")
    if tensor.shape[0] != len(channel_names):
        raise ValueError("Raw channel count does not match channel_names")
    if not math.isfinite(float(sfreq_hz)) or abs(
        float(sfreq_hz) - float(expected_sfreq_hz)
    ) > 1e-6:
        raise ValueError(
            f"Expected preprocessed {expected_sfreq_hz:g} Hz EEG, got {sfreq_hz:g} Hz"
        )
    if not math.isfinite(float(source_sfreq_hz)) or float(source_sfreq_hz) <= 0:
        raise ValueError("source_sfreq_hz must be finite and positive")
    if flatline_tolerance_volts < 0:
        raise ValueError("flatline_tolerance_volts must be non-negative")
    filter_version = _require_version(filter_version, name="filter_version")
    resample_version = _require_version(resample_version, name="resample_version")
    expected_reference = _expected_source_reference(
        reference_policy=reference_policy,
        sensitivity_reference=sensitivity_reference,
    )
    normalized_reference_policy = str(reference_policy).strip().lower()
    if normalized_reference_policy == "unlabeled_common_car19" and not apply_car19:
        raise ValueError("unlabeled_common_car19 requires apply_car19=True")

    n_raw_channels = len(channel_names)
    per_channel_units = _as_per_channel_units(input_unit, n_raw_channels)
    gap_detected = _as_qc_flags(
        channel_gap_detected,
        n_channels=n_raw_channels,
        name="channel_gap_detected",
    )
    clipping_detected = _as_qc_flags(
        channel_clipping_detected,
        n_channels=n_raw_channels,
        name="channel_clipping_detected",
    )

    candidates: dict[str, list[tuple[int, str, str | None]]] = {
        channel: [] for channel in STANDARD_19
    }
    for index, raw_name in enumerate(channel_names):
        canonical = normalize_electrode_name(raw_name)
        if canonical in candidates:
            candidates[canonical].append(
                (index, str(raw_name), _reference_suffix(raw_name))
            )
    missing = [channel for channel, rows in candidates.items() if not rows]
    duplicates = {
        channel: tuple(raw_name for _, raw_name, _ in rows)
        for channel, rows in candidates.items()
        if len(rows) > 1
    }
    if missing or duplicates:
        raise ValueError(
            "Complete unambiguous standard-19 physical channels are required; "
            f"missing={missing}, duplicates={duplicates}"
        )

    selected_rows = tuple(candidates[channel][0] for channel in STANDARD_19)
    indices = [row[0] for row in selected_rows]
    selected_names = tuple(row[1] for row in selected_rows)
    selected_references = tuple(
        _UNSPECIFIED_COMMON_REFERENCE if row[2] is None else row[2]
        for row in selected_rows
    )
    mismatched_references = {
        semantic: {"raw_name": raw_name, "observed": reference}
        for semantic, raw_name, reference in zip(
            STANDARD_19, selected_names, selected_references
        )
        if reference != expected_reference
    }
    if mismatched_references:
        raise ValueError(
            f"Source reference mismatch under {reference_policy!r}; "
            f"expected all {expected_reference}, mismatches={mismatched_references}"
        )

    selected_raw = tensor.index_select(
        0, torch.tensor(indices, device=tensor.device, dtype=torch.long)
    ).to(dtype=torch.float32)
    selected_units = tuple(per_channel_units[index] for index in indices)
    scales = selected_raw.new_tensor([_unit_scale(unit) for unit in selected_units])
    selected_volts = selected_raw * scales.unsqueeze(1)

    qc_rows: list[ChannelSignalQC] = []
    qc_failures: list[str] = []
    for channel_index, (semantic, raw_name, reference, source_unit) in enumerate(
        zip(STANDARD_19, selected_names, selected_references, selected_units)
    ):
        channel = selected_volts[channel_index]
        finite_pass = bool(torch.isfinite(channel).all())
        gap_pass = not gap_detected[indices[channel_index]]
        clipping_pass = not clipping_detected[indices[channel_index]]
        if finite_pass:
            peak_to_peak = float((channel.max() - channel.min()).detach().cpu())
            peak_abs = float(channel.abs().max().detach().cpu())
            flatline_pass = peak_to_peak > float(flatline_tolerance_volts)
        else:
            peak_to_peak = float("nan")
            peak_abs = float("nan")
            flatline_pass = False
        failed = [
            name
            for name, passed in (
                ("finite", finite_pass),
                ("gap", gap_pass),
                ("flatline", flatline_pass),
                ("clipping", clipping_pass),
            )
            if not passed
        ]
        if failed:
            qc_failures.append(f"{semantic}:{','.join(failed)}")
            continue
        qc_rows.append(
            ChannelSignalQC(
                semantic_name=semantic,
                raw_name=raw_name,
                source_unit=source_unit,
                source_reference=str(reference),
                n_samples=int(channel.numel()),
                finite_pass=finite_pass,
                gap_pass=gap_pass,
                flatline_pass=flatline_pass,
                clipping_pass=clipping_pass,
                peak_to_peak_volts=peak_to_peak,
                peak_abs_volts=peak_abs,
            )
        )
    if qc_failures:
        raise ValueError(
            "Selected physical EEG failed channel QC: " + ";".join(qc_failures)
        )

    output_reference = f"source_uniform_{expected_reference}"
    selected = selected_volts
    if apply_car19:
        selected = selected - selected.mean(dim=0, keepdim=True)
        output_reference = "common_average_standard19"
    receipt = SignalProcessingReceipt(
        semantic_channels=STANDARD_19,
        selected_raw_names=selected_names,
        source_units=selected_units,
        source_references=tuple(str(value) for value in selected_references),
        source_sfreq_hz=float(source_sfreq_hz),
        output_sfreq_hz=float(sfreq_hz),
        reference_policy=normalized_reference_policy,
        sensitivity_reference=(
            None
            if sensitivity_reference is None
            else str(sensitivity_reference).strip().upper()
        ),
        output_reference=output_reference,
        filter_version=filter_version,
        resample_version=resample_version,
        flatline_tolerance_volts=float(flatline_tolerance_volts),
        channel_qc=tuple(qc_rows),
    )
    return PhysicalEEG(
        data=selected,
        sfreq_hz=float(sfreq_hz),
        reference=output_reference,
        receipt=receipt,
    )


def crop_event_window(
    record: PhysicalEEG,
    onset_sec: float,
    *,
    pre_onset_sec: float = 12.0,
    post_onset_sec: float = 48.0,
) -> EventEEGWindow:
    """Crop ``[onset-pre, onset+post)`` after frozen half-up alignment."""

    if pre_onset_sec <= 0 or post_onset_sec <= 0:
        raise ValueError("pre_onset_sec and post_onset_sec must be positive")
    sfreq = float(record.sfreq_hz)
    requested_onset = float(onset_sec)
    onset_sample = _round_half_up_samples(requested_onset, sfreq)
    aligned_onset = onset_sample / sfreq
    alignment_error = aligned_onset - requested_onset
    pre_samples = _round_half_up_samples(float(pre_onset_sec), sfreq)
    post_samples = _round_half_up_samples(float(post_onset_sec), sfreq)
    start = onset_sample - pre_samples
    stop = onset_sample + post_samples
    if start < 0 or stop > record.data.shape[1]:
        duration = record.data.shape[1] / sfreq
        raise ValueError(
            "Incomplete event window after onset alignment: "
            f"requested onset={requested_onset:.9f}, "
            f"aligned onset={aligned_onset:.9f}, "
            f"requested interval=[{requested_onset-pre_onset_sec:.3f},"
            f"{requested_onset+post_onset_sec:.3f}), "
            f"recording=[0,{duration:.3f})"
        )
    window = record.data[:, start:stop]
    expected_samples = pre_samples + post_samples
    if window.shape != (N_STANDARD_CHANNELS, expected_samples):
        raise RuntimeError("Event crop produced an unexpected shape")
    return EventEEGWindow(
        data=window,
        sfreq_hz=sfreq,
        start_sec=start / sfreq,
        stop_sec=stop / sfreq,
        onset_index=pre_samples,
        onset_sample_in_record=onset_sample,
        requested_onset_sec=requested_onset,
        aligned_onset_sec=aligned_onset,
        alignment_error_sec=alignment_error,
    )
