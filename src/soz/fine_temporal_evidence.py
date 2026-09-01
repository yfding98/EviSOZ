"""Target-free, sub-second temporal evidence for scalp SOZ reasoning.

The extractor in this module deliberately does not consume seizure-onset
electrode labels, DeepSOZ targets, or private annotations.  It summarizes a
60-second, standard-19, onset-aligned scalp EEG window at one-second analysis
resolution and 250-ms stride.  The global TUSZ seizure time is only the
temporal anchor at ``t=0``; a detected signal change is therefore *not* a
cortical SOZ onset and a relative delay is *not* propagation ground truth.

The returned feature tensor is intended to be combined with a frozen
pretrained EEG representation.  It is not a replacement foundation model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal

import torch

from .geometry import (
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    STANDARD_19,
    TCP_20_EDGES,
    edge_endpoint_indices,
)


FINE_TEMPORAL_EVIDENCE_SCHEMA: Final[str] = (
    "soz_target_free_fine_temporal_evidence_v11"
)
FINE_WINDOW_SECONDS: Final[float] = 1.0
FINE_STRIDE_SECONDS: Final[float] = 0.25
FINE_BASELINE_START_SECONDS: Final[float] = -12.0
FINE_ONSET_HORIZON_SECONDS: Final[float] = 48.0
FINE_CHANGE_THRESHOLD: Final[float] = 2.5
FINE_SUSTAINED_WINDOWS: Final[int] = 3

FINE_TEMPORAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "node_change_peak_0_4s",
    "node_change_mean_positive_0_12s",
    "node_change_persistence_0_12s",
    "node_change_point_strength",
    "node_change_latency_sec_censored",
    "node_change_detected",
    "node_relative_recruitment_delay_sec_censored",
    "node_dominant_frequency_shift_hz_0_12s",
    "node_dominant_frequency_slope_hz_per_sec_0_12s",
    "node_rhythmicity_change_0_12s",
    "node_spectral_entropy_drop_0_12s",
    "node_late_minus_early_change",
    "node_late_change_persistence_12_36s",
    "bipolar_incident_change_peak_0_4s",
    "bipolar_incident_change_mean_positive_0_12s",
    "bipolar_incident_change_latency_sec_censored",
    "bipolar_incident_change_detected",
    "bipolar_incident_relative_delay_sec_censored",
    "high_frequency_ratio_change_0_12s",
    "artifact_burden_0_12s",
)


@dataclass(frozen=True)
class FineTemporalEvidence:
    """Fine-resolution signal evidence for one standard-19 event.

    ``features`` is always finite.  Non-detected change times are represented
    by a censored horizon in the feature matrix and separately by a zero
    detection indicator.  The diagnostic latency vectors retain ``NaN`` for
    non-detections so downstream reports cannot accidentally present a
    censored value as an observed onset.
    """

    features: torch.Tensor
    feature_names: tuple[str, ...]
    composite_trace: torch.Tensor
    dominant_frequency_hz: torch.Tensor
    window_center_sec: torch.Tensor
    node_change_detected: torch.Tensor
    node_change_latency_sec: torch.Tensor
    bipolar_change_detected: torch.Tensor
    bipolar_change_latency_sec: torch.Tensor

    def __post_init__(self) -> None:
        n_features = len(FINE_TEMPORAL_FEATURE_NAMES)
        if tuple(self.features.shape) != (N_STANDARD_CHANNELS, n_features):
            raise ValueError("fine evidence features must have shape [19,D]")
        if self.feature_names != FINE_TEMPORAL_FEATURE_NAMES:
            raise ValueError("fine evidence feature vocabulary changed")
        if self.composite_trace.ndim != 2 or self.composite_trace.shape[0] != 19:
            raise ValueError("composite_trace must have shape [19,W]")
        windows = int(self.composite_trace.shape[1])
        if tuple(self.dominant_frequency_hz.shape) != (19, windows):
            raise ValueError("dominant-frequency trace must have shape [19,W]")
        if tuple(self.window_center_sec.shape) != (windows,):
            raise ValueError("window_center_sec must have shape [W]")
        if tuple(self.node_change_detected.shape) != (19,) or (
            self.node_change_detected.dtype != torch.bool
        ):
            raise TypeError("node_change_detected must be bool [19]")
        if tuple(self.node_change_latency_sec.shape) != (19,):
            raise ValueError("node_change_latency_sec must have shape [19]")
        if tuple(self.bipolar_change_detected.shape) != (N_TCP_EDGES,) or (
            self.bipolar_change_detected.dtype != torch.bool
        ):
            raise TypeError("bipolar_change_detected must be bool [20]")
        if tuple(self.bipolar_change_latency_sec.shape) != (N_TCP_EDGES,):
            raise ValueError("bipolar_change_latency_sec must have shape [20]")
        finite_tensors = (
            self.features,
            self.composite_trace,
            self.dominant_frequency_hz,
            self.window_center_sec,
        )
        if any(not value.is_floating_point() for value in finite_tensors):
            raise TypeError("fine temporal evidence tensors must be floating point")
        if any(not torch.isfinite(value).all() for value in finite_tensors):
            raise ValueError("fine temporal evidence contains non-finite values")
        if len({value.device for value in (*finite_tensors, self.node_change_detected,
                                           self.node_change_latency_sec,
                                           self.bipolar_change_detected,
                                           self.bipolar_change_latency_sec)}) != 1:
            raise ValueError("all fine temporal evidence tensors must share a device")
        if not torch.equal(
            torch.isfinite(self.node_change_latency_sec), self.node_change_detected
        ):
            raise ValueError("node latency finiteness must encode detection")
        if not torch.equal(
            torch.isfinite(self.bipolar_change_latency_sec),
            self.bipolar_change_detected,
        ):
            raise ValueError("bipolar latency finiteness must encode detection")

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def feature(self, name: str) -> torch.Tensor:
        """Return one named standard-19 feature without copying it."""

        try:
            index = self.feature_names.index(str(name))
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.features[:, index]

    def to(self, device: str | torch.device) -> "FineTemporalEvidence":
        return FineTemporalEvidence(
            features=self.features.to(device),
            feature_names=self.feature_names,
            composite_trace=self.composite_trace.to(device),
            dominant_frequency_hz=self.dominant_frequency_hz.to(device),
            window_center_sec=self.window_center_sec.to(device),
            node_change_detected=self.node_change_detected.to(device),
            node_change_latency_sec=self.node_change_latency_sec.to(device),
            bipolar_change_detected=self.bipolar_change_detected.to(device),
            bipolar_change_latency_sec=self.bipolar_change_latency_sec.to(device),
        )


@dataclass(frozen=True)
class _WindowDescriptors:
    log_rms: torch.Tensor
    log_line_length: torch.Tensor
    dominant_frequency_hz: torch.Tensor
    spectral_entropy: torch.Tensor
    rhythmicity: torch.Tensor
    high_frequency_ratio: torch.Tensor
    peak_to_peak: torch.Tensor


@dataclass(frozen=True)
class _ChangeSummary:
    composite: torch.Tensor
    detected: torch.Tensor
    latency_sec: torch.Tensor
    censored_latency_sec: torch.Tensor
    relative_delay_sec: torch.Tensor
    peak_0_4: torch.Tensor
    mean_positive_0_12: torch.Tensor
    persistence_0_12: torch.Tensor
    change_point_strength: torch.Tensor
    dominant_frequency_shift: torch.Tensor
    dominant_frequency_slope: torch.Tensor
    rhythmicity_change: torch.Tensor
    entropy_drop: torch.Tensor
    late_minus_early: torch.Tensor
    late_persistence: torch.Tensor
    high_frequency_ratio_change: torch.Tensor
    artifact_burden: torch.Tensor


def _require_eeg(eeg: torch.Tensor, *, sfreq_hz: float) -> None:
    if not isinstance(eeg, torch.Tensor):
        raise TypeError("eeg must be a torch.Tensor")
    if eeg.ndim != 2 or eeg.shape[0] != N_STANDARD_CHANNELS:
        raise ValueError("eeg must have shape [19,T]")
    if not eeg.is_floating_point() or eeg.requires_grad:
        raise TypeError("eeg must be detached floating point")
    if not torch.isfinite(eeg).all():
        raise ValueError("eeg must be finite")
    if not math.isfinite(float(sfreq_hz)) or float(sfreq_hz) <= 0:
        raise ValueError("sfreq_hz must be positive and finite")
    expected = int(round(60.0 * float(sfreq_hz)))
    if eeg.shape[1] != expected:
        raise ValueError(f"eeg must contain exactly 60 seconds ({expected} samples)")


def _analysis_windows(
    signal: torch.Tensor,
    *,
    sfreq_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window_samples = int(round(FINE_WINDOW_SECONDS * sfreq_hz))
    stride_samples = int(round(FINE_STRIDE_SECONDS * sfreq_hz))
    if window_samples < 4 or stride_samples < 1:
        raise ValueError("sampling rate is too low for the frozen fine-time grid")
    if abs(window_samples / sfreq_hz - FINE_WINDOW_SECONDS) > 1e-9 or abs(
        stride_samples / sfreq_hz - FINE_STRIDE_SECONDS
    ) > 1e-9:
        raise ValueError("sampling rate cannot represent the frozen time grid exactly")
    windows = signal.unfold(-1, window_samples, stride_samples).contiguous()
    starts = (
        torch.arange(windows.shape[-2], device=signal.device, dtype=signal.dtype)
        * (stride_samples / sfreq_hz)
        + FINE_BASELINE_START_SECONDS
    )
    centers = starts + 0.5 * FINE_WINDOW_SECONDS
    return windows, starts, centers


def _window_descriptors(
    windows: torch.Tensor,
    *,
    sfreq_hz: float,
) -> _WindowDescriptors:
    if windows.ndim != 3:
        raise ValueError("analysis windows must have shape [C,W,S]")
    eps = torch.finfo(windows.dtype).eps
    centered = windows - windows.mean(dim=-1, keepdim=True)
    log_rms = (centered.square().mean(dim=-1).sqrt() + eps).log()
    log_line = (centered.diff(dim=-1).abs().mean(dim=-1) + eps).log()
    peak_to_peak = windows.amax(dim=-1) - windows.amin(dim=-1)

    taper = torch.hann_window(
        windows.shape[-1],
        periodic=False,
        dtype=windows.dtype,
        device=windows.device,
    )
    spectrum = torch.fft.rfft(centered * taper, dim=-1)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(
        windows.shape[-1], d=1.0 / sfreq_hz, device=windows.device
    ).to(windows.dtype)

    analysis_mask = (frequencies >= 1.0) & (frequencies <= 30.0)
    high_mask = (frequencies >= 30.0) & (frequencies <= 45.0)
    broad_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    if int(analysis_mask.sum()) < 4 or not high_mask.any() or not broad_mask.any():
        raise ValueError("sampling rate lacks the frozen 1--45 Hz evidence bands")

    analysis_power = power[..., analysis_mask]
    analysis_frequencies = frequencies[analysis_mask]
    analysis_total = analysis_power.sum(dim=-1)
    nonzero = analysis_total > eps
    probability = analysis_power / analysis_total.unsqueeze(-1).clamp_min(eps)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=-1)
    entropy = entropy / math.log(float(analysis_power.shape[-1]))
    entropy = torch.where(nonzero, entropy, torch.ones_like(entropy))

    # A three-bin concentration is less sensitive than a single FFT bin while
    # retaining the one-second temporal resolution.
    padded = torch.nn.functional.pad(analysis_power, (1, 1))
    concentration = (
        padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]
    )
    rhythmicity = concentration.amax(dim=-1) / (
        3.0 * analysis_total.clamp_min(eps)
    )
    rhythmicity = torch.where(nonzero, rhythmicity, torch.zeros_like(rhythmicity))
    dominant_index = analysis_power.argmax(dim=-1)
    dominant_frequency = analysis_frequencies[dominant_index]
    dominant_frequency = torch.where(
        nonzero, dominant_frequency, torch.zeros_like(dominant_frequency)
    )

    broad_total = power[..., broad_mask].sum(dim=-1)
    high_total = power[..., high_mask].sum(dim=-1)
    high_ratio = high_total / broad_total.clamp_min(eps)
    high_ratio = torch.where(
        broad_total > eps, high_ratio, torch.zeros_like(high_ratio)
    )
    return _WindowDescriptors(
        log_rms=log_rms,
        log_line_length=log_line,
        dominant_frequency_hz=dominant_frequency,
        spectral_entropy=entropy,
        rhythmicity=rhythmicity,
        high_frequency_ratio=high_ratio,
        peak_to_peak=peak_to_peak,
    )


def _robust_z(values: torch.Tensor, baseline_mask: torch.Tensor) -> torch.Tensor:
    baseline = values[:, baseline_mask]
    if baseline.shape[1] < 4:
        raise ValueError("fine temporal evidence requires at least four baseline windows")
    median = baseline.median(dim=1).values
    mad = (baseline - median.unsqueeze(1)).abs().median(dim=1).values
    standard = baseline.std(dim=1, unbiased=False)
    # The absolute floor prevents an exactly flat baseline from turning
    # floating-point noise into an arbitrarily large change score.
    scale = torch.maximum(1.4826 * mad, 0.1 * standard).clamp_min(1e-3)
    return (values - median.unsqueeze(1)) / scale.unsqueeze(1)


def _linear_slope(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or times.ndim != 1 or values.shape[1] != times.numel():
        raise ValueError("slope inputs must have shape [C,W] and [W]")
    centered_time = times - times.mean()
    denominator = centered_time.square().sum().clamp_min(
        torch.finfo(values.dtype).eps
    )
    centered_values = values - values.mean(dim=1, keepdim=True)
    return (centered_values * centered_time.unsqueeze(0)).sum(dim=1) / denominator


def _sustained_change(
    composite: torch.Tensor,
    starts: torch.Tensor,
    *,
    threshold: float = FINE_CHANGE_THRESHOLD,
    run_windows: int = FINE_SUSTAINED_WINDOWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if composite.ndim != 2 or starts.ndim != 1 or composite.shape[1] != starts.numel():
        raise ValueError("change trace and time grid are not aligned")
    if run_windows < 1 or run_windows > composite.shape[1]:
        raise ValueError("invalid sustained-change run length")
    above = composite >= float(threshold)
    sustained = above.unfold(1, run_windows, 1).all(dim=-1)
    candidate_start = starts[: sustained.shape[1]]
    sustained &= candidate_start.unsqueeze(0) >= 0.0
    detected = sustained.any(dim=1)
    first_index = sustained.to(torch.int64).argmax(dim=1)
    latency = candidate_start[first_index]
    latency = torch.where(
        detected,
        latency,
        torch.full_like(latency, torch.nan),
    )
    return detected, latency


def _summarize_change(
    descriptors: _WindowDescriptors,
    *,
    starts: torch.Tensor,
    centers: torch.Tensor,
) -> _ChangeSummary:
    baseline_mask = (starts >= FINE_BASELINE_START_SECONDS) & (
        starts + FINE_WINDOW_SECONDS <= 0.0
    )
    early4 = (starts >= 0.0) & (starts < 4.0)
    early12 = (starts >= 0.0) & (starts < 12.0)
    late = (starts >= 12.0) & (starts < 36.0)
    if not baseline_mask.any() or not early4.any() or not early12.any() or not late.any():
        raise RuntimeError("frozen 60-second phase masks could not be constructed")

    z_rms = _robust_z(descriptors.log_rms, baseline_mask)
    z_line = _robust_z(descriptors.log_line_length, baseline_mask)
    z_rhythm = _robust_z(descriptors.rhythmicity, baseline_mask)
    z_entropy_drop = _robust_z(-descriptors.spectral_entropy, baseline_mask)
    composite = torch.stack((z_rms, z_line, z_rhythm, z_entropy_drop), dim=-1)
    composite = composite.mean(dim=-1).clamp(-20.0, 20.0)

    detected, latency = _sustained_change(composite, starts)
    horizon = torch.full_like(latency, FINE_ONSET_HORIZON_SECONDS)
    censored = torch.where(detected, latency, horizon)
    if detected.any():
        global_first = latency[detected].amin()
        relative = torch.where(detected, latency - global_first, horizon)
    else:
        relative = horizon

    positive = composite.clamp_min(0.0)
    peak_0_4 = composite[:, early4].amax(dim=1)
    mean_positive = positive[:, early12].mean(dim=1)
    persistence = (composite[:, early12] >= FINE_CHANGE_THRESHOLD).to(
        composite.dtype
    ).mean(dim=1)
    strength = torch.where(
        detected,
        composite.gather(1, ((latency - starts[0]) / FINE_STRIDE_SECONDS)
                         .round().long().clamp(0, composite.shape[1] - 1)
                         .unsqueeze(1)).squeeze(1),
        composite[:, early12].amax(dim=1),
    )

    baseline_dom = descriptors.dominant_frequency_hz[:, baseline_mask].median(
        dim=1
    ).values
    early_dom = descriptors.dominant_frequency_hz[:, early12]
    dominant_shift = early_dom.median(dim=1).values - baseline_dom
    dominant_slope = _linear_slope(early_dom, centers[early12])
    rhythm_change = descriptors.rhythmicity[:, early12].mean(dim=1) - (
        descriptors.rhythmicity[:, baseline_mask].mean(dim=1)
    )
    entropy_drop = descriptors.spectral_entropy[:, baseline_mask].mean(dim=1) - (
        descriptors.spectral_entropy[:, early12].mean(dim=1)
    )
    early_change = composite[:, early12].mean(dim=1)
    late_change = composite[:, late].mean(dim=1)
    late_minus_early = late_change - early_change
    late_persistence = (composite[:, late] >= FINE_CHANGE_THRESHOLD).to(
        composite.dtype
    ).mean(dim=1)
    high_change = descriptors.high_frequency_ratio[:, early12].mean(dim=1) - (
        descriptors.high_frequency_ratio[:, baseline_mask].mean(dim=1)
    )

    # An artifact flag is intentionally conservative and observable.  It does
    # not suppress evidence or assign a diagnosis; it is passed to the
    # reasoner/report as a reliability warning.
    early_peak = descriptors.peak_to_peak[:, early12]
    early_high = descriptors.high_frequency_ratio[:, early12]
    artifact = (early_peak > 1e-3) | (early_high > 0.65)
    artifact_burden = artifact.to(composite.dtype).mean(dim=1)

    return _ChangeSummary(
        composite=composite,
        detected=detected,
        latency_sec=latency,
        censored_latency_sec=censored,
        relative_delay_sec=relative,
        peak_0_4=peak_0_4,
        mean_positive_0_12=mean_positive,
        persistence_0_12=persistence,
        change_point_strength=strength,
        dominant_frequency_shift=dominant_shift,
        dominant_frequency_slope=dominant_slope,
        rhythmicity_change=rhythm_change,
        entropy_drop=entropy_drop,
        late_minus_early=late_minus_early,
        late_persistence=late_persistence,
        high_frequency_ratio_change=high_change,
        artifact_burden=artifact_burden,
    )


def route_bipolar_support_to_nodes(
    edge_support: torch.Tensor,
    *,
    reduction: Literal["max", "mean"] = "max",
) -> torch.Tensor:
    """Route symmetric TCP-edge evidence to physical endpoints.

    The operation never attempts to decide which endpoint generated an edge.
    ``max`` is the default because a single supported edge contributes the
    same value to both endpoints independent of their graph degree.
    """

    if not isinstance(edge_support, torch.Tensor) or edge_support.ndim < 1:
        raise TypeError("edge_support must be a tensor with a final edge axis")
    if edge_support.shape[-1] != N_TCP_EDGES:
        raise ValueError("edge_support final axis must contain TCP-20 edges")
    if not edge_support.is_floating_point() or not torch.isfinite(edge_support).all():
        raise ValueError("edge_support must be finite floating point")
    if reduction not in {"max", "mean"}:
        raise ValueError("reduction must be 'max' or 'mean'")
    endpoints = edge_endpoint_indices(device=edge_support.device)
    rows = []
    for channel in range(N_STANDARD_CHANNELS):
        incident = torch.nonzero(
            (endpoints[:, 0] == channel) | (endpoints[:, 1] == channel),
            as_tuple=False,
        ).flatten()
        if incident.numel() == 0:
            rows.append(torch.zeros_like(edge_support[..., 0]))
            continue
        values = edge_support.index_select(-1, incident)
        rows.append(
            values.amax(dim=-1) if reduction == "max" else values.mean(dim=-1)
        )
    return torch.stack(rows, dim=-1)


def _route_incident_latency(
    detected: torch.Tensor,
    latency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    endpoints = edge_endpoint_indices(device=detected.device)
    node_detected = torch.zeros(
        N_STANDARD_CHANNELS, dtype=torch.bool, device=detected.device
    )
    node_latency = torch.full(
        (N_STANDARD_CHANNELS,),
        FINE_ONSET_HORIZON_SECONDS,
        dtype=latency.dtype,
        device=latency.device,
    )
    for channel in range(N_STANDARD_CHANNELS):
        incident = (endpoints[:, 0] == channel) | (endpoints[:, 1] == channel)
        valid = incident & detected
        if valid.any():
            node_detected[channel] = True
            node_latency[channel] = latency[valid].amin()
    if node_detected.any():
        first = node_latency[node_detected].amin()
        relative = torch.where(
            node_detected,
            node_latency - first,
            torch.full_like(node_latency, FINE_ONSET_HORIZON_SECONDS),
        )
    else:
        relative = torch.full_like(node_latency, FINE_ONSET_HORIZON_SECONDS)
    return node_detected, node_latency, relative


def extract_fine_temporal_evidence(
    eeg: torch.Tensor,
    *,
    sfreq_hz: float = 200.0,
) -> FineTemporalEvidence:
    """Extract target-free fine temporal evidence from one 60-second event.

    Args:
        eeg: Detached standard-19 EEG in volts, shape ``[19, 12000]`` at
            200 Hz.  The interval is ``[-12, +48)`` seconds around the TUSZ
            global seizure time.
        sfreq_hz: Sampling frequency.  The formal protocol uses 200 Hz.

    Returns:
        A validated :class:`FineTemporalEvidence` instance.  None of its
        outputs is an SOZ label, onset-electrode label, or propagation label.
    """

    _require_eeg(eeg, sfreq_hz=sfreq_hz)
    if abs(float(sfreq_hz) - 200.0) > 1e-9:
        raise ValueError("formal fine temporal evidence is frozen at 200 Hz")
    values = eeg.detach().to(dtype=torch.float32).contiguous()
    node_windows, starts, centers = _analysis_windows(values, sfreq_hz=sfreq_hz)
    node_descriptors = _window_descriptors(node_windows, sfreq_hz=sfreq_hz)
    node = _summarize_change(node_descriptors, starts=starts, centers=centers)

    endpoints = edge_endpoint_indices(device=values.device)
    bipolar = values.index_select(0, endpoints[:, 0]) - values.index_select(
        0, endpoints[:, 1]
    )
    edge_windows, edge_starts, edge_centers = _analysis_windows(
        bipolar, sfreq_hz=sfreq_hz
    )
    if not torch.equal(starts, edge_starts) or not torch.equal(centers, edge_centers):
        raise RuntimeError("node and bipolar temporal grids diverged")
    edge_descriptors = _window_descriptors(edge_windows, sfreq_hz=sfreq_hz)
    edge = _summarize_change(edge_descriptors, starts=starts, centers=centers)

    incident_peak = route_bipolar_support_to_nodes(edge.peak_0_4, reduction="max")
    incident_mean = route_bipolar_support_to_nodes(
        edge.mean_positive_0_12, reduction="max"
    )
    incident_detected, incident_latency, incident_relative = _route_incident_latency(
        edge.detected, edge.latency_sec
    )

    features = torch.stack(
        (
            node.peak_0_4,
            node.mean_positive_0_12,
            node.persistence_0_12,
            node.change_point_strength,
            node.censored_latency_sec,
            node.detected.to(values.dtype),
            node.relative_delay_sec,
            node.dominant_frequency_shift,
            node.dominant_frequency_slope,
            node.rhythmicity_change,
            node.entropy_drop,
            node.late_minus_early,
            node.late_persistence,
            incident_peak,
            incident_mean,
            incident_latency,
            incident_detected.to(values.dtype),
            incident_relative,
            node.high_frequency_ratio_change,
            node.artifact_burden,
        ),
        dim=1,
    ).contiguous()
    if not torch.isfinite(features).all():
        raise RuntimeError("fine temporal feature construction produced non-finite values")
    return FineTemporalEvidence(
        features=features,
        feature_names=FINE_TEMPORAL_FEATURE_NAMES,
        composite_trace=node.composite.contiguous(),
        dominant_frequency_hz=node_descriptors.dominant_frequency_hz.contiguous(),
        window_center_sec=centers.contiguous(),
        node_change_detected=node.detected.contiguous(),
        node_change_latency_sec=node.latency_sec.contiguous(),
        bipolar_change_detected=edge.detected.contiguous(),
        bipolar_change_latency_sec=edge.latency_sec.contiguous(),
    )


if tuple(STANDARD_19) != tuple(channel for channel in STANDARD_19):
    raise RuntimeError("standard-19 ontology changed unexpectedly")
if len(TCP_20_EDGES) != N_TCP_EDGES:
    raise RuntimeError("TCP-20 ontology changed unexpectedly")


__all__ = [
    "FINE_BASELINE_START_SECONDS",
    "FINE_CHANGE_THRESHOLD",
    "FINE_ONSET_HORIZON_SECONDS",
    "FINE_STRIDE_SECONDS",
    "FINE_SUSTAINED_WINDOWS",
    "FINE_TEMPORAL_EVIDENCE_SCHEMA",
    "FINE_TEMPORAL_FEATURE_NAMES",
    "FINE_WINDOW_SECONDS",
    "FineTemporalEvidence",
    "extract_fine_temporal_evidence",
    "route_bipolar_support_to_nodes",
]
