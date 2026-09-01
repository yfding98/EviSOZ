"""Signal-only, abstention-capable findings for detector-selected EEG windows.

The producer consumes only the frozen standard-19 ``[-12,+48]`` second tensor.
It never reads annotations, spreadsheets, research electrode rankings or
clinical metadata.  Its default policy deliberately withholds spike/sharp-wave
morphology, electrographic onset and clinical spread terminology because no
independent morphology/spread producer has a passing promotion receipt.

Qualified output is therefore limited to reproducible quantitative scalp EEG
change candidates: involved bipolar derivations, frequency band/range,
rhythmicity, amplitude range, later derivation timing, quantitative
frequency/amplitude trajectory and return below the change threshold.  The
trajectory is explicitly not ACNS ``evolution`` and bipolar endpoints are not
promoted to maximal electrodes, regions or a source field.  Every fact remains
an ``algorithm_candidate`` and must be reviewed by an EEG reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.soz.geometry import STANDARD_19


SIGNAL_FINDINGS_RECEIPT_SCHEMA = "signal_qualified_event_findings_receipt_v1"
SIGNAL_FINDINGS_PRODUCER_ID = "quantitative_scalp_change_v1"
MORPHOLOGY_PRODUCER_PROMOTION_STATUS = "no_go"
SPATIAL_SPREAD_PRODUCER_PROMOTION_STATUS = "not_qualified"


_BIPOLAR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("FP1-F7", "FP1", "F7"),
    ("F7-T7", "F7", "T7"),
    ("T7-P7", "T7", "P7"),
    ("P7-O1", "P7", "O1"),
    ("FP1-F3", "FP1", "F3"),
    ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),
    ("P3-O1", "P3", "O1"),
    ("FP2-F8", "FP2", "F8"),
    ("F8-T8", "F8", "T8"),
    ("T8-P8", "T8", "P8"),
    ("P8-O2", "P8", "O2"),
    ("FP2-F4", "FP2", "F4"),
    ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),
    ("P4-O2", "P4", "O2"),
    ("FZ-CZ", "FZ", "CZ"),
    ("CZ-PZ", "CZ", "PZ"),
)

_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class SignalFindingPolicy:
    sampling_rate_hz: float = 200.0
    segment_duration_seconds: float = 60.0
    anchor_offset_seconds: float = 12.0
    baseline_start_seconds: float = 0.0
    baseline_stop_seconds: float = 8.0
    window_seconds: float = 1.0
    step_seconds: float = 0.5
    score_threshold: float = 4.0
    return_threshold: float = 2.0
    minimum_run_windows: int = 4
    minimum_active_derivations: int = 2
    stable_derivation_fraction: float = 0.5
    maximum_bad_channel_fraction: float = 0.2
    flat_channel_mad_uv: float = 0.25
    extreme_amplitude_uv: float = 5000.0
    extreme_step_uv: float = 2000.0
    extreme_sample_fraction: float = 0.01
    frequency_evolution_hz: float = 2.0
    amplitude_evolution_ratio: float = 1.5
    later_change_minimum_delay_seconds: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_SIGNAL_FINDING_POLICY = SignalFindingPolicy()


@dataclass(frozen=True)
class TrajectoryQualification:
    start_index: int
    stop_index_exclusive: int
    stable_derivation_indices: tuple[int, ...]


@dataclass(frozen=True)
class SignalFindingResult:
    status: str
    abstention_reason: str | None
    facts: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def qualify_sustained_trajectory(
    window_scores: np.ndarray,
    active_derivations: np.ndarray,
    *,
    score_threshold: float,
    minimum_run_windows: int,
    minimum_active_derivations: int,
    stable_derivation_fraction: float,
) -> TrajectoryQualification | None:
    """Return the first reproducible qualifying run, inclusive at threshold."""

    scores = np.asarray(window_scores, dtype=np.float64)
    active = np.asarray(active_derivations, dtype=bool)
    if scores.ndim != 1 or active.ndim != 2 or active.shape[0] != scores.size:
        raise ValueError("trajectory shapes are inconsistent")
    if not np.isfinite(scores).all():
        raise ValueError("trajectory scores must be finite")
    if minimum_run_windows < 2 or minimum_active_derivations < 1:
        raise ValueError("trajectory minimums are invalid")
    if not 0 < stable_derivation_fraction <= 1:
        raise ValueError("stable_derivation_fraction must be in (0,1]")

    eligible = (scores >= score_threshold) & (
        active.sum(axis=1) >= minimum_active_derivations
    )
    start = 0
    while start < scores.size:
        if not eligible[start]:
            start += 1
            continue
        stop = start + 1
        while stop < scores.size and eligible[stop]:
            stop += 1
        if stop - start >= minimum_run_windows:
            run_scores = scores[start:stop]
            odd = run_scores[::2]
            even = run_scores[1::2]
            stable = np.flatnonzero(
                active[start:stop].mean(axis=0) >= stable_derivation_fraction
            )
            if (
                odd.size
                and even.size
                and float(np.median(odd)) >= score_threshold
                and float(np.median(even)) >= score_threshold
                and stable.size >= minimum_active_derivations
            ):
                return TrajectoryQualification(
                    start_index=start,
                    stop_index_exclusive=stop,
                    stable_derivation_indices=tuple(int(item) for item in stable),
                )
        start = stop
    return None


def _round(value: float, digits: int = 3) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _qualification(policy: SignalFindingPolicy) -> dict[str, Any]:
    return {
        "producer_id": SIGNAL_FINDINGS_PRODUCER_ID,
        "policy_sha256": policy.sha256,
        "artifact_gate_passed": True,
        "sustained_change_gate_passed": True,
        "reproducibility_gate_passed": True,
        "source_signal_only": True,
        "external_context_used": False,
        "research_ranking_used": False,
        "morphology_terms_qualified": False,
        "spatial_spread_terms_qualified": False,
    }


def _receipt(
    *,
    policy: SignalFindingPolicy,
    analysis_scope: Mapping[str, Any],
    status: str,
    reason: str | None,
    bad_channels: Sequence[str],
    usable_derivations: int,
    candidate_window_count: int,
    emitted_fact_types: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SIGNAL_FINDINGS_RECEIPT_SCHEMA,
        "producer_id": SIGNAL_FINDINGS_PRODUCER_ID,
        "policy_sha256": policy.sha256,
        "status": status,
        "abstention_reason": reason,
        "input_scope": "processed_standard19_fixed_event_window_only",
        "analysis_scope": dict(analysis_scope),
        "quality": {
            "bad_channels": list(bad_channels),
            "usable_bipolar_derivation_count": int(usable_derivations),
            "candidate_window_count": int(candidate_window_count),
            "artifact_gate_passed": reason != "artifact_or_channel_quality_gate_failed",
        },
        "gates": {
            "sustained_change_gate_passed": status == "qualified",
            "reproducibility_gate_passed": status == "qualified",
            "morphology_producer_promotion_status": MORPHOLOGY_PRODUCER_PROMOTION_STATUS,
            "spatial_spread_producer_promotion_status": SPATIAL_SPREAD_PRODUCER_PROMOTION_STATUS,
        },
        "thresholds": policy.to_dict(),
        "emitted_fact_types": list(emitted_fact_types),
        "source_receipt": {
            "raw_eeg_used": True,
            "edf_annotations_used": False,
            "excel_used": False,
            "clinical_data_used": False,
            "research_ranking_used": False,
        },
    }


def _abstain(
    *,
    policy: SignalFindingPolicy,
    analysis_scope: Mapping[str, Any],
    reason: str,
    bad_channels: Sequence[str] = (),
    usable_derivations: int = 0,
    candidate_window_count: int = 0,
) -> SignalFindingResult:
    return SignalFindingResult(
        status="abstained",
        abstention_reason=reason,
        facts=(),
        receipt=_receipt(
            policy=policy,
            analysis_scope=analysis_scope,
            status="abstained",
            reason=reason,
            bad_channels=bad_channels,
            usable_derivations=usable_derivations,
            candidate_window_count=candidate_window_count,
            emitted_fact_types=(),
        ),
    )


def _finite_interval(
    value: object,
    *,
    context: str,
) -> tuple[float, float]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise TypeError(f"{context} must contain exactly two finite numbers")
    start, stop = (float(item) for item in value)
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError(f"{context} must contain finite numbers")
    if stop <= start:
        raise ValueError(f"{context} must be non-empty and increasing")
    return start, stop


def _resolve_analysis_scope(
    *,
    policy: SignalFindingPolicy,
    candidate_support_interval_relative_to_anchor: tuple[float, float] | None,
    evidence_interval_seconds_relative_to_anchor: tuple[float, float] | None,
    evidence_anchor_offset_seconds: float | None,
) -> dict[str, Any]:
    """Resolve optional evidence coordinates onto the frozen segment timebase.

    ``candidate_support_interval_relative_to_anchor`` is the legacy detector
    support interval relative to the fixed candidate anchor.  The optional
    evidence interval may instead be relative to a refined evidence anchor.
    When both are supplied, analysis is restricted to their intersection.
    Missing intervals deliberately mean the entire immutable carrier, and the
    choice is persisted in the receipt rather than being inferred later.
    """

    if evidence_anchor_offset_seconds is None:
        evidence_anchor = float(policy.anchor_offset_seconds)
        anchor_source = "fixed_carrier_default"
    else:
        if isinstance(evidence_anchor_offset_seconds, bool) or not isinstance(
            evidence_anchor_offset_seconds, (int, float)
        ):
            raise TypeError("evidence anchor offset must be a finite number")
        evidence_anchor = float(evidence_anchor_offset_seconds)
        if not math.isfinite(evidence_anchor):
            raise ValueError("evidence anchor offset must be finite")
        anchor_source = "explicit"
    if not 0.0 <= evidence_anchor <= policy.segment_duration_seconds:
        raise ValueError("evidence anchor lies outside the fixed EEG carrier")

    carrier = (0.0, float(policy.segment_duration_seconds))
    candidate_local: tuple[float, float] | None = None
    candidate_relative: tuple[float, float] | None = None
    if candidate_support_interval_relative_to_anchor is not None:
        candidate_relative = _finite_interval(
            candidate_support_interval_relative_to_anchor,
            context="candidate support interval relative to anchor",
        )
        candidate_local = (
            candidate_relative[0] + float(policy.anchor_offset_seconds),
            candidate_relative[1] + float(policy.anchor_offset_seconds),
        )
        if candidate_local[0] < carrier[0] or candidate_local[1] > carrier[1]:
            raise ValueError("candidate support lies outside the fixed EEG carrier")

    evidence_local: tuple[float, float] | None = None
    evidence_relative: tuple[float, float] | None = None
    if evidence_interval_seconds_relative_to_anchor is not None:
        evidence_relative = _finite_interval(
            evidence_interval_seconds_relative_to_anchor,
            context="evidence interval relative to anchor",
        )
        evidence_local = (
            evidence_relative[0] + evidence_anchor,
            evidence_relative[1] + evidence_anchor,
        )
        if evidence_local[0] < carrier[0] or evidence_local[1] > carrier[1]:
            raise ValueError("evidence interval lies outside the fixed EEG carrier")

    constraints = [item for item in (candidate_local, evidence_local) if item is not None]
    effective_start = max([carrier[0], *(item[0] for item in constraints)])
    effective_stop = min([carrier[1], *(item[1] for item in constraints)])
    if effective_stop <= effective_start:
        raise ValueError("candidate and evidence intervals do not overlap")

    return {
        "timebase": "processed_segment_start_seconds",
        "segment_duration_seconds": float(policy.segment_duration_seconds),
        "fixed_candidate_anchor_offset_seconds": float(policy.anchor_offset_seconds),
        "evidence_anchor_offset_seconds": evidence_anchor,
        "evidence_anchor_source": anchor_source,
        "candidate_support_interval_relative_to_fixed_anchor": (
            list(candidate_relative) if candidate_relative is not None else None
        ),
        "evidence_interval_seconds_relative_to_anchor": (
            list(evidence_relative) if evidence_relative is not None else None
        ),
        "effective_interval_seconds_in_segment": [effective_start, effective_stop],
        "default_full_carrier_used": not constraints,
        "interval_combination_policy": "intersection",
    }


def _window_features(
    data_uv: np.ndarray,
    starts: np.ndarray,
    *,
    policy: SignalFindingPolicy,
) -> dict[str, np.ndarray]:
    window_samples = int(round(policy.window_seconds * policy.sampling_rate_hz))
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / policy.sampling_rate_hz)
    taper = np.hanning(window_samples).astype(np.float64)
    rms_rows: list[np.ndarray] = []
    line_rows: list[np.ndarray] = []
    band_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    peak_frequency_rows: list[np.ndarray] = []
    concentration_rows: list[np.ndarray] = []
    amplitude_rows: list[np.ndarray] = []
    analysis_mask = (frequencies >= 0.5) & (frequencies <= 45.0)
    peak_mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    band_masks = [
        (frequencies >= low) & (frequencies < high)
        for _, low, high in _BANDS
    ]
    for start in starts:
        sample = int(round(float(start) * policy.sampling_rate_hz))
        window = data_uv[:, sample : sample + window_samples]
        centered = window - np.mean(window, axis=1, keepdims=True)
        rms_rows.append(np.sqrt(np.mean(centered * centered, axis=1) + 1e-12))
        line_rows.append(np.mean(np.abs(np.diff(centered, axis=1)), axis=1) + 1e-12)
        amplitude_rows.append(np.ptp(centered, axis=1))
        spectrum = np.abs(np.fft.rfft(centered * taper, axis=1)) ** 2
        total = np.sum(spectrum[:, analysis_mask], axis=1) + 1e-12
        band_rows.append(
            np.stack(
                [np.sum(spectrum[:, mask], axis=1) / total for mask in band_masks],
                axis=1,
            )
        )
        normalized = spectrum[:, analysis_mask] / total[:, None]
        entropy_rows.append(
            -np.sum(normalized * np.log(normalized + 1e-12), axis=1)
            / math.log(max(2, int(np.count_nonzero(analysis_mask))))
        )
        peak_power = spectrum[:, peak_mask]
        peak_indices = np.argmax(peak_power, axis=1)
        peak_frequency_rows.append(frequencies[peak_mask][peak_indices])
        concentration_rows.append(
            np.max(peak_power, axis=1) / (np.sum(peak_power, axis=1) + 1e-12)
        )
    return {
        "rms": np.stack(rms_rows),
        "line_length": np.stack(line_rows),
        "band_ratio": np.stack(band_rows),
        "spectral_entropy": np.stack(entropy_rows),
        "peak_frequency": np.stack(peak_frequency_rows),
        "spectral_concentration": np.stack(concentration_rows),
        "amplitude": np.stack(amplitude_rows),
    }


def measure_native_signal_window_features(
    data_uv: np.ndarray,
    starts_seconds: Sequence[float],
    *,
    sampling_rate_hz: float,
    window_seconds: float = 1.0,
) -> dict[str, np.ndarray]:
    """Expose the established numerical window kernel for native producers.

    This is deliberately a measurement-only API.  It accepts a finite
    ``[unit,time]`` array in microvolts and returns the same RMS, line-length,
    relative-band-power, entropy, peak-frequency, spectral-concentration and
    peak-to-peak arrays used by :func:`extract_signal_qualified_event_findings`.
    It does not qualify a seizure, morphology term, onset, or SOZ.

    Keeping this thin public wrapper prevents adaptive event analyzers from
    silently cloning a second spectral/amplitude implementation while letting
    them operate on common-17 and ragged physical supports.
    """

    values = np.asarray(data_uv, dtype=np.float64)
    starts = np.asarray(tuple(starts_seconds), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("native window features require [unit,time] data")
    if not np.isfinite(values).all():
        raise ValueError("native window features require finite data")
    if starts.ndim != 1 or starts.size < 1 or not np.isfinite(starts).all():
        raise ValueError("native window starts must be a finite non-empty vector")
    rate = float(sampling_rate_hz)
    duration = float(window_seconds)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("window_seconds must be positive and finite")
    window_samples = int(round(rate * duration))
    if window_samples < 4:
        raise ValueError("native feature windows require at least four samples")
    last_stop = int(round(float(starts[-1]) * rate)) + window_samples
    if float(starts[0]) < 0.0 or last_stop > values.shape[1]:
        raise ValueError("native feature window lies outside the supplied tensor")
    policy = SignalFindingPolicy(
        sampling_rate_hz=rate,
        segment_duration_seconds=values.shape[1] / rate,
        anchor_offset_seconds=0.0,
        baseline_start_seconds=0.0,
        baseline_stop_seconds=min(duration, values.shape[1] / rate),
        window_seconds=duration,
    )
    return _window_features(values, starts, policy=policy)


def _robust_z(values: np.ndarray, baseline: np.ndarray, floor: float) -> np.ndarray:
    center = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - center), axis=0)
    scale = np.maximum(1.4826 * mad, floor)
    return np.maximum(0.0, (values - center) / scale)


def _change_scores(
    features: Mapping[str, np.ndarray], baseline_mask: np.ndarray
) -> np.ndarray:
    log_rms = np.log(features["rms"] + 1e-12)
    log_line = np.log(features["line_length"] + 1e-12)
    rms_z = _robust_z(log_rms, log_rms[baseline_mask], 0.12)
    line_z = _robust_z(log_line, log_line[baseline_mask], 0.12)
    baseline_band = features["band_ratio"][baseline_mask]
    band_center = np.median(baseline_band, axis=0)
    band_mad = np.median(np.abs(baseline_band - band_center), axis=0)
    band_scale = np.maximum(1.4826 * band_mad, 0.04)
    band_z = np.max(
        np.abs(features["band_ratio"] - band_center) / band_scale,
        axis=2,
    )
    entropy_center = np.median(features["spectral_entropy"][baseline_mask], axis=0)
    entropy_drop = np.maximum(
        0.0, (entropy_center - features["spectral_entropy"]) / 0.08
    )
    components = np.stack((rms_z, line_z, band_z, entropy_drop), axis=2)
    # The second strongest component is used so a single unstable feature
    # cannot qualify a clinical-facing signal observation.
    return np.sort(np.clip(components, 0.0, 20.0), axis=2)[:, :, -2]


def _rank_derivations(
    derivation_indices: Sequence[int],
    derivation_scores: np.ndarray,
) -> list[str]:
    order = sorted(
        derivation_indices,
        key=lambda index: (-float(derivation_scores[index]), _BIPOLAR_PAIRS[index][0]),
    )[:4]
    return [_BIPOLAR_PAIRS[index][0] for index in order]


def extract_signal_qualified_event_findings(
    eeg: torch.Tensor,
    *,
    candidate_support_interval_relative_to_anchor: tuple[float, float] | None = None,
    evidence_interval_seconds_relative_to_anchor: tuple[float, float] | None = None,
    evidence_anchor_offset_seconds: float | None = None,
    policy: SignalFindingPolicy = DEFAULT_SIGNAL_FINDING_POLICY,
) -> SignalFindingResult:
    """Extract qualified quantitative event findings or explicitly abstain.

    All coordinates are resolved onto the immutable processed-segment
    timebase.  With no candidate/evidence interval, the full fixed carrier is
    the explicit default; baseline windows are still excluded from candidate
    qualification.  An optional refined evidence interval can be supplied
    relative to an explicit evidence anchor without changing the tensor.
    """

    signal = eeg.detach().cpu().to(torch.float64).contiguous()
    expected_samples = int(
        round(policy.segment_duration_seconds * policy.sampling_rate_hz)
    )
    if tuple(signal.shape) != (len(STANDARD_19), expected_samples):
        raise ValueError("signal findings require standard-19 fixed-duration EEG")
    if not torch.isfinite(signal).all():
        raise ValueError("signal findings require finite EEG")
    analysis_scope = _resolve_analysis_scope(
        policy=policy,
        candidate_support_interval_relative_to_anchor=(
            candidate_support_interval_relative_to_anchor
        ),
        evidence_interval_seconds_relative_to_anchor=(
            evidence_interval_seconds_relative_to_anchor
        ),
        evidence_anchor_offset_seconds=evidence_anchor_offset_seconds,
    )
    support_start, support_stop = analysis_scope[
        "effective_interval_seconds_in_segment"
    ]

    values_uv = signal.numpy() * 1e6
    channel_center = np.median(values_uv, axis=1, keepdims=True)
    channel_mad = np.median(np.abs(values_uv - channel_center), axis=1)
    extreme_fraction = np.mean(
        np.abs(values_uv - channel_center) >= policy.extreme_amplitude_uv,
        axis=1,
    )
    maximum_step = np.max(np.abs(np.diff(values_uv, axis=1)), axis=1)
    bad_mask = (
        (channel_mad < policy.flat_channel_mad_uv)
        | (extreme_fraction > policy.extreme_sample_fraction)
        | (maximum_step > policy.extreme_step_uv)
    )
    bad_channels = [
        electrode for electrode, bad in zip(STANDARD_19, bad_mask) if bool(bad)
    ]
    if float(np.mean(bad_mask)) > policy.maximum_bad_channel_fraction:
        return _abstain(
            policy=policy,
            analysis_scope=analysis_scope,
            reason="artifact_or_channel_quality_gate_failed",
            bad_channels=bad_channels,
        )

    channel_index = {name: index for index, name in enumerate(STANDARD_19)}
    usable_pairs = [
        pair
        for pair in _BIPOLAR_PAIRS
        if not bad_mask[channel_index[pair[1]]]
        and not bad_mask[channel_index[pair[2]]]
    ]
    if len(usable_pairs) < policy.minimum_active_derivations:
        return _abstain(
            policy=policy,
            analysis_scope=analysis_scope,
            reason="artifact_or_channel_quality_gate_failed",
            bad_channels=bad_channels,
            usable_derivations=len(usable_pairs),
        )
    bipolar_uv = np.stack(
        [
            values_uv[channel_index[first]] - values_uv[channel_index[second]]
            for _, first, second in usable_pairs
        ]
    )
    # Preserve global pair indices for stable clinical labels.
    global_pair_indices = [
        next(index for index, item in enumerate(_BIPOLAR_PAIRS) if item == pair)
        for pair in usable_pairs
    ]

    last_start = policy.segment_duration_seconds - policy.window_seconds
    starts = np.arange(
        0.0,
        last_start + policy.step_seconds / 2,
        policy.step_seconds,
        dtype=np.float64,
    )
    baseline_mask = (
        (starts >= policy.baseline_start_seconds)
        & (starts + policy.window_seconds <= policy.baseline_stop_seconds)
    )
    candidate_mask = (
        (starts >= support_start)
        & (starts + policy.window_seconds <= support_stop)
        & ~baseline_mask
    )
    if np.count_nonzero(baseline_mask) < 4 or np.count_nonzero(candidate_mask) < 2:
        return _abstain(
            policy=policy,
            analysis_scope=analysis_scope,
            reason="insufficient_baseline_or_candidate_windows",
            bad_channels=bad_channels,
            usable_derivations=len(usable_pairs),
            candidate_window_count=int(np.count_nonzero(candidate_mask)),
        )
    features = _window_features(bipolar_uv, starts, policy=policy)
    per_derivation_scores = _change_scores(features, baseline_mask)
    active = per_derivation_scores >= policy.score_threshold
    aggregate = np.zeros(starts.size, dtype=np.float64)
    for index in range(starts.size):
        row = per_derivation_scores[index]
        active_count = int(np.count_nonzero(active[index]))
        if active_count < policy.minimum_active_derivations:
            continue
        top = np.sort(row)[-min(3, row.size) :]
        aggregate[index] = float(np.median(top))
    candidate_indices = np.flatnonzero(candidate_mask)
    trajectory = qualify_sustained_trajectory(
        aggregate[candidate_indices],
        active[candidate_indices],
        score_threshold=policy.score_threshold,
        minimum_run_windows=policy.minimum_run_windows,
        minimum_active_derivations=policy.minimum_active_derivations,
        stable_derivation_fraction=policy.stable_derivation_fraction,
    )
    if trajectory is None:
        return _abstain(
            policy=policy,
            analysis_scope=analysis_scope,
            reason="sustained_or_reproducibility_gate_not_met",
            bad_channels=bad_channels,
            usable_derivations=len(usable_pairs),
            candidate_window_count=int(candidate_indices.size),
        )

    run_indices = candidate_indices[
        trajectory.start_index : trajectory.stop_index_exclusive
    ]
    stable_local = list(trajectory.stable_derivation_indices)
    stable_scores = np.median(per_derivation_scores[run_indices], axis=0)
    stable_global = [global_pair_indices[index] for index in stable_local]
    global_scores = np.zeros(len(_BIPOLAR_PAIRS), dtype=np.float64)
    for local_index, global_index in enumerate(global_pair_indices):
        global_scores[global_index] = stable_scores[local_index]
    derivations = _rank_derivations(stable_global, global_scores)
    selected_local = [
        global_pair_indices.index(index)
        for index in stable_global
        if index in global_pair_indices
    ]
    selected_peak = features["peak_frequency"][np.ix_(run_indices, selected_local)]
    selected_concentration = features["spectral_concentration"][
        np.ix_(run_indices, selected_local)
    ]
    selected_amplitude = features["amplitude"][np.ix_(run_indices, selected_local)]
    median_peak_frequency = float(np.median(selected_peak))
    band_index = next(
        (
            index
            for index, (_, low, high) in enumerate(_BANDS)
            if low <= median_peak_frequency < high
        ),
        len(_BANDS) - 1,
    )
    frequency_band = _BANDS[band_index][0]
    concentration = float(np.median(selected_concentration))
    rhythmicity = (
        "rhythmic"
        if concentration >= 0.25
        else "quasi_rhythmic"
        if concentration >= 0.12
        else "nonrhythmic"
    )
    # Report a frequency range that is semantically contained in the declared
    # dominant band.  Cross-band window peaks remain part of the trajectory
    # qualification but must not yield contradictions such as “delta,
    # 1–15 Hz”.  With an even number of observations, NumPy may interpolate
    # the median between peaks from two different bands.  The interpolated
    # value can fall in a third band that has no observed supporting peak; that
    # is an evidence-level abstention, not a technical failure of the record.
    _, band_low, band_high = _BANDS[band_index]
    band_peaks = selected_peak[
        (selected_peak >= band_low) & (selected_peak < band_high)
    ]
    if band_peaks.size == 0:
        return _abstain(
            policy=policy,
            analysis_scope=analysis_scope,
            reason="dominant_frequency_band_not_supported",
            bad_channels=bad_channels,
            usable_derivations=len(usable_pairs),
            candidate_window_count=int(candidate_indices.size),
        )
    frequency_low = _round(float(np.quantile(band_peaks, 0.1)), 1)
    frequency_high = _round(float(np.quantile(band_peaks, 0.9)), 1)
    amplitude_low = _round(float(np.quantile(selected_amplitude, 0.1)), 1)
    amplitude_high = _round(float(np.quantile(selected_amplitude, 0.9)), 1)
    start_seconds = _round(float(starts[run_indices[0]]))
    end_seconds = _round(
        float(starts[run_indices[-1]] + policy.window_seconds)
    )
    qualification = _qualification(policy)
    band_zh = {
        "delta": "δ",
        "theta": "θ",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
    }[frequency_band]
    rhythm_zh = {
        "rhythmic": "节律性",
        "quasi_rhythmic": "近节律性",
        "nonrhythmic": "非节律性",
    }[rhythmicity]
    sustained_value: dict[str, Any] = {
        "start_offset_seconds": start_seconds,
        "end_offset_seconds": end_seconds,
        "derivations": derivations,
        "frequency_hz": {"min": frequency_low, "max": frequency_high},
        "frequency_band": frequency_band,
        "amplitude_uv": {"min": amplitude_low, "max": amplitude_high},
        "rhythmicity": rhythmicity,
        "qualification": qualification,
        "text_zh": (
            f"待复核候选：信号门控在{'、'.join(derivations)}识别到"
            f"持续{band_zh}频段{rhythm_zh}变化，主频"
            f"{frequency_low:g}–{frequency_high:g} Hz；仅表示量化头皮信号变化。"
        ),
    }
    facts: list[dict[str, Any]] = [
        {
            "fact_type": "algorithmic_sustained_eeg_change",
            "value": sustained_value,
            "method": "固结双极量化特征、伪迹门控与跨窗口可重复性检测",
        }
    ]

    # A neutral quantitative trajectory is retained only when the first and
    # last halves differ by a frozen threshold.  This is not ACNS evolution:
    # it contains one comparison only, and amplitude change alone never
    # establishes evolution.  Morphology remains NO-GO.
    if run_indices.size >= 4:
        midpoint = run_indices.size // 2
        early = run_indices[:midpoint]
        late = run_indices[midpoint:]
        early_frequency = float(
            np.median(features["peak_frequency"][np.ix_(early, selected_local)])
        )
        late_frequency = float(
            np.median(features["peak_frequency"][np.ix_(late, selected_local)])
        )
        early_amplitude = float(
            np.median(features["amplitude"][np.ix_(early, selected_local)])
        )
        late_amplitude = float(
            np.median(features["amplitude"][np.ix_(late, selected_local)])
        )
        dimensions: list[str] = []
        if abs(late_frequency - early_frequency) >= policy.frequency_evolution_hz:
            dimensions.append("frequency")
        amplitude_ratio = late_amplitude / max(early_amplitude, 1e-6)
        if (
            amplitude_ratio >= policy.amplitude_evolution_ratio
            or amplitude_ratio <= 1.0 / policy.amplitude_evolution_ratio
        ):
            dimensions.append("amplitude")
        if dimensions:
            sustained_value["quantitative_trajectory"] = {
                "comparison_offset_seconds": _round(
                    float(starts[late[0]] - starts[run_indices[0]])
                ),
                "change_dimensions": dimensions,
                "early_frequency_hz": _round(early_frequency, 1),
                "late_frequency_hz": _round(late_frequency, 1),
                "early_amplitude_uv": _round(early_amplitude, 1),
                "late_amplitude_uv": _round(late_amplitude, 1),
                "amplitude_change_alone_is_not_ictal_evolution": True,
            }

    # Later observations remain derivation-level temporal relations.  A
    # bipolar endpoint is not a maximal electrode, source field or spread.
    first_active = set(np.flatnonzero(active[run_indices[0]]))
    later_observations: list[dict[str, Any]] = []
    seen_derivations: set[str] = set(derivations)
    for local_index in range(active.shape[1]):
        if local_index in first_active:
            continue
        active_run = np.flatnonzero(active[run_indices, local_index])
        if active_run.size < 2:
            continue
        first_position = int(active_run[0])
        delay = float(starts[run_indices[first_position]] - starts[run_indices[0]])
        if delay < policy.later_change_minimum_delay_seconds:
            continue
        derivation = usable_pairs[local_index][0]
        if derivation in seen_derivations:
            continue
        later_observations.append(
            {"derivation": derivation, "delay_seconds": _round(delay)}
        )
        seen_derivations.add(derivation)
        if len(later_observations) >= 3:
            break
    if later_observations:
        sustained_value["later_derivation_changes"] = later_observations

    # A return below threshold is searched after the qualified run.  It is an
    # neutral return-to-baseline candidate, not a confirmed seizure offset.
    first_after = int(run_indices[-1] + 1)
    required_return_windows = policy.minimum_run_windows
    termination_index: int | None = None
    for index in range(first_after, starts.size - required_return_windows + 1):
        if not np.all(candidate_mask[index : index + required_return_windows]):
            continue
        block = aggregate[index : index + required_return_windows]
        if np.all(block < policy.return_threshold):
            termination_index = index
            break
    if termination_index is not None:
        offset = _round(float(starts[termination_index] - starts[run_indices[0]]))
        if offset > 0:
            sustained_value["candidate_return_to_baseline_offset_seconds"] = offset

    fact_types = [str(item["fact_type"]) for item in facts]
    receipt = _receipt(
        policy=policy,
        analysis_scope=analysis_scope,
        status="qualified",
        reason=None,
        bad_channels=bad_channels,
        usable_derivations=len(usable_pairs),
        candidate_window_count=int(candidate_indices.size),
        emitted_fact_types=fact_types,
    )
    receipt["qualified_interval_seconds_in_segment"] = [start_seconds, end_seconds]
    evidence_anchor = float(analysis_scope["evidence_anchor_offset_seconds"])
    receipt["qualified_interval_seconds_relative_to_evidence_anchor"] = [
        _round(start_seconds - evidence_anchor),
        _round(end_seconds - evidence_anchor),
    ]
    receipt["qualified_derivations"] = derivations
    receipt["frequency_band"] = frequency_band
    receipt["rhythmicity"] = rhythmicity
    return SignalFindingResult(
        status="qualified",
        abstention_reason=None,
        facts=tuple(facts),
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_SIGNAL_FINDING_POLICY",
    "MORPHOLOGY_PRODUCER_PROMOTION_STATUS",
    "SIGNAL_FINDINGS_PRODUCER_ID",
    "SIGNAL_FINDINGS_RECEIPT_SCHEMA",
    "SPATIAL_SPREAD_PRODUCER_PROMOTION_STATUS",
    "SignalFindingPolicy",
    "SignalFindingResult",
    "TrajectoryQualification",
    "extract_signal_qualified_event_findings",
    "measure_native_signal_window_features",
    "qualify_sustained_trajectory",
]
