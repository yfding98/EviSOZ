"""Continuous target-blind coarse sentinel cache for common-17 EEG.

This module continuously screens every physical sample in a caller-supplied
legal horizon.  One-second base cells form an exact, gap-free partition and
are deterministically aggregated into four- and sixteen-second cells.  The
cache may only propose intervals for later native-EEG materialization.  It is
not an event detector, an onset estimator, a clinical Finding producer, or a
channel/SOZ ranker.

The coarse pass and any later native fine analysis have separate accounting
ledgers.  In particular, samples consumed by this cache were read and may not
be advertised as "unread" merely because the downstream fine analyzer has not
materialized them.

Only common-17 EEG arrays, a sample-level EEG-derived validity mask, sampling
parameters, opaque identifiers and a legal sample interval have an input
route.  There is no annotation, label, spreadsheet, clinical text, behavior,
sleep, provocation, auxiliary physiology or LLM input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .adaptive_native_evidence_common17 import COMMON17_CHANNELS


SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_common17_continuous_coarse_sentinel_cache_v1"
)
METHOD_ID: Final[str] = (
    "COMMON17-CONTINUOUS-1S-4S-16S-COARSE-SENTINEL-CACHE-V1"
)

_SCALE_SECONDS: Final[tuple[int, ...]] = (1, 4, 16)
_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)
_PRIMITIVE_KEYS: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
    "delta_relative_power",
    "theta_relative_power",
    "alpha_relative_power",
    "beta_relative_power",
    "gamma_relative_power",
    "dominant_frequency_hz",
    "spectral_entropy",
    "spectral_concentration",
)
_LOG_CHANGE_KEYS: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "line_length_uv_per_sample",
)
_SPECTRAL_CHANGE_KEYS: Final[tuple[str, ...]] = (
    "delta_relative_power",
    "theta_relative_power",
    "alpha_relative_power",
    "beta_relative_power",
    "gamma_relative_power",
)
_AUTHORIZATION: Final[dict[str, object]] = {
    "output_namespace": "coarse_native_query_screening_proposal_only",
    "may_trigger_downstream_native_query": True,
    "may_assert_eeg_finding": False,
    "may_assert_clinical_term": False,
    "may_assert_seizure": False,
    "may_assert_onset_or_offset": False,
    "may_rank_channels_regions_or_laterality": False,
    "may_assert_SOZ_EZ_or_diagnosis": False,
    "may_enter_clinical_report_as_fact": False,
}
_SCOPE: Final[dict[str, object]] = {
    "common17_EEG_samples_used": True,
    "sampling_rate_recording_length_and_legal_horizon_used": True,
    "EEG_derived_sample_QC_used_if_supplied": True,
    "detector_score_or_anchor_used": False,
    "TERM_or_reference_seizure_interval_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_or_sidecar_used": False,
    "spreadsheet_doctor_or_clinical_text_used": False,
    "patient_metadata_history_video_or_behaviour_used": False,
    "sleep_activation_or_provocation_label_used": False,
    "ECG_EMG_EOG_or_auxiliary_signal_used": False,
    "FZ_or_PZ_samples_used": False,
    "zero_fill_interpolation_or_missing_electrode_synthesis_used": False,
    "LLM_output_used": False,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray, *, domain: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _round(value: float, digits: int = 8) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty identifier")
    return value


def _finite(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} is outside its allowed range")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ContinuousCoarseSentinelPolicyV1:
    """Frozen engineering defaults for a screening-only cache."""

    base_cell_seconds: float = 1.0
    aggregation_scales_seconds: tuple[int, int] = (4, 16)
    minimum_sampling_rate_hz: float = 16.0
    minimum_time_domain_samples: int = 8
    minimum_spectral_samples: int = 16
    minimum_qc_valid_fraction: float = 0.90
    repeated_transition_tolerance_volts: float = 1.0e-15
    channel_change_score_threshold: float = 2.5
    global_change_score_threshold: float = 3.0
    minimum_active_channels: int = 2
    global_top_channel_count: int = 3
    log_amplitude_change_floor: float = 0.35
    relative_band_l1_change_floor: float = 0.20
    dominant_frequency_change_floor_hz: float = 2.0
    spectral_entropy_change_floor: float = 0.08
    spectral_concentration_change_floor: float = 0.08

    def __post_init__(self) -> None:
        if self.base_cell_seconds != 1.0:
            raise ValueError("continuous sentinel v1 requires one-second base cells")
        if self.aggregation_scales_seconds != (4, 16):
            raise ValueError("continuous sentinel v1 requires four/sixteen-second aggregation")
        if self.minimum_sampling_rate_hz < 8.0:
            raise ValueError("minimum sampling rate is unsafe")
        if not 0.0 < self.minimum_qc_valid_fraction <= 1.0:
            raise ValueError("minimum QC fraction must be in (0,1]")
        if self.minimum_active_channels < 1:
            raise ValueError("minimum active channels must be positive")
        if not 1 <= self.global_top_channel_count <= len(COMMON17_CHANNELS):
            raise ValueError("global top-channel count is invalid")
        for value in (
            self.repeated_transition_tolerance_volts,
            self.channel_change_score_threshold,
            self.global_change_score_threshold,
            self.log_amplitude_change_floor,
            self.relative_band_l1_change_floor,
            self.dominant_frequency_change_floor_hz,
            self.spectral_entropy_change_floor,
            self.spectral_concentration_change_floor,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("continuous sentinel policy values must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "aggregation_scales_seconds": list(self.aggregation_scales_seconds),
            "threshold_source": (
                "frozen_engineering_screening_defaults_not_TERM_SOZ_or_clinical_tuned"
            ),
            "clinical_term_qualification_authorized": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_POLICY = ContinuousCoarseSentinelPolicyV1()


def _clock_aligned_intervals(
    start_sample: int,
    stop_sample: int,
    *,
    rate: float,
    scale_seconds: int,
) -> tuple[tuple[int, int], ...]:
    """Clip a recording-clock-aligned grid to one legal horizon.

    Grid boundaries are derived from recording sample zero, never from the
    candidate horizon start.  Moving a navigation anchor therefore exposes
    more or fewer already frozen cells instead of translating every coarse
    transition boundary with the anchor.
    """

    intervals: list[tuple[int, int]] = []
    left = start_sample
    period = float(scale_seconds) * rate
    boundary_index = int(math.floor(left / period)) + 1
    while left < stop_sample:
        nominal_right = int(round(boundary_index * period))
        while nominal_right <= left:
            boundary_index += 1
            nominal_right = int(round(boundary_index * period))
        right = min(stop_sample, nominal_right)
        intervals.append((left, right))
        left = right
        boundary_index += 1
    return tuple(intervals)


def _spectral_primitives(
    values_uv: np.ndarray,
    *,
    rate: float,
) -> dict[str, float | None]:
    count = values_uv.size
    centered = values_uv - float(np.mean(values_uv))
    taper = np.hanning(count).astype(np.float64)
    frequencies = np.fft.rfftfreq(count, d=1.0 / rate)
    power = np.abs(np.fft.rfft(centered * taper)) ** 2
    high = min(45.0, 0.45 * rate)
    analysis = (frequencies >= 0.5) & (frequencies <= high)
    if int(np.count_nonzero(analysis)) < 2:
        return {
            **{f"{name}_relative_power": None for name, _, _ in _BANDS},
            "dominant_frequency_hz": None,
            "spectral_entropy": None,
            "spectral_concentration": None,
        }
    total = float(np.sum(power[analysis]))
    if not math.isfinite(total) or total <= 1.0e-18:
        return {
            **{f"{name}_relative_power": None for name, _, _ in _BANDS},
            "dominant_frequency_hz": None,
            "spectral_entropy": None,
            "spectral_concentration": None,
        }
    normalized = power[analysis] / total
    entropy = -float(np.sum(normalized * np.log(normalized + 1.0e-15))) / math.log(
        max(2, int(np.count_nonzero(analysis)))
    )
    analysis_frequencies = frequencies[analysis]
    peak_index = int(np.argmax(power[analysis]))
    result: dict[str, float | None] = {}
    for name, low, band_high in _BANDS:
        mask = (frequencies >= low) & (frequencies < min(band_high, high + 1.0e-12))
        result[f"{name}_relative_power"] = (
            _round(float(np.sum(power[mask])) / total)
            if bool(np.any(mask)) and low < high
            else None
        )
    result.update(
        {
            "dominant_frequency_hz": _round(float(analysis_frequencies[peak_index])),
            "spectral_entropy": _round(float(np.clip(entropy, 0.0, 1.0))),
            "spectral_concentration": _round(
                float(np.clip(np.max(power[analysis]) / total, 0.0, 1.0))
            ),
        }
    )
    return result


def _measure_channel_cell(
    signal_volts: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rate: float,
    policy: ContinuousCoarseSentinelPolicyV1,
) -> dict[str, Any]:
    valid = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(signal_volts, dtype=np.float64)
    count = values.size
    valid_count = int(np.count_nonzero(valid))
    usable_fraction = valid_count / max(1, count)
    valid_pairs = valid[:-1] & valid[1:] if count > 1 else np.zeros(0, dtype=bool)
    pair_count = int(np.count_nonzero(valid_pairs))
    repeated_fraction = (
        float(
            np.mean(
                np.abs(np.diff(values)[valid_pairs])
                <= policy.repeated_transition_tolerance_volts
            )
        )
        if pair_count
        else None
    )
    time_evaluable = (
        usable_fraction >= policy.minimum_qc_valid_fraction
        and valid_count >= policy.minimum_time_domain_samples
        and pair_count >= max(1, policy.minimum_time_domain_samples - 1)
    )
    spectral_evaluable = (
        time_evaluable
        and valid_count == count
        and count >= policy.minimum_spectral_samples
    )
    primitives: dict[str, float | None] = {
        key: None for key in _PRIMITIVE_KEYS
    }
    if time_evaluable:
        selected = values[valid] * 1.0e6
        centered = selected - float(np.mean(selected))
        differences_uv = np.diff(values)[valid_pairs] * 1.0e6
        primitives.update(
            {
                "rms_uv": _round(float(np.sqrt(np.mean(centered * centered)))),
                "peak_to_peak_uv": _round(float(np.ptp(selected))),
                "line_length_uv_per_sample": _round(
                    float(np.mean(np.abs(differences_uv)))
                ),
            }
        )
    if spectral_evaluable:
        primitives.update(_spectral_primitives(values * 1.0e6, rate=rate))
    return {
        "usable_fraction": _round(usable_fraction),
        "valid_sample_count": valid_count,
        "valid_adjacent_pair_count": pair_count,
        "repeated_value_transition_fraction": (
            _round(repeated_fraction) if repeated_fraction is not None else None
        ),
        "time_domain_evaluable": bool(time_evaluable),
        "spectral_evaluable": bool(spectral_evaluable),
        "quality_semantics": "signal_derived_screening_proxy_not_clinical_artifact_label",
        "primitives": primitives,
    }


def _build_base_cells(
    signal_volts: np.ndarray,
    valid_mask: np.ndarray,
    *,
    horizon_start_sample: int,
    rate: float,
    policy: ContinuousCoarseSentinelPolicyV1,
) -> list[dict[str, Any]]:
    intervals = _clock_aligned_intervals(
        horizon_start_sample,
        horizon_start_sample + signal_volts.shape[1],
        rate=rate,
        scale_seconds=1,
    )
    rows: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(intervals):
        local_start = start - horizon_start_sample
        local_stop = stop - horizon_start_sample
        channels = []
        for channel_index, channel in enumerate(COMMON17_CHANNELS):
            measurement = _measure_channel_cell(
                signal_volts[channel_index, local_start:local_stop],
                valid_mask[channel_index, local_start:local_stop],
                rate=rate,
                policy=policy,
            )
            channels.append({"channel": channel, **measurement})
        rows.append(
            {
                "cell_id": f"B{index:06d}",
                "ordinal": index,
                "interval_samples": [start, stop],
                "interval_recording_seconds": [
                    _round(start / rate),
                    _round(stop / rate),
                ],
                "duration_seconds": _round((stop - start) / rate),
                "nominal_scale_seconds": 1,
                "complete_nominal_cell": bool(
                    abs((stop - start) / rate - 1.0) <= 1.0 / rate + 1.0e-12
                ),
                "per_channel": channels,
            }
        )
    return rows


def _median_or_none(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return _round(float(np.median(available))) if available else None


def _aggregate_cells(
    base_cells: Sequence[Mapping[str, Any]],
    *,
    scale_seconds: int,
    rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon_start = int(base_cells[0]["interval_samples"][0])
    horizon_stop = int(base_cells[-1]["interval_samples"][1])
    aggregate_intervals = _clock_aligned_intervals(
        horizon_start,
        horizon_stop,
        rate=rate,
        scale_seconds=scale_seconds,
    )
    base_index = 0
    for ordinal, (start, stop) in enumerate(aggregate_intervals):
        group: list[Mapping[str, Any]] = []
        while base_index < len(base_cells):
            source = base_cells[base_index]
            source_start = int(source["interval_samples"][0])
            source_stop = int(source["interval_samples"][1])
            if source_start >= stop:
                break
            if source_start < start or source_stop > stop:
                raise RuntimeError("clock-aligned aggregate split a one-second base cell")
            group.append(source)
            base_index += 1
        if not group:
            raise RuntimeError("clock-aligned aggregate interval has no base cells")
        per_channel: list[dict[str, Any]] = []
        for channel_index, channel in enumerate(COMMON17_CHANNELS):
            source = [row["per_channel"][channel_index] for row in group]
            primitives = {
                key: _median_or_none([item["primitives"][key] for item in source])
                for key in _PRIMITIVE_KEYS
            }
            per_channel.append(
                {
                    "channel": channel,
                    "usable_fraction": _round(
                        float(
                            np.average(
                                [item["usable_fraction"] for item in source],
                                weights=[
                                    row["interval_samples"][1]
                                    - row["interval_samples"][0]
                                    for row in group
                                ],
                            )
                        )
                    ),
                    "source_time_domain_evaluable_fraction": _round(
                        float(
                            np.mean(
                                [item["time_domain_evaluable"] for item in source]
                            )
                        )
                    ),
                    "source_spectral_evaluable_fraction": _round(
                        float(np.mean([item["spectral_evaluable"] for item in source]))
                    ),
                    "primitives_median": primitives,
                    "aggregation_semantics": (
                        "median_of_gap_free_one_second_base_measurements_no_clinical_term"
                    ),
                }
            )
        rows.append(
            {
                "cell_id": f"A{scale_seconds:02d}-{ordinal:06d}",
                "ordinal": ordinal,
                "interval_samples": [start, stop],
                "interval_recording_seconds": [
                    _round(start / rate),
                    _round(stop / rate),
                ],
                "duration_seconds": _round((stop - start) / rate),
                "nominal_scale_seconds": scale_seconds,
                "complete_nominal_cell": bool(
                    abs((stop - start) / rate - scale_seconds)
                    <= 1.0 / rate + 1.0e-12
                ),
                "source_base_cell_ids": [str(row["cell_id"]) for row in group],
                "per_channel": per_channel,
            }
        )
    if base_index != len(base_cells):
        raise RuntimeError("clock-aligned aggregation left base cells unconsumed")
    return rows


def _primitives_for_transition(
    cell: Mapping[str, Any],
    channel_index: int,
) -> tuple[float, dict[str, float | None]]:
    item = cell["per_channel"][channel_index]
    if "primitives" in item:
        return float(item["usable_fraction"]), dict(item["primitives"])
    return float(item["usable_fraction"]), dict(item["primitives_median"])


def _channel_transition_score(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    channel_index: int,
    *,
    policy: ContinuousCoarseSentinelPolicyV1,
) -> float | None:
    left_qc, a = _primitives_for_transition(left, channel_index)
    right_qc, b = _primitives_for_transition(right, channel_index)
    if min(left_qc, right_qc) < policy.minimum_qc_valid_fraction:
        return None
    components: list[float] = []
    for key in _LOG_CHANGE_KEYS:
        if a[key] is not None and b[key] is not None:
            components.append(
                abs(math.log((float(b[key]) + 1.0e-6) / (float(a[key]) + 1.0e-6)))
                / policy.log_amplitude_change_floor
            )
    bands = [
        abs(float(b[key]) - float(a[key]))
        for key in _SPECTRAL_CHANGE_KEYS
        if a[key] is not None and b[key] is not None
    ]
    if len(bands) == len(_SPECTRAL_CHANGE_KEYS):
        components.append(sum(bands) / policy.relative_band_l1_change_floor)
    if a["dominant_frequency_hz"] is not None and b["dominant_frequency_hz"] is not None:
        components.append(
            abs(float(b["dominant_frequency_hz"]) - float(a["dominant_frequency_hz"]))
            / policy.dominant_frequency_change_floor_hz
        )
    if a["spectral_entropy"] is not None and b["spectral_entropy"] is not None:
        components.append(
            abs(float(b["spectral_entropy"]) - float(a["spectral_entropy"]))
            / policy.spectral_entropy_change_floor
        )
    if (
        a["spectral_concentration"] is not None
        and b["spectral_concentration"] is not None
    ):
        components.append(
            abs(
                float(b["spectral_concentration"])
                - float(a["spectral_concentration"])
            )
            / policy.spectral_concentration_change_floor
        )
    if len(components) < 2:
        return None
    ranked = sorted(min(20.0, max(0.0, value)) for value in components)
    return float(np.mean(ranked[-2:]))


def _screen_transitions(
    cells: Sequence[Mapping[str, Any]],
    *,
    scale_seconds: int,
    policy: ContinuousCoarseSentinelPolicyV1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(cells, cells[1:])):
        scores = [
            _channel_transition_score(left, right, channel_index, policy=policy)
            for channel_index in range(len(COMMON17_CHANNELS))
        ]
        evaluable = [score for score in scores if score is not None]
        active = [
            COMMON17_CHANNELS[channel_index]
            for channel_index, score in enumerate(scores)
            if score is not None and score >= policy.channel_change_score_threshold
        ]
        if evaluable:
            ordered = sorted(evaluable)
            take = min(policy.global_top_channel_count, len(ordered))
            global_score: float | None = float(np.mean(ordered[-take:]))
        else:
            global_score = None
        trigger = bool(
            global_score is not None
            and global_score >= policy.global_change_score_threshold
            and len(active) >= policy.minimum_active_channels
        )
        rows.append(
            {
                "transition_id": f"T{scale_seconds:02d}-{index:06d}",
                "ordinal": index,
                "scale_seconds": scale_seconds,
                "left_cell_id": str(left["cell_id"]),
                "right_cell_id": str(right["cell_id"]),
                "boundary_between_samples": int(left["interval_samples"][1]),
                "screened_interval_samples": [
                    int(left["interval_samples"][0]),
                    int(right["interval_samples"][1]),
                ],
                "evaluable_channel_count": len(evaluable),
                "active_channel_count": len(active),
                "active_channels": active,
                "global_screening_change_score": (
                    _round(global_score) if global_score is not None else None
                ),
                "trigger_native_query": trigger,
                "permission": "native_query_screening_only",
                "clinical_assertion_authorized": False,
            }
        )
    return rows


def _merge_triggered_proposals(
    transition_groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    triggered = [
        row
        for rows in transition_groups.values()
        for row in rows
        if bool(row["trigger_native_query"])
    ]
    triggered.sort(
        key=lambda row: (
            int(row["screened_interval_samples"][0]),
            int(row["screened_interval_samples"][1]),
            str(row["transition_id"]),
        )
    )
    merged: list[dict[str, Any]] = []
    for row in triggered:
        start, stop = (int(value) for value in row["screened_interval_samples"])
        if not merged or start > int(merged[-1]["interval_samples"][1]):
            merged.append(
                {
                    "proposal_id": "PENDING",
                    "interval_samples": [start, stop],
                    "source_transition_ids": [str(row["transition_id"])],
                    "source_scales_seconds": [int(row["scale_seconds"])],
                    "maximum_screening_score": float(
                        row["global_screening_change_score"]
                    ),
                    "trigger_native_query": True,
                    "permission": "native_query_only",
                    "clinical_assertion_authorized": False,
                }
            )
            continue
        target = merged[-1]
        target["interval_samples"][1] = max(
            int(target["interval_samples"][1]), stop
        )
        target["source_transition_ids"].append(str(row["transition_id"]))
        target["source_scales_seconds"] = sorted(
            set(target["source_scales_seconds"]).union({int(row["scale_seconds"])})
        )
        target["maximum_screening_score"] = max(
            float(target["maximum_screening_score"]),
            float(row["global_screening_change_score"]),
        )
    for index, row in enumerate(merged):
        row["proposal_id"] = f"NQ{index:06d}"
        row["maximum_screening_score"] = _round(row["maximum_screening_score"])
    return merged


def _normalize_inputs(
    signal_volts: object,
    valid_sample_mask: object | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    signal = np.asarray(signal_volts, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != len(COMMON17_CHANNELS):
        raise ValueError("continuous sentinel requires [common17,time] EEG")
    if signal.shape[1] < 1 or not bool(np.isfinite(signal).all()):
        raise ValueError("continuous sentinel EEG must be finite and non-empty")
    if valid_sample_mask is None:
        valid = np.ones(signal.shape, dtype=bool)
        supplied = False
    else:
        valid = np.asarray(valid_sample_mask)
        if valid.shape != signal.shape or valid.dtype != np.bool_:
            raise ValueError("continuous sentinel QC mask must be boolean and match EEG")
        valid = np.ascontiguousarray(valid, dtype=bool)
        supplied = True
    return np.ascontiguousarray(signal), valid, supplied


def _normalize_reader_result(
    value: object,
    *,
    expected_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    signal: object
    qc: object | None
    if isinstance(value, Mapping):
        if set(value) not in ({"signal_volts"}, {"signal_volts", "valid_sample_mask"}):
            raise ValueError("reader mapping fields are not fail-closed")
        signal = value["signal_volts"]
        qc = value.get("valid_sample_mask")
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("reader tuple must contain signal and QC")
        signal, qc = value
    else:
        signal, qc = value, None
    array = np.asarray(signal)
    if array.shape != (len(COMMON17_CHANNELS), expected_samples):
        raise ValueError("reader returned the wrong common17 interval shape")
    return array, None if qc is None else np.asarray(qc)


def _materialize_from_arrays(
    *,
    recording_id: str,
    candidate_group_id: str,
    horizon_start_sample: int,
    recording_sample_count: int,
    sampling_rate_hz: float,
    signal_volts: object,
    valid_sample_mask: object | None,
    channel_order: Sequence[str],
    policy: ContinuousCoarseSentinelPolicyV1,
    source_mode: str,
    reader_query_intervals: Sequence[Sequence[int]],
) -> dict[str, Any]:
    recording = _identifier(recording_id, "recording_id")
    candidate = _identifier(candidate_group_id, "candidate_group_id")
    start = _integer(horizon_start_sample, "horizon_start_sample")
    recording_count = _integer(
        recording_sample_count, "recording_sample_count", minimum=1
    )
    rate = _finite(
        sampling_rate_hz,
        "sampling_rate_hz",
        minimum=policy.minimum_sampling_rate_hz,
    )
    if tuple(channel_order) != COMMON17_CHANNELS:
        raise ValueError(
            "continuous sentinel requires exact directly observed common17 order"
        )
    signal, valid, qc_supplied = _normalize_inputs(signal_volts, valid_sample_mask)
    stop = start + signal.shape[1]
    if start >= stop or stop > recording_count:
        raise ValueError("continuous sentinel legal horizon lies outside recording")
    base_cells = _build_base_cells(
        signal,
        valid,
        horizon_start_sample=start,
        rate=rate,
        policy=policy,
    )
    aggregate_cells = {
        str(scale): _aggregate_cells(base_cells, scale_seconds=scale, rate=rate)
        for scale in policy.aggregation_scales_seconds
    }
    scale_cells: dict[str, Sequence[Mapping[str, Any]]] = {
        "1": base_cells,
        **aggregate_cells,
    }
    transitions = {
        scale: _screen_transitions(
            cells,
            scale_seconds=int(scale),
            policy=policy,
        )
        for scale, cells in scale_cells.items()
    }
    proposals = _merge_triggered_proposals(transitions)
    fft_evaluations = sum(
        int(channel["spectral_evaluable"])
        for cell in base_cells
        for channel in cell["per_channel"]
    )
    query_rows = [
        [_integer(row[0], "reader query start"), _integer(row[1], "reader query stop")]
        for row in reader_query_intervals
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "recording_id": recording,
        "candidate_group_id": candidate,
        "source_binding": {
            "source_mode": source_mode,
            "signal_sha256": _array_sha256(
                signal.astype("<f8", copy=False),
                domain="common17-continuous-coarse-sentinel-volts-v1",
            ),
            "valid_sample_mask_sha256": _array_sha256(
                valid.astype(np.uint8),
                domain="common17-continuous-coarse-sentinel-qc-v1",
            ),
            "valid_sample_mask_supplied": qc_supplied,
            "raw_signal_content_addressed": True,
        },
        "acquisition": {
            "channel_order": list(COMMON17_CHANNELS),
            "removed_channels": ["FZ", "PZ"],
            "signal_unit": "V",
            "sampling_rate_hz": rate,
            "recording_sample_count": recording_count,
            "missing_channel_imputation_used": False,
            "montage_synthesis_used": False,
            "digital_screening_band_hz": [0.5, _round(min(45.0, 0.45 * rate))],
            "digital_band_is_acquisition_capability_claim": False,
        },
        "legal_horizon": {
            "interval_samples": [start, stop],
            "interval_recording_seconds": [_round(start / rate), _round(stop / rate)],
            "sample_count_per_channel": signal.shape[1],
            "continuous_gap_free_screening_required": True,
            "horizon_selection_is_onset_or_finding_assertion": False,
        },
        "coverage_ledger": {
            "base_cell_intervals_samples": [
                list(cell["interval_samples"]) for cell in base_cells
            ],
            "base_cells_exactly_partition_legal_horizon": True,
            "base_cell_gap_samples": 0,
            "aggregate_source_base_cells_exactly_once_per_scale": {
                str(scale): True for scale in policy.aggregation_scales_seconds
            },
            "uncovered_legal_horizon_samples": 0,
            "sparse_probe_schedule_used": False,
        },
        "base_cells_1s": base_cells,
        "aggregate_cells": aggregate_cells,
        "screening_transitions": transitions,
        "native_query_proposals": proposals,
        "compute_ledger": {
            "coarse_cache_compute": {
                "executed": True,
                "source_mode": source_mode,
                "reader_query_count": len(query_rows),
                "reader_query_intervals_samples": query_rows,
                "unique_physical_samples_per_channel_read": signal.shape[1],
                "channel_samples_read": signal.shape[1]
                * len(COMMON17_CHANNELS),
                "base_cell_count": len(base_cells),
                "aggregate_cell_count_by_scale": {
                    scale: len(rows) for scale, rows in aggregate_cells.items()
                },
                "screened_transition_count_by_scale": {
                    scale: len(rows) for scale, rows in transitions.items()
                },
                "per_channel_fft_evaluation_count": fft_evaluations,
            },
            "downstream_native_fine_compute": {
                "executed": False,
                "query_intervals_samples": [],
                "unique_physical_samples_per_channel_read": 0,
                "channel_samples_read": 0,
            },
            "accounting_semantics": {
                "coarse_and_native_fine_ledgers_are_separate": True,
                "coarse_physical_samples_were_read": True,
                "coarse_samples_may_be_claimed_unread": False,
                "efficiency_claim_must_include_coarse_compute": True,
                "native_query_proposals_are_executed_here": False,
                "total_physical_source_samples_exposed_per_channel": signal.shape[1],
            },
        },
        "scope_receipt": deepcopy(_SCOPE),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_common17_continuous_coarse_sentinel_cache_v1(body)


def materialize_common17_continuous_coarse_sentinel_cache_from_arrays_v1(
    *,
    recording_id: str,
    candidate_group_id: str,
    horizon_start_sample: int,
    recording_sample_count: int,
    sampling_rate_hz: float,
    signal_volts: object,
    valid_sample_mask: object | None = None,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: ContinuousCoarseSentinelPolicyV1 = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Materialize a cache from an already exposed legal-horizon array."""

    return _materialize_from_arrays(
        recording_id=recording_id,
        candidate_group_id=candidate_group_id,
        horizon_start_sample=horizon_start_sample,
        recording_sample_count=recording_sample_count,
        sampling_rate_hz=sampling_rate_hz,
        signal_volts=signal_volts,
        valid_sample_mask=valid_sample_mask,
        channel_order=channel_order,
        policy=policy,
        source_mode="in_memory_legal_horizon_arrays",
        reader_query_intervals=(),
    )


def materialize_common17_continuous_coarse_sentinel_cache_v1(
    *,
    recording_id: str,
    candidate_group_id: str,
    horizon_start_sample: int,
    horizon_stop_sample: int,
    recording_sample_count: int,
    sampling_rate_hz: float,
    query_reader: Any,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: ContinuousCoarseSentinelPolicyV1 = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Read exactly one legal horizon and materialize its continuous cache."""

    start = _integer(horizon_start_sample, "horizon_start_sample")
    stop = _integer(horizon_stop_sample, "horizon_stop_sample", minimum=1)
    recording_count = _integer(
        recording_sample_count, "recording_sample_count", minimum=1
    )
    if not 0 <= start < stop <= recording_count:
        raise ValueError("reader legal horizon lies outside recording")
    if not callable(query_reader):
        raise TypeError("query_reader must be callable")
    raw = query_reader(start, stop)
    signal, qc = _normalize_reader_result(raw, expected_samples=stop - start)
    return _materialize_from_arrays(
        recording_id=recording_id,
        candidate_group_id=candidate_group_id,
        horizon_start_sample=start,
        recording_sample_count=recording_count,
        sampling_rate_hz=sampling_rate_hz,
        signal_volts=signal,
        valid_sample_mask=qc,
        channel_order=channel_order,
        policy=policy,
        source_mode="single_exact_legal_horizon_reader_query",
        reader_query_intervals=((start, stop),),
    )


def _validate_partition(
    intervals: Sequence[Sequence[int]],
    *,
    start: int,
    stop: int,
    field: str,
) -> None:
    if not intervals:
        raise ValueError(f"{field} must not be empty")
    cursor = start
    for row in intervals:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or isinstance(row[0], bool)
            or isinstance(row[1], bool)
            or not isinstance(row[0], int)
            or not isinstance(row[1], int)
            or row[0] != cursor
            or row[1] <= row[0]
            or row[1] > stop
        ):
            raise ValueError(f"{field} is not a gap-free ordered partition")
        cursor = row[1]
    if cursor != stop:
        raise ValueError(f"{field} does not cover the legal horizon")


def _validate_channel_rows(
    rows: object,
    *,
    aggregate: bool,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(COMMON17_CHANNELS):
        raise ValueError("coarse cell channel roster drifted")
    expected = (
        {
            "channel",
            "usable_fraction",
            "source_time_domain_evaluable_fraction",
            "source_spectral_evaluable_fraction",
            "primitives_median",
            "aggregation_semantics",
        }
        if aggregate
        else {
            "channel",
            "usable_fraction",
            "valid_sample_count",
            "valid_adjacent_pair_count",
            "repeated_value_transition_fraction",
            "time_domain_evaluable",
            "spectral_evaluable",
            "quality_semantics",
            "primitives",
        }
    )
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != expected:
            raise ValueError("coarse per-channel fields drifted")
        if row["channel"] != COMMON17_CHANNELS[index]:
            raise ValueError("coarse channel order drifted")
        usable = _finite(row["usable_fraction"], "usable_fraction", minimum=0.0)
        if usable > 1.0:
            raise ValueError("usable fraction exceeds one")
        primitives = row["primitives_median"] if aggregate else row["primitives"]
        if type(primitives) is not dict or tuple(primitives) != _PRIMITIVE_KEYS:
            raise ValueError("coarse primitive roster drifted")
        for value in primitives.values():
            if value is not None:
                _finite(value, "coarse primitive")
        if aggregate:
            if row["aggregation_semantics"] != (
                "median_of_gap_free_one_second_base_measurements_no_clinical_term"
            ):
                raise ValueError("aggregation semantics drifted")
            for field in (
                "source_time_domain_evaluable_fraction",
                "source_spectral_evaluable_fraction",
            ):
                value = _finite(row[field], field, minimum=0.0)
                if value > 1.0:
                    raise ValueError("aggregate opportunity fraction exceeds one")
        else:
            if row["quality_semantics"] != (
                "signal_derived_screening_proxy_not_clinical_artifact_label"
            ):
                raise ValueError("quality semantics drifted")
            _integer(row["valid_sample_count"], "valid_sample_count")
            _integer(row["valid_adjacent_pair_count"], "valid_adjacent_pair_count")
            repeated = row["repeated_value_transition_fraction"]
            if repeated is not None:
                repeated_value = _finite(repeated, "repeated fraction", minimum=0.0)
                if repeated_value > 1.0:
                    raise ValueError("repeated fraction exceeds one")
            if type(row["time_domain_evaluable"]) is not bool or type(
                row["spectral_evaluable"]
            ) is not bool:
                raise ValueError("coarse evaluability must be boolean")


def validate_common17_continuous_coarse_sentinel_cache_v1(
    payload: object,
) -> dict[str, Any]:
    """Fail-closed structural, permission, coverage and content validator."""

    if type(payload) is not dict:
        raise TypeError("continuous coarse sentinel receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "policy",
        "policy_sha256",
        "recording_id",
        "candidate_group_id",
        "source_binding",
        "acquisition",
        "legal_horizon",
        "coverage_ledger",
        "base_cells_1s",
        "aggregate_cells",
        "screening_transitions",
        "native_query_proposals",
        "compute_ledger",
        "scope_receipt",
        "authorization",
    }
    if set(data) != required:
        raise ValueError("continuous coarse sentinel top-level fields drifted")
    if data["schema_version"] != SCHEMA_VERSION or data["method_id"] != METHOD_ID:
        raise ValueError("continuous coarse sentinel method binding drifted")
    _identifier(data["recording_id"], "recording_id")
    _identifier(data["candidate_group_id"], "candidate_group_id")
    if data["policy_sha256"] != _canonical_sha256(data["policy"]):
        raise ValueError("continuous coarse sentinel policy hash drifted")
    if data["policy"] != DEFAULT_POLICY.to_dict():
        raise ValueError("continuous coarse sentinel validator accepts only frozen v1 policy")
    source = data["source_binding"]
    if type(source) is not dict or set(source) != {
        "source_mode",
        "signal_sha256",
        "valid_sample_mask_sha256",
        "valid_sample_mask_supplied",
        "raw_signal_content_addressed",
    }:
        raise ValueError("continuous sentinel source binding drifted")
    if source["source_mode"] not in {
        "in_memory_legal_horizon_arrays",
        "single_exact_legal_horizon_reader_query",
    }:
        raise ValueError("continuous sentinel source mode is invalid")
    for field in ("signal_sha256", "valid_sample_mask_sha256"):
        if (
            not isinstance(source[field], str)
            or len(source[field]) != 64
            or any(character not in "0123456789abcdef" for character in source[field])
        ):
            raise ValueError("continuous sentinel source hash is invalid")
    if type(source["valid_sample_mask_supplied"]) is not bool or source[
        "raw_signal_content_addressed"
    ] is not True:
        raise ValueError("continuous sentinel source receipt is not fail-closed")
    acquisition = data["acquisition"]
    if (
        acquisition.get("channel_order") != list(COMMON17_CHANNELS)
        or acquisition.get("removed_channels") != ["FZ", "PZ"]
        or acquisition.get("signal_unit") != "V"
        or acquisition.get("missing_channel_imputation_used") is not False
        or acquisition.get("montage_synthesis_used") is not False
        or acquisition.get("digital_band_is_acquisition_capability_claim") is not False
    ):
        raise ValueError("continuous sentinel common17/acquisition contract drifted")
    rate = _finite(
        acquisition.get("sampling_rate_hz"),
        "sampling_rate_hz",
        minimum=DEFAULT_POLICY.minimum_sampling_rate_hz,
    )
    recording_count = _integer(
        acquisition.get("recording_sample_count"),
        "recording_sample_count",
        minimum=1,
    )
    horizon = data["legal_horizon"]
    if type(horizon) is not dict or set(horizon) != {
        "interval_samples",
        "interval_recording_seconds",
        "sample_count_per_channel",
        "continuous_gap_free_screening_required",
        "horizon_selection_is_onset_or_finding_assertion",
    }:
        raise ValueError("continuous sentinel legal horizon fields drifted")
    interval = horizon["interval_samples"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or isinstance(interval[0], bool)
        or isinstance(interval[1], bool)
        or not isinstance(interval[0], int)
        or not isinstance(interval[1], int)
        or not 0 <= interval[0] < interval[1] <= recording_count
        or horizon["sample_count_per_channel"] != interval[1] - interval[0]
        or horizon["continuous_gap_free_screening_required"] is not True
        or horizon["horizon_selection_is_onset_or_finding_assertion"] is not False
    ):
        raise ValueError("continuous sentinel legal horizon is invalid")
    start, stop = interval
    base = data["base_cells_1s"]
    if not isinstance(base, list):
        raise ValueError("continuous sentinel base cells must be a list")
    base_intervals: list[list[int]] = []
    base_ids: list[str] = []
    base_required = {
        "cell_id",
        "ordinal",
        "interval_samples",
        "interval_recording_seconds",
        "duration_seconds",
        "nominal_scale_seconds",
        "complete_nominal_cell",
        "per_channel",
    }
    for index, cell in enumerate(base):
        if type(cell) is not dict or set(cell) != base_required:
            raise ValueError("continuous sentinel base-cell fields drifted")
        if cell["cell_id"] != f"B{index:06d}" or cell["ordinal"] != index:
            raise ValueError("continuous sentinel base-cell identity drifted")
        if cell["nominal_scale_seconds"] != 1:
            raise ValueError("continuous sentinel base scale drifted")
        base_intervals.append(cell["interval_samples"])
        base_ids.append(cell["cell_id"])
        _validate_channel_rows(cell["per_channel"], aggregate=False)
    _validate_partition(base_intervals, start=start, stop=stop, field="base cells")
    coverage = data["coverage_ledger"]
    if (
        type(coverage) is not dict
        or coverage.get("base_cell_intervals_samples") != base_intervals
        or coverage.get("base_cells_exactly_partition_legal_horizon") is not True
        or coverage.get("base_cell_gap_samples") != 0
        or coverage.get("uncovered_legal_horizon_samples") != 0
        or coverage.get("sparse_probe_schedule_used") is not False
        or coverage.get("aggregate_source_base_cells_exactly_once_per_scale")
        != {"4": True, "16": True}
    ):
        raise ValueError("continuous sentinel coverage ledger drifted")
    aggregates = data["aggregate_cells"]
    if type(aggregates) is not dict or set(aggregates) != {"4", "16"}:
        raise ValueError("continuous sentinel aggregate scales drifted")
    cells_by_scale: dict[str, list[dict[str, Any]]] = {"1": base}
    aggregate_required = base_required.union({"source_base_cell_ids"})
    for scale in (4, 16):
        rows = aggregates[str(scale)]
        if not isinstance(rows, list):
            raise ValueError("continuous sentinel aggregate rows must be a list")
        source_ids: list[str] = []
        intervals: list[list[int]] = []
        for index, cell in enumerate(rows):
            if type(cell) is not dict or set(cell) != aggregate_required:
                raise ValueError("continuous sentinel aggregate-cell fields drifted")
            if (
                cell["cell_id"] != f"A{scale:02d}-{index:06d}"
                or cell["ordinal"] != index
                or cell["nominal_scale_seconds"] != scale
                or not isinstance(cell["source_base_cell_ids"], list)
                or not 1 <= len(cell["source_base_cell_ids"]) <= scale
            ):
                raise ValueError("continuous sentinel aggregate identity drifted")
            source_ids.extend(cell["source_base_cell_ids"])
            intervals.append(cell["interval_samples"])
            _validate_channel_rows(cell["per_channel"], aggregate=True)
        if source_ids != base_ids:
            raise ValueError("aggregate cells do not consume each base cell exactly once")
        _validate_partition(
            intervals,
            start=start,
            stop=stop,
            field=f"{scale}-second aggregate cells",
        )
        cells_by_scale[str(scale)] = rows
    transitions = data["screening_transitions"]
    if type(transitions) is not dict or set(transitions) != {"1", "4", "16"}:
        raise ValueError("continuous sentinel transition scales drifted")
    transition_ids: set[str] = set()
    triggered_ids: set[str] = set()
    transition_required = {
        "transition_id",
        "ordinal",
        "scale_seconds",
        "left_cell_id",
        "right_cell_id",
        "boundary_between_samples",
        "screened_interval_samples",
        "evaluable_channel_count",
        "active_channel_count",
        "active_channels",
        "global_screening_change_score",
        "trigger_native_query",
        "permission",
        "clinical_assertion_authorized",
    }
    for scale, cells in cells_by_scale.items():
        rows = transitions[scale]
        if not isinstance(rows, list) or len(rows) != max(0, len(cells) - 1):
            raise ValueError("continuous sentinel did not screen every adjacent cell")
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != transition_required:
                raise ValueError("continuous sentinel transition fields drifted")
            expected_id = f"T{int(scale):02d}-{index:06d}"
            if (
                row["transition_id"] != expected_id
                or row["ordinal"] != index
                or row["scale_seconds"] != int(scale)
                or row["left_cell_id"] != cells[index]["cell_id"]
                or row["right_cell_id"] != cells[index + 1]["cell_id"]
                or row["boundary_between_samples"]
                != cells[index]["interval_samples"][1]
                or row["screened_interval_samples"]
                != [
                    cells[index]["interval_samples"][0],
                    cells[index + 1]["interval_samples"][1],
                ]
                or row["permission"] != "native_query_screening_only"
                or row["clinical_assertion_authorized"] is not False
                or type(row["trigger_native_query"]) is not bool
            ):
                raise ValueError("continuous sentinel transition semantics drifted")
            transition_ids.add(expected_id)
            if row["trigger_native_query"]:
                triggered_ids.add(expected_id)
    proposals = data["native_query_proposals"]
    if not isinstance(proposals, list):
        raise ValueError("continuous sentinel proposals must be a list")
    bound_transition_ids: set[str] = set()
    proposal_required = {
        "proposal_id",
        "interval_samples",
        "source_transition_ids",
        "source_scales_seconds",
        "maximum_screening_score",
        "trigger_native_query",
        "permission",
        "clinical_assertion_authorized",
    }
    previous_stop: int | None = None
    for index, row in enumerate(proposals):
        if type(row) is not dict or set(row) != proposal_required:
            raise ValueError("continuous sentinel proposal fields drifted")
        proposal_interval = row["interval_samples"]
        if (
            row["proposal_id"] != f"NQ{index:06d}"
            or row["trigger_native_query"] is not True
            or row["permission"] != "native_query_only"
            or row["clinical_assertion_authorized"] is not False
            or not isinstance(proposal_interval, list)
            or len(proposal_interval) != 2
            or not start <= proposal_interval[0] < proposal_interval[1] <= stop
            or previous_stop is not None
            and proposal_interval[0] <= previous_stop
            or not isinstance(row["source_transition_ids"], list)
            or not row["source_transition_ids"]
            or not set(row["source_transition_ids"]).issubset(triggered_ids)
            or not isinstance(row["source_scales_seconds"], list)
            or not set(row["source_scales_seconds"]).issubset({1, 4, 16})
        ):
            raise ValueError("continuous sentinel native-query proposal drifted")
        previous_stop = proposal_interval[1]
        bound_transition_ids.update(row["source_transition_ids"])
    if bound_transition_ids != triggered_ids:
        raise ValueError("triggered transitions are not exactly bound to proposals")
    compute = data["compute_ledger"]
    if type(compute) is not dict or set(compute) != {
        "coarse_cache_compute",
        "downstream_native_fine_compute",
        "accounting_semantics",
    }:
        raise ValueError("continuous sentinel compute ledger fields drifted")
    coarse = compute["coarse_cache_compute"]
    fine = compute["downstream_native_fine_compute"]
    accounting = compute["accounting_semantics"]
    expected_samples = stop - start
    if (
        coarse.get("executed") is not True
        or coarse.get("unique_physical_samples_per_channel_read") != expected_samples
        or coarse.get("channel_samples_read")
        != expected_samples * len(COMMON17_CHANNELS)
        or coarse.get("base_cell_count") != len(base)
        or coarse.get("aggregate_cell_count_by_scale")
        != {scale: len(rows) for scale, rows in aggregates.items()}
        or coarse.get("screened_transition_count_by_scale")
        != {scale: len(rows) for scale, rows in transitions.items()}
        or fine
        != {
            "executed": False,
            "query_intervals_samples": [],
            "unique_physical_samples_per_channel_read": 0,
            "channel_samples_read": 0,
        }
        or accounting
        != {
            "coarse_and_native_fine_ledgers_are_separate": True,
            "coarse_physical_samples_were_read": True,
            "coarse_samples_may_be_claimed_unread": False,
            "efficiency_claim_must_include_coarse_compute": True,
            "native_query_proposals_are_executed_here": False,
            "total_physical_source_samples_exposed_per_channel": expected_samples,
        }
    ):
        raise ValueError("continuous sentinel coarse/fine accounting drifted")
    source_mode = source["source_mode"]
    expected_reader_count = 1 if source_mode == "single_exact_legal_horizon_reader_query" else 0
    expected_reader_rows = [[start, stop]] if expected_reader_count else []
    if (
        coarse.get("source_mode") != source_mode
        or coarse.get("reader_query_count") != expected_reader_count
        or coarse.get("reader_query_intervals_samples") != expected_reader_rows
    ):
        raise ValueError("continuous sentinel reader accounting drifted")
    if data["scope_receipt"] != _SCOPE or data["authorization"] != _AUTHORIZATION:
        raise ValueError("continuous sentinel scope or permission escalated")
    expected_hash = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected_hash:
        raise ValueError("continuous sentinel content hash mismatch")
    return data


__all__ = [
    "COMMON17_CHANNELS",
    "ContinuousCoarseSentinelPolicyV1",
    "DEFAULT_POLICY",
    "METHOD_ID",
    "SCHEMA_VERSION",
    "materialize_common17_continuous_coarse_sentinel_cache_from_arrays_v1",
    "materialize_common17_continuous_coarse_sentinel_cache_v1",
    "validate_common17_continuous_coarse_sentinel_cache_v1",
]
