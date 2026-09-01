"""Common-17 adaptive support v2 with a pre-anchor background bank.

The global candidate lattice ``anchor + [-60,+60]`` is frozen before any EEG
query.  Acquisition reveals only part of that lattice at a time; unrevealed
parts are explicitly censored.  The lattice definition never receives an
acquired support boundary or a baseline endpoint.

Primary onset normalization uses at least two mutually consistent, stationary
pre-anchor blocks outside the global lattice and every supplied frozen detector
candidate envelope.  Near-anchor blocks are screening evidence only.  TERM,
SOZ, annotations, clinical text and post-anchor baseline data have no input
route.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .adaptive_native_evidence_common17 import (
    COMMON17_CHANNELS,
    NativeEEGQueryReader,
    _AcquiredChunk,
    _FEATURE_NAMES,
    _array_sha256,
    _components,
    _global_score,
    _identifier,
    _js_similarity,
    _native_feature_tensor,
    _normalize_query_result,
    _persistent_start,
    _robust_change_scores,
    _round,
    _summarize_primitives,
    _transform_feature_tensor,
    _window_qc_mask,
    _window_starts,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH: Final[Path] = (
    ROOT / "configs/clinical_eeg_common17_adaptive_support_v2_contract.json"
)
ADAPTIVE_SUPPORT_V2_SCHEMA: Final[str] = (
    "clinical_eeg_common17_adaptive_support_event_v2"
)
ADAPTIVE_SUPPORT_V2_METHOD_ID: Final[str] = (
    "COMMON17-REMOTE-BASELINE-BANK-ANCHOR-LATTICE-MULTISCALE-CP-V2"
)
ANCHOR_JITTER_SHADOW_V2_SCHEMA: Final[str] = (
    "clinical_eeg_common17_anchor_jitter_shadow_v2"
)
ANCHOR_JITTER_SHADOW_V2_METHOD_ID: Final[str] = (
    "COMMON17-FROZEN-BASELINE-FIXED-120S-ANCHOR-JITTER-V2"
)

_SIGNATURE_NAMES: Final[tuple[str, ...]] = (
    "rms_uv",
    "line_length_uv_per_sample",
    "dominant_frequency_hz",
    "spectral_entropy",
    "rhythmicity",
)
_SIGNATURE_FLOORS: Final[np.ndarray] = np.asarray(
    (0.12, 0.12, 0.75, 0.04, 0.025), dtype=np.float64
)
_LEFT_REASONS: Final[frozenset[str]] = frozenset(
    {
        "left_edge_clear_closed",
        "recording_start",
        "global_left_lattice_cap_60s",
        "background_censored",
    }
)
_RIGHT_REASONS: Final[frozenset[str]] = frozenset(
    {
        "postchange_recovery_closed",
        "multiscale_spatial_plateau_closed",
        "recording_stop",
        "right_evolution_cap_60s",
        "background_censored",
    }
)
_SCOPE: Final[dict[str, object]] = {
    "common17_EEG_samples_used": True,
    "acquisition_parameters_used": True,
    "EEG_derived_QC_used_if_supplied": True,
    "navigation_anchor_used": True,
    "frozen_detector_candidate_envelopes_used_for_exclusion_only": True,
    "TERM_or_seizure_interval_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_API_used": False,
    "spreadsheet_doctor_or_clinical_text_used": False,
    "video_sleep_activation_or_behaviour_used": False,
    "postanchor_samples_used_for_onset_normalization": False,
    "FZ_or_PZ_samples_used": False,
    "zero_fill_interpolation_or_montage_synthesis_used": False,
    "LLM_output_used": False,
}
_AUTHORIZATION: Final[dict[str, object]] = {
    "candidate_is_confirmed_seizure": False,
    "channel_ranking_is_cortical_SOZ": False,
    "adaptive_superiority_authorized": False,
    "patient_held_out_SOZ_GT_required_for_superiority": True,
    "clinical_deployment_allowed": False,
}
_JITTER_SCOPE: Final[dict[str, object]] = {
    "same_physical_120s_EEG_shadow_used_for_every_offset": True,
    "frozen_preanchor_baseline_reused_for_every_offset": True,
    "adaptive_support_selection_rerun_for_jitter": False,
    "TERM_or_seizure_interval_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotations_or_sidecars_used": False,
    "clinical_text_or_spreadsheet_used": False,
    "jitter_result_used_to_tune_policy": False,
    "promotion_or_efficacy_claim_authorized": False,
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


def _contract() -> dict[str, Any]:
    value = json.loads(
        DEFAULT_CONTRACT_PATH.resolve(strict=True).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict) or value.get("contract_id") != ADAPTIVE_SUPPORT_V2_METHOD_ID:
        raise ValueError("adaptive-support v2 design contract drifted")
    if value.get("status") != "frozen_before_real_edf_v2_extraction":
        raise ValueError("adaptive-support v2 design contract is not frozen")
    return value


@dataclass(frozen=True)
class AdaptiveSupportV2Policy:
    near_screening_blocks_seconds: tuple[tuple[float, float], ...] = (
        (-24.0, -12.0),
        (-36.0, -24.0),
        (-48.0, -36.0),
        (-60.0, -48.0),
    )
    preanchor_calibration_blocks_seconds: tuple[tuple[float, float], ...] = (
        (-84.0, -72.0),
        (-108.0, -96.0),
        (-132.0, -120.0),
        (-156.0, -144.0),
        (-180.0, -168.0),
        (-216.0, -204.0),
        (-252.0, -240.0),
        (-288.0, -276.0),
    )
    window_seconds: float = 1.0
    step_seconds: float = 0.5
    minimum_baseline_windows: int = 8
    maximum_baseline_windows: int = 12
    minimum_consistent_baseline_blocks: int = 2
    minimum_qc_valid_fraction_per_window: float = 0.90
    minimum_evaluable_channel_fraction: float = 0.70
    baseline_stationarity_threshold: float = 2.5
    baseline_trend_threshold: float = 1.5
    cross_block_distance_threshold: float = 3.0
    detector_envelope_guard_seconds: float = 2.0
    global_lattice_start_relative_seconds: float = -60.0
    global_lattice_stop_relative_seconds: float = 60.0
    left_expansion_extents_seconds: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0)
    right_expansion_extents_seconds: tuple[float, ...] = (8.0, 16.0, 32.0, 48.0, 60.0)
    changepoint_scales_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)
    minimum_changepoint_scales: int = 2
    change_score_threshold: float = 3.0
    shift_score_threshold: float = 0.75
    minimum_active_channels: int = 2
    minimum_onset_run_windows: int = 2
    posterior_temperature: float = 1.5
    left_edge_context_seconds: float = 4.0
    left_edge_candidate_guard_seconds: float = 2.0
    left_edge_evidence_mass_threshold: float = 0.30
    left_edge_high_window_fraction_threshold: float = 0.50
    return_score_threshold: float = 1.6
    recovery_duration_seconds: float = 3.0
    minimum_postcandidate_seconds: float = 8.0
    plateau_context_seconds: float = 4.0
    plateau_spatial_similarity_threshold: float = 0.85
    plateau_absolute_slope_threshold: float = 0.5
    earliest_field_tolerance_seconds: float = 1.0

    def __post_init__(self) -> None:
        contract = _contract()
        baseline = contract["remote_baseline_bank"]
        lattice = contract["candidate_search_lattice"]
        expected_screen = tuple(
            tuple(map(float, row))
            for row in baseline["near_screening_blocks_relative_to_anchor_seconds"]
        )
        expected_far = tuple(
            tuple(map(float, row))
            for row in baseline[
                "preanchor_calibration_blocks_relative_to_anchor_seconds_in_query_order"
            ]
        )
        if self.near_screening_blocks_seconds != expected_screen:
            raise ValueError("v2 near screening bank drifted")
        if self.preanchor_calibration_blocks_seconds != expected_far:
            raise ValueError("v2 preanchor calibration bank drifted")
        if (
            self.global_lattice_start_relative_seconds
            != float(lattice["global_start_relative_to_anchor_seconds"])
            or self.global_lattice_stop_relative_seconds
            != float(lattice["global_stop_relative_to_anchor_seconds"])
        ):
            raise ValueError("v2 global candidate lattice drifted")
        if self.left_expansion_extents_seconds != (10.0, 20.0, 40.0, 60.0):
            raise ValueError("v2 left expansion schedule drifted")
        if self.right_expansion_extents_seconds != (8.0, 16.0, 32.0, 48.0, 60.0):
            raise ValueError("v2 right expansion schedule drifted")
        if self.changepoint_scales_seconds != (1.0, 2.0, 4.0):
            raise ValueError("v2 changepoint scale contract drifted")
        expected_edge_fraction = float(
            lattice["left_edge_high_window_fraction_minimum"]
        )
        if self.left_edge_high_window_fraction_threshold != expected_edge_fraction:
            raise ValueError("v2 left-edge persistence threshold drifted")
        if not 3 <= self.minimum_baseline_windows <= self.maximum_baseline_windows:
            raise ValueError("v2 baseline window counts are invalid")
        if self.minimum_consistent_baseline_blocks < 2:
            raise ValueError("v2 cannot freeze a single-block baseline")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "design_contract_sha256": _canonical_sha256(_contract()),
            "global_lattice_frozen_before_query": True,
            "support_start_is_global_lattice_input": False,
            "baseline_endpoint_is_global_lattice_input": False,
            "primary_onset_normalization_is_preanchor_only": True,
            "threshold_source": "frozen_engineering_contract_not_TERM_or_SOZ_tuned",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY = AdaptiveSupportV2Policy()


@dataclass(frozen=True)
class _StableBlock:
    ordinal: int
    interval_samples: tuple[int, int]
    raw_features: np.ndarray
    transformed_features: np.ndarray
    window_qc: np.ndarray
    starts_absolute: np.ndarray
    signature_center: np.ndarray
    stationarity_score: float
    trend_score: float
    signal_sha256: str
    qc_sha256: str


@dataclass(frozen=True)
class FrozenBaselineV2:
    raw_features: np.ndarray
    transformed_features: np.ndarray
    window_qc: np.ndarray
    starts_absolute: np.ndarray
    selected_blocks: tuple[_StableBlock, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class V2EvidenceSnapshot:
    serializable: dict[str, Any]
    candidate_sample: int | None
    recovery_sample: int | None
    left_edge_touched: bool
    right_terminal_reason: str | None


@dataclass(frozen=True)
class AdaptiveSupportV2RunState:
    receipt: dict[str, Any]
    baseline: FrozenBaselineV2 | None
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]]


def _safe_envelopes(
    values: Sequence[Sequence[float]] | None,
    *,
    anchor_sample: int,
    recording_samples: int,
    rate: float,
    policy: AdaptiveSupportV2Policy,
) -> tuple[tuple[int, int], ...]:
    if values is None:
        start = max(
            0,
            anchor_sample + int(round(policy.global_lattice_start_relative_seconds * rate)),
        )
        stop = min(
            recording_samples,
            anchor_sample + int(round(policy.global_lattice_stop_relative_seconds * rate)),
        )
        return ((start, stop),)
    rows: list[tuple[int, int]] = []
    for value in values:
        if len(value) != 2:
            raise ValueError("detector candidate envelope must contain start/stop seconds")
        start_seconds, stop_seconds = map(float, value)
        if not math.isfinite(start_seconds) or not math.isfinite(stop_seconds) or stop_seconds <= start_seconds:
            raise ValueError("detector candidate envelope is invalid")
        start = max(0, int(round(start_seconds * rate)))
        stop = min(recording_samples, int(round(stop_seconds * rate)))
        if stop > start:
            rows.append((start, stop))
    if not rows:
        raise ValueError("detector candidate envelope set is empty after clipping")
    rows.sort()
    return tuple(rows)


def _overlaps_detector_envelope(
    start: int,
    stop: int,
    envelopes: Sequence[tuple[int, int]],
    *,
    guard_samples: int,
) -> bool:
    return any(start < right + guard_samples and stop > left - guard_samples for left, right in envelopes)


def _read_chunk(
    reader: NativeEEGQueryReader,
    *,
    start: int,
    stop: int,
    rate: float,
    phase: str,
) -> tuple[_AcquiredChunk, dict[str, Any]]:
    result = reader(start, stop)
    signal, qc = _normalize_query_result(result, expected_samples=stop - start)
    chunk = _AcquiredChunk(
        start_sample=start,
        stop_sample=stop,
        signal_volts=signal,
        valid_sample_mask=qc,
        signal_sha256=_array_sha256(signal.astype("<f8", copy=False), prefix="common17-v2-volts"),
        qc_sha256=_array_sha256(qc.astype(np.uint8), prefix="common17-v2-qc"),
    )
    return chunk, {
        "phase": phase,
        "interval_samples": [start, stop],
        "interval_recording_seconds": [_round(start / rate), _round(stop / rate)],
        "samples_per_channel": stop - start,
        "raw_EEG_sha256": chunk.signal_sha256,
        "EEG_QC_sha256": chunk.qc_sha256,
    }


def _stable_block(
    *,
    chunk: _AcquiredChunk,
    ordinal: int,
    rate: float,
    policy: AdaptiveSupportV2Policy,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> _StableBlock | None:
    signal, qc = chunk.signal_volts, chunk.valid_sample_mask
    starts_local = _window_starts(signal.shape[1], rate=rate, policy=policy)
    window_samples = int(round(policy.window_seconds * rate))
    if len(starts_local) < policy.minimum_baseline_windows:
        return None
    raw = _native_feature_tensor(
        signal,
        starts_local,
        global_start_sample=chunk.start_sample,
        rate=rate,
        morphology_cache=morphology_cache,
    )
    transformed = _transform_feature_tensor(raw)
    opportunity = _window_qc_mask(
        qc,
        starts_local,
        window_samples=window_samples,
        threshold=policy.minimum_qc_valid_fraction_per_window,
    )
    feature_indices = [_FEATURE_NAMES.index(name) for name in _SIGNATURE_NAMES]
    masked = np.where(opportunity[:, :, None], transformed[:, :, feature_indices], np.nan)
    with np.errstate(all="ignore"):
        signatures = np.nanmedian(masked, axis=1)
    eligible = (
        (np.mean(opportunity, axis=1) >= policy.minimum_evaluable_channel_fraction)
        & np.isfinite(signatures).all(axis=1)
    )
    width = policy.minimum_baseline_windows
    candidates: list[tuple[float, float, np.ndarray]] = []
    for start_index in range(len(starts_local) - width + 1):
        indices = np.arange(start_index, start_index + width)
        if not bool(np.all(eligible[indices])):
            continue
        values = signatures[indices]
        step = np.abs(np.diff(values, axis=0)) / _SIGNATURE_FLOORS
        dispersion = (
            1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0)
            / _SIGNATURE_FLOORS
        )
        trend = np.abs(values[-1] - values[0]) / _SIGNATURE_FLOORS
        stationarity = float(
            np.median(np.concatenate((np.median(step, axis=0), dispersion)))
        )
        robust_trend = float(np.median(np.sort(trend)[-2:]))
        candidates.append((stationarity, robust_trend, indices))
    if not candidates:
        return None
    stationarity, trend, indices = min(
        candidates, key=lambda value: (value[0] + value[1], value[0], value[1])
    )
    if stationarity > policy.baseline_stationarity_threshold or trend > policy.baseline_trend_threshold:
        return None
    signature_center = np.median(signatures[indices], axis=0)
    return _StableBlock(
        ordinal=ordinal,
        interval_samples=(chunk.start_sample, chunk.stop_sample),
        raw_features=np.ascontiguousarray(raw[indices]),
        transformed_features=np.ascontiguousarray(transformed[indices]),
        window_qc=np.ascontiguousarray(opportunity[indices]),
        starts_absolute=np.ascontiguousarray(chunk.start_sample + starts_local[indices]),
        signature_center=np.ascontiguousarray(signature_center),
        stationarity_score=stationarity,
        trend_score=trend,
        signal_sha256=chunk.signal_sha256,
        qc_sha256=chunk.qc_sha256,
    )


def _consistent_cluster(
    blocks: Sequence[_StableBlock], policy: AdaptiveSupportV2Policy
) -> tuple[_StableBlock, ...]:
    if len(blocks) < policy.minimum_consistent_baseline_blocks:
        return ()
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(blocks))}
    for left in range(len(blocks)):
        for right in range(left + 1, len(blocks)):
            left_interval = blocks[left].interval_samples
            right_interval = blocks[right].interval_samples
            nonoverlapping = (
                left_interval[1] <= right_interval[0]
                or right_interval[1] <= left_interval[0]
            )
            distance = float(
                np.median(
                    np.abs(blocks[left].signature_center - blocks[right].signature_center)
                    / _SIGNATURE_FLOORS
                )
            )
            if nonoverlapping and distance <= policy.cross_block_distance_threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    # Connected components are insufficient here: A~B and B~C must not make
    # A and C share one normalization baseline when A!~C.  The bank contains
    # at most eight blocks, so an exact maximum-clique search is both cheap and
    # easier to audit than an approximate graph routine.  Equal-sized cliques
    # are resolved by their ordinal tuple, preserving deterministic query-order
    # preference without consulting any event target.
    minimum = policy.minimum_consistent_baseline_blocks
    for size in range(len(blocks), minimum - 1, -1):
        eligible: list[tuple[int, ...]] = []
        for indices in combinations(range(len(blocks)), size):
            if all(right in adjacency[left] for left, right in combinations(indices, 2)):
                eligible.append(indices)
        if eligible:
            chosen = min(
                eligible,
                key=lambda indices: tuple(
                    sorted(blocks[index].ordinal for index in indices)
                ),
            )
            selected = (blocks[index] for index in chosen)
            return tuple(sorted(selected, key=lambda block: block.ordinal))
    return ()


def _freeze_baseline_bank(
    *,
    reader: NativeEEGQueryReader,
    anchor_sample: int,
    recording_samples: int,
    rate: float,
    envelopes: Sequence[tuple[int, int]],
    policy: AdaptiveSupportV2Policy,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> tuple[FrozenBaselineV2 | None, dict[str, Any]]:
    screening_trace: list[dict[str, Any]] = []
    screening_stable: list[_StableBlock] = []
    for ordinal, (relative_start, relative_stop) in enumerate(policy.near_screening_blocks_seconds):
        start = anchor_sample + int(round(relative_start * rate))
        stop = anchor_sample + int(round(relative_stop * rate))
        if start < 0 or stop <= 0:
            screening_trace.append({"ordinal": ordinal, "decision": "skip_recording_start"})
            continue
        chunk, ledger = _read_chunk(
            reader,
            start=start,
            stop=stop,
            rate=rate,
            phase="near_screening_not_normalization",
        )
        block = _stable_block(
            chunk=chunk,
            ordinal=ordinal,
            rate=rate,
            policy=policy,
            morphology_cache=morphology_cache,
        )
        if block is not None:
            screening_stable.append(block)
        ledger["stationary_block_available"] = block is not None
        ledger["normalization_permission"] = False
        screening_trace.append(ledger)
        if len(_consistent_cluster(screening_stable, policy)) >= 2:
            break

    calibration_trace: list[dict[str, Any]] = []
    stable_blocks: list[_StableBlock] = []
    cluster: tuple[_StableBlock, ...] = ()
    guard = int(round(policy.detector_envelope_guard_seconds * rate))
    for ordinal, (relative_start, relative_stop) in enumerate(
        policy.preanchor_calibration_blocks_seconds
    ):
        start = anchor_sample + int(round(relative_start * rate))
        stop = anchor_sample + int(round(relative_stop * rate))
        base = {
            "ordinal": ordinal,
            "relative_interval_seconds": [relative_start, relative_stop],
            "normalization_permission": "primary_onset_preanchor",
        }
        if start < 0 or stop <= 0:
            calibration_trace.append({**base, "decision": "skip_recording_start"})
            continue
        if _overlaps_detector_envelope(start, stop, envelopes, guard_samples=guard):
            calibration_trace.append({**base, "decision": "skip_detector_envelope_guard"})
            continue
        chunk, ledger = _read_chunk(
            reader,
            start=start,
            stop=stop,
            rate=rate,
            phase="preanchor_calibration_candidate",
        )
        block = _stable_block(
            chunk=chunk,
            ordinal=ordinal,
            rate=rate,
            policy=policy,
            morphology_cache=morphology_cache,
        )
        if block is not None:
            stable_blocks.append(block)
        cluster = _consistent_cluster(stable_blocks, policy)
        ledger.update(base)
        ledger["stationary_block_available"] = block is not None
        ledger["consistent_cluster_size_after"] = len(cluster)
        ledger["decision"] = "freeze_background" if cluster else "expand_farther"
        calibration_trace.append(ledger)
        if cluster:
            break

    near_cluster = _consistent_cluster(screening_stable, policy)
    bank_receipt: dict[str, Any] = {
        "status": (
            "qualified_consistent_preanchor_background"
            if cluster
            else "background_censored"
        ),
        "near_screening_trace": screening_trace,
        "near_screening_consistent_cluster_size": len(near_cluster),
        "near_screening_used_for_normalization": False,
        "preanchor_calibration_trace": calibration_trace,
        "selected_block_count": len(cluster),
        "minimum_required_blocks": policy.minimum_consistent_baseline_blocks,
        "single_block_forced_selection": False,
        "postanchor_block_used_for_onset_normalization": False,
        "all_detector_envelopes_excluded_with_guard": True,
        "event_sharing": {
            "used": False,
            "saved_samples_per_channel": 0,
            "sharing_key_components": [
                "recording_id",
                "detector_envelope_set_sha256",
                "design_contract_sha256",
            ],
        },
    }
    if not cluster:
        bank_receipt["selected_blocks"] = []
        return None, bank_receipt
    raw = np.concatenate([block.raw_features for block in cluster], axis=0)
    transformed = np.concatenate(
        [block.transformed_features for block in cluster], axis=0
    )
    opportunity = np.concatenate([block.window_qc for block in cluster], axis=0)
    starts = np.concatenate([block.starts_absolute for block in cluster], axis=0)
    selected_rows = [
        {
            "ordinal": block.ordinal,
            "interval_samples": list(block.interval_samples),
            "stationarity_score": _round(block.stationarity_score),
            "trend_score": _round(block.trend_score),
            "raw_EEG_sha256": block.signal_sha256,
            "EEG_QC_sha256": block.qc_sha256,
        }
        for block in cluster
    ]
    bank_receipt["selected_blocks"] = selected_rows
    bank_receipt["selected_feature_windows"] = len(raw)
    bank_receipt["baseline_feature_sha256"] = _array_sha256(
        raw.astype("<f8", copy=False), prefix="common17-v2-baseline-features"
    )
    return (
        FrozenBaselineV2(
            raw_features=np.ascontiguousarray(raw),
            transformed_features=np.ascontiguousarray(transformed),
            window_qc=np.ascontiguousarray(opportunity),
            starts_absolute=np.ascontiguousarray(starts),
            selected_blocks=cluster,
            receipt=deepcopy(bank_receipt),
        ),
        bank_receipt,
    )


def _multiscale_shifts(
    values: np.ndarray, policy: AdaptiveSupportV2Policy
) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.full(
        (len(values), len(policy.changepoint_scales_seconds)), np.nan, dtype=np.float64
    )
    available = np.zeros(shifts.shape, dtype=bool)
    for column, seconds in enumerate(policy.changepoint_scales_seconds):
        width = max(1, int(round(seconds / policy.step_seconds)))
        for index in range(width, len(values) - width + 1):
            before = float(np.median(values[index - width : index]))
            after = float(np.median(values[index : index + width]))
            shifts[index, column] = max(0.0, after - before)
            available[index, column] = True
    return shifts, available


def evaluate_common17_analysis_support_v2(
    *,
    baseline: FrozenBaselineV2,
    analysis_signal_volts: np.ndarray,
    analysis_qc: np.ndarray,
    analysis_start_sample: int,
    anchor_sample: int,
    rate: float,
    policy: AdaptiveSupportV2Policy = DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]] | None = None,
) -> V2EvidenceSnapshot:
    cache = {} if morphology_cache is None else morphology_cache
    signal = np.asarray(analysis_signal_volts, dtype=np.float64)
    qc = np.asarray(analysis_qc, dtype=bool)
    if signal.ndim != 2 or signal.shape[0] != len(COMMON17_CHANNELS) or qc.shape != signal.shape:
        raise ValueError("v2 analysis support must be exact common17 signal/QC")
    starts_local = _window_starts(signal.shape[1], rate=rate, policy=policy)
    if not len(starts_local):
        raise ValueError("v2 analysis support is too short")
    starts_absolute = analysis_start_sample + starts_local
    window_samples = int(round(policy.window_seconds * rate))
    analysis_raw = _native_feature_tensor(
        signal,
        starts_local,
        global_start_sample=analysis_start_sample,
        rate=rate,
        morphology_cache=cache,
    )
    analysis_transformed = _transform_feature_tensor(analysis_raw)
    analysis_opportunity = _window_qc_mask(
        qc,
        starts_local,
        window_samples=window_samples,
        threshold=policy.minimum_qc_valid_fraction_per_window,
    )
    raw = np.concatenate((baseline.raw_features, analysis_raw), axis=0)
    transformed = np.concatenate(
        (baseline.transformed_features, analysis_transformed), axis=0
    )
    opportunity = np.concatenate((baseline.window_qc, analysis_opportunity), axis=0)
    baseline_mask = np.zeros(len(raw), dtype=bool)
    baseline_mask[: len(baseline.raw_features)] = True
    scores, probability, spatial, centers, scales = _robust_change_scores(
        transformed, opportunity, baseline_mask, policy=policy
    )
    scores = scores[len(baseline.raw_features) :]
    probability = probability[len(baseline.raw_features) :]
    spatial = spatial[len(baseline.raw_features) :]
    evaluable = scales[:, 0] > 0.0
    global_scores = _global_score(scores, evaluable)
    active = np.sum(scores >= policy.change_score_threshold, axis=1)
    shifts, available = _multiscale_shifts(global_scores, policy)
    passing = available & (shifts >= policy.shift_score_threshold)
    global_start = anchor_sample + int(
        round(policy.global_lattice_start_relative_seconds * rate)
    )
    global_stop = anchor_sample + int(
        round(policy.global_lattice_stop_relative_seconds * rate)
    )
    search = (starts_absolute >= global_start) & (starts_absolute <= global_stop)
    candidate_flags = (
        search
        & (global_scores >= policy.change_score_threshold)
        & (active >= policy.minimum_active_channels)
        & (np.sum(passing, axis=1) >= policy.minimum_changepoint_scales)
    )
    candidate_index = _persistent_start(
        candidate_flags, minimum_length=policy.minimum_onset_run_windows
    )
    candidate_sample = (
        int(starts_absolute[candidate_index]) if candidate_index is not None else None
    )
    onset_rows = (
        np.flatnonzero(
            (starts_absolute >= candidate_sample)
            & (starts_absolute <= candidate_sample + int(round(2.0 * rate)))
        )
        if candidate_sample is not None
        else np.zeros(0, dtype=np.int64)
    )
    if len(onset_rows):
        distribution = np.mean(spatial[onset_rows], axis=0)
    elif np.any(evaluable):
        distribution = spatial[int(np.argmax(global_scores))]
    else:
        distribution = np.zeros(len(COMMON17_CHANNELS), dtype=np.float64)
    distribution /= max(float(np.sum(distribution)), 1e-12)

    context_rows = max(1, int(round(policy.left_edge_context_seconds / policy.step_seconds)))
    edge_rows = np.arange(min(context_rows, len(global_scores)))
    evidence_mass = np.maximum(global_scores, 0.0)
    edge_mass = float(
        np.sum(evidence_mass[edge_rows]) / max(float(np.sum(evidence_mass)), 1e-12)
    )
    edge_level = float(np.max(global_scores[edge_rows])) if len(edge_rows) else 0.0
    edge_high_fraction = (
        float(
            np.mean(
                global_scores[edge_rows] >= policy.change_score_threshold,
                dtype=np.float64,
            )
        )
        if len(edge_rows)
        else 0.0
    )
    candidate_touches = (
        candidate_sample is not None
        and candidate_sample - analysis_start_sample
        <= int(round(policy.left_edge_candidate_guard_seconds * rate))
    )
    left_edge_touched = bool(
        candidate_touches
        or (
            edge_level >= policy.change_score_threshold
            and (
                edge_mass >= policy.left_edge_evidence_mass_threshold
                or edge_high_fraction
                >= policy.left_edge_high_window_fraction_threshold
            )
        )
    )

    channel_onsets: dict[str, int | None] = {}
    channel_search = search.copy()
    if candidate_sample is not None:
        channel_search &= starts_absolute >= candidate_sample - int(round(rate))
    for index, channel in enumerate(COMMON17_CHANNELS):
        onset = _persistent_start(
            channel_search & (scores[:, index] >= policy.change_score_threshold),
            minimum_length=policy.minimum_onset_run_windows,
        )
        channel_onsets[channel] = int(starts_absolute[onset]) if onset is not None else None
    onset_values = [value for value in channel_onsets.values() if value is not None]
    earliest = min(onset_values) if onset_values else None
    tolerance = int(round(policy.earliest_field_tolerance_seconds * rate))
    earliest_channels = (
        [
            channel
            for channel in COMMON17_CHANNELS
            if channel_onsets[channel] is not None
            and int(channel_onsets[channel]) <= int(earliest) + tolerance
        ]
        if earliest is not None
        else []
    )
    components = _components(earliest_channels)

    recovery_index: int | None = None
    if candidate_index is not None:
        recovery_windows = max(
            1, int(math.ceil(policy.recovery_duration_seconds / policy.step_seconds))
        )
        recovery_index = _persistent_start(
            global_scores < policy.return_score_threshold,
            minimum_length=recovery_windows,
            start_index=candidate_index + policy.minimum_onset_run_windows,
        )
    recovery_sample = (
        int(starts_absolute[recovery_index]) if recovery_index is not None else None
    )
    plateau_similarity: float | None = None
    plateau_slope: float | None = None
    plateau = False
    if candidate_sample is not None:
        after = np.flatnonzero(starts_absolute >= candidate_sample)
        context = max(1, int(round(policy.plateau_context_seconds / policy.step_seconds)))
        elapsed = (analysis_start_sample + signal.shape[1] - candidate_sample) / rate
        if elapsed >= policy.minimum_postcandidate_seconds and len(after) >= 2 * context:
            previous, current = after[-2 * context : -context], after[-context:]
            plateau_similarity = _js_similarity(
                np.mean(spatial[previous], axis=0), np.mean(spatial[current], axis=0)
            )
            plateau_slope = (
                float(np.median(global_scores[current]))
                - float(np.median(global_scores[previous]))
            ) / policy.plateau_context_seconds
            plateau = (
                plateau_similarity >= policy.plateau_spatial_similarity_threshold
                and abs(plateau_slope) <= policy.plateau_absolute_slope_threshold
            )
    if recovery_sample is not None:
        right_terminal = "postchange_recovery_closed"
    elif plateau:
        right_terminal = "multiscale_spatial_plateau_closed"
    else:
        right_terminal = None

    channel_rows = []
    for index, channel in enumerate(COMMON17_CHANNELS):
        peak = int(np.argmax(scores[:, index]))
        channel_rows.append(
            {
                "channel": channel,
                "evaluable": bool(evaluable[index]),
                "onset_spatial_posterior_mass": _round(distribution[index]),
                "earliest_change_recording_seconds": (
                    _round(channel_onsets[channel] / rate)
                    if channel_onsets[channel] is not None
                    else None
                ),
                "peak_change_score": _round(scores[peak, index]),
                "peak_algorithmic_change_posterior": _round(probability[peak, index]),
            }
        )
    channel_rows.sort(
        key=lambda row: (
            -float(row["onset_spatial_posterior_mass"]),
            -float(row["peak_change_score"]),
            COMMON17_CHANNELS.index(str(row["channel"])),
        )
    )
    candidate_indices = len(baseline.raw_features) + np.flatnonzero(search)
    primitives = _summarize_primitives(
        raw,
        transformed,
        centers,
        scales,
        baseline_mask,
        opportunity,
        candidate_indices,
    )
    serializable = {
        "status": (
            "qualified_multiscale_change_candidate"
            if candidate_sample is not None
            else "no_multiscale_change_candidate"
        ),
        "baseline_binding": deepcopy(baseline.receipt),
        "global_candidate_lattice": {
            "interval_relative_to_anchor_seconds": [
                policy.global_lattice_start_relative_seconds,
                policy.global_lattice_stop_relative_seconds,
            ],
            "interval_samples_unclipped": [global_start, global_stop],
            "frozen_before_first_query": True,
            "support_start_used_to_define_lattice": False,
            "baseline_endpoint_used_to_define_lattice": False,
            "acquired_interval_samples": [
                analysis_start_sample,
                analysis_start_sample + signal.shape[1],
            ],
            "left_unacquired_typed_censor": analysis_start_sample > global_start,
            "right_unacquired_typed_censor": analysis_start_sample + signal.shape[1] < global_stop,
        },
        "multiscale_changepoint": {
            "scales_seconds": list(policy.changepoint_scales_seconds),
            "minimum_passing_scales": policy.minimum_changepoint_scales,
            "level_threshold": policy.change_score_threshold,
            "shift_threshold": policy.shift_score_threshold,
            "minimum_active_channels": policy.minimum_active_channels,
        },
        "left_edge_audit": {
            "context_seconds": policy.left_edge_context_seconds,
            "candidate_touches_edge": candidate_touches,
            "edge_global_score_maximum": _round(edge_level),
            "edge_positive_evidence_mass_fraction": _round(edge_mass),
            "edge_high_window_fraction": _round(edge_high_fraction),
            "edge_high_window_fraction_minimum": (
                policy.left_edge_high_window_fraction_threshold
            ),
            "left_edge_touched": left_edge_touched,
        },
        "onset_candidate": (
            {
                "recording_seconds": _round(candidate_sample / rate),
                "relative_to_navigation_anchor_seconds": _round(
                    (candidate_sample - anchor_sample) / rate
                ),
                "clinical_onset_claim_authorized": False,
            }
            if candidate_sample is not None
            else None
        ),
        "per_channel_evidence": channel_rows,
        "earliest_field": (
            {
                "recording_seconds": _round(earliest / rate),
                "channels": earliest_channels,
                "dominant_connected_component": components[0] if components else [],
                "clinical_SOZ_claim_authorized": False,
            }
            if earliest is not None
            else None
        ),
        "evolution": {
            "candidate_return_to_baseline_recording_seconds": (
                _round(recovery_sample / rate) if recovery_sample is not None else None
            ),
            "plateau_spatial_JS_similarity": (
                _round(plateau_similarity) if plateau_similarity is not None else None
            ),
            "plateau_global_score_slope_per_second": (
                _round(plateau_slope) if plateau_slope is not None else None
            ),
            "right_terminal_reason": right_terminal,
            "trajectory_is_ACNS_evolution": False,
        },
        "native_primitives": primitives,
        "change_trajectory": [
            {
                "window_start_recording_seconds": _round(starts_absolute[index] / rate),
                "global_change_score": _round(global_scores[index]),
                "active_channel_count": int(active[index]),
                "passing_changepoint_scales": int(np.sum(passing[index])),
                "multiscale_positive_shifts": [
                    _round(value) if math.isfinite(float(value)) else None
                    for value in shifts[index]
                ],
            }
            for index in range(len(starts_absolute))
        ],
    }
    return V2EvidenceSnapshot(
        serializable=serializable,
        candidate_sample=candidate_sample,
        recovery_sample=recovery_sample,
        left_edge_touched=left_edge_touched,
        right_terminal_reason=right_terminal,
    )


def evaluate_common17_anchor_jitter_shadow_v2(
    *,
    event_id: str,
    recording_id: str,
    baseline: FrozenBaselineV2,
    shadow_signal_volts: np.ndarray,
    shadow_qc: np.ndarray,
    shadow_start_sample: int,
    navigation_anchor_sample: int,
    rate: float,
    policy: AdaptiveSupportV2Policy = DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
    morphology_cache: dict[
        tuple[int, int], tuple[float, float, float, float]
    ] | None = None,
) -> dict[str, Any]:
    """Replay anchor offsets on one frozen physical 120-second EEG shadow.

    The baseline and physical signal support are immutable inputs.  Only the
    navigation coordinate supplied to the candidate lattice changes.  This is
    a target-blind robustness diagnostic; it never reruns adaptive acquisition
    and cannot promote a policy or establish clinical onset accuracy.
    """

    event = _identifier(event_id, "event_id")
    recording = _identifier(recording_id, "recording_id")
    if not isinstance(baseline, FrozenBaselineV2):
        raise TypeError("v2 jitter replay requires a frozen v2 baseline")
    sampling_rate = float(rate)
    if not math.isfinite(sampling_rate) or sampling_rate < 10.0:
        raise ValueError("v2 jitter replay sampling rate is invalid")
    if isinstance(shadow_start_sample, bool) or not isinstance(shadow_start_sample, int):
        raise TypeError("v2 jitter shadow start must be an integer sample")
    if isinstance(navigation_anchor_sample, bool) or not isinstance(
        navigation_anchor_sample, int
    ):
        raise TypeError("v2 jitter anchor must be an integer sample")
    signal = np.asarray(shadow_signal_volts, dtype=np.float64)
    qc = np.asarray(shadow_qc, dtype=bool)
    expected_samples = int(round(120.0 * sampling_rate))
    if (
        signal.shape != (len(COMMON17_CHANNELS), expected_samples)
        or qc.shape != signal.shape
        or not np.isfinite(signal).all()
    ):
        raise ValueError("v2 jitter replay requires exact finite common17 120s shadow")
    expected_start = navigation_anchor_sample - int(round(60.0 * sampling_rate))
    if shadow_start_sample != expected_start:
        raise ValueError("v2 jitter shadow is not centered on the frozen base anchor")

    contract = _contract()["target_blind_anchor_jitter_audit"]
    offsets = tuple(float(value) for value in contract["offsets_seconds"])
    if offsets != (-10.0, -5.0, 0.0, 5.0, 10.0):
        raise ValueError("v2 jitter offset contract drifted")
    cache = {} if morphology_cache is None else morphology_cache
    rows: list[dict[str, Any]] = []
    for offset_seconds in offsets:
        shifted_anchor = navigation_anchor_sample + int(
            round(offset_seconds * sampling_rate)
        )
        snapshot = evaluate_common17_analysis_support_v2(
            baseline=baseline,
            analysis_signal_volts=signal,
            analysis_qc=qc,
            analysis_start_sample=shadow_start_sample,
            anchor_sample=shifted_anchor,
            rate=sampling_rate,
            policy=policy,
            morphology_cache=cache,
        )
        candidate_seconds = (
            _round(snapshot.candidate_sample / sampling_rate)
            if snapshot.candidate_sample is not None
            else None
        )
        ranked = snapshot.serializable["per_channel_evidence"]
        ranked_channels = (
            [str(row["channel"]) for row in ranked]
            if snapshot.candidate_sample is not None
            else []
        )
        rows.append(
            {
                "anchor_offset_seconds": offset_seconds,
                "shifted_navigation_anchor_recording_seconds": _round(
                    shifted_anchor / sampling_rate
                ),
                "evidence_status": snapshot.serializable["status"],
                "candidate_recording_seconds": candidate_seconds,
                "candidate_translation_from_zero_offset_seconds": None,
                "candidate_relative_to_shifted_anchor_seconds": (
                    _round((snapshot.candidate_sample - shifted_anchor) / sampling_rate)
                    if snapshot.candidate_sample is not None
                    else None
                ),
                "top1_channel": ranked_channels[0] if ranked_channels else None,
                "top3_channels": ranked_channels[:3],
                "fixed_shadow_left_typed_censor": snapshot.serializable[
                    "global_candidate_lattice"
                ]["left_unacquired_typed_censor"],
                "fixed_shadow_right_typed_censor": snapshot.serializable[
                    "global_candidate_lattice"
                ]["right_unacquired_typed_censor"],
            }
        )

    zero_row = next(row for row in rows if row["anchor_offset_seconds"] == 0.0)
    zero_candidate = zero_row["candidate_recording_seconds"]
    for row in rows:
        candidate = row["candidate_recording_seconds"]
        if candidate is not None and zero_candidate is not None:
            row["candidate_translation_from_zero_offset_seconds"] = _round(
                float(candidate) - float(zero_candidate)
            )

    evaluable = [row for row in rows if row["candidate_recording_seconds"] is not None]
    opportunity_rate = len(evaluable) / len(rows)
    candidate_times = [float(row["candidate_recording_seconds"]) for row in evaluable]
    candidate_range = (
        max(candidate_times) - min(candidate_times) if candidate_times else None
    )
    top1_counts: dict[str, int] = {}
    for row in evaluable:
        channel = str(row["top1_channel"])
        top1_counts[channel] = top1_counts.get(channel, 0) + 1
    top1_consensus = (
        max(top1_counts.values()) / len(evaluable) if evaluable else 0.0
    )
    pairwise_jaccards: list[float] = []
    for left, right in combinations(evaluable, 2):
        left_set = set(left["top3_channels"])
        right_set = set(right["top3_channels"])
        union = left_set | right_set
        pairwise_jaccards.append(
            len(left_set & right_set) / len(union) if union else 0.0
        )
    minimum_jaccard = min(pairwise_jaccards) if pairwise_jaccards else 0.0
    range_limit = float(contract["candidate_recording_time_range_maximum_seconds"])
    consensus_limit = float(contract["top1_consensus_rate_minimum"])
    jaccard_limit = float(contract["top3_pairwise_jaccard_minimum"])
    opportunity_limit = float(contract["cohort_evaluable_rate_minimum"])
    descriptive_pass = bool(
        candidate_range is not None
        and candidate_range <= range_limit
        and top1_consensus >= consensus_limit
        and minimum_jaccard >= jaccard_limit
        and opportunity_rate >= opportunity_limit
    )
    summary = {
        "offset_count": len(rows),
        "candidate_evaluable_offset_count": len(evaluable),
        "candidate_evaluable_rate": _round(opportunity_rate),
        "candidate_recording_time_range_seconds": (
            _round(candidate_range) if candidate_range is not None else None
        ),
        "top1_consensus_rate": _round(top1_consensus),
        "top3_pairwise_jaccard_minimum": _round(minimum_jaccard),
        "engineering_gate_thresholds": {
            "candidate_recording_time_range_maximum_seconds": range_limit,
            "top1_consensus_rate_minimum": consensus_limit,
            "top3_pairwise_jaccard_minimum": jaccard_limit,
            "candidate_evaluable_rate_minimum": opportunity_limit,
        },
        "descriptive_engineering_gate_pass": descriptive_pass,
        "gate_scope": "within_fixed_shadow_anchor_offsets_only_not_base_adaptive_support_agreement",
        "promotion_or_efficacy_permission_granted": False,
    }
    body: dict[str, Any] = {
        "schema_version": ANCHOR_JITTER_SHADOW_V2_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": ANCHOR_JITTER_SHADOW_V2_METHOD_ID,
        "event_id": event,
        "recording_id": recording,
        "policy_sha256": policy.sha256,
        "frozen_baseline_receipt_sha256": _canonical_sha256(baseline.receipt),
        "fixed_shadow": {
            "interval_samples": [
                shadow_start_sample,
                shadow_start_sample + expected_samples,
            ],
            "interval_recording_seconds": [
                _round(shadow_start_sample / sampling_rate),
                _round((shadow_start_sample + expected_samples) / sampling_rate),
            ],
            "duration_seconds": 120.0,
            "sampling_rate_hz": sampling_rate,
            "raw_EEG_sha256": _array_sha256(
                signal.astype("<f8", copy=False), prefix="common17-v2-jitter-shadow-volts"
            ),
            "EEG_QC_sha256": _array_sha256(
                qc.astype(np.uint8), prefix="common17-v2-jitter-shadow-qc"
            ),
        },
        "base_navigation_anchor_recording_seconds": _round(
            navigation_anchor_sample / sampling_rate
        ),
        "offsets_seconds": list(offsets),
        "replays": rows,
        "summary": summary,
        "scope_receipt": deepcopy(_JITTER_SCOPE),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_common17_anchor_jitter_shadow_v2(body)


def validate_common17_anchor_jitter_shadow_v2(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("v2 anchor-jitter receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "event_id",
        "recording_id",
        "policy_sha256",
        "frozen_baseline_receipt_sha256",
        "fixed_shadow",
        "base_navigation_anchor_recording_seconds",
        "offsets_seconds",
        "replays",
        "summary",
        "scope_receipt",
    }
    if set(data) != required:
        raise ValueError("v2 anchor-jitter receipt fields drifted")
    if (
        data["schema_version"] != ANCHOR_JITTER_SHADOW_V2_SCHEMA
        or data["method_id"] != ANCHOR_JITTER_SHADOW_V2_METHOD_ID
        or data["policy_sha256"] != DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY.sha256
    ):
        raise ValueError("v2 anchor-jitter method or policy drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    if data["offsets_seconds"] != [-10.0, -5.0, 0.0, 5.0, 10.0]:
        raise ValueError("v2 anchor-jitter offsets drifted")
    if len(data["replays"]) != 5 or data["fixed_shadow"].get(
        "duration_seconds"
    ) != 120.0:
        raise ValueError("v2 anchor-jitter shadow/replay count drifted")
    if data["scope_receipt"] != _JITTER_SCOPE:
        raise ValueError("v2 anchor-jitter target-blind scope drifted")
    if data["summary"].get("promotion_or_efficacy_permission_granted") is not False:
        raise ValueError("v2 anchor-jitter receipt granted unauthorized promotion")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("v2 anchor-jitter receipt content hash mismatch")
    return data


def _assemble_analysis(chunks: Sequence[_AcquiredChunk]) -> tuple[int, np.ndarray, np.ndarray]:
    ordered = sorted(chunks, key=lambda value: value.start_sample)
    if not ordered:
        raise ValueError("v2 analysis chunk set is empty")
    cursor = ordered[0].start_sample
    signals, masks = [], []
    for chunk in ordered:
        if chunk.start_sample != cursor:
            raise ValueError("v2 analysis support must remain contiguous")
        signals.append(chunk.signal_volts)
        masks.append(chunk.valid_sample_mask)
        cursor = chunk.stop_sample
    return ordered[0].start_sample, np.concatenate(signals, axis=1), np.concatenate(masks, axis=1)


def _materialize_common17_adaptive_support_v2_with_state(
    *,
    event_id: str,
    recording_id: str,
    navigation_anchor_recording_seconds: float,
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader: NativeEEGQueryReader,
    frozen_detector_candidate_envelopes_recording_seconds: Sequence[Sequence[float]] | None = None,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: AdaptiveSupportV2Policy = DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
) -> AdaptiveSupportV2RunState:
    event = _identifier(event_id, "event_id")
    recording = _identifier(recording_id, "recording_id")
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate < 10.0:
        raise ValueError("v2 sampling rate is invalid")
    if tuple(channel_order) != COMMON17_CHANNELS:
        raise ValueError("v2 requires exact directly observed common17")
    if isinstance(recording_sample_count, bool) or recording_sample_count < int(2 * rate):
        raise ValueError("v2 recording sample count is invalid")
    if not callable(query_reader):
        raise TypeError("v2 query_reader must be callable")
    anchor_sample = int(round(float(navigation_anchor_recording_seconds) * rate))
    if not 0 <= anchor_sample <= recording_sample_count:
        raise ValueError("v2 navigation anchor lies outside recording")
    envelopes = _safe_envelopes(
        frozen_detector_candidate_envelopes_recording_seconds,
        anchor_sample=anchor_sample,
        recording_samples=recording_sample_count,
        rate=rate,
        policy=policy,
    )
    envelope_seconds = [[_round(left / rate), _round(right / rate)] for left, right in envelopes]
    envelope_sha = _canonical_sha256(envelope_seconds)
    cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    baseline, bank_receipt = _freeze_baseline_bank(
        reader=query_reader,
        anchor_sample=anchor_sample,
        recording_samples=recording_sample_count,
        rate=rate,
        envelopes=envelopes,
        policy=policy,
        morphology_cache=cache,
    )

    analysis_trace: list[dict[str, Any]] = []
    analysis_chunks: list[_AcquiredChunk] = []
    initial_start = max(
        0,
        anchor_sample - int(round(policy.left_expansion_extents_seconds[0] * rate)),
    )
    initial_stop = min(
        recording_sample_count,
        anchor_sample + int(round(policy.right_expansion_extents_seconds[0] * rate)),
    )
    initial, ledger = _read_chunk(
        query_reader,
        start=initial_start,
        stop=initial_stop,
        rate=rate,
        phase="initial_candidate_and_evolution_support",
    )
    analysis_chunks.append(initial)
    analysis_trace.append(ledger)
    left_index = 0
    right_index = 0
    left_reason: str | None = "background_censored" if baseline is None else None
    right_reason: str | None = "background_censored" if baseline is None else None
    snapshot: V2EvidenceSnapshot | None = None
    while baseline is not None and (left_reason is None or right_reason is None):
        support_start, signal, qc = _assemble_analysis(analysis_chunks)
        snapshot = evaluate_common17_analysis_support_v2(
            baseline=baseline,
            analysis_signal_volts=signal,
            analysis_qc=qc,
            analysis_start_sample=support_start,
            anchor_sample=anchor_sample,
            rate=rate,
            policy=policy,
            morphology_cache=cache,
        )
        analysis_trace[-1]["evidence_status_after"] = snapshot.serializable["status"]
        analysis_trace[-1]["left_edge_touched_after"] = snapshot.left_edge_touched
        analysis_trace[-1]["right_terminal_reason_after"] = snapshot.right_terminal_reason

        if left_reason is None:
            if not snapshot.left_edge_touched:
                left_reason = "left_edge_clear_closed"
            elif left_index + 1 < len(policy.left_expansion_extents_seconds):
                next_index = left_index + 1
                target = max(
                    0,
                    anchor_sample
                    - int(round(policy.left_expansion_extents_seconds[next_index] * rate)),
                )
                if target < support_start:
                    chunk, row = _read_chunk(
                        query_reader,
                        start=target,
                        stop=support_start,
                        rate=rate,
                        phase=f"left_edge_extension_to_{int(policy.left_expansion_extents_seconds[next_index])}s",
                    )
                    analysis_chunks.append(chunk)
                    analysis_trace.append(row)
                    left_index = next_index
                    continue
                left_reason = "recording_start"
            else:
                left_reason = "global_left_lattice_cap_60s"

        if right_reason is None:
            if snapshot.right_terminal_reason is not None:
                right_reason = snapshot.right_terminal_reason
            elif right_index + 1 < len(policy.right_expansion_extents_seconds):
                next_index = right_index + 1
                current_stop = support_start + signal.shape[1]
                target = min(
                    recording_sample_count,
                    anchor_sample
                    + int(round(policy.right_expansion_extents_seconds[next_index] * rate)),
                )
                if target > current_stop:
                    chunk, row = _read_chunk(
                        query_reader,
                        start=current_stop,
                        stop=target,
                        rate=rate,
                        phase=f"right_evolution_extension_to_{int(policy.right_expansion_extents_seconds[next_index])}s",
                    )
                    analysis_chunks.append(chunk)
                    analysis_trace.append(row)
                    right_index = next_index
                    continue
                right_reason = "recording_stop"
            else:
                right_reason = "right_evolution_cap_60s"
    support_start, signal, qc = _assemble_analysis(analysis_chunks)
    if baseline is not None:
        snapshot = evaluate_common17_analysis_support_v2(
            baseline=baseline,
            analysis_signal_volts=signal,
            analysis_qc=qc,
            analysis_start_sample=support_start,
            anchor_sample=anchor_sample,
            rate=rate,
            policy=policy,
            morphology_cache=cache,
        )
    analysis_trace[-1]["decision"] = "stop"
    evidence = (
        deepcopy(snapshot.serializable)
        if snapshot is not None
        else {
            "status": "background_censored",
            "baseline_binding": None,
            "global_candidate_lattice": {
                "interval_relative_to_anchor_seconds": [-60.0, 60.0],
                "frozen_before_first_query": True,
                "support_start_used_to_define_lattice": False,
                "baseline_endpoint_used_to_define_lattice": False,
                "left_unacquired_typed_censor": True,
                "right_unacquired_typed_censor": True,
            },
            "onset_candidate": None,
            "per_channel_evidence": [],
            "earliest_field": None,
            "evolution": {"right_terminal_reason": "background_censored"},
            "native_primitives": None,
        }
    )
    baseline_query_samples = sum(
        int(row.get("samples_per_channel", 0))
        for row in bank_receipt["near_screening_trace"]
        + bank_receipt["preanchor_calibration_trace"]
    )
    body: dict[str, Any] = {
        "schema_version": ADAPTIVE_SUPPORT_V2_SCHEMA,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": ADAPTIVE_SUPPORT_V2_METHOD_ID,
        "design_contract_sha256": _canonical_sha256(_contract()),
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "event_id": event,
        "recording_id": recording,
        "navigation_anchor_recording_seconds": _round(anchor_sample / rate),
        "frozen_detector_candidate_envelopes": {
            "intervals_recording_seconds": envelope_seconds,
            "sha256": envelope_sha,
            "source": (
                "explicit_frozen_detector_provider"
                if frozen_detector_candidate_envelopes_recording_seconds is not None
                else "conservative_global_lattice_fallback"
            ),
            "used_for_baseline_exclusion_only": True,
        },
        "acquisition": {
            "sampling_rate_hz": rate,
            "recording_sample_count": recording_sample_count,
            "channel_order": list(COMMON17_CHANNELS),
            "removed_channels": ["FZ", "PZ"],
            "signal_unit": "V",
            "missing_channel_imputation_used": False,
        },
        "phase_order": [
            "freeze_global_candidate_lattice",
            "screen_near_preanchor_blocks_without_normalization_permission",
            "freeze_two_or_more_consistent_far_preanchor_baseline_blocks",
            "run_incremental_left_and_right_analysis_queries",
        ],
        "remote_baseline_bank": bank_receipt,
        "adaptive_analysis_support": {
            "query_trace": analysis_trace,
            "interval_samples": [support_start, support_start + signal.shape[1]],
            "interval_relative_to_anchor_seconds": [
                _round((support_start - anchor_sample) / rate),
                _round((support_start + signal.shape[1] - anchor_sample) / rate),
            ],
            "left_extent_seconds": _round((anchor_sample - support_start) / rate),
            "right_extent_seconds": _round(
                (support_start + signal.shape[1] - anchor_sample) / rate
            ),
            "left_terminal_reason": left_reason,
            "right_terminal_reason": right_reason,
            "analysis_samples_per_channel": signal.shape[1],
        },
        "final_evidence": evidence,
        "budget_ledger": {
            "baseline_bank_query_samples_per_channel": baseline_query_samples,
            "analysis_query_samples_per_channel": signal.shape[1],
            "total_query_samples_per_channel": baseline_query_samples + signal.shape[1],
            "event_shared_baseline_used": False,
            "event_sharing_saved_samples_per_channel": 0,
        },
        "target_blind_anchor_jitter_audit": {
            "embedded": False,
            "scheduled_after_all_arm_receipts_freeze_on_120s_shadow": True,
            "may_change_support_selection": False,
        },
        "scope_receipt": deepcopy(_SCOPE),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return AdaptiveSupportV2RunState(
        receipt=validate_common17_adaptive_support_v2(body),
        baseline=baseline,
        morphology_cache=cache,
    )


def materialize_common17_adaptive_support_v2(
    *,
    event_id: str,
    recording_id: str,
    navigation_anchor_recording_seconds: float,
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader: NativeEEGQueryReader,
    frozen_detector_candidate_envelopes_recording_seconds: Sequence[Sequence[float]] | None = None,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: AdaptiveSupportV2Policy = DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
) -> dict[str, Any]:
    return _materialize_common17_adaptive_support_v2_with_state(
        event_id=event_id,
        recording_id=recording_id,
        navigation_anchor_recording_seconds=navigation_anchor_recording_seconds,
        sampling_rate_hz=sampling_rate_hz,
        recording_sample_count=recording_sample_count,
        query_reader=query_reader,
        frozen_detector_candidate_envelopes_recording_seconds=frozen_detector_candidate_envelopes_recording_seconds,
        channel_order=channel_order,
        policy=policy,
    ).receipt


def validate_common17_adaptive_support_v2(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("adaptive-support v2 receipt must be an object")
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "design_contract_sha256",
        "policy",
        "policy_sha256",
        "event_id",
        "recording_id",
        "navigation_anchor_recording_seconds",
        "frozen_detector_candidate_envelopes",
        "acquisition",
        "phase_order",
        "remote_baseline_bank",
        "adaptive_analysis_support",
        "final_evidence",
        "budget_ledger",
        "target_blind_anchor_jitter_audit",
        "scope_receipt",
        "authorization",
    }
    if set(payload) != required:
        raise ValueError("adaptive-support v2 receipt fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != ADAPTIVE_SUPPORT_V2_SCHEMA:
        raise ValueError("adaptive-support v2 schema drifted")
    if data["method_id"] != ADAPTIVE_SUPPORT_V2_METHOD_ID:
        raise ValueError("adaptive-support v2 method drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    if data["design_contract_sha256"] != _canonical_sha256(_contract()):
        raise ValueError("adaptive-support v2 contract hash drifted")
    if data["policy_sha256"] != _canonical_sha256(data["policy"]):
        raise ValueError("adaptive-support v2 policy hash drifted")
    envelope = data["frozen_detector_candidate_envelopes"]
    if envelope.get("used_for_baseline_exclusion_only") is not True:
        raise ValueError("v2 detector envelopes escaped exclusion-only permission")
    if envelope.get("sha256") != _canonical_sha256(
        envelope.get("intervals_recording_seconds")
    ):
        raise ValueError("v2 detector envelope hash mismatch")
    acquisition = data["acquisition"]
    if acquisition.get("channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("adaptive-support v2 is not common17")
    if acquisition.get("removed_channels") != ["FZ", "PZ"] or acquisition.get(
        "missing_channel_imputation_used"
    ) is not False:
        raise ValueError("adaptive-support v2 imputed removed channels")
    if data["phase_order"] != [
        "freeze_global_candidate_lattice",
        "screen_near_preanchor_blocks_without_normalization_permission",
        "freeze_two_or_more_consistent_far_preanchor_baseline_blocks",
        "run_incremental_left_and_right_analysis_queries",
    ]:
        raise ValueError("adaptive-support v2 phase order drifted")
    bank = data["remote_baseline_bank"]
    if bank.get("near_screening_used_for_normalization") is not False:
        raise ValueError("v2 near screening leaked into onset baseline")
    if bank.get("postanchor_block_used_for_onset_normalization") is not False:
        raise ValueError("v2 postanchor data leaked into onset baseline")
    if bank.get("single_block_forced_selection") is not False:
        raise ValueError("v2 forced a single-block baseline")
    if bank.get("status") == "qualified_consistent_preanchor_background":
        if int(bank.get("selected_block_count", 0)) < 2:
            raise ValueError("v2 qualified fewer than two baseline blocks")
    elif bank.get("status") != "background_censored":
        raise ValueError("v2 baseline state is invalid")
    analysis = data["adaptive_analysis_support"]
    if analysis.get("left_terminal_reason") not in _LEFT_REASONS:
        raise ValueError("v2 left terminal reason drifted")
    if analysis.get("right_terminal_reason") not in _RIGHT_REASONS:
        raise ValueError("v2 right terminal reason drifted")
    lattice = data["final_evidence"].get("global_candidate_lattice", {})
    if lattice.get("interval_relative_to_anchor_seconds") != [-60.0, 60.0]:
        raise ValueError("v2 global lattice drifted")
    if lattice.get("frozen_before_first_query") is not True:
        raise ValueError("v2 global lattice was not frozen before query")
    if lattice.get("support_start_used_to_define_lattice") is not False or lattice.get(
        "baseline_endpoint_used_to_define_lattice"
    ) is not False:
        raise ValueError("v2 restored support-start search coupling")
    ledger = data["budget_ledger"]
    if ledger.get("total_query_samples_per_channel") != (
        ledger.get("baseline_bank_query_samples_per_channel")
        + ledger.get("analysis_query_samples_per_channel")
    ):
        raise ValueError("v2 query budget does not close")
    if data["target_blind_anchor_jitter_audit"] != {
        "embedded": False,
        "scheduled_after_all_arm_receipts_freeze_on_120s_shadow": True,
        "may_change_support_selection": False,
    }:
        raise ValueError("v2 jitter phase boundary drifted")
    if data["scope_receipt"] != _SCOPE or data["authorization"] != _AUTHORIZATION:
        raise ValueError("v2 target-blind scope drifted")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("v2 receipt content hash mismatch")
    return data


__all__ = [
    "ADAPTIVE_SUPPORT_V2_METHOD_ID",
    "ADAPTIVE_SUPPORT_V2_SCHEMA",
    "ANCHOR_JITTER_SHADOW_V2_METHOD_ID",
    "ANCHOR_JITTER_SHADOW_V2_SCHEMA",
    "DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY",
    "AdaptiveSupportV2Policy",
    "AdaptiveSupportV2RunState",
    "FrozenBaselineV2",
    "V2EvidenceSnapshot",
    "evaluate_common17_anchor_jitter_shadow_v2",
    "evaluate_common17_analysis_support_v2",
    "materialize_common17_adaptive_support_v2",
    "validate_common17_anchor_jitter_shadow_v2",
    "validate_common17_adaptive_support_v2",
]
