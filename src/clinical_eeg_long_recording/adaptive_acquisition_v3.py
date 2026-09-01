"""Target-blind common-17 adaptive acquisition v3 engineering prototype.

V3 adds two acquisition opportunities absent from v2: mandatory coverage of
the frozen detector envelope and sparse, anchor-relative sentinel probes over
the still-hidden global lattice.  A sentinel can only request a dense local
refinement; it cannot itself become an onset candidate.  Qualified candidates
still require the v2 persistent multiscale EEG changepoint kernel under a
frozen remote pre-anchor baseline.

The module has no reference, annotation, text, or target input.  It is a
narrow engineering prototype and does not authorize efficacy claims.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .adaptive_native_evidence_common17 import (
    COMMON17_CHANNELS,
    NativeEEGQueryReader,
    _AcquiredChunk,
    _array_sha256,
    _native_feature_tensor,
    _persistent_start,
    _robust_change_scores,
    _transform_feature_tensor,
    _window_qc_mask,
    _window_starts,
)
from .adaptive_support_v2 import (
    ADAPTIVE_SUPPORT_V2_METHOD_ID,
    DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
    AdaptiveSupportV2Policy,
    FrozenBaselineV2,
    V2EvidenceSnapshot,
    _canonical_sha256 as _v2_canonical_sha256,
    _contract as _v2_contract,
    _freeze_baseline_bank,
    _identifier,
    _read_chunk,
    _round,
    _safe_envelopes,
    evaluate_common17_analysis_support_v2,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH: Final[Path] = (
    ROOT / "configs/clinical_eeg_common17_adaptive_acquisition_v3_contract.json"
)
SCHEMA_VERSION: Final[str] = "clinical_eeg_common17_adaptive_acquisition_event_v3"
METHOD_ID: Final[str] = (
    "COMMON17-ENVELOPE-SENTINEL-ISLAND-RECOVERY-ACQUISITION-V3"
)

_SCOPE: Final[dict[str, object]] = {
    "common17_EEG_samples_used": True,
    "sampling_rate_recording_length_and_EEG_QC_used": True,
    "frozen_detector_anchor_and_envelope_used_for_navigation_acquisition": True,
    "detector_envelope_used_as_clinical_onset": False,
    "sentinel_probe_used_as_onset_candidate": False,
    "TERM_or_seizure_interval_used": False,
    "SOZ_or_channel_target_used": False,
    "EDF_annotation_or_sidecar_used": False,
    "clinical_text_or_spreadsheet_used": False,
    "source_eval_used": False,
    "FZ_or_PZ_samples_used": False,
    "zero_fill_interpolation_or_montage_synthesis_used": False,
}
_CLAIMS: Final[dict[str, object]] = {
    "engineering_prototype_only": True,
    "broad_source_dev_evaluation_authorized": False,
    "adaptive_superiority_authorized": False,
    "detector_or_SOZ_efficacy_claim_authorized": False,
    "clinical_onset_cortical_SOZ_or_EZ_claim_authorized": False,
    "clinical_deployment_allowed": False,
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
    if (
        not isinstance(value, dict)
        or value.get("contract_id") != METHOD_ID
        or value.get("status")
        != "frozen_engineering_prototype_before_target_blind_synthetic_audit"
    ):
        raise ValueError("adaptive acquisition v3 contract drifted")
    expected_v2 = _v2_canonical_sha256(_v2_contract())
    if value["background_binding"]["design_contract_canonical_sha256"] != expected_v2:
        raise ValueError("adaptive acquisition v3 lost its v2 baseline binding")
    return value


@dataclass(frozen=True)
class AdaptiveAcquisitionV3Policy:
    global_start_relative_seconds: float = -60.0
    global_stop_relative_seconds: float = 60.0
    initial_interval_relative_seconds: tuple[float, float] = (-10.0, 8.0)
    sentinel_intervals_relative_seconds: tuple[tuple[float, float], ...] = (
        (-58.0, -54.0),
        (-46.0, -42.0),
        (-34.0, -30.0),
        (-22.0, -18.0),
        (10.0, 14.0),
        (22.0, 26.0),
        (34.0, 38.0),
        (46.0, 50.0),
        (56.0, 60.0),
    )
    sentinel_global_score_threshold: float = 3.0
    sentinel_minimum_active_channels: int = 2
    sentinel_minimum_persistent_windows: int = 2
    refinement_left_margin_seconds: float = 8.0
    refinement_right_margin_seconds: float = 8.0
    positive_ranking_prefix_seconds: float = 2.0

    def __post_init__(self) -> None:
        contract = _contract()
        lattice = contract["global_candidate_lattice"]
        mandatory = contract["mandatory_dense_acquisition"]
        sentinel = contract["sparse_sentinel_acquisition"]
        refinement = contract["dense_refinement"]
        firewall = contract["onset_causal_evidence_firewall"]
        prefix = firewall["positive_ranking_prefix_relative_to_candidate_seconds"]
        if (
            self.global_start_relative_seconds
            != float(lattice["start_relative_to_navigation_anchor_seconds"])
            or self.global_stop_relative_seconds
            != float(lattice["stop_relative_to_navigation_anchor_seconds"])
            or list(self.initial_interval_relative_seconds)
            != mandatory["initial_interval_relative_to_anchor_seconds"]
            or [list(row) for row in self.sentinel_intervals_relative_seconds]
            != sentinel["probe_intervals_relative_to_anchor_seconds"]
            or self.sentinel_global_score_threshold
            != float(sentinel["global_change_score_minimum"])
            or self.sentinel_minimum_active_channels
            != int(sentinel["active_channel_minimum"])
            or self.sentinel_minimum_persistent_windows
            != int(sentinel["persistent_window_minimum"])
            or self.refinement_left_margin_seconds
            != float(refinement["left_margin_seconds"])
            or self.refinement_right_margin_seconds
            != float(refinement["right_margin_seconds"])
            or prefix[0] != 0.0
            or self.positive_ranking_prefix_seconds != float(prefix[1])
        ):
            raise ValueError("adaptive acquisition v3 policy drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "design_contract_sha256": _canonical_sha256(_contract()),
            "threshold_source": (
                "frozen_engineering_smoke_not_TERM_SOZ_or_source_dev_tuned"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_POLICY = AdaptiveAcquisitionV3Policy()


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    rows = sorted((int(start), int(stop)) for start, stop in intervals if stop > start)
    merged: list[list[int]] = []
    for start, stop in rows:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((start, stop) for start, stop in merged)


def _interval_samples(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in _merge_intervals(intervals))


def _validated_interval_rows(
    value: object,
    *,
    field: str,
    allow_zero_width: bool = False,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ValueError(f"adaptive acquisition v3 {field} must be a list")
    rows: list[tuple[int, int]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or isinstance(row[0], bool)
            or isinstance(row[1], bool)
            or not isinstance(row[0], int)
            or not isinstance(row[1], int)
            or row[0] < 0
            or row[1] < row[0]
            or (row[1] == row[0] and not allow_zero_width)
        ):
            raise ValueError(f"adaptive acquisition v3 {field} interval drifted")
        rows.append((row[0], row[1]))
    return tuple(rows)


def _missing_intervals(
    start: int, stop: int, covered: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    cursor = start
    missing: list[tuple[int, int]] = []
    for left, right in _merge_intervals(covered):
        if right <= cursor or left >= stop:
            continue
        if left > cursor:
            missing.append((cursor, min(left, stop)))
        cursor = max(cursor, right)
        if cursor >= stop:
            break
    if cursor < stop:
        missing.append((cursor, stop))
    return tuple(row for row in missing if row[1] > row[0])


def _interval_is_covered(
    interval: tuple[int, int], covered: Sequence[tuple[int, int]]
) -> bool:
    return not _missing_intervals(interval[0], interval[1], covered)


class _AnalysisStore:
    def __init__(
        self,
        reader: NativeEEGQueryReader,
        *,
        rate: float,
    ) -> None:
        self.reader = reader
        self.rate = rate
        self.chunks: list[_AcquiredChunk] = []
        self.phase_ledger: list[dict[str, Any]] = []

    @property
    def coverage(self) -> tuple[tuple[int, int], ...]:
        return _merge_intervals(
            [(chunk.start_sample, chunk.stop_sample) for chunk in self.chunks]
        )

    def ensure(self, start: int, stop: int, *, phase: str) -> None:
        missing = _missing_intervals(start, stop, self.coverage)
        query_rows: list[dict[str, Any]] = []
        for left, right in missing:
            chunk, ledger = _read_chunk(
                self.reader,
                start=left,
                stop=right,
                rate=self.rate,
                phase=phase,
            )
            self.chunks.append(chunk)
            query_rows.append(ledger)
        self.phase_ledger.append(
            {
                "phase": phase,
                "requested_interval_samples": [start, stop],
                "new_query_intervals_samples": [list(row) for row in missing],
                "incremental_unique_samples_per_channel": sum(
                    right - left for left, right in missing
                ),
                "raw_query_receipts": query_rows,
            }
        )

    def view(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        count = stop - start
        signal = np.empty((len(COMMON17_CHANNELS), count), dtype=np.float64)
        qc = np.empty((len(COMMON17_CHANNELS), count), dtype=bool)
        filled = np.zeros(count, dtype=bool)
        for chunk in self.chunks:
            left = max(start, chunk.start_sample)
            right = min(stop, chunk.stop_sample)
            if right <= left:
                continue
            target = slice(left - start, right - start)
            source = slice(left - chunk.start_sample, right - chunk.start_sample)
            signal[:, target] = chunk.signal_volts[:, source]
            qc[:, target] = chunk.valid_sample_mask[:, source]
            filled[target] = True
        if not bool(np.all(filled)):
            raise RuntimeError("adaptive acquisition v3 attempted an unqueried dense view")
        return np.ascontiguousarray(signal), np.ascontiguousarray(qc)


def _record_clipped_relative_interval(
    relative: tuple[float, float],
    *,
    anchor_sample: int,
    rate: float,
    lower: int,
    upper: int,
) -> tuple[int, int] | None:
    start = max(lower, anchor_sample + int(round(relative[0] * rate)))
    stop = min(upper, anchor_sample + int(round(relative[1] * rate)))
    return (start, stop) if stop > start else None


def _screen_sentinel(
    *,
    baseline: FrozenBaselineV2,
    signal: np.ndarray,
    qc: np.ndarray,
    start: int,
    anchor_sample: int,
    rate: float,
    policy: AdaptiveAcquisitionV3Policy,
    v2_policy: AdaptiveSupportV2Policy,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> tuple[bool, dict[str, Any]]:
    snapshot = evaluate_common17_analysis_support_v2(
        baseline=baseline,
        analysis_signal_volts=signal,
        analysis_qc=qc,
        analysis_start_sample=start,
        anchor_sample=anchor_sample,
        rate=rate,
        policy=v2_policy,
        morphology_cache=morphology_cache,
    )
    trajectory = snapshot.serializable["change_trajectory"]
    flags = np.asarray(
        [
            float(row["global_change_score"])
            >= policy.sentinel_global_score_threshold
            and int(row["active_channel_count"])
            >= policy.sentinel_minimum_active_channels
            for row in trajectory
        ],
        dtype=bool,
    )
    trigger = _persistent_start(
        flags,
        minimum_length=policy.sentinel_minimum_persistent_windows,
    )
    return trigger is not None, {
        "screening_only": True,
        "may_assert_onset_candidate": False,
        "window_count": len(trajectory),
        "persistent_trigger": trigger is not None,
        "first_trigger_window_index": int(trigger) if trigger is not None else None,
        "maximum_global_change_score": (
            max(float(row["global_change_score"]) for row in trajectory)
            if trajectory
            else None
        ),
        "maximum_active_channel_count": (
            max(int(row["active_channel_count"]) for row in trajectory)
            if trajectory
            else 0
        ),
    }


def _baseline_query_intervals(bank_receipt: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    for key in ("near_screening_trace", "preanchor_calibration_trace"):
        for row in bank_receipt[key]:
            interval = row.get("interval_samples")
            if isinstance(interval, list) and len(interval) == 2:
                rows.append((int(interval[0]), int(interval[1])))
    return _merge_intervals(rows)


def _budget_matched_interval(
    samples: int,
    *,
    anchor_sample: int,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    if samples < 0 or samples > upper - lower:
        raise ValueError("adaptive acquisition v3 budget cannot fit its lattice")
    start = anchor_sample - samples // 2
    stop = start + samples
    if start < lower:
        start, stop = lower, lower + samples
    if stop > upper:
        start, stop = upper - samples, upper
    return start, stop


def _candidate_locked_onset_masses(
    *,
    baseline: FrozenBaselineV2,
    signal: np.ndarray,
    qc: np.ndarray,
    start_sample: int,
    rate: float,
    policy: AdaptiveSupportV2Policy,
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Rank channels from an exact raw-EEG prefix and the frozen baseline.

    This deliberately does not reuse ``V2EvidenceSnapshot.per_channel_evidence``.
    Those rows are valid for the v2 evidence summary, but their final feature
    window may extend past a boundary described only in terms of window starts.
    V3 instead truncates the *raw samples* first, then extracts only complete
    windows inside that prefix.  No sample after the candidate-locked boundary
    has a computational route into positive channel rank.
    """

    samples = np.asarray(signal, dtype=np.float64)
    sample_qc = np.asarray(qc, dtype=bool)
    if (
        samples.ndim != 2
        or samples.shape[0] != len(COMMON17_CHANNELS)
        or sample_qc.shape != samples.shape
    ):
        raise ValueError("candidate-locked ranking requires exact common17 signal/QC")
    starts = _window_starts(samples.shape[1], rate=rate, policy=policy)
    if not len(starts):
        raise ValueError("candidate-locked ranking prefix is too short")
    window_samples = int(round(policy.window_seconds * rate))
    raw = _native_feature_tensor(
        samples,
        starts,
        global_start_sample=start_sample,
        rate=rate,
        morphology_cache=morphology_cache,
    )
    transformed = _transform_feature_tensor(raw)
    opportunity = _window_qc_mask(
        sample_qc,
        starts,
        window_samples=window_samples,
        threshold=policy.minimum_qc_valid_fraction_per_window,
    )
    joint_transformed = np.concatenate(
        (baseline.transformed_features, transformed), axis=0
    )
    joint_opportunity = np.concatenate((baseline.window_qc, opportunity), axis=0)
    baseline_mask = np.zeros(len(joint_transformed), dtype=bool)
    baseline_mask[: len(baseline.transformed_features)] = True
    _scores, _probability, spatial, _centers, _scales = _robust_change_scores(
        joint_transformed,
        joint_opportunity,
        baseline_mask,
        policy=policy,
    )
    prefix_spatial = spatial[len(baseline.transformed_features) :]
    distribution = np.mean(prefix_spatial, axis=0)
    distribution /= max(float(np.sum(distribution)), 1e-12)
    masses = {
        channel: float(distribution[index])
        for index, channel in enumerate(COMMON17_CHANNELS)
    }
    audit = {
        "raw_interval_samples": [start_sample, start_sample + samples.shape[1]],
        "samples_per_channel": samples.shape[1],
        "complete_feature_window_count": len(starts),
        "raw_EEG_sha256": _array_sha256(
            samples.astype("<f8", copy=False),
            prefix="common17-v3-candidate-locked-prefix-volts",
        ),
        "EEG_QC_sha256": _array_sha256(
            sample_qc.astype(np.uint8),
            prefix="common17-v3-candidate-locked-prefix-qc",
        ),
        "raw_sample_truncation_precedes_feature_extraction": True,
        "postprefix_samples_used_for_positive_rank": False,
    }
    return masses, audit


def materialize_common17_adaptive_acquisition_v3(
    *,
    event_id: str,
    recording_id: str,
    navigation_anchor_recording_seconds: float,
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader: NativeEEGQueryReader,
    frozen_detector_candidate_envelopes_recording_seconds: Sequence[
        Sequence[float]
    ],
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    policy: AdaptiveAcquisitionV3Policy = DEFAULT_POLICY,
    v2_policy: AdaptiveSupportV2Policy = DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY,
) -> dict[str, Any]:
    event = _identifier(event_id, "event_id")
    recording = _identifier(recording_id, "recording_id")
    if tuple(channel_order) != COMMON17_CHANNELS:
        raise ValueError("adaptive acquisition v3 requires exact observed common17")
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate < 10.0:
        raise ValueError("adaptive acquisition v3 sampling rate is invalid")
    if isinstance(recording_sample_count, bool) or recording_sample_count < int(2 * rate):
        raise ValueError("adaptive acquisition v3 recording length is invalid")
    if frozen_detector_candidate_envelopes_recording_seconds is None:
        raise ValueError("adaptive acquisition v3 requires an explicit frozen detector envelope")
    anchor_sample = int(round(float(navigation_anchor_recording_seconds) * rate))
    if not 0 <= anchor_sample <= recording_sample_count:
        raise ValueError("adaptive acquisition v3 anchor lies outside recording")
    envelopes = _safe_envelopes(
        frozen_detector_candidate_envelopes_recording_seconds,
        anchor_sample=anchor_sample,
        recording_samples=recording_sample_count,
        rate=rate,
        policy=v2_policy,
    )
    morphology_cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    baseline, bank_receipt = _freeze_baseline_bank(
        reader=query_reader,
        anchor_sample=anchor_sample,
        recording_samples=recording_sample_count,
        rate=rate,
        envelopes=envelopes,
        policy=v2_policy,
        morphology_cache=morphology_cache,
    )
    lattice_start = max(
        0,
        anchor_sample + int(round(policy.global_start_relative_seconds * rate)),
    )
    lattice_stop = min(
        recording_sample_count,
        anchor_sample + int(round(policy.global_stop_relative_seconds * rate)),
    )
    store = _AnalysisStore(query_reader, rate=rate)
    sentinel_rows: list[dict[str, Any]] = []
    dense_intervals: list[tuple[int, int]] = []
    dense_evidence: list[dict[str, Any]] = []
    candidate_rows: list[tuple[int, int, V2EvidenceSnapshot]] = []

    if baseline is not None:
        initial = _record_clipped_relative_interval(
            policy.initial_interval_relative_seconds,
            anchor_sample=anchor_sample,
            rate=rate,
            lower=lattice_start,
            upper=lattice_stop,
        )
        if initial is not None:
            store.ensure(*initial, phase="mandatory_initial_dense")
            dense_intervals.append(initial)

        envelope_coverage = _merge_intervals(
            [
                (max(left, lattice_start), min(right, lattice_stop))
                for left, right in envelopes
                if min(right, lattice_stop) > max(left, lattice_start)
            ]
        )
        for index, interval in enumerate(envelope_coverage):
            store.ensure(
                *interval,
                phase=f"mandatory_detector_envelope_coverage_{index}",
            )
            dense_intervals.append(interval)

        triggered: list[tuple[int, int]] = []
        for index, relative in enumerate(policy.sentinel_intervals_relative_seconds):
            interval = _record_clipped_relative_interval(
                relative,
                anchor_sample=anchor_sample,
                rate=rate,
                lower=lattice_start,
                upper=lattice_stop,
            )
            if interval is None:
                sentinel_rows.append(
                    {
                        "ordinal": index,
                        "relative_interval_seconds": list(relative),
                        "status": "record_or_lattice_censored",
                        "screening_only": True,
                        "may_assert_onset_candidate": False,
                    }
                )
                continue
            store.ensure(*interval, phase=f"sparse_sentinel_probe_{index}")
            signal, qc = store.view(*interval)
            fired, screening = _screen_sentinel(
                baseline=baseline,
                signal=signal,
                qc=qc,
                start=interval[0],
                anchor_sample=anchor_sample,
                rate=rate,
                policy=policy,
                v2_policy=v2_policy,
                morphology_cache=morphology_cache,
            )
            refinement = None
            if fired:
                refinement = (
                    max(
                        lattice_start,
                        interval[0]
                        - int(round(policy.refinement_left_margin_seconds * rate)),
                    ),
                    min(
                        lattice_stop,
                        interval[1]
                        + int(round(policy.refinement_right_margin_seconds * rate)),
                    ),
                )
                triggered.append(refinement)
            sentinel_rows.append(
                {
                    "ordinal": index,
                    "relative_interval_seconds": list(relative),
                    "interval_samples": list(interval),
                    "status": "trigger_dense_refinement" if fired else "screen_clear",
                    "screening": screening,
                    "requested_refinement_interval_samples": (
                        list(refinement) if refinement is not None else None
                    ),
                }
            )

        for index, interval in enumerate(_merge_intervals(triggered)):
            store.ensure(*interval, phase=f"sentinel_triggered_dense_refinement_{index}")
            dense_intervals.append(interval)

        dense_intervals = list(_merge_intervals(dense_intervals))
        minimum_dense_samples = int(round(v2_policy.window_seconds * rate))
        for ordinal, interval in enumerate(dense_intervals):
            if interval[1] - interval[0] < minimum_dense_samples:
                dense_evidence.append(
                    {
                        "ordinal": ordinal,
                        "interval_samples": list(interval),
                        "status": "dense_interval_too_short_for_kernel",
                        "candidate_recording_seconds": None,
                    }
                )
                continue
            signal, qc = store.view(*interval)
            snapshot = evaluate_common17_analysis_support_v2(
                baseline=baseline,
                analysis_signal_volts=signal,
                analysis_qc=qc,
                analysis_start_sample=interval[0],
                anchor_sample=anchor_sample,
                rate=rate,
                policy=v2_policy,
                morphology_cache=morphology_cache,
            )
            dense_evidence.append(
                {
                    "ordinal": ordinal,
                    "interval_samples": list(interval),
                    "interval_relative_to_anchor_seconds": [
                        _round((interval[0] - anchor_sample) / rate),
                        _round((interval[1] - anchor_sample) / rate),
                    ],
                    "status": snapshot.serializable["status"],
                    "candidate_recording_seconds": (
                        _round(snapshot.candidate_sample / rate)
                        if snapshot.candidate_sample is not None
                        else None
                    ),
                }
            )
            if snapshot.candidate_sample is not None:
                candidate_rows.append((snapshot.candidate_sample, ordinal, snapshot))

    selected_snapshot: V2EvidenceSnapshot | None = None
    selected_dense_ordinal: int | None = None
    if candidate_rows:
        _candidate_sample, selected_dense_ordinal, selected_snapshot = min(
            candidate_rows, key=lambda row: (row[0], row[1])
        )
    selected_candidate = (
        {
            "recording_seconds": _round(selected_snapshot.candidate_sample / rate),
            "relative_to_navigation_anchor_seconds": _round(
                (selected_snapshot.candidate_sample - anchor_sample) / rate
            ),
            "dense_interval_ordinal": selected_dense_ordinal,
            "selection_rule": (
                "earliest_qualified_physical_candidate_then_interval_order"
            ),
            "detector_anchor_is_clinical_onset": False,
            "sentinel_probe_is_candidate": False,
        }
        if selected_snapshot is not None
        else None
    )

    if selected_snapshot is not None:
        candidate_sample = int(selected_snapshot.candidate_sample)
        prefix_stop = candidate_sample + int(
            round(policy.positive_ranking_prefix_seconds * rate)
        )
        prefix_signal, prefix_qc = store.view(candidate_sample, prefix_stop)
        masses, prefix_audit = _candidate_locked_onset_masses(
            baseline=baseline,
            signal=prefix_signal,
            qc=prefix_qc,
            start_sample=candidate_sample,
            rate=rate,
            policy=v2_policy,
            morphology_cache=morphology_cache,
        )
        rounded_masses = {
            channel: float(_round(masses[channel])) for channel in COMMON17_CHANNELS
        }
        ranked_channels = sorted(
            COMMON17_CHANNELS,
            key=lambda channel: (
                -rounded_masses[channel],
                COMMON17_CHANNELS.index(channel),
            ),
        )
        ranking = {
            "status": "candidate_locked_positive_ranking_available",
            "candidate_locked_prefix_recording_seconds": [
                _round(candidate_sample / rate),
                _round(prefix_stop / rate),
            ],
            "candidate_locked_prefix_audit": prefix_audit,
            "ranking": [
                {
                    "rank": index + 1,
                    "channel": channel,
                    "onset_spatial_posterior_mass": rounded_masses[channel],
                }
                for index, channel in enumerate(ranked_channels)
            ],
            "ranking_sha256": "CONTENT-ADDRESS-PENDING",
            "late_peak_score_used_for_tie_break": False,
            "late_spread_course_or_recovery_may_increase_positive_rank": False,
            "clinical_SOZ_claim_authorized": False,
        }
        ranking["ranking_sha256"] = _canonical_sha256(ranking["ranking"])
    else:
        ranking = {
            "status": (
                "background_censored"
                if baseline is None
                else "no_qualified_multiscale_candidate"
            ),
            "candidate_locked_prefix_recording_seconds": None,
            "candidate_locked_prefix_audit": None,
            "ranking": [],
            "ranking_sha256": _canonical_sha256([]),
            "late_peak_score_used_for_tie_break": False,
            "late_spread_course_or_recovery_may_increase_positive_rank": False,
            "clinical_SOZ_claim_authorized": False,
        }

    analysis_union = store.coverage
    baseline_union = _baseline_query_intervals(bank_receipt)
    physical_union = _merge_intervals((*baseline_union, *analysis_union))
    analysis_samples = _interval_samples(analysis_union)
    comparator = _budget_matched_interval(
        analysis_samples,
        anchor_sample=anchor_sample,
        lower=lattice_start,
        upper=lattice_stop,
    )
    phase_incremental = sum(
        int(row["incremental_unique_samples_per_channel"])
        for row in store.phase_ledger
    )
    budget = {
        "analysis_query_phase_ledger": deepcopy(store.phase_ledger),
        "analysis_physical_union_intervals_samples": [
            list(row) for row in analysis_union
        ],
        "analysis_unique_samples_per_channel": analysis_samples,
        "phase_incremental_samples_sum": phase_incremental,
        "phase_sum_equals_analysis_unique_samples": phase_incremental
        == analysis_samples,
        "baseline_physical_union_intervals_samples": [
            list(row) for row in baseline_union
        ],
        "baseline_unique_samples_per_channel": _interval_samples(baseline_union),
        "total_physical_union_intervals_samples": [
            list(row) for row in physical_union
        ],
        "total_physical_union_samples_per_channel": _interval_samples(
            physical_union
        ),
        "budget_matched_contiguous_comparator": {
            "executed": False,
            "interval_samples": list(comparator),
            "samples_per_channel": comparator[1] - comparator[0],
            "exact_match_to_analysis_unique_samples": (
                comparator[1] - comparator[0] == analysis_samples
            ),
            "same_global_lattice": True,
            "background_overhead_shared_and_excluded_from_match": True,
        },
    }

    envelope_seconds = [
        [_round(start / rate), _round(stop / rate)] for start, stop in envelopes
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": METHOD_ID,
        "design_contract_sha256": _canonical_sha256(_contract()),
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "v2_baseline_and_changepoint_method_id": ADAPTIVE_SUPPORT_V2_METHOD_ID,
        "v2_policy_sha256": v2_policy.sha256,
        "event_id": event,
        "recording_id": recording,
        "navigation_anchor_recording_seconds": _round(anchor_sample / rate),
        "acquisition": {
            "sampling_rate_hz": rate,
            "recording_sample_count": recording_sample_count,
            "channel_order": list(COMMON17_CHANNELS),
            "removed_channels": ["FZ", "PZ"],
            "signal_unit": "V",
            "missing_channel_imputation_used": False,
        },
        "global_candidate_lattice": {
            "interval_samples": [lattice_start, lattice_stop],
            "interval_relative_to_anchor_seconds": [
                policy.global_start_relative_seconds,
                policy.global_stop_relative_seconds,
            ],
            "support_boundaries_used_to_define_lattice": False,
        },
        "frozen_detector_candidate_envelopes": {
            "intervals_recording_seconds": envelope_seconds,
            "sha256": _canonical_sha256(envelope_seconds),
            "explicit_provider_required": True,
            "mandatory_lattice_clipped_acquisition_coverage": True,
            "used_as_clinical_onset": False,
            "used_for_background_normalization": False,
        },
        "remote_baseline_bank": bank_receipt,
        "sparse_sentinel_screening": {
            "probe_schedule_frozen_before_query": True,
            "probe_may_assert_onset_candidate": False,
            "rows": sentinel_rows,
        },
        "dense_candidate_supports": dense_evidence,
        "selected_candidate": selected_candidate,
        "positive_onset_channel_ranking": ranking,
        "onset_causal_evidence_firewall": {
            "positive_rank_source": (
                "selected_candidate_onset_spatial_posterior_mass_only"
            ),
            "candidate_locked_prefix_seconds": policy.positive_ranking_prefix_seconds,
            "late_peak_score_used_for_tie_break": False,
            "late_spread_course_or_recovery_may_increase_positive_rank": False,
            "late_evidence_permission": "description_or_confidence_reduction_only",
        },
        "query_budget_ledger": budget,
        "scope_receipt": deepcopy(_SCOPE),
        "claim_limits": deepcopy(_CLAIMS),
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_common17_adaptive_acquisition_v3(body)


def validate_common17_adaptive_acquisition_v3(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("adaptive acquisition v3 receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "design_contract_sha256",
        "policy",
        "policy_sha256",
        "v2_baseline_and_changepoint_method_id",
        "v2_policy_sha256",
        "event_id",
        "recording_id",
        "navigation_anchor_recording_seconds",
        "acquisition",
        "global_candidate_lattice",
        "frozen_detector_candidate_envelopes",
        "remote_baseline_bank",
        "sparse_sentinel_screening",
        "dense_candidate_supports",
        "selected_candidate",
        "positive_onset_channel_ranking",
        "onset_causal_evidence_firewall",
        "query_budget_ledger",
        "scope_receipt",
        "claim_limits",
    }
    if set(data) != required:
        raise ValueError("adaptive acquisition v3 receipt fields drifted")
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["method_id"] != METHOD_ID
        or data["design_contract_sha256"] != _canonical_sha256(_contract())
        or data["policy_sha256"] != _canonical_sha256(data["policy"])
        or data["v2_baseline_and_changepoint_method_id"]
        != ADAPTIVE_SUPPORT_V2_METHOD_ID
        or data["v2_policy_sha256"] != DEFAULT_ADAPTIVE_SUPPORT_V2_POLICY.sha256
    ):
        raise ValueError("adaptive acquisition v3 method binding drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    acquisition = data["acquisition"]
    if (
        acquisition.get("channel_order") != list(COMMON17_CHANNELS)
        or acquisition.get("removed_channels") != ["FZ", "PZ"]
        or acquisition.get("missing_channel_imputation_used") is not False
    ):
        raise ValueError("adaptive acquisition v3 common17 contract drifted")
    lattice = data["global_candidate_lattice"]
    if (
        lattice.get("interval_relative_to_anchor_seconds") != [-60.0, 60.0]
        or lattice.get("support_boundaries_used_to_define_lattice") is not False
    ):
        raise ValueError("adaptive acquisition v3 lattice drifted")
    detector = data["frozen_detector_candidate_envelopes"]
    if (
        detector.get("explicit_provider_required") is not True
        or detector.get("mandatory_lattice_clipped_acquisition_coverage") is not True
        or detector.get("used_as_clinical_onset") is not False
        or detector.get("used_for_background_normalization") is not False
        or detector.get("sha256")
        != _canonical_sha256(detector.get("intervals_recording_seconds"))
    ):
        raise ValueError("adaptive acquisition v3 detector envelope escaped its role")
    sentinels = data["sparse_sentinel_screening"]
    sentinel_rows = sentinels.get("rows", [])
    expected_sentinels = _contract()["sparse_sentinel_acquisition"][
        "probe_intervals_relative_to_anchor_seconds"
    ]
    if (
        sentinels.get("probe_schedule_frozen_before_query") is not True
        or sentinels.get("probe_may_assert_onset_candidate") is not False
        or not isinstance(sentinel_rows, list)
        or len(sentinel_rows) not in {0, len(expected_sentinels)}
        or (
            len(sentinel_rows) == len(expected_sentinels)
            and any(
                row.get("ordinal") != index
                or row.get("relative_interval_seconds") != relative
                for index, (row, relative) in enumerate(
                    zip(sentinel_rows, expected_sentinels)
                )
            )
        )
        or any(
            row.get("screening", {}).get("may_assert_onset_candidate") is not False
            for row in sentinel_rows
            if row.get("screening") is not None
        )
    ):
        raise ValueError("adaptive acquisition v3 sentinel gained onset authority")
    ranking = data["positive_onset_channel_ranking"]
    if (
        ranking.get("ranking_sha256")
        != _canonical_sha256(ranking.get("ranking"))
        or ranking.get("late_peak_score_used_for_tie_break") is not False
        or ranking.get("late_spread_course_or_recovery_may_increase_positive_rank")
        is not False
        or ranking.get("clinical_SOZ_claim_authorized") is not False
    ):
        raise ValueError("adaptive acquisition v3 positive evidence firewall drifted")
    prefix_audit = ranking.get("candidate_locked_prefix_audit")
    if ranking.get("status") == "candidate_locked_positive_ranking_available":
        rows = ranking.get("ranking")
        if (
            not isinstance(rows, list)
            or len(rows) != len(COMMON17_CHANNELS)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get("channel") for row in rows} != set(COMMON17_CHANNELS)
            or [row.get("rank") for row in rows]
            != list(range(1, len(COMMON17_CHANNELS) + 1))
            or not isinstance(prefix_audit, dict)
            or prefix_audit.get("raw_sample_truncation_precedes_feature_extraction")
            is not True
            or prefix_audit.get("postprefix_samples_used_for_positive_rank")
            is not False
        ):
            raise ValueError("adaptive acquisition v3 candidate-locked ranking drifted")
        try:
            masses = {
                str(row["channel"]): float(row["onset_spatial_posterior_mass"])
                for row in rows
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "adaptive acquisition v3 ranking mass drifted"
            ) from error
        expected_channels = sorted(
            COMMON17_CHANNELS,
            key=lambda channel: (
                -masses[channel],
                COMMON17_CHANNELS.index(channel),
            ),
        )
        if (
            [row["channel"] for row in rows] != expected_channels
            or any(not math.isfinite(value) or value < 0.0 for value in masses.values())
            or not math.isclose(sum(masses.values()), 1.0, abs_tol=2.0e-5)
        ):
            raise ValueError("adaptive acquisition v3 ranking order drifted")
        interval = prefix_audit.get("raw_interval_samples")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or interval[1] - interval[0]
            != prefix_audit.get("samples_per_channel")
            or prefix_audit.get("samples_per_channel")
            != int(
                round(
                    float(data["policy"]["positive_ranking_prefix_seconds"])
                    * float(acquisition["sampling_rate_hz"])
                )
            )
        ):
            raise ValueError("adaptive acquisition v3 ranking prefix budget drifted")
    elif (
        ranking.get("status")
        not in {"background_censored", "no_qualified_multiscale_candidate"}
        or ranking.get("ranking") != []
        or prefix_audit is not None
    ):
        raise ValueError("adaptive acquisition v3 empty ranking state drifted")
    firewall = data["onset_causal_evidence_firewall"]
    if firewall != {
        "positive_rank_source": (
            "selected_candidate_onset_spatial_posterior_mass_only"
        ),
        "candidate_locked_prefix_seconds": 2.0,
        "late_peak_score_used_for_tie_break": False,
        "late_spread_course_or_recovery_may_increase_positive_rank": False,
        "late_evidence_permission": "description_or_confidence_reduction_only",
    }:
        raise ValueError("adaptive acquisition v3 onset-causal firewall drifted")
    budget = data["query_budget_ledger"]
    comparator = budget.get("budget_matched_contiguous_comparator", {})
    analysis_rows = _validated_interval_rows(
        budget.get("analysis_physical_union_intervals_samples"),
        field="analysis union",
    )
    baseline_rows = _validated_interval_rows(
        budget.get("baseline_physical_union_intervals_samples"),
        field="baseline union",
    )
    physical_rows = _validated_interval_rows(
        budget.get("total_physical_union_intervals_samples"),
        field="total physical union",
    )
    phase_rows = budget.get("analysis_query_phase_ledger")
    if not isinstance(phase_rows, list):
        raise ValueError("adaptive acquisition v3 query phase ledger drifted")
    phase_new_intervals: list[tuple[int, int]] = []
    recomputed_phase_sum = 0
    for row in phase_rows:
        if not isinstance(row, dict):
            raise ValueError("adaptive acquisition v3 query phase row drifted")
        new_rows = _validated_interval_rows(
            row.get("new_query_intervals_samples"),
            field="phase new-query",
        )
        incremental = sum(stop - start for start, stop in new_rows)
        if row.get("incremental_unique_samples_per_channel") != incremental:
            raise ValueError("adaptive acquisition v3 phase increment drifted")
        phase_new_intervals.extend(new_rows)
        recomputed_phase_sum += incremental
    analysis_samples = _interval_samples(analysis_rows)
    baseline_samples = _interval_samples(baseline_rows)
    total_samples = _interval_samples(physical_rows)
    comparator_interval = _validated_interval_rows(
        [comparator.get("interval_samples")],
        field="matched comparator",
        allow_zero_width=True,
    )
    lattice_interval = _validated_interval_rows(
        [lattice.get("interval_samples")], field="global lattice"
    )[0]
    if data["remote_baseline_bank"].get("status") == (
        "qualified_consistent_preanchor_background"
    ):
        rate = float(acquisition["sampling_rate_hz"])
        anchor_sample = int(
            round(float(data["navigation_anchor_recording_seconds"]) * rate)
        )
        required_analysis: list[tuple[int, int]] = []
        initial_relative = data["policy"]["initial_interval_relative_seconds"]
        initial = _record_clipped_relative_interval(
            (float(initial_relative[0]), float(initial_relative[1])),
            anchor_sample=anchor_sample,
            rate=rate,
            lower=lattice_interval[0],
            upper=lattice_interval[1],
        )
        if initial is not None:
            required_analysis.append(initial)
        for start_seconds, stop_seconds in detector[
            "intervals_recording_seconds"
        ]:
            clipped = (
                max(lattice_interval[0], int(round(float(start_seconds) * rate))),
                min(lattice_interval[1], int(round(float(stop_seconds) * rate))),
            )
            if clipped[1] > clipped[0]:
                required_analysis.append(clipped)
        for row in sentinel_rows:
            interval = row.get("interval_samples")
            if interval is not None:
                required_analysis.extend(
                    _validated_interval_rows([interval], field="sentinel acquisition")
                )
        if any(
            not _interval_is_covered(interval, analysis_rows)
            for interval in required_analysis
        ):
            raise ValueError("adaptive acquisition v3 mandatory coverage drifted")
    if (
        analysis_rows != _merge_intervals(analysis_rows)
        or baseline_rows != _merge_intervals(baseline_rows)
        or physical_rows != _merge_intervals((*baseline_rows, *analysis_rows))
        or _merge_intervals(phase_new_intervals) != analysis_rows
        or sum(stop - start for start, stop in phase_new_intervals)
        != analysis_samples
        or budget.get("analysis_unique_samples_per_channel") != analysis_samples
        or budget.get("baseline_unique_samples_per_channel") != baseline_samples
        or budget.get("total_physical_union_samples_per_channel") != total_samples
        or recomputed_phase_sum != analysis_samples
        or budget.get("phase_incremental_samples_sum")
        != budget.get("analysis_unique_samples_per_channel")
        or budget.get("phase_sum_equals_analysis_unique_samples") is not True
        or comparator.get("executed") is not False
        or comparator.get("samples_per_channel")
        != budget.get("analysis_unique_samples_per_channel")
        or comparator_interval[0][1] - comparator_interval[0][0]
        != analysis_samples
        or comparator_interval[0][0] < lattice_interval[0]
        or comparator_interval[0][1] > lattice_interval[1]
        or comparator.get("exact_match_to_analysis_unique_samples") is not True
        or comparator.get("background_overhead_shared_and_excluded_from_match")
        is not True
    ):
        raise ValueError("adaptive acquisition v3 matched budget does not close")
    if data["scope_receipt"] != _SCOPE or data["claim_limits"] != _CLAIMS:
        raise ValueError("adaptive acquisition v3 claim boundary drifted")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("adaptive acquisition v3 content hash mismatch")
    return data


__all__ = [
    "DEFAULT_POLICY",
    "METHOD_ID",
    "SCHEMA_VERSION",
    "AdaptiveAcquisitionV3Policy",
    "materialize_common17_adaptive_acquisition_v3",
    "validate_common17_adaptive_acquisition_v3",
]
