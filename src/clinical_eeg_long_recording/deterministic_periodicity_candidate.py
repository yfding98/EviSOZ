"""Replayable signal-only element-interval candidate for EEG periodicity.

The clinical word *periodic* cannot be recovered from a spectral maximum or
an autocorrelation peak alone.  It requires repeated, individually
identifiable waveform elements and a quantifiable series of successive
inter-element intervals.  This module implements only that missing native
measurement primitive.  It deliberately does **not** qualify periodic
discharges, an ictal pattern, an onset channel, a scalp SOZ, or a clinical
diagnosis.

The producer consumes an already materialised canonical EEG receipt and one
or more trusted Findings views.  It performs no I/O and has no route to EDF
annotations, spreadsheets, doctor labels, clinical text, video, behaviour or
patient metadata.  A deterministic robust-amplitude segmentation is run on
one explicitly named analysis unit.  A candidate is emitted only when the
following objects can all be serialized and replayed:

* a de-duplicated roster of bounded waveform elements;
* the onset-to-onset and peak-to-peak interval between every successive pair;
* element duration, interval variability, occurrence and burden summaries;
* exact view/sample/time/source bindings and v2 raw-sample dependencies.

Insufficient duration, physical bandwidth, morphology/amplitude eligibility,
quality opportunity, robust scale, bounded elements or successive intervals
returns ``not_evaluable``.  Such a result carries ``count=None`` and never
means zero/absence.  Spectral and autocorrelation features are neither read
nor computed.  Even when a future-free causal view is used as a fallback, the
candidate itself remains context-only and ``onset_support_eligible=False``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .canonical_signal_views import (
    validate_canonical_signal_receipt,
    view_tensor_index_to_recording_seconds,
)
from .deterministic_event_findings import (
    DeterministicViewInput,
    _canonical_sha256,
    _identifier,
    _invalid_samples,
    _prepare_views,
)
from .deterministic_event_findings_v2 import (
    DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
    _EvidenceBuilder,
    _mask_component_sha256,
    _unit_is_eligible,
    _view_bandwidth,
    _view_role,
)
from .event_findings_v2_validation import _validate_raw_sample_dependency


DETERMINISTIC_PERIODICITY_CANDIDATE_SCHEMA_VERSION: Final[str] = (
    "deterministic_eeg_element_interval_candidate_v1"
)
DETERMINISTIC_PERIODICITY_CANDIDATE_METHOD_ID: Final[str] = (
    "DETERMINISTIC-EEG-ELEMENT-INTERVAL-CANDIDATE-V1"
)
DETERMINISTIC_PERIODICITY_CANDIDATE_TERM_ID: Final[str] = (
    "deterministic_element_interval_profile_candidate"
)
DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID: Final[str] = (
    "DETERMINISTIC-ELEMENT-SEGMENTATION-AND-INTERVAL-POLICY-V1"
)
DETERMINISTIC_PERIODICITY_DEDUPLICATION_POLICY_ID: Final[str] = (
    "ROBUST-EXCURSION-OVERLAP-AND-REFRACTORY-DEDUPLICATION-V1"
)
_CANDIDATE_ID_DOMAIN: Final[str] = (
    "clinical-eeg-deterministic-element-interval-candidate-id-v1"
)
_CANDIDATE_DIGEST_DOMAIN: Final[str] = (
    "clinical-eeg-deterministic-element-interval-candidate-digest-v1"
)
_ELEMENT_ROSTER_DIGEST_DOMAIN: Final[str] = (
    "clinical-eeg-deterministic-element-roster-v1"
)
_INTERVAL_LEDGER_DIGEST_DOMAIN: Final[str] = (
    "clinical-eeg-deterministic-successive-element-interval-ledger-v1"
)
_VIEW_PRIORITY: Final[dict[str, int]] = {
    "context_offline": 0,
    "canonical_physical_evidence": 1,
    "onset_causal": 2,
}
_TOL: Final[float] = 1e-9


@dataclass(frozen=True)
class DeterministicPeriodicityCandidatePolicy:
    """Frozen engineering policy for explicit element segmentation.

    These thresholds produce a research measurement candidate.  They are not
    clinical periodic-discharge criteria and have no qualification receipt.
    ``minimum_required_upper_bandwidth_hz`` is derived from the shortest
    permitted element as ``1 / (2 * minimum_element_duration_seconds)``.
    """

    minimum_analyzable_seconds: float = 2.0
    minimum_element_count: int = 4
    minimum_element_duration_seconds: float = 0.030
    maximum_element_duration_seconds: float = 0.250
    minimum_samples_per_element: int = 3
    peak_threshold_robust_z: float = 5.0
    return_threshold_robust_z: float = 1.5
    envelope_smoothing_seconds: float = 0.010
    minimum_peak_separation_seconds: float = 0.060
    merge_gap_seconds: float = 0.020
    boundary_guard_seconds: float = 0.015
    minimum_robust_scale_uv: float = 0.10

    def __post_init__(self) -> None:
        positive_float_fields = (
            "minimum_analyzable_seconds",
            "minimum_element_duration_seconds",
            "maximum_element_duration_seconds",
            "peak_threshold_robust_z",
            "return_threshold_robust_z",
            "envelope_smoothing_seconds",
            "minimum_peak_separation_seconds",
            "merge_gap_seconds",
            "boundary_guard_seconds",
            "minimum_robust_scale_uv",
        )
        for name in positive_float_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.minimum_element_count) is not int or self.minimum_element_count < 3:
            raise ValueError("minimum_element_count must be an integer >= 3")
        if (
            type(self.minimum_samples_per_element) is not int
            or self.minimum_samples_per_element < 3
        ):
            raise ValueError("minimum_samples_per_element must be an integer >= 3")
        if self.maximum_element_duration_seconds <= self.minimum_element_duration_seconds:
            raise ValueError("maximum element duration must exceed minimum duration")
        if self.return_threshold_robust_z >= self.peak_threshold_robust_z:
            raise ValueError("return threshold must be lower than peak threshold")
        if self.merge_gap_seconds > self.minimum_peak_separation_seconds:
            raise ValueError("merge gap cannot exceed the peak-separation interval")

    @property
    def minimum_interval_count(self) -> int:
        return self.minimum_element_count - 1

    @property
    def minimum_required_upper_bandwidth_hz(self) -> float:
        return 1.0 / (2.0 * self.minimum_element_duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["minimum_interval_count"] = self.minimum_interval_count
        result["minimum_required_upper_bandwidth_hz"] = (
            self.minimum_required_upper_bandwidth_hz
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "policy_id": DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID,
                "policy": self.to_dict(),
            }
        )


DEFAULT_DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY = (
    DeterministicPeriodicityCandidatePolicy()
)


def _finite_interval(
    value: Sequence[float], name: str
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    start, stop = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError(f"{name} must be a finite positive interval")
    return start, stop


def _overlap_samples(
    interval: Sequence[int], target: tuple[int, int]
) -> int:
    start, stop = (int(item) for item in interval)
    return max(0, min(stop, target[1]) - max(start, target[0]))


def _quality_reason_codes(
    view: Any,
    *,
    unit_id: str,
    unit_index: int,
    tensor_interval: tuple[int, int],
) -> tuple[list[str], float]:
    reasons: list[str] = []
    unit_catalog = {
        str(row["unit_id"]): row for row in view.receipt["output_units"]
    }
    unit = unit_catalog[unit_id]
    required_families = ("amplitude", "morphology", "waveform")
    if any(not _unit_is_eligible(view, unit_id, family) for family in required_families):
        reasons.append("morphology_amplitude_or_waveform_evidence_ineligible")

    masks = view.receipt["masks"]
    if any(
        _overlap_samples(row, tensor_interval) > 0
        for row in masks["padding_intervals"]
    ):
        reasons.append("padding_overlap_in_analysis_opportunity")
    if any(
        _overlap_samples(row, tensor_interval) > 0
        for row in masks["edge_invalid_intervals"]
    ):
        reasons.append("filter_edge_overlap_in_analysis_opportunity")
    for row in masks["quality_invalid_intervals"]:
        if str(row["unit_id"]) != unit_id:
            continue
        if not set(required_families).intersection(row["disabled_evidence_families"]):
            continue
        if _overlap_samples(row["tensor_sample_interval"], tensor_interval) > 0:
            reasons.append("unusable_quality_overlap_in_analysis_opportunity")
            break

    invalid = np.zeros(view.tensor.shape[1], dtype=bool)
    for family in required_families:
        invalid |= _invalid_samples(view, family=family)[unit_index]
    start, stop = tensor_interval
    usable_fraction = float(1.0 - np.mean(invalid[start:stop]))
    if not bool(unit["observed"]) or bool(unit["imputed"]):
        usable_fraction = 0.0
    return sorted(set(reasons)), usable_fraction


def _view_attempt(
    view: Any,
    *,
    unit_id: str,
    unit_index: int,
    policy: DeterministicPeriodicityCandidatePolicy,
) -> dict[str, Any]:
    role = _view_role(view)
    if role not in _VIEW_PRIORITY:
        raise ValueError("periodicity candidate requires a qualified Findings view")
    start, stop = view.final_tensor_interval
    actual_start = view_tensor_index_to_recording_seconds(
        view.receipt, tensor_sample_index=start
    )
    actual_stop = view_tensor_index_to_recording_seconds(
        view.receipt, tensor_sample_index=stop
    )
    bandwidth = _view_bandwidth(view, [unit_id])
    reasons, usable_fraction = _quality_reason_codes(
        view,
        unit_id=unit_id,
        unit_index=unit_index,
        tensor_interval=(start, stop),
    )
    duration = actual_stop - actual_start
    if duration + _TOL < policy.minimum_analyzable_seconds:
        reasons.append("analysis_opportunity_too_short_for_interval_series")
    if bandwidth[1] + _TOL < policy.minimum_required_upper_bandwidth_hz:
        reasons.append("effective_bandwidth_insufficient_for_element_duration")
    minimum_samples = (
        policy.minimum_element_duration_seconds * view.sampling_rate_hz
    )
    if minimum_samples + _TOL < policy.minimum_samples_per_element:
        reasons.append("sampling_clock_insufficient_for_element_duration")
    return {
        "view": view,
        "view_role": role,
        "unit_index": unit_index,
        "tensor_interval": (start, stop),
        "actual_interval": (actual_start, actual_stop),
        "effective_bandwidth_hz": bandwidth,
        "usable_fraction": usable_fraction,
        "reason_codes": sorted(set(reasons)),
        "capable": not reasons,
    }


def _select_attempt(
    prepared: Sequence[Any],
    *,
    analysis_unit_id: str,
    policy: DeterministicPeriodicityCandidatePolicy,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for view in prepared:
        if analysis_unit_id not in view.unit_ids:
            continue
        attempt = _view_attempt(
            view,
            unit_id=analysis_unit_id,
            unit_index=view.unit_ids.index(analysis_unit_id),
            policy=policy,
        )
        attempts.append(attempt)
    if not attempts:
        raise ValueError("analysis_unit_id is absent from all Findings views")
    attempts.sort(
        key=lambda row: (
            0 if row["capable"] else 1,
            _VIEW_PRIORITY[str(row["view_role"])],
            -float(row["effective_bandwidth_hz"][1]),
            str(row["view"].receipt["view_id"]),
        )
    )
    return attempts[0]


def _robust_linear_detrend(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64, copy=True)
    sample_count = centered.size
    time = np.arange(sample_count, dtype=np.float64)
    time -= float(np.mean(time))
    location = float(np.median(centered))
    denominator = float(np.dot(time, time))
    slope = (
        0.0
        if denominator <= 0.0
        else float(np.dot(time, centered - location) / denominator)
    )
    return centered - (location + slope * time)


def _moving_rms(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return np.abs(values)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.sqrt(
        np.maximum(
            np.convolve(values * values, kernel, mode="same"),
            0.0,
        )
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(
        [np.array([False]), mask.astype(bool, copy=False), np.array([False])]
    )
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _candidate_peak_indices(
    robust_z: np.ndarray,
    *,
    threshold: float,
) -> list[int]:
    peaks: list[int] = []
    for start, stop in _true_runs(robust_z >= threshold):
        local = robust_z[start:stop]
        if local.size:
            # np.argmax resolves a plateau to its earliest sample, which is
            # part of the frozen deterministic tie policy.
            peaks.append(start + int(np.argmax(local)))
    return peaks


def _segment_peak(
    peak: int,
    *,
    envelope_z: np.ndarray,
    return_threshold: float,
    guard_samples: int,
) -> tuple[int, int] | None:
    start = int(peak)
    while start > 0 and envelope_z[start - 1] >= return_threshold:
        start -= 1
    stop = int(peak) + 1
    while stop < envelope_z.size and envelope_z[stop] >= return_threshold:
        stop += 1
    if start < guard_samples or stop > envelope_z.size - guard_samples:
        return None
    return start, stop


def _deduplicate_segments(
    segments: Sequence[dict[str, Any]],
    *,
    minimum_peak_separation_samples: int,
    merge_gap_samples: int,
) -> list[dict[str, Any]]:
    """Cluster overlapping/nearby proposals and keep one deterministic peak."""

    if not segments:
        return []
    ordered = sorted(
        (deepcopy(dict(row)) for row in segments),
        key=lambda row: (int(row["start"]), int(row["stop"]), int(row["peak"])),
    )
    clusters: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not clusters:
            clusters.append([row])
            continue
        previous = clusters[-1]
        cluster_stop = max(int(item["stop"]) for item in previous)
        cluster_peaks = [int(item["peak"]) for item in previous]
        near_interval = min(abs(int(row["peak"]) - item) for item in cluster_peaks)
        if (
            int(row["start"]) <= cluster_stop + merge_gap_samples
            or near_interval < minimum_peak_separation_samples
        ):
            previous.append(row)
        else:
            clusters.append([row])

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        selected = min(
            cluster,
            key=lambda row: (-float(row["peak_z"]), int(row["peak"])),
        )
        result.append(
            {
                "start": min(int(row["start"]) for row in cluster),
                "stop": max(int(row["stop"]) for row in cluster),
                "peak": int(selected["peak"]),
                "peak_z": float(selected["peak_z"]),
                "proposal_count": len(cluster),
            }
        )
    return sorted(result, key=lambda row: (int(row["start"]), int(row["peak"])))


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def _segmentation(
    values_volts: np.ndarray,
    *,
    sampling_rate_hz: float,
    policy: DeterministicPeriodicityCandidatePolicy,
) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    detrended = _robust_linear_detrend(values_volts)
    centered = detrended - float(np.median(detrended))
    robust_scale_volts = 1.4826 * float(np.median(np.abs(centered)))
    robust_scale_uv = robust_scale_volts * 1e6
    if (
        not math.isfinite(robust_scale_uv)
        or robust_scale_uv < policy.minimum_robust_scale_uv
    ):
        return [], None

    robust_z = np.abs(centered) / robust_scale_volts
    smoothing_samples = max(
        1, int(round(policy.envelope_smoothing_seconds * sampling_rate_hz))
    )
    envelope_z = _moving_rms(centered, smoothing_samples) / robust_scale_volts
    guard_samples = max(
        1, int(math.ceil(policy.boundary_guard_seconds * sampling_rate_hz))
    )
    proposals: list[dict[str, Any]] = []
    for peak in _candidate_peak_indices(
        robust_z, threshold=policy.peak_threshold_robust_z
    ):
        segment = _segment_peak(
            peak,
            envelope_z=envelope_z,
            return_threshold=policy.return_threshold_robust_z,
            guard_samples=guard_samples,
        )
        if segment is None:
            continue
        proposals.append(
            {
                "start": int(segment[0]),
                "stop": int(segment[1]),
                "peak": int(peak),
                "peak_z": float(robust_z[peak]),
            }
        )

    deduplicated = _deduplicate_segments(
        proposals,
        minimum_peak_separation_samples=max(
            1,
            int(round(policy.minimum_peak_separation_seconds * sampling_rate_hz)),
        ),
        merge_gap_samples=max(
            0, int(round(policy.merge_gap_seconds * sampling_rate_hz))
        ),
    )
    minimum_duration_samples = max(
        policy.minimum_samples_per_element,
        int(math.ceil(policy.minimum_element_duration_seconds * sampling_rate_hz)),
    )
    maximum_duration_samples = max(
        minimum_duration_samples,
        int(math.floor(policy.maximum_element_duration_seconds * sampling_rate_hz)),
    )
    accepted = [
        row
        for row in deduplicated
        if minimum_duration_samples
        <= int(row["stop"]) - int(row["start"])
        <= maximum_duration_samples
    ]
    diagnostics = {
        "robust_scale_uv": robust_scale_uv,
        "smoothing_samples": float(smoothing_samples),
        "raw_excursion_count": float(len(proposals)),
        "deduplicated_segment_count": float(len(deduplicated)),
        "accepted_segment_count": float(len(accepted)),
    }
    return accepted, diagnostics


def _source_binding(
    attempt: Mapping[str, Any],
    *,
    canonical_signal_sha256: str,
    analysis_unit_id: str,
    requested_interval: tuple[float, float],
    policy: DeterministicPeriodicityCandidatePolicy,
) -> dict[str, Any]:
    view = attempt["view"]
    start, stop = attempt["tensor_interval"]
    return {
        "canonical_signal_sha256": canonical_signal_sha256,
        "source_view_id": str(view.receipt["view_id"]),
        "source_view_role": str(attempt["view_role"]),
        "source_view_receipt_id": str(view.receipt["view_receipt_id"]),
        "source_view_receipt_sha256": str(view.receipt["receipt_sha256"]),
        "transform_spec_sha256": str(
            view.receipt["transform_spec"]["transform_spec_sha256"]
        ),
        "processed_view_sha256": str(view.receipt["processed_view_sha256"]),
        "analysis_unit_id": analysis_unit_id,
        "requested_recording_interval": [
            float(requested_interval[0]),
            float(requested_interval[1]),
        ],
        "actual_recording_interval": [
            float(attempt["actual_interval"][0]),
            float(attempt["actual_interval"][1]),
        ],
        "tensor_sample_interval": [int(start), int(stop)],
        "sample_rate_hz": float(view.sampling_rate_hz),
        "effective_bandwidth_hz": [
            float(item) for item in attempt["effective_bandwidth_hz"]
        ],
        "minimum_required_upper_bandwidth_hz": float(
            policy.minimum_required_upper_bandwidth_hz
        ),
        "reference_type": str(
            view.receipt["transform_spec"]["reference"]["reference_type"]
        ),
        "quality_mask_sha256": str(view.receipt["masks"]["mask_sha256"]),
        "edge_mask_sha256": _mask_component_sha256(
            view, "edge_invalid_intervals"
        ),
        "padding_mask_sha256": _mask_component_sha256(
            view, "padding_intervals"
        ),
        "usable_fraction": float(attempt["usable_fraction"]),
        "raw_dependency_method_id": DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
    }


def _empty_measurement_blocks() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    occurrence = {
        "status": "not_evaluable",
        "count": None,
        "evaluable_seconds": None,
        "rate_per_minute": None,
        "observed_seconds": None,
        "burden_fraction": None,
        "element_roster": [],
        "element_roster_sha256": None,
        "deduplication_policy_id": None,
    }
    intervals = {
        "status": "not_evaluable",
        "interval_count": None,
        "successive_intervals": [],
        "interval_ledger_sha256": None,
        "onset_to_onset_median_seconds": None,
        "onset_to_onset_mad_seconds": None,
        "onset_to_onset_robust_cv": None,
        "onset_to_onset_cv": None,
        "peak_to_peak_median_seconds": None,
        "peak_to_peak_mad_seconds": None,
    }
    durations = {
        "status": "not_evaluable",
        "element_duration_median_seconds": None,
        "element_duration_mad_seconds": None,
        "element_duration_min_seconds": None,
        "element_duration_max_seconds": None,
    }
    return occurrence, intervals, durations


def _finalize_candidate(body: Mapping[str, Any]) -> dict[str, Any]:
    identifier_source = deepcopy(dict(body))
    identifier = "ELEMINT-" + _canonical_sha256(
        {"domain": _CANDIDATE_ID_DOMAIN, "candidate": identifier_source}
    )[:24]
    result = {"candidate_id": identifier, **identifier_source}
    digest_source = deepcopy(result)
    digest_source["candidate_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["candidate_sha256"] = _canonical_sha256(
        {"domain": _CANDIDATE_DIGEST_DOMAIN, "candidate": digest_source}
    )
    return result


def _not_evaluable_candidate(
    *,
    event_id: str,
    analysis_unit_id: str,
    requested_interval: tuple[float, float],
    source_binding: Mapping[str, Any],
    opportunity_reasons: Sequence[str],
    policy: DeterministicPeriodicityCandidatePolicy,
) -> dict[str, Any]:
    occurrence, intervals, durations = _empty_measurement_blocks()
    reasons = sorted(set(str(item) for item in opportunity_reasons))
    body = {
        "schema_version": DETERMINISTIC_PERIODICITY_CANDIDATE_SCHEMA_VERSION,
        "method_id": DETERMINISTIC_PERIODICITY_CANDIDATE_METHOD_ID,
        "policy_id": DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID,
        "policy_sha256": policy.sha256,
        "policy": policy.to_dict(),
        "event_id": event_id,
        "term_id": DETERMINISTIC_PERIODICITY_CANDIDATE_TERM_ID,
        "analysis_unit_id": analysis_unit_id,
        "analysis_scope": "signal_only_event_context",
        "requested_recording_interval": list(requested_interval),
        "qualification_status": "not_evaluable",
        "assertion_level": "model_candidate",
        "intrinsic_evidence_role": "early_context",
        "onset_support_eligible": False,
        "soz_support_eligible": False,
        "periodicity_inference_source": (
            "explicit_waveform_element_segmentation_only"
        ),
        "spectral_peak_used": False,
        "autocorrelation_used": False,
        "source_binding": deepcopy(dict(source_binding)),
        "evaluation_opportunity": {
            "status": "not_evaluable",
            "reason_codes": reasons,
        },
        "segmentation": {
            "status": "not_evaluable",
            "algorithm_id": (
                "ROBUST-DETRENDED-AMPLITUDE-EXCURSION-SEGMENTATION-V1"
            ),
            "diagnostics": None,
            "reason_codes": reasons,
        },
        "occurrence": occurrence,
        "successive_interval_profile": intervals,
        "element_duration_profile": durations,
        "analysis_raw_sample_dependency_id": None,
        "raw_sample_dependencies": [],
        "capability_receipt_ids": [],
        "sensitivity_receipt_ids": [],
        "term_decision_receipt_ids": [],
        "clinical_qualification_receipt_ids": [],
        "clinical_term_claims": [],
        "reason_codes": reasons,
        "input_firewall": {
            "edf_annotations_used": False,
            "excel_used": False,
            "doctor_labels_used": False,
            "clinical_text_used": False,
            "patient_metadata_used": False,
        },
    }
    return validate_deterministic_periodicity_candidate(_finalize_candidate(body))


def _element_rows(
    segments: Sequence[Mapping[str, Any]],
    *,
    local_values_volts: np.ndarray,
    local_start_tensor_index: int,
    view: Any,
    analysis_unit_id: str,
    builder: _EvidenceBuilder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(segments, start=1):
        local_start = int(segment["start"])
        local_stop = int(segment["stop"])
        local_peak = int(segment["peak"])
        tensor_start = local_start_tensor_index + local_start
        tensor_stop = local_start_tensor_index + local_stop
        tensor_peak = local_start_tensor_index + local_peak
        recording_start = view_tensor_index_to_recording_seconds(
            view.receipt, tensor_sample_index=tensor_start
        )
        recording_stop = view_tensor_index_to_recording_seconds(
            view.receipt, tensor_sample_index=tensor_stop
        )
        recording_peak = view_tensor_index_to_recording_seconds(
            view.receipt, tensor_sample_index=tensor_peak
        )
        raw_dependency = builder.raw_sample_dependency(
            view=view,
            unit_ids=[analysis_unit_id],
            interval=(recording_start, recording_stop),
            tensor_interval=(tensor_start, tensor_stop),
        )
        peak_uv = float(local_values_volts[local_peak] * 1e6)
        element_body = {
            "ordinal": ordinal,
            "tensor_sample_interval": [tensor_start, tensor_stop],
            "peak_tensor_sample_index": tensor_peak,
            "recording_interval": [recording_start, recording_stop],
            "peak_recording_seconds": recording_peak,
            "duration_seconds": recording_stop - recording_start,
            "peak_amplitude_uv": peak_uv,
            "absolute_peak_robust_z": float(segment["peak_z"]),
            "polarity": "positive" if peak_uv >= 0.0 else "negative",
            "merged_proposal_count": int(segment["proposal_count"]),
            "raw_sample_dependency_id": str(raw_dependency["dependency_id"]),
        }
        element_id = "ELEMENT-" + _canonical_sha256(
            {
                "domain": "clinical-eeg-explicit-waveform-element-v1",
                "event_id": builder.event_id,
                "analysis_unit_id": analysis_unit_id,
                "element": element_body,
            }
        )[:24]
        rows.append({"element_id": element_id, **element_body})
        dependencies.append(
            {
                "dependency_role": "element_waveform",
                "element_id": element_id,
                "raw_sample_dependency": raw_dependency,
            }
        )
    return rows, dependencies


def _successive_interval_rows(
    elements: Sequence[Mapping[str, Any]],
    *,
    sampling_rate_hz: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous, current in zip(elements, elements[1:]):
        onset_samples = int(current["tensor_sample_interval"][0]) - int(
            previous["tensor_sample_interval"][0]
        )
        peak_samples = int(current["peak_tensor_sample_index"]) - int(
            previous["peak_tensor_sample_index"]
        )
        body = {
            "from_element_id": str(previous["element_id"]),
            "to_element_id": str(current["element_id"]),
            "onset_to_onset_samples": onset_samples,
            "onset_to_onset_seconds": onset_samples / sampling_rate_hz,
            "peak_to_peak_samples": peak_samples,
            "peak_to_peak_seconds": peak_samples / sampling_rate_hz,
        }
        interval_id = "ELEMINTV-" + _canonical_sha256(
            {
                "domain": "clinical-eeg-successive-element-interval-v1",
                "interval": body,
            }
        )[:24]
        rows.append({"interval_id": interval_id, **body})
    return rows


def produce_deterministic_periodicity_candidate(
    *,
    event_id: str,
    analysis_unit_id: str,
    analysis_interval_recording_seconds: Sequence[float],
    canonical_receipt: object,
    views: Sequence[DeterministicViewInput],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
    policy: DeterministicPeriodicityCandidatePolicy = (
        DEFAULT_DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY
    ),
) -> dict[str, Any]:
    """Measure one explicit element/interval profile from signal alone.

    Technical receipt/hash/canonical-integrity failures raise.  Ordinary
    signal insufficiency returns a complete ``not_evaluable`` artifact.
    """

    _identifier(event_id, "event_id")
    _identifier(analysis_unit_id, "analysis_unit_id")
    if not isinstance(policy, DeterministicPeriodicityCandidatePolicy):
        raise TypeError("policy must be DeterministicPeriodicityCandidatePolicy")
    requested_interval = _finite_interval(
        analysis_interval_recording_seconds,
        "analysis_interval_recording_seconds",
    )
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    if (
        requested_interval[0] < -_TOL
        or requested_interval[1]
        > float(canonical["recording_duration_seconds"]) + _TOL
    ):
        raise ValueError("analysis interval lies outside the canonical recording")
    prepared = _prepare_views(
        canonical=canonical,
        views=views,
        final_interval=requested_interval,
        trusted_parent_views=trusted_parent_views,
    )
    attempt = _select_attempt(
        prepared,
        analysis_unit_id=analysis_unit_id,
        policy=policy,
    )
    binding = _source_binding(
        attempt,
        canonical_signal_sha256=str(canonical["source_signal_sha256"]),
        analysis_unit_id=analysis_unit_id,
        requested_interval=requested_interval,
        policy=policy,
    )
    if attempt["reason_codes"]:
        return _not_evaluable_candidate(
            event_id=event_id,
            analysis_unit_id=analysis_unit_id,
            requested_interval=requested_interval,
            source_binding=binding,
            opportunity_reasons=attempt["reason_codes"],
            policy=policy,
        )

    view = attempt["view"]
    tensor_start, tensor_stop = attempt["tensor_interval"]
    unit_index = int(attempt["unit_index"])
    values = view.tensor[unit_index, tensor_start:tensor_stop]
    segments, diagnostics = _segmentation(
        values,
        sampling_rate_hz=view.sampling_rate_hz,
        policy=policy,
    )
    if diagnostics is None:
        return _not_evaluable_candidate(
            event_id=event_id,
            analysis_unit_id=analysis_unit_id,
            requested_interval=requested_interval,
            source_binding=binding,
            opportunity_reasons=["robust_baseline_scale_not_measurable"],
            policy=policy,
        )
    if len(segments) < policy.minimum_element_count:
        return _not_evaluable_candidate(
            event_id=event_id,
            analysis_unit_id=analysis_unit_id,
            requested_interval=requested_interval,
            source_binding=binding,
            opportunity_reasons=[
                "too_few_explicit_bounded_elements_for_successive_interval_series"
            ],
            policy=policy,
        )

    montage_eligibility = {
        str(row["unit_id"]): bool(
            row["observed"] and not row["imputed"] and row["evidence_eligible"]
        )
        for row in view.receipt["output_units"]
    }
    builder = _EvidenceBuilder(
        event_id=event_id,
        canonical_receipt=canonical,
        policy_sha256=policy.sha256,
        protection_zone=requested_interval,
        protection_zone_id="PZ-" + _canonical_sha256(
            [event_id, requested_interval]
        )[:20],
        state_spans={},
        montage_eligibility=montage_eligibility,
        resolution=1.0 / view.sampling_rate_hz,
    )
    analysis_dependency = builder.raw_sample_dependency(
        view=view,
        unit_ids=[analysis_unit_id],
        interval=tuple(float(item) for item in attempt["actual_interval"]),
        tensor_interval=(tensor_start, tensor_stop),
    )
    elements, element_dependencies = _element_rows(
        segments,
        local_values_volts=values,
        local_start_tensor_index=tensor_start,
        view=view,
        analysis_unit_id=analysis_unit_id,
        builder=builder,
    )
    interval_rows = _successive_interval_rows(
        elements, sampling_rate_hz=view.sampling_rate_hz
    )
    if len(interval_rows) < policy.minimum_interval_count:
        # Defensive: the element-count gate should make this impossible, but
        # an incomplete series must still fail closed rather than encode zero.
        return _not_evaluable_candidate(
            event_id=event_id,
            analysis_unit_id=analysis_unit_id,
            requested_interval=requested_interval,
            source_binding=binding,
            opportunity_reasons=["successive_interval_series_incomplete"],
            policy=policy,
        )

    onset_intervals = [
        float(row["onset_to_onset_seconds"]) for row in interval_rows
    ]
    peak_intervals = [
        float(row["peak_to_peak_seconds"]) for row in interval_rows
    ]
    durations = [float(row["duration_seconds"]) for row in elements]
    onset_median, onset_mad = _median_mad(onset_intervals)
    peak_median, peak_mad = _median_mad(peak_intervals)
    duration_median, duration_mad = _median_mad(durations)
    onset_mean = float(np.mean(onset_intervals))
    onset_cv = float(np.std(onset_intervals, ddof=0) / onset_mean)
    onset_robust_cv = float(1.4826 * onset_mad / onset_median)
    evaluable_seconds = float(
        attempt["actual_interval"][1] - attempt["actual_interval"][0]
    )
    observed_seconds = float(sum(durations))
    roster_sha256 = _canonical_sha256(
        {
            "domain": _ELEMENT_ROSTER_DIGEST_DOMAIN,
            "event_id": event_id,
            "analysis_unit_id": analysis_unit_id,
            "elements": elements,
        }
    )
    interval_sha256 = _canonical_sha256(
        {
            "domain": _INTERVAL_LEDGER_DIGEST_DOMAIN,
            "event_id": event_id,
            "analysis_unit_id": analysis_unit_id,
            "successive_intervals": interval_rows,
        }
    )
    all_dependencies = [
        {
            "dependency_role": "analysis_scope",
            "element_id": None,
            "raw_sample_dependency": analysis_dependency,
        },
        *element_dependencies,
    ]
    reason_codes = [
        "explicit_elements_and_successive_intervals_measured",
        "candidate_not_clinically_qualified",
        "context_only_not_onset_or_soz_evidence",
    ]
    body = {
        "schema_version": DETERMINISTIC_PERIODICITY_CANDIDATE_SCHEMA_VERSION,
        "method_id": DETERMINISTIC_PERIODICITY_CANDIDATE_METHOD_ID,
        "policy_id": DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID,
        "policy_sha256": policy.sha256,
        "policy": policy.to_dict(),
        "event_id": event_id,
        "term_id": DETERMINISTIC_PERIODICITY_CANDIDATE_TERM_ID,
        "analysis_unit_id": analysis_unit_id,
        "analysis_scope": "signal_only_event_context",
        "requested_recording_interval": list(requested_interval),
        "qualification_status": "candidate_only",
        "assertion_level": "model_candidate",
        "intrinsic_evidence_role": "early_context",
        "onset_support_eligible": False,
        "soz_support_eligible": False,
        "periodicity_inference_source": (
            "explicit_waveform_element_segmentation_only"
        ),
        "spectral_peak_used": False,
        "autocorrelation_used": False,
        "source_binding": binding,
        "evaluation_opportunity": {
            "status": "sufficient",
            "reason_codes": [],
        },
        "segmentation": {
            "status": "measured",
            "algorithm_id": (
                "ROBUST-DETRENDED-AMPLITUDE-EXCURSION-SEGMENTATION-V1"
            ),
            "diagnostics": diagnostics,
            "reason_codes": [],
        },
        "occurrence": {
            "status": "measured",
            "count": len(elements),
            "evaluable_seconds": evaluable_seconds,
            "rate_per_minute": len(elements) * 60.0 / evaluable_seconds,
            "observed_seconds": observed_seconds,
            "burden_fraction": observed_seconds / evaluable_seconds,
            "element_roster": elements,
            "element_roster_sha256": roster_sha256,
            "deduplication_policy_id": (
                DETERMINISTIC_PERIODICITY_DEDUPLICATION_POLICY_ID
            ),
        },
        "successive_interval_profile": {
            "status": "measured",
            "interval_count": len(interval_rows),
            "successive_intervals": interval_rows,
            "interval_ledger_sha256": interval_sha256,
            "onset_to_onset_median_seconds": onset_median,
            "onset_to_onset_mad_seconds": onset_mad,
            "onset_to_onset_robust_cv": onset_robust_cv,
            "onset_to_onset_cv": onset_cv,
            "peak_to_peak_median_seconds": peak_median,
            "peak_to_peak_mad_seconds": peak_mad,
        },
        "element_duration_profile": {
            "status": "measured",
            "element_duration_median_seconds": duration_median,
            "element_duration_mad_seconds": duration_mad,
            "element_duration_min_seconds": min(durations),
            "element_duration_max_seconds": max(durations),
        },
        "analysis_raw_sample_dependency_id": str(
            analysis_dependency["dependency_id"]
        ),
        "raw_sample_dependencies": all_dependencies,
        "capability_receipt_ids": [],
        "sensitivity_receipt_ids": [],
        "term_decision_receipt_ids": [],
        "clinical_qualification_receipt_ids": [],
        "clinical_term_claims": [],
        "reason_codes": reason_codes,
        "input_firewall": {
            "edf_annotations_used": False,
            "excel_used": False,
            "doctor_labels_used": False,
            "clinical_text_used": False,
            "patient_metadata_used": False,
        },
    }
    return validate_deterministic_periodicity_candidate(_finalize_candidate(body))


def validate_deterministic_periodicity_candidate(value: object) -> dict[str, Any]:
    """Validate content hashes, four-state semantics and replay closure."""

    if type(value) is not dict:
        raise TypeError("periodicity candidate must be an object")
    candidate = deepcopy(value)
    if (
        candidate.get("schema_version")
        != DETERMINISTIC_PERIODICITY_CANDIDATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported periodicity candidate schema version")
    if candidate.get("method_id") != DETERMINISTIC_PERIODICITY_CANDIDATE_METHOD_ID:
        raise ValueError("periodicity candidate method_id drifted")
    if candidate.get("policy_id") != DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID:
        raise ValueError("periodicity candidate policy_id drifted")
    policy = DeterministicPeriodicityCandidatePolicy(
        **{
            key: candidate["policy"][key]
            for key in asdict(DEFAULT_DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY)
        }
    )
    if candidate["policy"] != policy.to_dict() or candidate["policy_sha256"] != policy.sha256:
        raise ValueError("periodicity candidate policy content/hash drifted")
    _identifier(candidate["event_id"], "event_id")
    _identifier(candidate["analysis_unit_id"], "analysis_unit_id")
    _finite_interval(
        candidate["requested_recording_interval"],
        "requested_recording_interval",
    )
    if candidate["term_id"] != DETERMINISTIC_PERIODICITY_CANDIDATE_TERM_ID:
        raise ValueError("periodicity candidate term_id drifted")
    if candidate["analysis_scope"] != "signal_only_event_context":
        raise ValueError("periodicity candidate analysis scope is unsupported")
    if candidate["assertion_level"] != "model_candidate":
        raise ValueError("periodicity candidate cannot be a clinical assertion")
    if candidate["intrinsic_evidence_role"] != "early_context":
        raise ValueError("periodicity candidate must remain context-only")
    if candidate["onset_support_eligible"] or candidate["soz_support_eligible"]:
        raise ValueError("periodicity candidate cannot support onset/SOZ")
    if (
        candidate["periodicity_inference_source"]
        != "explicit_waveform_element_segmentation_only"
        or candidate["spectral_peak_used"]
        or candidate["autocorrelation_used"]
    ):
        raise ValueError("periodicity candidate cannot be inferred from spectrum/autocorrelation")
    for key in (
        "capability_receipt_ids",
        "sensitivity_receipt_ids",
        "term_decision_receipt_ids",
        "clinical_qualification_receipt_ids",
        "clinical_term_claims",
    ):
        if candidate[key]:
            raise ValueError(f"deterministic candidate must leave {key} empty")
    if any(bool(item) for item in candidate["input_firewall"].values()):
        raise ValueError("periodicity candidate violates the signal-only firewall")

    source = candidate["source_binding"]
    source_interval = _finite_interval(
        source["actual_recording_interval"], "source actual_recording_interval"
    )
    tensor_interval = tuple(int(item) for item in source["tensor_sample_interval"])
    if tensor_interval[1] <= tensor_interval[0]:
        raise ValueError("source tensor interval is empty")
    if source["analysis_unit_id"] != candidate["analysis_unit_id"]:
        raise ValueError("source analysis unit drifted")
    if source["source_view_role"] not in _VIEW_PRIORITY:
        raise ValueError("source view role is not a qualified Findings role")

    status = candidate["qualification_status"]
    if status == "not_evaluable":
        if candidate["evaluation_opportunity"]["status"] != "not_evaluable":
            raise ValueError("not-evaluable candidate needs a closed opportunity")
        if not candidate["reason_codes"]:
            raise ValueError("not-evaluable candidate requires reason codes")
        if candidate["segmentation"]["status"] != "not_evaluable":
            raise ValueError("not-evaluable candidate cannot carry measured segmentation")
        if (
            sorted(set(candidate["reason_codes"]))
            != sorted(set(candidate["evaluation_opportunity"]["reason_codes"]))
            or sorted(set(candidate["reason_codes"]))
            != sorted(set(candidate["segmentation"]["reason_codes"]))
        ):
            raise ValueError("not-evaluable reason ledgers must agree")
        if candidate["raw_sample_dependencies"] or candidate["analysis_raw_sample_dependency_id"] is not None:
            raise ValueError("not-evaluable candidate cannot carry positive raw support")
        occurrence = candidate["occurrence"]
        interval_profile = candidate["successive_interval_profile"]
        durations = candidate["element_duration_profile"]
        if occurrence["status"] != "not_evaluable" or any(
            occurrence[key] is not None
            for key in (
                "count",
                "evaluable_seconds",
                "rate_per_minute",
                "observed_seconds",
                "burden_fraction",
                "element_roster_sha256",
                "deduplication_policy_id",
            )
        ) or occurrence["element_roster"]:
            raise ValueError("not-evaluable occurrence cannot encode zero/absence")
        if interval_profile["status"] != "not_evaluable" or interval_profile["successive_intervals"]:
            raise ValueError("not-evaluable candidate cannot carry an interval series")
        if any(
            interval_profile[key] is not None
            for key in interval_profile
            if key not in {"status", "successive_intervals"}
        ):
            raise ValueError("not-evaluable interval summary must remain null")
        if durations["status"] != "not_evaluable" or any(
            durations[key] is not None for key in durations if key != "status"
        ):
            raise ValueError("not-evaluable duration summary must remain null")
    elif status == "candidate_only":
        if candidate["evaluation_opportunity"] != {
            "status": "sufficient",
            "reason_codes": [],
        }:
            raise ValueError("candidate-only periodicity requires a sufficient opportunity")
        if candidate["segmentation"]["status"] != "measured":
            raise ValueError("candidate-only periodicity requires measured segmentation")
        occurrence = candidate["occurrence"]
        interval_profile = candidate["successive_interval_profile"]
        durations = candidate["element_duration_profile"]
        if occurrence["status"] != "measured" or interval_profile["status"] != "measured" or durations["status"] != "measured":
            raise ValueError("candidate-only periodicity requires complete measurements")
        elements = occurrence["element_roster"]
        intervals = interval_profile["successive_intervals"]
        if int(occurrence["count"]) != len(elements) or len(elements) < policy.minimum_element_count:
            raise ValueError("element roster does not meet the frozen minimum")
        if occurrence["deduplication_policy_id"] != DETERMINISTIC_PERIODICITY_DEDUPLICATION_POLICY_ID:
            raise ValueError("element roster deduplication policy drifted")
        if int(interval_profile["interval_count"]) != len(intervals) or len(intervals) != len(elements) - 1:
            raise ValueError("successive interval ledger is incomplete")
        if [int(row["ordinal"]) for row in elements] != list(range(1, len(elements) + 1)):
            raise ValueError("element roster ordinals are not replayable")
        if elements != sorted(
            elements,
            key=lambda row: (
                int(row["tensor_sample_interval"][0]),
                int(row["peak_tensor_sample_index"]),
            ),
        ):
            raise ValueError("element roster is not in physical-time order")
        previous_stop: int | None = None
        for index, element in enumerate(elements):
            start, stop = (int(item) for item in element["tensor_sample_interval"])
            peak = int(element["peak_tensor_sample_index"])
            if not (tensor_interval[0] <= start <= peak < stop <= tensor_interval[1]):
                raise ValueError("element samples lie outside the analysis opportunity")
            if previous_stop is not None and start < previous_stop:
                raise ValueError("de-duplicated waveform elements cannot overlap")
            previous_stop = stop
            element_interval = _finite_interval(
                element["recording_interval"], f"element[{index}].recording_interval"
            )
            if element_interval[0] < source_interval[0] - _TOL or element_interval[1] > source_interval[1] + _TOL:
                raise ValueError("element time lies outside the analysis opportunity")
            if abs(float(element["duration_seconds"]) - (element_interval[1] - element_interval[0])) > _TOL:
                raise ValueError("element duration is not replayable")
            element_body = deepcopy(element)
            serialized_element_id = str(element_body.pop("element_id"))
            expected_element_id = "ELEMENT-" + _canonical_sha256(
                {
                    "domain": "clinical-eeg-explicit-waveform-element-v1",
                    "event_id": candidate["event_id"],
                    "analysis_unit_id": candidate["analysis_unit_id"],
                    "element": element_body,
                }
            )[:24]
            if serialized_element_id != expected_element_id:
                raise ValueError("element_id does not bind its waveform content")
        for index, row in enumerate(intervals):
            previous = elements[index]
            current = elements[index + 1]
            if row["from_element_id"] != previous["element_id"] or row["to_element_id"] != current["element_id"]:
                raise ValueError("successive interval endpoints drifted")
            onset_samples = int(current["tensor_sample_interval"][0]) - int(previous["tensor_sample_interval"][0])
            peak_samples = int(current["peak_tensor_sample_index"]) - int(previous["peak_tensor_sample_index"])
            if int(row["onset_to_onset_samples"]) != onset_samples or int(row["peak_to_peak_samples"]) != peak_samples:
                raise ValueError("successive sample interval is not replayable")
            sampling_rate = float(source["sample_rate_hz"])
            if abs(float(row["onset_to_onset_seconds"]) - onset_samples / sampling_rate) > _TOL or abs(float(row["peak_to_peak_seconds"]) - peak_samples / sampling_rate) > _TOL:
                raise ValueError("successive time interval is not replayable")
            interval_body = deepcopy(row)
            serialized_interval_id = str(interval_body.pop("interval_id"))
            expected_interval_id = "ELEMINTV-" + _canonical_sha256(
                {
                    "domain": "clinical-eeg-successive-element-interval-v1",
                    "interval": interval_body,
                }
            )[:24]
            if serialized_interval_id != expected_interval_id:
                raise ValueError("interval_id does not bind its successive elements")

        evaluable_seconds = source_interval[1] - source_interval[0]
        element_durations = [float(row["duration_seconds"]) for row in elements]
        observed_seconds = float(sum(element_durations))
        expected_occurrence_values = {
            "evaluable_seconds": evaluable_seconds,
            "rate_per_minute": len(elements) * 60.0 / evaluable_seconds,
            "observed_seconds": observed_seconds,
            "burden_fraction": observed_seconds / evaluable_seconds,
        }
        for name, expected in expected_occurrence_values.items():
            if not math.isclose(
                float(occurrence[name]), expected, rel_tol=1e-9, abs_tol=_TOL
            ):
                raise ValueError(f"occurrence {name} is not replayable")

        onset_values = [
            float(row["onset_to_onset_seconds"]) for row in intervals
        ]
        peak_values = [
            float(row["peak_to_peak_seconds"]) for row in intervals
        ]
        onset_median, onset_mad = _median_mad(onset_values)
        peak_median, peak_mad = _median_mad(peak_values)
        duration_median, duration_mad = _median_mad(element_durations)
        onset_mean = float(np.mean(onset_values))
        expected_interval_values = {
            "onset_to_onset_median_seconds": onset_median,
            "onset_to_onset_mad_seconds": onset_mad,
            "onset_to_onset_robust_cv": 1.4826 * onset_mad / onset_median,
            "onset_to_onset_cv": float(
                np.std(onset_values, ddof=0) / onset_mean
            ),
            "peak_to_peak_median_seconds": peak_median,
            "peak_to_peak_mad_seconds": peak_mad,
        }
        for name, expected in expected_interval_values.items():
            if not math.isclose(
                float(interval_profile[name]),
                expected,
                rel_tol=1e-9,
                abs_tol=_TOL,
            ):
                raise ValueError(f"interval summary {name} is not replayable")
        expected_duration_values = {
            "element_duration_median_seconds": duration_median,
            "element_duration_mad_seconds": duration_mad,
            "element_duration_min_seconds": min(element_durations),
            "element_duration_max_seconds": max(element_durations),
        }
        for name, expected in expected_duration_values.items():
            if not math.isclose(
                float(durations[name]), expected, rel_tol=1e-9, abs_tol=_TOL
            ):
                raise ValueError(f"duration summary {name} is not replayable")
        diagnostics = candidate["segmentation"]["diagnostics"]
        if (
            not isinstance(diagnostics, Mapping)
            or int(float(diagnostics["accepted_segment_count"])) != len(elements)
            or float(diagnostics["deduplicated_segment_count"])
            < float(diagnostics["accepted_segment_count"])
            or float(diagnostics["raw_excursion_count"])
            < float(diagnostics["deduplicated_segment_count"])
        ):
            raise ValueError("segmentation diagnostics do not replay the roster")
        expected_roster_sha = _canonical_sha256(
            {
                "domain": _ELEMENT_ROSTER_DIGEST_DOMAIN,
                "event_id": candidate["event_id"],
                "analysis_unit_id": candidate["analysis_unit_id"],
                "elements": elements,
            }
        )
        expected_interval_sha = _canonical_sha256(
            {
                "domain": _INTERVAL_LEDGER_DIGEST_DOMAIN,
                "event_id": candidate["event_id"],
                "analysis_unit_id": candidate["analysis_unit_id"],
                "successive_intervals": intervals,
            }
        )
        if occurrence["element_roster_sha256"] != expected_roster_sha or interval_profile["interval_ledger_sha256"] != expected_interval_sha:
            raise ValueError("element/interval content digest drifted")
        dependency_rows = candidate["raw_sample_dependencies"]
        if len(dependency_rows) != len(elements) + 1:
            raise ValueError("raw dependency ledger does not close over the profile")
        dependency_ids: set[str] = set()
        element_by_id = {str(row["element_id"]): row for row in elements}
        analysis_dependencies = 0
        analysis_dependency_id: str | None = None
        for index, row in enumerate(dependency_rows):
            dependency = row["raw_sample_dependency"]
            if row["dependency_role"] == "analysis_scope":
                analysis_dependencies += 1
                if row["element_id"] is not None:
                    raise ValueError("analysis raw dependency cannot bind an element")
                evidence_interval = source_interval
                dependency_tensor = tensor_interval
                analysis_dependency_id = str(dependency["dependency_id"])
            elif row["dependency_role"] == "element_waveform":
                element_id = str(row["element_id"])
                if element_id not in element_by_id:
                    raise ValueError("raw dependency references an unknown element")
                element = element_by_id[element_id]
                evidence_interval = _finite_interval(
                    element["recording_interval"], "element dependency interval"
                )
                dependency_tensor = tuple(
                    int(item) for item in element["tensor_sample_interval"]
                )
                if element["raw_sample_dependency_id"] != dependency["dependency_id"]:
                    raise ValueError("element raw dependency ID drifted")
            else:
                raise ValueError("unsupported raw dependency role")
            dependency_id = _validate_raw_sample_dependency(
                dependency,
                context=f"raw_sample_dependencies[{index}]",
                canonical_signal_sha256=str(source["canonical_signal_sha256"]),
                source_view_id=str(source["source_view_id"]),
                view_role=str(source["source_view_role"]),
                evidence_interval=evidence_interval,
                view_tensor_interval=dependency_tensor,
                view_receipt_id=str(source["source_view_receipt_id"]),
                view_receipt_sha256=str(source["source_view_receipt_sha256"]),
                transform_spec_sha256=str(source["transform_spec_sha256"]),
            )
            if dependency_id in dependency_ids:
                raise ValueError("raw dependency ledger contains duplicates")
            dependency_ids.add(dependency_id)
        if (
            analysis_dependencies != 1
            or candidate["analysis_raw_sample_dependency_id"]
            != analysis_dependency_id
            or analysis_dependency_id not in dependency_ids
        ):
            raise ValueError("analysis raw dependency is missing")
    else:
        raise ValueError("periodicity candidate qualification status is unsupported")

    identifier_source = deepcopy(candidate)
    identifier_source.pop("candidate_id", None)
    identifier_source.pop("candidate_sha256", None)
    expected_id = "ELEMINT-" + _canonical_sha256(
        {"domain": _CANDIDATE_ID_DOMAIN, "candidate": identifier_source}
    )[:24]
    if candidate.get("candidate_id") != expected_id:
        raise ValueError("periodicity candidate_id does not bind its content")
    digest_source = deepcopy(candidate)
    digest_source["candidate_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_digest = _canonical_sha256(
        {"domain": _CANDIDATE_DIGEST_DOMAIN, "candidate": digest_source}
    )
    if candidate.get("candidate_sha256") != expected_digest:
        raise ValueError("periodicity candidate_sha256 does not bind its content")
    return candidate


__all__ = [
    "DEFAULT_DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY",
    "DETERMINISTIC_PERIODICITY_CANDIDATE_METHOD_ID",
    "DETERMINISTIC_PERIODICITY_CANDIDATE_POLICY_ID",
    "DETERMINISTIC_PERIODICITY_CANDIDATE_SCHEMA_VERSION",
    "DETERMINISTIC_PERIODICITY_CANDIDATE_TERM_ID",
    "DETERMINISTIC_PERIODICITY_DEDUPLICATION_POLICY_ID",
    "DeterministicPeriodicityCandidatePolicy",
    "produce_deterministic_periodicity_candidate",
    "validate_deterministic_periodicity_candidate",
]
