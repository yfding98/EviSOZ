"""Target-blind common-17 support-policy comparator for event Findings.

This module changes only the EEG support made visible to the numerical
Findings kernel.  The adaptive arm and every fixed arm use the exact same
``_evaluate_support`` implementation from
``adaptive_native_evidence_common17``.  This makes the comparison a support
policy ablation rather than a feature-implementation ablation.

The primary materializer has no route for seizure references, channel labels,
doctor text, annotations, spreadsheets or LLM output.  A deliberately separate
post-freeze function can audit already materialized receipts against global
TERM seizure intervals.  Reference values can therefore never influence a
query, a stopping decision, or a feature value.

The budget-matched arm is a counterfactual measurement arm, not a deployable
fixed policy: after the adaptive arm has stopped, it exposes an anchor-centred
interval with exactly the same number of unique samples (translated at a
recording boundary when necessary).  It receives no adaptive side geometry or
feature values.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import median
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from .adaptive_native_evidence_common17 import (
    ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID,
    COMMON17_CHANNELS,
    DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY,
    AdaptiveNativeEvidencePolicy,
    NativeEEGQueryReader,
    _AcquiredChunk,
    _array_sha256,
    _evaluate_support,
    _normalize_query_result,
    materialize_common17_adaptive_native_event_evidence,
    validate_common17_adaptive_native_event_evidence,
)


COMMON17_SUPPORT_COMPARISON_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_common17_native_support_comparison_v1"
)
COMMON17_SUPPORT_COMPARISON_METHOD_ID: Final[str] = (
    "COMMON17-SAME-KERNEL-ADAPTIVE-VS-FIXED-SUPPORT-V1"
)
COMMON17_SUPPORT_COMPARISON_COHORT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_common17_native_support_comparison_cohort_v1"
)
COMMON17_SUPPORT_POSTFREEZE_AUDIT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_common17_native_support_postfreeze_reference_audit_v1"
)

ADAPTIVE_STRATEGY_ID: Final[str] = "adaptive_q0_lr_8_16_32"
LEGACY_STRATEGY_ID: Final[str] = "fixed_legacy_m12_p48"
SYMMETRIC60_STRATEGY_ID: Final[str] = "fixed_symmetric_m30_p30"
FIXED120_STRATEGY_ID: Final[str] = "fixed_symmetric_m60_p60_shadow"
BUDGET_MATCHED_STRATEGY_ID: Final[str] = "fixed_budget_matched_symmetric"
STRATEGY_ORDER: Final[tuple[str, ...]] = (
    ADAPTIVE_STRATEGY_ID,
    LEGACY_STRATEGY_ID,
    SYMMETRIC60_STRATEGY_ID,
    FIXED120_STRATEGY_ID,
    BUDGET_MATCHED_STRATEGY_ID,
)

_EEG_ONLY_SCOPE: Final[dict[str, object]] = {
    "EEG_samples_used": True,
    "acquisition_parameters_used": True,
    "EEG_derived_QC_used_if_supplied": True,
    "navigation_anchor_used": True,
    "global_seizure_reference_used": False,
    "channel_or_SOZ_target_used": False,
    "EDF_annotation_API_used": False,
    "spreadsheet_or_doctor_text_used": False,
    "clinical_text_used": False,
    "video_behaviour_sleep_activation_used": False,
    "LLM_output_used": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    if len(value) > 200 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{name} is not a safe identifier")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class Common17SupportComparisonPolicyV1:
    """Frozen geometries for the same-kernel event-support ablation."""

    legacy_before_seconds: float = 12.0
    legacy_after_seconds: float = 48.0
    symmetric_60_each_side_seconds: float = 30.0
    fixed_120_each_side_seconds: float = 60.0
    budget_matched_geometry: str = (
        "anchor_centered_exact_adaptive_sample_budget_boundary_translated_v1"
    )
    high_budget_shadow_strategy_id: str = FIXED120_STRATEGY_ID

    def __post_init__(self) -> None:
        if (
            float(self.legacy_before_seconds) != 12.0
            or float(self.legacy_after_seconds) != 48.0
            or float(self.symmetric_60_each_side_seconds) != 30.0
            or float(self.fixed_120_each_side_seconds) != 60.0
        ):
            raise ValueError("v1 freezes [-12,+48], [-30,+30], and [-60,+60]")
        if self.budget_matched_geometry != (
            "anchor_centered_exact_adaptive_sample_budget_boundary_translated_v1"
        ):
            raise ValueError("v1 budget-matched geometry drifted")
        if self.high_budget_shadow_strategy_id != FIXED120_STRATEGY_ID:
            raise ValueError("v1 high-budget shadow must remain fixed 120 s")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_COMMON17_SUPPORT_COMPARISON_POLICY_V1 = Common17SupportComparisonPolicyV1()


NativeEEGQueryReaderFactory = Callable[[str], NativeEEGQueryReader]


def _clipped_relative_interval(
    *,
    anchor_sample: int,
    recording_sample_count: int,
    rate: float,
    before_seconds: float,
    after_seconds: float,
) -> tuple[int, int]:
    start = max(0, anchor_sample - int(round(before_seconds * rate)))
    stop = min(
        recording_sample_count,
        anchor_sample + int(round(after_seconds * rate)),
    )
    if stop <= start:
        raise ValueError("fixed support is empty after recording-boundary clipping")
    return start, stop


def _budget_matched_interval(
    *,
    anchor_sample: int,
    recording_sample_count: int,
    sample_budget: int,
) -> tuple[int, int]:
    if (
        isinstance(sample_budget, bool)
        or not isinstance(sample_budget, int)
        or not 1 <= sample_budget <= recording_sample_count
    ):
        raise ValueError("adaptive sample budget is invalid")
    start = anchor_sample - sample_budget // 2
    stop = start + sample_budget
    if start < 0:
        stop -= start
        start = 0
    if stop > recording_sample_count:
        start -= stop - recording_sample_count
        stop = recording_sample_count
    start = max(0, start)
    if stop - start != sample_budget or not start <= anchor_sample <= stop:
        raise RuntimeError("budget-matched interval does not preserve the sample ledger")
    return start, stop


def _status_from_snapshot(snapshot: object) -> str:
    onset = getattr(snapshot, "onset_sample")
    baseline = getattr(snapshot, "baseline_status")
    if onset is not None:
        return "qualified_scalp_change_candidate"
    if baseline != "qualified_robust_matched_baseline":
        return "unresolved_baseline_censored"
    return "unresolved_no_sustained_change"


def _fixed_arm(
    *,
    strategy_id: str,
    recording_id: str,
    anchor_sample: int,
    start_sample: int,
    stop_sample: int,
    rate: float,
    query_reader: NativeEEGQueryReader,
    adaptive_policy: AdaptiveNativeEvidencePolicy,
    geometry: Mapping[str, object],
) -> dict[str, Any]:
    result = query_reader(start_sample, stop_sample)
    signal, qc = _normalize_query_result(
        result,
        expected_samples=stop_sample - start_sample,
    )
    signal_sha = _array_sha256(
        signal.astype("<f8", copy=False), prefix="common17-volts"
    )
    qc_sha = _array_sha256(qc.astype(np.uint8), prefix="common17-eeg-qc")
    chunk = _AcquiredChunk(
        start_sample=start_sample,
        stop_sample=stop_sample,
        signal_volts=signal,
        valid_sample_mask=qc,
        signal_sha256=signal_sha,
        qc_sha256=qc_sha,
    )
    snapshot = _evaluate_support(
        [chunk],
        anchor_sample=anchor_sample,
        rate=rate,
        policy=adaptive_policy,
        morphology_cache={},
    )
    evidence = deepcopy(snapshot.serializable)
    return {
        "strategy_id": strategy_id,
        "strategy_family": "fixed_support",
        "support_selection_used_EEG_evidence": False,
        "geometry": deepcopy(dict(geometry)),
        "interval_samples": [start_sample, stop_sample],
        "interval_recording_seconds": [
            _round(start_sample / rate),
            _round(stop_sample / rate),
        ],
        "interval_relative_to_anchor_seconds": [
            _round((start_sample - anchor_sample) / rate),
            _round((stop_sample - anchor_sample) / rate),
        ],
        "unique_samples_per_channel": stop_sample - start_sample,
        "queried_physical_seconds_per_channel": _round(
            (stop_sample - start_sample) / rate
        ),
        "query_count": 1,
        "query_intervals_samples": [[start_sample, stop_sample]],
        "status": _status_from_snapshot(snapshot),
        "final_evidence": evidence,
        "source_bindings": {
            "recording_id": recording_id,
            "queried_EEG_sha256": signal_sha,
            "queried_EEG_QC_sha256": qc_sha,
        },
    }


def _adaptive_arm(receipt: Mapping[str, object]) -> dict[str, Any]:
    validated = validate_common17_adaptive_native_event_evidence(receipt)
    support = validated["final_variable_support"]
    return {
        "strategy_id": ADAPTIVE_STRATEGY_ID,
        "strategy_family": "adaptive_support",
        "support_selection_used_EEG_evidence": True,
        "geometry": {
            "q0_extent_seconds_each_side": 4.0,
            "independent_expansion_extents_seconds_each_side": [8.0, 16.0, 32.0],
            "stopping_rule": "EEG_native_evidence_side_specific_v1",
            "side_closure": deepcopy(support["side_closure"]),
        },
        "interval_samples": deepcopy(support["interval_samples"]),
        "interval_recording_seconds": deepcopy(
            support["interval_recording_seconds"]
        ),
        "interval_relative_to_anchor_seconds": deepcopy(
            support["interval_relative_to_anchor_seconds"]
        ),
        "unique_samples_per_channel": int(
            support["unique_physical_samples_per_channel"]
        ),
        "queried_physical_seconds_per_channel": _round(
            float(support["unique_physical_samples_per_channel"])
            / float(validated["acquisition"]["sampling_rate_hz"])
        ),
        "query_count": int(support["query_count"]),
        "query_intervals_samples": [
            deepcopy(row["action"]["interval_samples"])
            for row in validated["query_trace"]
        ],
        "status": validated["status"],
        "final_evidence": deepcopy(validated["final_evidence"]),
        "source_bindings": {
            "recording_id": validated["recording_id"],
            "adaptive_receipt_sha256": validated["receipt_sha256"],
            "final_acquired_EEG_sha256": validated["source_bindings"][
                "final_acquired_eeg_sha256"
            ],
            "final_EEG_QC_sha256": validated["source_bindings"][
                "final_eeg_qc_sha256"
            ],
        },
    }


def _spatial_vector(arm: Mapping[str, object]) -> np.ndarray | None:
    evidence = arm["final_evidence"]
    rows = evidence.get("per_channel_evidence", [])
    if not rows:
        return None
    by_channel = {str(row["channel"]): row for row in rows}
    if set(by_channel) != set(COMMON17_CHANNELS):
        return None
    values = np.asarray(
        [float(by_channel[channel]["onset_spatial_posterior_mass"]) for channel in COMMON17_CHANNELS],
        dtype=np.float64,
    )
    total = float(np.sum(values))
    if not np.isfinite(values).all() or total <= 0.0:
        return None
    return values / total


def _js_divergence_base2(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return max(0.0, min(1.0, 0.5 * kl(left) + 0.5 * kl(right)))


def _top_channels(arm: Mapping[str, object], count: int) -> list[str]:
    rows = arm["final_evidence"].get("per_channel_evidence", [])
    return [str(row["channel"]) for row in rows[:count]]


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float | None:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else None


def _candidate_time(arm: Mapping[str, object]) -> float | None:
    candidate = arm["final_evidence"].get("onset_candidate")
    return None if candidate is None else float(candidate["recording_seconds"])


def _recovery_available(arm: Mapping[str, object]) -> bool:
    evolution = arm["final_evidence"].get("evolution")
    return bool(
        isinstance(evolution, dict)
        and evolution.get("candidate_return_to_baseline_recording_seconds") is not None
    )


def _earliest_channels(arm: Mapping[str, object]) -> list[str]:
    field = arm["final_evidence"].get("earliest_field")
    return [] if field is None else [str(value) for value in field.get("channels", [])]


def _pairwise_target_blind_metrics(
    arm: Mapping[str, object], shadow: Mapping[str, object]
) -> dict[str, object]:
    left_vector, right_vector = _spatial_vector(arm), _spatial_vector(shadow)
    divergence = (
        _js_divergence_base2(left_vector, right_vector)
        if left_vector is not None and right_vector is not None
        else None
    )
    left_time, right_time = _candidate_time(arm), _candidate_time(shadow)
    left_top1, right_top1 = _top_channels(arm, 1), _top_channels(shadow, 1)
    left_baseline = arm["final_evidence"]["robust_matched_baseline"]["status"]
    right_baseline = shadow["final_evidence"]["robust_matched_baseline"]["status"]
    left_reference = arm["final_evidence"]["reference_stability"]["status"]
    right_reference = shadow["final_evidence"]["reference_stability"]["status"]
    return {
        "comparison_strategy_id": arm["strategy_id"],
        "shadow_strategy_id": shadow["strategy_id"],
        "query_budget_ratio_to_shadow": _round(
            float(arm["unique_samples_per_channel"])
            / float(shadow["unique_samples_per_channel"])
        ),
        "baseline_status_agreement": left_baseline == right_baseline,
        "candidate_presence_agreement": (left_time is None) == (right_time is None),
        "onset_candidate_absolute_delta_seconds": (
            _round(abs(left_time - right_time))
            if left_time is not None and right_time is not None
            else None
        ),
        "spatial_JSD_base2": _round(divergence) if divergence is not None else None,
        "spatial_JS_similarity": _round(1.0 - divergence) if divergence is not None else None,
        "top1_channel_agreement": (
            left_top1 == right_top1 if left_top1 and right_top1 else None
        ),
        "top3_channel_jaccard": _jaccard(
            _top_channels(arm, 3), _top_channels(shadow, 3)
        ),
        "earliest_field_jaccard": _jaccard(
            _earliest_channels(arm), _earliest_channels(shadow)
        ),
        "reference_stability_status_agreement": left_reference == right_reference,
        "recovery_evaluability_agreement": (
            _recovery_available(arm) == _recovery_available(shadow)
        ),
    }


def _arm_target_blind_indicators(arm: Mapping[str, object]) -> dict[str, object]:
    evidence = arm["final_evidence"]
    baseline_status = evidence["robust_matched_baseline"]["status"]
    reference_status = evidence["reference_stability"]["status"]
    return {
        "queried_physical_seconds_per_channel": arm[
            "queried_physical_seconds_per_channel"
        ],
        "baseline_evaluable": baseline_status == "qualified_robust_matched_baseline",
        "baseline_status": baseline_status,
        "change_candidate_available": evidence.get("onset_candidate") is not None,
        "earliest_field_evaluable": evidence.get("earliest_field") is not None,
        "reference_view_evaluable": reference_status in {"stable", "reference_sensitive"},
        "reference_view_stable": reference_status == "stable",
        "reference_stability_status": reference_status,
        "evolution_trajectory_evaluable": evidence.get("evolution") is not None,
        "recovery_evaluable": _recovery_available(arm),
    }


def materialize_common17_support_policy_comparison_v1(
    *,
    event_id: str,
    recording_id: str,
    navigation_anchor_recording_seconds: float,
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader_factory: NativeEEGQueryReaderFactory,
    channel_order: Sequence[str] = COMMON17_CHANNELS,
    adaptive_policy: AdaptiveNativeEvidencePolicy = (
        DEFAULT_ADAPTIVE_NATIVE_EVIDENCE_POLICY
    ),
    comparison_policy: Common17SupportComparisonPolicyV1 = (
        DEFAULT_COMMON17_SUPPORT_COMPARISON_POLICY_V1
    ),
) -> dict[str, Any]:
    """Materialize five same-kernel support arms without target access."""

    event = _identifier(event_id, "event_id")
    recording = _identifier(recording_id, "recording_id")
    rate = _finite(sampling_rate_hz, "sampling_rate_hz", minimum=10.0)
    if tuple(channel_order) != COMMON17_CHANNELS:
        raise ValueError("support comparison requires exact directly observed common-17")
    if (
        isinstance(recording_sample_count, bool)
        or not isinstance(recording_sample_count, int)
        or recording_sample_count < int(round(2.0 * rate))
    ):
        raise ValueError("recording_sample_count is invalid")
    if not callable(query_reader_factory):
        raise TypeError("query_reader_factory must be callable")
    anchor_seconds = _finite(
        navigation_anchor_recording_seconds,
        "navigation_anchor_recording_seconds",
        minimum=0.0,
    )
    anchor_sample = int(round(anchor_seconds * rate))
    if not 0 <= anchor_sample <= recording_sample_count:
        raise ValueError("navigation anchor lies outside the recording")

    readers: dict[str, NativeEEGQueryReader] = {}
    arms: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        def reader_for(strategy_id: str) -> NativeEEGQueryReader:
            candidate = query_reader_factory(strategy_id)
            if hasattr(candidate, "__enter__") and hasattr(candidate, "__exit__"):
                candidate = stack.enter_context(candidate)  # type: ignore[arg-type]
            if not callable(candidate):
                raise TypeError("query_reader_factory returned a non-callable reader")
            readers[strategy_id] = candidate
            return candidate

        adaptive_receipt = materialize_common17_adaptive_native_event_evidence(
            event_id=event,
            recording_id=recording,
            navigation_anchor_recording_seconds=anchor_seconds,
            sampling_rate_hz=rate,
            recording_sample_count=recording_sample_count,
            query_reader=reader_for(ADAPTIVE_STRATEGY_ID),
            channel_order=channel_order,
            policy=adaptive_policy,
        )
        arms[ADAPTIVE_STRATEGY_ID] = _adaptive_arm(adaptive_receipt)

        fixed_specs = (
            (
                LEGACY_STRATEGY_ID,
                comparison_policy.legacy_before_seconds,
                comparison_policy.legacy_after_seconds,
                {
                    "nominal_relative_seconds": [-12.0, 48.0],
                    "boundary_policy": "clip_without_budget_reallocation",
                },
            ),
            (
                SYMMETRIC60_STRATEGY_ID,
                comparison_policy.symmetric_60_each_side_seconds,
                comparison_policy.symmetric_60_each_side_seconds,
                {
                    "nominal_relative_seconds": [-30.0, 30.0],
                    "boundary_policy": "clip_without_budget_reallocation",
                },
            ),
            (
                FIXED120_STRATEGY_ID,
                comparison_policy.fixed_120_each_side_seconds,
                comparison_policy.fixed_120_each_side_seconds,
                {
                    "nominal_relative_seconds": [-60.0, 60.0],
                    "boundary_policy": "clip_without_budget_reallocation",
                    "role": "high_budget_shadow_not_ground_truth",
                },
            ),
        )
        for strategy_id, before, after, geometry in fixed_specs:
            start, stop = _clipped_relative_interval(
                anchor_sample=anchor_sample,
                recording_sample_count=recording_sample_count,
                rate=rate,
                before_seconds=before,
                after_seconds=after,
            )
            arms[strategy_id] = _fixed_arm(
                strategy_id=strategy_id,
                recording_id=recording,
                anchor_sample=anchor_sample,
                start_sample=start,
                stop_sample=stop,
                rate=rate,
                query_reader=reader_for(strategy_id),
                adaptive_policy=adaptive_policy,
                geometry=geometry,
            )

        adaptive_budget = int(
            arms[ADAPTIVE_STRATEGY_ID]["unique_samples_per_channel"]
        )
        start, stop = _budget_matched_interval(
            anchor_sample=anchor_sample,
            recording_sample_count=recording_sample_count,
            sample_budget=adaptive_budget,
        )
        arms[BUDGET_MATCHED_STRATEGY_ID] = _fixed_arm(
            strategy_id=BUDGET_MATCHED_STRATEGY_ID,
            recording_id=recording,
            anchor_sample=anchor_sample,
            start_sample=start,
            stop_sample=stop,
            rate=rate,
            query_reader=reader_for(BUDGET_MATCHED_STRATEGY_ID),
            adaptive_policy=adaptive_policy,
            geometry={
                "sample_budget_source": "adaptive_unique_sample_count_only",
                "adaptive_side_geometry_copied": False,
                "nominal_geometry": "anchor_centered_symmetric",
                "boundary_policy": "translate_to_preserve_exact_sample_budget",
            },
        )

    ordered_arms = [arms[strategy_id] for strategy_id in STRATEGY_ORDER]
    shadow = arms[FIXED120_STRATEGY_ID]
    body: dict[str, Any] = {
        "schema_version": COMMON17_SUPPORT_COMPARISON_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": COMMON17_SUPPORT_COMPARISON_METHOD_ID,
        "event_id": event,
        "recording_id": recording,
        "navigation_anchor_recording_seconds": _round(anchor_sample / rate),
        "acquisition": {
            "sampling_rate_hz": rate,
            "recording_sample_count": recording_sample_count,
            "recording_duration_seconds": _round(recording_sample_count / rate),
            "channel_order": list(COMMON17_CHANNELS),
            "signal_unit": "V",
            "removed_channels": ["FZ", "PZ"],
            "missing_channel_imputation_used": False,
        },
        "kernel_binding": {
            "adaptive_method_id": ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID,
            "adaptive_policy": adaptive_policy.to_dict(),
            "adaptive_policy_sha256": adaptive_policy.sha256,
            "same_evaluate_support_kernel_for_all_arms": True,
            "only_support_policy_varies": True,
        },
        "comparison_policy": comparison_policy.to_dict(),
        "comparison_policy_sha256": comparison_policy.sha256,
        "strategy_order": list(STRATEGY_ORDER),
        "arms": ordered_arms,
        "target_blind_indicators": {
            arm["strategy_id"]: _arm_target_blind_indicators(arm)
            for arm in ordered_arms
        },
        "high_budget_shadow_comparisons": {
            arm["strategy_id"]: _pairwise_target_blind_metrics(arm, shadow)
            for arm in ordered_arms
            if arm["strategy_id"] != FIXED120_STRATEGY_ID
        },
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "claim_limits": {
            "high_budget_shadow_is_ground_truth": False,
            "adaptive_superiority_claim_authorized_from_single_event": False,
            "clinical_onset_or_SOZ_claim_authorized": False,
            "postfreeze_reference_metric_embedded": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return validate_common17_support_policy_comparison_v1(body)


def validate_common17_support_policy_comparison_v1(payload: object) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("support comparison receipt must be an object")
    required = {
        "schema_version",
        "receipt_sha256",
        "method_id",
        "event_id",
        "recording_id",
        "navigation_anchor_recording_seconds",
        "acquisition",
        "kernel_binding",
        "comparison_policy",
        "comparison_policy_sha256",
        "strategy_order",
        "arms",
        "target_blind_indicators",
        "high_budget_shadow_comparisons",
        "scope_receipt",
        "claim_limits",
    }
    if set(payload) != required:
        raise ValueError("support comparison receipt fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != COMMON17_SUPPORT_COMPARISON_SCHEMA_VERSION:
        raise ValueError("support comparison schema drifted")
    if data["method_id"] != COMMON17_SUPPORT_COMPARISON_METHOD_ID:
        raise ValueError("support comparison method drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    acquisition = data["acquisition"]
    if acquisition.get("channel_order") != list(COMMON17_CHANNELS):
        raise ValueError("support comparison is not exact common-17")
    if acquisition.get("removed_channels") != ["FZ", "PZ"] or acquisition.get(
        "missing_channel_imputation_used"
    ) is not False:
        raise ValueError("support comparison imputed removed midline channels")
    if data["comparison_policy_sha256"] != _canonical_sha256(
        data["comparison_policy"]
    ):
        raise ValueError("support comparison policy hash mismatch")
    if data["strategy_order"] != list(STRATEGY_ORDER):
        raise ValueError("support comparison strategy order drifted")
    arms = data["arms"]
    if not isinstance(arms, list) or [row.get("strategy_id") for row in arms] != list(
        STRATEGY_ORDER
    ):
        raise ValueError("support comparison arms drifted")
    arm_map = {row["strategy_id"]: row for row in arms}
    if len(arm_map) != len(STRATEGY_ORDER):
        raise ValueError("support comparison arms are duplicated")
    samples = int(acquisition["recording_sample_count"])
    for arm in arms:
        interval = arm.get("interval_samples")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in interval)
            or not 0 <= interval[0] < interval[1] <= samples
        ):
            raise ValueError("support arm interval is invalid")
        if arm.get("unique_samples_per_channel") != interval[1] - interval[0]:
            raise ValueError("support arm sample ledger does not close")
        query_intervals = arm.get("query_intervals_samples")
        if not isinstance(query_intervals, list) or len(query_intervals) != arm.get(
            "query_count"
        ):
            raise ValueError("support arm query ledger is invalid")
        if arm["strategy_id"] == ADAPTIVE_STRATEGY_ID:
            if arm.get("strategy_family") != "adaptive_support" or arm.get(
                "support_selection_used_EEG_evidence"
            ) is not True:
                raise ValueError("adaptive support semantics drifted")
        elif arm.get("strategy_family") != "fixed_support" or arm.get(
            "support_selection_used_EEG_evidence"
        ) is not False:
            raise ValueError("fixed support semantics drifted")
        evidence = arm.get("final_evidence")
        if not isinstance(evidence, dict) or "robust_matched_baseline" not in evidence:
            raise ValueError("support arm lacks common numerical evidence")
    if arm_map[BUDGET_MATCHED_STRATEGY_ID]["unique_samples_per_channel"] != arm_map[
        ADAPTIVE_STRATEGY_ID
    ]["unique_samples_per_channel"]:
        raise ValueError("budget-matched arm does not match adaptive samples")
    if set(data["target_blind_indicators"]) != set(STRATEGY_ORDER):
        raise ValueError("target-blind indicators do not cover every strategy")
    if set(data["high_budget_shadow_comparisons"]) != (
        set(STRATEGY_ORDER) - {FIXED120_STRATEGY_ID}
    ):
        raise ValueError("shadow comparisons do not cover non-shadow strategies")
    if data["scope_receipt"] != _EEG_ONLY_SCOPE:
        raise ValueError("support comparison violated the target-blind firewall")
    if data["claim_limits"].get("postfreeze_reference_metric_embedded") is not False:
        raise ValueError("post-freeze reference values leaked into extraction")
    expected = _canonical_sha256(
        {key: value for key, value in data.items() if key != "receipt_sha256"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("support comparison content hash mismatch")
    return data


def _rate(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator) if denominator else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return _round(median(values)) if values else None


def summarize_common17_support_policy_comparison_cohort_v1(
    *,
    receipts: Sequence[Mapping[str, object]],
    group_id_by_event_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate only target-blind outcomes; group IDs are never extractor inputs."""

    if not receipts:
        raise ValueError("cannot summarize an empty support-comparison cohort")
    rows = [validate_common17_support_policy_comparison_v1(row) for row in receipts]
    event_ids = [str(row["event_id"]) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("support comparison event IDs are duplicated")
    groups: list[str] = []
    if group_id_by_event_id is not None:
        if set(group_id_by_event_id) != set(event_ids):
            raise ValueError("group mapping must exactly cover the frozen event cohort")
        groups = [_identifier(group_id_by_event_id[event], "group_id") for event in event_ids]

    strategy_summary: dict[str, object] = {}
    for strategy_id in STRATEGY_ORDER:
        indicators = [row["target_blind_indicators"][strategy_id] for row in rows]
        seconds = [float(item["queried_physical_seconds_per_channel"]) for item in indicators]
        strategy_summary[strategy_id] = {
            "event_count": len(indicators),
            "queried_seconds_per_channel": {
                "total": _round(sum(seconds)),
                "median": _median_or_none(seconds),
                "minimum": _round(min(seconds)),
                "maximum": _round(max(seconds)),
            },
            "baseline_evaluable_rate": _rate(
                sum(bool(item["baseline_evaluable"]) for item in indicators),
                len(indicators),
            ),
            "change_candidate_available_rate": _rate(
                sum(bool(item["change_candidate_available"]) for item in indicators),
                len(indicators),
            ),
            "earliest_field_evaluable_rate": _rate(
                sum(bool(item["earliest_field_evaluable"]) for item in indicators),
                len(indicators),
            ),
            "reference_view_evaluable_rate": _rate(
                sum(bool(item["reference_view_evaluable"]) for item in indicators),
                len(indicators),
            ),
            "reference_view_stable_rate_all_events": _rate(
                sum(bool(item["reference_view_stable"]) for item in indicators),
                len(indicators),
            ),
            "recovery_evaluable_rate": _rate(
                sum(bool(item["recovery_evaluable"]) for item in indicators),
                len(indicators),
            ),
        }

    comparison_summary: dict[str, object] = {}
    for strategy_id in STRATEGY_ORDER:
        if strategy_id == FIXED120_STRATEGY_ID:
            continue
        comparisons = [
            row["high_budget_shadow_comparisons"][strategy_id] for row in rows
        ]
        comparison_summary[strategy_id] = {
            "event_count": len(comparisons),
            "median_query_budget_ratio_to_shadow": _median_or_none(
                [float(item["query_budget_ratio_to_shadow"]) for item in comparisons]
            ),
            "candidate_presence_agreement_rate": _rate(
                sum(bool(item["candidate_presence_agreement"]) for item in comparisons),
                len(comparisons),
            ),
            "onset_delta_seconds_median_paired_evaluable": _median_or_none(
                [
                    float(item["onset_candidate_absolute_delta_seconds"])
                    for item in comparisons
                    if item["onset_candidate_absolute_delta_seconds"] is not None
                ]
            ),
            "spatial_JSD_base2_median_paired_evaluable": _median_or_none(
                [
                    float(item["spatial_JSD_base2"])
                    for item in comparisons
                    if item["spatial_JSD_base2"] is not None
                ]
            ),
            "top1_channel_agreement_rate_paired_evaluable": _rate(
                sum(item["top1_channel_agreement"] is True for item in comparisons),
                sum(item["top1_channel_agreement"] is not None for item in comparisons),
            ),
            "top3_channel_jaccard_median_paired_evaluable": _median_or_none(
                [
                    float(item["top3_channel_jaccard"])
                    for item in comparisons
                    if item["top3_channel_jaccard"] is not None
                ]
            ),
            "reference_stability_status_agreement_rate": _rate(
                sum(
                    bool(item["reference_stability_status_agreement"])
                    for item in comparisons
                ),
                len(comparisons),
            ),
            "recovery_evaluability_agreement_rate": _rate(
                sum(bool(item["recovery_evaluability_agreement"]) for item in comparisons),
                len(comparisons),
            ),
        }

    same_budget_rows = []
    exact_budget_matches = 0
    for row in rows:
        arms = {arm["strategy_id"]: arm for arm in row["arms"]}
        adaptive = arms[ADAPTIVE_STRATEGY_ID]
        fixed_budget = arms[BUDGET_MATCHED_STRATEGY_ID]
        exact_budget_matches += int(
            adaptive["unique_samples_per_channel"]
            == fixed_budget["unique_samples_per_channel"]
        )
        same_budget_rows.append(_pairwise_target_blind_metrics(adaptive, fixed_budget))
    same_budget_summary: dict[str, object] = {
        "adaptive_strategy_id": ADAPTIVE_STRATEGY_ID,
        "fixed_strategy_id": BUDGET_MATCHED_STRATEGY_ID,
        "event_count": len(same_budget_rows),
        "exact_sample_budget_match_rate": _rate(exact_budget_matches, len(rows)),
        "candidate_presence_agreement_rate": _rate(
            sum(bool(item["candidate_presence_agreement"]) for item in same_budget_rows),
            len(same_budget_rows),
        ),
        "onset_delta_seconds_median_paired_evaluable": _median_or_none(
            [
                float(item["onset_candidate_absolute_delta_seconds"])
                for item in same_budget_rows
                if item["onset_candidate_absolute_delta_seconds"] is not None
            ]
        ),
        "spatial_JSD_base2_median_paired_evaluable": _median_or_none(
            [
                float(item["spatial_JSD_base2"])
                for item in same_budget_rows
                if item["spatial_JSD_base2"] is not None
            ]
        ),
        "top1_channel_agreement_rate_paired_evaluable": _rate(
            sum(item["top1_channel_agreement"] is True for item in same_budget_rows),
            sum(item["top1_channel_agreement"] is not None for item in same_budget_rows),
        ),
        "top3_channel_jaccard_median_paired_evaluable": _median_or_none(
            [
                float(item["top3_channel_jaccard"])
                for item in same_budget_rows
                if item["top3_channel_jaccard"] is not None
            ]
        ),
        "earliest_field_jaccard_median_paired_evaluable": _median_or_none(
            [
                float(item["earliest_field_jaccard"])
                for item in same_budget_rows
                if item["earliest_field_jaccard"] is not None
            ]
        ),
        "baseline_status_agreement_rate": _rate(
            sum(bool(item["baseline_status_agreement"]) for item in same_budget_rows),
            len(same_budget_rows),
        ),
        "reference_stability_status_agreement_rate": _rate(
            sum(
                bool(item["reference_stability_status_agreement"])
                for item in same_budget_rows
            ),
            len(same_budget_rows),
        ),
        "recovery_evaluability_agreement_rate": _rate(
            sum(bool(item["recovery_evaluability_agreement"]) for item in same_budget_rows),
            len(same_budget_rows),
        ),
    }

    body: dict[str, Any] = {
        "schema_version": COMMON17_SUPPORT_COMPARISON_COHORT_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": COMMON17_SUPPORT_COMPARISON_METHOD_ID,
        "event_count": len(rows),
        "recording_count": len({str(row["recording_id"]) for row in rows}),
        "group_audit": {
            "group_mapping_supplied_to_summarizer_only": group_id_by_event_id is not None,
            "group_mapping_supplied_to_event_extractor": False,
            "unique_group_count": len(set(groups)) if groups else None,
            "one_event_per_group": len(groups) == len(set(groups)) if groups else None,
        },
        "strategy_summary": strategy_summary,
        "same_budget_adaptive_vs_fixed_summary": same_budget_summary,
        "high_budget_shadow_comparison_summary": comparison_summary,
        "event_receipt_sha256s": [row["receipt_sha256"] for row in rows],
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "interpretation": {
            "metrics_are_target_blind": True,
            "high_budget_shadow_is_ground_truth": False,
            "adaptive_superiority_requires_prespecified_cohort_statistics": True,
            "clinical_accuracy_measured": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


def evaluate_common17_support_policy_postfreeze_references_v1(
    *,
    frozen_receipts: Sequence[Mapping[str, object]],
    reference_intervals_by_event_id: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Audit frozen outputs against global intervals without rematerialization."""

    rows = [validate_common17_support_policy_comparison_v1(row) for row in frozen_receipts]
    event_ids = {str(row["event_id"]) for row in rows}
    if set(reference_intervals_by_event_id) != event_ids:
        raise ValueError("post-freeze references must exactly cover frozen receipts")
    per_strategy: dict[str, object] = {}
    for strategy_id in STRATEGY_ORDER:
        errors: list[float] = []
        onset_covered = 0
        full_interval_covered = 0
        hits = {1.0: 0, 3.0: 0, 5.0: 0, 10.0: 0}
        candidate_count = 0
        for receipt in rows:
            event_id = str(receipt["event_id"])
            reference = reference_intervals_by_event_id[event_id]
            if set(reference) != {"onset_seconds", "offset_seconds"}:
                raise ValueError("post-freeze reference fields drifted")
            onset = _finite(reference["onset_seconds"], "reference onset", minimum=0.0)
            offset = _finite(reference["offset_seconds"], "reference offset", minimum=0.0)
            if offset <= onset:
                raise ValueError("post-freeze reference interval is invalid")
            arm = next(
                item for item in receipt["arms"] if item["strategy_id"] == strategy_id
            )
            support_start, support_stop = map(float, arm["interval_recording_seconds"])
            onset_covered += int(support_start <= onset <= support_stop)
            full_interval_covered += int(
                support_start <= onset and offset <= support_stop
            )
            candidate = _candidate_time(arm)
            if candidate is not None:
                candidate_count += 1
                error = abs(candidate - onset)
                errors.append(error)
                for tolerance in hits:
                    hits[tolerance] += int(error <= tolerance)
        per_strategy[strategy_id] = {
            "event_denominator": len(rows),
            "reference_onset_support_coverage_rate": _rate(onset_covered, len(rows)),
            "reference_full_interval_support_coverage_rate": _rate(
                full_interval_covered, len(rows)
            ),
            "candidate_available_rate": _rate(candidate_count, len(rows)),
            "onset_absolute_error_seconds": {
                "median_candidate_only": _median_or_none(errors),
                "mean_candidate_only": _round(float(np.mean(errors))) if errors else None,
                "candidate_denominator": candidate_count,
            },
            "onset_hit_rate_reference_denominator": {
                f"{int(tolerance)}s": _rate(count, len(rows))
                for tolerance, count in hits.items()
            },
        }
    body: dict[str, Any] = {
        "schema_version": COMMON17_SUPPORT_POSTFREEZE_AUDIT_SCHEMA_VERSION,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        "method_id": COMMON17_SUPPORT_COMPARISON_METHOD_ID,
        "frozen_event_receipt_sha256s": [row["receipt_sha256"] for row in rows],
        "reference_event_count": len(rows),
        "per_strategy": per_strategy,
        "firewall_receipt": {
            "frozen_EEG_receipts_rematerialized": False,
            "reference_used_for_query_or_stopping": False,
            "reference_used_for_feature_measurement": False,
            "reference_used_only_after_receipt_hash_freeze": True,
            "channel_or_SOZ_target_used": False,
        },
    }
    body["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


__all__ = [
    "ADAPTIVE_STRATEGY_ID",
    "BUDGET_MATCHED_STRATEGY_ID",
    "COMMON17_SUPPORT_COMPARISON_COHORT_SCHEMA_VERSION",
    "COMMON17_SUPPORT_COMPARISON_METHOD_ID",
    "COMMON17_SUPPORT_COMPARISON_SCHEMA_VERSION",
    "COMMON17_SUPPORT_POSTFREEZE_AUDIT_SCHEMA_VERSION",
    "DEFAULT_COMMON17_SUPPORT_COMPARISON_POLICY_V1",
    "FIXED120_STRATEGY_ID",
    "LEGACY_STRATEGY_ID",
    "STRATEGY_ORDER",
    "SYMMETRIC60_STRATEGY_ID",
    "Common17SupportComparisonPolicyV1",
    "NativeEEGQueryReaderFactory",
    "evaluate_common17_support_policy_postfreeze_references_v1",
    "materialize_common17_support_policy_comparison_v1",
    "summarize_common17_support_policy_comparison_cohort_v1",
    "validate_common17_support_policy_comparison_v1",
]
