"""Typed adaptive-Findings to record-level common-17 SOZ evidence adapter.

This module closes one deliberately narrow interface:

``adaptive_native_evidence_common17`` event receipts
    -> typed, uncalibrated event support
    -> ``common17_record_soz_evidence_aggregation_v1``.

The adapter never turns an engineering softmax or a normalized native feature
score into a probability.  Event channel/state values enter the aggregator as
``uncalibrated_nonnegative_score`` and the record output is therefore a
``normalized_support_score`` unless a separate, patient-disjoint calibration
stage is explicitly added elsewhere.  This v1 adapter has no calibration
argument and binds no calibration receipt.

Only onset-safe EEG evidence is allowed to create channel support.  Native
change evidence is gated by its first-change time; channels whose first change
falls outside the frozen early-onset horizon cannot gain support from a later
peak.  The spatial mass, earliest connected field, early channel delays and
cross-reference views are all measured by the upstream EEG-only receipt.

The independent pattern-state axis retains localized, bilateral
near-synchronous, generalized synchronous and unresolved support.  A
nonlocalized state is never mapped to CZ.  FZ/PZ are absent on the signal and
prediction sides, and CZ is eligible only when the observed CZ signal itself
contributes onset-safe evidence.

Mode handling is intentionally conservative.  Until a target-free,
patient-disjoint clustering model is trained and frozen, every event receives
its own explicitly named onset-safe shadow mode.  Late involvement and course
features do not enter that identifier, so the adapter does not fabricate a
stable seizure type.

The runner accepts detector navigation anchors, common-17 EEG/QC through the
upstream query reader, acquisition parameters and content hashes.  It exposes
no annotation, spreadsheet, doctor-text, label, video, behaviour, sleep,
provocation, ECG/EMG/EOG or LLM input route.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .adaptive_native_evidence_common17 import (
    ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID,
    ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION,
    COMMON17_CHANNELS,
    AdaptiveNativeEvidencePolicy,
    NativeEEGQueryReader,
    materialize_common17_adaptive_native_event_evidence,
    validate_common17_adaptive_native_event_evidence,
)
from .common17_record_soz_evidence_aggregation_v1 import (
    COMMON17_CHANNEL_IDS,
    COMMON17_CHANNEL_TO_LATERALITY,
    COMMON17_CHANNEL_TO_REGION,
    INDEPENDENT_PATTERN_STATE_IDS,
    NORMALIZED_SUPPORT_SCORE,
    NOT_APPLICABLE_NONLOCALIZED,
    UNCALIBRATED_NONNEGATIVE_SCORE,
    Common17CompleteEventRosterV1,
    Common17EventQCProfileV1,
    Common17EventSOZEvidenceV1,
    Common17RecordSOZAggregationPolicyV1,
    aggregate_common17_record_soz_evidence_v1,
    validate_common17_record_soz_evidence_aggregation_v1,
)


COMMON17_ADAPTIVE_RECORD_ADAPTER_SCHEMA_VERSION = (
    "clinical_eeg_common17_adaptive_event_to_record_soz_adapter_v1"
)
COMMON17_ADAPTIVE_RECORD_ADAPTER_METHOD_ID = (
    "common17_onset_safe_native_support_unknown_calibration_adapter_v1"
)
COMMON17_ADAPTIVE_RECORD_RUN_SCHEMA_VERSION = (
    "clinical_eeg_common17_adaptive_findings_record_soz_run_v1"
)
COMMON17_ADAPTIVE_RECORD_RUN_METHOD_ID = (
    "complete_detector_roster_adaptive_native_to_record_support_v1"
)
COMMON17_RECORD_SOZ_EVALUATION_PREDICTION_SCHEMA_VERSION = (
    "clinical_eeg_common17_record_soz_label_free_prediction_v1"
)

if tuple(COMMON17_CHANNELS) != tuple(COMMON17_CHANNEL_IDS):
    raise RuntimeError("adaptive evidence and record aggregation common-17 drifted")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 2e-6

_CHANNEL_COMPONENT_IDS = (
    "onset_spatial_support",
    "early_native_change_support",
    "early_evolution_priority_support",
    "earliest_connected_field_support",
    "cross_reference_consensus_support",
)

_EEG_ONLY_SCOPE = {
    "eeg_samples_used": True,
    "acquisition_parameters_used": True,
    "eeg_derived_qc_used_if_supplied": True,
    "detector_navigation_anchors_used": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behaviour_used": False,
    "sleep_or_activation_labels_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _round(value: float) -> float:
    return round(float(value), 12)


def _normalize(values: Sequence[float]) -> tuple[float, ...] | None:
    rows: list[float] = []
    for index, raw in enumerate(values):
        value = _finite(raw, f"support[{index}]")
        if value < 0.0:
            raise ValueError("support values must be nonnegative")
        rows.append(value)
    total = math.fsum(rows)
    if total <= 0.0:
        return None
    return tuple(value / total for value in rows)


def _support_mapping(
    identifiers: Sequence[str], values: Sequence[float] | None
) -> dict[str, float] | None:
    if values is None:
        return None
    return {
        identifier: _round(value)
        for identifier, value in zip(identifiers, values)
    }


def _distribution_from_mapping(
    value: object,
    identifiers: Sequence[str],
    *,
    name: str,
    allow_none: bool,
) -> tuple[float, ...] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Mapping) or set(value) != set(identifiers):
        raise ValueError(f"{name} must exactly cover its frozen ontology")
    normalized = _normalize([value[item] for item in identifiers])
    if normalized is None:
        raise ValueError(f"{name} must contain positive mass")
    return normalized


@dataclass(frozen=True)
class Common17AdaptiveEventAdapterPolicyV1:
    """Frozen EEG-only support composition; values are not clinical norms."""

    onset_spatial_weight: float = 0.36
    early_native_change_weight: float = 0.24
    early_evolution_priority_weight: float = 0.18
    earliest_connected_field_weight: float = 0.12
    cross_reference_consensus_weight: float = 0.10
    onset_safe_horizon_seconds: float = 2.0
    early_delay_decay_seconds: float = 0.75
    generalized_localized_axis_gate: float = 0.08
    neutral_extractor_multiplier: float = 1.0

    def __post_init__(self) -> None:
        weights = self.channel_component_weights
        if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("channel component weights must sum to one")
        if any(_finite(value, name) < 0.0 for name, value in weights.items()):
            raise ValueError("channel component weights must be nonnegative")
        if _finite(
            self.onset_safe_horizon_seconds, "onset_safe_horizon_seconds"
        ) <= 0.0:
            raise ValueError("onset_safe_horizon_seconds must be positive")
        if _finite(
            self.early_delay_decay_seconds, "early_delay_decay_seconds"
        ) <= 0.0:
            raise ValueError("early_delay_decay_seconds must be positive")
        _unit_interval(
            self.generalized_localized_axis_gate,
            "generalized_localized_axis_gate",
        )
        _unit_interval(
            self.neutral_extractor_multiplier,
            "neutral_extractor_multiplier",
        )
        if self.neutral_extractor_multiplier != 1.0:
            raise ValueError("v1 freezes the neutral extractor multiplier to one")

    @property
    def channel_component_weights(self) -> dict[str, float]:
        return {
            "onset_spatial_support": self.onset_spatial_weight,
            "early_native_change_support": self.early_native_change_weight,
            "early_evolution_priority_support": (
                self.early_evolution_priority_weight
            ),
            "earliest_connected_field_support": (
                self.earliest_connected_field_weight
            ),
            "cross_reference_consensus_support": (
                self.cross_reference_consensus_weight
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "channel_component_weights": self.channel_component_weights,
            "channel_contract": "STANDARD_19_minus_FZ_PZ_no_imputation_v1",
            "mode_strategy": (
                "one_event_one_onset_safe_shadow_until_target_free_clustering_is_frozen"
            ),
            "late_involvement_used_for_channel_support": False,
            "late_involvement_or_course_used_for_mode": False,
            "default_event_calibration_binding": None,
            "event_value_semantics": UNCALIBRATED_NONNEGATIVE_SCORE,
            "record_display_semantics": NORMALIZED_SUPPORT_SCORE,
        }

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_COMMON17_ADAPTIVE_EVENT_ADAPTER_POLICY_V1 = (
    Common17AdaptiveEventAdapterPolicyV1()
)


@dataclass(frozen=True)
class Common17DetectedNavigationEventV1:
    """One detector output used only as an adaptive EEG navigation anchor."""

    event_id: str
    anchor_recording_seconds: float

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        anchor = _finite(
            self.anchor_recording_seconds, "anchor_recording_seconds"
        )
        if anchor < 0.0:
            raise ValueError("anchor_recording_seconds must be nonnegative")
        object.__setattr__(self, "anchor_recording_seconds", anchor)


def _extract_channel_components(
    receipt: Mapping[str, Any],
    policy: Common17AdaptiveEventAdapterPolicyV1,
) -> tuple[
    dict[str, tuple[float, ...] | None],
    tuple[float, ...] | None,
    dict[str, Any],
]:
    final = receipt["final_evidence"]
    rows = final.get("per_channel_evidence", [])
    if receipt["status"] != "qualified_scalp_change_candidate" or not rows:
        return (
            {name: None for name in _CHANNEL_COMPONENT_IDS},
            None,
            {
                "earliest_change_recording_seconds": None,
                "onset_safe_channel_ids": [],
                "late_change_channel_ids_excluded": [],
                "late_spread_created_onset_support": False,
            },
        )
    by_channel = {str(row["channel"]): row for row in rows}
    if set(by_channel) != set(COMMON17_CHANNEL_IDS):
        raise ValueError("adaptive per-channel evidence is not exact common-17")

    earliest_observed = [
        float(row["earliest_change_recording_seconds"])
        for row in rows
        if row.get("evaluable") is True
        and row.get("earliest_change_recording_seconds") is not None
    ]
    earliest = min(earliest_observed) if earliest_observed else None
    onset_safe: set[str] = set()
    excluded_late: set[str] = set()
    for channel in COMMON17_CHANNEL_IDS:
        row = by_channel[channel]
        when = row.get("earliest_change_recording_seconds")
        if row.get("evaluable") is not True or when is None or earliest is None:
            continue
        delay = float(when) - earliest
        if -_TOL <= delay <= policy.onset_safe_horizon_seconds + _TOL:
            onset_safe.add(channel)
        elif delay > policy.onset_safe_horizon_seconds:
            excluded_late.add(channel)

    spatial_raw = []
    change_raw = []
    evolution_raw = []
    for channel in COMMON17_CHANNEL_IDS:
        row = by_channel[channel]
        safe = channel in onset_safe
        spatial_raw.append(
            max(0.0, float(row["onset_spatial_posterior_mass"])) if safe else 0.0
        )
        change = max(0.0, float(row["peak_change_score"])) if safe else 0.0
        change_raw.append(math.log1p(change))
        when = row.get("earliest_change_recording_seconds")
        delay = (
            0.0
            if earliest is None or when is None
            else max(0.0, float(when) - earliest)
        )
        evolution_raw.append(
            math.exp(-delay / policy.early_delay_decay_seconds)
            * math.log1p(change)
            if safe
            else 0.0
        )

    earliest_field = final.get("earliest_field")
    field_raw = [0.0] * len(COMMON17_CHANNEL_IDS)
    if isinstance(earliest_field, Mapping):
        field_channels = set(earliest_field.get("channels", [])) & onset_safe
        dominant = set(earliest_field.get("dominant_connected_component", []))
        for channel in field_channels:
            field_raw[COMMON17_CHANNEL_IDS.index(channel)] = (
                1.0 if channel in dominant else 0.5
            )

    reference = final.get("reference_stability")
    reference_raw = [0.0] * len(COMMON17_CHANNEL_IDS)
    reference_view_count = 0
    if isinstance(reference, Mapping):
        views = reference.get("view_channel_mass")
        if isinstance(views, Mapping):
            for view in views.values():
                if not isinstance(view, Mapping) or set(view) != set(
                    COMMON17_CHANNEL_IDS
                ):
                    continue
                reference_view_count += 1
                for index, channel in enumerate(COMMON17_CHANNEL_IDS):
                    if channel in onset_safe:
                        reference_raw[index] += max(0.0, float(view[channel]))
    if reference_view_count:
        reference_raw = [value / reference_view_count for value in reference_raw]

    components = {
        "onset_spatial_support": _normalize(spatial_raw),
        "early_native_change_support": _normalize(change_raw),
        "early_evolution_priority_support": _normalize(evolution_raw),
        "earliest_connected_field_support": _normalize(field_raw),
        "cross_reference_consensus_support": _normalize(reference_raw),
    }
    available_weights = {
        name: policy.channel_component_weights[name]
        for name, values in components.items()
        if values is not None and policy.channel_component_weights[name] > 0.0
    }
    combined: tuple[float, ...] | None = None
    if available_weights:
        denominator = math.fsum(available_weights.values())
        raw_combined = [0.0] * len(COMMON17_CHANNEL_IDS)
        for name, weight in available_weights.items():
            values = components[name]
            assert values is not None
            for index, value in enumerate(values):
                raw_combined[index] += weight / denominator * value
        combined = _normalize(raw_combined)
    return (
        components,
        combined,
        {
            "earliest_change_recording_seconds": (
                None if earliest is None else _round(earliest)
            ),
            "onset_safe_channel_ids": [
                channel for channel in COMMON17_CHANNEL_IDS if channel in onset_safe
            ],
            "late_change_channel_ids_excluded": [
                channel
                for channel in COMMON17_CHANNEL_IDS
                if channel in excluded_late
            ],
            "late_spread_created_onset_support": False,
        },
    )


def _entropy(values: Sequence[float]) -> float:
    positive = [float(value) for value in values if value > 0.0]
    if len(positive) <= 1:
        return 0.0
    return min(
        1.0,
        -math.fsum(value * math.log(value) for value in positive)
        / math.log(len(COMMON17_CHANNEL_IDS)),
    )


def _derive_independent_state_support(
    receipt: Mapping[str, Any],
    channel_support: tuple[float, ...] | None,
    temporal_receipt: Mapping[str, Any],
    policy: Common17AdaptiveEventAdapterPolicyV1,
) -> tuple[tuple[float, ...], tuple[float, ...] | None, dict[str, Any]]:
    if receipt["status"] != "qualified_scalp_change_candidate" or channel_support is None:
        return (
            (0.0, 0.0, 0.0, 1.0),
            None,
            {
                "status": "unresolved_without_qualified_onset_safe_support",
                "localized_raw": 0.0,
                "bilateral_raw": 0.0,
                "generalized_raw": 0.0,
                "unresolved_raw": 1.0,
                "late_involvement_used": False,
            },
        )

    final = receipt["final_evidence"]
    earliest_field = final.get("earliest_field")
    field_channels = (
        []
        if not isinstance(earliest_field, Mapping)
        else [
            channel
            for channel in earliest_field.get("channels", [])
            if channel in COMMON17_CHANNEL_IDS
        ]
    )
    if not field_channels:
        return (
            (0.0, 0.0, 0.0, 1.0),
            None,
            {
                "status": "unresolved_without_earliest_field",
                "localized_raw": 0.0,
                "bilateral_raw": 0.0,
                "generalized_raw": 0.0,
                "unresolved_raw": 1.0,
                "late_involvement_used": False,
            },
        )

    left_mass = math.fsum(
        channel_support[index]
        for index, channel in enumerate(COMMON17_CHANNEL_IDS)
        if COMMON17_CHANNEL_TO_LATERALITY[channel] == "left"
    )
    right_mass = math.fsum(
        channel_support[index]
        for index, channel in enumerate(COMMON17_CHANNEL_IDS)
        if COMMON17_CHANNEL_TO_LATERALITY[channel] == "right"
    )
    midline_mass = channel_support[COMMON17_CHANNEL_IDS.index("CZ")]
    hemispheric_mass = left_mass + right_mass
    bilateral_balance = (
        0.0
        if hemispheric_mass <= 0.0
        else 2.0 * min(left_mass, right_mass) / hemispheric_mass
    )
    hemispheric_dominance = (
        0.0
        if hemispheric_mass <= 0.0
        else abs(left_mass - right_mass) / hemispheric_mass
    )

    field_lateralities = {
        COMMON17_CHANNEL_TO_LATERALITY[channel] for channel in field_channels
    }
    field_regions = {COMMON17_CHANNEL_TO_REGION[channel] for channel in field_channels}
    bilateral_field = "left" in field_lateralities and "right" in field_lateralities
    field_fraction = len(field_channels) / len(COMMON17_CHANNEL_IDS)
    region_fraction = len(field_regions) / len(set(COMMON17_CHANNEL_TO_REGION.values()))

    by_channel = {
        str(row["channel"]): row
        for row in final.get("per_channel_evidence", [])
        if isinstance(row, Mapping)
    }
    field_times = [
        float(by_channel[channel]["earliest_change_recording_seconds"])
        for channel in field_channels
        if channel in by_channel
        and by_channel[channel].get("earliest_change_recording_seconds") is not None
    ]
    field_span = max(field_times) - min(field_times) if len(field_times) >= 2 else 0.0
    synchrony = math.exp(
        -field_span / max(policy.early_delay_decay_seconds, 1e-12)
    )
    concentration_top3 = math.fsum(sorted(channel_support, reverse=True)[:3])
    entropy = _entropy(channel_support)
    connectivity = final.get("spatial_connectivity", {})
    connected_fraction = (
        float(connectivity.get("dominant_component_fraction", 0.0))
        if isinstance(connectivity, Mapping)
        else 0.0
    )
    connected_fraction = min(1.0, max(0.0, connected_fraction))

    generalized_raw = (
        bilateral_balance
        * synchrony
        * region_fraction
        * math.sqrt(field_fraction)
        * entropy
        if bilateral_field
        else 0.0
    )
    bilateral_raw = (
        bilateral_balance
        * synchrony
        * max(0.0, 1.0 - region_fraction * math.sqrt(field_fraction))
        if bilateral_field
        else 0.0
    )
    focal_asymmetry = max(hemispheric_dominance, midline_mass)
    localized_raw = (
        concentration_top3
        * (0.30 + 0.70 * focal_asymmetry)
        * (0.40 + 0.60 * connected_fraction)
        * max(0.15, 1.0 - 0.65 * entropy)
    )

    reference = final.get("reference_stability", {})
    minimum_reference = (
        reference.get("minimum_similarity")
        if isinstance(reference, Mapping)
        else None
    )
    unresolved_raw = 0.04
    if minimum_reference is None:
        unresolved_raw += 0.30
    else:
        unresolved_raw += max(0.0, 0.75 - float(minimum_reference)) * 0.5
    if not temporal_receipt["onset_safe_channel_ids"]:
        unresolved_raw += 0.50

    state = _normalize(
        (localized_raw, bilateral_raw, generalized_raw, unresolved_raw)
    )
    assert state is not None
    gated_channel = channel_support
    if (
        state[2] >= max(state[0], state[1], state[3])
        and state[0] < policy.generalized_localized_axis_gate
    ):
        state_without_localized = _normalize((0.0, state[1], state[2], state[3]))
        assert state_without_localized is not None
        state = state_without_localized
        gated_channel = None

    return (
        state,
        gated_channel,
        {
            "status": "estimated_from_onset_safe_eeg_support",
            "localized_raw": _round(localized_raw),
            "bilateral_raw": _round(bilateral_raw),
            "generalized_raw": _round(generalized_raw),
            "unresolved_raw": _round(unresolved_raw),
            "left_support": _round(left_mass),
            "right_support": _round(right_mass),
            "observed_cz_support": _round(midline_mass),
            "bilateral_balance": _round(bilateral_balance),
            "earliest_field_channel_fraction": _round(field_fraction),
            "earliest_field_region_fraction": _round(region_fraction),
            "earliest_field_span_seconds": _round(field_span),
            "onset_synchrony_score": _round(synchrony),
            "channel_support_entropy": _round(entropy),
            "top3_channel_concentration": _round(concentration_top3),
            "dominant_connected_component_fraction": _round(connected_fraction),
            "late_involvement_used": False,
        },
    )


def _derive_qc_profile(
    receipt: Mapping[str, Any],
) -> tuple[Common17EventQCProfileV1, dict[str, Any]]:
    final = receipt["final_evidence"]
    rows = final.get("per_channel_evidence", [])
    evaluable = sum(
        1 for row in rows if isinstance(row, Mapping) and row.get("evaluable") is True
    )
    evaluable_fraction = evaluable / len(COMMON17_CHANNEL_IDS)
    baseline = final.get("robust_matched_baseline", {})
    baseline_qualified = (
        isinstance(baseline, Mapping)
        and baseline.get("status") == "qualified_robust_matched_baseline"
    )
    artifact_proxy = evaluable_fraction if baseline_qualified else 0.0

    reference = final.get("reference_stability", {})
    minimum_similarity = (
        reference.get("minimum_similarity")
        if isinstance(reference, Mapping)
        else None
    )
    reference_score = (
        0.0
        if minimum_similarity is None
        else min(1.0, max(0.0, float(minimum_similarity)))
    )

    support = receipt["final_variable_support"]
    interval = support["interval_recording_seconds"]
    onset = final.get("onset_candidate")
    if isinstance(onset, Mapping):
        preonset_margin = max(0.0, float(onset["recording_seconds"]) - float(interval[0]))
        onset_boundary_score = min(1.0, preonset_margin / 2.0)
    else:
        preonset_margin = None
        onset_boundary_score = 0.0

    side_scores: list[float] = []
    for side, extent_key in (
        ("left", "left_extent_seconds"),
        ("right", "right_extent_seconds"),
    ):
        closure = support["side_closure"][side]
        reasons = set(closure["reason_codes"])
        extent = max(0.0, float(support[extent_key]))
        if closure["state"] == "normal_closed" or "search_cap_32s" in reasons:
            score = 1.0
        elif "impassable_qc_gap" in reasons:
            score = 0.0
        else:
            score = min(1.0, extent / 4.0)
        side_scores.append(score)
    adaptive_coverage = math.fsum(side_scores) / 2.0

    profile = Common17EventQCProfileV1(
        signal_valid_fraction=evaluable_fraction,
        common17_channel_coverage_fraction=evaluable_fraction,
        artifact_free_fraction=artifact_proxy,
        reference_stability=reference_score,
        onset_boundary_support=onset_boundary_score,
        adaptive_support_coverage=adaptive_coverage,
    )
    return (
        profile,
        {
            "evaluable_channel_count": evaluable,
            "evaluable_channel_fraction": _round(evaluable_fraction),
            "baseline_qualified": baseline_qualified,
            "artifact_free_fraction_is_conservative_baseline_opportunity_proxy": True,
            "reference_minimum_similarity": (
                None if minimum_similarity is None else _round(float(minimum_similarity))
            ),
            "preonset_support_margin_seconds": (
                None if preonset_margin is None else _round(preonset_margin)
            ),
            "side_support_scores": {
                "left": _round(side_scores[0]),
                "right": _round(side_scores[1]),
            },
            "all_qc_factors_are_eeg_derived": True,
        },
    )


def _onset_evidence_ids(
    source_sha256: str,
    components: Mapping[str, tuple[float, ...] | None],
    state_status: str,
) -> tuple[str, ...]:
    prefix = source_sha256[:16]
    identifiers = [
        f"ONSET-EVIDENCE:{name.upper().replace('_', '-')}:{prefix}"
        for name in _CHANNEL_COMPONENT_IDS
        if components[name] is not None
    ]
    if not identifiers:
        identifiers.append(f"ONSET-EVIDENCE:{state_status.upper().replace('_', '-')}:{prefix}")
    return tuple(identifiers)


def adapt_common17_adaptive_event_evidence_v1(
    receipt: object,
    *,
    policy: Common17AdaptiveEventAdapterPolicyV1 = (
        DEFAULT_COMMON17_ADAPTIVE_EVENT_ADAPTER_POLICY_V1
    ),
) -> tuple[Common17EventSOZEvidenceV1, dict[str, Any]]:
    """Project one validated native event receipt to typed uncalibrated support."""

    if not isinstance(policy, Common17AdaptiveEventAdapterPolicyV1):
        raise TypeError("policy must be Common17AdaptiveEventAdapterPolicyV1")
    source = validate_common17_adaptive_native_event_evidence(receipt)
    source_sha256 = _sha256(source["receipt_sha256"], "source receipt_sha256")
    components, channel_support, temporal = _extract_channel_components(
        source, policy
    )
    state, channel_support, state_derivation = _derive_independent_state_support(
        source,
        channel_support,
        temporal,
        policy,
    )
    qc, qc_derivation = _derive_qc_profile(source)
    evidence_ids = _onset_evidence_ids(
        source_sha256, components, str(state_derivation["status"])
    )
    mode_id = f"ONSET-SAFE-EVENT-SHADOW-V1:{source_sha256[:24]}"
    channel_semantics = (
        NOT_APPLICABLE_NONLOCALIZED
        if channel_support is None
        else UNCALIBRATED_NONNEGATIVE_SCORE
    )
    event = Common17EventSOZEvidenceV1(
        event_id=source["event_id"],
        source_event_evidence_sha256=source_sha256,
        mode_id=mode_id,
        channel_values=channel_support,
        channel_value_semantics=channel_semantics,
        state_values=state,
        state_value_semantics=UNCALIBRATED_NONNEGATIVE_SCORE,
        model_reliability=policy.neutral_extractor_multiplier,
        qc=qc,
        onset_evidence_ids=evidence_ids,
        channel_calibration=None,
        state_calibration=None,
        labels_or_external_context_present=False,
    )

    body: dict[str, Any] = {
        "schema_version": COMMON17_ADAPTIVE_RECORD_ADAPTER_SCHEMA_VERSION,
        "method_id": COMMON17_ADAPTIVE_RECORD_ADAPTER_METHOD_ID,
        "adaptation_sha256": "CONTENT-ADDRESS-PENDING",
        "record_id": source["recording_id"],
        "event_id": source["event_id"],
        "source_binding": {
            "schema_version": source["schema_version"],
            "method_id": source["method_id"],
            "receipt_sha256": source_sha256,
        },
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "common17_contract": {
            "channel_ids": list(COMMON17_CHANNEL_IDS),
            "excluded_signal_and_prediction_channels": ["FZ", "PZ"],
            "missing_channel_imputation_used": False,
            "fz_pz_to_cz_prediction_fallback_used": False,
            "cz_requires_observed_cz_onset_safe_evidence": True,
        },
        "mode_binding": {
            "mode_id": mode_id,
            "strategy": "one_event_one_onset_safe_shadow",
            "stable_seizure_type_claimed": False,
            "target_free_clustering_model_bound": False,
            "late_involvement_used": False,
            "course_used": False,
        },
        "value_contract": {
            "channel_input_semantics": channel_semantics,
            "state_input_semantics": UNCALIBRATED_NONNEGATIVE_SCORE,
            "output_facing_term": NORMALIZED_SUPPORT_SCORE,
            "calibration_status": "unknown_no_patient_disjoint_calibrator_bound",
            "channel_calibration_receipt_id": None,
            "state_calibration_receipt_id": None,
            "probability_language_authorized": False,
        },
        "channel_evidence": {
            "normalized_support_scores": _support_mapping(
                COMMON17_CHANNEL_IDS, channel_support
            ),
            "components": {
                name: {
                    "available": components[name] is not None,
                    "normalized_support_scores": _support_mapping(
                        COMMON17_CHANNEL_IDS, components[name]
                    ),
                    "configured_weight": _round(
                        policy.channel_component_weights[name]
                    ),
                }
                for name in _CHANNEL_COMPONENT_IDS
            },
            "temporal_gate": temporal,
            "late_spread_created_onset_support": False,
        },
        "independent_pattern_state": {
            "state_ids": list(INDEPENDENT_PATTERN_STATE_IDS),
            "normalized_support_scores": _support_mapping(
                INDEPENDENT_PATTERN_STATE_IDS, state
            ),
            "derivation": state_derivation,
            "nonlocalized_state_projected_to_channel_axis": False,
        },
        "qc_profile": qc.as_dict(),
        "qc_derivation": qc_derivation,
        "typed_event_projection": {
            "event_content_sha256": event.content_sha256,
            "onset_evidence_ids": list(evidence_ids),
            "neutral_extractor_multiplier_not_accuracy": (
                policy.neutral_extractor_multiplier
            ),
            "effective_reliability": _round(event.effective_reliability),
        },
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
    }
    body["adaptation_sha256"] = _self_hash(body, "adaptation_sha256")
    validated = validate_common17_adaptive_event_adapter_receipt_v1(body)
    return event, validated


def validate_common17_adaptive_event_adapter_receipt_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate the event adapter receipt and its no-calibration boundary."""

    if type(payload) is not dict:
        raise TypeError("adaptive event adapter receipt must be an object")
    required = {
        "schema_version",
        "method_id",
        "adaptation_sha256",
        "record_id",
        "event_id",
        "source_binding",
        "policy",
        "policy_sha256",
        "common17_contract",
        "mode_binding",
        "value_contract",
        "channel_evidence",
        "independent_pattern_state",
        "qc_profile",
        "qc_derivation",
        "typed_event_projection",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("adaptive event adapter receipt fields drifted")
    result = deepcopy(payload)
    if result["schema_version"] != COMMON17_ADAPTIVE_RECORD_ADAPTER_SCHEMA_VERSION:
        raise ValueError("adaptive event adapter schema drifted")
    if result["method_id"] != COMMON17_ADAPTIVE_RECORD_ADAPTER_METHOD_ID:
        raise ValueError("adaptive event adapter method drifted")
    _identifier(result["record_id"], "record_id")
    _identifier(result["event_id"], "event_id")
    source = result["source_binding"]
    if not isinstance(source, Mapping):
        raise ValueError("source binding is invalid")
    if source.get("schema_version") != ADAPTIVE_NATIVE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("source adaptive schema drifted")
    if source.get("method_id") != ADAPTIVE_NATIVE_EVIDENCE_METHOD_ID:
        raise ValueError("source adaptive method drifted")
    _sha256(source.get("receipt_sha256"), "source receipt_sha256")
    if result["policy_sha256"] != _canonical_sha256(result["policy"]):
        raise ValueError("adapter policy hash mismatch")

    common17 = result["common17_contract"]
    if not isinstance(common17, Mapping) or common17.get("channel_ids") != list(
        COMMON17_CHANNEL_IDS
    ):
        raise ValueError("adapter common-17 ontology drifted")
    if common17.get("excluded_signal_and_prediction_channels") != ["FZ", "PZ"]:
        raise ValueError("adapter no longer excludes FZ/PZ")
    if any(
        common17.get(field) is not expected
        for field, expected in (
            ("missing_channel_imputation_used", False),
            ("fz_pz_to_cz_prediction_fallback_used", False),
            ("cz_requires_observed_cz_onset_safe_evidence", True),
        )
    ):
        raise ValueError("adapter introduced a missing-channel/CZ fallback")

    mode = result["mode_binding"]
    if not isinstance(mode, Mapping) or mode.get("strategy") != (
        "one_event_one_onset_safe_shadow"
    ):
        raise ValueError("adapter mode strategy drifted")
    _identifier(mode.get("mode_id"), "mode_id")
    if any(
        mode.get(field) is not False
        for field in (
            "stable_seizure_type_claimed",
            "target_free_clustering_model_bound",
            "late_involvement_used",
            "course_used",
        )
    ):
        raise ValueError("adapter mode leaked course or fabricated a seizure type")

    values = result["value_contract"]
    if not isinstance(values, Mapping):
        raise ValueError("adapter value contract is invalid")
    if values.get("channel_input_semantics") not in {
        UNCALIBRATED_NONNEGATIVE_SCORE,
        NOT_APPLICABLE_NONLOCALIZED,
    }:
        raise ValueError("adapter channel semantics drifted")
    if values.get("state_input_semantics") != UNCALIBRATED_NONNEGATIVE_SCORE:
        raise ValueError("adapter state semantics drifted")
    if values.get("output_facing_term") != NORMALIZED_SUPPORT_SCORE:
        raise ValueError("adapter called uncalibrated values by another term")
    if values.get("calibration_status") != (
        "unknown_no_patient_disjoint_calibrator_bound"
    ):
        raise ValueError("adapter calibration status drifted")
    if any(
        values.get(field) is not None
        for field in (
            "channel_calibration_receipt_id",
            "state_calibration_receipt_id",
        )
    ) or values.get("probability_language_authorized") is not False:
        raise ValueError("adapter silently bound calibration or probability language")

    channel = result["channel_evidence"]
    if not isinstance(channel, Mapping):
        raise ValueError("adapter channel evidence is invalid")
    channel_values = _distribution_from_mapping(
        channel.get("normalized_support_scores"),
        COMMON17_CHANNEL_IDS,
        name="channel normalized support",
        allow_none=True,
    )
    components = channel.get("components")
    if not isinstance(components, Mapping) or set(components) != set(
        _CHANNEL_COMPONENT_IDS
    ):
        raise ValueError("adapter channel components drifted")
    for name in _CHANNEL_COMPONENT_IDS:
        row = components[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"adapter component {name} is invalid")
        component = _distribution_from_mapping(
            row.get("normalized_support_scores"),
            COMMON17_CHANNEL_IDS,
            name=f"component {name}",
            allow_none=True,
        )
        if (component is not None) is not (row.get("available") is True):
            raise ValueError(f"adapter component {name} availability drifted")
        _unit_interval(row.get("configured_weight"), f"{name} weight")
    temporal = channel.get("temporal_gate")
    if not isinstance(temporal, Mapping) or temporal.get(
        "late_spread_created_onset_support"
    ) is not False:
        raise ValueError("late spread leaked into adapter channel support")
    if channel.get("late_spread_created_onset_support") is not False:
        raise ValueError("late spread leaked into adapter channel support")

    state = result["independent_pattern_state"]
    if not isinstance(state, Mapping) or state.get("state_ids") != list(
        INDEPENDENT_PATTERN_STATE_IDS
    ):
        raise ValueError("adapter independent state ontology drifted")
    state_values = _distribution_from_mapping(
        state.get("normalized_support_scores"),
        INDEPENDENT_PATTERN_STATE_IDS,
        name="state normalized support",
        allow_none=False,
    )
    assert state_values is not None
    if state.get("nonlocalized_state_projected_to_channel_axis") is not False:
        raise ValueError("adapter projected nonlocalized state onto a channel")
    if channel_values is None and state_values[0] > _TOL:
        raise ValueError("localized state support lacks a channel axis")
    if channel_values is not None and values.get("channel_input_semantics") != (
        UNCALIBRATED_NONNEGATIVE_SCORE
    ):
        raise ValueError("available channel support has invalid semantics")
    if channel_values is None and values.get("channel_input_semantics") != (
        NOT_APPLICABLE_NONLOCALIZED
    ):
        raise ValueError("absent channel support has invalid semantics")

    qc = result["qc_profile"]
    if not isinstance(qc, Mapping):
        raise ValueError("adapter QC profile is invalid")
    for field in (
        "signal_valid_fraction",
        "common17_channel_coverage_fraction",
        "artifact_free_fraction",
        "reference_stability",
        "onset_boundary_support",
        "adaptive_support_coverage",
        "geometric_quality",
    ):
        _unit_interval(qc.get(field), f"qc {field}")
    projection = result["typed_event_projection"]
    if not isinstance(projection, Mapping):
        raise ValueError("typed event projection is invalid")
    _sha256(projection.get("event_content_sha256"), "event content_sha256")
    if projection.get("neutral_extractor_multiplier_not_accuracy") != 1.0:
        raise ValueError("adapter neutral multiplier drifted")
    _unit_interval(
        projection.get("effective_reliability"), "effective_reliability"
    )
    if result["scope_receipt"] != _EEG_ONLY_SCOPE:
        raise ValueError("adaptive event adapter violated the EEG-only scope")
    if result["adaptation_sha256"] != _self_hash(result, "adaptation_sha256"):
        raise ValueError("adaptive event adapter content hash mismatch")
    return result


def _typed_event_from_adapter_receipt(
    receipt: Mapping[str, Any],
) -> Common17EventSOZEvidenceV1:
    channel = _distribution_from_mapping(
        receipt["channel_evidence"]["normalized_support_scores"],
        COMMON17_CHANNEL_IDS,
        name="channel normalized support",
        allow_none=True,
    )
    state = _distribution_from_mapping(
        receipt["independent_pattern_state"]["normalized_support_scores"],
        INDEPENDENT_PATTERN_STATE_IDS,
        name="state normalized support",
        allow_none=False,
    )
    assert state is not None
    qc_values = receipt["qc_profile"]
    qc = Common17EventQCProfileV1(
        signal_valid_fraction=qc_values["signal_valid_fraction"],
        common17_channel_coverage_fraction=qc_values[
            "common17_channel_coverage_fraction"
        ],
        artifact_free_fraction=qc_values["artifact_free_fraction"],
        reference_stability=qc_values["reference_stability"],
        onset_boundary_support=qc_values["onset_boundary_support"],
        adaptive_support_coverage=qc_values["adaptive_support_coverage"],
    )
    return Common17EventSOZEvidenceV1(
        event_id=receipt["event_id"],
        source_event_evidence_sha256=receipt["source_binding"]["receipt_sha256"],
        mode_id=receipt["mode_binding"]["mode_id"],
        channel_values=channel,
        channel_value_semantics=receipt["value_contract"][
            "channel_input_semantics"
        ],
        state_values=state,
        state_value_semantics=UNCALIBRATED_NONNEGATIVE_SCORE,
        model_reliability=receipt["typed_event_projection"][
            "neutral_extractor_multiplier_not_accuracy"
        ],
        qc=qc,
        onset_evidence_ids=tuple(
            receipt["typed_event_projection"]["onset_evidence_ids"]
        ),
        channel_calibration=None,
        state_calibration=None,
        labels_or_external_context_present=False,
    )


def build_common17_complete_roster_from_adaptive_receipts_v1(
    receipts: Sequence[object],
    *,
    record_id: str,
    canonical_signal_sha256: str,
    upstream_model_artifact_sha256: str,
    source_scope: str = "deployment_eeg_only",
    adapter_policy: Common17AdaptiveEventAdapterPolicyV1 = (
        DEFAULT_COMMON17_ADAPTIVE_EVENT_ADAPTER_POLICY_V1
    ),
) -> tuple[Common17CompleteEventRosterV1, tuple[dict[str, Any], ...]]:
    """Adapt a complete chronological detector receipt roster without filtering."""

    record = _identifier(record_id, "record_id")
    canonical = _sha256(canonical_signal_sha256, "canonical_signal_sha256")
    upstream = _sha256(
        upstream_model_artifact_sha256, "upstream_model_artifact_sha256"
    )
    if isinstance(receipts, (str, bytes)) or not receipts:
        raise ValueError("receipts must contain the complete non-empty detector roster")
    event_rows: list[Common17EventSOZEvidenceV1] = []
    adapter_rows: list[dict[str, Any]] = []
    anchors: list[float] = []
    for receipt in receipts:
        source = validate_common17_adaptive_native_event_evidence(receipt)
        if source["recording_id"] != record:
            raise ValueError("adaptive receipt/record identifier mismatch")
        event, adapted = adapt_common17_adaptive_event_evidence_v1(
            source, policy=adapter_policy
        )
        event_rows.append(event)
        adapter_rows.append(adapted)
        anchors.append(float(source["navigation_anchor_recording_seconds"]))
    if any(right < left for left, right in zip(anchors, anchors[1:])):
        raise ValueError("adaptive receipts are not in detector chronological order")
    bag = Common17CompleteEventRosterV1(
        record_id=record,
        canonical_signal_sha256=canonical,
        upstream_model_artifact_sha256=upstream,
        detector_event_roster=tuple(event.event_id for event in event_rows),
        events=tuple(event_rows),
        source_scope=source_scope,
        labels_or_external_context_present=False,
    )
    return bag, tuple(adapter_rows)


def run_common17_adaptive_findings_record_soz_v1(
    *,
    record_id: str,
    canonical_signal_sha256: str,
    upstream_model_artifact_sha256: str,
    detected_events: Sequence[Common17DetectedNavigationEventV1],
    sampling_rate_hz: float,
    recording_sample_count: int,
    query_reader: NativeEEGQueryReader,
    source_scope: str = "deployment_eeg_only",
    adaptive_policy: AdaptiveNativeEvidencePolicy | None = None,
    adapter_policy: Common17AdaptiveEventAdapterPolicyV1 = (
        DEFAULT_COMMON17_ADAPTIVE_EVENT_ADAPTER_POLICY_V1
    ),
    aggregation_policy: Common17RecordSOZAggregationPolicyV1 | None = None,
) -> dict[str, Any]:
    """Run adaptive EEG measurement, typed adaptation and record aggregation."""

    record = _identifier(record_id, "record_id")
    if isinstance(detected_events, (str, bytes)) or not detected_events:
        raise ValueError("detected_events must be a non-empty complete roster")
    if not all(
        isinstance(event, Common17DetectedNavigationEventV1)
        for event in detected_events
    ):
        raise TypeError("detected_events must be typed navigation events")
    event_ids = [event.event_id for event in detected_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("detected event roster contains duplicate identifiers")
    anchors = [event.anchor_recording_seconds for event in detected_events]
    if any(right < left for left, right in zip(anchors, anchors[1:])):
        raise ValueError("detected event roster must be chronological")
    native_policy = adaptive_policy or AdaptiveNativeEvidencePolicy()
    if not isinstance(native_policy, AdaptiveNativeEvidencePolicy):
        raise TypeError("adaptive_policy must be AdaptiveNativeEvidencePolicy")

    native_receipts = [
        materialize_common17_adaptive_native_event_evidence(
            event_id=event.event_id,
            recording_id=record,
            navigation_anchor_recording_seconds=event.anchor_recording_seconds,
            sampling_rate_hz=sampling_rate_hz,
            recording_sample_count=recording_sample_count,
            query_reader=query_reader,
            policy=native_policy,
        )
        for event in detected_events
    ]
    bag, adapter_receipts = build_common17_complete_roster_from_adaptive_receipts_v1(
        native_receipts,
        record_id=record,
        canonical_signal_sha256=canonical_signal_sha256,
        upstream_model_artifact_sha256=upstream_model_artifact_sha256,
        source_scope=source_scope,
        adapter_policy=adapter_policy,
    )
    record_result = aggregate_common17_record_soz_evidence_v1(
        bag,
        policy=aggregation_policy,
    )
    body: dict[str, Any] = {
        "schema_version": COMMON17_ADAPTIVE_RECORD_RUN_SCHEMA_VERSION,
        "method_id": COMMON17_ADAPTIVE_RECORD_RUN_METHOD_ID,
        "run_sha256": "CONTENT-ADDRESS-PENDING",
        "record_id": record,
        "detector_event_roster": event_ids,
        "native_event_receipt_bindings": [
            {
                "event_id": receipt["event_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
                "navigation_anchor_recording_seconds": receipt[
                    "navigation_anchor_recording_seconds"
                ],
            }
            for receipt in native_receipts
        ],
        "event_adapter_receipts": list(adapter_receipts),
        "record_aggregation": record_result,
        "calibration_boundary": {
            "event_calibration_receipts_bound": False,
            "record_calibration_receipt_bound": False,
            "record_output_value_semantics": NORMALIZED_SUPPORT_SCORE,
            "probability_language_authorized": False,
        },
        "complete_roster_receipt": {
            "input_detector_event_count": len(detected_events),
            "native_receipt_count": len(native_receipts),
            "adapter_receipt_count": len(adapter_receipts),
            "aggregation_ledger_event_count": record_result["evidence_ledger"][
                "ledger_event_count"
            ],
            "all_detected_events_retained_even_if_zero_weight": True,
            "excluded_event_ids": [],
        },
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
    }
    body["run_sha256"] = _self_hash(body, "run_sha256")
    return validate_common17_adaptive_findings_record_soz_run_v1(body)


def validate_common17_adaptive_findings_record_soz_run_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate end-to-end roster closure and the no-fallback/calibration gates."""

    if type(payload) is not dict:
        raise TypeError("adaptive findings record SOZ run must be an object")
    required = {
        "schema_version",
        "method_id",
        "run_sha256",
        "record_id",
        "detector_event_roster",
        "native_event_receipt_bindings",
        "event_adapter_receipts",
        "record_aggregation",
        "calibration_boundary",
        "complete_roster_receipt",
        "scope_receipt",
    }
    if set(payload) != required:
        raise ValueError("adaptive findings record run fields drifted")
    result = deepcopy(payload)
    if result["schema_version"] != COMMON17_ADAPTIVE_RECORD_RUN_SCHEMA_VERSION:
        raise ValueError("adaptive findings record run schema drifted")
    if result["method_id"] != COMMON17_ADAPTIVE_RECORD_RUN_METHOD_ID:
        raise ValueError("adaptive findings record run method drifted")
    record = _identifier(result["record_id"], "record_id")
    roster = result["detector_event_roster"]
    if not isinstance(roster, list) or not roster:
        raise ValueError("adaptive findings record run lacks a detector roster")
    if len(roster) != len(set(roster)):
        raise ValueError("adaptive findings record detector roster repeats IDs")
    for event_id in roster:
        _identifier(event_id, "detector event_id")

    native = result["native_event_receipt_bindings"]
    adapters = result["event_adapter_receipts"]
    if not isinstance(native, list) or not isinstance(adapters, list):
        raise ValueError("adaptive findings record evidence bindings are invalid")
    if [row.get("event_id") for row in native if isinstance(row, Mapping)] != roster:
        raise ValueError("native event bindings do not close the detector roster")
    for row in native:
        if not isinstance(row, Mapping):
            raise ValueError("native event binding is invalid")
        _sha256(row.get("receipt_sha256"), "native receipt_sha256")
    validated_adapters = [
        validate_common17_adaptive_event_adapter_receipt_v1(row)
        for row in adapters
    ]
    if [row["event_id"] for row in validated_adapters] != roster:
        raise ValueError("adapter receipts do not close the detector roster")
    for native_row, adapter_row in zip(native, validated_adapters):
        if native_row["receipt_sha256"] != adapter_row["source_binding"][
            "receipt_sha256"
        ]:
            raise ValueError("adapter receipt is not bound to its native evidence")
        if adapter_row["record_id"] != record:
            raise ValueError("adapter receipt/record identifier mismatch")

    aggregation = validate_common17_record_soz_evidence_aggregation_v1(
        result["record_aggregation"]
    )
    ledger = aggregation["evidence_ledger"]
    if ledger["detector_event_roster"] != roster or ledger[
        "ledger_event_count"
    ] != len(roster):
        raise ValueError("record aggregation ledger does not close detector roster")
    if ledger["excluded_event_ids"] or ledger["all_detector_events_entered"] is not True:
        raise ValueError("record aggregation silently excluded a detected event")
    for row in ledger["event_rows"]:
        if row["roster_included"] is not True:
            raise ValueError("record aggregation contains an excluded event row")
        if row["channel_calibration_receipt_id"] is not None or row[
            "state_calibration_receipt_id"
        ] is not None:
            raise ValueError("default adapter run unexpectedly bound event calibration")
    adapter_by_event = {row["event_id"]: row for row in validated_adapters}
    for ledger_row in ledger["event_rows"]:
        adapter_row = adapter_by_event[ledger_row["event_id"]]
        if ledger_row["source_event_evidence_sha256"] != adapter_row[
            "source_binding"
        ]["receipt_sha256"]:
            raise ValueError("aggregation ledger lost its native evidence binding")
        if ledger_row["event_content_sha256"] != adapter_row[
            "typed_event_projection"
        ]["event_content_sha256"]:
            raise ValueError("aggregation ledger lost its typed event binding")
    ontology = aggregation["common17_ontology"]
    if ontology["channel_ids"] != list(COMMON17_CHANNEL_IDS):
        raise ValueError("record aggregation common-17 ontology drifted")
    if ontology["prediction_side_fz_pz_to_cz_mapping_used"] is not False:
        raise ValueError("record aggregation introduced an FZ/PZ-to-CZ fallback")
    if aggregation["spatial_localization"][
        "nonlocalized_states_projected_to_channels"
    ] is not False:
        raise ValueError("record aggregation projected nonlocalized state to channels")

    calibration = result["calibration_boundary"]
    if calibration != {
        "event_calibration_receipts_bound": False,
        "record_calibration_receipt_bound": False,
        "record_output_value_semantics": NORMALIZED_SUPPORT_SCORE,
        "probability_language_authorized": False,
    }:
        raise ValueError("adaptive findings record calibration boundary drifted")
    complete = result["complete_roster_receipt"]
    if not isinstance(complete, Mapping) or any(
        complete.get(field) != len(roster)
        for field in (
            "input_detector_event_count",
            "native_receipt_count",
            "adapter_receipt_count",
            "aggregation_ledger_event_count",
        )
    ):
        raise ValueError("adaptive findings record counts do not close")
    if complete.get("all_detected_events_retained_even_if_zero_weight") is not True:
        raise ValueError("adaptive findings record dropped its zero-weight guarantee")
    if complete.get("excluded_event_ids") != []:
        raise ValueError("adaptive findings record contains excluded event IDs")
    if result["scope_receipt"] != _EEG_ONLY_SCOPE:
        raise ValueError("adaptive findings record run violated EEG-only scope")
    if result["run_sha256"] != _self_hash(result, "run_sha256"):
        raise ValueError("adaptive findings record run content hash mismatch")
    return result


def project_common17_record_soz_label_free_prediction_v1(
    run: object,
) -> dict[str, Any]:
    """Export the frozen, label-free row consumed by later SOZ evaluation joins.

    Ground truth is intentionally absent.  Exact Top-1 and DeepSOZ-style N2/N4
    are computed only after this prediction row and its content hash have been
    frozen, using an evaluation-only join outside this inference module.
    """

    source = validate_common17_adaptive_findings_record_soz_run_v1(run)
    aggregation = source["record_aggregation"]
    spatial = aggregation["spatial_localization"]
    state = aggregation["independent_pattern_state"]
    channel_mapping = spatial["channel_values"]
    channel_values = (
        None
        if channel_mapping is None
        else [float(channel_mapping[channel]) for channel in COMMON17_CHANNEL_IDS]
    )
    top1 = (
        None
        if not spatial["channel_ranking"]
        else spatial["channel_ranking"][0]["candidate_id"]
    )
    body: dict[str, Any] = {
        "schema_version": COMMON17_RECORD_SOZ_EVALUATION_PREDICTION_SCHEMA_VERSION,
        "prediction_sha256": "CONTENT-ADDRESS-PENDING",
        "record_id": source["record_id"],
        "run_sha256": source["run_sha256"],
        "aggregation_sha256": aggregation["aggregation_sha256"],
        "channel_ids": list(COMMON17_CHANNEL_IDS),
        "channel_normalized_support_scores": channel_values,
        "predicted_top1_channel": top1,
        "laterality_ranking": deepcopy(spatial["laterality_ranking"]),
        "region_ranking": deepcopy(spatial["region_ranking"]),
        "independent_pattern_state_normalized_support_scores": [
            float(state["mass_values"][identifier])
            for identifier in INDEPENDENT_PATTERN_STATE_IDS
        ],
        "independent_pattern_state_ids": list(INDEPENDENT_PATTERN_STATE_IDS),
        "value_semantics": NORMALIZED_SUPPORT_SCORE,
        "probability_language_authorized": False,
        "label_free_at_export": True,
        "evaluation_join_allowed_only_after_content_freeze": True,
        "exact_top1_endpoint_supported": True,
        "deepsoz_neighbor_tolerant_n2_n4_endpoints_supported": True,
        "fz_pz_to_cz_prediction_fallback_used": False,
    }
    body["prediction_sha256"] = _self_hash(body, "prediction_sha256")
    return validate_common17_record_soz_label_free_prediction_v1(body)


def validate_common17_record_soz_label_free_prediction_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate a label-free common-17 SOZ evaluation prediction row."""

    if type(payload) is not dict:
        raise TypeError("SOZ evaluation prediction must be an object")
    required = {
        "schema_version",
        "prediction_sha256",
        "record_id",
        "run_sha256",
        "aggregation_sha256",
        "channel_ids",
        "channel_normalized_support_scores",
        "predicted_top1_channel",
        "laterality_ranking",
        "region_ranking",
        "independent_pattern_state_normalized_support_scores",
        "independent_pattern_state_ids",
        "value_semantics",
        "probability_language_authorized",
        "label_free_at_export",
        "evaluation_join_allowed_only_after_content_freeze",
        "exact_top1_endpoint_supported",
        "deepsoz_neighbor_tolerant_n2_n4_endpoints_supported",
        "fz_pz_to_cz_prediction_fallback_used",
    }
    if set(payload) != required:
        raise ValueError("SOZ evaluation prediction fields drifted")
    result = deepcopy(payload)
    if result["schema_version"] != (
        COMMON17_RECORD_SOZ_EVALUATION_PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("SOZ evaluation prediction schema drifted")
    _identifier(result["record_id"], "record_id")
    _sha256(result["run_sha256"], "run_sha256")
    _sha256(result["aggregation_sha256"], "aggregation_sha256")
    if result["channel_ids"] != list(COMMON17_CHANNEL_IDS):
        raise ValueError("SOZ evaluation prediction channel ontology drifted")
    values = result["channel_normalized_support_scores"]
    if values is None:
        if result["predicted_top1_channel"] is not None:
            raise ValueError("SOZ evaluation prediction fabricates a Top-1")
    else:
        normalized = _normalize(values)
        if normalized is None or len(normalized) != len(COMMON17_CHANNEL_IDS):
            raise ValueError("SOZ evaluation prediction channel scores are invalid")
        if any(abs(left - right) > _TOL for left, right in zip(normalized, values)):
            raise ValueError("SOZ evaluation prediction channel scores do not close")
        expected_top1 = min(
            range(len(COMMON17_CHANNEL_IDS)),
            key=lambda index: (-float(values[index]), COMMON17_CHANNEL_IDS[index]),
        )
        if result["predicted_top1_channel"] != COMMON17_CHANNEL_IDS[expected_top1]:
            raise ValueError("SOZ evaluation prediction Top-1 is inconsistent")
    if result["independent_pattern_state_ids"] != list(
        INDEPENDENT_PATTERN_STATE_IDS
    ):
        raise ValueError("SOZ evaluation prediction state ontology drifted")
    state = _normalize(result["independent_pattern_state_normalized_support_scores"])
    if state is None or any(
        abs(left - right) > _TOL
        for left, right in zip(
            state, result["independent_pattern_state_normalized_support_scores"]
        )
    ):
        raise ValueError("SOZ evaluation prediction state scores do not close")
    if result["value_semantics"] != NORMALIZED_SUPPORT_SCORE:
        raise ValueError("SOZ evaluation prediction value semantics drifted")
    for field, expected in (
        ("probability_language_authorized", False),
        ("label_free_at_export", True),
        ("evaluation_join_allowed_only_after_content_freeze", True),
        ("exact_top1_endpoint_supported", True),
        ("deepsoz_neighbor_tolerant_n2_n4_endpoints_supported", True),
        ("fz_pz_to_cz_prediction_fallback_used", False),
    ):
        if result[field] is not expected:
            raise ValueError(f"SOZ evaluation prediction {field} drifted")
    if result["prediction_sha256"] != _self_hash(result, "prediction_sha256"):
        raise ValueError("SOZ evaluation prediction content hash mismatch")
    return result


__all__ = [
    "COMMON17_ADAPTIVE_RECORD_ADAPTER_METHOD_ID",
    "COMMON17_ADAPTIVE_RECORD_ADAPTER_SCHEMA_VERSION",
    "COMMON17_ADAPTIVE_RECORD_RUN_METHOD_ID",
    "COMMON17_ADAPTIVE_RECORD_RUN_SCHEMA_VERSION",
    "COMMON17_RECORD_SOZ_EVALUATION_PREDICTION_SCHEMA_VERSION",
    "DEFAULT_COMMON17_ADAPTIVE_EVENT_ADAPTER_POLICY_V1",
    "Common17AdaptiveEventAdapterPolicyV1",
    "Common17DetectedNavigationEventV1",
    "adapt_common17_adaptive_event_evidence_v1",
    "build_common17_complete_roster_from_adaptive_receipts_v1",
    "project_common17_record_soz_label_free_prediction_v1",
    "run_common17_adaptive_findings_record_soz_v1",
    "validate_common17_adaptive_event_adapter_receipt_v1",
    "validate_common17_adaptive_findings_record_soz_run_v1",
    "validate_common17_record_soz_label_free_prediction_v1",
]
