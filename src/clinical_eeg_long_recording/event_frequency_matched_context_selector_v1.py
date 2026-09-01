"""EEG-only, typed-unit-common matched-context selector for S03.

The selector consumes a dense *measurement-opportunity* ledger and physical
event/support intervals.  It never consumes the numerical measurement values,
detector scores, reference labels, SOZ targets, annotations, spreadsheets or
clinical text.  A pre-event context interval is selected only when the same
policy-aligned physical window is QC-clean and spectrally evaluable for every
typed unit.  This prevents per-channel post-hoc context selection from
creating an artificial spatial contrast.

Selection is deliberately geometry/QC based.  It does not choose the context
whose waveform is most similar to, or most different from, the event.  If no
common pre-event opportunity exists, an optional future interval may be
recorded as ``course_only``.  Future context and every selector output have
``onset_support_permission=forbidden``: the receipt can define an S03 physical
context delta, but it can never add an SOZ score or create an onset fact.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_dense_measurement_sidecar import (
    BAIEGDenseMeasurementSidecar,
    BAIEGDenseMeasurementRowBinding,
    BAIEGDenseMeasurementViewBinding,
    BAIEGDenseMeasurementPolicy,
    BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
)
from .ba_ieg_training_contract import BA_IEG_DETERMINISTIC_TARGETS


S03_MATCHED_CONTEXT_SELECTOR_SCHEMA_VERSION_V1: Final[str] = (
    "clinical_eeg_s03_typed_eeg_matched_context_selector_v1"
)
S03_MATCHED_CONTEXT_SELECTOR_METHOD_ID_V1: Final[str] = (
    "S03-TYPED-EEG-COMMON-MATCHED-CONTEXT-SELECTOR-V1"
)

_TOL = 1e-8
_HEX = frozenset("0123456789abcdef")
_SPECTRAL_TARGETS = (
    "dominant_frequency_hz",
    "spectral_concentration",
    "spectral_entropy",
)
_SPECTRAL_INDICES = tuple(
    BA_IEG_DETERMINISTIC_TARGETS.index(name) for name in _SPECTRAL_TARGETS
)

_FIREWALL: Final[dict[str, bool]] = {
    "canonical_or_native_eeg_measurement_opportunity_used": True,
    "physical_event_and_support_intervals_used": True,
    "typed_unit_reference_qc_and_raw_dependency_used": True,
    "measurement_target_values_used": False,
    "event_waveform_values_used_for_context_similarity": False,
    "detector_score_logit_embedding_used": False,
    "reference_onset_or_event_label_used": False,
    "soz_rank_or_channel_target_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "selector_only_defines_context_delta": True,
    "normal_background_claim_authorized": False,
    "clinical_term_qualification_authorized": False,
    "direct_onset_support_authorized": False,
    "direct_soz_score_authorized": False,
    "bipolar_endpoint_fact_projection_authorized": False,
    "report_promotion_authorized": False,
    "onset_support_permission": "forbidden",
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


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum - _TOL:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _fraction(value: object, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0 + _TOL:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _interval(value: Sequence[object], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number interval")
    start = _finite(value[0], f"{name}[0]", minimum=0.0)
    stop = _finite(value[1], f"{name}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _contains(carrier: tuple[float, float], item: tuple[float, float]) -> bool:
    return item[0] >= carrier[0] - _TOL and item[1] <= carrier[1] + _TOL


def _normalise_intervals(
    values: Sequence[Sequence[float]],
    *,
    name: str,
    support: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    result = tuple(_interval(value, f"{name}[{index}]") for index, value in enumerate(values))
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted")
    if any(not _contains(support, item) for item in result):
        raise ValueError(f"{name} must remain inside analysis support")
    if any(_overlap(left, right) > _TOL for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be disjoint")
    return result


@dataclass(frozen=True)
class S03MatchedContextSelectorPolicyV1:
    """Frozen engineering defaults; none are clinical normality thresholds."""

    pre_event_protection_seconds: float = 0.0
    post_event_protection_seconds: float = 0.0
    distant_minimum_separation_seconds: float = 10.0
    require_all_typed_units: bool = True
    minimum_common_spectral_unit_fraction: float = 1.0
    minimum_artifact_free_unit_fraction: float = 1.0
    minimum_reference_match_fraction: float = 1.0
    minimum_bandwidth_compatible_fraction: float = 1.0
    allow_course_only_future_fallback: bool = True
    usable_weight: float = 0.35
    artifact_free_weight: float = 0.20
    gap_free_weight: float = 0.15
    reference_match_weight: float = 0.10
    bandwidth_match_weight: float = 0.10
    temporal_proximity_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "pre_event_protection_seconds",
            "post_event_protection_seconds",
            "distant_minimum_separation_seconds",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, minimum=0.0),
            )
        for name in (
            "minimum_common_spectral_unit_fraction",
            "minimum_artifact_free_unit_fraction",
            "minimum_reference_match_fraction",
            "minimum_bandwidth_compatible_fraction",
            "usable_weight",
            "artifact_free_weight",
            "gap_free_weight",
            "reference_match_weight",
            "bandwidth_match_weight",
            "temporal_proximity_weight",
        ):
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        if type(self.require_all_typed_units) is not bool:
            raise TypeError("require_all_typed_units must be boolean")
        if type(self.allow_course_only_future_fallback) is not bool:
            raise TypeError("allow_course_only_future_fallback must be boolean")
        if not self.require_all_typed_units:
            raise ValueError("v1 requires one common interval across all typed units")
        if not math.isclose(
            self.minimum_common_spectral_unit_fraction,
            1.0,
            abs_tol=_TOL,
        ):
            raise ValueError("v1 requires spectral opportunity for every typed unit")
        weights = (
            self.usable_weight,
            self.artifact_free_weight,
            self.gap_free_weight,
            self.reference_match_weight,
            self.bandwidth_match_weight,
            self.temporal_proximity_weight,
        )
        if not math.isclose(sum(weights), 1.0, abs_tol=_TOL):
            raise ValueError("selector weights must sum to one")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value = asdict(self)
        value["selection_basis"] = (
            "geometry_qc_opportunity_only_no_event_similarity_or_target_values"
        )
        value["shared_interval_semantics"] = (
            "one_physical_interval_shared_by_all_typed_units"
        )
        if include_hash:
            value["policy_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> "S03MatchedContextSelectorPolicyV1":
        if type(value) is not dict:
            raise TypeError("selector policy must be an object")
        expected = set(cls.__dataclass_fields__) | {
            "selection_basis",
            "shared_interval_semantics",
            "policy_sha256",
        }
        if set(value) != expected:
            raise ValueError("selector policy fields drifted")
        payload = deepcopy(value)
        observed_hash = payload.pop("policy_sha256")
        if payload.pop("selection_basis") != (
            "geometry_qc_opportunity_only_no_event_similarity_or_target_values"
        ):
            raise ValueError("selector policy basis drifted")
        if payload.pop("shared_interval_semantics") != (
            "one_physical_interval_shared_by_all_typed_units"
        ):
            raise ValueError("selector shared-interval semantics drifted")
        policy = cls(**payload)
        if observed_hash != policy.policy_sha256:
            raise ValueError("selector policy hash drifted")
        return policy


DEFAULT_S03_MATCHED_CONTEXT_SELECTOR_POLICY_V1 = (
    S03MatchedContextSelectorPolicyV1()
)


def _typed_roster(
    sidecar: BAIEGDenseMeasurementSidecar,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]]]:
    view_by_index: dict[int, BAIEGDenseMeasurementViewBinding] = {
        item.view_index: item for item in sidecar.view_bindings
    }
    first_rows: dict[tuple[int, int], BAIEGDenseMeasurementRowBinding] = {}
    for row in sidecar.row_bindings:
        first_rows.setdefault((row.view_index, row.unit_index), row)
    roster: list[dict[str, object]] = []
    by_key: dict[tuple[int, int], dict[str, object]] = {}
    for key in sorted(first_rows):
        row = first_rows[key]
        view = view_by_index[row.view_index]
        kind = "whole_bipolar_lead" if row.unit_type == "lead" else "electrode"
        if row.unit_type not in {"lead", "electrode"}:
            raise ValueError("S03 selector supports electrodes and whole leads only")
        item: dict[str, object] = {
            "view_index": row.view_index,
            "unit_index": row.unit_index,
            "view_id": row.view_id,
            "typed_unit_id": row.unit_id,
            "typed_unit_kind": kind,
            "source_unit_type": row.unit_type,
            "reference_type": row.reference_type,
            "reference_row_sha256": row.reference_row_sha256,
            "reference_matrix_sha256": view.reference_matrix_sha256,
            "canonical_source_channel_ids": list(row.canonical_source_channel_ids),
            "effective_bandwidth_hz": list(row.effective_bandwidth_hz),
            "quality_mask_sha256": row.quality_mask_sha256,
            "whole_output_unit_identity_preserved": True,
            "bipolar_endpoint_fact_projection_authorized": False,
        }
        roster.append(item)
        by_key[key] = item
    if not roster:
        raise ValueError("selector requires at least one typed unit")
    for row in sidecar.row_bindings:
        expected = by_key[(row.view_index, row.unit_index)]
        comparisons = {
            "view_id": row.view_id,
            "typed_unit_id": row.unit_id,
            "source_unit_type": row.unit_type,
            "reference_type": row.reference_type,
            "reference_row_sha256": row.reference_row_sha256,
            "canonical_source_channel_ids": list(row.canonical_source_channel_ids),
            "effective_bandwidth_hz": list(row.effective_bandwidth_hz),
            "quality_mask_sha256": row.quality_mask_sha256,
        }
        if any(expected[name] != observed for name, observed in comparisons.items()):
            raise ValueError("typed-unit/reference identity changed across windows")
    return roster, by_key


def _group_rows(
    sidecar: BAIEGDenseMeasurementSidecar,
) -> list[tuple[tuple[float, float], list[BAIEGDenseMeasurementRowBinding]]]:
    groups: dict[tuple[float, float], list[BAIEGDenseMeasurementRowBinding]] = {}
    for row in sidecar.row_bindings:
        groups.setdefault(tuple(row.requested_recording_interval_seconds), []).append(row)
    return [
        (interval, sorted(rows, key=lambda item: (item.view_index, item.unit_index)))
        for interval, rows in sorted(groups.items())
    ]


def _expanded(
    interval: tuple[float, float],
    *,
    support: tuple[float, float],
    pre_seconds: float,
    post_seconds: float,
) -> tuple[float, float]:
    return (
        max(support[0], interval[0] - pre_seconds),
        min(support[1], interval[1] + post_seconds),
    )


def _score_components(
    *,
    rows: Sequence[BAIEGDenseMeasurementRowBinding],
    typed_roster: Mapping[tuple[int, int], Mapping[str, object]],
    interval: tuple[float, float],
    temporal_class: str,
    event_protection: tuple[float, float],
    gap_free: bool,
    measurement_policy: BAIEGDenseMeasurementPolicy,
    policy: S03MatchedContextSelectorPolicyV1,
) -> dict[str, float]:
    expected_keys = set(typed_roster)
    observed_keys = {(row.view_index, row.unit_index) for row in rows}
    total = len(expected_keys)
    spectral_usable = 0
    artifact_free = 0
    reference_match = 0
    bandwidth_compatible = 0
    for row in rows:
        key = (row.view_index, row.unit_index)
        if key not in expected_keys:
            continue
        if all(row.target_value_mask[index] for index in _SPECTRAL_INDICES):
            spectral_usable += 1
        if not row.overlapping_quality_reason_codes:
            artifact_free += 1
        roster = typed_roster[key]
        if (
            row.reference_type == roster["reference_type"]
            and row.reference_row_sha256 == roster["reference_row_sha256"]
        ):
            reference_match += 1
        low, high = row.effective_bandwidth_hz
        if (
            low <= measurement_policy.analysis_low_hz + _TOL
            and high >= measurement_policy.analysis_high_hz - _TOL
        ):
            bandwidth_compatible += 1
    complete = observed_keys == expected_keys and len(rows) == total
    if not complete:
        spectral_usable = min(spectral_usable, len(observed_keys & expected_keys))
    if temporal_class == "pre_event":
        distance = max(0.0, event_protection[0] - interval[1])
    elif temporal_class == "future_course":
        distance = max(0.0, interval[0] - event_protection[1])
    else:
        distance = 0.0
    return {
        "common_typed_unit_row_fraction": float(len(observed_keys & expected_keys) / total),
        "spectral_usable_unit_fraction": float(spectral_usable / total),
        "artifact_free_unit_fraction": float(artifact_free / total),
        "gap_free_fraction": 1.0 if gap_free else 0.0,
        "reference_match_fraction": float(reference_match / total),
        "bandwidth_compatible_fraction": float(bandwidth_compatible / total),
        "temporal_distance_seconds": float(distance),
        "temporal_proximity_score": float(1.0 / (1.0 + distance)),
    }


def _selection_score(
    components: Mapping[str, float], policy: S03MatchedContextSelectorPolicyV1
) -> float:
    return float(
        policy.usable_weight * components["spectral_usable_unit_fraction"]
        + policy.artifact_free_weight * components["artifact_free_unit_fraction"]
        + policy.gap_free_weight * components["gap_free_fraction"]
        + policy.reference_match_weight * components["reference_match_fraction"]
        + policy.bandwidth_match_weight
        * components["bandwidth_compatible_fraction"]
        + policy.temporal_proximity_weight
        * components["temporal_proximity_score"]
    )


def _evaluated_candidate(
    *,
    candidate_index: int,
    interval: tuple[float, float],
    rows: Sequence[BAIEGDenseMeasurementRowBinding],
    temporal_class: str,
    event_protection: tuple[float, float],
    typed_roster: Mapping[tuple[int, int], Mapping[str, object]],
    gaps: Sequence[tuple[float, float]],
    measurement_policy: BAIEGDenseMeasurementPolicy,
    policy: S03MatchedContextSelectorPolicyV1,
) -> dict[str, object]:
    gap_free = not any(_overlap(interval, gap) > _TOL for gap in gaps)
    components = _score_components(
        rows=rows,
        typed_roster=typed_roster,
        interval=interval,
        temporal_class=temporal_class,
        event_protection=event_protection,
        gap_free=gap_free,
        measurement_policy=measurement_policy,
        policy=policy,
    )
    reasons: list[str] = []
    if components["common_typed_unit_row_fraction"] < 1.0 - _TOL:
        reasons.append("incomplete_common_typed_unit_opportunity")
    if (
        components["spectral_usable_unit_fraction"]
        < policy.minimum_common_spectral_unit_fraction - _TOL
    ):
        reasons.append("incomplete_common_spectral_opportunity")
    if (
        components["artifact_free_unit_fraction"]
        < policy.minimum_artifact_free_unit_fraction - _TOL
    ):
        reasons.append("qc_or_artifact_overlap")
    if not gap_free:
        reasons.append("signal_gap_overlap")
    if (
        components["reference_match_fraction"]
        < policy.minimum_reference_match_fraction - _TOL
    ):
        reasons.append("reference_identity_mismatch")
    if (
        components["bandwidth_compatible_fraction"]
        < policy.minimum_bandwidth_compatible_fraction - _TOL
    ):
        reasons.append("effective_bandwidth_incompatible")
    score = _selection_score(components, policy)
    return {
        "candidate_index": candidate_index,
        "interval_seconds": list(interval),
        "temporal_class": temporal_class,
        "evaluation_status": "eligible" if not reasons else "not_evaluable",
        "score_components": components,
        "selection_score": score,
        "reason_codes": sorted(reasons),
        "contamination_and_protection": {
            "event_protection_overlap": False,
            "other_candidate_protection_overlap": False,
            "signal_gap_overlap": not gap_free,
            "qc_or_artifact_overlap": "qc_or_artifact_overlap" in reasons,
            "normal_background_claim_authorized": False,
        },
        "opportunity": {
            "required_typed_unit_count": len(typed_roster),
            "observed_row_count": len(rows),
            "all_typed_units_share_interval": (
                components["common_typed_unit_row_fraction"] >= 1.0 - _TOL
            ),
        },
        "decision_dependency_row_sha256s": [
            row.source_binding_sha256 for row in rows
        ],
        "measurement_target_values_used": False,
    }


def _blocked_candidate(
    *,
    candidate_index: int,
    interval: tuple[float, float],
    temporal_class: str,
    reasons: Sequence[str],
    status: str = "not_evaluable",
) -> dict[str, object]:
    normalized = sorted(set(str(item) for item in reasons))
    return {
        "candidate_index": candidate_index,
        "interval_seconds": list(interval),
        "temporal_class": temporal_class,
        "evaluation_status": status,
        "score_components": None,
        "selection_score": None,
        "reason_codes": normalized,
        "contamination_and_protection": {
            "event_protection_overlap": "event_protection_overlap" in normalized,
            "other_candidate_protection_overlap": (
                "other_candidate_protection_overlap" in normalized
            ),
            "signal_gap_overlap": "signal_gap_overlap" in normalized,
            "qc_or_artifact_overlap": False,
            "normal_background_claim_authorized": False,
        },
        "opportunity": {
            "required_typed_unit_count": None,
            "observed_row_count": None,
            "all_typed_units_share_interval": None,
        },
        "decision_dependency_row_sha256s": [],
        "measurement_target_values_used": False,
    }


def _selected_slot(
    candidate: Mapping[str, object], *, temporal_role: str
) -> dict[str, object]:
    return {
        "status": "selected",
        "temporal_role": temporal_role,
        "interval_seconds": deepcopy(candidate["interval_seconds"]),
        "candidate_index": candidate["candidate_index"],
        "selection_score": candidate["selection_score"],
        "reason_codes": [],
        "future_context_selected": candidate["temporal_class"] == "future_course",
        "onset_support_permission": "forbidden",
        "direct_soz_score_authorized": False,
    }


def _empty_slot(*, temporal_role: str, reasons: Sequence[str]) -> dict[str, object]:
    return {
        "status": "not_evaluable",
        "temporal_role": temporal_role,
        "interval_seconds": None,
        "candidate_index": None,
        "selection_score": None,
        "reason_codes": sorted(set(str(item) for item in reasons)),
        "future_context_selected": False,
        "onset_support_permission": "forbidden",
        "direct_soz_score_authorized": False,
    }


def _decision_hash_material(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": payload["schema_version"],
        "method_id": payload["method_id"],
        "event_id": payload["event_id"],
        "policy": payload["policy"],
        "event_support": payload["event_support"],
        "typed_unit_roster": payload["typed_unit_roster"],
        "candidate_audit": payload["candidate_audit"],
        "selections": payload["selections"],
        "decision_semantics": payload["decision_semantics"],
        "firewall": payload["firewall"],
        "authorization": payload["authorization"],
    }


def materialize_s03_matched_context_selector_v1(
    *,
    event_id: str,
    dense_measurement_opportunity: BAIEGDenseMeasurementSidecar,
    event_interval_seconds: Sequence[float],
    other_candidate_intervals_seconds: Sequence[Sequence[float]] = (),
    signal_gap_intervals_seconds: Sequence[Sequence[float]] = (),
    policy: S03MatchedContextSelectorPolicyV1 = (
        DEFAULT_S03_MATCHED_CONTEXT_SELECTOR_POLICY_V1
    ),
) -> dict[str, Any]:
    """Select common pre-event context without consulting measurement values."""

    _identifier(event_id, "event_id")
    if not isinstance(dense_measurement_opportunity, BAIEGDenseMeasurementSidecar):
        raise TypeError(
            "dense_measurement_opportunity must be BAIEGDenseMeasurementSidecar"
        )
    if not isinstance(policy, S03MatchedContextSelectorPolicyV1):
        raise TypeError("policy must be S03MatchedContextSelectorPolicyV1")
    dense_measurement_opportunity.verify_integrity()
    if dense_measurement_opportunity.background_intervals_seconds:
        raise ValueError(
            "selector opportunity sidecar must not contain preselected backgrounds"
        )
    support = dense_measurement_opportunity.analysis_interval_seconds
    event_interval = _interval(event_interval_seconds, "event_interval_seconds")
    if not _contains(support, event_interval):
        raise ValueError("event interval lies outside measurement support")
    other_candidates = _normalise_intervals(
        other_candidate_intervals_seconds,
        name="other_candidate_intervals_seconds",
        support=support,
    )
    gaps = _normalise_intervals(
        signal_gap_intervals_seconds,
        name="signal_gap_intervals_seconds",
        support=support,
    )
    event_protection = _expanded(
        event_interval,
        support=support,
        pre_seconds=policy.pre_event_protection_seconds,
        post_seconds=policy.post_event_protection_seconds,
    )
    other_protections = tuple(
        _expanded(
            item,
            support=support,
            pre_seconds=policy.pre_event_protection_seconds,
            post_seconds=policy.post_event_protection_seconds,
        )
        for item in other_candidates
    )
    roster, roster_by_key = _typed_roster(dense_measurement_opportunity)
    groups = _group_rows(dense_measurement_opportunity)

    audit_by_index: dict[int, dict[str, object]] = {}
    pre_evaluated: list[dict[str, object]] = []
    future_groups: list[
        tuple[int, tuple[float, float], list[BAIEGDenseMeasurementRowBinding]]
    ] = []
    for index, (interval, rows) in enumerate(groups):
        event_overlap = _overlap(interval, event_protection) > _TOL
        other_overlap = any(
            _overlap(interval, protected) > _TOL for protected in other_protections
        )
        if interval[1] <= event_protection[0] + _TOL:
            temporal_class = "pre_event"
        elif interval[0] >= event_protection[1] - _TOL:
            temporal_class = "future_course"
        else:
            temporal_class = "protected_event_neighborhood"
        blocked: list[str] = []
        if event_overlap:
            blocked.append("event_protection_overlap")
        if other_overlap:
            blocked.append("other_candidate_protection_overlap")
        if blocked or temporal_class == "protected_event_neighborhood":
            if not blocked:
                blocked.append("event_protection_overlap")
            audit_by_index[index] = _blocked_candidate(
                candidate_index=index,
                interval=interval,
                temporal_class=temporal_class,
                reasons=blocked,
            )
            continue
        if temporal_class == "future_course":
            future_groups.append((index, interval, rows))
            continue
        candidate = _evaluated_candidate(
            candidate_index=index,
            interval=interval,
            rows=rows,
            temporal_class=temporal_class,
            event_protection=event_protection,
            typed_roster=roster_by_key,
            gaps=gaps,
            measurement_policy=dense_measurement_opportunity.policy,
            policy=policy,
        )
        audit_by_index[index] = candidate
        if candidate["evaluation_status"] == "eligible":
            pre_evaluated.append(candidate)

    def preference(candidate: Mapping[str, object]) -> tuple[float, float, float]:
        interval = candidate["interval_seconds"]
        return (
            float(candidate["selection_score"]),
            float(interval[1]),
            -float(interval[0]),
        )

    local = max(pre_evaluated, key=preference) if pre_evaluated else None
    distant = None
    if local is not None:
        local_start = float(local["interval_seconds"][0])
        distant_candidates = [
            candidate
            for candidate in pre_evaluated
            if float(candidate["interval_seconds"][1])
            <= local_start - policy.distant_minimum_separation_seconds + _TOL
        ]
        if distant_candidates:
            distant = max(distant_candidates, key=preference)

    course = None
    future_rows_consulted = False
    if local is None and policy.allow_course_only_future_fallback:
        future_rows_consulted = True
        future_evaluated: list[dict[str, object]] = []
        for index, interval, rows in future_groups:
            candidate = _evaluated_candidate(
                candidate_index=index,
                interval=interval,
                rows=rows,
                temporal_class="future_course",
                event_protection=event_protection,
                typed_roster=roster_by_key,
                gaps=gaps,
                measurement_policy=dense_measurement_opportunity.policy,
                policy=policy,
            )
            audit_by_index[index] = candidate
            if candidate["evaluation_status"] == "eligible":
                future_evaluated.append(candidate)
        if future_evaluated:
            course = max(
                future_evaluated,
                key=lambda candidate: (
                    float(candidate["selection_score"]),
                    -float(candidate["interval_seconds"][0]),
                ),
            )
    else:
        for index, interval, _ in future_groups:
            audit_by_index[index] = _blocked_candidate(
                candidate_index=index,
                interval=interval,
                temporal_class="future_course",
                status="not_consulted",
                reasons=("pre_event_context_available_future_rows_not_consulted",),
            )

    if not policy.allow_course_only_future_fallback and local is None:
        for index, interval, _ in future_groups:
            audit_by_index[index] = _blocked_candidate(
                candidate_index=index,
                interval=interval,
                temporal_class="future_course",
                status="not_consulted",
                reasons=("course_only_future_fallback_disabled",),
            )

    selections = {
        "local_pre_event": (
            _selected_slot(local, temporal_role="matched_context_onset_facing_delta")
            if local is not None
            else _empty_slot(
                temporal_role="matched_context_onset_facing_delta",
                reasons=("no_common_qc_clean_pre_event_opportunity",),
            )
        ),
        "distant_pre_event": (
            _selected_slot(
                distant,
                temporal_role="matched_context_distant_pre_event_delta",
            )
            if distant is not None
            else _empty_slot(
                temporal_role="matched_context_distant_pre_event_delta",
                reasons=("no_common_qc_clean_distant_pre_event_opportunity",),
            )
        ),
        "course_only_future_context": (
            _selected_slot(course, temporal_role="matched_context_course_only")
            if course is not None
            else _empty_slot(
                temporal_role="matched_context_course_only",
                reasons=(
                    "future_context_not_needed_or_no_common_qc_clean_opportunity",
                ),
            )
        ),
    }
    selected_pre = [
        slot["interval_seconds"]
        for name, slot in selections.items()
        if name in {"local_pre_event", "distant_pre_event"}
        and slot["status"] == "selected"
    ]
    selected_pre.sort()
    payload: dict[str, Any] = {
        "schema_version": S03_MATCHED_CONTEXT_SELECTOR_SCHEMA_VERSION_V1,
        "method_id": S03_MATCHED_CONTEXT_SELECTOR_METHOD_ID_V1,
        "event_id": event_id,
        "policy": policy.to_dict(),
        "source": {
            "schema_version": BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
            "recording_id": dense_measurement_opportunity.recording_id,
            "canonical_signal_id": dense_measurement_opportunity.canonical_signal_id,
            "canonical_receipt_sha256": (
                dense_measurement_opportunity.canonical_receipt_sha256
            ),
            "source_signal_sha256": dense_measurement_opportunity.source_signal_sha256,
            "measurement_opportunity_sidecar_receipt_sha256": (
                dense_measurement_opportunity.receipt_sha256
            ),
            "measurement_opportunity_source_binding_sha256": (
                dense_measurement_opportunity.source_binding_sha256
            ),
            "measurement_policy_sha256": dense_measurement_opportunity.policy.sha256,
            "target_values_used_for_selection": False,
        },
        "event_support": {
            "analysis_interval_seconds": list(support),
            "event_interval_seconds": list(event_interval),
            "event_protection_interval_seconds": list(event_protection),
            "other_candidate_intervals_seconds": [list(item) for item in other_candidates],
            "other_candidate_protection_intervals_seconds": [
                list(item) for item in other_protections
            ],
            "signal_gap_intervals_seconds": [list(item) for item in gaps],
        },
        "typed_unit_roster": roster,
        "candidate_audit": [audit_by_index[index] for index in range(len(groups))],
        "selections": selections,
        "s03_comparison_context_intervals_seconds": selected_pre,
        "decision_semantics": {
            "common_interval_across_all_typed_units_required": True,
            "event_similarity_used_for_selection": False,
            "measurement_target_values_used_for_selection": False,
            "future_measurement_rows_consulted_for_selection": future_rows_consulted,
            "no_future_context_selected_for_onset_facing_delta": all(
                float(interval[1]) <= event_protection[0] + _TOL
                for interval in selected_pre
            ),
            "future_context_if_selected_is_course_only": True,
            "selector_only_defines_context_delta": True,
            "onset_support_permission": "forbidden",
            "direct_soz_score_authorized": False,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "decision_receipt_sha256": "",
        "receipt_sha256": "",
    }
    payload["decision_receipt_sha256"] = _canonical_sha256(
        _decision_hash_material(payload)
    )
    payload["receipt_sha256"] = _self_hash(payload, "receipt_sha256")
    return validate_s03_matched_context_selector_v1(payload)


def validate_s03_matched_context_selector_v1(value: object) -> dict[str, Any]:
    """Validate the strict disk contract without replaying its native source."""

    if type(value) is not dict:
        raise TypeError("S03 matched-context selector receipt must be an object")
    payload = deepcopy(value)
    expected_top = {
        "schema_version",
        "method_id",
        "event_id",
        "policy",
        "source",
        "event_support",
        "typed_unit_roster",
        "candidate_audit",
        "selections",
        "s03_comparison_context_intervals_seconds",
        "decision_semantics",
        "firewall",
        "authorization",
        "decision_receipt_sha256",
        "receipt_sha256",
    }
    if set(payload) != expected_top:
        raise ValueError("S03 matched-context selector top-level fields drifted")
    if payload["schema_version"] != S03_MATCHED_CONTEXT_SELECTOR_SCHEMA_VERSION_V1:
        raise ValueError("S03 selector schema version drifted")
    if payload["method_id"] != S03_MATCHED_CONTEXT_SELECTOR_METHOD_ID_V1:
        raise ValueError("S03 selector method drifted")
    _identifier(payload["event_id"], "event_id")
    policy = S03MatchedContextSelectorPolicyV1.from_dict(payload["policy"])
    source = payload["source"]
    if type(source) is not dict or set(source) != {
        "schema_version",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "measurement_opportunity_sidecar_receipt_sha256",
        "measurement_opportunity_source_binding_sha256",
        "measurement_policy_sha256",
        "target_values_used_for_selection",
    }:
        raise ValueError("S03 selector source fields drifted")
    if source["schema_version"] != BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION:
        raise ValueError("S03 selector source schema drifted")
    _identifier(source["recording_id"], "source.recording_id")
    _identifier(source["canonical_signal_id"], "source.canonical_signal_id")
    for name in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "measurement_opportunity_sidecar_receipt_sha256",
        "measurement_opportunity_source_binding_sha256",
        "measurement_policy_sha256",
    ):
        _sha(source[name], f"source.{name}")
    if source["target_values_used_for_selection"] is not False:
        raise ValueError("S03 selector cannot consume measurement target values")
    support_payload = payload["event_support"]
    if type(support_payload) is not dict or set(support_payload) != {
        "analysis_interval_seconds",
        "event_interval_seconds",
        "event_protection_interval_seconds",
        "other_candidate_intervals_seconds",
        "other_candidate_protection_intervals_seconds",
        "signal_gap_intervals_seconds",
    }:
        raise ValueError("S03 selector event-support fields drifted")
    support = _interval(support_payload["analysis_interval_seconds"], "analysis")
    event = _interval(support_payload["event_interval_seconds"], "event")
    protection = _interval(
        support_payload["event_protection_interval_seconds"], "event protection"
    )
    if not _contains(support, event) or not _contains(support, protection):
        raise ValueError("S03 selector event/protection lies outside support")
    expected_protection = _expanded(
        event,
        support=support,
        pre_seconds=policy.pre_event_protection_seconds,
        post_seconds=policy.post_event_protection_seconds,
    )
    if any(
        not math.isclose(observed, expected, abs_tol=_TOL)
        for observed, expected in zip(protection, expected_protection)
    ):
        raise ValueError("S03 selector event protection drifted")
    others = _normalise_intervals(
        support_payload["other_candidate_intervals_seconds"],
        name="other candidates",
        support=support,
    )
    expected_other_protections = [
        list(
            _expanded(
                item,
                support=support,
                pre_seconds=policy.pre_event_protection_seconds,
                post_seconds=policy.post_event_protection_seconds,
            )
        )
        for item in others
    ]
    if support_payload["other_candidate_protection_intervals_seconds"] != expected_other_protections:
        raise ValueError("S03 selector other-candidate protection drifted")
    _normalise_intervals(
        support_payload["signal_gap_intervals_seconds"],
        name="signal gaps",
        support=support,
    )
    roster = payload["typed_unit_roster"]
    if not isinstance(roster, list) or not roster:
        raise ValueError("S03 selector requires a typed-unit roster")
    roster_keys: list[tuple[int, int]] = []
    for unit in roster:
        if type(unit) is not dict or set(unit) != {
            "view_index",
            "unit_index",
            "view_id",
            "typed_unit_id",
            "typed_unit_kind",
            "source_unit_type",
            "reference_type",
            "reference_row_sha256",
            "reference_matrix_sha256",
            "canonical_source_channel_ids",
            "effective_bandwidth_hz",
            "quality_mask_sha256",
            "whole_output_unit_identity_preserved",
            "bipolar_endpoint_fact_projection_authorized",
        }:
            raise ValueError("S03 selector typed-unit roster fields drifted")
        key = (unit["view_index"], unit["unit_index"])
        roster_keys.append(key)
        _identifier(unit["view_id"], "typed unit view_id")
        _identifier(unit["typed_unit_id"], "typed unit id")
        _identifier(unit["reference_type"], "typed unit reference")
        if unit["typed_unit_kind"] not in {"electrode", "whole_bipolar_lead"}:
            raise ValueError("S03 selector typed-unit kind drifted")
        expected_source_type = (
            "lead" if unit["typed_unit_kind"] == "whole_bipolar_lead" else "electrode"
        )
        if unit["source_unit_type"] != expected_source_type:
            raise ValueError("S03 selector source/typed-unit kind mismatch")
        if (
            unit["whole_output_unit_identity_preserved"] is not True
            or unit["bipolar_endpoint_fact_projection_authorized"] is not False
        ):
            raise ValueError("S03 selector whole-lead permissions drifted")
        channels = unit["canonical_source_channel_ids"]
        if not isinstance(channels, list) or not channels:
            raise ValueError("typed unit must preserve canonical source channels")
        for name in (
            "reference_row_sha256",
            "reference_matrix_sha256",
            "quality_mask_sha256",
        ):
            _sha(unit[name], f"typed_unit.{name}")
        _interval(unit["effective_bandwidth_hz"], "typed-unit bandwidth")
    if roster_keys != sorted(roster_keys) or len(roster_keys) != len(set(roster_keys)):
        raise ValueError("S03 selector typed-unit roster order drifted")
    audits = payload["candidate_audit"]
    if not isinstance(audits, list) or not audits:
        raise ValueError("S03 selector requires candidate opportunity accounting")
    for expected_index, candidate in enumerate(audits):
        if type(candidate) is not dict or set(candidate) != {
            "candidate_index",
            "interval_seconds",
            "temporal_class",
            "evaluation_status",
            "score_components",
            "selection_score",
            "reason_codes",
            "contamination_and_protection",
            "opportunity",
            "decision_dependency_row_sha256s",
            "measurement_target_values_used",
        }:
            raise ValueError("S03 selector candidate-audit fields drifted")
        if candidate["candidate_index"] != expected_index:
            raise ValueError("S03 selector candidate indices drifted")
        _interval(candidate["interval_seconds"], "candidate interval")
        if candidate["temporal_class"] not in {
            "pre_event",
            "protected_event_neighborhood",
            "future_course",
        }:
            raise ValueError("S03 selector temporal class drifted")
        if candidate["evaluation_status"] not in {
            "eligible",
            "not_evaluable",
            "not_consulted",
        }:
            raise ValueError("S03 selector candidate status drifted")
        if candidate["measurement_target_values_used"] is not False:
            raise ValueError("S03 selector candidate consumed target values")
        for digest in candidate["decision_dependency_row_sha256s"]:
            _sha(digest, "candidate decision dependency")
        components = candidate["score_components"]
        if candidate["evaluation_status"] in {"eligible", "not_evaluable"} and components is not None:
            expected_component_keys = {
                "common_typed_unit_row_fraction",
                "spectral_usable_unit_fraction",
                "artifact_free_unit_fraction",
                "gap_free_fraction",
                "reference_match_fraction",
                "bandwidth_compatible_fraction",
                "temporal_distance_seconds",
                "temporal_proximity_score",
            }
            if type(components) is not dict or set(components) != expected_component_keys:
                raise ValueError("S03 selector score-component fields drifted")
            for name in expected_component_keys - {"temporal_distance_seconds"}:
                _fraction(components[name], f"score component {name}")
            _finite(
                components["temporal_distance_seconds"],
                "temporal distance",
                minimum=0.0,
            )
            expected_score = _selection_score(components, policy)
            if not math.isclose(
                float(candidate["selection_score"]),
                expected_score,
                abs_tol=_TOL,
            ):
                raise ValueError("S03 selector selection score drifted")
        elif components is not None or candidate["selection_score"] is not None:
            raise ValueError("unconsulted/blocked candidate carries a score")
    selections = payload["selections"]
    if type(selections) is not dict or set(selections) != {
        "local_pre_event",
        "distant_pre_event",
        "course_only_future_context",
    }:
        raise ValueError("S03 selector selection slots drifted")
    selected_pre: list[list[float]] = []
    for name, slot in selections.items():
        if type(slot) is not dict or set(slot) != {
            "status",
            "temporal_role",
            "interval_seconds",
            "candidate_index",
            "selection_score",
            "reason_codes",
            "future_context_selected",
            "onset_support_permission",
            "direct_soz_score_authorized",
        }:
            raise ValueError("S03 selector selection-slot fields drifted")
        if slot["status"] not in {"selected", "not_evaluable"}:
            raise ValueError("S03 selector selection-slot status drifted")
        if (
            slot["onset_support_permission"] != "forbidden"
            or slot["direct_soz_score_authorized"] is not False
        ):
            raise ValueError("S03 selector selection permissions drifted")
        if slot["status"] == "selected":
            index = slot["candidate_index"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(audits):
                raise ValueError("selected S03 context has an invalid candidate index")
            candidate = audits[index]
            if candidate["evaluation_status"] != "eligible":
                raise ValueError("selected S03 context is not eligible")
            if slot["interval_seconds"] != candidate["interval_seconds"]:
                raise ValueError("selected S03 context interval drifted")
            if not math.isclose(
                float(slot["selection_score"]),
                float(candidate["selection_score"]),
                abs_tol=_TOL,
            ):
                raise ValueError("selected S03 context score drifted")
            if name != "course_only_future_context":
                selected_pre.append(slot["interval_seconds"])
        elif any(
            slot[field] is not None
            for field in ("interval_seconds", "candidate_index", "selection_score")
        ):
            raise ValueError("unevaluable S03 context slot carries a selection")
    if sorted(selected_pre) != payload["s03_comparison_context_intervals_seconds"]:
        raise ValueError("S03 comparison-context projection drifted")
    semantics = payload["decision_semantics"]
    if type(semantics) is not dict or set(semantics) != {
        "common_interval_across_all_typed_units_required",
        "event_similarity_used_for_selection",
        "measurement_target_values_used_for_selection",
        "future_measurement_rows_consulted_for_selection",
        "no_future_context_selected_for_onset_facing_delta",
        "future_context_if_selected_is_course_only",
        "selector_only_defines_context_delta",
        "onset_support_permission",
        "direct_soz_score_authorized",
    }:
        raise ValueError("S03 selector decision semantics drifted")
    if semantics != {
        "common_interval_across_all_typed_units_required": True,
        "event_similarity_used_for_selection": False,
        "measurement_target_values_used_for_selection": False,
        "future_measurement_rows_consulted_for_selection": semantics[
            "future_measurement_rows_consulted_for_selection"
        ],
        "no_future_context_selected_for_onset_facing_delta": True,
        "future_context_if_selected_is_course_only": True,
        "selector_only_defines_context_delta": True,
        "onset_support_permission": "forbidden",
        "direct_soz_score_authorized": False,
    } or type(semantics["future_measurement_rows_consulted_for_selection"]) is not bool:
        raise ValueError("S03 selector decision permissions drifted")
    if payload["firewall"] != _FIREWALL or payload["authorization"] != _AUTHORIZATION:
        raise ValueError("S03 selector firewall/authorization drifted")
    _sha(payload["decision_receipt_sha256"], "decision_receipt_sha256")
    if payload["decision_receipt_sha256"] != _canonical_sha256(
        _decision_hash_material(payload)
    ):
        raise ValueError("S03 selector decision receipt drifted")
    _sha(payload["receipt_sha256"], "receipt_sha256")
    if payload["receipt_sha256"] != _self_hash(payload, "receipt_sha256"):
        raise ValueError("S03 selector content receipt drifted")
    return payload


def replay_s03_matched_context_selector_v1(
    expected: object,
    *,
    dense_measurement_opportunity: BAIEGDenseMeasurementSidecar,
) -> dict[str, Any]:
    """Rebuild the selector from its native opportunity ledger exactly."""

    payload = validate_s03_matched_context_selector_v1(expected)
    replayed = materialize_s03_matched_context_selector_v1(
        event_id=payload["event_id"],
        dense_measurement_opportunity=dense_measurement_opportunity,
        event_interval_seconds=payload["event_support"]["event_interval_seconds"],
        other_candidate_intervals_seconds=payload["event_support"][
            "other_candidate_intervals_seconds"
        ],
        signal_gap_intervals_seconds=payload["event_support"][
            "signal_gap_intervals_seconds"
        ],
        policy=S03MatchedContextSelectorPolicyV1.from_dict(payload["policy"]),
    )
    if replayed != payload:
        raise ValueError("S03 matched-context selector does not replay exactly")
    return replayed
