"""Fail-closed action selection for boundary-adaptive EEG fine analysis.

This module implements the *outer* controller of BA-IEG.  Whole-record EEG
navigation has already happened; the controller only decides which additional
physical interval should receive expensive fine analysis.  Candidate utility
predictions must come from frozen, patient-disjoint hidden-chunk
counterfactual supervision.  They are computational routing signals, never
clinical Findings or positive onset/SOZ evidence.

The contract deliberately separates:

* left/right event-context acquisition;
* distant-background retrieval;
* split/merge/boundary refinement actions;
* action utility and report authorization.

Offline future samples may help decide whether to acquire more context, but
every decision receipt explicitly denies permission to create or strengthen a
positive onset/SOZ claim.  Private labels, EDF annotations, spreadsheets,
doctor text, and clinical context are forbidden inputs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


BOUNDARY_ADAPTIVE_STATE_SCHEMA_VERSION = (
    "clinical_eeg_boundary_adaptive_fine_analysis_state_v1"
)
BOUNDARY_ADAPTIVE_POLICY_SCHEMA_VERSION = (
    "clinical_eeg_boundary_adaptive_fine_analysis_policy_v1"
)
BOUNDARY_ADAPTIVE_DECISION_SCHEMA_VERSION = (
    "clinical_eeg_boundary_adaptive_fine_analysis_decision_v1"
)
BOUNDARY_ADAPTIVE_METHOD_ID = (
    "counterfactual_utility_boundary_adaptive_fine_analysis_v1"
)

ACTION_TYPES = (
    "query_left",
    "query_right",
    "refine_boundary",
    "retrieve_distant_background",
    "split_review",
    "merge_review",
)

SIDE_STATES = (
    "open",
    "closed",
    "record_censored",
    "neighbor_censored",
    "quality_limited",
)

UTILITY_AUTHORITIES = (
    "hidden_chunk_counterfactual_oof",
    "frozen_source_dev_counterfactual_model",
    "oracle_reference_boundary_error_decomposition_only",
)

SELECTION_MODES = (
    "research_inference",
    "oracle_error_decomposition",
)

_STATE_KEYS = {
    "schema_version",
    "recording_id",
    "event_id",
    "step_index",
    "canonical_evidence_root_sha256",
    "detector_candidate_group_sha256",
    "recording_duration_seconds",
    "current_event_interval_recording_seconds",
    "retrieved_background_intervals_recording_seconds",
    "neighbor_protection_intervals_recording_seconds",
    "side_state",
    "boundary_posterior",
    "budget",
    "action_candidates",
    "scope_receipt",
}

_POLICY_KEYS = {
    "schema_version",
    "policy_id",
    "selection_mode",
    "weights",
    "minimum_action_utility",
    "utility_tie_tolerance",
    "maximum_single_action_eeg_seconds",
    "scope_receipt",
}

_ACTION_KEYS = {
    "action_id",
    "action_type",
    "side",
    "proposed_intervals_recording_seconds",
    "target_event_ids",
    "predicted_gain",
    "predicted_cost",
    "utility_prediction_receipt",
    "temporal_authority",
}

_GAIN_KEYS = {
    "onset_entropy_reduction_nats",
    "offset_entropy_reduction_nats",
    "earliest_field_stability_gain",
    "soz_rank_stability_gain",
    "finding_opportunity_gain",
}

_COST_KEYS = {
    "incremental_eeg_seconds",
    "incremental_gpu_seconds",
    "bad_quality_fraction",
    "neighbor_merge_risk",
}

_WEIGHT_KEYS = {
    "onset_entropy_reduction",
    "offset_entropy_reduction",
    "earliest_field_stability",
    "soz_rank_stability",
    "finding_opportunity",
    "eeg_seconds_cost",
    "gpu_seconds_cost",
    "bad_quality_fraction_cost",
    "neighbor_merge_risk_cost",
}

_SCOPE_KEYS = {
    "eeg_signal_only",
    "edf_annotations_used",
    "excel_used",
    "physician_labels_or_report_used",
    "clinical_context_used",
}

_TEMPORAL_AUTHORITY_KEYS = {
    "offline_future_context_may_control_acquisition",
    "may_authorize_positive_onset_or_soz_evidence",
    "may_create_report_eligible_finding",
}

_UTILITY_RECEIPT_KEYS = {
    "receipt_id",
    "authority",
    "model_or_rule_sha256",
    "training_data_manifest_sha256",
    "patient_disjoint_from_current_recording",
    "frozen_before_current_recording",
    "same_downstream_endpoint_used_for_hidden_chunk_targets",
    "current_record_reference_boundary_used",
    "private_evaluation_labels_used",
    "excel_annotation_doctor_text_or_clinical_context_used",
}

_BUDGET_KEYS = {
    "maximum_query_eeg_seconds",
    "used_query_eeg_seconds",
    "maximum_gpu_seconds",
    "used_gpu_seconds",
    "maximum_steps",
    "used_steps",
}

_BOUNDARY_POSTERIOR_KEYS = {
    "credible_interval_recording_seconds",
    "credible_mass",
    "entropy_nats",
    "touches_current_left_boundary",
    "touches_current_right_boundary",
    "censoring",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"{context} keys drifted; missing={missing}, extra={extra}")


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty identifier")
    if len(value) > 160 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} is not a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _unit_interval(value: object, context: str) -> float:
    result = _finite(value, context)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{context} must lie in [0,1]")
    return result


def _interval(
    value: object,
    context: str,
    *,
    lower: float,
    upper: float,
) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element list")
    start = _finite(value[0], f"{context}[0]")
    stop = _finite(value[1], f"{context}[1]")
    if start < lower - 1e-9 or stop > upper + 1e-9 or stop <= start:
        raise ValueError(f"{context} must be a positive interval within bounds")
    return [start, stop]


def _intervals(
    value: object,
    context: str,
    *,
    lower: float,
    upper: float,
) -> list[list[float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = [
        _interval(item, f"{context}[{index}]", lower=lower, upper=upper)
        for index, item in enumerate(value)
    ]
    ordered = sorted(result, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1] - 1e-9:
            raise ValueError(f"{context} intervals must not overlap")
    return ordered


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))


def _overlaps_any(interval: Sequence[float], others: Iterable[Sequence[float]]) -> bool:
    return any(_overlap(interval, other) > 1e-9 for other in others)


def _validate_scope(value: object, context: str) -> dict[str, bool]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    _exact_keys(value, _SCOPE_KEYS, context)
    for key in _SCOPE_KEYS:
        if not isinstance(value[key], bool):
            raise TypeError(f"{context}.{key} must be boolean")
    if value["eeg_signal_only"] is not True or any(
        value[key]
        for key in (
            "edf_annotations_used",
            "excel_used",
            "physician_labels_or_report_used",
            "clinical_context_used",
        )
    ):
        raise ValueError(f"{context} violates the EEG-only inference firewall")
    return dict(value)


def _validate_boundary_posterior(
    value: object,
    context: str,
    *,
    duration: float,
    current_interval: Sequence[float],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    _exact_keys(value, _BOUNDARY_POSTERIOR_KEYS, context)
    credible = _interval(
        value["credible_interval_recording_seconds"],
        f"{context}.credible_interval_recording_seconds",
        lower=0.0,
        upper=duration,
    )
    if credible[0] < current_interval[0] - 1e-9 or credible[1] > current_interval[1] + 1e-9:
        raise ValueError(f"{context} credible interval must lie inside current event interval")
    mass = _unit_interval(value["credible_mass"], f"{context}.credible_mass")
    entropy = _finite(value["entropy_nats"], f"{context}.entropy_nats", minimum=0.0)
    for key in (
        "touches_current_left_boundary",
        "touches_current_right_boundary",
    ):
        if not isinstance(value[key], bool):
            raise TypeError(f"{context}.{key} must be boolean")
    censoring = value["censoring"]
    if censoring not in {"none", "record_left", "record_right", "search_cap", "neighbor"}:
        raise ValueError(f"{context}.censoring is invalid")
    return {
        "credible_interval_recording_seconds": credible,
        "credible_mass": mass,
        "entropy_nats": entropy,
        "touches_current_left_boundary": value["touches_current_left_boundary"],
        "touches_current_right_boundary": value["touches_current_right_boundary"],
        "censoring": censoring,
    }


def _validate_utility_prediction_receipt(value: object, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    _exact_keys(value, _UTILITY_RECEIPT_KEYS, context)
    receipt = {
        "receipt_id": _identifier(value["receipt_id"], f"{context}.receipt_id"),
        "authority": value["authority"],
        "model_or_rule_sha256": _sha256(
            value["model_or_rule_sha256"], f"{context}.model_or_rule_sha256"
        ),
        "training_data_manifest_sha256": _sha256(
            value["training_data_manifest_sha256"],
            f"{context}.training_data_manifest_sha256",
        ),
    }
    if receipt["authority"] not in UTILITY_AUTHORITIES:
        raise ValueError(f"{context}.authority is invalid")
    for key in (
        "patient_disjoint_from_current_recording",
        "frozen_before_current_recording",
        "same_downstream_endpoint_used_for_hidden_chunk_targets",
        "current_record_reference_boundary_used",
        "private_evaluation_labels_used",
        "excel_annotation_doctor_text_or_clinical_context_used",
    ):
        if not isinstance(value[key], bool):
            raise TypeError(f"{context}.{key} must be boolean")
        receipt[key] = value[key]
    if receipt["patient_disjoint_from_current_recording"] is not True:
        raise ValueError(f"{context} must be patient-disjoint from the current recording")
    if receipt["frozen_before_current_recording"] is not True:
        raise ValueError(f"{context} must be frozen before the current recording")
    if receipt["same_downstream_endpoint_used_for_hidden_chunk_targets"] is not True:
        raise ValueError(f"{context} must use the frozen downstream endpoint")
    if receipt["private_evaluation_labels_used"] is not False:
        raise ValueError(f"{context} may not use private evaluation labels")
    if receipt["excel_annotation_doctor_text_or_clinical_context_used"] is not False:
        raise ValueError(f"{context} violates the EEG-only firewall")
    oracle = receipt["authority"] == "oracle_reference_boundary_error_decomposition_only"
    if receipt["current_record_reference_boundary_used"] is not oracle:
        raise ValueError(
            f"{context}.current_record_reference_boundary_used must identify oracle-only actions"
        )
    return receipt


def _validate_action(
    value: object,
    index: int,
    *,
    duration: float,
    current_interval: Sequence[float],
    backgrounds: Sequence[Sequence[float]],
    neighbor_protection: Sequence[Sequence[float]],
) -> dict[str, Any]:
    context = f"action_candidates[{index}]"
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    _exact_keys(value, _ACTION_KEYS, context)
    action_id = _identifier(value["action_id"], f"{context}.action_id")
    action_type = value["action_type"]
    if action_type not in ACTION_TYPES:
        raise ValueError(f"{context}.action_type is invalid")
    side = value["side"]
    expected_side = {
        "query_left": {"left"},
        "query_right": {"right"},
        "refine_boundary": {"left", "right"},
        "retrieve_distant_background": {"none"},
        "split_review": {"none"},
        "merge_review": {"none"},
    }[action_type]
    if side not in expected_side:
        raise ValueError(f"{context}.side is invalid for {action_type}")
    intervals = _intervals(
        value["proposed_intervals_recording_seconds"],
        f"{context}.proposed_intervals_recording_seconds",
        lower=0.0,
        upper=duration,
    )
    target_ids_raw = value["target_event_ids"]
    if not isinstance(target_ids_raw, list):
        raise TypeError(f"{context}.target_event_ids must be a list")
    target_ids = [
        _identifier(item, f"{context}.target_event_ids[{target_index}]")
        for target_index, item in enumerate(target_ids_raw)
    ]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError(f"{context}.target_event_ids must be unique")

    if action_type == "query_left":
        if len(intervals) != 1 or abs(intervals[0][1] - current_interval[0]) > 1e-6:
            raise ValueError(f"{context} must extend contiguously from the current left edge")
    elif action_type == "query_right":
        if len(intervals) != 1 or abs(intervals[0][0] - current_interval[1]) > 1e-6:
            raise ValueError(f"{context} must extend contiguously from the current right edge")
    elif action_type == "retrieve_distant_background":
        if not intervals:
            raise ValueError(f"{context} must retrieve at least one physical interval")
        for interval in intervals:
            if _overlap(interval, current_interval) > 1e-9 or _overlaps_any(interval, backgrounds):
                raise ValueError(f"{context} distant background overlaps already queried evidence")
    elif intervals:
        raise ValueError(f"{context} may not acquire intervals for {action_type}")

    for interval in intervals:
        if _overlaps_any(interval, neighbor_protection):
            raise ValueError(f"{context} overlaps a neighboring-event protection interval")

    gain_raw = value["predicted_gain"]
    if type(gain_raw) is not dict:
        raise TypeError(f"{context}.predicted_gain must be an object")
    _exact_keys(gain_raw, _GAIN_KEYS, f"{context}.predicted_gain")
    gain = {
        "onset_entropy_reduction_nats": _finite(
            gain_raw["onset_entropy_reduction_nats"],
            f"{context}.predicted_gain.onset_entropy_reduction_nats",
            minimum=0.0,
        ),
        "offset_entropy_reduction_nats": _finite(
            gain_raw["offset_entropy_reduction_nats"],
            f"{context}.predicted_gain.offset_entropy_reduction_nats",
            minimum=0.0,
        ),
        "earliest_field_stability_gain": _unit_interval(
            gain_raw["earliest_field_stability_gain"],
            f"{context}.predicted_gain.earliest_field_stability_gain",
        ),
        "soz_rank_stability_gain": _unit_interval(
            gain_raw["soz_rank_stability_gain"],
            f"{context}.predicted_gain.soz_rank_stability_gain",
        ),
        "finding_opportunity_gain": _unit_interval(
            gain_raw["finding_opportunity_gain"],
            f"{context}.predicted_gain.finding_opportunity_gain",
        ),
    }

    cost_raw = value["predicted_cost"]
    if type(cost_raw) is not dict:
        raise TypeError(f"{context}.predicted_cost must be an object")
    _exact_keys(cost_raw, _COST_KEYS, f"{context}.predicted_cost")
    computed_eeg_seconds = sum(stop - start for start, stop in intervals)
    declared_eeg_seconds = _finite(
        cost_raw["incremental_eeg_seconds"],
        f"{context}.predicted_cost.incremental_eeg_seconds",
        minimum=0.0,
    )
    if abs(computed_eeg_seconds - declared_eeg_seconds) > 1e-6:
        raise ValueError(f"{context} incremental EEG cost does not match physical intervals")
    cost = {
        "incremental_eeg_seconds": computed_eeg_seconds,
        "incremental_gpu_seconds": _finite(
            cost_raw["incremental_gpu_seconds"],
            f"{context}.predicted_cost.incremental_gpu_seconds",
            minimum=0.0,
        ),
        "bad_quality_fraction": _unit_interval(
            cost_raw["bad_quality_fraction"],
            f"{context}.predicted_cost.bad_quality_fraction",
        ),
        "neighbor_merge_risk": _unit_interval(
            cost_raw["neighbor_merge_risk"],
            f"{context}.predicted_cost.neighbor_merge_risk",
        ),
    }

    temporal_raw = value["temporal_authority"]
    if type(temporal_raw) is not dict:
        raise TypeError(f"{context}.temporal_authority must be an object")
    _exact_keys(temporal_raw, _TEMPORAL_AUTHORITY_KEYS, f"{context}.temporal_authority")
    for key in _TEMPORAL_AUTHORITY_KEYS:
        if not isinstance(temporal_raw[key], bool):
            raise TypeError(f"{context}.temporal_authority.{key} must be boolean")
    if temporal_raw["may_authorize_positive_onset_or_soz_evidence"] is not False:
        raise ValueError(f"{context} may not authorize positive onset/SOZ evidence")
    if temporal_raw["may_create_report_eligible_finding"] is not False:
        raise ValueError(f"{context} may not create report-eligible Findings")

    return {
        "action_id": action_id,
        "action_type": action_type,
        "side": side,
        "proposed_intervals_recording_seconds": intervals,
        "target_event_ids": target_ids,
        "predicted_gain": gain,
        "predicted_cost": cost,
        "utility_prediction_receipt": _validate_utility_prediction_receipt(
            value["utility_prediction_receipt"],
            f"{context}.utility_prediction_receipt",
        ),
        "temporal_authority": dict(temporal_raw),
    }


def validate_boundary_adaptive_fine_analysis_state(payload: object) -> dict[str, Any]:
    """Validate one event-specific controller state without selecting an action."""

    if type(payload) is not dict:
        raise TypeError("boundary-adaptive state must be an object")
    _exact_keys(payload, _STATE_KEYS, "state")
    if payload["schema_version"] != BOUNDARY_ADAPTIVE_STATE_SCHEMA_VERSION:
        raise ValueError("boundary-adaptive state schema version drifted")
    recording_id = _identifier(payload["recording_id"], "state.recording_id")
    event_id = _identifier(payload["event_id"], "state.event_id")
    step_index = _integer(payload["step_index"], "state.step_index")
    duration = _finite(
        payload["recording_duration_seconds"],
        "state.recording_duration_seconds",
        minimum=1e-9,
    )
    current_interval = _interval(
        payload["current_event_interval_recording_seconds"],
        "state.current_event_interval_recording_seconds",
        lower=0.0,
        upper=duration,
    )
    backgrounds = _intervals(
        payload["retrieved_background_intervals_recording_seconds"],
        "state.retrieved_background_intervals_recording_seconds",
        lower=0.0,
        upper=duration,
    )
    neighbors = _intervals(
        payload["neighbor_protection_intervals_recording_seconds"],
        "state.neighbor_protection_intervals_recording_seconds",
        lower=0.0,
        upper=duration,
    )
    if any(_overlap(item, current_interval) > 1e-9 for item in backgrounds):
        raise ValueError("retrieved background may not overlap the current event interval")
    if any(_overlaps_any(item, neighbors) for item in backgrounds):
        raise ValueError("retrieved background may not overlap neighboring-event protection")

    side_raw = payload["side_state"]
    if type(side_raw) is not dict or set(side_raw) != {"left", "right"}:
        raise ValueError("state.side_state must contain exactly left and right")
    side_state: dict[str, str] = {}
    for side in ("left", "right"):
        value = side_raw[side]
        if value not in SIDE_STATES:
            raise ValueError(f"state.side_state.{side} is invalid")
        side_state[side] = value

    posterior_raw = payload["boundary_posterior"]
    if type(posterior_raw) is not dict or set(posterior_raw) != {"onset", "offset"}:
        raise ValueError("state.boundary_posterior must contain exactly onset and offset")
    posterior = {
        boundary: _validate_boundary_posterior(
            posterior_raw[boundary],
            f"state.boundary_posterior.{boundary}",
            duration=duration,
            current_interval=current_interval,
        )
        for boundary in ("onset", "offset")
    }

    budget_raw = payload["budget"]
    if type(budget_raw) is not dict:
        raise TypeError("state.budget must be an object")
    _exact_keys(budget_raw, _BUDGET_KEYS, "state.budget")
    budget = {
        "maximum_query_eeg_seconds": _finite(
            budget_raw["maximum_query_eeg_seconds"],
            "state.budget.maximum_query_eeg_seconds",
            minimum=0.0,
        ),
        "used_query_eeg_seconds": _finite(
            budget_raw["used_query_eeg_seconds"],
            "state.budget.used_query_eeg_seconds",
            minimum=0.0,
        ),
        "maximum_gpu_seconds": _finite(
            budget_raw["maximum_gpu_seconds"],
            "state.budget.maximum_gpu_seconds",
            minimum=0.0,
        ),
        "used_gpu_seconds": _finite(
            budget_raw["used_gpu_seconds"],
            "state.budget.used_gpu_seconds",
            minimum=0.0,
        ),
        "maximum_steps": _integer(
            budget_raw["maximum_steps"], "state.budget.maximum_steps", minimum=1
        ),
        "used_steps": _integer(
            budget_raw["used_steps"], "state.budget.used_steps"
        ),
    }
    if budget["used_query_eeg_seconds"] > budget["maximum_query_eeg_seconds"] + 1e-9:
        raise ValueError("used EEG budget exceeds maximum")
    if budget["used_gpu_seconds"] > budget["maximum_gpu_seconds"] + 1e-9:
        raise ValueError("used GPU budget exceeds maximum")
    if budget["used_steps"] > budget["maximum_steps"]:
        raise ValueError("used step budget exceeds maximum")
    if step_index != budget["used_steps"]:
        raise ValueError("state.step_index must equal budget.used_steps")

    candidates_raw = payload["action_candidates"]
    if not isinstance(candidates_raw, list):
        raise TypeError("state.action_candidates must be a list")
    candidates = [
        _validate_action(
            value,
            index,
            duration=duration,
            current_interval=current_interval,
            backgrounds=backgrounds,
            neighbor_protection=neighbors,
        )
        for index, value in enumerate(candidates_raw)
    ]
    action_ids = [item["action_id"] for item in candidates]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("state.action_candidates action_id values must be unique")
    # A candidate roster is a set of possible actions, not a priority list.
    # Canonicalize it before hashing so caller serialization order cannot
    # change the selected action or the decision content address.
    candidates.sort(key=lambda item: item["action_id"])

    return {
        "schema_version": BOUNDARY_ADAPTIVE_STATE_SCHEMA_VERSION,
        "recording_id": recording_id,
        "event_id": event_id,
        "step_index": step_index,
        "canonical_evidence_root_sha256": _sha256(
            payload["canonical_evidence_root_sha256"],
            "state.canonical_evidence_root_sha256",
        ),
        "detector_candidate_group_sha256": _sha256(
            payload["detector_candidate_group_sha256"],
            "state.detector_candidate_group_sha256",
        ),
        "recording_duration_seconds": duration,
        "current_event_interval_recording_seconds": current_interval,
        "retrieved_background_intervals_recording_seconds": backgrounds,
        "neighbor_protection_intervals_recording_seconds": neighbors,
        "side_state": side_state,
        "boundary_posterior": posterior,
        "budget": budget,
        "action_candidates": candidates,
        "scope_receipt": _validate_scope(payload["scope_receipt"], "state.scope_receipt"),
    }


def validate_boundary_adaptive_fine_analysis_policy(payload: object) -> dict[str, Any]:
    """Validate utility weights and fixed-compute selection rules."""

    if type(payload) is not dict:
        raise TypeError("boundary-adaptive policy must be an object")
    _exact_keys(payload, _POLICY_KEYS, "policy")
    if payload["schema_version"] != BOUNDARY_ADAPTIVE_POLICY_SCHEMA_VERSION:
        raise ValueError("boundary-adaptive policy schema version drifted")
    selection_mode = payload["selection_mode"]
    if selection_mode not in SELECTION_MODES:
        raise ValueError("policy.selection_mode is invalid")
    weights_raw = payload["weights"]
    if type(weights_raw) is not dict:
        raise TypeError("policy.weights must be an object")
    _exact_keys(weights_raw, _WEIGHT_KEYS, "policy.weights")
    weights = {
        key: _finite(weights_raw[key], f"policy.weights.{key}", minimum=0.0)
        for key in sorted(_WEIGHT_KEYS)
    }
    if not any(weights[key] > 0.0 for key in _WEIGHT_KEYS):
        raise ValueError("policy must assign positive weight to at least one utility component")
    minimum_action_utility = _finite(
        payload["minimum_action_utility"], "policy.minimum_action_utility"
    )
    tie_tolerance = _finite(
        payload["utility_tie_tolerance"],
        "policy.utility_tie_tolerance",
        minimum=0.0,
    )
    maximum_single = _finite(
        payload["maximum_single_action_eeg_seconds"],
        "policy.maximum_single_action_eeg_seconds",
        minimum=0.0,
    )
    return {
        "schema_version": BOUNDARY_ADAPTIVE_POLICY_SCHEMA_VERSION,
        "policy_id": _identifier(payload["policy_id"], "policy.policy_id"),
        "selection_mode": selection_mode,
        "weights": weights,
        "minimum_action_utility": minimum_action_utility,
        "utility_tie_tolerance": tie_tolerance,
        "maximum_single_action_eeg_seconds": maximum_single,
        "scope_receipt": _validate_scope(payload["scope_receipt"], "policy.scope_receipt"),
    }


def _utility(action: Mapping[str, Any], weights: Mapping[str, float]) -> tuple[dict[str, float], float]:
    gain = action["predicted_gain"]
    cost = action["predicted_cost"]
    components = {
        "onset_entropy_reduction": weights["onset_entropy_reduction"]
        * gain["onset_entropy_reduction_nats"],
        "offset_entropy_reduction": weights["offset_entropy_reduction"]
        * gain["offset_entropy_reduction_nats"],
        "earliest_field_stability": weights["earliest_field_stability"]
        * gain["earliest_field_stability_gain"],
        "soz_rank_stability": weights["soz_rank_stability"]
        * gain["soz_rank_stability_gain"],
        "finding_opportunity": weights["finding_opportunity"]
        * gain["finding_opportunity_gain"],
        "eeg_seconds_cost": -weights["eeg_seconds_cost"]
        * cost["incremental_eeg_seconds"],
        "gpu_seconds_cost": -weights["gpu_seconds_cost"]
        * cost["incremental_gpu_seconds"],
        "bad_quality_fraction_cost": -weights["bad_quality_fraction_cost"]
        * cost["bad_quality_fraction"],
        "neighbor_merge_risk_cost": -weights["neighbor_merge_risk_cost"]
        * cost["neighbor_merge_risk"],
    }
    return components, sum(components.values())


def _candidate_exclusion_reasons(
    action: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    authority = action["utility_prediction_receipt"]["authority"]
    oracle = authority == "oracle_reference_boundary_error_decomposition_only"
    if policy["selection_mode"] == "research_inference" and oracle:
        reasons.append("oracle_authority_forbidden_in_research_inference")
    if policy["selection_mode"] == "oracle_error_decomposition" and not oracle:
        reasons.append("non_oracle_action_excluded_from_oracle_arm")
    if action["side"] in {"left", "right"} and state["side_state"][action["side"]] != "open":
        reasons.append(f"{action['side']}_side_not_open")
    budget = state["budget"]
    cost = action["predicted_cost"]
    if budget["used_steps"] >= budget["maximum_steps"]:
        reasons.append("step_budget_exhausted")
    if (
        budget["used_query_eeg_seconds"] + cost["incremental_eeg_seconds"]
        > budget["maximum_query_eeg_seconds"] + 1e-9
    ):
        reasons.append("eeg_second_budget_exceeded")
    if (
        budget["used_gpu_seconds"] + cost["incremental_gpu_seconds"]
        > budget["maximum_gpu_seconds"] + 1e-9
    ):
        reasons.append("gpu_budget_exceeded")
    if (
        cost["incremental_eeg_seconds"]
        > policy["maximum_single_action_eeg_seconds"] + 1e-9
    ):
        reasons.append("single_action_eeg_second_cap_exceeded")
    return sorted(set(reasons))


def select_boundary_adaptive_fine_analysis_action(
    state_payload: object,
    policy_payload: object,
) -> dict[str, Any]:
    """Select one qualified action or a typed stop/fallback outcome.

    Candidate order is irrelevant.  Ties within ``utility_tie_tolerance`` are
    resolved by lower EEG cost, lower GPU cost, action type, then action ID.
    The receipt always denies clinical/report authorization.
    """

    state = validate_boundary_adaptive_fine_analysis_state(state_payload)
    policy = validate_boundary_adaptive_fine_analysis_policy(policy_payload)
    evaluations: list[dict[str, Any]] = []
    for action in state["action_candidates"]:
        components, utility = _utility(action, policy["weights"])
        reasons = _candidate_exclusion_reasons(action, state=state, policy=policy)
        evaluations.append(
            {
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "side": action["side"],
                "utility": utility,
                "utility_components": components,
                "incremental_eeg_seconds": action["predicted_cost"]["incremental_eeg_seconds"],
                "incremental_gpu_seconds": action["predicted_cost"]["incremental_gpu_seconds"],
                "eligible": not reasons,
                "exclusion_reason_codes": reasons,
                "utility_prediction_receipt_id": action["utility_prediction_receipt"]["receipt_id"],
            }
        )
    evaluations.sort(key=lambda item: item["action_id"])

    eligible = [item for item in evaluations if item["eligible"]]
    selected: dict[str, Any] | None = None
    if eligible:
        best_utility = max(item["utility"] for item in eligible)
        tied = [
            item
            for item in eligible
            if best_utility - item["utility"] <= policy["utility_tie_tolerance"]
        ]
        tied.sort(
            key=lambda item: (
                item["incremental_eeg_seconds"],
                item["incremental_gpu_seconds"],
                item["action_type"],
                item["action_id"],
            )
        )
        candidate = tied[0]
        if candidate["utility"] > policy["minimum_action_utility"]:
            selected = candidate

    if selected is not None:
        status = "selected_action"
        selected_action_id: str | None = selected["action_id"]
        selected_action_type = selected["action_type"]
        selected_utility: float | None = selected["utility"]
        stop_reasons: list[str] = []
    elif not state["action_candidates"]:
        status = "fallback_required"
        selected_action_id = None
        selected_action_type = "stop"
        selected_utility = None
        stop_reasons = ["no_action_candidates_materialized"]
    elif not eligible and all(
        any("oracle" in reason for reason in item["exclusion_reason_codes"])
        for item in evaluations
    ):
        status = "fallback_required"
        selected_action_id = None
        selected_action_type = "stop"
        selected_utility = None
        stop_reasons = ["no_action_candidate_with_authorized_utility_scope"]
    else:
        status = "stop"
        selected_action_id = None
        selected_action_type = "stop"
        selected_utility = None
        stop_reasons = []
        if not eligible:
            stop_reasons.extend(
                sorted(
                    {
                        reason
                        for item in evaluations
                        for reason in item["exclusion_reason_codes"]
                    }
                )
            )
        else:
            stop_reasons.append("no_positive_qualified_marginal_utility")
        if state["side_state"]["left"] != "open" and state["side_state"]["right"] != "open":
            stop_reasons.append("both_event_context_sides_closed_or_censored")
        stop_reasons = sorted(set(stop_reasons))

    body: dict[str, Any] = {
        "schema_version": BOUNDARY_ADAPTIVE_DECISION_SCHEMA_VERSION,
        "decision_id": "CONTENT-ADDRESS-PENDING",
        "method_id": BOUNDARY_ADAPTIVE_METHOD_ID,
        "recording_id": state["recording_id"],
        "event_id": state["event_id"],
        "step_index": state["step_index"],
        "state_sha256": _sha256_payload(state),
        "policy_sha256": _sha256_payload(policy),
        "status": status,
        "selected_action_id": selected_action_id,
        "selected_action_type": selected_action_type,
        "selected_utility": selected_utility,
        "candidate_evaluations": evaluations,
        "stop_reason_codes": stop_reasons,
        "authorization": {
            "computational_routing_only": True,
            "may_authorize_positive_onset_or_soz_evidence": False,
            "may_create_report_eligible_finding": False,
            "may_create_or_strengthen_report_claim": False,
            "future_context_role": "acquisition_control_or_counterevidence_only",
        },
        "scope_receipt": deepcopy(state["scope_receipt"]),
    }
    body["decision_id"] = "BAFA-" + _sha256_payload(body)[:24]
    return validate_boundary_adaptive_fine_analysis_decision(body)


def validate_boundary_adaptive_fine_analysis_decision(payload: object) -> dict[str, Any]:
    """Validate an action-decision receipt and its content address."""

    if type(payload) is not dict:
        raise TypeError("boundary-adaptive decision must be an object")
    expected_keys = {
        "schema_version",
        "decision_id",
        "method_id",
        "recording_id",
        "event_id",
        "step_index",
        "state_sha256",
        "policy_sha256",
        "status",
        "selected_action_id",
        "selected_action_type",
        "selected_utility",
        "candidate_evaluations",
        "stop_reason_codes",
        "authorization",
        "scope_receipt",
    }
    _exact_keys(payload, expected_keys, "decision")
    if payload["schema_version"] != BOUNDARY_ADAPTIVE_DECISION_SCHEMA_VERSION:
        raise ValueError("boundary-adaptive decision schema version drifted")
    if payload["method_id"] != BOUNDARY_ADAPTIVE_METHOD_ID:
        raise ValueError("boundary-adaptive decision method drifted")
    _identifier(payload["recording_id"], "decision.recording_id")
    _identifier(payload["event_id"], "decision.event_id")
    _integer(payload["step_index"], "decision.step_index")
    _sha256(payload["state_sha256"], "decision.state_sha256")
    _sha256(payload["policy_sha256"], "decision.policy_sha256")
    if payload["status"] not in {"selected_action", "stop", "fallback_required"}:
        raise ValueError("decision.status is invalid")
    if not isinstance(payload["candidate_evaluations"], list):
        raise TypeError("decision.candidate_evaluations must be a list")
    evaluations = payload["candidate_evaluations"]
    action_ids: list[str] = []
    for index, item in enumerate(evaluations):
        context = f"decision.candidate_evaluations[{index}]"
        if type(item) is not dict:
            raise TypeError(f"{context} must be an object")
        expected_eval = {
            "action_id",
            "action_type",
            "side",
            "utility",
            "utility_components",
            "incremental_eeg_seconds",
            "incremental_gpu_seconds",
            "eligible",
            "exclusion_reason_codes",
            "utility_prediction_receipt_id",
        }
        _exact_keys(item, expected_eval, context)
        action_ids.append(_identifier(item["action_id"], f"{context}.action_id"))
        if item["action_type"] not in ACTION_TYPES:
            raise ValueError(f"{context}.action_type is invalid")
        if item["side"] not in {"left", "right", "none"}:
            raise ValueError(f"{context}.side is invalid")
        _finite(item["utility"], f"{context}.utility")
        if type(item["utility_components"]) is not dict or set(item["utility_components"]) != _WEIGHT_KEYS:
            raise ValueError(f"{context}.utility_components keys drifted")
        components = [
            _finite(value, f"{context}.utility_components.{key}")
            for key, value in item["utility_components"].items()
        ]
        if abs(sum(components) - float(item["utility"])) > 1e-8:
            raise ValueError(f"{context}.utility does not equal component sum")
        _finite(item["incremental_eeg_seconds"], f"{context}.incremental_eeg_seconds", minimum=0.0)
        _finite(item["incremental_gpu_seconds"], f"{context}.incremental_gpu_seconds", minimum=0.0)
        if not isinstance(item["eligible"], bool):
            raise TypeError(f"{context}.eligible must be boolean")
        if not isinstance(item["exclusion_reason_codes"], list) or not all(
            isinstance(reason, str) and reason for reason in item["exclusion_reason_codes"]
        ):
            raise TypeError(f"{context}.exclusion_reason_codes must contain strings")
        _identifier(
            item["utility_prediction_receipt_id"],
            f"{context}.utility_prediction_receipt_id",
        )
    if action_ids != sorted(action_ids) or len(action_ids) != len(set(action_ids)):
        raise ValueError("decision candidate evaluations must be uniquely sorted by action_id")
    selected_id = payload["selected_action_id"]
    selected_type = payload["selected_action_type"]
    selected_utility = payload["selected_utility"]
    if payload["status"] == "selected_action":
        if selected_id not in action_ids or selected_type not in ACTION_TYPES:
            raise ValueError("selected decision must reference a known action")
        match = next(item for item in evaluations if item["action_id"] == selected_id)
        if not match["eligible"] or match["action_type"] != selected_type:
            raise ValueError("selected action is not eligible or type-consistent")
        if abs(_finite(selected_utility, "decision.selected_utility") - match["utility"]) > 1e-8:
            raise ValueError("selected utility does not match candidate evaluation")
        if payload["stop_reason_codes"] != []:
            raise ValueError("selected action may not carry stop reasons")
    else:
        if selected_id is not None or selected_type != "stop" or selected_utility is not None:
            raise ValueError("stop/fallback decision may not select an action")
        if not isinstance(payload["stop_reason_codes"], list) or not payload["stop_reason_codes"]:
            raise ValueError("stop/fallback decision requires reason codes")

    authorization = payload["authorization"]
    expected_authorization = {
        "computational_routing_only": True,
        "may_authorize_positive_onset_or_soz_evidence": False,
        "may_create_report_eligible_finding": False,
        "may_create_or_strengthen_report_claim": False,
        "future_context_role": "acquisition_control_or_counterevidence_only",
    }
    if authorization != expected_authorization:
        raise ValueError("decision clinical/report authorization drifted")
    _validate_scope(payload["scope_receipt"], "decision.scope_receipt")

    decision_id = _identifier(payload["decision_id"], "decision.decision_id")
    body = deepcopy(payload)
    body["decision_id"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "BAFA-" + _sha256_payload(body)[:24]
    if decision_id != expected_id:
        raise ValueError("boundary-adaptive decision content address drifted")
    return deepcopy(payload)


def replay_boundary_adaptive_fine_analysis_decision(
    state_payload: object,
    policy_payload: object,
    decision_payload: object,
) -> dict[str, Any]:
    """Recompute and byte-logically compare a decision receipt."""

    observed = validate_boundary_adaptive_fine_analysis_decision(decision_payload)
    expected = select_boundary_adaptive_fine_analysis_action(state_payload, policy_payload)
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError("boundary-adaptive decision does not replay from state and policy")
    return observed


__all__ = [
    "ACTION_TYPES",
    "BOUNDARY_ADAPTIVE_DECISION_SCHEMA_VERSION",
    "BOUNDARY_ADAPTIVE_METHOD_ID",
    "BOUNDARY_ADAPTIVE_POLICY_SCHEMA_VERSION",
    "BOUNDARY_ADAPTIVE_STATE_SCHEMA_VERSION",
    "SELECTION_MODES",
    "SIDE_STATES",
    "UTILITY_AUTHORITIES",
    "replay_boundary_adaptive_fine_analysis_decision",
    "select_boundary_adaptive_fine_analysis_action",
    "validate_boundary_adaptive_fine_analysis_decision",
    "validate_boundary_adaptive_fine_analysis_policy",
    "validate_boundary_adaptive_fine_analysis_state",
]
