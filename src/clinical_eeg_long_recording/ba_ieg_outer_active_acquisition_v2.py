"""Trainable outer active-acquisition bridge for BA-IEG v2.

The repository already has two important, but previously disconnected, pieces:

* replayable hidden-chunk counterfactual targets; and
* a fail-closed boundary-adaptive action selector whose ``predicted_gain``
  values must be supplied by another component.

This module closes that gap as a contextual, fixed-budget acquisition policy.
For each detector-frozen event state it:

1. materialises a state-action feature vector from information available
   *before* the proposed EEG interval is revealed;
2. learns positive benefit magnitude, negative harm magnitude, and harm
   probability for the five frozen downstream endpoints;
3. ranks mutually exclusive left/right/background actions by expected signed
   counterfactual utility minus physical/compute/contamination costs; and
4. evaluates the learned policy on source-dev action sets under exactly the
   same per-decision EEG-second and GPU-second budget as deterministic
   comparators.

This is deliberately a contextual-bandit closure, not yet a claim that a full
multi-step public EEG rollout has been trained.  A later state must be
materialised after the selected chunk is actually revealed; additive rewards
from alternative hidden chunks are never treated as a synthetic trajectory.

Only EEG-derived state, detector navigation, quality estimates, and frozen
endpoint snapshots are accepted.  The revealed chunk and its endpoint delta
are targets only.  EDF annotations, spreadsheets, doctor labels/reports,
clinical context, and private data have no input field.  Acquisition utility
cannot authorise a clinical Finding, onset/SOZ evidence, or report claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as torch_functional

from .ba_ieg_counterfactual_utility_supervision_v1 import (
    BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES,
    BA_IEG_COUNTERFACTUAL_GAIN_NAMES,
    validate_hidden_chunk_counterfactual_target_v1,
)
from .boundary_adaptive_fine_analysis import (
    validate_boundary_adaptive_fine_analysis_state,
)


BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2: Final[str] = (
    "ba_ieg_boundary_posterior_outer_active_acquisition_v2"
)
BA_IEG_OUTER_ACTIVE_ACQUISITION_EXAMPLE_SCHEMA_VERSION_V2: Final[str] = (
    "ba_ieg_outer_active_acquisition_example_v2"
)
BA_IEG_OUTER_ACTIVE_ACQUISITION_EVALUATION_SCHEMA_VERSION_V2: Final[str] = (
    "ba_ieg_outer_active_acquisition_fixed_budget_evaluation_v2"
)

BA_IEG_OUTER_ACTION_TYPES_V2: Final[tuple[str, ...]] = (
    "query_left",
    "query_right",
    "retrieve_distant_background",
)
BA_IEG_OUTER_COST_NAMES_V2: Final[tuple[str, ...]] = (
    "incremental_eeg_seconds",
    "incremental_gpu_seconds",
    "bad_quality_fraction",
    "neighbor_merge_risk",
)

# Every feature is available before the hidden chunk is revealed.  Ratios use
# physical seconds and the current controller budget; no learned fold statistic
# or target-derived normalisation is embedded in this representation.
BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2: Final[tuple[str, ...]] = (
    "action_is_query_left",
    "action_is_query_right",
    "action_is_distant_background",
    "action_duration_over_current_duration",
    "action_signed_center_distance_over_current_duration",
    "action_gap_over_current_duration",
    "action_gpu_over_remaining_gpu_budget",
    "action_bad_quality_fraction",
    "action_neighbor_merge_risk",
    "step_used_fraction",
    "eeg_budget_remaining_fraction",
    "gpu_budget_remaining_fraction",
    "current_duration_over_recording_duration",
    "retrieved_background_seconds_over_recording_duration",
    "retrieved_background_count_log1p",
    "left_side_open",
    "right_side_open",
    "onset_entropy_log1p",
    "onset_credible_mass",
    "onset_credible_width_over_current_duration",
    "onset_credible_center_fraction",
    "onset_touches_current_left_boundary",
    "onset_touches_current_right_boundary",
    "onset_is_censored",
    "offset_entropy_log1p",
    "offset_credible_mass",
    "offset_credible_width_over_current_duration",
    "offset_credible_center_fraction",
    "offset_touches_current_left_boundary",
    "offset_touches_current_right_boundary",
    "offset_is_censored",
    "base_onset_entropy_log1p",
    "base_offset_entropy_log1p",
    "base_earliest_field_stability",
    "base_soz_rank_stability",
    "base_finding_opportunity_fraction",
    "base_onset_entropy_evaluable",
    "base_offset_entropy_evaluable",
    "base_earliest_field_stability_evaluable",
    "base_soz_rank_stability_evaluable",
    "base_finding_opportunity_evaluable",
    "nearest_left_neighbor_gap_over_current_duration",
    "nearest_right_neighbor_gap_over_current_duration",
    "left_neighbor_present",
    "right_neighbor_present",
)

_ENDPOINT_TO_FEATURE_NAME: Final[Mapping[str, str]] = {
    "onset_entropy_nats": "base_onset_entropy_log1p",
    "offset_entropy_nats": "base_offset_entropy_log1p",
    "earliest_field_stability": "base_earliest_field_stability",
    "soz_rank_stability": "base_soz_rank_stability",
    "finding_opportunity_fraction": "base_finding_opportunity_fraction",
}
_ENDPOINT_TO_MASK_FEATURE_NAME: Final[Mapping[str, str]] = {
    "onset_entropy_nats": "base_onset_entropy_evaluable",
    "offset_entropy_nats": "base_offset_entropy_evaluable",
    "earliest_field_stability": "base_earliest_field_stability_evaluable",
    "soz_rank_stability": "base_soz_rank_stability_evaluable",
    "finding_opportunity_fraction": "base_finding_opportunity_evaluable",
}
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-8

_COUNTERFACTUAL_FIREWALL: Final[dict[str, bool]] = {
    "eeg_signal_only": True,
    "downstream_endpoint_patient_disjoint_from_current_patient": True,
    "downstream_endpoint_frozen_before_current_target": True,
    "hidden_chunk_used_as_predictor_input": False,
    "reference_boundary_used": False,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_context_used": False,
    "video_or_semiology_used": False,
    "sleep_activation_or_other_physiology_used": False,
    "private_data_used": False,
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


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
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


def _same_floats(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= _TOLERANCE for a, b in zip(left, right)
    )


def _same_intervals(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> bool:
    return len(left) == len(right) and all(
        _same_floats(a, b) for a, b in zip(left, right)
    )


def _visible_intervals(state: Mapping[str, Any]) -> list[list[float]]:
    rows = [
        list(state["current_event_interval_recording_seconds"]),
        *[
            list(interval)
            for interval in state["retrieved_background_intervals_recording_seconds"]
        ],
    ]
    return sorted(rows, key=lambda interval: (interval[0], interval[1]))


def _validate_base_snapshot_for_predictor(
    value: object,
    *,
    visible_intervals: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("base_snapshot must be an object")
    expected_keys = {
        "endpoint_bundle_sha256",
        "input_evidence_union_sha256",
        "evidence_interval_roster_sha256",
        "metrics",
        "firewall",
    }
    if set(value) != expected_keys:
        raise ValueError("base_snapshot keys drifted")
    metrics_raw = value["metrics"]
    if type(metrics_raw) is not dict or set(metrics_raw) != set(
        BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES
    ):
        raise ValueError("base_snapshot.metrics vocabulary drifted")
    metrics: dict[str, float | None] = {}
    for name in BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES:
        raw = metrics_raw[name]
        if raw is None:
            metrics[name] = None
            continue
        number = _finite(raw, f"base_snapshot.metrics.{name}", minimum=0.0)
        if name in {
            "earliest_field_stability",
            "soz_rank_stability",
            "finding_opportunity_fraction",
        } and number > 1.0 + _TOLERANCE:
            raise ValueError(f"base_snapshot.metrics.{name} must lie in [0,1]")
        metrics[name] = min(1.0, number) if name.endswith("stability") or name.endswith("fraction") else number
    if all(value is None for value in metrics.values()):
        raise ValueError("base_snapshot cannot be entirely not-evaluable")
    if value["firewall"] != _COUNTERFACTUAL_FIREWALL:
        raise ValueError("base_snapshot violates the EEG-only predictor firewall")
    expected_roster = _canonical_sha256(
        {
            "schema_version": "ba_ieg_counterfactual_physical_interval_roster_v1",
            "visible_intervals_recording_seconds": [
                [float(interval[0]), float(interval[1])]
                for interval in visible_intervals
            ],
        }
    )
    roster = _sha256(
        value["evidence_interval_roster_sha256"],
        "base_snapshot.evidence_interval_roster_sha256",
    )
    if roster != expected_roster:
        raise ValueError("base_snapshot does not bind the visible controller support")
    return {
        "endpoint_bundle_sha256": _sha256(
            value["endpoint_bundle_sha256"],
            "base_snapshot.endpoint_bundle_sha256",
        ),
        "input_evidence_union_sha256": _sha256(
            value["input_evidence_union_sha256"],
            "base_snapshot.input_evidence_union_sha256",
        ),
        "evidence_interval_roster_sha256": roster,
        "metrics": metrics,
        "firewall": dict(_COUNTERFACTUAL_FIREWALL),
    }


def _candidate_action(
    state: Mapping[str, Any], action_id: str
) -> Mapping[str, Any]:
    matches = [
        action for action in state["action_candidates"] if action["action_id"] == action_id
    ]
    if len(matches) != 1:
        raise ValueError("action_id must identify exactly one controller candidate")
    action = matches[0]
    if action["action_type"] not in BA_IEG_OUTER_ACTION_TYPES_V2:
        raise ValueError("outer trainable v2 only supports interval-acquisition actions")
    return action


def outer_target_independent_candidate_roster_sha256_v2(
    state_payload: Mapping[str, Any],
) -> str:
    """Hash the candidate geometry/cost roster without predicted utilities."""

    state = validate_boundary_adaptive_fine_analysis_state(state_payload)
    candidates = []
    for action in state["action_candidates"]:
        if action["action_type"] not in BA_IEG_OUTER_ACTION_TYPES_V2:
            continue
        candidates.append(
            {
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "side": action["side"],
                "proposed_intervals_recording_seconds": action[
                    "proposed_intervals_recording_seconds"
                ],
                "predicted_cost": action["predicted_cost"],
            }
        )
    if not candidates:
        raise ValueError("state has no interval-acquisition candidate")
    return _canonical_sha256(
        {
            "schema_version": "ba_ieg_outer_target_independent_candidate_roster_v2",
            "recording_id": state["recording_id"],
            "event_id": state["event_id"],
            "step_index": state["step_index"],
            "candidates": candidates,
        }
    )


def _state_context_descriptor(
    state: Mapping[str, Any], base_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "ba_ieg_outer_visible_state_context_v2",
        "recording_id": state["recording_id"],
        "event_id": state["event_id"],
        "step_index": state["step_index"],
        "canonical_evidence_root_sha256": state[
            "canonical_evidence_root_sha256"
        ],
        "detector_candidate_group_sha256": state[
            "detector_candidate_group_sha256"
        ],
        "recording_duration_seconds": state["recording_duration_seconds"],
        "current_event_interval_recording_seconds": state[
            "current_event_interval_recording_seconds"
        ],
        "retrieved_background_intervals_recording_seconds": state[
            "retrieved_background_intervals_recording_seconds"
        ],
        "neighbor_protection_intervals_recording_seconds": state[
            "neighbor_protection_intervals_recording_seconds"
        ],
        "side_state": state["side_state"],
        "boundary_posterior": state["boundary_posterior"],
        "budget": state["budget"],
        "base_endpoint_bundle_sha256": base_snapshot["endpoint_bundle_sha256"],
        "base_input_evidence_union_sha256": base_snapshot[
            "input_evidence_union_sha256"
        ],
        "base_evidence_interval_roster_sha256": base_snapshot[
            "evidence_interval_roster_sha256"
        ],
        "base_metrics": base_snapshot["metrics"],
        "candidate_roster_sha256": outer_target_independent_candidate_roster_sha256_v2(
            state
        ),
    }


def _nearest_neighbor_gaps(
    state: Mapping[str, Any], current_duration: float
) -> tuple[float, float, float, float]:
    current_start, current_stop = state["current_event_interval_recording_seconds"]
    left_gaps = [
        max(0.0, current_start - interval[1])
        for interval in state["neighbor_protection_intervals_recording_seconds"]
        if interval[0] < current_start
    ]
    right_gaps = [
        max(0.0, interval[0] - current_stop)
        for interval in state["neighbor_protection_intervals_recording_seconds"]
        if interval[1] > current_stop
    ]
    left_present = float(bool(left_gaps))
    right_present = float(bool(right_gaps))
    left_gap = min(left_gaps) / current_duration if left_gaps else 1.0
    right_gap = min(right_gaps) / current_duration if right_gaps else 1.0
    return left_gap, right_gap, left_present, right_present


def _feature_values(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    base_snapshot: Mapping[str, Any],
) -> tuple[float, ...]:
    current_start, current_stop = state["current_event_interval_recording_seconds"]
    current_duration = current_stop - current_start
    current_center = 0.5 * (current_start + current_stop)
    recording_duration = state["recording_duration_seconds"]
    intervals = action["proposed_intervals_recording_seconds"]
    action_seconds = sum(stop - start for start, stop in intervals)
    weighted_center = (
        sum(0.5 * (start + stop) * (stop - start) for start, stop in intervals)
        / action_seconds
        if action_seconds > 0.0
        else current_center
    )
    if action["action_type"] == "query_left":
        action_gap = max(0.0, current_start - intervals[-1][1])
    elif action["action_type"] == "query_right":
        action_gap = max(0.0, intervals[0][0] - current_stop)
    else:
        action_gap = min(
            max(0.0, current_start - stop, start - current_stop)
            for start, stop in intervals
        )
    budget = state["budget"]
    remaining_eeg = max(
        0.0,
        budget["maximum_query_eeg_seconds"] - budget["used_query_eeg_seconds"],
    )
    remaining_gpu = max(
        0.0,
        budget["maximum_gpu_seconds"] - budget["used_gpu_seconds"],
    )
    step_remaining = max(0, budget["maximum_steps"] - budget["used_steps"])
    background_seconds = sum(
        stop - start
        for start, stop in state[
            "retrieved_background_intervals_recording_seconds"
        ]
    )

    features: dict[str, float] = {
        "action_is_query_left": float(action["action_type"] == "query_left"),
        "action_is_query_right": float(action["action_type"] == "query_right"),
        "action_is_distant_background": float(
            action["action_type"] == "retrieve_distant_background"
        ),
        "action_duration_over_current_duration": action_seconds / current_duration,
        "action_signed_center_distance_over_current_duration": (
            weighted_center - current_center
        )
        / current_duration,
        "action_gap_over_current_duration": action_gap / current_duration,
        "action_gpu_over_remaining_gpu_budget": action["predicted_cost"][
            "incremental_gpu_seconds"
        ]
        / max(remaining_gpu, 1e-6),
        "action_bad_quality_fraction": action["predicted_cost"][
            "bad_quality_fraction"
        ],
        "action_neighbor_merge_risk": action["predicted_cost"][
            "neighbor_merge_risk"
        ],
        "step_used_fraction": budget["used_steps"]
        / max(1, budget["maximum_steps"]),
        "eeg_budget_remaining_fraction": remaining_eeg
        / max(budget["maximum_query_eeg_seconds"], 1e-6),
        "gpu_budget_remaining_fraction": remaining_gpu
        / max(budget["maximum_gpu_seconds"], 1e-6),
        "current_duration_over_recording_duration": current_duration
        / recording_duration,
        "retrieved_background_seconds_over_recording_duration": background_seconds
        / recording_duration,
        "retrieved_background_count_log1p": math.log1p(
            len(state["retrieved_background_intervals_recording_seconds"])
        ),
        "left_side_open": float(state["side_state"]["left"] == "open"),
        "right_side_open": float(state["side_state"]["right"] == "open"),
    }

    for boundary in ("onset", "offset"):
        posterior = state["boundary_posterior"][boundary]
        credible_start, credible_stop = posterior[
            "credible_interval_recording_seconds"
        ]
        features.update(
            {
                f"{boundary}_entropy_log1p": math.log1p(
                    posterior["entropy_nats"]
                ),
                f"{boundary}_credible_mass": posterior["credible_mass"],
                f"{boundary}_credible_width_over_current_duration": (
                    credible_stop - credible_start
                )
                / current_duration,
                f"{boundary}_credible_center_fraction": (
                    0.5 * (credible_start + credible_stop) - current_start
                )
                / current_duration,
                f"{boundary}_touches_current_left_boundary": float(
                    posterior["touches_current_left_boundary"]
                ),
                f"{boundary}_touches_current_right_boundary": float(
                    posterior["touches_current_right_boundary"]
                ),
                f"{boundary}_is_censored": float(
                    posterior["censoring"] != "none"
                ),
            }
        )

    for endpoint_name in BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES:
        value = base_snapshot["metrics"][endpoint_name]
        feature_name = _ENDPOINT_TO_FEATURE_NAME[endpoint_name]
        features[feature_name] = (
            math.log1p(value)
            if value is not None and endpoint_name.endswith("entropy_nats")
            else float(value or 0.0)
        )
        features[_ENDPOINT_TO_MASK_FEATURE_NAME[endpoint_name]] = float(
            value is not None
        )

    left_gap, right_gap, left_present, right_present = _nearest_neighbor_gaps(
        state, current_duration
    )
    features.update(
        {
            "nearest_left_neighbor_gap_over_current_duration": left_gap,
            "nearest_right_neighbor_gap_over_current_duration": right_gap,
            "left_neighbor_present": left_present,
            "right_neighbor_present": right_present,
        }
    )
    if set(features) != set(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2):
        raise RuntimeError("outer acquisition feature implementation drifted")
    values = tuple(float(features[name]) for name in BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("outer acquisition features must be finite")
    if step_remaining < 0:  # defensive: validated state should make this impossible
        raise RuntimeError("negative remaining step budget")
    return values


def outer_action_predictor_input_receipt_sha256_v2(
    state_payload: Mapping[str, Any],
    *,
    action_id: str,
    base_snapshot: Mapping[str, Any],
) -> str:
    """Hash the exact visible-only state-action model input.

    ``predicted_gain`` and its model receipt are intentionally excluded.  This
    prevents circularly feeding a previous utility prediction back as a
    feature and makes the receipt usable before a hidden target exists.
    """

    state = validate_boundary_adaptive_fine_analysis_state(state_payload)
    visible = _visible_intervals(state)
    base = _validate_base_snapshot_for_predictor(
        base_snapshot, visible_intervals=visible
    )
    action = _candidate_action(state, action_id)
    features = _feature_values(state, action, base)
    state_context_sha256 = _canonical_sha256(_state_context_descriptor(state, base))
    descriptor = {
        "schema_version": "ba_ieg_outer_visible_state_action_predictor_input_v2",
        "method_id": BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2,
        "state_context_sha256": state_context_sha256,
        "action": {
            "action_id": action["action_id"],
            "action_type": action["action_type"],
            "side": action["side"],
            "proposed_intervals_recording_seconds": action[
                "proposed_intervals_recording_seconds"
            ],
            "predicted_cost": action["predicted_cost"],
        },
        "feature_names": list(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2),
        "feature_values": list(features),
        "forbidden_inputs": {
            "hidden_or_revealed_chunk": False,
            "counterfactual_delta": False,
            "reference_boundary": False,
            "annotation_excel_doctor_text_private_label": False,
            "previous_predicted_gain": False,
        },
    }
    return _canonical_sha256(descriptor)


@dataclass(frozen=True)
class BAIEGOuterAcquisitionExampleV2:
    """One visible state-action input paired with its hidden-chunk target."""

    target_id: str
    patient_uid: str
    recording_id: str
    event_id: str
    step_index: int
    model_split: str
    source_data_manifest_sha256: str
    context_id: str
    action_id: str
    action_type: str
    side: str
    predictor_input_receipt_sha256: str
    features: torch.Tensor
    signed_gain_target: torch.Tensor
    benefit_target: torch.Tensor
    harm_magnitude_target: torch.Tensor
    harm_target: torch.Tensor
    evaluable_mask: torch.Tensor
    costs: torch.Tensor
    example_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        feature_count = len(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2)
        gain_count = len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES)
        tensors = {
            "features": (self.features, (feature_count,), torch.float32),
            "signed_gain_target": (
                self.signed_gain_target,
                (gain_count,),
                torch.float32,
            ),
            "benefit_target": (self.benefit_target, (gain_count,), torch.float32),
            "harm_magnitude_target": (
                self.harm_magnitude_target,
                (gain_count,),
                torch.float32,
            ),
            "harm_target": (self.harm_target, (gain_count,), torch.float32),
            "evaluable_mask": (
                self.evaluable_mask,
                (gain_count,),
                torch.bool,
            ),
            "costs": (self.costs, (len(BA_IEG_OUTER_COST_NAMES_V2),), torch.float32),
        }
        for name, (tensor, shape, dtype) in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} has invalid shape")
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must use dtype {dtype}")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be detached")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, tensor.detach().clone().contiguous())
        if not self.evaluable_mask.any():
            raise ValueError("outer acquisition example needs an evaluable endpoint")
        if (self.benefit_target < 0).any() or (self.harm_magnitude_target < 0).any():
            raise ValueError("benefit/harm magnitudes must be non-negative")
        if ((self.harm_target < 0) | (self.harm_target > 1)).any():
            raise ValueError("harm targets must lie in [0,1]")
        if (self.costs < 0).any():
            raise ValueError("outer acquisition costs must be non-negative")
        _sha256(self.predictor_input_receipt_sha256, "predictor_input_receipt_sha256")
        _sha256(self.source_data_manifest_sha256, "source_data_manifest_sha256")
        descriptor = {
            "schema_version": BA_IEG_OUTER_ACTIVE_ACQUISITION_EXAMPLE_SCHEMA_VERSION_V2,
            "target_id": self.target_id,
            "patient_uid": self.patient_uid,
            "recording_id": self.recording_id,
            "event_id": self.event_id,
            "step_index": self.step_index,
            "model_split": self.model_split,
            "source_data_manifest_sha256": self.source_data_manifest_sha256,
            "context_id": self.context_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "side": self.side,
            "predictor_input_receipt_sha256": self.predictor_input_receipt_sha256,
            "features": self.features.tolist(),
            "signed_gain_target": self.signed_gain_target.tolist(),
            "evaluable_mask": self.evaluable_mask.tolist(),
            "costs": self.costs.tolist(),
        }
        object.__setattr__(self, "example_sha256", _canonical_sha256(descriptor))


def materialize_ba_ieg_outer_acquisition_example_v2(
    *,
    state: Mapping[str, Any],
    counterfactual_target: Mapping[str, Any],
) -> BAIEGOuterAcquisitionExampleV2:
    """Join one visible controller state to one exact counterfactual target."""

    validated_state = validate_boundary_adaptive_fine_analysis_state(state)
    target = validate_hidden_chunk_counterfactual_target_v1(counterfactual_target)
    if target["recording_id"] != validated_state["recording_id"] or target[
        "event_id"
    ] != validated_state["event_id"]:
        raise ValueError("controller state and target identify different events")
    target_action = target["action"]
    action = _candidate_action(validated_state, target_action["action_id"])
    if target_action["action_type"] != action["action_type"] or target_action[
        "side"
    ] != action["side"]:
        raise ValueError("target action type/side does not match controller candidate")
    if not _same_floats(
        target_action["current_event_interval_recording_seconds"],
        validated_state["current_event_interval_recording_seconds"],
    ):
        raise ValueError("target current event interval does not match controller state")
    visible = _visible_intervals(validated_state)
    if not _same_intervals(
        target_action["visible_intervals_recording_seconds"], visible
    ):
        raise ValueError("target visible intervals do not match controller support")
    if not _same_intervals(
        target_action["proposed_intervals_recording_seconds"],
        action["proposed_intervals_recording_seconds"],
    ):
        raise ValueError("target proposed interval does not match controller action")
    expected_roster = outer_target_independent_candidate_roster_sha256_v2(
        validated_state
    )
    if target_action["target_independent_candidate_roster_sha256"] != expected_roster:
        raise ValueError("target was not drawn from the frozen candidate roster")

    base = _validate_base_snapshot_for_predictor(
        target["base_snapshot"], visible_intervals=visible
    )
    for boundary, endpoint in (
        ("onset", "onset_entropy_nats"),
        ("offset", "offset_entropy_nats"),
    ):
        base_entropy = base["metrics"][endpoint]
        if base_entropy is not None and abs(
            base_entropy
            - validated_state["boundary_posterior"][boundary]["entropy_nats"]
        ) > _TOLERANCE:
            raise ValueError(
                f"base {boundary} entropy does not match the visible controller state"
            )
    expected_receipt = outer_action_predictor_input_receipt_sha256_v2(
        validated_state,
        action_id=action["action_id"],
        base_snapshot=base,
    )
    if target_action["predictor_input_receipt_sha256"] != expected_receipt:
        raise ValueError(
            "counterfactual target does not bind the exact visible-only predictor input"
        )
    features = torch.tensor(
        _feature_values(validated_state, action, base), dtype=torch.float32
    )
    signed_values = [
        float(target["raw_signed_delta"][name] or 0.0)
        for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES
    ]
    signed = torch.tensor(signed_values, dtype=torch.float32)
    benefit = torch.clamp_min(signed, 0.0)
    harm_magnitude = torch.clamp_min(-signed, 0.0)
    harm = torch.tensor(
        [float(target["harm_target"][name] or 0.0) for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES],
        dtype=torch.float32,
    )
    evaluable = torch.tensor(
        [bool(target["evaluable_mask"][name]) for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES],
        dtype=torch.bool,
    )
    costs = torch.tensor(
        [float(action["predicted_cost"][name]) for name in BA_IEG_OUTER_COST_NAMES_V2],
        dtype=torch.float32,
    )
    state_context_sha256 = _canonical_sha256(
        _state_context_descriptor(validated_state, base)
    )
    context_id = "BAIEG-OAC-CONTEXT-" + _canonical_sha256(
        {
            "patient_uid": target["patient_uid"],
            "state_context_sha256": state_context_sha256,
        }
    )[:24]
    return BAIEGOuterAcquisitionExampleV2(
        target_id=target["target_id"],
        patient_uid=target["patient_uid"],
        recording_id=target["recording_id"],
        event_id=target["event_id"],
        step_index=validated_state["step_index"],
        model_split=target["model_split"],
        source_data_manifest_sha256=target["source_data_manifest_sha256"],
        context_id=context_id,
        action_id=action["action_id"],
        action_type=action["action_type"],
        side=action["side"],
        predictor_input_receipt_sha256=expected_receipt,
        features=features,
        signed_gain_target=signed,
        benefit_target=benefit,
        harm_magnitude_target=harm_magnitude,
        harm_target=harm,
        evaluable_mask=evaluable,
        costs=costs,
    )


@dataclass(frozen=True)
class BAIEGOuterAcquisitionBatchV2:
    """Patient/context/action-balanced tensor batch."""

    target_ids: tuple[str, ...]
    patient_uids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    action_types: tuple[str, ...]
    sides: tuple[str, ...]
    step_indices: tuple[int, ...]
    model_split: str
    features: torch.Tensor
    signed_gain_target: torch.Tensor
    benefit_target: torch.Tensor
    harm_magnitude_target: torch.Tensor
    harm_target: torch.Tensor
    evaluable_mask: torch.Tensor
    comparison_mask: torch.Tensor
    costs: torch.Tensor
    row_weight: torch.Tensor
    batch_sha256: str

    def __post_init__(self) -> None:
        row_count = len(self.target_ids)
        if row_count < 1:
            raise ValueError("outer acquisition batch cannot be empty")
        identifier_columns = (
            self.patient_uids,
            self.recording_ids,
            self.event_ids,
            self.context_ids,
            self.action_ids,
            self.action_types,
            self.sides,
            self.step_indices,
        )
        if any(len(column) != row_count for column in identifier_columns):
            raise ValueError("outer acquisition batch identifiers are misaligned")
        gain_shape = (row_count, len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES))
        if self.features.shape != (
            row_count,
            len(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2),
        ):
            raise ValueError("outer acquisition batch feature shape drifted")
        for tensor in (
            self.signed_gain_target,
            self.benefit_target,
            self.harm_magnitude_target,
            self.harm_target,
            self.evaluable_mask,
            self.comparison_mask,
        ):
            if tensor.shape != gain_shape:
                raise ValueError("outer acquisition batch gain shape drifted")
        if self.costs.shape != (row_count, len(BA_IEG_OUTER_COST_NAMES_V2)):
            raise ValueError("outer acquisition batch cost shape drifted")
        if self.row_weight.shape != (row_count,):
            raise ValueError("outer acquisition row weights are misaligned")
        if self.evaluable_mask.dtype != torch.bool or self.comparison_mask.dtype != torch.bool:
            raise TypeError("outer acquisition masks must be boolean")
        if not torch.all(self.comparison_mask <= self.evaluable_mask):
            raise ValueError("comparison mask cannot exceed per-action evaluability")
        for tensor in (
            self.features,
            self.signed_gain_target,
            self.benefit_target,
            self.harm_magnitude_target,
            self.harm_target,
            self.costs,
            self.row_weight,
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("outer acquisition batch tensors must be finite floats")
        if (self.row_weight <= 0).any() or not torch.isclose(
            self.row_weight.sum(),
            torch.tensor(1.0, dtype=self.row_weight.dtype),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("outer acquisition row weights must be positive and sum to one")
        if self.model_split not in {"source_train", "source_dev"}:
            raise ValueError("outer acquisition batch split is invalid")
        _sha256(self.batch_sha256, "batch_sha256")


def collate_ba_ieg_outer_acquisition_examples_v2(
    examples: Sequence[BAIEGOuterAcquisitionExampleV2],
) -> BAIEGOuterAcquisitionBatchV2:
    """Collate with equal patient, then context, then action mass."""

    if isinstance(examples, (str, bytes)) or not examples:
        raise ValueError("outer acquisition examples must be non-empty")
    rows = sorted(examples, key=lambda item: (item.context_id, item.action_id))
    if len({item.target_id for item in rows}) != len(rows):
        raise ValueError("outer acquisition target IDs must be unique")
    if len({item.example_sha256 for item in rows}) != len(rows):
        raise ValueError("outer acquisition examples must be content-distinct")
    splits = {item.model_split for item in rows}
    manifests = {item.source_data_manifest_sha256 for item in rows}
    if len(splits) != 1 or len(manifests) != 1:
        raise ValueError("one outer acquisition batch cannot mix split/manifests")
    context_to_rows: dict[str, list[int]] = {}
    context_to_patient: dict[str, str] = {}
    for index, item in enumerate(rows):
        context_to_rows.setdefault(item.context_id, []).append(index)
        previous = context_to_patient.setdefault(item.context_id, item.patient_uid)
        if previous != item.patient_uid:
            raise ValueError("one acquisition context cannot span patients")
    patient_to_contexts: dict[str, list[str]] = {}
    for context_id, patient_uid in context_to_patient.items():
        patient_to_contexts.setdefault(patient_uid, []).append(context_id)
    patient_count = len(patient_to_contexts)
    row_weights = torch.empty(len(rows), dtype=torch.float32)
    comparison = torch.zeros(
        (len(rows), len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES)), dtype=torch.bool
    )
    for context_id, indices in context_to_rows.items():
        patient_uid = context_to_patient[context_id]
        common_mask = torch.stack([rows[index].evaluable_mask for index in indices]).all(
            dim=0
        )
        if not common_mask.any():
            raise ValueError(
                "actions in one context share no counterfactual endpoint denominator"
            )
        for index in indices:
            comparison[index] = common_mask
            row_weights[index] = (
                1.0
                / patient_count
                / len(patient_to_contexts[patient_uid])
                / len(indices)
            )
    descriptor = {
        "schema_version": "ba_ieg_outer_active_acquisition_batch_v2",
        "example_sha256": [item.example_sha256 for item in rows],
        "model_split": next(iter(splits)),
        "source_data_manifest_sha256": next(iter(manifests)),
        "weighting": "equal_patient_then_equal_context_then_equal_action",
    }
    return BAIEGOuterAcquisitionBatchV2(
        target_ids=tuple(item.target_id for item in rows),
        patient_uids=tuple(item.patient_uid for item in rows),
        recording_ids=tuple(item.recording_id for item in rows),
        event_ids=tuple(item.event_id for item in rows),
        context_ids=tuple(item.context_id for item in rows),
        action_ids=tuple(item.action_id for item in rows),
        action_types=tuple(item.action_type for item in rows),
        sides=tuple(item.side for item in rows),
        step_indices=tuple(item.step_index for item in rows),
        model_split=next(iter(splits)),
        features=torch.stack([item.features for item in rows]),
        signed_gain_target=torch.stack([item.signed_gain_target for item in rows]),
        benefit_target=torch.stack([item.benefit_target for item in rows]),
        harm_magnitude_target=torch.stack(
            [item.harm_magnitude_target for item in rows]
        ),
        harm_target=torch.stack([item.harm_target for item in rows]),
        evaluable_mask=torch.stack([item.evaluable_mask for item in rows]),
        comparison_mask=comparison,
        costs=torch.stack([item.costs for item in rows]),
        row_weight=row_weights,
        batch_sha256=_canonical_sha256(descriptor),
    )


@dataclass(frozen=True)
class BAIEGOuterAcquisitionUtilityWeightsV2:
    """Pre-registered endpoint and resource coefficients."""

    endpoint_weights: tuple[float, ...] = (1.0, 1.0, 0.5, 0.5, 0.25)
    eeg_seconds_cost: float = 0.01
    gpu_seconds_cost: float = 0.01
    bad_quality_fraction_cost: float = 0.5
    neighbor_merge_risk_cost: float = 1.0

    def __post_init__(self) -> None:
        if len(self.endpoint_weights) != len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES):
            raise ValueError("endpoint utility weights have invalid length")
        values = [
            *self.endpoint_weights,
            self.eeg_seconds_cost,
            self.gpu_seconds_cost,
            self.bad_quality_fraction_cost,
            self.neighbor_merge_risk_cost,
        ]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("outer utility weights must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("outer utility cannot have all-zero weights")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": "ba_ieg_outer_utility_weights_v2",
                "gain_names": list(BA_IEG_COUNTERFACTUAL_GAIN_NAMES),
                "endpoint_weights": list(self.endpoint_weights),
                "cost_names": list(BA_IEG_OUTER_COST_NAMES_V2),
                "cost_weights": [
                    self.eeg_seconds_cost,
                    self.gpu_seconds_cost,
                    self.bad_quality_fraction_cost,
                    self.neighbor_merge_risk_cost,
                ],
            }
        )


class BAIEGOuterAcquisitionUtilityModelV2(nn.Module):
    """Small state-action MLP with interpretable signed-utility heads."""

    def __init__(self, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 4:
            raise ValueError("hidden_dim must be an integer >= 4")
        input_dim = len(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2)
        output_dim = len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.benefit_head = nn.Linear(hidden_dim, output_dim)
        self.harm_magnitude_head = nn.Linear(hidden_dim, output_dim)
        self.harm_logit_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("outer acquisition features must be [N,F]")
        if features.shape[1] != len(BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2):
            raise ValueError("outer acquisition feature width drifted")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("outer acquisition features must be finite floats")
        hidden = self.encoder(features)
        benefit = torch_functional.softplus(self.benefit_head(hidden))
        harm_magnitude = torch_functional.softplus(
            self.harm_magnitude_head(hidden)
        )
        harm_logits = self.harm_logit_head(hidden)
        harm_probability = torch.sigmoid(harm_logits)
        expected_signed_gain = (
            (1.0 - harm_probability) * benefit
            - harm_probability * harm_magnitude
        )
        return {
            "benefit": benefit,
            "harm_magnitude": harm_magnitude,
            "harm_logits": harm_logits,
            "harm_probability": harm_probability,
            "expected_signed_gain": expected_signed_gain,
        }


def _validate_predictions(
    predictions: Mapping[str, torch.Tensor], batch: BAIEGOuterAcquisitionBatchV2
) -> None:
    expected = {
        "benefit",
        "harm_magnitude",
        "harm_logits",
        "harm_probability",
        "expected_signed_gain",
    }
    if set(predictions) != expected:
        raise ValueError("outer acquisition prediction heads drifted")
    shape = batch.signed_gain_target.shape
    for name in expected:
        tensor = predictions[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
            raise ValueError(f"outer acquisition prediction {name} shape drifted")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"outer acquisition prediction {name} must be finite")
    if (predictions["benefit"] < 0).any() or (
        predictions["harm_magnitude"] < 0
    ).any():
        raise ValueError("outer acquisition magnitude heads must be non-negative")
    if ((predictions["harm_probability"] < 0) | (predictions["harm_probability"] > 1)).any():
        raise ValueError("outer acquisition harm probability must lie in [0,1]")


def _utility_tensor(
    signed_gain: torch.Tensor,
    costs: torch.Tensor,
    comparison_mask: torch.Tensor,
    weights: BAIEGOuterAcquisitionUtilityWeightsV2,
) -> torch.Tensor:
    endpoint_weight = torch.tensor(
        weights.endpoint_weights, device=signed_gain.device, dtype=signed_gain.dtype
    )
    cost_weight = torch.tensor(
        [
            weights.eeg_seconds_cost,
            weights.gpu_seconds_cost,
            weights.bad_quality_fraction_cost,
            weights.neighbor_merge_risk_cost,
        ],
        device=signed_gain.device,
        dtype=signed_gain.dtype,
    )
    return (
        signed_gain
        * comparison_mask.to(device=signed_gain.device, dtype=signed_gain.dtype)
        * endpoint_weight
    ).sum(dim=1) - (
        costs.to(device=signed_gain.device, dtype=signed_gain.dtype) * cost_weight
    ).sum(dim=1)


def compute_ba_ieg_outer_acquisition_training_loss_v2(
    *,
    predictions: Mapping[str, torch.Tensor],
    batch: BAIEGOuterAcquisitionBatchV2,
    utility_weights: BAIEGOuterAcquisitionUtilityWeightsV2,
    benefit_weight: float = 1.0,
    harm_magnitude_weight: float = 1.0,
    harm_classification_weight: float = 1.0,
    ranking_weight: float = 0.5,
    ranking_temperature: float = 0.25,
    ranking_tie_margin: float = 1e-5,
) -> dict[str, torch.Tensor | str | int]:
    """Train component heads and within-state counterfactual action ranking."""

    if batch.model_split != "source_train":
        raise ValueError("gradient-bearing outer acquisition loss is source_train-only")
    _validate_predictions(predictions, batch)
    scalar_weights = {
        "benefit_weight": benefit_weight,
        "harm_magnitude_weight": harm_magnitude_weight,
        "harm_classification_weight": harm_classification_weight,
        "ranking_weight": ranking_weight,
    }
    parsed = {
        name: _finite(value, name, minimum=0.0)
        for name, value in scalar_weights.items()
    }
    temperature = _finite(
        ranking_temperature, "ranking_temperature", minimum=1e-9
    )
    tie_margin = _finite(ranking_tie_margin, "ranking_tie_margin", minimum=0.0)
    device = predictions["benefit"].device
    dtype = predictions["benefit"].dtype
    mask = batch.evaluable_mask.to(device=device)
    mask_float = mask.to(dtype=dtype)
    row_weight = batch.row_weight.to(device=device, dtype=dtype)
    denominator = mask_float.sum(dim=1)

    def aggregate(item_loss: torch.Tensor) -> torch.Tensor:
        per_row = (item_loss * mask_float).sum(dim=1) / denominator
        return (per_row * row_weight).sum()

    benefit_loss = aggregate(
        torch_functional.smooth_l1_loss(
            predictions["benefit"],
            batch.benefit_target.to(device=device, dtype=dtype),
            reduction="none",
        )
    )
    harm_magnitude_loss = aggregate(
        torch_functional.smooth_l1_loss(
            predictions["harm_magnitude"],
            batch.harm_magnitude_target.to(device=device, dtype=dtype),
            reduction="none",
        )
    )
    harm_classification_loss = aggregate(
        torch_functional.binary_cross_entropy_with_logits(
            predictions["harm_logits"],
            batch.harm_target.to(device=device, dtype=dtype),
            reduction="none",
        )
    )
    predicted_utility = _utility_tensor(
        predictions["expected_signed_gain"],
        batch.costs,
        batch.comparison_mask,
        utility_weights,
    )
    target_utility = _utility_tensor(
        batch.signed_gain_target.to(device=device, dtype=dtype),
        batch.costs,
        batch.comparison_mask,
        utility_weights,
    )

    context_to_indices: dict[str, list[int]] = {}
    for index, context_id in enumerate(batch.context_ids):
        context_to_indices.setdefault(context_id, []).append(index)
    patient_to_contexts: dict[str, set[str]] = {}
    for patient_uid, context_id in zip(batch.patient_uids, batch.context_ids):
        patient_to_contexts.setdefault(patient_uid, set()).add(context_id)
    pair_losses: list[torch.Tensor] = []
    pair_weights: list[float] = []
    for context_id, indices in context_to_indices.items():
        patient_uid = batch.patient_uids[indices[0]]
        local_losses: list[torch.Tensor] = []
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                target_difference = target_utility[left_index] - target_utility[right_index]
                if abs(float(target_difference.detach().cpu())) <= tie_margin:
                    continue
                sign = torch.sign(target_difference.detach())
                predicted_difference = (
                    predicted_utility[left_index] - predicted_utility[right_index]
                )
                local_losses.append(
                    torch_functional.softplus(
                        -sign * predicted_difference / temperature
                    )
                )
        if local_losses:
            pair_losses.append(torch.stack(local_losses).mean())
            pair_weights.append(
                1.0
                / len(patient_to_contexts)
                / len(patient_to_contexts[patient_uid])
            )
    if pair_losses:
        ranking_loss = sum(
            loss * weight for loss, weight in zip(pair_losses, pair_weights)
        )
        ranking_pair_count = sum(
            1
            for indices in context_to_indices.values()
            for left_position, left_index in enumerate(indices)
            for right_index in indices[left_position + 1 :]
            if abs(
                float(
                    (target_utility[left_index] - target_utility[right_index])
                    .detach()
                    .cpu()
                )
            )
            > tie_margin
        )
    else:
        ranking_loss = predicted_utility.sum() * 0.0
        ranking_pair_count = 0
    total_loss = (
        parsed["benefit_weight"] * benefit_loss
        + parsed["harm_magnitude_weight"] * harm_magnitude_loss
        + parsed["harm_classification_weight"] * harm_classification_loss
        + parsed["ranking_weight"] * ranking_loss
    )
    return {
        "method_id": BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2,
        "batch_sha256": batch.batch_sha256,
        "utility_weights_sha256": utility_weights.sha256,
        "benefit_loss": benefit_loss,
        "harm_magnitude_loss": harm_magnitude_loss,
        "harm_classification_loss": harm_classification_loss,
        "ranking_loss": ranking_loss,
        "ranking_pair_count": ranking_pair_count,
        "total_loss": total_loss,
    }


def _context_weights(batch: BAIEGOuterAcquisitionBatchV2) -> dict[str, float]:
    patient_to_contexts: dict[str, set[str]] = {}
    context_to_patient: dict[str, str] = {}
    for patient_uid, context_id in zip(batch.patient_uids, batch.context_ids):
        patient_to_contexts.setdefault(patient_uid, set()).add(context_id)
        context_to_patient[context_id] = patient_uid
    return {
        context_id: 1.0 / len(patient_to_contexts) / len(patient_to_contexts[patient_uid])
        for context_id, patient_uid in context_to_patient.items()
    }


def evaluate_fixed_budget_ba_ieg_outer_acquisition_v2(
    *,
    predictions: Mapping[str, torch.Tensor],
    batch: BAIEGOuterAcquisitionBatchV2,
    utility_weights: BAIEGOuterAcquisitionUtilityWeightsV2,
    maximum_eeg_seconds_per_decision: float,
    maximum_gpu_seconds_per_decision: float,
) -> dict[str, Any]:
    """Offline source-dev policy value under equal per-decision budgets.

    Alternative actions in one context are mutually exclusive.  The evaluator
    selects at most one or ``stop``; it never adds marginal utilities from
    chunks that were not jointly replayed.  The deterministic comparator
    alternates left/right by controller step and ignores the boundary
    posterior, providing a fixed symmetric watchdog at the same budget.
    """

    if batch.model_split != "source_dev":
        raise ValueError("fixed-budget outer acquisition evaluation is source_dev-only")
    _validate_predictions(predictions, batch)
    if any(tensor.requires_grad for tensor in predictions.values()):
        raise ValueError("evaluation predictions must be detached/frozen")
    eeg_budget = _finite(
        maximum_eeg_seconds_per_decision,
        "maximum_eeg_seconds_per_decision",
        minimum=0.0,
    )
    gpu_budget = _finite(
        maximum_gpu_seconds_per_decision,
        "maximum_gpu_seconds_per_decision",
        minimum=0.0,
    )
    predicted_utility = _utility_tensor(
        predictions["expected_signed_gain"].detach().cpu(),
        batch.costs,
        batch.comparison_mask,
        utility_weights,
    )
    target_utility = _utility_tensor(
        batch.signed_gain_target,
        batch.costs,
        batch.comparison_mask,
        utility_weights,
    )
    context_to_indices: dict[str, list[int]] = {}
    for index, context_id in enumerate(batch.context_ids):
        context_to_indices.setdefault(context_id, []).append(index)
    context_weight = _context_weights(batch)

    selections: dict[str, list[int | None]] = {
        "learned": [],
        "oracle": [],
        "alternating_symmetric": [],
        "lowest_cost": [],
    }
    context_order = sorted(context_to_indices)
    context_rows: list[dict[str, Any]] = []
    for context_id in context_order:
        indices = context_to_indices[context_id]
        eligible = [
            index
            for index in indices
            if float(batch.costs[index, 0]) <= eeg_budget + _TOLERANCE
            and float(batch.costs[index, 1]) <= gpu_budget + _TOLERANCE
        ]

        def positive_argmax(values: torch.Tensor) -> int | None:
            ranked = sorted(
                eligible,
                key=lambda index: (-float(values[index]), batch.action_ids[index]),
            )
            return ranked[0] if ranked and float(values[ranked[0]]) > 0.0 else None

        learned = positive_argmax(predicted_utility)
        oracle = positive_argmax(target_utility)
        step_index = batch.step_indices[indices[0]]
        preferred_side = "left" if step_index % 2 == 0 else "right"
        preferred = [index for index in eligible if batch.sides[index] == preferred_side]
        if not preferred:
            preferred = [
                index
                for index in eligible
                if batch.sides[index] in {"left", "right"}
            ]
        symmetric = (
            sorted(
                preferred,
                key=lambda index: (-float(batch.costs[index, 0]), batch.action_ids[index]),
            )[0]
            if preferred
            else None
        )
        lowest_cost = (
            sorted(
                eligible,
                key=lambda index: (
                    float(batch.costs[index, 0]),
                    float(batch.costs[index, 1]),
                    batch.action_ids[index],
                ),
            )[0]
            if eligible
            else None
        )
        for policy_name, selection in (
            ("learned", learned),
            ("oracle", oracle),
            ("alternating_symmetric", symmetric),
            ("lowest_cost", lowest_cost),
        ):
            selections[policy_name].append(selection)
        context_rows.append(
            {
                "context_id": context_id,
                "patient_uid": batch.patient_uids[indices[0]],
                "recording_id": batch.recording_ids[indices[0]],
                "event_id": batch.event_ids[indices[0]],
                "step_index": step_index,
                "eligible_action_ids": [batch.action_ids[index] for index in eligible],
                "learned_action_id": None if learned is None else batch.action_ids[learned],
                "oracle_action_id": None if oracle is None else batch.action_ids[oracle],
                "alternating_symmetric_action_id": (
                    None if symmetric is None else batch.action_ids[symmetric]
                ),
                "lowest_cost_action_id": (
                    None if lowest_cost is None else batch.action_ids[lowest_cost]
                ),
            }
        )

    policy_metrics: dict[str, dict[str, float]] = {}
    oracle_value_by_context = {
        context_id: (
            0.0
            if selection is None
            else float(target_utility[selection])
        )
        for context_id, selection in zip(context_order, selections["oracle"])
    }
    for policy_name, chosen_rows in selections.items():
        totals = {
            "counterfactual_policy_value": 0.0,
            "oracle_regret": 0.0,
            "best_action_accuracy": 0.0,
            "queried_eeg_seconds": 0.0,
            "queried_gpu_seconds": 0.0,
            "selected_harm_rate": 0.0,
            "stop_rate": 0.0,
            "left_selection_rate": 0.0,
            "right_selection_rate": 0.0,
            "background_selection_rate": 0.0,
            "budget_violation_rate": 0.0,
        }
        for context_id, selected, oracle_selected in zip(
            context_order, chosen_rows, selections["oracle"]
        ):
            weight = context_weight[context_id]
            value = 0.0 if selected is None else float(target_utility[selected])
            totals["counterfactual_policy_value"] += weight * value
            totals["oracle_regret"] += weight * max(
                0.0, oracle_value_by_context[context_id] - value
            )
            totals["best_action_accuracy"] += weight * float(
                selected == oracle_selected
            )
            totals["stop_rate"] += weight * float(selected is None)
            if selected is None:
                continue
            eeg_seconds = float(batch.costs[selected, 0])
            gpu_seconds = float(batch.costs[selected, 1])
            totals["queried_eeg_seconds"] += weight * eeg_seconds
            totals["queried_gpu_seconds"] += weight * gpu_seconds
            harm = bool(
                (
                    (batch.harm_target[selected] > 0.5)
                    & batch.evaluable_mask[selected]
                ).any()
            )
            totals["selected_harm_rate"] += weight * float(harm)
            totals["left_selection_rate"] += weight * float(
                batch.sides[selected] == "left"
            )
            totals["right_selection_rate"] += weight * float(
                batch.sides[selected] == "right"
            )
            totals["background_selection_rate"] += weight * float(
                batch.sides[selected] == "none"
            )
            totals["budget_violation_rate"] += weight * float(
                eeg_seconds > eeg_budget + _TOLERANCE
                or gpu_seconds > gpu_budget + _TOLERANCE
            )
        policy_metrics[policy_name] = totals

    body: dict[str, Any] = {
        "schema_version": BA_IEG_OUTER_ACTIVE_ACQUISITION_EVALUATION_SCHEMA_VERSION_V2,
        "evaluation_id": "CONTENT-ADDRESS-PENDING",
        "method_id": BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2,
        "evaluation_scope": "source_dev_contextual_counterfactual_replay_only",
        "batch_sha256": batch.batch_sha256,
        "utility_weights_sha256": utility_weights.sha256,
        "fixed_budget": {
            "maximum_eeg_seconds_per_decision": eeg_budget,
            "maximum_gpu_seconds_per_decision": gpu_budget,
            "at_most_one_mutually_exclusive_action": True,
            "stop_has_zero_utility": True,
        },
        "context_count": len(context_order),
        "patient_count": len(set(batch.patient_uids)),
        "policy_metrics": policy_metrics,
        "context_rows": context_rows,
        "interpretation_limits": {
            "multi_step_rollout_claimed": False,
            "alternative_action_rewards_added_together": False,
            "clinical_finding_or_soz_authority": False,
            "private_or_doctor_labels_used": False,
        },
    }
    body["evaluation_id"] = "BAIEG-OAC-EVAL-" + _canonical_sha256(body)[:24]
    return body


def risk_adjusted_outer_predicted_gain_for_selector_v2(
    predictions: Mapping[str, torch.Tensor],
    *,
    evaluable_mask: torch.Tensor,
) -> torch.Tensor:
    """Project trainable heads to the selector's non-negative gain fields.

    Negative expected utility remains represented by harm discounting and the
    fixed-budget scorer.  The returned tensor is computational routing input
    only and carries no Finding/onset/SOZ/report authority.
    """

    required = {"benefit", "harm_probability"}
    if not required.issubset(predictions):
        raise ValueError("benefit and harm_probability heads are required")
    benefit = predictions["benefit"]
    harm_probability = predictions["harm_probability"]
    if benefit.shape != harm_probability.shape or benefit.shape != evaluable_mask.shape:
        raise ValueError("selector projection tensors must align")
    if evaluable_mask.dtype != torch.bool:
        raise TypeError("selector projection evaluable_mask must be boolean")
    adjusted = benefit * (1.0 - harm_probability)
    adjusted = torch.where(evaluable_mask, adjusted, torch.zeros_like(adjusted))
    # The final three gains are bounded endpoint changes by construction.
    if adjusted.shape[1] != len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES):
        raise ValueError("selector projection gain width drifted")
    bounded = adjusted.clone()
    bounded[:, 2:] = bounded[:, 2:].clamp(max=1.0)
    return bounded


__all__ = [
    "BA_IEG_OUTER_ACTION_FEATURE_NAMES_V2",
    "BA_IEG_OUTER_ACTION_TYPES_V2",
    "BA_IEG_OUTER_ACTIVE_ACQUISITION_EVALUATION_SCHEMA_VERSION_V2",
    "BA_IEG_OUTER_ACTIVE_ACQUISITION_EXAMPLE_SCHEMA_VERSION_V2",
    "BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2",
    "BA_IEG_OUTER_COST_NAMES_V2",
    "BAIEGOuterAcquisitionBatchV2",
    "BAIEGOuterAcquisitionExampleV2",
    "BAIEGOuterAcquisitionUtilityModelV2",
    "BAIEGOuterAcquisitionUtilityWeightsV2",
    "collate_ba_ieg_outer_acquisition_examples_v2",
    "compute_ba_ieg_outer_acquisition_training_loss_v2",
    "evaluate_fixed_budget_ba_ieg_outer_acquisition_v2",
    "materialize_ba_ieg_outer_acquisition_example_v2",
    "outer_action_predictor_input_receipt_sha256_v2",
    "outer_target_independent_candidate_roster_sha256_v2",
    "risk_adjusted_outer_predicted_gain_for_selector_v2",
]
