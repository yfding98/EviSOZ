"""Replayable evidence-closure trajectories for additive NS-CEC v1.5.

An endpoint-aligned counterfactual row is a *one-step* conditional target.  It
is not a return that may be added to alternatives or to later steps.  This
module records the only legal multi-step object: after one action is selected,
the hidden native EEG is actually revealed, the complete downstream endpoint
is recomputed, and the resulting state becomes the next action's parent.

For every executed query the trajectory stores:

* mathematical support before/after the query and the selected action;
* boundary/onset-distribution recomputation receipts;
* which Finding atoms were first observed, stabilized, changed, or invalidated;
* each atom's candidate-minimal raw dependency union and exact receipts; and
* a native-remeasurement deletion counterfactual, including honest
  ``not_proven``/``not_evaluable`` outcomes.

The contract never treats attention, action value, entropy, or saliency as a
Finding.  It does not verify referenced raw bytes, certify minimality merely
because a receipt exists, authorize clinical terminology, or authorize a
report claim.  It leaves all frozen v1.4 files unchanged.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .nscec_external_endpoint_counterfactual_target_v1 import (
    NSCEC_COST_NAMES_V1,
    validate_nscec_external_endpoint_counterfactual_target_v1,
)


NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_SCHEMA_VERSION_V1: Final[
    str
] = "nscec_selected_query_evidence_closure_trajectory_v1"
NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_METHOD_ID_V1: Final[
    str
] = "nscec_actual_reveal_recompute_atom_dependency_trajectory_v1"

_ATOM_TRANSITIONS: Final[frozenset[str]] = frozenset(
    {
        "first_observed",
        "first_observed_and_stabilized",
        "updated_unstable",
        "stabilized",
        "changed_after_stabilization",
        "invalidated",
    }
)
_ATOM_STATES: Final[frozenset[str]] = frozenset(
    {"present", "uncertain", "not_evaluable", "absent_with_opportunity"}
)
_DELETION_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "atom_disappears",
        "atom_becomes_uncertain",
        "atom_becomes_not_evaluable",
        "atom_measurement_changes",
        "no_material_effect",
        "not_evaluable",
    }
)
_MINIMALITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"proven_by_registered_deletion", "not_proven", "not_evaluable"}
)
_DEPENDENCY_ROLES: Final[frozenset[str]] = frozenset(
    {"necessary", "supporting_nonnecessary", "no_detected_effect", "not_evaluable"}
)
_SIDE_STATES: Final[frozenset[str]] = frozenset(
    {"open", "normal_closed", "typed_censored"}
)
_POST_QUERY_DECISIONS: Final[frozenset[str]] = frozenset({"continue", "stop"})
_POST_QUERY_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "left_boundary_open",
        "right_boundary_open",
        "course_evidence_open",
        "clean_context_deficit",
        "both_sides_closed",
        "record_edge_censor",
        "impassable_qc_gap_censor",
        "neighbor_event_protection_censor",
        "missing_acquisition_support_censor",
        "budget_exhausted_censor",
        "mixed_normal_and_censored_closure",
    }
)
_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-8

_TRAJECTORY_AUTHORIZATION: Final[dict[str, object]] = {
    "software_contract_replayable": True,
    "referenced_raw_eeg_or_dependency_bytes_verified_here": False,
    "dependency_minimality_inferred_from_receipt_presence": False,
    "action_value_attention_or_saliency_is_finding": False,
    "clinical_term_authorized": False,
    "positive_onset_or_soz_claim_authorized": False,
    "report_claim_authorized": False,
}

_NONADDITIVE_TRAJECTORY_SEMANTICS: Final[dict[str, object]] = {
    "only_executed_selected_action_rows_included": True,
    "alternative_counterfactual_rows_included": False,
    "per_step_external_endpoint_deltas_preserved": True,
    "summed_external_endpoint_delta": None,
    "summed_external_endpoint_delta_permitted": False,
    "trajectory_return_computed_by_independent_delta_sum": False,
    "each_next_state_requires_actual_reveal_and_full_recompute": True,
    "final_endpoint_bound_to_last_revealed_snapshot": True,
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


def _exact_object(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValueError(
            f"{context} fields drifted; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed identifier")
    if len(value) > 256 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} is not a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum - _TOLERANCE:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _interval(value: object, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element list")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOLERANCE:
        raise ValueError(f"{context} must have positive duration")
    return [start, stop]


def _canonical_union(value: object, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    intervals = [
        _interval(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    intervals.sort(key=lambda item: (item[0], item[1]))
    union: list[list[float]] = []
    for start, stop in intervals:
        if not union or start > union[-1][1] + _TOLERANCE:
            union.append([start, stop])
        else:
            union[-1][1] = max(union[-1][1], stop)
    return union


def _same_union(left: object, right: object, context: str) -> bool:
    left_union = _canonical_union(left, f"{context}.left")
    right_union = _canonical_union(right, f"{context}.right")
    return len(left_union) == len(right_union) and all(
        abs(a[0] - b[0]) <= _TOLERANCE and abs(a[1] - b[1]) <= _TOLERANCE
        for a, b in zip(left_union, right_union)
    )


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _is_subset(
    subset: Sequence[Sequence[float]], superset: Sequence[Sequence[float]]
) -> bool:
    for interval in subset:
        if not any(
            interval[0] >= container[0] - _TOLERANCE
            and interval[1] <= container[1] + _TOLERANCE
            for container in superset
        ):
            return False
    return True


def _sorted_hashes(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    hashes = [_sha256(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if hashes != sorted(set(hashes)):
        raise ValueError(f"{context} must be unique and canonically sorted")
    return hashes


def _post_query_decision(value: object, context: str) -> dict[str, Any]:
    fields = {
        "decision",
        "reason_codes",
        "left_state",
        "right_state",
        "decision_receipt_sha256",
        "rule_closure_evaluated",
        "learned_stop_overrode_unclosed_rule",
    }
    data = _exact_object(value, fields, context)
    decision = data["decision"]
    if decision not in _POST_QUERY_DECISIONS:
        raise ValueError(f"{context}.decision is unsupported")
    reasons_raw = data["reason_codes"]
    if not isinstance(reasons_raw, list) or not reasons_raw:
        raise ValueError(f"{context}.reason_codes must be non-empty")
    reasons = [
        _identifier(reason, f"{context}.reason_codes[{index}]")
        for index, reason in enumerate(reasons_raw)
    ]
    if reasons != sorted(set(reasons)):
        raise ValueError(f"{context}.reason_codes must be sorted and unique")
    if any(reason not in _POST_QUERY_REASON_CODES for reason in reasons):
        raise ValueError(f"{context}.reason_codes contain an unsupported reason")
    left = data["left_state"]
    right = data["right_state"]
    if left not in _SIDE_STATES or right not in _SIDE_STATES:
        raise ValueError(f"{context} side closure state is unsupported")
    if decision == "stop" and "open" in {left, right}:
        raise ValueError(f"{context} cannot stop while one side remains open")
    if decision == "continue" and "open" not in {left, right}:
        raise ValueError(f"{context} cannot continue after both sides close")
    if "typed_censored" in {left, right} and not any(
        reason.endswith("_censor") or reason == "mixed_normal_and_censored_closure"
        for reason in reasons
    ):
        raise ValueError(f"{context} typed censor state lacks a censor reason")
    if data["rule_closure_evaluated"] is not True:
        raise ValueError(f"{context} lacks the mandatory rule-closure check")
    if data["learned_stop_overrode_unclosed_rule"] is not False:
        raise ValueError(f"{context} lets learned stop override an unclosed rule")
    return {
        "decision": decision,
        "reason_codes": reasons,
        "left_state": left,
        "right_state": right,
        "decision_receipt_sha256": _sha256(
            data["decision_receipt_sha256"], f"{context}.decision_receipt_sha256"
        ),
        "rule_closure_evaluated": True,
        "learned_stop_overrode_unclosed_rule": False,
    }


def _deletion_effect(
    value: object,
    context: str,
    *,
    raw_hashes: Sequence[str],
    dependency_union: Sequence[Sequence[float]],
    minimality_status: str,
) -> dict[str, Any]:
    fields = {
        "deleted_raw_dependency_sha256s",
        "deleted_raw_dependency_union_recording_seconds",
        "deletion_recompute_receipt_sha256",
        "effect_status",
        "atom_state_without_dependency",
        "measurement_delta",
        "measurement_unit",
        "native_eeg_remeasurement_used",
        "action_value_attention_or_saliency_used",
    }
    data = _exact_object(value, fields, context)
    deleted_hashes = _sorted_hashes(
        data["deleted_raw_dependency_sha256s"],
        f"{context}.deleted_raw_dependency_sha256s",
    )
    if list(raw_hashes) != deleted_hashes:
        raise ValueError(f"{context} did not delete the exact raw dependency roster")
    deleted_union = _canonical_union(
        data["deleted_raw_dependency_union_recording_seconds"],
        f"{context}.deleted_raw_dependency_union_recording_seconds",
    )
    if not _same_union(deleted_union, dependency_union, context):
        raise ValueError(f"{context} did not delete the exact dependency union")
    effect = data["effect_status"]
    if effect not in _DELETION_EFFECTS:
        raise ValueError(f"{context}.effect_status is unsupported")
    state = data["atom_state_without_dependency"]
    if state is not None and state not in _ATOM_STATES:
        raise ValueError(f"{context}.atom_state_without_dependency is unsupported")
    magnitude_raw = data["measurement_delta"]
    unit_raw = data["measurement_unit"]
    if magnitude_raw is None:
        magnitude = None
        if unit_raw is not None:
            raise ValueError(f"{context} measurement unit exists without a delta")
        unit = None
    else:
        magnitude = _finite(magnitude_raw, f"{context}.measurement_delta")
        unit = _identifier(unit_raw, f"{context}.measurement_unit")
    if effect == "atom_measurement_changes" and magnitude is None:
        raise ValueError(f"{context} measurement-change effect lacks a delta")
    if effect != "atom_measurement_changes" and magnitude is not None:
        raise ValueError(f"{context} non-measurement effect carries a numeric delta")
    if data["native_eeg_remeasurement_used"] is not True:
        raise ValueError(f"{context} deletion effect was not natively remeasured")
    if data["action_value_attention_or_saliency_used"] is not False:
        raise ValueError(f"{context} confuses model attribution with native deletion")
    if minimality_status == "proven_by_registered_deletion" and effect in {
        "no_material_effect",
        "not_evaluable",
    }:
        raise ValueError(
            f"{context} cannot claim proven minimality without a material deletion effect"
        )
    if minimality_status == "not_proven" and effect not in {
        "no_material_effect",
        "atom_measurement_changes",
    }:
        raise ValueError(
            f"{context} material state-loss effect must be marked proven or not_evaluable"
        )
    if minimality_status == "not_evaluable" and effect != "not_evaluable":
        raise ValueError(f"{context} minimality/effect evaluability disagree")
    return {
        "deleted_raw_dependency_sha256s": deleted_hashes,
        "deleted_raw_dependency_union_recording_seconds": deleted_union,
        "deletion_recompute_receipt_sha256": _sha256(
            data["deletion_recompute_receipt_sha256"],
            f"{context}.deletion_recompute_receipt_sha256",
        ),
        "effect_status": effect,
        "atom_state_without_dependency": state,
        "measurement_delta": magnitude,
        "measurement_unit": unit,
        "native_eeg_remeasurement_used": True,
        "action_value_attention_or_saliency_used": False,
    }


def _atom_observation(
    value: object,
    context: str,
    *,
    query_index: int,
    support_after: Sequence[Sequence[float]],
    proposed: Sequence[Sequence[float]],
) -> dict[str, Any]:
    fields = {
        "atom_id",
        "slot_id",
        "transition",
        "atom_state_after",
        "finding_atom_sha256_before",
        "finding_atom_sha256_after",
        "first_observed_query_index",
        "stabilized_query_index",
        "raw_dependency_sha256s",
        "candidate_minimal_raw_dependency_union_recording_seconds",
        "raw_dependency_union_receipt_sha256",
        "minimality_status",
        "minimality_test_receipt_sha256",
        "minimality_verification",
        "dependency_role",
        "deletion_effect",
    }
    data = _exact_object(value, fields, context)
    atom_id = _identifier(data["atom_id"], f"{context}.atom_id")
    slot_id = _identifier(data["slot_id"], f"{context}.slot_id")
    transition = data["transition"]
    if transition not in _ATOM_TRANSITIONS:
        raise ValueError(f"{context}.transition is unsupported")
    state = data["atom_state_after"]
    if state not in _ATOM_STATES:
        raise ValueError(f"{context}.atom_state_after is unsupported")
    before = _optional_sha256(
        data["finding_atom_sha256_before"], f"{context}.finding_atom_sha256_before"
    )
    after = _optional_sha256(
        data["finding_atom_sha256_after"], f"{context}.finding_atom_sha256_after"
    )
    first = _nonnegative_int(
        data["first_observed_query_index"],
        f"{context}.first_observed_query_index",
    )
    stabilized_raw = data["stabilized_query_index"]
    stabilized = (
        None
        if stabilized_raw is None
        else _nonnegative_int(stabilized_raw, f"{context}.stabilized_query_index")
    )
    if first > query_index or (stabilized is not None and stabilized > query_index):
        raise ValueError(f"{context} observation indices cannot lie in the future")
    if stabilized is not None and stabilized < first:
        raise ValueError(f"{context} stabilized before first observation")
    if transition == "first_observed":
        if (
            before is not None
            or after is None
            or first != query_index
            or stabilized is not None
        ):
            raise ValueError(f"{context} first_observed history is inconsistent")
    elif transition == "first_observed_and_stabilized":
        if (
            before is not None
            or after is None
            or first != query_index
            or stabilized != query_index
        ):
            raise ValueError(
                f"{context} first-observed/stabilized history is inconsistent"
            )
    elif transition == "updated_unstable":
        if (
            before is None
            or after is None
            or first >= query_index
            or stabilized is not None
        ):
            raise ValueError(f"{context} unstable update history is inconsistent")
    elif transition == "stabilized":
        if (
            before is None
            or after is None
            or first >= query_index
            or stabilized != query_index
        ):
            raise ValueError(f"{context} stabilization history is inconsistent")
    elif transition == "changed_after_stabilization":
        if (
            before is None
            or after is None
            or first >= query_index
            or stabilized is None
            or stabilized >= query_index
        ):
            raise ValueError(
                f"{context} post-stabilization change history is inconsistent"
            )
    else:
        if before is None or after is not None or first >= query_index:
            raise ValueError(f"{context} invalidation history is inconsistent")

    raw_hashes = _sorted_hashes(
        data["raw_dependency_sha256s"], f"{context}.raw_dependency_sha256s"
    )
    dependency_union = _canonical_union(
        data["candidate_minimal_raw_dependency_union_recording_seconds"],
        f"{context}.candidate_minimal_raw_dependency_union_recording_seconds",
    )
    if not dependency_union:
        raise ValueError(f"{context} has an empty raw dependency union")
    if not _is_subset(dependency_union, support_after):
        raise ValueError(f"{context} raw dependency escapes acquired support")
    if not any(
        _overlap(dependency, new_interval) > _TOLERANCE
        for dependency in dependency_union
        for new_interval in proposed
    ):
        raise ValueError(f"{context} transition has no dependency on the new query")
    minimality = data["minimality_status"]
    if minimality not in _MINIMALITY_STATUSES:
        raise ValueError(f"{context}.minimality_status is unsupported")
    role = data["dependency_role"]
    if role not in _DEPENDENCY_ROLES:
        raise ValueError(f"{context}.dependency_role is unsupported")
    expected_role = {
        "proven_by_registered_deletion": "necessary",
        "not_proven": {"supporting_nonnecessary", "no_detected_effect"},
        "not_evaluable": "not_evaluable",
    }[minimality]
    if isinstance(expected_role, set):
        if role not in expected_role:
            raise ValueError(f"{context} dependency role/minimality disagree")
    elif role != expected_role:
        raise ValueError(f"{context} dependency role/minimality disagree")
    verification = _exact_object(
        data["minimality_verification"],
        {
            "full_union_deletion_tested",
            "every_dependency_leave_one_out_tested",
            "temporal_edge_shrink_tests_completed",
            "native_eeg_remeasurement_used",
            "action_value_attention_or_saliency_used",
        },
        f"{context}.minimality_verification",
    )
    if (
        verification["native_eeg_remeasurement_used"] is not True
        or verification["action_value_attention_or_saliency_used"] is not False
    ):
        raise ValueError(f"{context} minimality verification is not native/replayable")
    proof_flags = (
        verification["full_union_deletion_tested"],
        verification["every_dependency_leave_one_out_tested"],
        verification["temporal_edge_shrink_tests_completed"],
    )
    if any(type(flag) is not bool for flag in proof_flags):
        raise TypeError(f"{context} minimality proof flags must be boolean")
    if minimality == "proven_by_registered_deletion" and not all(proof_flags):
        raise ValueError(
            f"{context} claims minimality without full/leave-one-out/edge-shrink tests"
        )
    if minimality == "not_evaluable" and any(proof_flags):
        raise ValueError(f"{context} not-evaluable minimality claims completed tests")
    deletion = _deletion_effect(
        data["deletion_effect"],
        f"{context}.deletion_effect",
        raw_hashes=raw_hashes,
        dependency_union=dependency_union,
        minimality_status=minimality,
    )
    if (
        minimality == "not_proven"
        and deletion["effect_status"] == "no_material_effect"
        and role != "no_detected_effect"
    ):
        raise ValueError(
            f"{context} no-effect deletion must use no_detected_effect role"
        )
    if (
        minimality == "not_proven"
        and deletion["effect_status"] != "no_material_effect"
        and role != "supporting_nonnecessary"
    ):
        raise ValueError(
            f"{context} nonzero unproven deletion must be supporting_nonnecessary"
        )
    return {
        "atom_id": atom_id,
        "slot_id": slot_id,
        "transition": transition,
        "atom_state_after": state,
        "finding_atom_sha256_before": before,
        "finding_atom_sha256_after": after,
        "first_observed_query_index": first,
        "stabilized_query_index": stabilized,
        "raw_dependency_sha256s": raw_hashes,
        "candidate_minimal_raw_dependency_union_recording_seconds": dependency_union,
        "raw_dependency_union_receipt_sha256": _sha256(
            data["raw_dependency_union_receipt_sha256"],
            f"{context}.raw_dependency_union_receipt_sha256",
        ),
        "minimality_status": minimality,
        "minimality_test_receipt_sha256": _sha256(
            data["minimality_test_receipt_sha256"],
            f"{context}.minimality_test_receipt_sha256",
        ),
        "minimality_verification": {
            "full_union_deletion_tested": verification["full_union_deletion_tested"],
            "every_dependency_leave_one_out_tested": verification[
                "every_dependency_leave_one_out_tested"
            ],
            "temporal_edge_shrink_tests_completed": verification[
                "temporal_edge_shrink_tests_completed"
            ],
            "native_eeg_remeasurement_used": True,
            "action_value_attention_or_saliency_used": False,
        },
        "dependency_role": role,
        "deletion_effect": deletion,
    }


def _step_evidence(
    value: object,
    context: str,
    *,
    target: Mapping[str, Any],
    support_before: Sequence[Sequence[float]],
    support_after: Sequence[Sequence[float]],
) -> dict[str, Any]:
    fields = {
        "target_id",
        "selected_action_execution_receipt_sha256",
        "endpoint_recompute_receipt_sha256",
        "boundary_posterior_delta_receipt_sha256",
        "typed_onset_distribution_delta_receipt_sha256",
        "atom_observations",
        "post_query_decision",
    }
    data = _exact_object(value, fields, context)
    if data["target_id"] != target["target_id"]:
        raise ValueError(f"{context} belongs to another target row")
    endpoint_receipt = _sha256(
        data["endpoint_recompute_receipt_sha256"],
        f"{context}.endpoint_recompute_receipt_sha256",
    )
    if (
        endpoint_receipt
        != target["revealed_snapshot"]["endpoint_recompute_receipt_sha256"]
    ):
        raise ValueError(f"{context} does not bind the revealed endpoint recomputation")
    observations_raw = data["atom_observations"]
    if not isinstance(observations_raw, list):
        raise TypeError(f"{context}.atom_observations must be a list")
    query_index = target["counterfactual_context"]["decision_index"]
    proposed = _canonical_union(
        target["action"]["proposed_intervals_recording_seconds"],
        f"{context}.proposed_intervals",
    )
    observations = [
        _atom_observation(
            row,
            f"{context}.atom_observations[{index}]",
            query_index=query_index,
            support_after=support_after,
            proposed=proposed,
        )
        for index, row in enumerate(observations_raw)
    ]
    atom_ids = [row["atom_id"] for row in observations]
    if atom_ids != sorted(atom_ids) or len(set(atom_ids)) != len(atom_ids):
        raise ValueError(
            f"{context}.atom_observations must be sorted by unique atom_id"
        )
    return {
        "target_id": target["target_id"],
        "selected_action_execution_receipt_sha256": _sha256(
            data["selected_action_execution_receipt_sha256"],
            f"{context}.selected_action_execution_receipt_sha256",
        ),
        "endpoint_recompute_receipt_sha256": endpoint_receipt,
        "boundary_posterior_delta_receipt_sha256": _sha256(
            data["boundary_posterior_delta_receipt_sha256"],
            f"{context}.boundary_posterior_delta_receipt_sha256",
        ),
        "typed_onset_distribution_delta_receipt_sha256": _sha256(
            data["typed_onset_distribution_delta_receipt_sha256"],
            f"{context}.typed_onset_distribution_delta_receipt_sha256",
        ),
        "support_before_recording_seconds": deepcopy(list(support_before)),
        "support_after_recording_seconds": deepcopy(list(support_after)),
        "atom_observations": observations,
        "new_atom_ids": sorted(
            row["atom_id"]
            for row in observations
            if row["transition"] in {"first_observed", "first_observed_and_stabilized"}
        ),
        "changed_atom_ids": sorted(
            row["atom_id"]
            for row in observations
            if row["transition"] in {"updated_unstable", "changed_after_stabilization"}
        ),
        "stabilized_atom_ids": sorted(
            row["atom_id"]
            for row in observations
            if row["transition"] in {"stabilized", "first_observed_and_stabilized"}
        ),
        "invalidated_atom_ids": sorted(
            row["atom_id"] for row in observations if row["transition"] == "invalidated"
        ),
        "post_query_decision": _post_query_decision(
            data["post_query_decision"], f"{context}.post_query_decision"
        ),
    }


def _cost_values(target: Mapping[str, Any]) -> dict[str, float | int]:
    cost = target["observed_cost_vector"]
    return {name: cost[name] for name in NSCEC_COST_NAMES_V1}


def _add_cost(
    cumulative: Mapping[str, float | int], incremental: Mapping[str, float | int]
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in NSCEC_COST_NAMES_V1:
        if name in {"native_samples", "model_tokens", "io_bytes"}:
            result[name] = int(cumulative[name]) + int(incremental[name])
        else:
            result[name] = float(cumulative[name]) + float(incremental[name])
    return result


def _validate_atom_history(
    observations: Sequence[Mapping[str, Any]],
    histories: dict[str, dict[str, Any]],
    *,
    query_index: int,
) -> None:
    for row in observations:
        atom_id = str(row["atom_id"])
        transition = str(row["transition"])
        previous = histories.get(atom_id)
        if transition in {"first_observed", "first_observed_and_stabilized"}:
            if previous is not None:
                raise ValueError("Finding atom was first-observed more than once")
            histories[atom_id] = {
                "first_observed_query_index": query_index,
                "stabilized_query_index": row["stabilized_query_index"],
                "last_atom_sha256": row["finding_atom_sha256_after"],
                "invalidated_query_index": None,
            }
            continue
        if previous is None:
            raise ValueError("Finding atom update appears before first observation")
        if previous["invalidated_query_index"] is not None:
            raise ValueError(
                "invalidated Finding atom cannot reappear under the same atom_id"
            )
        if row["first_observed_query_index"] != previous["first_observed_query_index"]:
            raise ValueError("Finding atom first-observed index drifted")
        if row["finding_atom_sha256_before"] != previous["last_atom_sha256"]:
            raise ValueError("Finding atom hash chain is discontinuous")
        previous_stable = previous["stabilized_query_index"]
        if transition == "stabilized":
            if previous_stable is not None:
                raise ValueError("Finding atom stabilized more than once")
            previous["stabilized_query_index"] = query_index
        elif transition == "updated_unstable":
            if previous_stable is not None:
                raise ValueError(
                    "stabilized atom cannot return to an unstable-update phase"
                )
        elif transition == "changed_after_stabilization":
            if (
                previous_stable is None
                or row["stabilized_query_index"] != previous_stable
            ):
                raise ValueError(
                    "post-stabilization change lost its stabilization history"
                )
        elif transition == "invalidated":
            if row["stabilized_query_index"] != previous_stable:
                raise ValueError("invalidation lost its stabilization history")
            previous["invalidated_query_index"] = query_index
        if (
            transition != "stabilized"
            and row["stabilized_query_index"] != previous_stable
        ):
            raise ValueError("Finding atom stabilization index drifted")
        previous["last_atom_sha256"] = row["finding_atom_sha256_after"]


def _build_trajectory(
    *,
    selected_target_rows: object,
    step_evidence_rows: object,
    trajectory_status: object,
) -> dict[str, Any]:
    if not isinstance(selected_target_rows, list) or not selected_target_rows:
        raise ValueError("selected_target_rows must be a non-empty list")
    if not isinstance(step_evidence_rows, list) or len(step_evidence_rows) != len(
        selected_target_rows
    ):
        raise ValueError(
            "step_evidence_rows must align one-to-one with selected targets"
        )
    if trajectory_status not in {"in_progress", "complete"}:
        raise ValueError("trajectory_status must be in_progress or complete")
    targets = [
        validate_nscec_external_endpoint_counterfactual_target_v1(row)
        for row in selected_target_rows
    ]
    identity = {
        (
            row["patient_uid"],
            row["recording_id"],
            row["event_id"],
            row["model_split"],
            row["source_data_manifest_sha256"],
            row["downstream_crossfit_checkpoint_binding"]["binding_sha256"],
        )
        for row in targets
    }
    if len(identity) != 1:
        raise ValueError(
            "one evidence-closure trajectory must retain one event/checkpoint"
        )
    decision_indices = [
        row["counterfactual_context"]["decision_index"] for row in targets
    ]
    if decision_indices != list(range(len(targets))):
        raise ValueError(
            "a complete NS-CEC trajectory must start at zero and be contiguous"
        )

    histories: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    cumulative: dict[str, float | int] = {
        name: 0 if name in {"native_samples", "model_tokens", "io_bytes"} else 0.0
        for name in NSCEC_COST_NAMES_V1
    }
    previous_target: Mapping[str, Any] | None = None
    previous_support_after: list[list[float]] | None = None
    for index, (target, evidence_raw) in enumerate(zip(targets, step_evidence_rows)):
        action = target["action"]
        support_before = _canonical_union(
            action["visible_intervals_recording_seconds"],
            f"target[{index}].visible_intervals",
        )
        support_after = _canonical_union(
            [
                *action["visible_intervals_recording_seconds"],
                *action["proposed_intervals_recording_seconds"],
            ],
            f"target[{index}].visible_plus_proposed_intervals",
        )
        if previous_target is not None:
            if (
                target["counterfactual_context"]["previous_selected_action_target_id"]
                != previous_target["target_id"]
            ):
                raise ValueError(
                    "next query does not name the actually selected prior row"
                )
            if (
                target["base_snapshot"]["input_evidence_union_sha256"]
                != previous_target["revealed_snapshot"]["input_evidence_union_sha256"]
            ):
                raise ValueError(
                    "next query base endpoint was not the prior revealed endpoint"
                )
            if (
                target["counterfactual_context"]["parent_state_receipt_sha256"]
                != previous_target["revealed_snapshot"][
                    "endpoint_recompute_receipt_sha256"
                ]
            ):
                raise ValueError(
                    "next query parent-state receipt is not the prior full recomputation"
                )
            previous_endpoint_state = deepcopy(previous_target["revealed_snapshot"])
            current_base_endpoint_state = deepcopy(target["base_snapshot"])
            # The same mathematical support can be serialized as adjacent
            # action intervals at q and as a merged interval at q+1.  The
            # roster receipt may therefore change, but no endpoint value or
            # recomputation receipt may silently change between steps.
            previous_endpoint_state.pop("evidence_interval_roster_sha256")
            current_base_endpoint_state.pop("evidence_interval_roster_sha256")
            if _canonical_json(previous_endpoint_state) != _canonical_json(
                current_base_endpoint_state
            ):
                raise ValueError(
                    "next query base endpoint values differ from the prior revealed state"
                )
            assert previous_support_after is not None
            if not _same_union(
                support_before,
                previous_support_after,
                f"trajectory support continuity step {index}",
            ):
                raise ValueError("next query support is not the prior acquired support")
        evidence = _step_evidence(
            evidence_raw,
            f"step_evidence_rows[{index}]",
            target=target,
            support_before=support_before,
            support_after=support_after,
        )
        _validate_atom_history(
            evidence["atom_observations"], histories, query_index=index
        )
        incremental = _cost_values(target)
        cumulative = _add_cost(cumulative, incremental)
        steps.append(
            {
                "query_index": index,
                "target_id": target["target_id"],
                "context_id": target["counterfactual_context"]["context_id"],
                "action": deepcopy(target["action"]),
                "support_before_recording_seconds": support_before,
                "support_after_recording_seconds": support_after,
                "external_endpoint_delta": deepcopy(
                    target["primary_external_endpoint_delta"]
                ),
                "native_finding_opportunity_gain": deepcopy(
                    target["native_finding_opportunity_gain"]
                ),
                "harm_raw_signed_increase": deepcopy(
                    target["harm_raw_signed_increase"]
                ),
                "incremental_cost": incremental,
                "cumulative_cost": deepcopy(cumulative),
                "evidence": evidence,
            }
        )
        previous_target = target
        previous_support_after = support_after

    decisions = [step["evidence"]["post_query_decision"]["decision"] for step in steps]
    if "stop" in decisions[:-1]:
        raise ValueError("trajectory continued after a terminal stop")
    if trajectory_status == "complete" and decisions[-1] != "stop":
        raise ValueError("complete trajectory lacks a terminal stop")
    if trajectory_status == "in_progress" and decisions[-1] != "continue":
        raise ValueError("in-progress trajectory cannot already be stopped")

    patient, recording, event, split, source_manifest, checkpoint_receipt = next(
        iter(identity)
    )
    final_target = targets[-1]
    atom_history_rows = [
        {
            "atom_id": atom_id,
            **deepcopy(history),
        }
        for atom_id, history in sorted(histories.items())
    ]
    body: dict[str, Any] = {
        "schema_version": NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_SCHEMA_VERSION_V1,
        "trajectory_id": "CONTENT-ADDRESS-PENDING",
        "method_id": NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_METHOD_ID_V1,
        "trajectory_status": trajectory_status,
        "patient_uid": patient,
        "recording_id": recording,
        "event_id": event,
        "model_split": split,
        "source_data_manifest_sha256": source_manifest,
        "downstream_crossfit_checkpoint_binding_sha256": checkpoint_receipt,
        "selected_target_rows": deepcopy(targets),
        "steps": steps,
        "atom_history": atom_history_rows,
        "final_support_recording_seconds": deepcopy(previous_support_after),
        "final_endpoint_snapshot_sha256": _canonical_sha256(
            final_target["revealed_snapshot"]
        ),
        "final_endpoint_recompute_receipt_sha256": final_target["revealed_snapshot"][
            "endpoint_recompute_receipt_sha256"
        ],
        "cumulative_observed_cost": cumulative,
        "nonadditive_trajectory_semantics": deepcopy(_NONADDITIVE_TRAJECTORY_SEMANTICS),
        "authorization": deepcopy(_TRAJECTORY_AUTHORIZATION),
    }
    body["trajectory_id"] = "NSCECTRAJ-" + _canonical_sha256(body)[:24]
    return body


def materialize_nscec_evidence_closure_trajectory_v1(
    *,
    selected_target_rows: Sequence[Mapping[str, Any]],
    step_evidence_rows: Sequence[Mapping[str, Any]],
    trajectory_status: str,
) -> dict[str, Any]:
    """Materialize an actual-selected-action reveal/recompute trajectory."""

    return _build_trajectory(
        selected_target_rows=list(selected_target_rows),
        step_evidence_rows=list(step_evidence_rows),
        trajectory_status=trajectory_status,
    )


def validate_nscec_evidence_closure_trajectory_v1(payload: object) -> dict[str, Any]:
    """Replay a serialized trajectory and all atom/deletion invariants."""

    fields = {
        "schema_version",
        "trajectory_id",
        "method_id",
        "trajectory_status",
        "patient_uid",
        "recording_id",
        "event_id",
        "model_split",
        "source_data_manifest_sha256",
        "downstream_crossfit_checkpoint_binding_sha256",
        "selected_target_rows",
        "steps",
        "atom_history",
        "final_support_recording_seconds",
        "final_endpoint_snapshot_sha256",
        "final_endpoint_recompute_receipt_sha256",
        "cumulative_observed_cost",
        "nonadditive_trajectory_semantics",
        "authorization",
    }
    data = _exact_object(payload, fields, "NS-CEC evidence-closure trajectory")
    if data["schema_version"] != NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_SCHEMA_VERSION_V1:
        raise ValueError("NS-CEC evidence-closure trajectory schema drifted")
    if data["method_id"] != NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_METHOD_ID_V1:
        raise ValueError("NS-CEC evidence-closure trajectory method drifted")
    selected_targets = data["selected_target_rows"]
    steps = data["steps"]
    if not isinstance(selected_targets, list) or not selected_targets:
        raise ValueError("serialized NS-CEC trajectory has no selected targets")
    if not isinstance(steps, list) or len(steps) != len(selected_targets):
        raise ValueError("serialized NS-CEC trajectory target/step length mismatch")
    evidence_rows = _extract_evidence_rows(steps)
    expected = _build_trajectory(
        selected_target_rows=selected_targets,
        step_evidence_rows=evidence_rows,
        trajectory_status=data["trajectory_status"],
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("NS-CEC evidence-closure trajectory does not replay")
    return expected


def _extract_evidence_rows(steps: Sequence[object]) -> list[dict[str, Any]]:
    evidence_rows: list[dict[str, Any]] = []
    for index, step_raw in enumerate(steps):
        step = _exact_object(
            step_raw,
            {
                "query_index",
                "target_id",
                "context_id",
                "action",
                "support_before_recording_seconds",
                "support_after_recording_seconds",
                "external_endpoint_delta",
                "native_finding_opportunity_gain",
                "harm_raw_signed_increase",
                "incremental_cost",
                "cumulative_cost",
                "evidence",
            },
            f"trajectory.steps[{index}]",
        )
        evidence = _exact_object(
            step["evidence"],
            {
                "target_id",
                "selected_action_execution_receipt_sha256",
                "endpoint_recompute_receipt_sha256",
                "boundary_posterior_delta_receipt_sha256",
                "typed_onset_distribution_delta_receipt_sha256",
                "support_before_recording_seconds",
                "support_after_recording_seconds",
                "atom_observations",
                "new_atom_ids",
                "changed_atom_ids",
                "stabilized_atom_ids",
                "invalidated_atom_ids",
                "post_query_decision",
            },
            f"trajectory.steps[{index}].evidence",
        )
        evidence_rows.append(
            {
                name: evidence[name]
                for name in (
                    "target_id",
                    "selected_action_execution_receipt_sha256",
                    "endpoint_recompute_receipt_sha256",
                    "boundary_posterior_delta_receipt_sha256",
                    "typed_onset_distribution_delta_receipt_sha256",
                    "atom_observations",
                    "post_query_decision",
                )
            }
        )
    return evidence_rows


def validate_nscec_evidence_closure_trajectory_with_targets_v1(
    payload: object,
    *,
    selected_target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay a trajectory against its content-addressed selected-row sidecar."""

    fields = {
        "schema_version",
        "trajectory_id",
        "method_id",
        "trajectory_status",
        "patient_uid",
        "recording_id",
        "event_id",
        "model_split",
        "source_data_manifest_sha256",
        "downstream_crossfit_checkpoint_binding_sha256",
        "selected_target_rows",
        "steps",
        "atom_history",
        "final_support_recording_seconds",
        "final_endpoint_snapshot_sha256",
        "final_endpoint_recompute_receipt_sha256",
        "cumulative_observed_cost",
        "nonadditive_trajectory_semantics",
        "authorization",
    }
    data = _exact_object(payload, fields, "NS-CEC evidence-closure trajectory")
    if data["schema_version"] != NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_SCHEMA_VERSION_V1:
        raise ValueError("NS-CEC evidence-closure trajectory schema drifted")
    if data["method_id"] != NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_METHOD_ID_V1:
        raise ValueError("NS-CEC evidence-closure trajectory method drifted")
    embedded_targets = data["selected_target_rows"]
    if not isinstance(embedded_targets, list) or len(embedded_targets) != len(
        selected_target_rows
    ):
        raise ValueError("trajectory/selected-target sidecar length mismatch")
    validated_sidecar = [
        validate_nscec_external_endpoint_counterfactual_target_v1(row)
        for row in selected_target_rows
    ]
    if _canonical_json(embedded_targets) != _canonical_json(validated_sidecar):
        raise ValueError(
            "selected-target sidecar differs from embedded trajectory rows"
        )
    steps = data["steps"]
    if not isinstance(steps, list) or len(steps) != len(selected_target_rows):
        raise ValueError("trajectory/selected-target sidecar length mismatch")
    evidence_rows = _extract_evidence_rows(steps)
    expected = _build_trajectory(
        selected_target_rows=list(selected_target_rows),
        step_evidence_rows=evidence_rows,
        trajectory_status=data["trajectory_status"],
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("NS-CEC evidence-closure trajectory does not replay")
    return expected


__all__ = [
    "NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_METHOD_ID_V1",
    "NSCEC_EVIDENCE_CLOSURE_TRAJECTORY_SCHEMA_VERSION_V1",
    "materialize_nscec_evidence_closure_trajectory_v1",
    "validate_nscec_evidence_closure_trajectory_v1",
    "validate_nscec_evidence_closure_trajectory_with_targets_v1",
]
