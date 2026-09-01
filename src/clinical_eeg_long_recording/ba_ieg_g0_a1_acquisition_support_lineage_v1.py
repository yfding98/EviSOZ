"""Reference-free G0a acquisition/support lineage for A1 candidates.

The prediction-first candidate roster and the stable-origin registry prevent a
reference onset from becoming the neural clock origin.  They do not, by
themselves, prove that the *support* later analysed by BA-IEG was selected
without a reference.  In particular, the existing boundary-adaptive decision
receipt binds a state and a selected action identifier, but has no execution,
raw-sample, source-signal, or QC binding.

This additive module closes the software part of that gap.  It builds a
content-addressed, pre-reference ledger which jointly binds:

* one candidate from the frozen prediction roster;
* the same candidate in the stable-origin registry;
* a deterministic candidate-envelope initial-support policy;
* every outer acquisition action and its decision/state hashes;
* the canonical final physical-support union; and
* interval-level raw-dependency, source-signal, and QC receipts.

An old decision-only outer receipt is retained as a typed
``pending_outer_execution_lineage`` event.  It is never silently upgraded to
verified.  A fully bound event is still not G0a-primary-admitted until a later
reference join commits to this pre-reference ledger in an immutable ordering
registry; that downstream binding is intentionally reported as pending here.

The builder has no field for a public event, physician label, spreadsheet,
EDF annotation, clinical text, behaviour, sleep, stimulation, or ECG input.
The output is a lineage/software receipt, not a seizure/SOZ label or a
clinical claim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_g0_a1_candidate_roster_v1 import (
    BA_IEG_G0_A1_RECORD_OUTCOMES,
    validate_ba_ieg_g0_a1_prediction_roster_v1,
)
from .ba_ieg_g0_support_relative_shortcut_surface_v1 import (
    BAIEGG0StableOriginRegistryV1,
)


BA_IEG_G0_A1_INITIAL_SUPPORT_POLICY_SCHEMA_V1: Final[str] = (
    "ba_ieg_g0_a1_reference_free_candidate_envelope_support_policy_v1"
)
BA_IEG_G0_A1_ACQUISITION_SUPPORT_LINEAGE_SCHEMA_V1: Final[str] = (
    "ba_ieg_g0_a1_reference_free_acquisition_support_lineage_v1"
)

BA_IEG_G0_A1_OUTER_RECEIPT_CAPABILITIES_V1: Final[tuple[str, ...]] = (
    "decision_only_v1",
    "reference_free_execution_bound_v1",
)
BA_IEG_G0_A1_SUPPORT_BINDING_STATUSES_V1: Final[tuple[str, ...]] = (
    "verified_reference_free_execution_chain",
    "pending_outer_execution_lineage",
    "pending_raw_dependency_binding",
    "pending_stable_origin_registry",
    "pending_reference_free_split_child_registry",
)

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-8
_INITIAL_DEPENDENCY_ACTION_ID: Final[str] = "initial_candidate_support"
_ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"query_left", "query_right", "retrieve_distant_background"}
)
_ACTION_SCOPE: Final[dict[str, Any]] = {
    "eeg_signal_only": True,
    "public_event_reference_used": False,
    "edf_annotations_used": False,
    "excel_used": False,
    "physician_labels_or_report_used": False,
    "clinical_context_used": False,
    "video_or_semiology_used": False,
    "sleep_stimulation_or_ecg_used": False,
}

_EVENT_REQUEST_FIELDS: Final[set[str]] = {
    "event_id",
    "patient_uid",
    "recording_id",
    "candidate_id",
    "parent_candidate_id",
    "declared_final_support_union_recording_seconds",
    "raw_dependencies",
    "outer_actions",
}
_RAW_DEPENDENCY_FIELDS: Final[set[str]] = {
    "dependency_id",
    "acquisition_action_id",
    "interval_recording_seconds",
    "source_signal_sha256",
    "raw_dependency_sha256",
    "qc_receipt_sha256",
}
_OUTER_ACTION_FIELDS: Final[set[str]] = {
    "step_index",
    "action_id",
    "action_type",
    "acquired_intervals_recording_seconds",
    "outer_state_sha256",
    "outer_decision_id",
    "outer_decision_sha256",
    "receipt_capability",
    "raw_dependency_ids",
    "scope_receipt",
}
_OUTPUT_RECORD_FIELDS: Final[set[str]] = {
    "patient_uid",
    "recording_id",
    "recording_duration_seconds",
    "source_signal_sha256",
    "prediction_outcome",
    "failure_stage",
    "detector_candidate_count",
    "event_ids",
    "record_lineage_status",
}
_OUTPUT_EVENT_FIELDS: Final[set[str]] = {
    "event_id",
    "patient_uid",
    "recording_id",
    "candidate_id",
    "parent_candidate_id",
    "candidate_origin",
    "candidate_anchor_recording_seconds_output_only",
    "source_candidate_receipt_sha256",
    "stable_origin_registry_receipt_sha256",
    "initial_support_policy_receipt_sha256",
    "initial_support_union_recording_seconds",
    "outer_actions",
    "final_support_union_recording_seconds",
    "final_support_seconds",
    "final_support_union_sha256",
    "source_signal_sha256",
    "raw_dependencies",
    "raw_dependency_roster_sha256",
    "qc_receipt_roster_sha256",
    "acquisition_chain_tip_sha256",
    "support_binding_status",
    "pending_reason_codes",
    "reference_join_order_status",
    "primary_admitted",
}
_OUTPUT_ACTION_FIELDS: Final[set[str]] = _OUTER_ACTION_FIELDS.union(
    {
        "pre_support_union_recording_seconds",
        "post_support_union_recording_seconds",
        "execution_receipt_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_object(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} must be a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _interval(value: object, *, duration: float, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element interval")
    start = _finite(value[0], f"{context}[0]")
    stop = _finite(value[1], f"{context}[1]")
    if start < -_TOLERANCE or stop > duration + _TOLERANCE or stop <= start:
        raise ValueError(f"{context} lies outside the recording or is empty")
    return [max(0.0, start), min(duration, stop)]


def _canonical_union(
    value: object, *, duration: float, context: str
) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context} must be an interval sequence")
    rows = [
        _interval(item, duration=duration, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]
    rows.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for start, stop in rows:
        if not merged or start > merged[-1][1] + _TOLERANCE:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return merged


def _same_union(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a[0]) - float(b[0])) <= _TOLERANCE
        and abs(float(a[1]) - float(b[1])) <= _TOLERANCE
        for a, b in zip(left, right)
    )


def _union_seconds(value: Sequence[Sequence[float]]) -> float:
    return sum(float(stop) - float(start) for start, stop in value)


def _seal(body: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = prefix + "-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    id_source = deepcopy(result)
    result[id_field] = prefix + "-" + _canonical_sha256(id_source)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


def build_ba_ieg_g0_a1_initial_support_policy_v1(
    *, left_margin_seconds: float, right_margin_seconds: float
) -> dict[str, Any]:
    """Freeze target-free margins around the detector candidate envelope."""

    left = _finite(left_margin_seconds, "left support margin")
    right = _finite(right_margin_seconds, "right support margin")
    if left < 0 or right < 0:
        raise ValueError("initial support margins must be non-negative")
    body = {
        "schema_version": BA_IEG_G0_A1_INITIAL_SUPPORT_POLICY_SCHEMA_V1,
        "policy_id": "BAIEGG0A1SUPPOL-PENDING",
        "left_margin_seconds": left,
        "right_margin_seconds": right,
        "geometry_rule": "candidate_envelope_plus_fixed_margins_clipped_to_recording",
        "reference_fields_available": [],
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="policy_id", prefix="BAIEGG0A1SUPPOL")
    validate_ba_ieg_g0_a1_initial_support_policy_v1(result)
    return result


def validate_ba_ieg_g0_a1_initial_support_policy_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "policy_id",
        "left_margin_seconds",
        "right_margin_seconds",
        "geometry_rule",
        "reference_fields_available",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0a initial-support policy")
    if data["schema_version"] != BA_IEG_G0_A1_INITIAL_SUPPORT_POLICY_SCHEMA_V1:
        raise ValueError("initial-support policy schema drifted")
    if _finite(data["left_margin_seconds"], "left margin") < 0 or _finite(
        data["right_margin_seconds"], "right margin"
    ) < 0:
        raise ValueError("initial-support policy margin is negative")
    if data["geometry_rule"] != (
        "candidate_envelope_plus_fixed_margins_clipped_to_recording"
    ) or data["reference_fields_available"] != []:
        raise ValueError("initial-support policy became target-conditioned")
    expected = _seal(data, id_field="policy_id", prefix="BAIEGG0A1SUPPOL")
    if data["policy_id"] != expected["policy_id"] or data["receipt_sha256"] != expected[
        "receipt_sha256"
    ]:
        raise ValueError("initial-support policy content address does not replay")
    return data


def _normalize_dependency(
    value: object, *, duration: float, source_signal_sha256: str, index: int
) -> dict[str, Any]:
    row = _strict_object(value, _RAW_DEPENDENCY_FIELDS, f"raw dependency {index}")
    observed_source = _sha256(row["source_signal_sha256"], "dependency source signal")
    if observed_source != source_signal_sha256:
        raise ValueError("raw dependency source signal does not match prediction roster")
    return {
        "dependency_id": _identifier(row["dependency_id"], "dependency ID"),
        "acquisition_action_id": _identifier(
            row["acquisition_action_id"], "dependency acquisition action ID"
        ),
        "interval_recording_seconds": _interval(
            row["interval_recording_seconds"],
            duration=duration,
            context="raw dependency interval",
        ),
        "source_signal_sha256": observed_source,
        "raw_dependency_sha256": _sha256(
            row["raw_dependency_sha256"], "raw dependency receipt"
        ),
        "qc_receipt_sha256": _sha256(row["qc_receipt_sha256"], "QC receipt"),
    }


def _normalize_action(value: object, *, duration: float, index: int) -> dict[str, Any]:
    row = _strict_object(value, _OUTER_ACTION_FIELDS, f"outer action {index}")
    if isinstance(row["step_index"], bool) or not isinstance(row["step_index"], int):
        raise TypeError("outer action step_index must be an integer")
    action_type = row["action_type"]
    if action_type not in _ACTION_TYPES:
        raise ValueError("outer action is not an interval-acquisition action")
    capability = row["receipt_capability"]
    if capability not in BA_IEG_G0_A1_OUTER_RECEIPT_CAPABILITIES_V1:
        raise ValueError("outer receipt capability is unsupported")
    if row["scope_receipt"] != _ACTION_SCOPE:
        raise ValueError("outer action violates the EEG-only reference-free firewall")
    dependency_ids = row["raw_dependency_ids"]
    if not isinstance(dependency_ids, list) or len(dependency_ids) != len(
        set(dependency_ids)
    ):
        raise ValueError("outer action raw dependency IDs must be a unique list")
    return {
        "step_index": row["step_index"],
        "action_id": _identifier(row["action_id"], "outer action ID"),
        "action_type": action_type,
        "acquired_intervals_recording_seconds": _canonical_union(
            row["acquired_intervals_recording_seconds"],
            duration=duration,
            context="outer acquired intervals",
        ),
        "outer_state_sha256": _sha256(row["outer_state_sha256"], "outer state"),
        "outer_decision_id": _identifier(
            row["outer_decision_id"], "outer decision ID"
        ),
        "outer_decision_sha256": _sha256(
            row["outer_decision_sha256"], "outer decision"
        ),
        "receipt_capability": capability,
        "raw_dependency_ids": [
            _identifier(item, "outer raw dependency ID") for item in dependency_ids
        ],
        "scope_receipt": deepcopy(_ACTION_SCOPE),
    }


def _initial_support(
    candidate: Mapping[str, Any], policy: Mapping[str, Any], duration: float
) -> list[list[float]]:
    return [
        [
            max(
                0.0,
                float(candidate["start_offset_seconds"])
                - float(policy["left_margin_seconds"]),
            ),
            min(
                duration,
                float(candidate["stop_offset_seconds"])
                + float(policy["right_margin_seconds"]),
            ),
        ]
    ]


def _normalize_event_request(
    value: object,
    *,
    roster: Mapping[str, Any],
    policy: Mapping[str, Any],
    stable_lineage_by_event: Mapping[str, Mapping[str, Any]],
    stable_registry_receipt_sha256: str | None,
    index: int,
) -> dict[str, Any]:
    request = _strict_object(value, _EVENT_REQUEST_FIELDS, f"event request {index}")
    event_id = _identifier(request["event_id"], "event ID")
    patient_uid = _identifier(request["patient_uid"], "event patient UID")
    recording_id = _identifier(request["recording_id"], "event recording ID")
    candidate_id = _identifier(request["candidate_id"], "event candidate ID")
    parent_id_raw = request["parent_candidate_id"]
    parent_id = (
        None
        if parent_id_raw is None
        else _identifier(parent_id_raw, "split parent candidate ID")
    )
    records = {row["recording_id"]: row for row in roster["records"]}
    candidates = {row["candidate_id"]: row for row in roster["candidates"]}
    record = records.get(recording_id)
    if record is None or record["patient_uid"] != patient_uid:
        raise ValueError("event request crosses prediction record/patient identity")
    duration = float(record["recording_duration_seconds"])

    pending_reasons: list[str] = []
    if parent_id is None:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("event candidate is absent from the prediction roster")
        if (
            candidate["patient_uid"] != patient_uid
            or candidate["recording_id"] != recording_id
        ):
            raise ValueError("event candidate crosses prediction identity")
        stable = stable_lineage_by_event.get(event_id)
        if stable is None:
            pending_reasons.append("stable_origin_registry_row_not_materialized")
        elif (
            stable["candidate_id"] != candidate_id
            or stable["patient_uid"] != patient_uid
            or stable["recording_id"] != recording_id
            or stable["source_candidate_receipt_sha256"]
            != candidate["source_candidate_receipt_sha256"]
        ):
            raise ValueError("stable-origin row does not bind the prediction candidate")
    else:
        if candidate_id in candidates:
            raise ValueError("split child ID aliases a frozen parent candidate")
        candidate = candidates.get(parent_id)
        if candidate is None:
            raise ValueError("split parent candidate is absent from prediction roster")
        if (
            candidate["patient_uid"] != patient_uid
            or candidate["recording_id"] != recording_id
        ):
            raise ValueError("split parent crosses prediction identity")
        pending_reasons.append("reference_free_split_child_registry_not_materialized")

    initial = _initial_support(candidate, policy, duration)
    dependencies = [
        _normalize_dependency(
            row,
            duration=duration,
            source_signal_sha256=record["source_signal_sha256"],
            index=dependency_index,
        )
        for dependency_index, row in enumerate(request["raw_dependencies"])
    ]
    dependencies.sort(
        key=lambda row: (
            row["acquisition_action_id"],
            row["interval_recording_seconds"][0],
            row["dependency_id"],
        )
    )
    dependency_ids = [row["dependency_id"] for row in dependencies]
    if len(dependency_ids) != len(set(dependency_ids)):
        raise ValueError("event raw dependency IDs are not unique")
    dependencies_by_action: dict[str, list[dict[str, Any]]] = {}
    for row in dependencies:
        dependencies_by_action.setdefault(row["acquisition_action_id"], []).append(row)

    initial_dependencies = dependencies_by_action.get(_INITIAL_DEPENDENCY_ACTION_ID, [])
    initial_dependency_union = _canonical_union(
        [row["interval_recording_seconds"] for row in initial_dependencies],
        duration=duration,
        context="initial raw dependency union",
    )
    if not initial_dependencies or not _same_union(initial, initial_dependency_union):
        pending_reasons.append("initial_support_raw_dependency_not_fully_bound")

    actions = [
        _normalize_action(row, duration=duration, index=action_index)
        for action_index, row in enumerate(request["outer_actions"])
    ]
    actions.sort(key=lambda row: row["step_index"])
    if [row["step_index"] for row in actions] != list(range(len(actions))):
        raise ValueError("outer action steps must be contiguous from zero")
    if len({row["action_id"] for row in actions}) != len(actions):
        raise ValueError("outer action IDs are not unique")

    event_context = list(initial[0])
    backgrounds: list[list[float]] = []
    current_support = deepcopy(initial)
    normalized_actions: list[dict[str, Any]] = []
    predecessor_receipt = _canonical_sha256(
        {
            "schema": "ba_ieg_g0_a1_initial_support_freeze_v1",
            "candidate_receipt_sha256": candidate[
                "source_candidate_receipt_sha256"
            ],
            "initial_support_policy_receipt_sha256": policy["receipt_sha256"],
            "initial_support_union_recording_seconds": initial,
            "raw_dependency_sha256s": [
                row["raw_dependency_sha256"] for row in initial_dependencies
            ],
            "qc_receipt_sha256s": [
                row["qc_receipt_sha256"] for row in initial_dependencies
            ],
        }
    )
    known_action_ids = {_INITIAL_DEPENDENCY_ACTION_ID}
    for action in actions:
        action_id = action["action_id"]
        known_action_ids.add(action_id)
        acquired = action["acquired_intervals_recording_seconds"]
        if not acquired:
            raise ValueError("outer acquisition action has no physical interval")
        if action["action_type"] == "query_left":
            if len(acquired) != 1 or abs(acquired[0][1] - event_context[0]) > _TOLERANCE:
                raise ValueError("query_left is not contiguous with the event support")
            event_context[0] = acquired[0][0]
        elif action["action_type"] == "query_right":
            if len(acquired) != 1 or abs(acquired[0][0] - event_context[1]) > _TOLERANCE:
                raise ValueError("query_right is not contiguous with the event support")
            event_context[1] = acquired[0][1]
        else:
            if any(
                min(interval[1], visible[1]) - max(interval[0], visible[0])
                > _TOLERANCE
                for interval in acquired
                for visible in current_support
            ):
                raise ValueError("distant-background acquisition overlaps visible support")
            backgrounds.extend(acquired)

        action_dependencies = dependencies_by_action.get(action_id, [])
        listed_ids = action["raw_dependency_ids"]
        observed_ids = [row["dependency_id"] for row in action_dependencies]
        if listed_ids != observed_ids:
            raise ValueError("outer action raw-dependency roster does not replay")
        dependency_union = _canonical_union(
            [row["interval_recording_seconds"] for row in action_dependencies],
            duration=duration,
            context="outer action dependency union",
        )
        capability = action["receipt_capability"]
        execution_receipt: str | None = None
        pre_support = deepcopy(current_support)
        current_support = _canonical_union(
            [*current_support, *acquired],
            duration=duration,
            context="post-action support union",
        )
        if capability == "decision_only_v1":
            pending_reasons.append(
                f"outer_action:{action_id}:decision_receipt_has_no_execution_binding"
            )
        elif not action_dependencies or not _same_union(acquired, dependency_union):
            pending_reasons.append(
                f"outer_action:{action_id}:raw_dependency_not_fully_bound"
            )
        else:
            execution_receipt = _canonical_sha256(
                {
                    "schema": "ba_ieg_g0_a1_reference_free_outer_execution_receipt_v1",
                    "event_id": event_id,
                    "candidate_id": candidate_id,
                    "parent_candidate_id": parent_id,
                    "step_index": action["step_index"],
                    "action_id": action_id,
                    "action_type": action["action_type"],
                    "outer_state_sha256": action["outer_state_sha256"],
                    "outer_decision_id": action["outer_decision_id"],
                    "outer_decision_sha256": action["outer_decision_sha256"],
                    "pre_support_union_recording_seconds": pre_support,
                    "acquired_intervals_recording_seconds": acquired,
                    "post_support_union_recording_seconds": current_support,
                    "source_signal_sha256": record["source_signal_sha256"],
                    "raw_dependency_receipts": [
                        {
                            "dependency_id": row["dependency_id"],
                            "raw_dependency_sha256": row["raw_dependency_sha256"],
                            "qc_receipt_sha256": row["qc_receipt_sha256"],
                        }
                        for row in action_dependencies
                    ],
                    "predecessor_execution_receipt_sha256": predecessor_receipt,
                    "scope_receipt": _ACTION_SCOPE,
                }
            )
            predecessor_receipt = execution_receipt
        normalized_actions.append(
            {
                **action,
                "pre_support_union_recording_seconds": pre_support,
                "post_support_union_recording_seconds": deepcopy(current_support),
                "execution_receipt_sha256": execution_receipt,
            }
        )

    unknown_dependency_actions = set(dependencies_by_action).difference(known_action_ids)
    if unknown_dependency_actions:
        raise ValueError("raw dependency references an unknown acquisition action")
    declared_final = _canonical_union(
        request["declared_final_support_union_recording_seconds"],
        duration=duration,
        context="declared final support union",
    )
    final_support = _canonical_union(
        [event_context, *backgrounds],
        duration=duration,
        context="derived final support union",
    )
    if not _same_union(declared_final, final_support):
        raise ValueError("declared final support does not replay acquisition geometry")

    all_dependency_union = _canonical_union(
        [row["interval_recording_seconds"] for row in dependencies],
        duration=duration,
        context="event raw dependency union",
    )
    if not dependencies or not _same_union(all_dependency_union, final_support):
        pending_reasons.append("final_support_raw_dependency_union_incomplete")

    if any("split_child" in reason for reason in pending_reasons):
        support_status = "pending_reference_free_split_child_registry"
    elif any("stable_origin" in reason for reason in pending_reasons):
        support_status = "pending_stable_origin_registry"
    elif any("decision_receipt" in reason for reason in pending_reasons):
        support_status = "pending_outer_execution_lineage"
    elif pending_reasons:
        support_status = "pending_raw_dependency_binding"
    else:
        support_status = "verified_reference_free_execution_chain"

    return {
        "event_id": event_id,
        "patient_uid": patient_uid,
        "recording_id": recording_id,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_id,
        "candidate_origin": candidate["origin"],
        "candidate_anchor_recording_seconds_output_only": candidate[
            "anchor_offset_seconds"
        ],
        "source_candidate_receipt_sha256": candidate[
            "source_candidate_receipt_sha256"
        ],
        "stable_origin_registry_receipt_sha256": stable_registry_receipt_sha256,
        "initial_support_policy_receipt_sha256": policy["receipt_sha256"],
        "initial_support_union_recording_seconds": initial,
        "outer_actions": normalized_actions,
        "final_support_union_recording_seconds": final_support,
        "final_support_seconds": _union_seconds(final_support),
        "final_support_union_sha256": _canonical_sha256(
            {
                "recording_id": recording_id,
                "source_signal_sha256": record["source_signal_sha256"],
                "support_union_recording_seconds": final_support,
            }
        ),
        "source_signal_sha256": record["source_signal_sha256"],
        "raw_dependencies": dependencies,
        "raw_dependency_roster_sha256": _canonical_sha256(dependencies),
        "qc_receipt_roster_sha256": _canonical_sha256(
            [
                {
                    "dependency_id": row["dependency_id"],
                    "qc_receipt_sha256": row["qc_receipt_sha256"],
                }
                for row in dependencies
            ]
        ),
        "acquisition_chain_tip_sha256": predecessor_receipt,
        "support_binding_status": support_status,
        "pending_reason_codes": sorted(set(pending_reasons)),
        "reference_join_order_status": "pending_downstream_join_binding_to_this_ledger",
        "primary_admitted": False,
    }


def build_ba_ieg_g0_a1_acquisition_support_lineage_v1(
    *,
    prediction_roster: Mapping[str, Any],
    stable_origin_registry: BAIEGG0StableOriginRegistryV1 | None,
    initial_support_policy: Mapping[str, Any],
    event_requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the pre-reference acquisition/support freeze for the full roster."""

    roster = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    policy = validate_ba_ieg_g0_a1_initial_support_policy_v1(
        dict(initial_support_policy)
    )
    if not isinstance(event_requests, Sequence) or isinstance(
        event_requests, (str, bytes)
    ):
        raise TypeError("event requests must be a sequence")

    stable_rows: dict[str, Mapping[str, Any]] = {}
    stable_receipt: str | None = None
    if stable_origin_registry is not None:
        if not isinstance(stable_origin_registry, BAIEGG0StableOriginRegistryV1):
            raise TypeError("stable origin registry has an unsupported type")
        stable_origin_registry.verify_integrity()
        if (
            stable_origin_registry.prediction_roster_id != roster["roster_id"]
            or stable_origin_registry.prediction_roster_receipt_sha256
            != roster["receipt_sha256"]
        ):
            raise ValueError("stable origin registry crosses prediction freeze")
        stable_rows = {
            row["event_id"]: row for row in stable_origin_registry.provider_lineage()
        }
        stable_receipt = stable_origin_registry.receipt_sha256

    events = [
        _normalize_event_request(
            request,
            roster=roster,
            policy=policy,
            stable_lineage_by_event=stable_rows,
            stable_registry_receipt_sha256=stable_receipt,
            index=index,
        )
        for index, request in enumerate(event_requests)
    ]
    events.sort(key=lambda row: (row["patient_uid"], row["recording_id"], row["event_id"]))
    if len({row["event_id"] for row in events}) != len(events):
        raise ValueError("acquisition lineage repeats an event ID")
    if len({row["candidate_id"] for row in events}) != len(events):
        raise ValueError("one event candidate is reused without a child registry")

    event_ids_by_record: dict[str, list[str]] = {
        row["recording_id"]: [] for row in roster["records"]
    }
    for row in events:
        event_ids_by_record[row["recording_id"]].append(row["event_id"])
    records: list[dict[str, Any]] = []
    for record in roster["records"]:
        event_ids = sorted(event_ids_by_record[record["recording_id"]])
        outcome = record["outcome"]
        if outcome == "technical_failure":
            status = "not_evaluable_detector_technical_failure"
        elif outcome == "completed_zero_candidate":
            status = "not_evaluable_zero_detector_candidate"
        elif outcome == "partial_coverage":
            status = "pending_detector_partial_coverage"
        elif not event_ids:
            status = "pending_no_event_acquisition_request"
        else:
            status = "event_lineage_rows_materialized"
        records.append(
            {
                "patient_uid": record["patient_uid"],
                "recording_id": record["recording_id"],
                "recording_duration_seconds": record[
                    "recording_duration_seconds"
                ],
                "source_signal_sha256": record["source_signal_sha256"],
                "prediction_outcome": outcome,
                "failure_stage": record["failure_stage"],
                "detector_candidate_count": len(record["detector_candidate_ids"]),
                "event_ids": event_ids,
                "record_lineage_status": status,
            }
        )

    support_counts = {
        status: sum(row["support_binding_status"] == status for row in events)
        for status in BA_IEG_G0_A1_SUPPORT_BINDING_STATUSES_V1
    }
    body = {
        "schema_version": BA_IEG_G0_A1_ACQUISITION_SUPPORT_LINEAGE_SCHEMA_V1,
        "lineage_id": "BAIEGG0A1SUPPORT-PENDING",
        "prediction_roster_id": roster["roster_id"],
        "prediction_roster_receipt_sha256": roster["receipt_sha256"],
        "provider_prediction_receipt_sha256": roster[
            "provider_prediction_receipt_sha256"
        ],
        "decoder_policy_receipt_sha256": roster["decoder_policy_receipt_sha256"],
        "stable_origin_registry_receipt_sha256": stable_receipt,
        "initial_support_policy": policy,
        "records": records,
        "events": events,
        "counts": {
            "patients": roster["counts"]["patients"],
            "records": roster["counts"]["records"],
            "events": len(events),
            "support_binding_statuses": support_counts,
            "zero_detector_candidate_records": sum(
                row["prediction_outcome"] == "completed_zero_candidate"
                for row in records
            ),
            "technical_failure_records": sum(
                row["prediction_outcome"] == "technical_failure" for row in records
            ),
        },
        "pre_reference_freeze_barrier": {
            "accepted_input_surface": "prediction_roster_stable_origin_outer_action_signal_dependency_qc_only",
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "excel_or_spreadsheet_opened": 0,
            "physician_labels_or_reports_opened": 0,
            "clinical_text_or_context_opened": 0,
            "video_or_semiology_opened": 0,
            "sleep_stimulation_or_ecg_opened": 0,
            "support_actions_content_addressed_in_this_prejoin_ledger": True,
            "existing_outer_decision_v1_alone_is_execution_proof": False,
            "downstream_reference_join_commits_to_this_ledger": False,
            "immutable_cross_registry_order_receipt_materialized": False,
        },
        "admission": {
            "reference_free_support_software_lineage_complete": bool(events)
            and all(
                row["support_binding_status"]
                == "verified_reference_free_execution_chain"
                for row in events
            ),
            "downstream_reference_join_binding_complete": False,
            "g0a_primary_admitted": False,
            "pending_reason": "postfreeze_reference_join_must_bind_this_pre_reference_ledger",
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="lineage_id", prefix="BAIEGG0A1SUPPORT")
    validate_ba_ieg_g0_a1_acquisition_support_lineage_v1(result)
    return result


def validate_ba_ieg_g0_a1_acquisition_support_lineage_v1(
    payload: object,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "lineage_id",
        "prediction_roster_id",
        "prediction_roster_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "decoder_policy_receipt_sha256",
        "stable_origin_registry_receipt_sha256",
        "initial_support_policy",
        "records",
        "events",
        "counts",
        "pre_reference_freeze_barrier",
        "admission",
        "receipt_sha256",
    }
    data = _strict_object(payload, fields, "G0a acquisition/support lineage")
    if data["schema_version"] != BA_IEG_G0_A1_ACQUISITION_SUPPORT_LINEAGE_SCHEMA_V1:
        raise ValueError("G0a acquisition/support lineage schema drifted")
    _identifier(data["prediction_roster_id"], "prediction roster ID")
    for name in (
        "prediction_roster_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "decoder_policy_receipt_sha256",
    ):
        _sha256(data[name], name)
    if data["stable_origin_registry_receipt_sha256"] is not None:
        _sha256(data["stable_origin_registry_receipt_sha256"], "stable-origin registry")
    validate_ba_ieg_g0_a1_initial_support_policy_v1(data["initial_support_policy"])
    if not isinstance(data["records"], list) or not isinstance(data["events"], list):
        raise TypeError("G0a records/events must be lists")
    if len({row["recording_id"] for row in data["records"]}) != len(data["records"]):
        raise ValueError("G0a record denominator repeats a recording")
    if len({row["event_id"] for row in data["events"]}) != len(data["events"]):
        raise ValueError("G0a event denominator repeats an event")
    record_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(data["records"]):
        row = _strict_object(raw, _OUTPUT_RECORD_FIELDS, f"G0a record {index}")
        _identifier(row["patient_uid"], "G0a record patient UID")
        recording_id = _identifier(row["recording_id"], "G0a recording ID")
        duration = _finite(row["recording_duration_seconds"], "G0a recording duration")
        if duration <= 0:
            raise ValueError("G0a recording duration must be positive")
        _sha256(row["source_signal_sha256"], "G0a record source signal")
        if row["prediction_outcome"] not in BA_IEG_G0_A1_RECORD_OUTCOMES:
            raise ValueError("G0a record outcome drifted")
        if not isinstance(row["event_ids"], list) or row["event_ids"] != sorted(
            set(row["event_ids"])
        ):
            raise ValueError("G0a record event IDs must be uniquely sorted")
        for event_id in row["event_ids"]:
            _identifier(event_id, "G0a record event ID")
        expected_record_status = (
            "not_evaluable_detector_technical_failure"
            if row["prediction_outcome"] == "technical_failure"
            else "not_evaluable_zero_detector_candidate"
            if row["prediction_outcome"] == "completed_zero_candidate"
            else "pending_detector_partial_coverage"
            if row["prediction_outcome"] == "partial_coverage"
            else "event_lineage_rows_materialized"
            if row["event_ids"]
            else "pending_no_event_acquisition_request"
        )
        if row["record_lineage_status"] != expected_record_status:
            raise ValueError("G0a record lineage status does not replay")
        record_by_id[recording_id] = row

    observed_event_ids_by_record: dict[str, list[str]] = {
        recording_id: [] for recording_id in record_by_id
    }
    for index, raw in enumerate(data["events"]):
        row = _strict_object(raw, _OUTPUT_EVENT_FIELDS, f"G0a event {index}")
        event_id = _identifier(row["event_id"], "G0a event ID")
        recording_id = _identifier(row["recording_id"], "G0a event recording ID")
        record = record_by_id.get(recording_id)
        if record is None or row["patient_uid"] != record["patient_uid"]:
            raise ValueError("G0a event crosses record/patient identity")
        if row["source_signal_sha256"] != record["source_signal_sha256"]:
            raise ValueError("G0a event source signal crosses its record")
        duration = float(record["recording_duration_seconds"])
        observed_event_ids_by_record[recording_id].append(event_id)
        _identifier(row["candidate_id"], "G0a event candidate ID")
        if row["parent_candidate_id"] is not None:
            _identifier(row["parent_candidate_id"], "G0a split parent candidate ID")
        _identifier(row["candidate_origin"], "G0a candidate origin")
        _finite(row["candidate_anchor_recording_seconds_output_only"], "G0a candidate anchor")
        _sha256(row["source_candidate_receipt_sha256"], "G0a source candidate")
        if row["stable_origin_registry_receipt_sha256"] is not None:
            _sha256(
                row["stable_origin_registry_receipt_sha256"],
                "G0a event stable-origin registry",
            )
        _sha256(row["initial_support_policy_receipt_sha256"], "G0a support policy")
        initial = _canonical_union(
            row["initial_support_union_recording_seconds"],
            duration=duration,
            context="G0a output initial support",
        )
        if initial != row["initial_support_union_recording_seconds"]:
            raise ValueError("G0a output initial support is not canonical")
        final_support = _canonical_union(
            row["final_support_union_recording_seconds"],
            duration=duration,
            context="G0a output final support",
        )
        if final_support != row["final_support_union_recording_seconds"]:
            raise ValueError("G0a output final support is not canonical")
        if abs(_union_seconds(final_support) - _finite(row["final_support_seconds"], "G0a final support seconds")) > _TOLERANCE:
            raise ValueError("G0a final support seconds do not replay")
        expected_support_hash = _canonical_sha256(
            {
                "recording_id": recording_id,
                "source_signal_sha256": row["source_signal_sha256"],
                "support_union_recording_seconds": final_support,
            }
        )
        if row["final_support_union_sha256"] != expected_support_hash:
            raise ValueError("G0a final-support receipt does not replay")
        if not isinstance(row["raw_dependencies"], list):
            raise TypeError("G0a raw dependencies must be a list")
        dependencies = [
            _normalize_dependency(
                dependency,
                duration=duration,
                source_signal_sha256=row["source_signal_sha256"],
                index=dependency_index,
            )
            for dependency_index, dependency in enumerate(row["raw_dependencies"])
        ]
        if dependencies != row["raw_dependencies"]:
            raise ValueError("G0a raw dependencies are not canonically serialized")
        if row["raw_dependency_roster_sha256"] != _canonical_sha256(dependencies):
            raise ValueError("G0a raw-dependency roster does not replay")
        expected_qc_hash = _canonical_sha256(
            [
                {
                    "dependency_id": dependency["dependency_id"],
                    "qc_receipt_sha256": dependency["qc_receipt_sha256"],
                }
                for dependency in dependencies
            ]
        )
        if row["qc_receipt_roster_sha256"] != expected_qc_hash:
            raise ValueError("G0a QC receipt roster does not replay")

        dependencies_by_action: dict[str, list[dict[str, Any]]] = {}
        for dependency in dependencies:
            dependencies_by_action.setdefault(
                dependency["acquisition_action_id"], []
            ).append(dependency)
        initial_dependencies = dependencies_by_action.get(
            _INITIAL_DEPENDENCY_ACTION_ID, []
        )
        predecessor = _canonical_sha256(
            {
                "schema": "ba_ieg_g0_a1_initial_support_freeze_v1",
                "candidate_receipt_sha256": row[
                    "source_candidate_receipt_sha256"
                ],
                "initial_support_policy_receipt_sha256": row[
                    "initial_support_policy_receipt_sha256"
                ],
                "initial_support_union_recording_seconds": initial,
                "raw_dependency_sha256s": [
                    dependency["raw_dependency_sha256"]
                    for dependency in initial_dependencies
                ],
                "qc_receipt_sha256s": [
                    dependency["qc_receipt_sha256"]
                    for dependency in initial_dependencies
                ],
            }
        )
        if not isinstance(row["outer_actions"], list):
            raise TypeError("G0a output outer actions must be a list")
        current_support = deepcopy(initial)
        for action_index, raw_action in enumerate(row["outer_actions"]):
            action = _strict_object(
                raw_action, _OUTPUT_ACTION_FIELDS, f"G0a output action {action_index}"
            )
            if action["step_index"] != action_index:
                raise ValueError("G0a output outer action steps are not contiguous")
            _identifier(action["action_id"], "G0a output action ID")
            if action["action_type"] not in _ACTION_TYPES:
                raise ValueError("G0a output action type drifted")
            if action["scope_receipt"] != _ACTION_SCOPE:
                raise ValueError("G0a output action firewall drifted")
            for name in ("outer_state_sha256", "outer_decision_sha256"):
                _sha256(action[name], f"G0a action {name}")
            _identifier(action["outer_decision_id"], "G0a outer decision ID")
            acquired = _canonical_union(
                action["acquired_intervals_recording_seconds"],
                duration=duration,
                context="G0a output acquired support",
            )
            if not acquired:
                raise ValueError("G0a output acquisition has no physical interval")
            pre_support = _canonical_union(
                action["pre_support_union_recording_seconds"],
                duration=duration,
                context="G0a output pre-support",
            )
            post_support = _canonical_union(
                action["post_support_union_recording_seconds"],
                duration=duration,
                context="G0a output post-support",
            )
            if pre_support != current_support:
                raise ValueError("G0a output action chain pre-support drifted")
            expected_post = _canonical_union(
                [*pre_support, *acquired],
                duration=duration,
                context="G0a replayed action post-support",
            )
            if post_support != expected_post:
                raise ValueError("G0a output action chain post-support drifted")
            current_support = post_support
            action_dependencies = dependencies_by_action.get(action["action_id"], [])
            if action["raw_dependency_ids"] != [
                dependency["dependency_id"] for dependency in action_dependencies
            ]:
                raise ValueError("G0a output action dependency roster drifted")
            if action["receipt_capability"] == "decision_only_v1":
                if action["execution_receipt_sha256"] is not None:
                    raise ValueError("decision-only outer action acquired execution proof")
                continue
            if action["receipt_capability"] != "reference_free_execution_bound_v1":
                raise ValueError("G0a output action capability drifted")
            expected_execution = _canonical_sha256(
                {
                    "schema": "ba_ieg_g0_a1_reference_free_outer_execution_receipt_v1",
                    "event_id": event_id,
                    "candidate_id": row["candidate_id"],
                    "parent_candidate_id": row["parent_candidate_id"],
                    "step_index": action["step_index"],
                    "action_id": action["action_id"],
                    "action_type": action["action_type"],
                    "outer_state_sha256": action["outer_state_sha256"],
                    "outer_decision_id": action["outer_decision_id"],
                    "outer_decision_sha256": action["outer_decision_sha256"],
                    "pre_support_union_recording_seconds": pre_support,
                    "acquired_intervals_recording_seconds": acquired,
                    "post_support_union_recording_seconds": post_support,
                    "source_signal_sha256": row["source_signal_sha256"],
                    "raw_dependency_receipts": [
                        {
                            "dependency_id": dependency["dependency_id"],
                            "raw_dependency_sha256": dependency[
                                "raw_dependency_sha256"
                            ],
                            "qc_receipt_sha256": dependency["qc_receipt_sha256"],
                        }
                        for dependency in action_dependencies
                    ],
                    "predecessor_execution_receipt_sha256": predecessor,
                    "scope_receipt": _ACTION_SCOPE,
                }
            )
            if action["execution_receipt_sha256"] is not None:
                _sha256(action["execution_receipt_sha256"], "G0a execution receipt")
                if action["execution_receipt_sha256"] != expected_execution:
                    raise ValueError("G0a execution receipt does not replay")
                predecessor = expected_execution
        if not _same_union(current_support, final_support):
            raise ValueError("G0a output final support does not equal action chain")
        if row["acquisition_chain_tip_sha256"] != predecessor:
            raise ValueError("G0a acquisition-chain tip does not replay")
        if row["support_binding_status"] not in BA_IEG_G0_A1_SUPPORT_BINDING_STATUSES_V1:
            raise ValueError("G0a event support status drifted")
        _sha256(row["source_signal_sha256"], "event source signal")
        _sha256(row["raw_dependency_roster_sha256"], "raw-dependency roster")
        _sha256(row["qc_receipt_roster_sha256"], "QC roster")
        _sha256(row["final_support_union_sha256"], "final support union")
        _sha256(row["acquisition_chain_tip_sha256"], "acquisition chain tip")
        if row["primary_admitted"] is not False:
            raise ValueError("pre-reference event cannot be primary-admitted")
        if row["reference_join_order_status"] != (
            "pending_downstream_join_binding_to_this_ledger"
        ):
            raise ValueError("reference-join order status drifted")
    for recording_id, record in record_by_id.items():
        if sorted(observed_event_ids_by_record[recording_id]) != record["event_ids"]:
            raise ValueError("G0a record event roster does not replay")
    expected_counts = {
        "patients": len({row["patient_uid"] for row in data["records"]}),
        "records": len(data["records"]),
        "events": len(data["events"]),
        "support_binding_statuses": {
            status: sum(row["support_binding_status"] == status for row in data["events"])
            for status in BA_IEG_G0_A1_SUPPORT_BINDING_STATUSES_V1
        },
        "zero_detector_candidate_records": sum(
            row["prediction_outcome"] == "completed_zero_candidate"
            for row in data["records"]
        ),
        "technical_failure_records": sum(
            row["prediction_outcome"] == "technical_failure" for row in data["records"]
        ),
    }
    if data["counts"] != expected_counts:
        raise ValueError("G0a acquisition/support counts do not replay")
    expected_barrier = {
        "accepted_input_surface": "prediction_roster_stable_origin_outer_action_signal_dependency_qc_only",
        "public_event_intervals_opened": 0,
        "edf_annotations_opened": 0,
        "excel_or_spreadsheet_opened": 0,
        "physician_labels_or_reports_opened": 0,
        "clinical_text_or_context_opened": 0,
        "video_or_semiology_opened": 0,
        "sleep_stimulation_or_ecg_opened": 0,
        "support_actions_content_addressed_in_this_prejoin_ledger": True,
        "existing_outer_decision_v1_alone_is_execution_proof": False,
        "downstream_reference_join_commits_to_this_ledger": False,
        "immutable_cross_registry_order_receipt_materialized": False,
    }
    if data["pre_reference_freeze_barrier"] != expected_barrier:
        raise ValueError("G0a pre-reference firewall/order barrier drifted")
    software_complete = bool(data["events"]) and all(
        row["support_binding_status"] == "verified_reference_free_execution_chain"
        for row in data["events"]
    )
    if data["admission"] != {
        "reference_free_support_software_lineage_complete": software_complete,
        "downstream_reference_join_binding_complete": False,
        "g0a_primary_admitted": False,
        "pending_reason": "postfreeze_reference_join_must_bind_this_pre_reference_ledger",
    }:
        raise ValueError("G0a admission status drifted")
    expected = _seal(data, id_field="lineage_id", prefix="BAIEGG0A1SUPPORT")
    if data["lineage_id"] != expected["lineage_id"] or data["receipt_sha256"] != expected[
        "receipt_sha256"
    ]:
        raise ValueError("G0a acquisition/support content address does not replay")
    return data


__all__ = [
    "BA_IEG_G0_A1_INITIAL_SUPPORT_POLICY_SCHEMA_V1",
    "BA_IEG_G0_A1_ACQUISITION_SUPPORT_LINEAGE_SCHEMA_V1",
    "BA_IEG_G0_A1_OUTER_RECEIPT_CAPABILITIES_V1",
    "BA_IEG_G0_A1_SUPPORT_BINDING_STATUSES_V1",
    "build_ba_ieg_g0_a1_initial_support_policy_v1",
    "validate_ba_ieg_g0_a1_initial_support_policy_v1",
    "build_ba_ieg_g0_a1_acquisition_support_lineage_v1",
    "validate_ba_ieg_g0_a1_acquisition_support_lineage_v1",
]
