"""EEG-only hidden-chunk supervision for the BA-IEG acquisition router.

The boundary-adaptive controller needs to decide whether another physical EEG
interval is worth analysing.  Saliency, entropy, or detector confidence alone
cannot answer that question.  This module therefore materialises a replayable
counterfactual target from two executions of the *same frozen downstream
endpoint*:

``base``
    the candidate chunk is hidden from the expensive event encoder; and
``revealed``
    exactly that chunk is added while every downstream model and policy stays
    frozen.

The target records changes in onset/offset entropy, earliest-field stability,
SOZ-rank stability, and Finding opportunity.  Signed changes are preserved;
the router is trained with a non-negative benefit target and a separate harm
target so that an action which destabilises the evidence chain is not silently
clipped into a positive example.

Only public ``source_train`` and ``source_dev`` recordings are accepted.
Targets, hidden samples, reference boundaries, annotations, spreadsheets,
clinical text, and private labels are forbidden predictor inputs.  The output
is computational-routing supervision only and can never authorise a Finding,
an onset/SOZ leaf, or a report claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional


BA_IEG_COUNTERFACTUAL_UTILITY_TARGET_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_hidden_chunk_counterfactual_utility_target_v1"
BA_IEG_COUNTERFACTUAL_UTILITY_METHOD_ID: Final[
    str
] = "ba_ieg_hidden_chunk_counterfactual_marginal_utility_v1"

BA_IEG_COUNTERFACTUAL_GAIN_NAMES: Final[tuple[str, ...]] = (
    "onset_entropy_reduction_nats",
    "offset_entropy_reduction_nats",
    "earliest_field_stability_gain",
    "soz_rank_stability_gain",
    "finding_opportunity_gain",
)

BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES: Final[tuple[str, ...]] = (
    "onset_entropy_nats",
    "offset_entropy_nats",
    "earliest_field_stability",
    "soz_rank_stability",
    "finding_opportunity_fraction",
)

_GAIN_TO_ENDPOINT: Final[Mapping[str, str]] = dict(
    zip(BA_IEG_COUNTERFACTUAL_GAIN_NAMES, BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES)
)

_ENTROPY_NAMES: Final[frozenset[str]] = frozenset(
    {
        "onset_entropy_reduction_nats",
        "offset_entropy_reduction_nats",
    }
)
_BOUNDED_NAMES: Final[frozenset[str]] = frozenset(
    {
        "earliest_field_stability",
        "soz_rank_stability",
        "finding_opportunity_fraction",
    }
)
_ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"query_left", "query_right", "retrieve_distant_background"}
)
_ACTION_SIDES: Final[Mapping[str, str]] = {
    "query_left": "left",
    "query_right": "right",
    "retrieve_distant_background": "none",
}
_SPLIT_TO_ROLE: Final[Mapping[str, str]] = {
    "source_train": "optimize",
    "source_dev": "calibrate",
}
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-9

_FIREWALL: Final[dict[str, bool]] = {
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

_AUTHORIZATION: Final[dict[str, object]] = {
    "computational_routing_supervision_only": True,
    "may_authorize_positive_onset_or_soz_evidence": False,
    "may_create_report_eligible_finding": False,
    "may_create_or_strengthen_report_claim": False,
    "hidden_chunk_role": "counterfactual_training_target_only",
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{context} keys drifted; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed identifier")
    if len(value) > 192 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} is not a safe identifier")
    return value


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


def _interval(value: object, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element list")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start:
        raise ValueError(f"{context} must have positive duration")
    return [start, stop]


def _intervals(value: object, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    rows = [_interval(item, f"{context}[{index}]") for index, item in enumerate(value)]
    rows.sort(key=lambda item: (item[0], item[1]))
    for previous, current in zip(rows, rows[1:]):
        if current[0] < previous[1] - _TOLERANCE:
            raise ValueError(f"{context} intervals must not overlap")
    return rows


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    return max(
        0.0,
        min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])),
    )


def _metric_map(value: object, context: str) -> dict[str, float | None]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    _exact_keys(value, set(BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES), context)
    result: dict[str, float | None] = {}
    for name in BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES:
        raw = value[name]
        if raw is None:
            result[name] = None
            continue
        number = _finite(raw, f"{context}.{name}", minimum=0.0)
        if name in _BOUNDED_NAMES and number > 1.0 + _TOLERANCE:
            raise ValueError(f"{context}.{name} must lie in [0,1]")
        result[name] = min(1.0, number) if name in _BOUNDED_NAMES else number
    if all(item is None for item in result.values()):
        raise ValueError(f"{context} cannot be entirely not-evaluable")
    return result


def _validate_firewall(value: object, context: str) -> dict[str, bool]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if value != _FIREWALL:
        raise ValueError(f"{context} violates the EEG-only counterfactual firewall")
    return deepcopy(_FIREWALL)


def _validate_snapshot(value: object, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = {
        "endpoint_bundle_sha256",
        "input_evidence_union_sha256",
        "evidence_interval_roster_sha256",
        "metrics",
        "firewall",
    }
    _exact_keys(value, expected, context)
    return {
        "endpoint_bundle_sha256": _sha256(
            value["endpoint_bundle_sha256"],
            f"{context}.endpoint_bundle_sha256",
        ),
        "input_evidence_union_sha256": _sha256(
            value["input_evidence_union_sha256"],
            f"{context}.input_evidence_union_sha256",
        ),
        "evidence_interval_roster_sha256": _sha256(
            value["evidence_interval_roster_sha256"],
            f"{context}.evidence_interval_roster_sha256",
        ),
        "metrics": _metric_map(value["metrics"], f"{context}.metrics"),
        "firewall": _validate_firewall(value["firewall"], f"{context}.firewall"),
    }


def _validate_action(value: object, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    expected = {
        "action_id",
        "action_type",
        "side",
        "current_event_interval_recording_seconds",
        "visible_intervals_recording_seconds",
        "proposed_intervals_recording_seconds",
        "full_candidate_envelope_recording_seconds",
        "predictor_input_receipt_sha256",
        "hidden_chunk_receipt_sha256",
        "target_independent_candidate_roster_sha256",
        "hidden_chunk_was_masked_from_predictor_input",
    }
    _exact_keys(value, expected, context)
    action_type = value["action_type"]
    if action_type not in _ACTION_TYPES:
        raise ValueError(f"{context}.action_type is unsupported for hidden-chunk v1")
    side = value["side"]
    if side != _ACTION_SIDES[action_type]:
        raise ValueError(f"{context}.side is inconsistent with action_type")
    if value["hidden_chunk_was_masked_from_predictor_input"] is not True:
        raise ValueError(f"{context} leaks the hidden chunk into router input")

    current = _interval(
        value["current_event_interval_recording_seconds"],
        f"{context}.current_event_interval_recording_seconds",
    )
    visible = _intervals(
        value["visible_intervals_recording_seconds"],
        f"{context}.visible_intervals_recording_seconds",
    )
    proposed = _intervals(
        value["proposed_intervals_recording_seconds"],
        f"{context}.proposed_intervals_recording_seconds",
    )
    envelope = _interval(
        value["full_candidate_envelope_recording_seconds"],
        f"{context}.full_candidate_envelope_recording_seconds",
    )
    if not proposed:
        raise ValueError(f"{context} must reveal at least one physical interval")
    for index, interval in enumerate(visible):
        if (
            interval[0] < envelope[0] - _TOLERANCE
            or interval[1] > envelope[1] + _TOLERANCE
        ):
            raise ValueError(
                f"{context}.visible_intervals[{index}] exceeds the envelope"
            )
    for index, interval in enumerate(proposed):
        if (
            interval[0] < envelope[0] - _TOLERANCE
            or interval[1] > envelope[1] + _TOLERANCE
        ):
            raise ValueError(
                f"{context}.proposed_intervals[{index}] exceeds the envelope"
            )
        if any(_overlap(interval, observed) > _TOLERANCE for observed in visible):
            raise ValueError(
                f"{context} proposed hidden chunk overlaps visible evidence"
            )

    if action_type == "query_left":
        if len(proposed) != 1 or abs(proposed[0][1] - current[0]) > 1e-6:
            raise ValueError(f"{context} query_left must meet the current left edge")
    elif action_type == "query_right":
        if len(proposed) != 1 or abs(proposed[0][0] - current[1]) > 1e-6:
            raise ValueError(f"{context} query_right must meet the current right edge")
    else:
        if any(_overlap(interval, current) > _TOLERANCE for interval in proposed):
            raise ValueError(f"{context} distant background may not overlap the event")

    return {
        "action_id": _identifier(value["action_id"], f"{context}.action_id"),
        "action_type": action_type,
        "side": side,
        "current_event_interval_recording_seconds": current,
        "visible_intervals_recording_seconds": visible,
        "proposed_intervals_recording_seconds": proposed,
        "full_candidate_envelope_recording_seconds": envelope,
        "predictor_input_receipt_sha256": _sha256(
            value["predictor_input_receipt_sha256"],
            f"{context}.predictor_input_receipt_sha256",
        ),
        "hidden_chunk_receipt_sha256": _sha256(
            value["hidden_chunk_receipt_sha256"],
            f"{context}.hidden_chunk_receipt_sha256",
        ),
        "target_independent_candidate_roster_sha256": _sha256(
            value["target_independent_candidate_roster_sha256"],
            f"{context}.target_independent_candidate_roster_sha256",
        ),
        "hidden_chunk_was_masked_from_predictor_input": True,
    }


def _interval_roster_sha256_from_validated_action(
    action: Mapping[str, Any], *, revealed: bool
) -> str:
    descriptor: dict[str, object] = {
        "schema_version": "ba_ieg_counterfactual_physical_interval_roster_v1",
        "visible_intervals_recording_seconds": action[
            "visible_intervals_recording_seconds"
        ],
    }
    if revealed:
        descriptor["revealed_intervals_recording_seconds"] = action[
            "proposed_intervals_recording_seconds"
        ]
    return _canonical_sha256(descriptor)


def counterfactual_interval_roster_sha256_v1(
    action: Mapping[str, Any], *, revealed: bool
) -> str:
    """Hash the exact base or base-plus-action physical interval roster."""

    if type(revealed) is not bool:
        raise TypeError("revealed must be boolean")
    validated = _validate_action(action, "action")
    return _interval_roster_sha256_from_validated_action(
        validated,
        revealed=revealed,
    )


def _signed_changes(
    base: Mapping[str, float | None],
    revealed: Mapping[str, float | None],
) -> tuple[
    dict[str, float | None],
    dict[str, float | None],
    dict[str, int | None],
    dict[str, bool],
]:
    raw: dict[str, float | None] = {}
    benefit: dict[str, float | None] = {}
    harm: dict[str, int | None] = {}
    mask: dict[str, bool] = {}
    for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES:
        endpoint_name = _GAIN_TO_ENDPOINT[name]
        before = base[endpoint_name]
        after = revealed[endpoint_name]
        evaluable = before is not None and after is not None
        mask[name] = evaluable
        if not evaluable:
            raw[name] = None
            benefit[name] = None
            harm[name] = None
            continue
        assert before is not None and after is not None
        delta = before - after if name in _ENTROPY_NAMES else after - before
        if abs(delta) <= _TOLERANCE:
            delta = 0.0
        raw[name] = float(delta)
        benefit[name] = float(max(0.0, delta))
        harm[name] = int(delta < 0.0)
    if not any(mask.values()):
        raise ValueError("base/revealed snapshots share no evaluable endpoint")
    return raw, benefit, harm, mask


def _build_target(
    *,
    patient_uid: object,
    recording_id: object,
    event_id: object,
    model_split: object,
    source_data_manifest_sha256: object,
    action: object,
    base_snapshot: object,
    revealed_snapshot: object,
) -> dict[str, Any]:
    patient = _identifier(patient_uid, "patient_uid")
    recording = _identifier(recording_id, "recording_id")
    event = _identifier(event_id, "event_id")
    if model_split not in _SPLIT_TO_ROLE:
        raise ValueError(
            "counterfactual targets are restricted to public source_train/source_dev"
        )
    split = str(model_split)
    source_manifest = _sha256(
        source_data_manifest_sha256,
        "source_data_manifest_sha256",
    )
    validated_action = _validate_action(action, "action")
    base = _validate_snapshot(base_snapshot, "base_snapshot")
    revealed = _validate_snapshot(revealed_snapshot, "revealed_snapshot")
    if base["endpoint_bundle_sha256"] != revealed["endpoint_bundle_sha256"]:
        raise ValueError("counterfactual snapshots must use the same frozen endpoint")
    expected_base_roster = _interval_roster_sha256_from_validated_action(
        validated_action,
        revealed=False,
    )
    expected_revealed_roster = _interval_roster_sha256_from_validated_action(
        validated_action,
        revealed=True,
    )
    if base["evidence_interval_roster_sha256"] != expected_base_roster:
        raise ValueError(
            "base snapshot does not bind the exact visible interval roster"
        )
    if revealed["evidence_interval_roster_sha256"] != expected_revealed_roster:
        raise ValueError(
            "revealed snapshot is not the exact visible-plus-hidden interval roster"
        )
    if base["input_evidence_union_sha256"] == revealed["input_evidence_union_sha256"]:
        raise ValueError("revealed snapshot must bind a different evidence union")
    if (
        validated_action["predictor_input_receipt_sha256"]
        == validated_action["hidden_chunk_receipt_sha256"]
    ):
        raise ValueError(
            "hidden-chunk target receipt cannot equal predictor input receipt"
        )

    raw, benefit, harm, mask = _signed_changes(base["metrics"], revealed["metrics"])
    physical_seconds = sum(
        stop - start
        for start, stop in validated_action["proposed_intervals_recording_seconds"]
    )
    body: dict[str, Any] = {
        "schema_version": BA_IEG_COUNTERFACTUAL_UTILITY_TARGET_SCHEMA_VERSION,
        "target_id": "CONTENT-ADDRESS-PENDING",
        "method_id": BA_IEG_COUNTERFACTUAL_UTILITY_METHOD_ID,
        "patient_uid": patient,
        "recording_id": recording,
        "event_id": event,
        "model_split": split,
        "optimization_role": _SPLIT_TO_ROLE[split],
        "source_data_manifest_sha256": source_manifest,
        "action": validated_action,
        "base_snapshot": base,
        "revealed_snapshot": revealed,
        "raw_signed_delta": raw,
        "benefit_target": benefit,
        "harm_target": harm,
        "evaluable_mask": mask,
        "physical_eeg_seconds": physical_seconds,
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    body["target_id"] = "BAIEG-CFU-" + _canonical_sha256(body)[:24]
    return body


def materialize_hidden_chunk_counterfactual_target_v1(
    *,
    patient_uid: str,
    recording_id: str,
    event_id: str,
    model_split: str,
    source_data_manifest_sha256: str,
    action: Mapping[str, Any],
    base_snapshot: Mapping[str, Any],
    revealed_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialise one content-addressed counterfactual action target."""

    return _build_target(
        patient_uid=patient_uid,
        recording_id=recording_id,
        event_id=event_id,
        model_split=model_split,
        source_data_manifest_sha256=source_data_manifest_sha256,
        action=action,
        base_snapshot=base_snapshot,
        revealed_snapshot=revealed_snapshot,
    )


def validate_hidden_chunk_counterfactual_target_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate and exactly replay a serialised counterfactual target."""

    if type(payload) is not dict:
        raise TypeError("counterfactual utility target must be an object")
    expected = {
        "schema_version",
        "target_id",
        "method_id",
        "patient_uid",
        "recording_id",
        "event_id",
        "model_split",
        "optimization_role",
        "source_data_manifest_sha256",
        "action",
        "base_snapshot",
        "revealed_snapshot",
        "raw_signed_delta",
        "benefit_target",
        "harm_target",
        "evaluable_mask",
        "physical_eeg_seconds",
        "firewall",
        "authorization",
    }
    _exact_keys(payload, expected, "target")
    if payload["schema_version"] != BA_IEG_COUNTERFACTUAL_UTILITY_TARGET_SCHEMA_VERSION:
        raise ValueError("counterfactual utility target schema drifted")
    if payload["method_id"] != BA_IEG_COUNTERFACTUAL_UTILITY_METHOD_ID:
        raise ValueError("counterfactual utility method drifted")
    if payload["firewall"] != _FIREWALL:
        raise ValueError("counterfactual target firewall drifted")
    if payload["authorization"] != _AUTHORIZATION:
        raise ValueError("counterfactual target authorization drifted")

    expected_payload = _build_target(
        patient_uid=payload["patient_uid"],
        recording_id=payload["recording_id"],
        event_id=payload["event_id"],
        model_split=payload["model_split"],
        source_data_manifest_sha256=payload["source_data_manifest_sha256"],
        action=payload["action"],
        base_snapshot=payload["base_snapshot"],
        revealed_snapshot=payload["revealed_snapshot"],
    )
    if _canonical_json(payload) != _canonical_json(expected_payload):
        raise ValueError("counterfactual utility target does not replay")
    return deepcopy(expected_payload)


@dataclass(frozen=True)
class BAIEGCounterfactualUtilityBatchV1:
    """Patient/event-balanced tensor view of validated target rows."""

    target_ids: tuple[str, ...]
    patient_uids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    model_split: str
    benefit_target: torch.Tensor
    harm_target: torch.Tensor
    evaluable_mask: torch.Tensor
    row_weight: torch.Tensor
    batch_sha256: str

    def __post_init__(self) -> None:
        row_count = len(self.target_ids)
        if row_count == 0:
            raise ValueError("counterfactual utility batch cannot be empty")
        if not (
            len(self.patient_uids)
            == len(self.recording_ids)
            == len(self.event_ids)
            == row_count
        ):
            raise ValueError("counterfactual batch identifiers are misaligned")
        expected_shape = (row_count, len(BA_IEG_COUNTERFACTUAL_GAIN_NAMES))
        if self.benefit_target.shape != expected_shape:
            raise ValueError("counterfactual benefit target shape drifted")
        if self.harm_target.shape != expected_shape:
            raise ValueError("counterfactual harm target shape drifted")
        if self.evaluable_mask.shape != expected_shape:
            raise ValueError("counterfactual evaluable mask shape drifted")
        if self.row_weight.shape != (row_count,):
            raise ValueError("counterfactual row-weight shape drifted")
        if self.evaluable_mask.dtype != torch.bool:
            raise TypeError("counterfactual evaluable mask must be boolean")
        for tensor in (self.benefit_target, self.harm_target, self.row_weight):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(
                    "counterfactual batch tensors must be finite floating point"
                )
        if (self.benefit_target < 0).any():
            raise ValueError("counterfactual benefits must be non-negative")
        if ((self.harm_target < 0) | (self.harm_target > 1)).any():
            raise ValueError("counterfactual harm targets must lie in [0,1]")
        if (self.row_weight <= 0).any() or not torch.isclose(
            self.row_weight.sum(),
            torch.tensor(1.0, dtype=self.row_weight.dtype),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "counterfactual row weights must be positive and sum to one"
            )
        if self.model_split not in _SPLIT_TO_ROLE:
            raise ValueError("counterfactual batch split is invalid")
        _sha256(self.batch_sha256, "batch_sha256")


def collate_hidden_chunk_counterfactual_targets_v1(
    payloads: Sequence[Mapping[str, Any]],
) -> BAIEGCounterfactualUtilityBatchV1:
    """Collate targets without letting action multiplicity inflate patients.

    Weighting is hierarchical: actions are averaged within an event, events
    are averaged within a patient, and patients receive equal total mass.
    """

    if isinstance(payloads, (str, bytes)) or not payloads:
        raise ValueError("counterfactual target payloads must be non-empty")
    rows = [validate_hidden_chunk_counterfactual_target_v1(item) for item in payloads]
    rows.sort(key=lambda item: item["target_id"])
    target_ids = [str(item["target_id"]) for item in rows]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("counterfactual target IDs must be unique")
    splits = {str(item["model_split"]) for item in rows}
    manifests = {str(item["source_data_manifest_sha256"]) for item in rows}
    if len(splits) != 1:
        raise ValueError("one counterfactual batch cannot mix model splits")
    if len(manifests) != 1:
        raise ValueError("one counterfactual batch cannot mix source manifests")

    patient_to_events: dict[str, set[tuple[str, str]]] = {}
    event_to_count: dict[tuple[str, str, str], int] = {}
    for item in rows:
        patient = str(item["patient_uid"])
        event_key = (str(item["recording_id"]), str(item["event_id"]))
        patient_to_events.setdefault(patient, set()).add(event_key)
        full_key = (patient, *event_key)
        event_to_count[full_key] = event_to_count.get(full_key, 0) + 1
    patient_count = len(patient_to_events)

    benefits: list[list[float]] = []
    harms: list[list[float]] = []
    masks: list[list[bool]] = []
    weights: list[float] = []
    for item in rows:
        patient = str(item["patient_uid"])
        full_key = (
            patient,
            str(item["recording_id"]),
            str(item["event_id"]),
        )
        benefits.append(
            [
                float(item["benefit_target"][name] or 0.0)
                for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES
            ]
        )
        harms.append(
            [
                float(item["harm_target"][name] or 0.0)
                for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES
            ]
        )
        masks.append(
            [
                bool(item["evaluable_mask"][name])
                for name in BA_IEG_COUNTERFACTUAL_GAIN_NAMES
            ]
        )
        weights.append(
            1.0
            / patient_count
            / len(patient_to_events[patient])
            / event_to_count[full_key]
        )

    batch_descriptor = {
        "schema_version": "ba_ieg_counterfactual_utility_batch_v1",
        "target_ids": target_ids,
        "source_data_manifest_sha256": next(iter(manifests)),
        "model_split": next(iter(splits)),
        "weighting": "action_mean_within_event_then_event_mean_within_equal_patient",
    }
    return BAIEGCounterfactualUtilityBatchV1(
        target_ids=tuple(target_ids),
        patient_uids=tuple(str(item["patient_uid"]) for item in rows),
        recording_ids=tuple(str(item["recording_id"]) for item in rows),
        event_ids=tuple(str(item["event_id"]) for item in rows),
        model_split=next(iter(splits)),
        benefit_target=torch.tensor(benefits, dtype=torch.float32),
        harm_target=torch.tensor(harms, dtype=torch.float32),
        evaluable_mask=torch.tensor(masks, dtype=torch.bool),
        row_weight=torch.tensor(weights, dtype=torch.float32),
        batch_sha256=_canonical_sha256(batch_descriptor),
    )


def compute_hidden_chunk_counterfactual_training_loss_v1(
    *,
    predicted_benefit: torch.Tensor,
    predicted_harm_logits: torch.Tensor,
    batch: BAIEGCounterfactualUtilityBatchV1,
    harm_weight: float = 1.0,
) -> dict[str, torch.Tensor | str]:
    """Compute source-train-only benefit regression plus harm classification.

    ``predicted_benefit`` must already pass through a non-negative link (for
    example softplus).  ``source_dev`` is deliberately rejected here: it may
    calibrate/freeze thresholds but may not carry a gradient-bearing training
    objective.
    """

    if batch.model_split != "source_train":
        raise ValueError("gradient-bearing router loss is source_train-only")
    expected_shape = batch.benefit_target.shape
    if predicted_benefit.shape != expected_shape:
        raise ValueError("predicted_benefit shape does not match target batch")
    if predicted_harm_logits.shape != expected_shape:
        raise ValueError("predicted_harm_logits shape does not match target batch")
    if (
        not predicted_benefit.is_floating_point()
        or not predicted_harm_logits.is_floating_point()
    ):
        raise TypeError("counterfactual predictions must be floating point")
    if (
        not torch.isfinite(predicted_benefit).all()
        or not torch.isfinite(predicted_harm_logits).all()
    ):
        raise ValueError("counterfactual predictions must be finite")
    if (predicted_benefit < 0).any():
        raise ValueError("predicted benefit must use a non-negative link")
    harm_weight_value = _finite(harm_weight, "harm_weight", minimum=0.0)

    device = predicted_benefit.device
    dtype = predicted_benefit.dtype
    target_benefit = batch.benefit_target.to(device=device, dtype=dtype)
    target_harm = batch.harm_target.to(device=device, dtype=dtype)
    mask = batch.evaluable_mask.to(device=device)
    row_weight = batch.row_weight.to(device=device, dtype=dtype)

    benefit_item = torch_functional.smooth_l1_loss(
        predicted_benefit,
        target_benefit,
        reduction="none",
    )
    harm_item = torch_functional.binary_cross_entropy_with_logits(
        predicted_harm_logits,
        target_harm,
        reduction="none",
    )
    mask_float = mask.to(dtype=dtype)
    denominator = mask_float.sum(dim=1)
    if (denominator <= 0).any():
        raise ValueError("each counterfactual row needs an evaluable target")
    benefit_per_row = (benefit_item * mask_float).sum(dim=1) / denominator
    harm_per_row = (harm_item * mask_float).sum(dim=1) / denominator
    benefit_loss = (benefit_per_row * row_weight).sum()
    harm_loss = (harm_per_row * row_weight).sum()
    total_loss = benefit_loss + harm_weight_value * harm_loss
    return {
        "method_id": BA_IEG_COUNTERFACTUAL_UTILITY_METHOD_ID,
        "batch_sha256": batch.batch_sha256,
        "benefit_loss": benefit_loss,
        "harm_loss": harm_loss,
        "total_loss": total_loss,
    }


def risk_adjusted_predicted_gain_v1(
    *,
    predicted_benefit: torch.Tensor,
    predicted_harm_probability: torch.Tensor,
    evaluable_mask: torch.Tensor,
    harm_aversion: float = 1.0,
) -> torch.Tensor:
    """Convert model heads into non-negative controller gains.

    The controller receives ``benefit * (1 - P[harm])**harm_aversion``.  This
    encourages useful acquisition while continuously discounting actions that
    often destabilise the evidence chain.  It remains a routing score, not a
    calibrated clinical probability.
    """

    if predicted_benefit.shape != predicted_harm_probability.shape or (
        predicted_benefit.shape != evaluable_mask.shape
    ):
        raise ValueError("risk-adjusted gain tensors must have identical shape")
    if evaluable_mask.dtype != torch.bool:
        raise TypeError("risk-adjusted evaluable_mask must be boolean")
    if (
        not predicted_benefit.is_floating_point()
        or not predicted_harm_probability.is_floating_point()
    ):
        raise TypeError("risk-adjusted predictions must be floating point")
    if (
        not torch.isfinite(predicted_benefit).all()
        or not torch.isfinite(predicted_harm_probability).all()
    ):
        raise ValueError("risk-adjusted predictions must be finite")
    if (predicted_benefit < 0).any():
        raise ValueError("risk-adjusted benefit must be non-negative")
    if ((predicted_harm_probability < 0) | (predicted_harm_probability > 1)).any():
        raise ValueError("predicted harm probability must lie in [0,1]")
    aversion = _finite(harm_aversion, "harm_aversion", minimum=0.0)
    adjusted = predicted_benefit * (1.0 - predicted_harm_probability).pow(aversion)
    return torch.where(evaluable_mask, adjusted, torch.zeros_like(adjusted))


__all__ = [
    "BA_IEG_COUNTERFACTUAL_ENDPOINT_NAMES",
    "BA_IEG_COUNTERFACTUAL_GAIN_NAMES",
    "BA_IEG_COUNTERFACTUAL_UTILITY_METHOD_ID",
    "BA_IEG_COUNTERFACTUAL_UTILITY_TARGET_SCHEMA_VERSION",
    "BAIEGCounterfactualUtilityBatchV1",
    "collate_hidden_chunk_counterfactual_targets_v1",
    "compute_hidden_chunk_counterfactual_training_loss_v1",
    "counterfactual_interval_roster_sha256_v1",
    "materialize_hidden_chunk_counterfactual_target_v1",
    "risk_adjusted_predicted_gain_v1",
    "validate_hidden_chunk_counterfactual_target_v1",
]
