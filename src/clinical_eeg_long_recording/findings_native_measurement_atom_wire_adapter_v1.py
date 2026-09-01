"""Replayable native Findings producer -> onset measurement-atom wire.

The S03--S06 producers predate the strict atom schema used by
``onset_trigger_attribution_v1_5_1``.  This module closes that software wire
without upgrading a numerical measurement into a clinical term or silently
inventing missing authority.

Two rules are intentionally non-negotiable:

* a strict atom is emitted only when every mandatory atom field can be
  recovered from a validated producer receipt, its paired native sidecar, the
  locked-prefix attribution context, or an embedded content-addressed wire
  decision;
* when a producer does not carry enough information, the output is a typed
  ``not_evaluable`` wire row with ``measurement_atom_id=None``.  In
  particular, a masked zero is never inserted into the atom's mandatory
  numeric effect field.

At v1, S03 adjacent spectral changes and S04 adjacent physical-amplitude
changes have sufficient native lineage for strict numerical atoms.  The
current onset-threshold registry remains unadmitted, so these atoms retain
``measurement_state='present'`` while their effect-threshold and persistence
gates remain ``not_evaluable``.  They therefore cannot become positive onset
support.  S05 has no registered operator-specific required bandwidth in its
producer receipt, and the current S06 producer explicitly forbids onset
support; both are materialized as typed non-evaluable wire rows rather than
filled with guessed fields.

No file I/O occurs here.  EDF annotations, spreadsheets, doctor text,
clinical history, behaviour/video, sleep/activation, ECG/EMG/EOG and LLMs are
outside the callable interface.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import re
from typing import Any, Final

import torch

from .ba_ieg_dense_measurement_sidecar import (
    BA_IEG_DENSE_MEASUREMENT_METHOD_ID,
    BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION,
    BAIEGDenseMeasurementPolicy,
    BAIEGDenseMeasurementRowBinding,
    BAIEGDenseMeasurementSidecar,
    BAIEGDenseMeasurementViewBinding,
)
from .ba_ieg_training_contract import (
    BA_IEG_DETERMINISTIC_TARGETS,
    BAIEGDeterministicTargets,
)
from .deterministic_event_morphology_primitives_v1 import (
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES,
    validate_event_morphology_primitive_supervision_v1,
)
from .event_component_cycle_element_ledger_v1 import (
    validate_event_component_cycle_element_ledger_v1,
)
from .event_frequency_findings_v1 import (
    replay_event_frequency_findings_v1,
    validate_event_frequency_findings_v1,
)
from .event_physical_amplitude_findings_v1 import (
    validate_event_physical_amplitude_findings_v1,
)
from .onset_trigger_attribution_v1_5_1 import (
    ONSET_TRIGGER_ATTRIBUTION_CONTEXT_SCHEMA_VERSION,
    validate_onset_trigger_attribution_context_v1_5_1,
    validate_onset_trigger_measurement_atom_v1_5_1,
)


FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_native_measurement_atom_wire_adapter_v1"
)
FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_METHOD_ID: Final[str] = (
    "FINDINGS-NATIVE-MEASUREMENT-ATOM-WIRE-ADAPTER-V1"
)
FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_DECISION_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_native_measurement_atom_wire_decision_v1"
)
FINDINGS_NATIVE_MEASUREMENT_ATOM_SOURCE_PROPOSAL_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_native_measurement_atom_source_proposal_v1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-9
_SLOTS: Final[tuple[str, ...]] = ("S03", "S04", "S05", "S06")

_TRIGGER_FIREWALL: Final[dict[str, bool]] = {
    "deterministic_native_EEG_remeasurement_used": True,
    "attention_used": False,
    "saliency_used": False,
    "detector_posterior_used": False,
    "detector_score_used": False,
    "late_course_feature_used_as_trigger": False,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_history_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}

_FIREWALL: Final[dict[str, bool]] = {
    "EEG_samples_used_by_upstream_native_producers": True,
    "allowlisted_acquisition_metadata_used": True,
    "EDF_annotations_used": False,
    "spreadsheet_or_Excel_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_history_used": False,
    "patient_identity_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_or_activation_used": False,
    "ECG_EMG_EOG_used": False,
    "LLM_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str | list[str]]] = {
    "scope": "native_numeric_producer_to_strict_measurement_atom_wire_only",
    "automated_clinical_term_allowlist": [],
    "clinical_term_qualification_authorized": False,
    "positive_onset_trigger_authorized": False,
    "positive_rank_contribution_authorized": False,
    "SOZ_EZ_or_surgical_target_claim_authorized": False,
    "report_text_authorized": False,
    "whole_bipolar_lead_identity_preserved": True,
    "bipolar_endpoint_attribution_authorized": False,
    "masked_zero_may_fill_missing_numeric_effect": False,
    "typed_not_evaluable_required_when_atom_fields_are_missing": True,
    "strict_atom_slots_materialized_at_v1": ["S03", "S04"],
    "strict_atom_slots_fail_closed_at_v1": ["S05", "S06"],
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha(body)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be a two-number interval")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must be non-empty")
    return [start, stop]


def _inside(carrier: Sequence[float], item: Sequence[float]) -> bool:
    return (
        float(item[0]) >= float(carrier[0]) - _TOL
        and float(item[1]) <= float(carrier[1]) + _TOL
    )


def _canonical_union(intervals: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = sorted(
        (_interval(row, "raw dependency interval") for row in intervals),
        key=lambda row: (row[0], row[1]),
    )
    result: list[list[float]] = []
    for row in rows:
        if not result or row[0] > result[-1][1] + _TOL:
            result.append(row)
        else:
            result[-1][1] = max(result[-1][1], row[1])
    return result


def _typed_unit(unit_type: object, unit_id: object) -> dict[str, str]:
    typed = {
        "unit_type": _identifier(unit_type, "unit_type"),
        "unit_id": _identifier(unit_id, "unit_id"),
    }
    if typed["unit_type"] not in {"electrode", "lead"}:
        raise ValueError("wire adapter supports electrode or whole lead units")
    typed["unit_key"] = f"{typed['unit_type']}:{typed['unit_id']}"
    return typed


def _reference_family(typed: Mapping[str, str]) -> str:
    return "referential" if typed["unit_type"] == "electrode" else "bipolar"


def _source_channels(
    typed: Mapping[str, str], observed_channels: Sequence[object]
) -> list[str]:
    observed = [_identifier(str(item), "canonical source channel") for item in observed_channels]
    if len(observed) != len(set(observed)) or not observed:
        raise ValueError("canonical source channels must be non-empty and unique")
    if typed["unit_type"] == "electrode":
        expected = [typed["unit_id"]]
    else:
        expected = typed["unit_id"].split("-")
        if len(expected) != 2 or expected[0] == expected[1]:
            raise ValueError("whole bipolar lead ID must retain directed endpoints")
    if set(observed) != set(expected):
        raise ValueError("producer source channels disagree with typed unit identity")
    # Producer ledgers often sort channel lineage.  The directed lead ID is
    # the authority for orientation; restoring that order is not endpoint
    # attribution and does not create an electrode-level finding.
    return expected


def _dense_sidecar_from_dict(value: object) -> BAIEGDenseMeasurementSidecar:
    """Rehydrate and fully integrity-check a serialized dense sidecar."""

    if isinstance(value, BAIEGDenseMeasurementSidecar):
        value.verify_integrity()
        return value
    if not isinstance(value, Mapping):
        raise TypeError("S03 dense sidecar must be an object or sidecar instance")
    payload = deepcopy(dict(value))
    expected_top = {
        "schema_version",
        "method_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "recording_id",
        "analysis_interval_seconds",
        "background_intervals_seconds",
        "policy",
        "policy_sha256",
        "target_names",
        "view_bindings",
        "row_bindings",
        "targets",
        "source_binding_sha256",
        "receipt_sha256",
    }
    if set(payload) != expected_top:
        raise ValueError("serialized S03 dense-sidecar fields drifted")
    if (
        payload["schema_version"] != BA_IEG_DENSE_MEASUREMENT_SIDECAR_SCHEMA_VERSION
        or payload["method_id"] != BA_IEG_DENSE_MEASUREMENT_METHOD_ID
    ):
        raise ValueError("serialized S03 dense-sidecar identity drifted")
    policy_body = payload["policy"]
    if not isinstance(policy_body, Mapping):
        raise TypeError("serialized S03 dense policy must be an object")
    policy_fields = set(BAIEGDenseMeasurementPolicy.__dataclass_fields__)
    if not policy_fields.issubset(policy_body):
        raise ValueError("serialized S03 dense policy is incomplete")
    policy = BAIEGDenseMeasurementPolicy(
        **{name: policy_body[name] for name in policy_fields}
    )
    if dict(policy_body) != policy.to_dict() or payload["policy_sha256"] != policy.sha256:
        raise ValueError("serialized S03 dense policy does not replay")

    view_bindings: list[BAIEGDenseMeasurementViewBinding] = []
    for index, raw in enumerate(payload["view_bindings"]):
        if not isinstance(raw, Mapping):
            raise TypeError(f"view_bindings[{index}] must be an object")
        binding = BAIEGDenseMeasurementViewBinding(
            view_index=raw["view_index"],
            view_id=raw["view_id"],
            task_role=raw["task_role"],
            view_receipt_id=raw["view_receipt_id"],
            view_receipt_sha256=raw["view_receipt_sha256"],
            transform_spec_sha256=raw["transform_spec_sha256"],
            processed_view_sha256=raw["processed_view_sha256"],
            quality_mask_sha256=raw["quality_mask_sha256"],
            reference_type=raw["reference_type"],
            reference_matrix_sha256=raw["reference_matrix_sha256"],
            output_sampling_rate_hz=raw["output_sampling_rate_hz"],
            output_unit_ids=tuple(raw["output_unit_ids"]),
            unit_indices=tuple(raw["unit_indices"]),
        )
        if _canonical_json(dict(raw)) != _canonical_json(asdict(binding)):
            raise ValueError(f"view_bindings[{index}] does not replay")
        view_bindings.append(binding)

    row_bindings: list[BAIEGDenseMeasurementRowBinding] = []
    for index, raw in enumerate(payload["row_bindings"]):
        if not isinstance(raw, Mapping):
            raise TypeError(f"row_bindings[{index}] must be an object")
        binding = BAIEGDenseMeasurementRowBinding(
            requested_row_index=raw["requested_row_index"],
            training_row_index=raw["training_row_index"],
            view_index=raw["view_index"],
            unit_index=raw["unit_index"],
            view_id=raw["view_id"],
            unit_id=raw["unit_id"],
            unit_type=raw["unit_type"],
            requested_recording_interval_seconds=tuple(
                raw["requested_recording_interval_seconds"]
            ),
            recording_interval_seconds=tuple(raw["recording_interval_seconds"]),
            tensor_sample_interval=tuple(raw["tensor_sample_interval"]),
            reference_type=raw["reference_type"],
            reference_row_sha256=raw["reference_row_sha256"],
            canonical_source_channel_ids=tuple(raw["canonical_source_channel_ids"]),
            effective_bandwidth_hz=tuple(raw["effective_bandwidth_hz"]),
            quality_mask_sha256=raw["quality_mask_sha256"],
            overlapping_quality_reason_codes=tuple(
                raw["overlapping_quality_reason_codes"]
            ),
            target_value_mask=tuple(raw["target_value_mask"]),
            target_reason_codes=tuple(
                tuple(items) for items in raw["target_reason_codes"]
            ),
            policy_sha256=raw["policy_sha256"],
        )
        if _canonical_json(dict(raw)) != _canonical_json(asdict(binding)):
            raise ValueError(f"row_bindings[{index}] does not replay")
        row_bindings.append(binding)

    target_body = payload["targets"]
    if not isinstance(target_body, Mapping) or set(target_body) != {
        "values",
        "value_mask",
        "row_time_bounds_seconds",
        "row_unit_index",
        "row_view_index",
        "receipt_sha256",
    }:
        raise ValueError("serialized S03 dense target fields drifted")
    targets = BAIEGDeterministicTargets(
        values=torch.tensor(target_body["values"], dtype=torch.float32),
        value_mask=torch.tensor(target_body["value_mask"], dtype=torch.bool),
        row_time_bounds_seconds=torch.tensor(
            target_body["row_time_bounds_seconds"], dtype=torch.float64
        ),
        row_unit_index=torch.tensor(target_body["row_unit_index"], dtype=torch.long),
        row_view_index=torch.tensor(target_body["row_view_index"], dtype=torch.long),
        policy_sha256=policy.sha256,
        source_binding_sha256=payload["source_binding_sha256"],
    )
    if targets.receipt_sha256 != target_body["receipt_sha256"]:
        raise ValueError("serialized S03 dense target receipt does not replay")
    if payload["target_names"] != list(BA_IEG_DETERMINISTIC_TARGETS):
        raise ValueError("serialized S03 dense target vocabulary drifted")
    sidecar = BAIEGDenseMeasurementSidecar(
        canonical_signal_id=payload["canonical_signal_id"],
        canonical_receipt_sha256=payload["canonical_receipt_sha256"],
        source_signal_sha256=payload["source_signal_sha256"],
        recording_id=payload["recording_id"],
        analysis_interval_seconds=tuple(payload["analysis_interval_seconds"]),
        background_intervals_seconds=tuple(
            tuple(row) for row in payload["background_intervals_seconds"]
        ),
        policy=policy,
        view_bindings=tuple(view_bindings),
        row_bindings=tuple(row_bindings),
        targets=targets,
        source_binding_sha256=payload["source_binding_sha256"],
    )
    if sidecar.receipt_sha256 != payload["receipt_sha256"]:
        raise ValueError("serialized S03 dense sidecar receipt does not replay")
    if _canonical_json(sidecar.to_dict()) != _canonical_json(payload):
        raise ValueError("serialized S03 dense sidecar payload drifted")
    return sidecar


def _decision(
    *,
    decision_kind: str,
    source_slot_id: str,
    source_item_ids: Sequence[str],
    state: str,
    facts: Mapping[str, object],
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_DECISION_SCHEMA_VERSION,
        "decision_kind": _identifier(decision_kind, "decision_kind"),
        "source_slot_id": source_slot_id,
        "source_item_ids": sorted({_identifier(item, "source_item_id") for item in source_item_ids}),
        "state": _identifier(state, "decision state"),
        "facts": deepcopy(dict(facts)),
        "reason_codes": sorted({_identifier(item, "reason_code") for item in reason_codes}),
        "clinical_term_authorized": False,
        "positive_rank_contribution_authorized": False,
        "receipt_sha256": "",
    }
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return body


def _proposal(
    *,
    context: Mapping[str, Any],
    source_slot_id: str,
    source_item_ids: Sequence[str],
    interval: Sequence[float],
    typed_unit: Mapping[str, str],
    reference_family: str,
    producer_receipt_sha256: str,
    selection_receipt_sha256s: Sequence[str],
) -> dict[str, Any]:
    material = {
        "recording_id": context["recording_id"],
        "occurrence_id": context["occurrence_id"],
        "query_index": context["query_index"],
        "source_slot_id": source_slot_id,
        "source_item_ids": sorted(set(source_item_ids)),
        "recording_relative_half_open_interval_s": list(interval),
        "typed_unit": deepcopy(dict(typed_unit)),
        "reference_family": reference_family,
        "producer_receipt_sha256": producer_receipt_sha256,
        "selection_receipt_sha256s": sorted(set(selection_receipt_sha256s)),
    }
    proposal_id = f"WIREPROP-{source_slot_id}-{_sha(material)[:24]}"
    body: dict[str, Any] = {
        "schema_version": FINDINGS_NATIVE_MEASUREMENT_ATOM_SOURCE_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": proposal_id,
        **material,
        "proposal_semantics": "validated_signal_only_query_projection_not_a_finding",
        "future_sample_access": False,
        "clinical_term_authorized": False,
        "onset_or_soz_claim_authorized": False,
        "receipt_sha256": "",
    }
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return body


def _wire_row(
    *,
    source_slot_id: str,
    source_item_id: str,
    producer_receipt_sha256: str,
    source_measurement_name_id: str | None,
    source_measurement_state: str,
    typed_unit: Mapping[str, str] | None,
    reference_family: str | None,
    interval: Sequence[float] | None,
    measurement_atom_id: str | None,
    missing_required_atom_fields: Sequence[str],
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source_slot_id": source_slot_id,
        "source_item_id": _identifier(source_item_id, "source_item_id"),
        "producer_receipt_sha256": _sha256(
            producer_receipt_sha256, "producer_receipt_sha256"
        ),
        "source_measurement_name_id": source_measurement_name_id,
        "source_measurement_state": source_measurement_state,
        "typed_unit": deepcopy(dict(typed_unit)) if typed_unit is not None else None,
        "reference_family": reference_family,
        "recording_relative_half_open_interval_s": (
            list(interval) if interval is not None else None
        ),
        "wire_state": (
            "measurement_atom_materialized_trigger_gate_not_evaluable"
            if measurement_atom_id is not None
            else "not_evaluable"
        ),
        "measurement_atom_id": measurement_atom_id,
        "missing_required_atom_fields": sorted(set(missing_required_atom_fields)),
        "reason_codes": sorted(set(reason_codes)),
        "masked_zero_used_as_numeric_effect": False,
        "clinical_term_authorized": False,
    }
    body["wire_row_id"] = "WIRE-" + _sha(body)[:24]
    body["wire_row_sha256"] = _self_hash(body, "wire_row_sha256")
    return body


class _Collector:
    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = context
        self.proposals: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.atoms: list[dict[str, Any]] = []
        self.wire_rows: list[dict[str, Any]] = []
        self.trusted_receipts: set[str] = set()

    def add_decision(self, **kwargs: Any) -> str:
        row = _decision(**kwargs)
        receipt = row["receipt_sha256"]
        self.decisions[receipt] = row
        self.trusted_receipts.add(receipt)
        return receipt

    def add_proposal(self, **kwargs: Any) -> dict[str, Any]:
        row = _proposal(context=self.context, **kwargs)
        self.proposals[row["proposal_id"]] = row
        return row

    def add_atom(
        self,
        *,
        source_slot_id: str,
        measurement_domain: str,
        source_item_ids: Sequence[str],
        proposal: Mapping[str, Any],
        interval: Sequence[float],
        change_interval: Sequence[float],
        raw_dependency_sha256s: Sequence[str],
        raw_dependency_intervals: Sequence[Sequence[float]],
        typed_unit: Mapping[str, str],
        canonical_source_channels: Sequence[str],
        reference_family: str,
        sample_rate_hz: float,
        physical_unit: str,
        effective_bandwidth_hz: Sequence[float],
        required_bandwidth_hz: Sequence[float],
        operator_id: str,
        operator_version: str,
        measurement_name_id: str,
        value: float,
        transform_receipt_sha256: str,
        operator_parameter_receipt_sha256: str,
        reference_transform_receipt_sha256: str,
        query_closure_receipt_sha256: str,
        producer_receipt_sha256: str,
        permission_lane: str,
    ) -> dict[str, Any]:
        source_ids = sorted(set(source_item_ids))
        native_validation_receipt = self.add_decision(
            decision_kind="native_measurement_validation",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="present",
            facts={
                "measurement_name_id": measurement_name_id,
                "value": value,
                "unit": physical_unit,
                "producer_receipt_sha256": producer_receipt_sha256,
            },
            reason_codes=[],
        )
        raw_receipt = self.add_decision(
            decision_kind="raw_dependency_binding",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="bound",
            facts={
                "raw_dependency_sha256s": sorted(set(raw_dependency_sha256s)),
                "raw_dependency_interval_union_s": _canonical_union(
                    raw_dependency_intervals
                ),
            },
            reason_codes=[],
        )
        qc_receipt = self.add_decision(
            decision_kind="qc_opportunity",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="pass",
            facts={"qc_opportunity_censor": False},
            reason_codes=[],
        )
        bandwidth_receipt = self.add_decision(
            decision_kind="bandwidth",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="pass",
            facts={
                "effective_bandwidth_hz": list(effective_bandwidth_hz),
                "required_bandwidth_hz": list(required_bandwidth_hz),
            },
            reason_codes=[],
        )
        threshold_receipt = self.add_decision(
            decision_kind="effect_threshold",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="not_evaluable",
            facts={
                "upstream_registry_receipt_sha256": self.context[
                    "onset_trigger_threshold_registry_receipt_sha256"
                ],
                "threshold_registry_admitted": False,
                "threshold_value": None,
            },
            reason_codes=["no_admitted_real_onset_trigger_threshold_registry"],
        )
        persistence_receipt = self.add_decision(
            decision_kind="minimum_persistence",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state="not_evaluable",
            facts={
                "upstream_registry_receipt_sha256": self.context[
                    "onset_trigger_threshold_registry_receipt_sha256"
                ],
                "threshold_registry_admitted": False,
                "minimum_persistence_seconds": None,
            },
            reason_codes=["no_admitted_real_minimum_persistence_registry"],
        )
        permission_receipt = self.add_decision(
            decision_kind="permission_lane",
            source_slot_id=source_slot_id,
            source_item_ids=source_ids,
            state=permission_lane,
            facts={
                "locked_causal_prefix_receipt_sha256": self.context[
                    "locked_causal_prefix_receipt_sha256"
                ],
                "positive_trigger_authorized": False,
                "clinical_term_authorized": False,
            },
            reason_codes=["threshold_and_persistence_gates_remain_not_evaluable"],
        )
        for receipt in (
            transform_receipt_sha256,
            operator_parameter_receipt_sha256,
            reference_transform_receipt_sha256,
            query_closure_receipt_sha256,
            producer_receipt_sha256,
        ):
            self.trusted_receipts.add(_sha256(receipt, "atom source receipt"))
        atom_seed = {
            "source_slot_id": source_slot_id,
            "source_item_ids": source_ids,
            "measurement_name_id": measurement_name_id,
            "interval": list(interval),
            "typed_unit": dict(typed_unit),
            "producer_receipt_sha256": producer_receipt_sha256,
        }
        atom: dict[str, Any] = {
            "measurement_atom_id": f"ATOM-{source_slot_id}-{_sha(atom_seed)[:24]}",
            "source_proposal_ids": [proposal["proposal_id"]],
            "source_slot_id": source_slot_id,
            "measurement_domain": measurement_domain,
            "namespace": "measurement",
            "recording_id": self.context["recording_id"],
            "occurrence_id": self.context["occurrence_id"],
            "query_index": self.context["query_index"],
            "recording_relative_half_open_interval_s": list(interval),
            "change_interval_s": list(change_interval),
            "raw_dependency_sha256s": sorted(set(raw_dependency_sha256s)),
            "raw_dependency_interval_union_s": _canonical_union(
                raw_dependency_intervals
            ),
            "typed_unit": deepcopy(dict(typed_unit)),
            "canonical_source_channels": list(canonical_source_channels),
            "reference_family": reference_family,
            "sample_rate_hz": float(sample_rate_hz),
            "physical_unit": physical_unit,
            "effective_bandwidth_hz": list(effective_bandwidth_hz),
            "required_bandwidth_hz": list(required_bandwidth_hz),
            "qc_opportunity_censor": False,
            "qc_opportunity_state": "pass",
            "bandwidth_state": "pass",
            "measurement_opportunity_state": "sufficient",
            "effect_threshold_state": "not_evaluable",
            "minimum_persistence_state": "not_evaluable",
            # This adapter sees one validated producer snapshot.  It cannot
            # infer cross-query stability from a single snapshot, even when
            # the enclosing causal prefix is locked.  ``first_observed`` is
            # therefore the only truthful default; the attribution gate will
            # keep the atom non-positive until a separate trajectory replay
            # proves stabilization.
            "query_transition_state": "first_observed",
            "operator_id": operator_id,
            "operator_version": operator_version,
            "effect_size_and_unit": {
                "measurement_name_id": measurement_name_id,
                "value": float(value),
                "unit": physical_unit,
                "semantics": (
                    "native_measurement_effect_not_rank_delta_or_clinical_causality"
                ),
            },
            "uncertainty": {
                "status": "not_established",
                "lower": None,
                "upper": None,
                "unit": physical_unit,
            },
            "permission_lane": permission_lane,
            "native_remeasurement_verified": True,
            "future_sample_access": False,
            "late_course_feature_used": False,
            "whole_bipolar_lead_identity_preserved": True,
            "bipolar_endpoint_attribution_authorized": False,
            "trigger_source_firewall": deepcopy(_TRIGGER_FIREWALL),
            "transform_receipt_sha256": transform_receipt_sha256,
            "operator_parameter_receipt_sha256": operator_parameter_receipt_sha256,
            "native_measurement_validation_receipt_sha256": native_validation_receipt,
            "raw_dependency_receipt_sha256": raw_receipt,
            "reference_transform_receipt_sha256": reference_transform_receipt_sha256,
            "qc_opportunity_receipt_sha256": qc_receipt,
            "bandwidth_receipt_sha256": bandwidth_receipt,
            "effect_threshold_decision_receipt_sha256": threshold_receipt,
            "minimum_persistence_decision_receipt_sha256": persistence_receipt,
            "query_closure_receipt_sha256": query_closure_receipt_sha256,
            "producer_receipt_sha256": producer_receipt_sha256,
            "permission_receipt_sha256": permission_receipt,
            "measurement_state": "present",
            "measurement_content_sha256": "",
        }
        atom["measurement_content_sha256"] = _self_hash(
            atom, "measurement_content_sha256"
        )
        self.atoms.append(atom)
        return atom


def _check_identity(
    context: Mapping[str, Any], *, event_id: str, recording_id: str, slot: str
) -> None:
    if event_id != context["occurrence_id"]:
        raise ValueError(f"{slot} event_id crosses occurrence_id")
    if recording_id != context["recording_id"]:
        raise ValueError(f"{slot} recording_id crosses attribution context")


def _project_s03(
    collector: _Collector,
    findings_value: object,
    sidecar_value: object,
) -> str:
    findings = validate_event_frequency_findings_v1(findings_value)
    sidecar = _dense_sidecar_from_dict(sidecar_value)
    replay_event_frequency_findings_v1(
        findings, dense_measurement_sidecar=sidecar
    )
    _check_identity(
        collector.context,
        event_id=findings["event_id"],
        recording_id=findings["source"]["recording_id"],
        slot="S03",
    )
    producer_sha = findings["receipt_sha256"]
    collector.trusted_receipts.update(
        {
            producer_sha,
            sidecar.receipt_sha256,
            sidecar.policy.sha256,
        }
    )
    views = {row.view_index: row for row in sidecar.view_bindings}
    source_rows = {
        row.source_binding_sha256: row for row in sidecar.row_bindings
    }
    metric_specs = (
        (
            "dominant_frequency_delta_hz",
            "dominant_frequency_delta_hz",
            "Hz",
            "native-adjacent-dominant-frequency-delta-v1",
        ),
        (
            "spectral_concentration_delta",
            "spectral_concentration_delta",
            "ratio",
            "native-adjacent-spectral-concentration-delta-v1",
        ),
        (
            "spectral_entropy_delta",
            "spectral_entropy_delta",
            "ratio",
            "native-adjacent-spectral-entropy-delta-v1",
        ),
    )
    prefix = collector.context["locked_causal_prefix_interval_s"]
    atom_before = len(collector.atoms)
    for unit in findings["units"]:
        typed = _typed_unit(unit["unit_type"], unit["unit_id"])
        reference = _reference_family(typed)
        points = {
            tuple(point["recording_interval_seconds"]): point
            for point in unit["event_trajectory"]["points"]
        }
        transitions = unit["event_trajectory"]["adjacent_transitions"]
        if not transitions:
            collector.wire_rows.append(
                _wire_row(
                    source_slot_id="S03",
                    source_item_id=f"S03UNIT-{unit['view_index']}-{unit['unit_index']}",
                    producer_receipt_sha256=producer_sha,
                    source_measurement_name_id="adjacent_frequency_spectral_change",
                    source_measurement_state="not_evaluable",
                    typed_unit=typed,
                    reference_family=reference,
                    interval=findings["selection"]["event_course_interval_seconds"],
                    measurement_atom_id=None,
                    missing_required_atom_fields=["change_interval_s"],
                    reason_codes=["no_adjacent_measured_s03_transition"],
                )
            )
            continue
        for transition_index, transition in enumerate(transitions):
            left_key = tuple(transition["from_interval_seconds"])
            right_key = tuple(transition["to_interval_seconds"])
            left_point = points[left_key]
            right_point = points[right_key]
            left = source_rows[left_point["source_binding_sha256"]]
            right = source_rows[right_point["source_binding_sha256"]]
            if left.view_index != right.view_index or left.unit_index != right.unit_index:
                raise ValueError("S03 transition crosses native unit/view identity")
            view = views[left.view_index]
            channels = _source_channels(typed, left.canonical_source_channel_ids)
            if channels != _source_channels(typed, right.canonical_source_channel_ids):
                raise ValueError("S03 transition changes canonical source channels")
            if left.effective_bandwidth_hz != right.effective_bandwidth_hz:
                raise ValueError("S03 transition changes effective bandwidth")
            interval = [left.recording_interval_seconds[0], right.recording_interval_seconds[1]]
            change_interval = list(right.recording_interval_seconds)
            source_ids = [left.source_binding_sha256, right.source_binding_sha256]
            if not _inside(prefix, interval):
                collector.wire_rows.append(
                    _wire_row(
                        source_slot_id="S03",
                        source_item_id=f"S03TRANS-{unit['view_index']}-{unit['unit_index']}-{transition_index}",
                        producer_receipt_sha256=producer_sha,
                        source_measurement_name_id="adjacent_frequency_spectral_change",
                        source_measurement_state="present",
                        typed_unit=typed,
                        reference_family=reference,
                        interval=interval,
                        measurement_atom_id=None,
                        missing_required_atom_fields=["permission_lane"],
                        reason_codes=["source_interval_outside_locked_causal_prefix"],
                    )
                )
                continue
            proposal = collector.add_proposal(
                source_slot_id="S03",
                source_item_ids=source_ids,
                interval=interval,
                typed_unit=typed,
                reference_family=reference,
                producer_receipt_sha256=producer_sha,
                selection_receipt_sha256s=[
                    findings["selection"]["selection_receipt_sha256"]
                ],
            )
            for source_field, name, unit_id, operator in metric_specs:
                atom = collector.add_atom(
                    source_slot_id="S03",
                    measurement_domain="frequency_spectrum",
                    source_item_ids=source_ids,
                    proposal=proposal,
                    interval=interval,
                    change_interval=change_interval,
                    raw_dependency_sha256s=source_ids,
                    raw_dependency_intervals=[
                        left.recording_interval_seconds,
                        right.recording_interval_seconds,
                    ],
                    typed_unit=typed,
                    canonical_source_channels=channels,
                    reference_family=reference,
                    sample_rate_hz=view.output_sampling_rate_hz,
                    physical_unit=unit_id,
                    effective_bandwidth_hz=left.effective_bandwidth_hz,
                    required_bandwidth_hz=(
                        sidecar.policy.analysis_low_hz,
                        sidecar.policy.analysis_high_hz,
                    ),
                    operator_id=operator,
                    operator_version="1.0.0",
                    measurement_name_id=name,
                    value=transition[source_field],
                    transform_receipt_sha256=view.view_receipt_sha256,
                    operator_parameter_receipt_sha256=sidecar.policy.sha256,
                    reference_transform_receipt_sha256=view.view_receipt_sha256,
                    query_closure_receipt_sha256=collector.context[
                        "locked_causal_prefix_receipt_sha256"
                    ],
                    producer_receipt_sha256=producer_sha,
                    permission_lane="onset_causal",
                )
                collector.wire_rows.append(
                    _wire_row(
                        source_slot_id="S03",
                        source_item_id=f"S03TRANS-{unit['view_index']}-{unit['unit_index']}-{transition_index}-{name}",
                        producer_receipt_sha256=producer_sha,
                        source_measurement_name_id=name,
                        source_measurement_state="present",
                        typed_unit=typed,
                        reference_family=reference,
                        interval=interval,
                        measurement_atom_id=atom["measurement_atom_id"],
                        missing_required_atom_fields=[],
                        reason_codes=[
                            "numeric_measurement_present",
                            "effect_threshold_not_evaluable",
                            "minimum_persistence_not_evaluable",
                        ],
                    )
                )
    return "present" if len(collector.atoms) > atom_before else "not_evaluable"


def _project_s04(collector: _Collector, findings_value: object) -> str:
    findings = validate_event_physical_amplitude_findings_v1(findings_value)
    _check_identity(
        collector.context,
        event_id=findings["event_id"],
        recording_id=findings["recording_id"],
        slot="S04",
    )
    producer_sha = findings["receipt_sha256"]
    collector.trusted_receipts.update({producer_sha, findings["policy_sha256"]})
    measurements = {row["measurement_id"]: row for row in findings["measurements"]}
    calibration = {
        (row["view_id"], row["unit_id"]): row
        for row in findings["calibration_ledger"]
    }
    metric_specs = (
        ("delta_rms_uv", "rms_delta_uv", "uV", "native-adjacent-rms-delta-v1"),
        (
            "rms_slope_uv_per_second",
            "rms_slope_uv_per_s",
            "uV_per_s",
            "native-adjacent-rms-slope-v1",
        ),
        (
            "rms_ratio_to_previous",
            "rms_ratio_to_previous",
            "ratio",
            "native-adjacent-rms-ratio-v1",
        ),
        (
            "delta_peak_to_peak_uv",
            "peak_to_peak_delta_uv",
            "uV",
            "native-adjacent-peak-to-peak-delta-v1",
        ),
        (
            "peak_to_peak_slope_uv_per_second",
            "peak_to_peak_slope_uv_per_s",
            "uV_per_s",
            "native-adjacent-peak-to-peak-slope-v1",
        ),
        (
            "peak_to_peak_ratio_to_previous",
            "peak_to_peak_ratio_to_previous",
            "ratio",
            "native-adjacent-peak-to-peak-ratio-v1",
        ),
    )
    prefix = collector.context["locked_causal_prefix_interval_s"]
    atom_before = len(collector.atoms)
    for trajectory in findings["amplitude_trajectories"]:
        typed = _typed_unit(trajectory["unit_type"], trajectory["unit_id"])
        reference = _reference_family(typed)
        calibration_row = calibration[(trajectory["view_id"], trajectory["unit_id"])]
        observed_channels = [
            row["channel_id"] for row in calibration_row["canonical_source_channels"]
        ]
        channels = _source_channels(typed, observed_channels)
        points = {row["ordinal"]: row for row in trajectory["points"]}
        if not trajectory["transition_intervals"]:
            collector.wire_rows.append(
                _wire_row(
                    source_slot_id="S04",
                    source_item_id=trajectory["trajectory_id"],
                    producer_receipt_sha256=producer_sha,
                    source_measurement_name_id="adjacent_physical_amplitude_change",
                    source_measurement_state="not_evaluable",
                    typed_unit=typed,
                    reference_family=reference,
                    interval=findings["analysis_interval_seconds"],
                    measurement_atom_id=None,
                    missing_required_atom_fields=["change_interval_s"],
                    reason_codes=["no_adjacent_measured_s04_transition"],
                )
            )
            continue
        for transition in trajectory["transition_intervals"]:
            left_point = points[transition["from_point_ordinal"]]
            right_point = points[transition["to_point_ordinal"]]
            left = measurements[left_point["measurement_id"]]
            right = measurements[right_point["measurement_id"]]
            interval = [
                left["recording_interval_seconds"][0],
                right["recording_interval_seconds"][1],
            ]
            change_interval = list(right["recording_interval_seconds"])
            source_ids = [
                left["source_amplitude_row_binding_sha256"],
                right["source_amplitude_row_binding_sha256"],
            ]
            if not _inside(prefix, interval):
                collector.wire_rows.append(
                    _wire_row(
                        source_slot_id="S04",
                        source_item_id=transition["transition_sha256"],
                        producer_receipt_sha256=producer_sha,
                        source_measurement_name_id="adjacent_physical_amplitude_change",
                        source_measurement_state="present",
                        typed_unit=typed,
                        reference_family=reference,
                        interval=interval,
                        measurement_atom_id=None,
                        missing_required_atom_fields=["permission_lane"],
                        reason_codes=["source_interval_outside_locked_causal_prefix"],
                    )
                )
                continue
            proposal = collector.add_proposal(
                source_slot_id="S04",
                source_item_ids=source_ids,
                interval=interval,
                typed_unit=typed,
                reference_family=reference,
                producer_receipt_sha256=producer_sha,
                selection_receipt_sha256s=[
                    left["selection_receipt_sha256"],
                    right["selection_receipt_sha256"],
                ],
            )
            for source_field, name, unit_id, operator in metric_specs:
                if transition[source_field] is None:
                    continue
                atom = collector.add_atom(
                    source_slot_id="S04",
                    measurement_domain="physical_amplitude",
                    source_item_ids=source_ids,
                    proposal=proposal,
                    interval=interval,
                    change_interval=change_interval,
                    raw_dependency_sha256s=source_ids,
                    raw_dependency_intervals=[
                        left["recording_interval_seconds"],
                        right["recording_interval_seconds"],
                    ],
                    typed_unit=typed,
                    canonical_source_channels=channels,
                    reference_family=reference,
                    sample_rate_hz=left["sampling_rate_hz"],
                    physical_unit=unit_id,
                    effective_bandwidth_hz=left["effective_bandwidth_hz"],
                    required_bandwidth_hz=(
                        findings["policy"]["required_bandwidth_low_hz"],
                        findings["policy"]["required_bandwidth_high_hz"],
                    ),
                    operator_id=operator,
                    operator_version="1.0.0",
                    measurement_name_id=name,
                    value=transition[source_field],
                    transform_receipt_sha256=calibration_row[
                        "view_receipt_sha256"
                    ],
                    operator_parameter_receipt_sha256=findings["policy_sha256"],
                    reference_transform_receipt_sha256=calibration_row[
                        "view_receipt_sha256"
                    ],
                    query_closure_receipt_sha256=collector.context[
                        "locked_causal_prefix_receipt_sha256"
                    ],
                    producer_receipt_sha256=producer_sha,
                    permission_lane="onset_causal",
                )
                collector.wire_rows.append(
                    _wire_row(
                        source_slot_id="S04",
                        source_item_id=f"{transition['transition_sha256']}-{name}",
                        producer_receipt_sha256=producer_sha,
                        source_measurement_name_id=name,
                        source_measurement_state="present",
                        typed_unit=typed,
                        reference_family=reference,
                        interval=interval,
                        measurement_atom_id=atom["measurement_atom_id"],
                        missing_required_atom_fields=[],
                        reason_codes=[
                            "numeric_measurement_present",
                            "effect_threshold_not_evaluable",
                            "minimum_persistence_not_evaluable",
                        ],
                    )
                )
    return "present" if len(collector.atoms) > atom_before else "not_evaluable"


def _project_s05_unavailable(collector: _Collector, value: object) -> str:
    receipt = validate_event_morphology_primitive_supervision_v1(value)
    _check_identity(
        collector.context,
        event_id=receipt["event_id"],
        recording_id=receipt["recording_id"],
        slot="S05",
    )
    producer_sha = receipt["receipt_sha256"]
    collector.trusted_receipts.add(producer_sha)
    prefix = collector.context["locked_causal_prefix_interval_s"]
    for row in receipt["rows"]:
        source = row["source_binding"]
        typed = _typed_unit(source["unit_type"], source["unit_id"])
        reference = _reference_family(typed)
        interval = source["recording_interval_seconds"]
        available_names = [
            name
            for name, available in zip(
                EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES,
                row["opportunity"]["target_value_mask"],
            )
            if available
        ]
        reasons = []
        missing = []
        if not available_names:
            reasons.append("source_morphology_primitive_not_evaluable")
            missing.append("effect_size_and_unit")
        if not _inside(prefix, interval):
            reasons.append("source_interval_outside_locked_causal_prefix")
            missing.append("permission_lane")
        # The S05 producer declares effective carrier bandwidth but no frozen
        # operator-specific requirement.  Filling required_bandwidth_hz with
        # the carrier itself would be circular and is therefore forbidden.
        reasons.append("s05_required_bandwidth_registry_not_materialized")
        missing.append("required_bandwidth_hz")
        collector.wire_rows.append(
            _wire_row(
                source_slot_id="S05",
                source_item_id=row["row_id"],
                producer_receipt_sha256=producer_sha,
                source_measurement_name_id="waveform_numeric_primitive_roster",
                source_measurement_state=(
                    "present" if available_names else "not_evaluable"
                ),
                typed_unit=typed,
                reference_family=reference,
                interval=interval,
                measurement_atom_id=None,
                missing_required_atom_fields=missing,
                reason_codes=reasons,
            )
        )
    return "not_evaluable"


def _project_s06_unavailable(collector: _Collector, value: object) -> str:
    receipt = validate_event_component_cycle_element_ledger_v1(value)
    _check_identity(
        collector.context,
        event_id=receipt["event_id"],
        recording_id=receipt["recording_id"],
        slot="S06",
    )
    producer_sha = receipt["receipt_sha256"]
    collector.trusted_receipts.add(producer_sha)
    candidates = receipt["source_periodicity_candidates"]
    if not candidates:
        collector.wire_rows.append(
            _wire_row(
                source_slot_id="S06",
                source_item_id=f"S06-{receipt['event_id']}-NO-SOURCE-CANDIDATE",
                producer_receipt_sha256=producer_sha,
                source_measurement_name_id=None,
                source_measurement_state="not_evaluable",
                typed_unit=None,
                reference_family=None,
                interval=receipt["analysis_interval_seconds"],
                measurement_atom_id=None,
                missing_required_atom_fields=[
                    "effect_size_and_unit",
                    "typed_unit",
                    "reference_family",
                ],
                reason_codes=["native_s06_source_candidate_roster_empty"],
            )
        )
    for candidate in candidates:
        typed: dict[str, str] | None = None
        for ledger_name in (
            "element_instance_ledger",
            "cycle_instance_ledger",
            "component_instance_ledger",
        ):
            matching = [
                row
                for row in receipt[ledger_name]["instances"]
                if row["source_candidate_id"] == candidate["candidate_id"]
            ]
            if matching:
                unit = matching[0]["analysis_unit"]
                typed = _typed_unit(unit["unit_type"], unit["unit_id"])
                break
        reasons = [
            "s06_source_explicitly_forbids_onset_support",
            "s06_positive_onset_native_replay_not_admitted",
        ]
        reasons.extend(candidate["reason_codes"])
        collector.wire_rows.append(
            _wire_row(
                source_slot_id="S06",
                source_item_id=candidate["candidate_id"],
                producer_receipt_sha256=producer_sha,
                source_measurement_name_id="component_cycle_interval_roster",
                source_measurement_state=(
                    "present"
                    if candidate["qualification_status"] == "candidate_only"
                    else "not_evaluable"
                ),
                typed_unit=typed,
                reference_family=_reference_family(typed) if typed else None,
                interval=candidate["requested_recording_interval"],
                measurement_atom_id=None,
                missing_required_atom_fields=[
                    "permission_lane",
                    "effect_threshold_decision_receipt_sha256",
                    "minimum_persistence_decision_receipt_sha256",
                ],
                reason_codes=reasons,
            )
        )
    return "not_evaluable"


def _context_receipts(context: Mapping[str, Any]) -> set[str]:
    return {
        value
        for key, value in context.items()
        if key.endswith("_sha256") and isinstance(value, str) and _SHA_RE.fullmatch(value)
    }


def _build_body(
    *,
    context: object,
    trusted_context_receipt_sha256s: Collection[str],
    s03_frequency_findings_receipt: object | None,
    s03_dense_measurement_sidecar_receipt: object | None,
    s04_physical_amplitude_findings_receipt: object | None,
    s05_morphology_primitive_receipt: object | None,
    s06_component_cycle_element_ledger_receipt: object | None,
) -> dict[str, Any]:
    validated_context = validate_onset_trigger_attribution_context_v1_5_1(
        context,
        trusted_receipt_sha256s=trusted_context_receipt_sha256s,
    )
    if validated_context["schema_version"] != ONSET_TRIGGER_ATTRIBUTION_CONTEXT_SCHEMA_VERSION:
        raise ValueError("wire adapter context schema drifted")
    if (
        (s03_frequency_findings_receipt is None)
        != (s03_dense_measurement_sidecar_receipt is None)
    ):
        raise ValueError("S03 Findings and dense sidecar must be supplied together")
    if all(
        value is None
        for value in (
            s03_frequency_findings_receipt,
            s04_physical_amplitude_findings_receipt,
            s05_morphology_primitive_receipt,
            s06_component_cycle_element_ledger_receipt,
        )
    ):
        raise ValueError("wire adapter requires at least one native producer receipt")
    collector = _Collector(validated_context)
    collector.trusted_receipts.update(_context_receipts(validated_context))
    slot_status: dict[str, str] = {slot: "producer_not_supplied" for slot in _SLOTS}
    input_receipts: dict[str, Any] = {
        "s03_frequency_findings_receipt": deepcopy(s03_frequency_findings_receipt),
        "s03_dense_measurement_sidecar_receipt": (
            s03_dense_measurement_sidecar_receipt.to_dict()
            if isinstance(
                s03_dense_measurement_sidecar_receipt,
                BAIEGDenseMeasurementSidecar,
            )
            else deepcopy(s03_dense_measurement_sidecar_receipt)
        ),
        "s04_physical_amplitude_findings_receipt": deepcopy(
            s04_physical_amplitude_findings_receipt
        ),
        "s05_morphology_primitive_receipt": deepcopy(
            s05_morphology_primitive_receipt
        ),
        "s06_component_cycle_element_ledger_receipt": deepcopy(
            s06_component_cycle_element_ledger_receipt
        ),
    }
    if s03_frequency_findings_receipt is not None:
        slot_status["S03"] = _project_s03(
            collector,
            s03_frequency_findings_receipt,
            s03_dense_measurement_sidecar_receipt,
        )
    if s04_physical_amplitude_findings_receipt is not None:
        slot_status["S04"] = _project_s04(
            collector, s04_physical_amplitude_findings_receipt
        )
    if s05_morphology_primitive_receipt is not None:
        slot_status["S05"] = _project_s05_unavailable(
            collector, s05_morphology_primitive_receipt
        )
    if s06_component_cycle_element_ledger_receipt is not None:
        slot_status["S06"] = _project_s06_unavailable(
            collector, s06_component_cycle_element_ledger_receipt
        )

    atoms = sorted(collector.atoms, key=lambda row: row["measurement_atom_id"])
    atom_content = sorted(row["measurement_content_sha256"] for row in atoms)
    trusted_receipts = sorted(collector.trusted_receipts)
    for atom in atoms:
        validate_onset_trigger_measurement_atom_v1_5_1(
            atom,
            context=validated_context,
            trusted_receipt_sha256s=trusted_receipts,
            trusted_measurement_content_sha256s=atom_content,
        )
    wire_rows = sorted(
        collector.wire_rows,
        key=lambda row: (
            row["source_slot_id"],
            row["source_item_id"],
            row["source_measurement_name_id"] or "",
            row["wire_row_id"],
        ),
    )
    summaries = []
    for slot in _SLOTS:
        slot_rows = [row for row in wire_rows if row["source_slot_id"] == slot]
        slot_atoms = [row for row in atoms if row["source_slot_id"] == slot]
        summaries.append(
            {
                "source_slot_id": slot,
                "producer_status": slot_status[slot],
                "wire_row_count": len(slot_rows),
                "strict_measurement_atom_count": len(slot_atoms),
                "not_evaluable_wire_row_count": sum(
                    row["wire_state"] == "not_evaluable" for row in slot_rows
                ),
                "positive_onset_trigger_atom_count": 0,
            }
        )
    return {
        "schema_version": FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_SCHEMA_VERSION,
        "method_id": FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_METHOD_ID,
        "recording_id": validated_context["recording_id"],
        "occurrence_id": validated_context["occurrence_id"],
        "query_index": validated_context["query_index"],
        "inputs": {
            "context": validated_context,
            **input_receipts,
        },
        "source_proposal_receipts": sorted(
            collector.proposals.values(), key=lambda row: row["proposal_id"]
        ),
        "wire_decision_receipts": sorted(
            collector.decisions.values(),
            key=lambda row: (row["decision_kind"], row["receipt_sha256"]),
        ),
        "wire_rows": wire_rows,
        "measurement_atoms": atoms,
        "slot_summaries": summaries,
        "onset_trigger_trust_material": {
            "trusted_receipt_sha256s": trusted_receipts,
            "trusted_measurement_content_sha256s": atom_content,
        },
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "implementation_truth": {
            "S03_adjacent_native_change_atoms_materialized": slot_status["S03"] == "present",
            "S04_adjacent_native_change_atoms_materialized": slot_status["S04"] == "present",
            "S05_strict_atoms_materialized": False,
            "S06_strict_atoms_materialized": False,
            "real_onset_trigger_threshold_registry_admitted": False,
            "positive_onset_trigger_atom_count": 0,
            "clinical_term_qualification_completed": False,
            "SOZ_performance_established": False,
        },
    }


def materialize_findings_native_measurement_atom_wire_adapter_v1(
    *,
    context: object,
    trusted_context_receipt_sha256s: Collection[str],
    s03_frequency_findings_receipt: object | None = None,
    s03_dense_measurement_sidecar_receipt: object | None = None,
    s04_physical_amplitude_findings_receipt: object | None = None,
    s05_morphology_primitive_receipt: object | None = None,
    s06_component_cycle_element_ledger_receipt: object | None = None,
) -> dict[str, Any]:
    """Materialize the replayable S03--S06 producer-to-atom wire receipt."""

    body = _build_body(
        context=context,
        trusted_context_receipt_sha256s=trusted_context_receipt_sha256s,
        s03_frequency_findings_receipt=s03_frequency_findings_receipt,
        s03_dense_measurement_sidecar_receipt=s03_dense_measurement_sidecar_receipt,
        s04_physical_amplitude_findings_receipt=(
            s04_physical_amplitude_findings_receipt
        ),
        s05_morphology_primitive_receipt=s05_morphology_primitive_receipt,
        s06_component_cycle_element_ledger_receipt=(
            s06_component_cycle_element_ledger_receipt
        ),
    )
    body["receipt_sha256"] = _self_hash(body, "receipt_sha256")
    return validate_findings_native_measurement_atom_wire_adapter_v1(
        body,
        trusted_context_receipt_sha256s=trusted_context_receipt_sha256s,
    )


def validate_findings_native_measurement_atom_wire_adapter_v1(
    value: object,
    *,
    trusted_context_receipt_sha256s: Collection[str],
) -> dict[str, Any]:
    """Replay the complete wire receipt from embedded validated producers."""

    if not isinstance(value, Mapping):
        raise TypeError("native measurement-atom wire receipt must be an object")
    payload = deepcopy(dict(value))
    receipt = _sha256(payload.get("receipt_sha256"), "receipt_sha256")
    if receipt != _self_hash(payload, "receipt_sha256"):
        raise ValueError("native measurement-atom wire receipt hash does not replay")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "context",
        "s03_frequency_findings_receipt",
        "s03_dense_measurement_sidecar_receipt",
        "s04_physical_amplitude_findings_receipt",
        "s05_morphology_primitive_receipt",
        "s06_component_cycle_element_ledger_receipt",
    }:
        raise ValueError("native measurement-atom wire input fields drifted")
    expected = _build_body(
        context=inputs["context"],
        trusted_context_receipt_sha256s=trusted_context_receipt_sha256s,
        s03_frequency_findings_receipt=inputs["s03_frequency_findings_receipt"],
        s03_dense_measurement_sidecar_receipt=inputs[
            "s03_dense_measurement_sidecar_receipt"
        ],
        s04_physical_amplitude_findings_receipt=inputs[
            "s04_physical_amplitude_findings_receipt"
        ],
        s05_morphology_primitive_receipt=inputs[
            "s05_morphology_primitive_receipt"
        ],
        s06_component_cycle_element_ledger_receipt=inputs[
            "s06_component_cycle_element_ledger_receipt"
        ],
    )
    observed = deepcopy(payload)
    observed.pop("receipt_sha256")
    if observed != expected:
        raise ValueError("native measurement-atom wire does not replay from producers")
    return payload


def onset_trigger_trust_material_from_wire_adapter_v1(
    value: object,
    *,
    trusted_context_receipt_sha256s: Collection[str],
) -> tuple[set[str], set[str]]:
    """Return the exact receipt/content trust sets validated by this adapter."""

    payload = validate_findings_native_measurement_atom_wire_adapter_v1(
        value,
        trusted_context_receipt_sha256s=trusted_context_receipt_sha256s,
    )
    trust = payload["onset_trigger_trust_material"]
    return (
        set(trust["trusted_receipt_sha256s"]),
        set(trust["trusted_measurement_content_sha256s"]),
    )


__all__ = [
    "FINDINGS_NATIVE_MEASUREMENT_ATOM_SOURCE_PROPOSAL_SCHEMA_VERSION",
    "FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_DECISION_SCHEMA_VERSION",
    "FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_METHOD_ID",
    "FINDINGS_NATIVE_MEASUREMENT_ATOM_WIRE_SCHEMA_VERSION",
    "materialize_findings_native_measurement_atom_wire_adapter_v1",
    "onset_trigger_trust_material_from_wire_adapter_v1",
    "validate_findings_native_measurement_atom_wire_adapter_v1",
]
