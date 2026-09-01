"""Masked-variable DeepSOZ auxiliary target join for the v17 protocol.

This module is deliberately a data-only boundary.  It joins the complete,
target-independent DeepSOZ identity-overlay signal universe to a strictly
rebuilt target-v2 registry and publishes only the ``quarantine_variable_label``
patients that satisfy the prespecified masked-label rule:

* stable ``explicit_1``/``explicit_0`` channels retain their binary target and
  enter the loss;
* ``patient_variable`` and ``missing`` channels are zero placeholders with
  loss mask zero;
* canonical PZ is always a zero placeholder with loss mask zero;
* a patient is admitted only when at least one stable positive and at least
  one signal-eligible event are present.

No model, cache, prediction, private label, majority vote, positive union,
one-hop dilation, or label imputation is accepted by this producer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from ..geometry import CHANNEL_INDEX, STANDARD_19
from . import deepsoz_signal_preflight as _base
from .deepsoz import (
    BINARY_STATE_EXPLICIT_0,
    BINARY_STATE_EXPLICIT_1,
    BINARY_STATE_MISSING,
    BINARY_STATE_PATIENT_VARIABLE,
    PZ_PRIMARY_STATE,
    PatientSOZReference,
)
from .deepsoz_target_independent_signal_universe import (
    TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY,
    TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA,
    VerifiedTargetIndependentSignalUniverse,
    load_target_independent_signal_universe,
)
from .deepsoz_target_v2 import (
    TARGET_V2_POLICY_SHA256,
    TARGET_V2_POLICY_VERSION,
    TARGET_V2_SCHEMA_VERSION,
    VerifiedDeepSOZTargetV2Artifact,
    load_verified_deepsoz_target_v2_artifact,
)


MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA = (
    "soz_deepsoz_masked_variable_auxiliary_target_join_v17"
)
MASKED_VARIABLE_AUXILIARY_JOIN_ARTIFACT_SCHEMA = (
    "soz_deepsoz_masked_variable_auxiliary_target_join_artifact_v17"
)
MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME = (
    "deepsoz_masked_variable_auxiliary_target_join_v17.json"
)
MASKED_VARIABLE_AUXILIARY_ADMISSION_SCHEMA = (
    "soz_deepsoz_masked_variable_auxiliary_admission_only_v17"
)
MASKED_VARIABLE_AUXILIARY_ADMISSION_ARTIFACT_SCHEMA = (
    "soz_deepsoz_masked_variable_auxiliary_admission_only_artifact_v17"
)
MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME = (
    "signal_admission.json"
)
MASKED_VARIABLE_AUXILIARY_ADMISSION_POLICY = (
    "projection_of_v17_join_admitted_identity_event_fold_roster_no_targets"
)
MASKED_VARIABLE_AUXILIARY_JOIN_POLICY = (
    "verified_target_v2_quarantine_variable_label_stable_explicit_only_"
    "masked_conflict_missing_pz_signal_eligible_no_imputation_v17"
)
MASKED_VARIABLE_COHORT_STATUS = "quarantine_variable_label"
PREREGISTERED_AUXILIARY_PATIENT_COUNT = 9
N_AUX_OUTER_FOLDS = 5
_AUX_OUTER_FOLD_IDENTITY_SALT = (
    "labram-v17-masked-variable-aux-outer-fold-20260812"
)
_AUX_OUTER_FOLD_POLICY = (
    "descending_signal_eligible_event_burden_greedy_min_event_load_"
    "min_patient_count_salted_identity_cyclic_tiebreak_v1"
)
_AUX_OUTER_FOLD_TRAIN_CONTRACT = (
    "stable_outer_fold_k_may_train_only_aux_outer_fold_not_equal_k"
)

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MODEL_SPLIT_BY_OFFICIAL = {
    "train": "source_train",
    "dev": "source_dev",
    "eval": "source_eval",
}
_MASKED_STATES = frozenset(
    {BINARY_STATE_PATIENT_VARIABLE, BINARY_STATE_MISSING, PZ_PRIMARY_STATE}
)
_LINEAGE_AXIS_FIELDS = frozenset(
    {
        "direct_target_values",
        "upstream_target_conditioned_roster",
        "target_supervised_model",
    }
)
_LINEAGE_STATE_FIELDS = frozenset({"used", "evidence"})
_LABEL_TRANSFORMATION_FIELDS = frozenset(
    {
        "stable_explicit_1_retained",
        "stable_explicit_0_retained",
        "patient_variable_loss_mask_zero",
        "missing_loss_mask_zero",
        "pz_loss_mask_zero",
        "masked_target_placeholder",
        "majority_vote",
        "positive_union",
        "one_hop_dilation",
        "missing_positive_imputation",
        "private_labels",
        "prediction_based_selection",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "protocol_sha256",
        "signal_universe_artifact_sha256",
        "signal_universe_receipt_sha256",
        "signal_universe_schema",
        "signal_universe_policy",
        "signal_universe_identity_patient_roster_sha256",
        "signal_universe_eligible_event_roster_sha256",
        "signal_universe_identity_patient_count",
        "signal_universe_candidate_event_count",
        "signal_universe_eligible_event_count",
        "target_v2_target_artifact_sha256",
        "target_v2_summary_artifact_sha256",
        "target_v2_readme_artifact_sha256",
        "target_v2_source_input_sha256",
        "target_v2_split_input_sha256",
        "target_v2_policy_sha256",
        "target_v2_verified_receipt_sha256",
        "target_v2_patient_roster_sha256",
        "target_v2_patient_count",
        "target_v2_policy_version",
        "target_v2_schema_version",
    }
)
_PATIENT_FIELDS = frozenset(
    {
        "patient_id",
        "official_split",
        "target_model_split",
        "source_record_count",
        "source_target_exclusion_reason",
        "target",
        "loss_mask",
        "target_states",
        "stable_positive_channels",
        "stable_negative_channels",
        "masked_channels",
        "eligible_event_ids",
        "eligible_event_count",
        "aux_outer_fold",
        "admitted",
        "exclusion_reasons",
        "patient_record_sha256",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "patient_id",
        "official_split",
        "source_model_split",
        "event_record_sha256",
        "crosswalk_record_sha256",
        "processed_window_sha256",
        "preprocess_config_sha256",
        "target_patient_record_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "selected_cohort_status",
        "standard_19",
        "lineage_axes",
        "label_transformations",
        "private_data_accessed",
        "model_or_training_executed",
        "inputs",
        "target_patient_roster_matches_signal_identity_roster",
        "candidate_patient_count",
        "admitted_patient_count",
        "excluded_patient_count",
        "candidate_signal_eligible_event_count",
        "admitted_event_count",
        "excluded_candidate_event_count",
        "preregistered_auxiliary_patient_count",
        "startup_auxiliary_patient_count_gate_pass",
        "aux_outer_fold_count",
        "aux_outer_fold_assignment_policy",
        "aux_outer_fold_assignment_inputs",
        "aux_outer_fold_identity_salt_sha256",
        "aux_outer_fold_target_values_used",
        "aux_outer_fold_train_contract",
        "aux_outer_fold_assignment_sha256",
        "aux_outer_fold_patient_counts",
        "aux_outer_fold_event_counts",
        "candidate_patient_ids",
        "admitted_patient_ids",
        "excluded_patient_ids",
        "admitted_event_ids",
        "candidate_patient_roster_sha256",
        "admitted_patient_roster_sha256",
        "excluded_patient_roster_sha256",
        "admitted_event_roster_sha256",
        "official_split_admitted_patient_counts",
        "official_split_admitted_event_counts",
        "exclusion_reason_counts",
        "patients",
        "events",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "serialization", "receipt_sha256", "receipt"}
)
_ADMISSION_PATIENT_FIELDS = frozenset(
    {
        "patient_id",
        "official_split",
        "aux_outer_fold",
        "eligible_event_count",
        "eligible_event_ids",
    }
)
_ADMISSION_EVENT_FIELDS = _EVENT_FIELDS - {"target_patient_record_sha256"}
_ADMISSION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "lineage_axes",
        "private_data_accessed",
        "model_or_training_executed",
        "source_join_artifact_sha256",
        "source_join_receipt_sha256",
        "signal_universe_artifact_sha256",
        "signal_universe_receipt_sha256",
        "signal_universe_eligible_event_roster_sha256",
        "aux_outer_fold_count",
        "aux_outer_fold_assignment_policy",
        "aux_outer_fold_assignment_inputs",
        "aux_outer_fold_identity_salt_sha256",
        "aux_outer_fold_target_values_used",
        "aux_outer_fold_train_contract",
        "aux_outer_fold_assignment_sha256",
        "aux_outer_fold_patient_counts",
        "aux_outer_fold_event_counts",
        "admitted_patient_count",
        "admitted_event_count",
        "admitted_patient_ids",
        "admitted_event_ids",
        "admitted_patient_roster_sha256",
        "admitted_event_roster_sha256",
        "patients",
        "events",
    }
)
_INPUT_HASH_FIELDS = frozenset(
    field
    for field in _INPUT_FIELDS
    if field.endswith("_sha256")
)


def _lineage_axes() -> dict[str, dict[str, object]]:
    return {
        "direct_target_values": {
            "used": True,
            "evidence": (
                "verified target-v2 patient target states and binary values are "
                "read at this explicit join"
            ),
        },
        "upstream_target_conditioned_roster": {
            "used": True,
            "evidence": (
                "the auxiliary patient roster is selected by the target-derived "
                "quarantine_variable_label status and stable-positive rule; its "
                "event roster inherits that patient join"
            ),
        },
        "target_supervised_model": {
            "used": False,
            "evidence": (
                "this producer accepts no checkpoint, prediction, optimizer, "
                "feature cache, or model output"
            ),
        },
    }


def _admission_lineage_axes() -> dict[str, dict[str, object]]:
    """Lineage visible to a cache process that opens admission-only bytes."""

    return {
        "direct_target_values": {
            "used": False,
            "evidence": (
                "admission-only bytes contain no target, loss mask, target state, "
                "positive/negative channel, or exclusion-reason field"
            ),
        },
        "upstream_target_conditioned_roster": {
            "used": True,
            "evidence": (
                "patient/event admission and fold rows are an exact projection of "
                "the target-conditioned v17 join"
            ),
        },
        "target_supervised_model": {
            "used": False,
            "evidence": (
                "the projection contains identity and signal receipts only and "
                "accepts no model artifact"
            ),
        },
    }


def _label_transformations() -> dict[str, object]:
    return {
        "stable_explicit_1_retained": True,
        "stable_explicit_0_retained": True,
        "patient_variable_loss_mask_zero": True,
        "missing_loss_mask_zero": True,
        "pz_loss_mask_zero": True,
        "masked_target_placeholder": 0,
        "majority_vote": False,
        "positive_union": False,
        "one_hop_dilation": False,
        "missing_positive_imputation": False,
        "private_labels": False,
        "prediction_based_selection": False,
    }


def _salted_identity_sha256(patient_id: str) -> str:
    payload = f"{_AUX_OUTER_FOLD_IDENTITY_SALT}\x1f{patient_id}".encode("utf-8")
    return _base._bytes_sha256(payload)


def assign_aux_outer_folds(
    patient_event_counts: Mapping[str, int],
) -> dict[str, int]:
    """Assign five fixed auxiliary folds without accepting any target field.

    Largest event burdens are placed first into the currently lightest fold.
    Salted patient identity supplies deterministic ordering and a cyclic fold
    preference only when event/patient loads tie.  The intentionally narrow
    function signature prevents label values, channel names, predictions, and
    outcomes from participating in assignment.
    """

    normalized: list[tuple[str, int, str]] = []
    for raw_patient_id, raw_count in patient_event_counts.items():
        patient_id = str(raw_patient_id)
        if not patient_id:
            raise ValueError("Auxiliary fold assignment patient ID is empty")
        count = _strict_nonnegative_int(
            raw_count, field=f"patient_event_counts[{patient_id}]"
        )
        if count < 1:
            raise ValueError("Admitted auxiliary patients require at least one event")
        normalized.append((patient_id, count, _salted_identity_sha256(patient_id)))
    if len({patient_id for patient_id, _, _ in normalized}) != len(normalized):
        raise ValueError("Auxiliary fold assignment contains duplicate patients")
    normalized.sort(key=lambda item: (-item[1], item[2], item[0]))
    event_loads = [0] * N_AUX_OUTER_FOLDS
    patient_loads = [0] * N_AUX_OUTER_FOLDS
    assignment: dict[str, int] = {}
    for patient_id, event_count, identity_sha in normalized:
        preferred = int(identity_sha[:16], 16) % N_AUX_OUTER_FOLDS
        fold = min(
            range(N_AUX_OUTER_FOLDS),
            key=lambda candidate: (
                event_loads[candidate],
                patient_loads[candidate],
                (candidate - preferred) % N_AUX_OUTER_FOLDS,
                candidate,
            ),
        )
        assignment[patient_id] = fold
        event_loads[fold] += event_count
        patient_loads[fold] += 1
    return dict(sorted(assignment.items()))


def _fold_counts(
    admitted_patients: Sequence[Mapping[str, object]], *, count_events: bool
) -> list[list[int]]:
    counts = [0] * N_AUX_OUTER_FOLDS
    for row in admitted_patients:
        fold = row["aux_outer_fold"]
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError("Admitted patient has no integer aux_outer_fold")
        if not 0 <= fold < N_AUX_OUTER_FOLDS:
            raise ValueError("Admitted patient auxiliary fold is out of range")
        counts[fold] += int(row["eligible_event_count"]) if count_events else 1
    return [[fold, counts[fold]] for fold in range(N_AUX_OUTER_FOLDS)]


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _status_from_reference(reference: PatientSOZReference) -> str:
    """Recover the frozen cohort status from the verified exclusion receipt."""

    if not reference.exclusion_reason:
        return "included"
    return reference.exclusion_reason.split(";", 1)[0]


def _canonical_target(reference: PatientSOZReference) -> tuple[list[int], list[int]]:
    """Project verified target states without aggregating or imputing labels."""

    target: list[int] = []
    mask: list[int] = []
    for index, (channel, state) in enumerate(
        zip(STANDARD_19, reference.target_states)
    ):
        source_mask = bool(reference.mask[index].item())
        source_value = int(reference.values[index].item())
        if state == BINARY_STATE_EXPLICIT_1:
            expected = (1, 1)
        elif state == BINARY_STATE_EXPLICIT_0:
            expected = (0, 1)
        elif state in _MASKED_STATES:
            expected = (0, 0)
        else:
            raise ValueError(
                f"Unsupported verified target state for {reference.patient_id}/{channel}: "
                f"{state}"
            )
        if (source_value, int(source_mask)) != expected:
            raise ValueError(
                f"Verified target state/value/mask drifted for "
                f"{reference.patient_id}/{channel}"
            )
        target.append(expected[0])
        mask.append(expected[1])
    pz_index = CHANNEL_INDEX["PZ"]
    if target[pz_index] != 0 or mask[pz_index] != 0:
        raise ValueError("PZ must remain a zero placeholder with loss mask zero")
    return target, mask


def _patient_record_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {key: row[key] for key in sorted(_PATIENT_FIELDS - {"patient_record_sha256"})}


def _count_rows(values: Sequence[str]) -> list[list[object]]:
    counts = Counter(values)
    return [[key, counts[key]] for key in sorted(counts)]


def _split_counts(
    rows: Sequence[Mapping[str, object]],
) -> list[list[object]]:
    return [
        [split, sum(str(row["official_split"]) == split for row in rows)]
        for split in _MODEL_SPLIT_BY_OFFICIAL
    ]


def _input_receipt(
    signal: VerifiedTargetIndependentSignalUniverse,
    target: VerifiedDeepSOZTargetV2Artifact,
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    signal_receipt = signal.receipt
    target_receipt = target.receipt
    return {
        "protocol_sha256": _base._require_sha256(
            protocol_sha256, field="protocol_sha256"
        ),
        "signal_universe_artifact_sha256": signal.artifact_sha256,
        "signal_universe_receipt_sha256": signal.receipt_sha256,
        "signal_universe_schema": signal_receipt["schema_version"],
        "signal_universe_policy": signal_receipt["policy"],
        "signal_universe_identity_patient_roster_sha256": signal_receipt[
            "identity_patient_roster_sha256"
        ],
        "signal_universe_eligible_event_roster_sha256": signal_receipt[
            "eligible_event_roster_sha256"
        ],
        "signal_universe_identity_patient_count": signal_receipt[
            "identity_patient_count"
        ],
        "signal_universe_candidate_event_count": signal_receipt[
            "candidate_event_count"
        ],
        "signal_universe_eligible_event_count": signal_receipt[
            "eligible_event_count"
        ],
        "target_v2_target_artifact_sha256": target_receipt.target_artifact_sha256,
        "target_v2_summary_artifact_sha256": target_receipt.summary_artifact_sha256,
        "target_v2_readme_artifact_sha256": target_receipt.readme_artifact_sha256,
        "target_v2_source_input_sha256": target_receipt.source_input_sha256,
        "target_v2_split_input_sha256": target_receipt.split_input_sha256,
        "target_v2_policy_sha256": target_receipt.policy_sha256,
        "target_v2_verified_receipt_sha256": target_receipt.receipt_sha256,
        "target_v2_patient_roster_sha256": target_receipt.patient_roster_sha256,
        "target_v2_patient_count": target_receipt.patient_count,
        "target_v2_policy_version": target_receipt.policy_version,
        "target_v2_schema_version": target_receipt.target_schema_version,
    }


def _build_join_receipt(
    signal: VerifiedTargetIndependentSignalUniverse,
    target: VerifiedDeepSOZTargetV2Artifact,
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    """Build a join receipt from already strict-loaded public artifacts."""

    if not isinstance(signal, VerifiedTargetIndependentSignalUniverse):
        raise TypeError("signal must be a verified target-independent universe")
    if not isinstance(target, VerifiedDeepSOZTargetV2Artifact):
        raise TypeError("target must be a verified target-v2 artifact")
    signal_receipt = signal.receipt
    if signal_receipt["schema_version"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA:
        raise ValueError("Signal universe schema is not the strict v1 producer")
    if signal_receipt["policy"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY:
        raise ValueError("Signal universe policy drifted")
    if any(
        bool(state["used"])
        for state in signal_receipt["lineage_axes"].values()
    ):
        raise ValueError("Signal universe must precede all target/model conditioning")
    if target.receipt.policy_sha256 != TARGET_V2_POLICY_SHA256:
        raise ValueError("Target-v2 policy SHA drifted")

    signal_patient_ids = tuple(signal_receipt["identity_patient_ids"])
    target_patient_ids = target.receipt.patient_ids
    if signal_patient_ids != target_patient_ids:
        raise ValueError(
            "Target-v2 patient roster must exactly match the signal identity overlay"
        )

    events_by_patient: dict[str, list[Mapping[str, object]]] = {
        patient_id: [] for patient_id in signal_patient_ids
    }
    for event in signal_receipt["events"]:
        patient_id = str(event["patient_id"])
        if patient_id not in events_by_patient:
            raise ValueError("Signal event patient is outside the joined patient roster")
        events_by_patient[patient_id].append(event)
    for rows in events_by_patient.values():
        rows.sort(key=lambda row: str(row["event_id"]))

    patient_rows: list[dict[str, object]] = []
    candidate_signal_event_count = 0
    for reference in target.registry:
        cohort_status = _status_from_reference(reference)
        if cohort_status != MASKED_VARIABLE_COHORT_STATUS:
            continue
        if reference.model_split != "quarantine":
            raise ValueError(
                "quarantine_variable_label patient is not in the quarantine split"
            )
        target_values, loss_mask = _canonical_target(reference)
        positive_channels = [
            channel
            for channel, value, mask in zip(STANDARD_19, target_values, loss_mask)
            if value == 1 and mask == 1
        ]
        negative_channels = [
            channel
            for channel, value, mask in zip(STANDARD_19, target_values, loss_mask)
            if value == 0 and mask == 1
        ]
        masked_channels = [
            channel for channel, mask in zip(STANDARD_19, loss_mask) if mask == 0
        ]
        signal_events = events_by_patient[reference.patient_id]
        event_ids = [str(row["event_id"]) for row in signal_events]
        candidate_signal_event_count += len(event_ids)
        exclusion_reasons: list[str] = []
        if not positive_channels:
            exclusion_reasons.append("no_stable_unmasked_in_head_positive")
        if not event_ids:
            exclusion_reasons.append("no_signal_eligible_event")
        admitted = not exclusion_reasons
        patient_row: dict[str, object] = {
            "patient_id": reference.patient_id,
            "official_split": reference.official_split,
            "target_model_split": reference.model_split,
            "source_record_count": reference.source_record_count,
            "source_target_exclusion_reason": reference.exclusion_reason,
            "target": target_values,
            "loss_mask": loss_mask,
            "target_states": list(reference.target_states),
            "stable_positive_channels": positive_channels,
            "stable_negative_channels": negative_channels,
            "masked_channels": masked_channels,
            "eligible_event_ids": event_ids,
            "eligible_event_count": len(event_ids),
            "aux_outer_fold": None,
            "admitted": admitted,
            "exclusion_reasons": exclusion_reasons,
        }
        patient_row["patient_record_sha256"] = "0" * 64
        patient_rows.append(patient_row)

    patient_rows.sort(key=lambda row: str(row["patient_id"]))
    admitted_rows = [row for row in patient_rows if bool(row["admitted"])]
    fold_assignment = assign_aux_outer_folds(
        {
            str(row["patient_id"]): int(row["eligible_event_count"])
            for row in admitted_rows
        }
    )
    for patient_row in patient_rows:
        patient_id = str(patient_row["patient_id"])
        patient_row["aux_outer_fold"] = (
            fold_assignment[patient_id] if bool(patient_row["admitted"]) else None
        )
        patient_row["patient_record_sha256"] = _base._canonical_sha256(
            _patient_record_payload(patient_row)
        )

    admitted_events: list[dict[str, object]] = []
    for patient_row in admitted_rows:
        patient_id = str(patient_row["patient_id"])
        official_split = str(patient_row["official_split"])
        for event in events_by_patient[patient_id]:
            if event["official_split"] != official_split:
                raise ValueError("Signal/target official split disagrees for patient")
            expected_source_split = _MODEL_SPLIT_BY_OFFICIAL[official_split]
            if event["model_split"] != expected_source_split:
                raise ValueError("Signal model split is not source-native")
            admitted_events.append(
                {
                    "event_id": event["event_id"],
                    "patient_id": patient_id,
                    "official_split": official_split,
                    "source_model_split": event["model_split"],
                    "event_record_sha256": event["event_record_sha256"],
                    "crosswalk_record_sha256": event[
                        "crosswalk_record_sha256"
                    ],
                    "processed_window_sha256": event[
                        "processed_window_sha256"
                    ],
                    "preprocess_config_sha256": event[
                        "preprocess_config_sha256"
                    ],
                    "target_patient_record_sha256": patient_row[
                        "patient_record_sha256"
                    ],
                }
            )
    admitted_events.sort(key=lambda row: str(row["event_id"]))
    candidate_ids = [str(row["patient_id"]) for row in patient_rows]
    admitted_ids = [
        str(row["patient_id"]) for row in patient_rows if bool(row["admitted"])
    ]
    excluded_ids = [
        str(row["patient_id"]) for row in patient_rows if not bool(row["admitted"])
    ]
    admitted_event_ids = [str(row["event_id"]) for row in admitted_events]
    exclusion_codes = [
        str(reason)
        for row in patient_rows
        for reason in row["exclusion_reasons"]
    ]
    receipt: dict[str, object] = {
        "schema_version": MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA,
        "policy": MASKED_VARIABLE_AUXILIARY_JOIN_POLICY,
        "selected_cohort_status": MASKED_VARIABLE_COHORT_STATUS,
        "standard_19": list(STANDARD_19),
        "lineage_axes": _lineage_axes(),
        "label_transformations": _label_transformations(),
        "private_data_accessed": False,
        "model_or_training_executed": False,
        "inputs": _input_receipt(
            signal, target, protocol_sha256=protocol_sha256
        ),
        "target_patient_roster_matches_signal_identity_roster": True,
        "candidate_patient_count": len(candidate_ids),
        "admitted_patient_count": len(admitted_ids),
        "excluded_patient_count": len(excluded_ids),
        "candidate_signal_eligible_event_count": candidate_signal_event_count,
        "admitted_event_count": len(admitted_event_ids),
        "excluded_candidate_event_count": (
            candidate_signal_event_count - len(admitted_event_ids)
        ),
        "preregistered_auxiliary_patient_count": (
            PREREGISTERED_AUXILIARY_PATIENT_COUNT
        ),
        "startup_auxiliary_patient_count_gate_pass": (
            len(admitted_ids) == PREREGISTERED_AUXILIARY_PATIENT_COUNT
        ),
        "aux_outer_fold_count": N_AUX_OUTER_FOLDS,
        "aux_outer_fold_assignment_policy": _AUX_OUTER_FOLD_POLICY,
        "aux_outer_fold_assignment_inputs": [
            "patient_id",
            "eligible_event_count",
        ],
        "aux_outer_fold_identity_salt_sha256": _base._bytes_sha256(
            _AUX_OUTER_FOLD_IDENTITY_SALT.encode("utf-8")
        ),
        "aux_outer_fold_target_values_used": False,
        "aux_outer_fold_train_contract": _AUX_OUTER_FOLD_TRAIN_CONTRACT,
        "aux_outer_fold_assignment_sha256": _base._canonical_sha256(
            [[patient_id, fold_assignment[patient_id]] for patient_id in admitted_ids]
        ),
        "aux_outer_fold_patient_counts": _fold_counts(
            admitted_rows, count_events=False
        ),
        "aux_outer_fold_event_counts": _fold_counts(
            admitted_rows, count_events=True
        ),
        "candidate_patient_ids": candidate_ids,
        "admitted_patient_ids": admitted_ids,
        "excluded_patient_ids": excluded_ids,
        "admitted_event_ids": admitted_event_ids,
        "candidate_patient_roster_sha256": _base._roster_sha256(candidate_ids),
        "admitted_patient_roster_sha256": _base._roster_sha256(admitted_ids),
        "excluded_patient_roster_sha256": _base._roster_sha256(excluded_ids),
        "admitted_event_roster_sha256": _base._roster_sha256(
            admitted_event_ids
        ),
        "official_split_admitted_patient_counts": _split_counts(
            admitted_rows
        ),
        "official_split_admitted_event_counts": _split_counts(admitted_events),
        "exclusion_reason_counts": _count_rows(exclusion_codes),
        "patients": patient_rows,
        "events": admitted_events,
    }
    _validate_receipt(receipt)
    return receipt


def _validate_lineage_axes(value: object) -> None:
    axes = _base._closed_object(
        value, expected=_LINEAGE_AXIS_FIELDS, field="lineage_axes"
    )
    expected = _lineage_axes()
    for axis in sorted(_LINEAGE_AXIS_FIELDS):
        state = _base._closed_object(
            axes[axis],
            expected=_LINEAGE_STATE_FIELDS,
            field=f"lineage_axes.{axis}",
        )
        if state != expected[axis]:
            raise ValueError(f"Masked-variable join lineage axis drifted: {axis}")


def _validate_split_counts(
    value: object,
    *,
    expected_rows: Sequence[Mapping[str, object]],
    field: str,
) -> None:
    if value != _split_counts(expected_rows):
        raise ValueError(f"{field} disagrees with the stored roster")


def _validate_receipt(value: object) -> dict[str, object]:
    receipt = _base._closed_object(
        value, expected=_RECEIPT_FIELDS, field="masked-variable join receipt"
    )
    if receipt["schema_version"] != MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA:
        raise ValueError("Unsupported masked-variable join schema")
    if receipt["policy"] != MASKED_VARIABLE_AUXILIARY_JOIN_POLICY:
        raise ValueError("Masked-variable join policy drifted")
    if receipt["selected_cohort_status"] != MASKED_VARIABLE_COHORT_STATUS:
        raise ValueError("Masked-variable selected cohort status drifted")
    if receipt["standard_19"] != list(STANDARD_19):
        raise ValueError("Masked-variable join channel order drifted")
    _validate_lineage_axes(receipt["lineage_axes"])
    transformations = _base._closed_object(
        receipt["label_transformations"],
        expected=_LABEL_TRANSFORMATION_FIELDS,
        field="label_transformations",
    )
    if transformations != _label_transformations():
        raise ValueError("Masked-variable label transformations drifted")
    if _strict_bool(
        receipt["private_data_accessed"], field="private_data_accessed"
    ):
        raise ValueError("Private data are forbidden in the auxiliary join")
    if _strict_bool(
        receipt["model_or_training_executed"], field="model_or_training_executed"
    ):
        raise ValueError("Model/training execution is forbidden in the join")
    if receipt["target_patient_roster_matches_signal_identity_roster"] is not True:
        raise ValueError("Signal/target patient roster equality must be explicit")

    inputs = _base._closed_object(
        receipt["inputs"], expected=_INPUT_FIELDS, field="inputs"
    )
    for field in _INPUT_HASH_FIELDS:
        _base._require_sha256(inputs[field], field=f"inputs.{field}")
    if inputs["signal_universe_schema"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_SCHEMA:
        raise ValueError("Input signal-universe schema drifted")
    if inputs["signal_universe_policy"] != TARGET_INDEPENDENT_SIGNAL_UNIVERSE_POLICY:
        raise ValueError("Input signal-universe policy drifted")
    if inputs["target_v2_policy_sha256"] != TARGET_V2_POLICY_SHA256:
        raise ValueError("Input target-v2 policy SHA drifted")
    if inputs["target_v2_policy_version"] != TARGET_V2_POLICY_VERSION:
        raise ValueError("Input target-v2 policy version drifted")
    if inputs["target_v2_schema_version"] != TARGET_V2_SCHEMA_VERSION:
        raise ValueError("Input target-v2 schema drifted")
    for field in (
        "signal_universe_identity_patient_count",
        "signal_universe_candidate_event_count",
        "signal_universe_eligible_event_count",
        "target_v2_patient_count",
    ):
        _strict_nonnegative_int(inputs[field], field=f"inputs.{field}")

    patients_value = receipt["patients"]
    events_value = receipt["events"]
    if not isinstance(patients_value, list) or not isinstance(events_value, list):
        raise ValueError("Masked-variable patients/events must be JSON arrays")
    patients: list[dict[str, object]] = []
    patient_by_id: dict[str, dict[str, object]] = {}
    candidate_event_ids: set[str] = set()
    for index, raw in enumerate(patients_value):
        row = _base._closed_object(
            raw, expected=_PATIENT_FIELDS, field=f"patients[{index}]"
        )
        patient_id = str(row["patient_id"])
        if not patient_id or patient_id in patient_by_id:
            raise ValueError("Patient audit contains an empty or duplicate patient ID")
        if row["target_model_split"] != "quarantine":
            raise ValueError("Auxiliary candidate is not quarantined")
        official_split = str(row["official_split"])
        if official_split not in _MODEL_SPLIT_BY_OFFICIAL:
            raise ValueError("Auxiliary candidate has an invalid official split")
        if not str(row["source_target_exclusion_reason"]).split(";", 1)[0] == (
            MASKED_VARIABLE_COHORT_STATUS
        ):
            raise ValueError("Patient was not selected by quarantine_variable_label")
        _strict_nonnegative_int(
            row["source_record_count"], field=f"patients[{index}].source_record_count"
        )
        target = row["target"]
        loss_mask = row["loss_mask"]
        states = row["target_states"]
        if not all(isinstance(item, list) for item in (target, loss_mask, states)):
            raise ValueError("Patient target, mask, and states must be JSON arrays")
        if not (len(target) == len(loss_mask) == len(states) == len(STANDARD_19)):
            raise ValueError("Patient target/mask/state shape must be [19]")
        expected_positive: list[str] = []
        expected_negative: list[str] = []
        expected_masked: list[str] = []
        for channel_index, (channel, target_value, mask_value, state) in enumerate(
            zip(STANDARD_19, target, loss_mask, states)
        ):
            if isinstance(target_value, bool) or target_value not in (0, 1):
                raise ValueError("Auxiliary target values must be integer binary")
            if isinstance(mask_value, bool) or mask_value not in (0, 1):
                raise ValueError("Auxiliary loss masks must be integer binary")
            if state == BINARY_STATE_EXPLICIT_1:
                expected = (1, 1)
                expected_positive.append(channel)
            elif state == BINARY_STATE_EXPLICIT_0:
                expected = (0, 1)
                expected_negative.append(channel)
            elif state in _MASKED_STATES:
                expected = (0, 0)
                expected_masked.append(channel)
            else:
                raise ValueError("Patient target contains an unsupported state")
            if (target_value, mask_value) != expected:
                raise ValueError("Patient target/mask disagrees with target state")
            if channel_index == CHANNEL_INDEX["PZ"] and state != PZ_PRIMARY_STATE:
                raise ValueError("PZ state must stay masked by the primary policy")
        if row["stable_positive_channels"] != expected_positive:
            raise ValueError("stable_positive_channels disagrees with target")
        if row["stable_negative_channels"] != expected_negative:
            raise ValueError("stable_negative_channels disagrees with target")
        if row["masked_channels"] != expected_masked:
            raise ValueError("masked_channels disagrees with loss mask")
        event_ids = row["eligible_event_ids"]
        if (
            not isinstance(event_ids, list)
            or event_ids != sorted(set(event_ids))
            or any(not isinstance(event_id, str) or not event_id for event_id in event_ids)
        ):
            raise ValueError("Patient eligible-event roster is not sorted unique")
        if row["eligible_event_count"] != len(event_ids):
            raise ValueError("Patient eligible_event_count disagrees with its roster")
        if candidate_event_ids.intersection(event_ids):
            raise ValueError("Candidate patient signal-event rosters overlap")
        candidate_event_ids.update(event_ids)
        expected_exclusions: list[str] = []
        if not expected_positive:
            expected_exclusions.append("no_stable_unmasked_in_head_positive")
        if not event_ids:
            expected_exclusions.append("no_signal_eligible_event")
        if row["exclusion_reasons"] != expected_exclusions:
            raise ValueError("Patient exclusion reasons disagree with admission rules")
        admitted = _strict_bool(row["admitted"], field=f"patients[{index}].admitted")
        if admitted != (not expected_exclusions):
            raise ValueError("Patient admission flag disagrees with exclusion rules")
        aux_fold = row["aux_outer_fold"]
        if admitted:
            if (
                isinstance(aux_fold, bool)
                or not isinstance(aux_fold, int)
                or not 0 <= aux_fold < N_AUX_OUTER_FOLDS
            ):
                raise ValueError("Admitted patient requires a valid aux_outer_fold")
        elif aux_fold is not None:
            raise ValueError("Excluded patient must not receive an auxiliary fold")
        declared_record_sha = _base._require_sha256(
            row["patient_record_sha256"],
            field=f"patients[{index}].patient_record_sha256",
        )
        if declared_record_sha != _base._canonical_sha256(
            _patient_record_payload(row)
        ):
            raise ValueError("Patient audit record SHA mismatch")
        patients.append(row)
        patient_by_id[patient_id] = row

    patient_ids = [str(row["patient_id"]) for row in patients]
    if patient_ids != sorted(patient_ids):
        raise ValueError("Patient audit is not canonically ordered")
    events: list[dict[str, object]] = []
    for index, raw in enumerate(events_value):
        row = _base._closed_object(
            raw, expected=_EVENT_FIELDS, field=f"events[{index}]"
        )
        patient_id = str(row["patient_id"])
        patient = patient_by_id.get(patient_id)
        if patient is None or not bool(patient["admitted"]):
            raise ValueError("Admitted event is not bound to an admitted patient")
        event_id = str(row["event_id"])
        if event_id not in patient["eligible_event_ids"]:
            raise ValueError("Admitted event is outside the patient's signal roster")
        if row["official_split"] != patient["official_split"]:
            raise ValueError("Admitted event official split disagrees with patient")
        if row["source_model_split"] != _MODEL_SPLIT_BY_OFFICIAL[
            str(row["official_split"])
        ]:
            raise ValueError("Admitted event source split is not mechanically derived")
        for field in (
            "event_record_sha256",
            "crosswalk_record_sha256",
            "processed_window_sha256",
            "preprocess_config_sha256",
            "target_patient_record_sha256",
        ):
            _base._require_sha256(row[field], field=f"events[{index}].{field}")
        if row["target_patient_record_sha256"] != patient["patient_record_sha256"]:
            raise ValueError("Event target-patient binding SHA mismatch")
        events.append(row)
    event_ids = [str(row["event_id"]) for row in events]
    if event_ids != sorted(event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("Admitted event roster is not sorted unique")
    expected_event_ids = sorted(
        event_id
        for row in patients
        if bool(row["admitted"])
        for event_id in row["eligible_event_ids"]
    )
    if event_ids != expected_event_ids:
        raise ValueError("Admitted event rows do not close admitted patient rosters")

    candidate_ids = patient_ids
    admitted_ids = [
        str(row["patient_id"]) for row in patients if bool(row["admitted"])
    ]
    excluded_ids = [
        str(row["patient_id"]) for row in patients if not bool(row["admitted"])
    ]
    roster_checks = {
        "candidate_patient_ids": candidate_ids,
        "admitted_patient_ids": admitted_ids,
        "excluded_patient_ids": excluded_ids,
        "admitted_event_ids": event_ids,
    }
    for field, expected in roster_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"{field} disagrees with stored records")
    count_checks = {
        "candidate_patient_count": len(candidate_ids),
        "admitted_patient_count": len(admitted_ids),
        "excluded_patient_count": len(excluded_ids),
        "candidate_signal_eligible_event_count": len(candidate_event_ids),
        "admitted_event_count": len(event_ids),
        "excluded_candidate_event_count": len(candidate_event_ids) - len(event_ids),
    }
    for field, expected in count_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"{field} disagrees with stored records")
    if receipt["preregistered_auxiliary_patient_count"] != (
        PREREGISTERED_AUXILIARY_PATIENT_COUNT
    ):
        raise ValueError("Preregistered auxiliary patient count drifted")
    if receipt["startup_auxiliary_patient_count_gate_pass"] is not (
        len(admitted_ids) == PREREGISTERED_AUXILIARY_PATIENT_COUNT
    ):
        raise ValueError("Auxiliary startup count gate disagrees with roster")
    if receipt["aux_outer_fold_count"] != N_AUX_OUTER_FOLDS:
        raise ValueError("Auxiliary outer-fold count drifted")
    if receipt["aux_outer_fold_assignment_policy"] != _AUX_OUTER_FOLD_POLICY:
        raise ValueError("Auxiliary outer-fold policy drifted")
    if receipt["aux_outer_fold_assignment_inputs"] != [
        "patient_id",
        "eligible_event_count",
    ]:
        raise ValueError("Auxiliary fold assignment inputs drifted")
    if receipt["aux_outer_fold_identity_salt_sha256"] != _base._bytes_sha256(
        _AUX_OUTER_FOLD_IDENTITY_SALT.encode("utf-8")
    ):
        raise ValueError("Auxiliary fold identity salt drifted")
    if _strict_bool(
        receipt["aux_outer_fold_target_values_used"],
        field="aux_outer_fold_target_values_used",
    ):
        raise ValueError("Auxiliary fold assignment may not use target values")
    if receipt["aux_outer_fold_train_contract"] != _AUX_OUTER_FOLD_TRAIN_CONTRACT:
        raise ValueError("Auxiliary fold training contract drifted")
    admitted_rows = [row for row in patients if bool(row["admitted"])]
    expected_assignment = assign_aux_outer_folds(
        {
            str(row["patient_id"]): int(row["eligible_event_count"])
            for row in admitted_rows
        }
    )
    observed_assignment = {
        str(row["patient_id"]): int(row["aux_outer_fold"])
        for row in admitted_rows
    }
    if observed_assignment != expected_assignment:
        raise ValueError("Auxiliary outer-fold assignment is not reproducible")
    expected_assignment_sha = _base._canonical_sha256(
        [[patient_id, expected_assignment[patient_id]] for patient_id in admitted_ids]
    )
    if receipt["aux_outer_fold_assignment_sha256"] != expected_assignment_sha:
        raise ValueError("Auxiliary outer-fold assignment SHA mismatch")
    if receipt["aux_outer_fold_patient_counts"] != _fold_counts(
        admitted_rows, count_events=False
    ):
        raise ValueError("Auxiliary outer-fold patient counts drifted")
    if receipt["aux_outer_fold_event_counts"] != _fold_counts(
        admitted_rows, count_events=True
    ):
        raise ValueError("Auxiliary outer-fold event counts drifted")
    roster_hash_checks = {
        "candidate_patient_roster_sha256": candidate_ids,
        "admitted_patient_roster_sha256": admitted_ids,
        "excluded_patient_roster_sha256": excluded_ids,
        "admitted_event_roster_sha256": event_ids,
    }
    for field, roster in roster_hash_checks.items():
        if receipt[field] != _base._roster_sha256(roster):
            raise ValueError(f"{field} disagrees with its roster")
    _validate_split_counts(
        receipt["official_split_admitted_patient_counts"],
        expected_rows=[row for row in patients if bool(row["admitted"])],
        field="official_split_admitted_patient_counts",
    )
    _validate_split_counts(
        receipt["official_split_admitted_event_counts"],
        expected_rows=events,
        field="official_split_admitted_event_counts",
    )
    exclusion_codes = [
        str(reason)
        for row in patients
        for reason in row["exclusion_reasons"]
    ]
    if receipt["exclusion_reason_counts"] != _count_rows(exclusion_codes):
        raise ValueError("exclusion_reason_counts disagrees with patient audits")
    return receipt


def _build_admission_receipt(
    join_receipt: Mapping[str, object],
    *,
    source_join_artifact_sha256: str,
    source_join_receipt_sha256: str,
) -> dict[str, object]:
    """Project the joined roster without serializing any target-bearing field."""

    validated = _validate_receipt(join_receipt)
    admitted_patients = [
        row for row in validated["patients"] if bool(row["admitted"])
    ]
    patients = [
        {
            "patient_id": row["patient_id"],
            "official_split": row["official_split"],
            "aux_outer_fold": row["aux_outer_fold"],
            "eligible_event_count": row["eligible_event_count"],
            "eligible_event_ids": row["eligible_event_ids"],
        }
        for row in admitted_patients
    ]
    events = [
        {
            field: row[field]
            for field in sorted(_ADMISSION_EVENT_FIELDS)
        }
        for row in validated["events"]
    ]
    inputs = validated["inputs"]
    receipt: dict[str, object] = {
        "schema_version": MASKED_VARIABLE_AUXILIARY_ADMISSION_SCHEMA,
        "policy": MASKED_VARIABLE_AUXILIARY_ADMISSION_POLICY,
        "lineage_axes": _admission_lineage_axes(),
        "private_data_accessed": False,
        "model_or_training_executed": False,
        "source_join_artifact_sha256": _base._require_sha256(
            source_join_artifact_sha256, field="source_join_artifact_sha256"
        ),
        "source_join_receipt_sha256": _base._require_sha256(
            source_join_receipt_sha256, field="source_join_receipt_sha256"
        ),
        "signal_universe_artifact_sha256": inputs[
            "signal_universe_artifact_sha256"
        ],
        "signal_universe_receipt_sha256": inputs[
            "signal_universe_receipt_sha256"
        ],
        "signal_universe_eligible_event_roster_sha256": inputs[
            "signal_universe_eligible_event_roster_sha256"
        ],
        "aux_outer_fold_count": validated["aux_outer_fold_count"],
        "aux_outer_fold_assignment_policy": validated[
            "aux_outer_fold_assignment_policy"
        ],
        "aux_outer_fold_assignment_inputs": validated[
            "aux_outer_fold_assignment_inputs"
        ],
        "aux_outer_fold_identity_salt_sha256": validated[
            "aux_outer_fold_identity_salt_sha256"
        ],
        "aux_outer_fold_target_values_used": False,
        "aux_outer_fold_train_contract": validated[
            "aux_outer_fold_train_contract"
        ],
        "aux_outer_fold_assignment_sha256": validated[
            "aux_outer_fold_assignment_sha256"
        ],
        "aux_outer_fold_patient_counts": validated[
            "aux_outer_fold_patient_counts"
        ],
        "aux_outer_fold_event_counts": validated["aux_outer_fold_event_counts"],
        "admitted_patient_count": validated["admitted_patient_count"],
        "admitted_event_count": validated["admitted_event_count"],
        "admitted_patient_ids": validated["admitted_patient_ids"],
        "admitted_event_ids": validated["admitted_event_ids"],
        "admitted_patient_roster_sha256": validated[
            "admitted_patient_roster_sha256"
        ],
        "admitted_event_roster_sha256": validated[
            "admitted_event_roster_sha256"
        ],
        "patients": patients,
        "events": events,
    }
    _validate_admission_receipt(receipt)
    return receipt


def _validate_admission_lineage_axes(value: object) -> None:
    axes = _base._closed_object(
        value, expected=_LINEAGE_AXIS_FIELDS, field="admission lineage_axes"
    )
    expected = _admission_lineage_axes()
    for axis in sorted(_LINEAGE_AXIS_FIELDS):
        state = _base._closed_object(
            axes[axis],
            expected=_LINEAGE_STATE_FIELDS,
            field=f"admission lineage_axes.{axis}",
        )
        if state != expected[axis]:
            raise ValueError(f"Admission-only lineage axis drifted: {axis}")


def _validate_admission_receipt(value: object) -> dict[str, object]:
    receipt = _base._closed_object(
        value,
        expected=_ADMISSION_RECEIPT_FIELDS,
        field="masked-variable admission-only receipt",
    )
    if receipt["schema_version"] != MASKED_VARIABLE_AUXILIARY_ADMISSION_SCHEMA:
        raise ValueError("Unsupported admission-only schema")
    if receipt["policy"] != MASKED_VARIABLE_AUXILIARY_ADMISSION_POLICY:
        raise ValueError("Admission-only policy drifted")
    _validate_admission_lineage_axes(receipt["lineage_axes"])
    if _strict_bool(
        receipt["private_data_accessed"], field="admission.private_data_accessed"
    ):
        raise ValueError("Private data are forbidden in admission-only projection")
    if _strict_bool(
        receipt["model_or_training_executed"],
        field="admission.model_or_training_executed",
    ):
        raise ValueError("Model execution is forbidden in admission-only projection")
    for field in (
        "source_join_artifact_sha256",
        "source_join_receipt_sha256",
        "signal_universe_artifact_sha256",
        "signal_universe_receipt_sha256",
        "signal_universe_eligible_event_roster_sha256",
        "aux_outer_fold_identity_salt_sha256",
        "aux_outer_fold_assignment_sha256",
        "admitted_patient_roster_sha256",
        "admitted_event_roster_sha256",
    ):
        _base._require_sha256(receipt[field], field=f"admission.{field}")
    if receipt["aux_outer_fold_count"] != N_AUX_OUTER_FOLDS:
        raise ValueError("Admission-only outer-fold count drifted")
    if receipt["aux_outer_fold_assignment_policy"] != _AUX_OUTER_FOLD_POLICY:
        raise ValueError("Admission-only outer-fold policy drifted")
    if receipt["aux_outer_fold_assignment_inputs"] != [
        "patient_id",
        "eligible_event_count",
    ]:
        raise ValueError("Admission-only fold inputs drifted")
    if receipt["aux_outer_fold_identity_salt_sha256"] != _base._bytes_sha256(
        _AUX_OUTER_FOLD_IDENTITY_SALT.encode("utf-8")
    ):
        raise ValueError("Admission-only fold salt drifted")
    if _strict_bool(
        receipt["aux_outer_fold_target_values_used"],
        field="admission.aux_outer_fold_target_values_used",
    ):
        raise ValueError("Admission-only fold assignment cannot use targets")
    if receipt["aux_outer_fold_train_contract"] != _AUX_OUTER_FOLD_TRAIN_CONTRACT:
        raise ValueError("Admission-only training contract drifted")

    patients_value = receipt["patients"]
    events_value = receipt["events"]
    if not isinstance(patients_value, list) or not isinstance(events_value, list):
        raise ValueError("Admission-only patients/events must be JSON arrays")
    patients: list[dict[str, object]] = []
    patient_by_id: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(patients_value):
        row = _base._closed_object(
            raw,
            expected=_ADMISSION_PATIENT_FIELDS,
            field=f"admission.patients[{index}]",
        )
        patient_id = str(row["patient_id"])
        if not patient_id or patient_id in patient_by_id:
            raise ValueError("Admission-only patient IDs must be unique and non-empty")
        if str(row["official_split"]) not in _MODEL_SPLIT_BY_OFFICIAL:
            raise ValueError("Admission-only patient has an invalid official split")
        fold = row["aux_outer_fold"]
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or not 0 <= fold < N_AUX_OUTER_FOLDS
        ):
            raise ValueError("Admission-only patient has an invalid auxiliary fold")
        event_ids = row["eligible_event_ids"]
        if (
            not isinstance(event_ids, list)
            or event_ids != sorted(set(event_ids))
            or not event_ids
            or any(not isinstance(event_id, str) or not event_id for event_id in event_ids)
        ):
            raise ValueError("Admission-only event IDs must be non-empty sorted unique")
        if row["eligible_event_count"] != len(event_ids):
            raise ValueError("Admission-only patient event count drifted")
        patients.append(row)
        patient_by_id[patient_id] = row
    patient_ids = [str(row["patient_id"]) for row in patients]
    if patient_ids != sorted(patient_ids):
        raise ValueError("Admission-only patients are not canonically ordered")
    expected_assignment = assign_aux_outer_folds(
        {
            str(row["patient_id"]): int(row["eligible_event_count"])
            for row in patients
        }
    )
    if any(
        int(row["aux_outer_fold"]) != expected_assignment[str(row["patient_id"])]
        for row in patients
    ):
        raise ValueError("Admission-only fold assignment is not reproducible")
    expected_assignment_sha = _base._canonical_sha256(
        [[patient_id, expected_assignment[patient_id]] for patient_id in patient_ids]
    )
    if receipt["aux_outer_fold_assignment_sha256"] != expected_assignment_sha:
        raise ValueError("Admission-only fold assignment SHA mismatch")
    if receipt["aux_outer_fold_patient_counts"] != _fold_counts(
        patients, count_events=False
    ):
        raise ValueError("Admission-only fold patient counts drifted")
    if receipt["aux_outer_fold_event_counts"] != _fold_counts(
        patients, count_events=True
    ):
        raise ValueError("Admission-only fold event counts drifted")

    events: list[dict[str, object]] = []
    seen_event_ids: set[str] = set()
    for index, raw in enumerate(events_value):
        row = _base._closed_object(
            raw,
            expected=_ADMISSION_EVENT_FIELDS,
            field=f"admission.events[{index}]",
        )
        patient = patient_by_id.get(str(row["patient_id"]))
        event_id = str(row["event_id"])
        if patient is None or event_id not in patient["eligible_event_ids"]:
            raise ValueError("Admission-only event is outside its patient roster")
        if event_id in seen_event_ids:
            raise ValueError("Admission-only event roster contains duplicates")
        seen_event_ids.add(event_id)
        if row["official_split"] != patient["official_split"]:
            raise ValueError("Admission-only event/patient official split drifted")
        if row["source_model_split"] != _MODEL_SPLIT_BY_OFFICIAL[
            str(row["official_split"])
        ]:
            raise ValueError("Admission-only source split is not mechanical")
        for field in (
            "event_record_sha256",
            "crosswalk_record_sha256",
            "processed_window_sha256",
            "preprocess_config_sha256",
        ):
            _base._require_sha256(row[field], field=f"admission.events[{index}].{field}")
        events.append(row)
    event_ids = [str(row["event_id"]) for row in events]
    if event_ids != sorted(event_ids):
        raise ValueError("Admission-only events are not canonically ordered")
    expected_event_ids = sorted(
        event_id for row in patients for event_id in row["eligible_event_ids"]
    )
    if event_ids != expected_event_ids:
        raise ValueError("Admission-only event rows do not close patient rosters")
    count_checks = {
        "admitted_patient_count": len(patient_ids),
        "admitted_event_count": len(event_ids),
    }
    for field, expected in count_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"Admission-only {field} drifted")
    roster_checks = {
        "admitted_patient_ids": patient_ids,
        "admitted_event_ids": event_ids,
    }
    for field, expected in roster_checks.items():
        if receipt[field] != expected:
            raise ValueError(f"Admission-only {field} drifted")
    roster_hash_checks = {
        "admitted_patient_roster_sha256": patient_ids,
        "admitted_event_roster_sha256": event_ids,
    }
    for field, roster in roster_hash_checks.items():
        if receipt[field] != _base._roster_sha256(roster):
            raise ValueError(f"Admission-only {field} drifted")
    return receipt


@dataclass(frozen=True)
class VerifiedMaskedVariableAuxiliaryJoin:
    receipt: Mapping[str, object]
    artifact_sha256: str
    receipt_sha256: str
    admission_artifact_sha256: str
    admission_receipt_sha256: str

    @property
    def admitted_patient_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.receipt["admitted_patient_ids"])

    @property
    def admitted_event_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.receipt["admitted_event_ids"])


@dataclass(frozen=True)
class VerifiedMaskedVariableAuxiliaryAdmission:
    receipt: Mapping[str, object]
    artifact_sha256: str
    receipt_sha256: str


def _publish_receipt(
    receipt: Mapping[str, object], output_directory: str | Path
) -> VerifiedMaskedVariableAuxiliaryJoin:
    receipt_sha = _base._canonical_sha256(receipt)
    artifact = {
        "schema_version": MASKED_VARIABLE_AUXILIARY_JOIN_ARTIFACT_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "receipt_sha256": receipt_sha,
        "receipt": receipt,
    }
    encoded = _base._canonical_json_bytes(artifact)
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Masked-variable join artifact exceeds size limit")
    artifact_sha = _base._bytes_sha256(encoded)
    admission_receipt = _build_admission_receipt(
        receipt,
        source_join_artifact_sha256=artifact_sha,
        source_join_receipt_sha256=receipt_sha,
    )
    admission_receipt_sha = _base._canonical_sha256(admission_receipt)
    admission_artifact = {
        "schema_version": MASKED_VARIABLE_AUXILIARY_ADMISSION_ARTIFACT_SCHEMA,
        "serialization": "canonical_json_utf8_newline_no_pickle",
        "receipt_sha256": admission_receipt_sha,
        "receipt": admission_receipt,
    }
    admission_encoded = _base._canonical_json_bytes(admission_artifact)
    if len(admission_encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Admission-only artifact exceeds size limit")
    admission_artifact_sha = _base._bytes_sha256(admission_encoded)
    output = _base._reject_symlink_components(
        Path(output_directory), field="masked-variable join output"
    )
    if output.name in {"", ".", ".."}:
        raise ValueError("Output requires a concrete directory name")
    if os.path.lexists(output):
        raise FileExistsError("Masked-variable join destination exists")
    parent = _base._reject_symlink_components(output.parent, field="output parent")
    if not parent.is_dir():
        raise FileNotFoundError("Masked-variable join output parent is missing")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    published = False
    try:
        artifact_path = temporary / MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME
        with artifact_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        admission_path = temporary / MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME
        with admission_path.open("xb") as handle:
            handle.write(admission_encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _base._fsync_directory(temporary)
        if os.path.lexists(output):
            raise FileExistsError("Masked-variable join destination exists")
        os.rename(temporary, output)
        published = True
        _base._fsync_directory(parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return VerifiedMaskedVariableAuxiliaryJoin(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=receipt_sha,
        admission_artifact_sha256=admission_artifact_sha,
        admission_receipt_sha256=admission_receipt_sha,
    )


def build_masked_variable_auxiliary_join(
    signal_universe_directory: str | Path,
    target_v2_directory: str | Path,
    target_source_csv: str | Path,
    target_split_csv: str | Path,
    protocol_path: str | Path,
    output_directory: str | Path,
    *,
    expected_signal_universe_artifact_sha256: str,
    expected_target_artifact_sha256: str,
    expected_target_summary_artifact_sha256: str,
    expected_target_readme_artifact_sha256: str,
    expected_target_source_input_sha256: str,
    expected_target_split_input_sha256: str,
    expected_protocol_sha256: str,
) -> VerifiedMaskedVariableAuxiliaryJoin:
    """Strictly load, join, validate, and atomically publish the v17 artifact."""

    if os.path.lexists(output_directory):
        raise FileExistsError("Masked-variable join destination exists")
    protocol_bytes, protocol_sha = _base._read_stable_regular_file(
        protocol_path,
        field="v17 protocol",
        max_bytes=4 * 1024 * 1024,
    )
    del protocol_bytes
    _base._check_expected_sha(
        protocol_sha,
        expected_protocol_sha256,
        field="expected_protocol_sha256",
    )
    signal = load_target_independent_signal_universe(
        signal_universe_directory,
        expected_artifact_sha256=expected_signal_universe_artifact_sha256,
    )
    target = load_verified_deepsoz_target_v2_artifact(
        target_v2_directory,
        target_source_csv,
        target_split_csv,
        expected_target_artifact_sha256=expected_target_artifact_sha256,
        expected_summary_artifact_sha256=(
            expected_target_summary_artifact_sha256
        ),
        expected_readme_artifact_sha256=(
            expected_target_readme_artifact_sha256
        ),
        expected_source_input_sha256=expected_target_source_input_sha256,
        expected_split_input_sha256=expected_target_split_input_sha256,
    )
    receipt = _build_join_receipt(
        signal,
        target,
        protocol_sha256=protocol_sha,
    )
    return _publish_receipt(receipt, output_directory)


def _parse_artifact(encoded: bytes) -> tuple[dict[str, object], dict[str, object]]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    try:
        artifact = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Masked-variable join artifact is not strict JSON") from exc
    artifact = _base._closed_object(
        artifact, expected=_ARTIFACT_FIELDS, field="masked-variable join artifact"
    )
    if _base._canonical_json_bytes(artifact) != encoded:
        raise ValueError("Masked-variable join artifact bytes are not canonical")
    if artifact["schema_version"] != MASKED_VARIABLE_AUXILIARY_JOIN_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported masked-variable join artifact schema")
    if artifact["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Masked-variable join artifact serialization is unsafe")
    receipt = _validate_receipt(artifact["receipt"])
    declared = _base._require_sha256(
        artifact["receipt_sha256"], field="receipt_sha256"
    )
    if declared != _base._canonical_sha256(receipt):
        raise ValueError("Masked-variable join receipt SHA mismatch")
    return artifact, receipt


def _parse_admission_artifact(
    encoded: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    try:
        artifact = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Admission-only artifact is not strict JSON") from exc
    artifact = _base._closed_object(
        artifact, expected=_ARTIFACT_FIELDS, field="admission-only artifact"
    )
    if _base._canonical_json_bytes(artifact) != encoded:
        raise ValueError("Admission-only artifact bytes are not canonical")
    if artifact["schema_version"] != (
        MASKED_VARIABLE_AUXILIARY_ADMISSION_ARTIFACT_SCHEMA
    ):
        raise ValueError("Unsupported admission-only artifact schema")
    if artifact["serialization"] != "canonical_json_utf8_newline_no_pickle":
        raise ValueError("Admission-only artifact serialization is unsafe")
    receipt = _validate_admission_receipt(artifact["receipt"])
    declared = _base._require_sha256(
        artifact["receipt_sha256"], field="admission.receipt_sha256"
    )
    if declared != _base._canonical_sha256(receipt):
        raise ValueError("Admission-only receipt SHA mismatch")
    return artifact, receipt


def load_masked_variable_auxiliary_join(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_admission_artifact_sha256: str,
) -> VerifiedMaskedVariableAuxiliaryJoin:
    """Strictly load one published v17 auxiliary target join."""

    bundle = _base._reject_symlink_components(
        Path(bundle_directory), field="masked-variable join bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Masked-variable join bundle is missing")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    expected_names = {
        MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME,
        MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME,
    }
    if (
        len(entries) != 2
        or {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("Masked-variable join bundle violates closed schema")
    encoded, artifact_sha = _base._read_stable_regular_file(
        bundle / MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME,
        field="masked-variable join artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    _base._check_expected_sha(
        artifact_sha,
        expected_artifact_sha256,
        field="expected_masked_variable_join_artifact_sha256",
    )
    _, receipt = _parse_artifact(encoded)
    admission_encoded, admission_artifact_sha = _base._read_stable_regular_file(
        bundle / MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME,
        field="masked-variable admission-only artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    _base._check_expected_sha(
        admission_artifact_sha,
        expected_admission_artifact_sha256,
        field="expected_masked_variable_admission_artifact_sha256",
    )
    _, admission_receipt = _parse_admission_artifact(admission_encoded)
    expected_admission = _build_admission_receipt(
        receipt,
        source_join_artifact_sha256=artifact_sha,
        source_join_receipt_sha256=_base._canonical_sha256(receipt),
    )
    if admission_receipt != expected_admission:
        raise ValueError("Admission-only artifact is not the exact join projection")
    return VerifiedMaskedVariableAuxiliaryJoin(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=_base._canonical_sha256(receipt),
        admission_artifact_sha256=admission_artifact_sha,
        admission_receipt_sha256=_base._canonical_sha256(admission_receipt),
    )


def load_masked_variable_auxiliary_admission(
    bundle_directory: str | Path,
    *,
    expected_artifact_sha256: str,
) -> VerifiedMaskedVariableAuxiliaryAdmission:
    """Load only target-excluding ``signal_admission.json`` bytes.

    The target-bearing join file is required to exist as a regular bundle
    member but is intentionally not opened.  Cache producers can therefore
    bind the target-conditioned roster without loading direct target values.
    """

    bundle = _base._reject_symlink_components(
        Path(bundle_directory), field="masked-variable admission bundle"
    )
    if not bundle.is_dir():
        raise FileNotFoundError("Masked-variable admission bundle is missing")
    entries = tuple(sorted(bundle.iterdir(), key=lambda path: path.name))
    expected_names = {
        MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME,
        MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME,
    }
    if (
        len(entries) != 2
        or {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("Masked-variable admission bundle violates closed schema")
    encoded, artifact_sha = _base._read_stable_regular_file(
        bundle / MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME,
        field="masked-variable admission-only artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    _base._check_expected_sha(
        artifact_sha,
        expected_artifact_sha256,
        field="expected_masked_variable_admission_artifact_sha256",
    )
    _, receipt = _parse_admission_artifact(encoded)
    return VerifiedMaskedVariableAuxiliaryAdmission(
        receipt=receipt,
        artifact_sha256=artifact_sha,
        receipt_sha256=_base._canonical_sha256(receipt),
    )


__all__ = [
    "MASKED_VARIABLE_AUXILIARY_ADMISSION_ARTIFACT_SCHEMA",
    "MASKED_VARIABLE_AUXILIARY_ADMISSION_FILENAME",
    "MASKED_VARIABLE_AUXILIARY_ADMISSION_POLICY",
    "MASKED_VARIABLE_AUXILIARY_ADMISSION_SCHEMA",
    "MASKED_VARIABLE_AUXILIARY_JOIN_ARTIFACT_SCHEMA",
    "MASKED_VARIABLE_AUXILIARY_JOIN_FILENAME",
    "MASKED_VARIABLE_AUXILIARY_JOIN_POLICY",
    "MASKED_VARIABLE_AUXILIARY_JOIN_SCHEMA",
    "MASKED_VARIABLE_COHORT_STATUS",
    "N_AUX_OUTER_FOLDS",
    "PREREGISTERED_AUXILIARY_PATIENT_COUNT",
    "VerifiedMaskedVariableAuxiliaryAdmission",
    "VerifiedMaskedVariableAuxiliaryJoin",
    "assign_aux_outer_folds",
    "build_masked_variable_auxiliary_join",
    "load_masked_variable_auxiliary_admission",
    "load_masked_variable_auxiliary_join",
]
