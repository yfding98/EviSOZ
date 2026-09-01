"""Complete-record BA-IEG localization bridge with patient-level supervision.

The bridge is intentionally downstream of the future-free typed-unit head and
the event-to-record capped log-mean-exp module.  Its supervised surface contains
only physical-electrode logits.  Bipolar-lead logits remain available for audit
and Findings, but there is no field through which they can receive a DeepSOZ
electrode target or be scattered to either endpoint.

Two navigation arms are kept mutually exclusive:

``A0_conditional_on_oracle_navigation``
    Public TUSZ global seizure intervals may define event navigation/boundary
    support.  Channel involvement annotations are forbidden and the result is
    not detector-aware performance.

``A1_detector_frozen``
    Reserved for event candidates from a separately qualified, frozen detector.
    V1 rejects every A1 roster until a typed detector operating-point receipt
    validator exists; a digest-shaped string and claimed status are not proof.

The complete record roster is explicit.  Records with zero candidates remain in
the complete audit denominator and receipt, as do candidate-bearing records with
no qualified event.  The numerical capped-LME denominator is instead computed
per electrode from the evaluable record mask.  V1 neither imputes localization
logits nor applies a detection-miss penalty for zero/unqualified records.
Exact-container aliases contribute at most once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from src.soz.geometry import STANDARD_19

from .ba_ieg_capped_log_mean_exp_event_bag_v1 import (
    BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID,
    BAIEGCappedLogMeanExpEventBagOutput,
)
from .ba_ieg_training_contract import BA_IEG_C18, BAIEGDeepSOZPositiveSet
from .deepsoz_tusz_identity_binding_v1 import (
    deepsoz_patient_uid_lookup_v1,
    validate_deepsoz_tusz_source_train_identity_binding_v1,
)


BA_IEG_COMPLETE_PATIENT_POSITIVE_SET_BRIDGE_ID_V1: Final[str] = (
    "ba_ieg_complete_record_patient_capped_lme_positive_set_bridge_v1"
)
BA_IEG_NAVIGATION_ARM_A0: Final[str] = "A0_conditional_on_oracle_navigation"
BA_IEG_NAVIGATION_ARM_A1: Final[str] = "A1_detector_frozen"
BA_IEG_NAVIGATION_ARMS: Final[tuple[str, str]] = (
    BA_IEG_NAVIGATION_ARM_A0,
    BA_IEG_NAVIGATION_ARM_A1,
)

BA_IEG_RECORD_HAS_CANDIDATES: Final[str] = "qualified_event_candidates_present"
BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE: Final[str] = (
    "event_candidates_present_none_qualified"
)
BA_IEG_RECORD_ZERO_CANDIDATE: Final[str] = "zero_detector_candidate"
BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED: Final[str] = "exact_signal_alias_excluded"
BA_IEG_RECORD_CANDIDATE_STATUSES: Final[tuple[str, str, str, str]] = (
    BA_IEG_RECORD_HAS_CANDIDATES,
    BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE,
    BA_IEG_RECORD_ZERO_CANDIDATE,
    BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED,
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")


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


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or set(text) - _SHA256_ALPHABET:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class BAIEGPhysicalRecordEvidenceBatchV1:
    """Physical-electrode-only adapter for the event-bag output."""

    source_input_batch_sha256: str
    source_implementation_id: str
    recording_ids: tuple[str, ...]
    physical_electrode_record_logits: torch.Tensor
    physical_electrode_mask: torch.Tensor
    unique_occurrence_count: torch.Tensor
    qualified_unique_occurrence_count: torch.Tensor
    uncertain_unique_occurrence_count: torch.Tensor
    not_evaluable_unique_occurrence_count: torch.Tensor
    copied_event_count: torch.Tensor
    electrode_ids: tuple[str, ...] = STANDARD_19
    evidence_semantics: str = (
        "physical_electrode_record_onset_candidate_logits_no_bipolar_endpoint_projection"
    )

    def __post_init__(self) -> None:
        _sha256(self.source_input_batch_sha256, "source input batch receipt")
        if self.source_implementation_id != BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID:
            raise ValueError("physical record evidence must come from the frozen event bag")
        records = tuple(_identifier(item, "recording_id") for item in self.recording_ids)
        if len(records) != len(set(records)):
            raise ValueError("physical record evidence repeats recording_id")
        if tuple(self.electrode_ids) != STANDARD_19:
            raise ValueError("physical record evidence must use standard-19")
        expected = (len(records), len(STANDARD_19))
        logits = self.physical_electrode_record_logits
        mask = self.physical_electrode_mask
        if (
            not isinstance(logits, torch.Tensor)
            or tuple(logits.shape) != expected
            or not logits.is_floating_point()
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("physical record logits must be finite floating [R,19]")
        if (
            not isinstance(mask, torch.Tensor)
            or tuple(mask.shape) != expected
            or mask.dtype != torch.bool
            or mask.device != logits.device
        ):
            raise ValueError("physical opportunity mask must be bool [R,19]")
        for name, value in (
            ("unique_occurrence_count", self.unique_occurrence_count),
            (
                "qualified_unique_occurrence_count",
                self.qualified_unique_occurrence_count,
            ),
            (
                "uncertain_unique_occurrence_count",
                self.uncertain_unique_occurrence_count,
            ),
            (
                "not_evaluable_unique_occurrence_count",
                self.not_evaluable_unique_occurrence_count,
            ),
            ("copied_event_count", self.copied_event_count),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (len(records),)
                or value.dtype != torch.long
                or value.device != logits.device
            ):
                raise ValueError(f"{name} must be long [R] on the logits device")
        if (
            torch.any(self.unique_occurrence_count <= 0)
            or torch.any(self.qualified_unique_occurrence_count < 0)
            or torch.any(self.uncertain_unique_occurrence_count < 0)
            or torch.any(self.not_evaluable_unique_occurrence_count < 0)
            or torch.any(self.copied_event_count < 0)
        ):
            raise ValueError("materialized record evidence has invalid occurrence counts")
        if not torch.equal(
            self.unique_occurrence_count,
            self.qualified_unique_occurrence_count
            + self.uncertain_unique_occurrence_count
            + self.not_evaluable_unique_occurrence_count,
        ):
            raise ValueError("record aggregation-status counts do not sum to total")
        no_qualified = self.qualified_unique_occurrence_count == 0
        if torch.any(self.physical_electrode_mask[no_qualified]):
            raise ValueError("record with no qualified event exposed positive evidence")
        object.__setattr__(self, "recording_ids", records)

    @classmethod
    def from_event_bag_output(
        cls, output: BAIEGCappedLogMeanExpEventBagOutput
    ) -> "BAIEGPhysicalRecordEvidenceBatchV1":
        if not isinstance(output, BAIEGCappedLogMeanExpEventBagOutput):
            raise TypeError("physical record adapter requires event-bag output")
        return cls(
            source_input_batch_sha256=output.source_input_batch_sha256,
            source_implementation_id=output.implementation_id,
            recording_ids=output.recording_ids,
            physical_electrode_record_logits=(
                output.physical_electrode_record_logits
            ),
            physical_electrode_mask=output.physical_electrode_mask,
            unique_occurrence_count=output.unique_occurrence_count,
            qualified_unique_occurrence_count=(
                output.qualified_unique_occurrence_count
            ),
            uncertain_unique_occurrence_count=(
                output.uncertain_unique_occurrence_count
            ),
            not_evaluable_unique_occurrence_count=(
                output.not_evaluable_unique_occurrence_count
            ),
            copied_event_count=output.copied_event_count,
        )


@dataclass(frozen=True)
class BAIEGCompleteRecordRosterEntryV1:
    """One source record in the frozen, target-independent patient roster."""

    patient_uid: str
    source_recording_id: str
    model_recording_id: str
    source_container_sha256: str
    exact_container_equivalence_id: str
    model_source_binding_sha256: str
    candidate_status: str
    expected_unique_occurrence_count: int
    expected_qualified_unique_occurrence_count: int
    model_split: str = "source_train"

    def __post_init__(self) -> None:
        _identifier(self.patient_uid, "patient_uid")
        _identifier(self.source_recording_id, "source_recording_id")
        _identifier(self.model_recording_id, "model_recording_id")
        container = _sha256(self.source_container_sha256, "source container")
        _sha256(self.model_source_binding_sha256, "model source binding")
        expected_equivalence = "TUSZ-EDF-CONTAINER-" + container
        if self.exact_container_equivalence_id != expected_equivalence:
            raise ValueError("record equivalence ID must derive from exact container")
        if self.model_split != "source_train":
            raise ValueError("v1 positive-set bridge is source_train-only")
        if self.candidate_status not in BA_IEG_RECORD_CANDIDATE_STATUSES:
            raise ValueError("complete record has unsupported candidate status")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.expected_unique_occurrence_count,
                self.expected_qualified_unique_occurrence_count,
            )
        ):
            raise TypeError("expected occurrence counts must be integers")
        if self.candidate_status == BA_IEG_RECORD_HAS_CANDIDATES:
            if not (
                self.expected_unique_occurrence_count
                >= self.expected_qualified_unique_occurrence_count
                >= 1
            ):
                raise ValueError("qualified record needs valid total/qualified counts")
        elif self.candidate_status == BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE:
            if (
                self.expected_unique_occurrence_count < 1
                or self.expected_qualified_unique_occurrence_count != 0
            ):
                raise ValueError("unqualified record needs candidates but no qualified event")
        elif (
            self.expected_unique_occurrence_count != 0
            or self.expected_qualified_unique_occurrence_count != 0
        ):
            raise ValueError("zero/alias record cannot claim candidate occurrences")


@dataclass(frozen=True)
class BAIEGCompletePatientRecordRosterV1:
    """Complete source-train audit denominator for one optimizer cohort."""

    identity_binding_sha256: str
    candidate_roster_receipt_sha256: str
    navigation_arm: str
    records: tuple[BAIEGCompleteRecordRosterEntryV1, ...]
    oracle_navigation_receipt_sha256: str | None = None
    detector_operating_point_receipt_sha256: str | None = None
    detector_operating_point_qualification_status: str | None = None
    oracle_event_intervals_used_for_navigation: bool = False
    oracle_channel_annotations_used: bool = False
    deepsoz_targets_used_for_candidate_selection: bool = False
    private_labels_used: bool = False
    model_split: str = "source_train"
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        identity = _sha256(self.identity_binding_sha256, "identity binding")
        candidate = _sha256(
            self.candidate_roster_receipt_sha256, "candidate roster receipt"
        )
        if self.navigation_arm not in BA_IEG_NAVIGATION_ARMS:
            raise ValueError("complete record roster has an unsupported navigation arm")
        if self.model_split != "source_train":
            raise ValueError("complete patient positive-set roster is source_train-only")
        if not self.records or not all(
            isinstance(item, BAIEGCompleteRecordRosterEntryV1)
            for item in self.records
        ):
            raise TypeError("complete patient roster requires typed record rows")
        if any(item.model_split != self.model_split for item in self.records):
            raise ValueError("complete patient roster crosses model splits")
        source_ids = [item.source_recording_id for item in self.records]
        model_ids = [item.model_recording_id for item in self.records]
        if len(source_ids) != len(set(source_ids)) or len(model_ids) != len(
            set(model_ids)
        ):
            raise ValueError("complete patient roster repeats a recording identity")
        if (
            self.oracle_channel_annotations_used is not False
            or self.deepsoz_targets_used_for_candidate_selection is not False
            or self.private_labels_used is not False
        ):
            raise ValueError("channel/DeepSOZ/private labels cannot select event candidates")

        by_equivalence: dict[str, list[BAIEGCompleteRecordRosterEntryV1]] = {}
        for item in self.records:
            by_equivalence.setdefault(item.exact_container_equivalence_id, []).append(item)
        for rows in by_equivalence.values():
            patients = {item.patient_uid for item in rows}
            representatives = [
                item
                for item in rows
                if item.candidate_status != BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED
            ]
            if len(patients) != 1 or len(representatives) != 1:
                raise ValueError(
                    "one exact-container group needs one same-patient representative"
                )
            if any(
                item.candidate_status == BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED
                for item in rows
            ) and len(rows) < 2:
                raise ValueError("exact-signal alias exclusion needs another group member")

        if self.navigation_arm == BA_IEG_NAVIGATION_ARM_A0:
            _sha256(self.oracle_navigation_receipt_sha256, "A0 navigation receipt")
            if (
                self.detector_operating_point_receipt_sha256 is not None
                or self.detector_operating_point_qualification_status is not None
                or self.oracle_event_intervals_used_for_navigation is not True
            ):
                raise ValueError("A0 must be conditional on oracle interval navigation only")
        else:
            raise ValueError(
                "A1 is fail-closed until a typed detector operating-point "
                "receipt validator is implemented"
            )

        payload = {
            "schema": "ba_ieg_complete_patient_record_roster_v1",
            "identity_binding_sha256": identity,
            "candidate_roster_receipt_sha256": candidate,
            "navigation_arm": self.navigation_arm,
            "model_split": self.model_split,
            "oracle_navigation_receipt_sha256": self.oracle_navigation_receipt_sha256,
            "detector_operating_point_receipt_sha256": (
                self.detector_operating_point_receipt_sha256
            ),
            "detector_operating_point_qualification_status": (
                self.detector_operating_point_qualification_status
            ),
            "oracle_event_intervals_used_for_navigation": (
                self.oracle_event_intervals_used_for_navigation
            ),
            "oracle_channel_annotations_used": False,
            "deepsoz_targets_used_for_candidate_selection": False,
            "private_labels_used": False,
            "records": [
                {
                    "patient_uid": item.patient_uid,
                    "source_recording_id": item.source_recording_id,
                    "model_recording_id": item.model_recording_id,
                    "source_container_sha256": item.source_container_sha256,
                    "exact_container_equivalence_id": (
                        item.exact_container_equivalence_id
                    ),
                    "model_source_binding_sha256": (
                        item.model_source_binding_sha256
                    ),
                    "candidate_status": item.candidate_status,
                    "expected_unique_occurrence_count": (
                        item.expected_unique_occurrence_count
                    ),
                    "expected_qualified_unique_occurrence_count": (
                        item.expected_qualified_unique_occurrence_count
                    ),
                }
                for item in self.records
            ],
        }
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))


@dataclass(frozen=True)
class BAIEGCompletePatientAggregationOutputV1:
    implementation_id: str
    identity_binding_sha256: str
    complete_roster_receipt_sha256: str
    navigation_arm: str
    patient_uids: tuple[str, ...]
    exact_container_equivalence_ids: tuple[str, ...]
    representative_recording_ids: tuple[str, ...]
    record_patient_index: torch.Tensor
    complete_record_logits: torch.Tensor
    complete_record_mask: torch.Tensor
    record_candidate_present_mask: torch.Tensor
    record_qualified_candidate_present_mask: torch.Tensor
    zero_candidate_record_mask: torch.Tensor
    unique_occurrence_count: torch.Tensor
    qualified_unique_occurrence_count: torch.Tensor
    uncertain_unique_occurrence_count: torch.Tensor
    not_evaluable_unique_occurrence_count: torch.Tensor
    copied_event_count: torch.Tensor
    patient_logits: torch.Tensor
    patient_electrode_mask: torch.Tensor
    patient_unique_record_count: torch.Tensor
    patient_candidate_record_count: torch.Tensor
    patient_zero_candidate_record_count: torch.Tensor
    patient_evaluable_record_count: torch.Tensor
    output_semantics: str = (
        "patient_weak_supervision_logits_with_complete_record_audit_denominator_"
        "and_per_electrode_evaluable_record_numeric_denominator_no_detection_"
        "miss_imputation_or_penalty"
    )

    @property
    def patient_has_localization_evidence(self) -> torch.Tensor:
        return self.patient_electrode_mask.any(dim=1)


class BAIEGCompletePatientCappedLogMeanExpV1(nn.Module):
    """Capped-LME over each electrode's evaluable exact-unique record mask.

    Complete zero/unqualified records remain visible in audit counts, but V1
    supplies neither an imputed numeric logit nor a detection-miss penalty for
    them.
    """

    implementation_id: Final[str] = BA_IEG_COMPLETE_PATIENT_POSITIVE_SET_BRIDGE_ID_V1

    def __init__(
        self, *, temperature: float = 1.0, symmetric_record_logit_clip: float = 12.0
    ) -> None:
        super().__init__()
        if temperature != 1.0 or symmetric_record_logit_clip != 12.0:
            raise ValueError("v1 patient bag is frozen at tau=1 and clip=12")
        self.temperature = float(temperature)
        self.symmetric_record_logit_clip = float(symmetric_record_logit_clip)

    def forward(
        self,
        record_evidence: BAIEGPhysicalRecordEvidenceBatchV1 | None,
        roster: BAIEGCompletePatientRecordRosterV1,
        *,
        empty_dtype: torch.dtype = torch.float32,
        empty_device: torch.device | str = "cpu",
    ) -> BAIEGCompletePatientAggregationOutputV1:
        if not isinstance(roster, BAIEGCompletePatientRecordRosterV1):
            raise TypeError("patient aggregation requires a complete typed roster")
        if record_evidence is not None and not isinstance(
            record_evidence, BAIEGPhysicalRecordEvidenceBatchV1
        ):
            raise TypeError("patient aggregation accepts physical record evidence only")
        if record_evidence is None:
            dtype = empty_dtype
            device = torch.device(empty_device)
            evidence_by_record: dict[str, int] = {}
        else:
            dtype = record_evidence.physical_electrode_record_logits.dtype
            device = record_evidence.physical_electrode_record_logits.device
            evidence_by_record = {
                item: index for index, item in enumerate(record_evidence.recording_ids)
            }

        expected_candidate_records = {
            item.model_recording_id
            for item in roster.records
            if item.candidate_status
            in {
                BA_IEG_RECORD_HAS_CANDIDATES,
                BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE,
            }
        }
        if set(evidence_by_record) != expected_candidate_records:
            raise ValueError(
                "record evidence does not equal the frozen candidate-bearing roster"
            )

        groups: dict[str, list[BAIEGCompleteRecordRosterEntryV1]] = {}
        for item in roster.records:
            groups.setdefault(item.exact_container_equivalence_id, []).append(item)
        group_rows: list[
            tuple[str, BAIEGCompleteRecordRosterEntryV1, tuple[BAIEGCompleteRecordRosterEntryV1, ...]]
        ] = []
        for equivalence_id in sorted(groups):
            members = tuple(groups[equivalence_id])
            representatives = [
                item
                for item in members
                if item.candidate_status != BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED
            ]
            if len(representatives) != 1:
                raise RuntimeError("validated roster lost its unique representative")
            group_rows.append((equivalence_id, representatives[0], members))

        patients = tuple(sorted({item.patient_uid for item in roster.records}))
        patient_index = {item: index for index, item in enumerate(patients)}
        record_count = len(group_rows)
        record_logits = torch.zeros(
            (record_count, len(STANDARD_19)), dtype=dtype, device=device
        )
        record_mask = torch.zeros_like(record_logits, dtype=torch.bool)
        candidate_present = torch.zeros(record_count, dtype=torch.bool, device=device)
        qualified_candidate_present = torch.zeros(
            record_count, dtype=torch.bool, device=device
        )
        zero_candidate = torch.zeros(record_count, dtype=torch.bool, device=device)
        unique_counts = torch.zeros(record_count, dtype=torch.long, device=device)
        qualified_counts = torch.zeros_like(unique_counts)
        uncertain_counts = torch.zeros_like(unique_counts)
        not_evaluable_counts = torch.zeros_like(unique_counts)
        copied_counts = torch.zeros(record_count, dtype=torch.long, device=device)
        record_patient_index = torch.tensor(
            [patient_index[representative.patient_uid] for _, representative, _ in group_rows],
            dtype=torch.long,
            device=device,
        )

        for group_index, (_, representative, members) in enumerate(group_rows):
            alias_count = len(members) - 1
            if representative.candidate_status in {
                BA_IEG_RECORD_HAS_CANDIDATES,
                BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE,
            }:
                evidence_index = evidence_by_record[representative.model_recording_id]
                observed_count = int(record_evidence.unique_occurrence_count[evidence_index])
                observed_qualified = int(
                    record_evidence.qualified_unique_occurrence_count[evidence_index]
                )
                if (
                    observed_count != representative.expected_unique_occurrence_count
                    or observed_qualified
                    != representative.expected_qualified_unique_occurrence_count
                ):
                    raise ValueError("record evidence lost or duplicated a frozen occurrence")
                record_logits[group_index] = (
                    record_evidence.physical_electrode_record_logits[evidence_index]
                )
                record_mask[group_index] = record_evidence.physical_electrode_mask[
                    evidence_index
                ]
                unique_counts[group_index] = observed_count
                qualified_counts[group_index] = observed_qualified
                uncertain_counts[group_index] = (
                    record_evidence.uncertain_unique_occurrence_count[evidence_index]
                )
                not_evaluable_counts[group_index] = (
                    record_evidence.not_evaluable_unique_occurrence_count[evidence_index]
                )
                copied_counts[group_index] = (
                    record_evidence.copied_event_count[evidence_index] + alias_count
                )
                candidate_present[group_index] = True
                qualified_candidate_present[group_index] = observed_qualified > 0
            elif representative.candidate_status == BA_IEG_RECORD_ZERO_CANDIDATE:
                zero_candidate[group_index] = True
                copied_counts[group_index] = alias_count
            else:
                raise RuntimeError("exact alias cannot be a group representative")

        patient_logits = torch.zeros(
            (len(patients), len(STANDARD_19)), dtype=dtype, device=device
        )
        patient_mask = torch.zeros_like(patient_logits, dtype=torch.bool)
        patient_unique_counts = torch.zeros(len(patients), dtype=torch.long, device=device)
        patient_candidate_counts = torch.zeros_like(patient_unique_counts)
        patient_zero_counts = torch.zeros_like(patient_unique_counts)
        patient_evaluable_counts = torch.zeros_like(patient_unique_counts)
        clipped = record_logits.clamp(
            min=-self.symmetric_record_logit_clip,
            max=self.symmetric_record_logit_clip,
        )
        for patient_position in range(len(patients)):
            selected_records = record_patient_index == patient_position
            patient_unique_counts[patient_position] = selected_records.sum()
            patient_candidate_counts[patient_position] = (
                selected_records & candidate_present
            ).sum()
            patient_zero_counts[patient_position] = (
                selected_records & zero_candidate
            ).sum()
            patient_evaluable_counts[patient_position] = (
                selected_records & record_mask.any(dim=1)
            ).sum()
            for electrode_index in range(len(STANDARD_19)):
                selected = selected_records & record_mask[:, electrode_index]
                count = int(selected.sum())
                if count == 0:
                    continue
                values = clipped[selected, electrode_index] / self.temperature
                patient_logits[patient_position, electrode_index] = self.temperature * (
                    torch.logsumexp(values, dim=0) - math.log(count)
                )
                patient_mask[patient_position, electrode_index] = True

        return BAIEGCompletePatientAggregationOutputV1(
            implementation_id=self.implementation_id,
            identity_binding_sha256=roster.identity_binding_sha256,
            complete_roster_receipt_sha256=roster.receipt_sha256,
            navigation_arm=roster.navigation_arm,
            patient_uids=patients,
            exact_container_equivalence_ids=tuple(item[0] for item in group_rows),
            representative_recording_ids=tuple(
                item[1].model_recording_id for item in group_rows
            ),
            record_patient_index=record_patient_index,
            complete_record_logits=record_logits,
            complete_record_mask=record_mask,
            record_candidate_present_mask=candidate_present,
            record_qualified_candidate_present_mask=(
                qualified_candidate_present
            ),
            zero_candidate_record_mask=zero_candidate,
            unique_occurrence_count=unique_counts,
            qualified_unique_occurrence_count=qualified_counts,
            uncertain_unique_occurrence_count=uncertain_counts,
            not_evaluable_unique_occurrence_count=not_evaluable_counts,
            copied_event_count=copied_counts,
            patient_logits=patient_logits,
            patient_electrode_mask=patient_mask,
            patient_unique_record_count=patient_unique_counts,
            patient_candidate_record_count=patient_candidate_counts,
            patient_zero_candidate_record_count=patient_zero_counts,
            patient_evaluable_record_count=patient_evaluable_counts,
        )


@dataclass(frozen=True)
class BAIEGBoundDeepSOZPositiveSetV1:
    """DeepSOZ positive set after an explicit numeric-ID identity join."""

    deepsoz_patient_id: str
    patient_uid: str
    positive_electrode_ids: tuple[str, ...]
    candidate_electrode_ids: tuple[str, ...]
    source_reference_sha256: str
    source_target_receipt_sha256: str
    identity_binding_sha256: str
    model_split: str = "source_train"
    label_semantics: str = "patient_level_c18_positive_set_unlisted_unknown"
    bipolar_supervision_authorized: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        deepsoz_id = _identifier(self.deepsoz_patient_id, "deepsoz_patient_id")
        if not deepsoz_id.isdigit() or str(int(deepsoz_id)) != deepsoz_id:
            raise ValueError("bound DeepSOZ ID must use canonical decimal form")
        _identifier(self.patient_uid, "patient_uid")
        positives = tuple(self.positive_electrode_ids)
        if (
            not positives
            or len(positives) != len(set(positives))
            or not set(positives).issubset(BA_IEG_C18)
            or tuple(sorted(positives, key=BA_IEG_C18.index)) != positives
        ):
            raise ValueError("bound DeepSOZ positives must be a canonical C18 subset")
        if tuple(self.candidate_electrode_ids) != BA_IEG_C18:
            raise ValueError("bound DeepSOZ candidate denominator must be C18")
        _sha256(self.source_reference_sha256, "DeepSOZ source reference")
        _sha256(self.source_target_receipt_sha256, "DeepSOZ target receipt")
        _sha256(self.identity_binding_sha256, "identity binding")
        if (
            self.model_split != "source_train"
            or self.label_semantics
            != "patient_level_c18_positive_set_unlisted_unknown"
            or self.bipolar_supervision_authorized is not False
        ):
            raise ValueError("bound DeepSOZ supervision semantics drifted")
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema": "ba_ieg_bound_deepsoz_positive_set_v1",
                    "deepsoz_patient_id": deepsoz_id,
                    "patient_uid": self.patient_uid,
                    "positive_electrode_ids": list(positives),
                    "candidate_electrode_ids": list(self.candidate_electrode_ids),
                    "source_reference_sha256": self.source_reference_sha256,
                    "source_target_receipt_sha256": self.source_target_receipt_sha256,
                    "identity_binding_sha256": self.identity_binding_sha256,
                    "model_split": self.model_split,
                    "label_semantics": self.label_semantics,
                    "bipolar_supervision_authorized": False,
                }
            ),
        )


def bind_deepsoz_positive_sets_to_tusz_identity_v1(
    targets: Sequence[BAIEGDeepSOZPositiveSet],
    identity_binding: Mapping[str, Any],
) -> tuple[BAIEGBoundDeepSOZPositiveSetV1, ...]:
    """Perform an explicit, receipt-bound ID join; never silently override IDs."""

    validate_deepsoz_tusz_source_train_identity_binding_v1(identity_binding)
    lookup = deepsoz_patient_uid_lookup_v1(identity_binding)
    receipt = str(identity_binding["receipt_sha256"])
    if not targets or not all(isinstance(item, BAIEGDeepSOZPositiveSet) for item in targets):
        raise TypeError("identity binding requires typed DeepSOZ positive sets")
    rows: list[BAIEGBoundDeepSOZPositiveSetV1] = []
    seen: set[str] = set()
    for target in targets:
        deepsoz_id = target.patient_uid
        if (
            not deepsoz_id.isdigit()
            or str(int(deepsoz_id)) != deepsoz_id
            or deepsoz_id not in lookup
        ):
            raise ValueError("DeepSOZ numeric patient ID is absent from identity binding")
        if deepsoz_id in seen:
            raise ValueError("DeepSOZ identity join repeats a patient")
        seen.add(deepsoz_id)
        if target.model_split != "source_train":
            raise ValueError("positive-set identity binding accepts source_train only")
        rows.append(
            BAIEGBoundDeepSOZPositiveSetV1(
                deepsoz_patient_id=deepsoz_id,
                patient_uid=lookup[deepsoz_id],
                positive_electrode_ids=target.positive_electrode_ids,
                candidate_electrode_ids=target.candidate_electrode_ids,
                source_reference_sha256=target.source_reference_sha256,
                source_target_receipt_sha256=target.receipt_sha256,
                identity_binding_sha256=receipt,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.patient_uid))


@dataclass(frozen=True)
class BAIEGCompletePatientPositiveSetLossOutputV1:
    total_loss: torch.Tensor
    patient_loss: torch.Tensor
    patient_loss_mask: torch.Tensor
    patient_status: tuple[str, ...]
    patient_uids: tuple[str, ...]
    evaluable_patient_count: int
    complete_patient_count: int
    optimizer_step_allowed: bool
    identity_binding_sha256: str
    complete_roster_receipt_sha256: str
    navigation_arm: str
    target_applied_once_per_patient: bool = True
    bipolar_supervision_used: bool = False


def complete_patient_positive_set_mass_loss_v1(
    aggregation: BAIEGCompletePatientAggregationOutputV1,
    targets: Sequence[BAIEGBoundDeepSOZPositiveSetV1],
) -> BAIEGCompletePatientPositiveSetLossOutputV1:
    """Apply one partial-label C18 loss per complete, evaluable patient bag."""

    if not isinstance(aggregation, BAIEGCompletePatientAggregationOutputV1):
        raise TypeError("positive-set loss requires complete patient aggregation")
    if not targets or not all(
        isinstance(item, BAIEGBoundDeepSOZPositiveSetV1) for item in targets
    ):
        raise TypeError("positive-set loss requires explicitly bound targets")
    by_patient = {item.patient_uid: item for item in targets}
    if len(by_patient) != len(targets) or set(by_patient) != set(
        aggregation.patient_uids
    ):
        raise ValueError("positive-set targets must equal the complete patient roster")
    if any(
        item.identity_binding_sha256 != aggregation.identity_binding_sha256
        for item in targets
    ):
        raise ValueError("positive-set target identity receipt disagrees with model roster")

    logits = aggregation.patient_logits
    opportunity = aggregation.patient_electrode_mask
    patient_loss = torch.zeros(len(aggregation.patient_uids), dtype=logits.dtype, device=logits.device)
    loss_mask = torch.zeros(len(aggregation.patient_uids), dtype=torch.bool, device=logits.device)
    statuses: list[str] = []
    rows: list[torch.Tensor] = []
    candidate_set = set(BA_IEG_C18)
    for patient_index, patient_uid in enumerate(aggregation.patient_uids):
        target = by_patient[patient_uid]
        observed_candidates = torch.tensor(
            [item in candidate_set for item in STANDARD_19],
            dtype=torch.bool,
            device=logits.device,
        ) & opportunity[patient_index]
        positive_set = set(target.positive_electrode_ids)
        observed_positives = torch.tensor(
            [item in positive_set for item in STANDARD_19],
            dtype=torch.bool,
            device=logits.device,
        ) & opportunity[patient_index]
        if not bool(opportunity[patient_index].any()):
            statuses.append("not_evaluable_zero_candidate_or_no_physical_opportunity")
            continue
        if not bool(observed_candidates.any()):
            statuses.append("not_evaluable_no_c18_candidate_opportunity")
            continue
        if not bool(observed_positives.any()):
            statuses.append("not_evaluable_no_observed_positive_set_member")
            continue
        row = torch.logsumexp(
            logits[patient_index, observed_candidates], dim=0
        ) - torch.logsumexp(logits[patient_index, observed_positives], dim=0)
        patient_loss[patient_index] = row
        loss_mask[patient_index] = True
        rows.append(row)
        statuses.append("evaluable_patient_positive_set_loss")
    if rows:
        total = torch.stack(rows).mean()
    else:
        # Typed zero preserves a finite receipt; the runner must skip the
        # optimizer step when ``optimizer_step_allowed`` is false.
        total = logits.sum() * 0.0
    evaluable = len(rows)
    return BAIEGCompletePatientPositiveSetLossOutputV1(
        total_loss=total,
        patient_loss=patient_loss,
        patient_loss_mask=loss_mask,
        patient_status=tuple(statuses),
        patient_uids=aggregation.patient_uids,
        evaluable_patient_count=evaluable,
        complete_patient_count=len(aggregation.patient_uids),
        optimizer_step_allowed=evaluable > 0,
        identity_binding_sha256=aggregation.identity_binding_sha256,
        complete_roster_receipt_sha256=(
            aggregation.complete_roster_receipt_sha256
        ),
        navigation_arm=aggregation.navigation_arm,
    )


__all__ = [
    "BA_IEG_COMPLETE_PATIENT_POSITIVE_SET_BRIDGE_ID_V1",
    "BA_IEG_NAVIGATION_ARM_A0",
    "BA_IEG_NAVIGATION_ARM_A1",
    "BA_IEG_NAVIGATION_ARMS",
    "BA_IEG_RECORD_CANDIDATE_STATUSES",
    "BA_IEG_RECORD_EXACT_ALIAS_EXCLUDED",
    "BA_IEG_RECORD_HAS_CANDIDATES",
    "BA_IEG_RECORD_NO_QUALIFIED_CANDIDATE",
    "BA_IEG_RECORD_ZERO_CANDIDATE",
    "BAIEGBoundDeepSOZPositiveSetV1",
    "BAIEGCompletePatientAggregationOutputV1",
    "BAIEGCompletePatientCappedLogMeanExpV1",
    "BAIEGCompletePatientPositiveSetLossOutputV1",
    "BAIEGCompletePatientRecordRosterV1",
    "BAIEGCompleteRecordRosterEntryV1",
    "BAIEGPhysicalRecordEvidenceBatchV1",
    "bind_deepsoz_positive_sets_to_tusz_identity_v1",
    "complete_patient_positive_set_mass_loss_v1",
]
