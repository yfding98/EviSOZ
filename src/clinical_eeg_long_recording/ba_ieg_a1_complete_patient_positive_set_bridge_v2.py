"""A1-only complete-patient aggregation for the additive BA-IEG v2 trainer.

The frozen v1 bridge deliberately rejects A1 because no typed detector receipt
validator existed when it was written.  This module does not relax that class.
It accepts a validated prediction-first acquisition/support ledger, retains
zero-candidate and failed records in the complete patient denominator, and
provides a separately named A1 roster/aggregator.

At present only a content-bound synthetic software-fixture authority can be
constructed here.  The real post-freeze reference-join/training authority is
intentionally unimplemented and therefore fails closed.
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

from .ba_ieg_complete_patient_positive_set_bridge_v1 import (
    BAIEGCompletePatientAggregationOutputV1,
    BAIEGPhysicalRecordEvidenceBatchV1,
)
from .ba_ieg_g0_a1_acquisition_support_lineage_v1 import (
    validate_ba_ieg_g0_a1_acquisition_support_lineage_v1,
)


BA_IEG_A1_COMPLETE_PATIENT_BRIDGE_ID_V2: Final[str] = (
    "ba_ieg_a1_complete_patient_capped_lme_positive_set_bridge_v2"
)
BA_IEG_A1_SYNTHETIC_TRAINING_AUTHORITY_SCHEMA_V2: Final[str] = (
    "ba_ieg_a1_synthetic_software_fixture_training_authority_v2"
)
BA_IEG_A1_RECORD_STATUSES_V2: Final[tuple[str, ...]] = (
    "qualified_event_candidates_present",
    "event_candidates_present_none_qualified",
    "zero_detector_candidate",
    "detector_partial_coverage_not_evaluable",
    "detector_technical_failure_not_evaluable",
    "exact_signal_alias_excluded",
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")


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


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True)
class BAIEGA1SyntheticTrainingAuthorityV2:
    """Explicitly non-promotable authority for strict disk software tests."""

    fixture_id: str
    prediction_roster_receipt_sha256: str
    acquisition_support_lineage_receipt_sha256: str
    stable_origin_registry_receipt_sha256: str
    target_independent_candidate_roster_receipt_sha256: str
    schema_version: str = BA_IEG_A1_SYNTHETIC_TRAINING_AUTHORITY_SCHEMA_V2
    real_training_authorized: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != BA_IEG_A1_SYNTHETIC_TRAINING_AUTHORITY_SCHEMA_V2:
            raise ValueError("A1 synthetic training-authority schema drifted")
        _identifier(self.fixture_id, "synthetic fixture ID")
        for name in (
            "prediction_roster_receipt_sha256",
            "acquisition_support_lineage_receipt_sha256",
            "stable_origin_registry_receipt_sha256",
            "target_independent_candidate_roster_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.real_training_authorized is not False:
            raise ValueError("synthetic A1 fixture cannot authorize real training")
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema_version": self.schema_version,
                    "fixture_id": self.fixture_id,
                    "prediction_roster_receipt_sha256": (
                        self.prediction_roster_receipt_sha256
                    ),
                    "acquisition_support_lineage_receipt_sha256": (
                        self.acquisition_support_lineage_receipt_sha256
                    ),
                    "stable_origin_registry_receipt_sha256": (
                        self.stable_origin_registry_receipt_sha256
                    ),
                    "target_independent_candidate_roster_receipt_sha256": (
                        self.target_independent_candidate_roster_receipt_sha256
                    ),
                    "real_training_authorized": False,
                    "software_fixture_only": True,
                }
            ),
        )


def require_real_a1_postfreeze_training_authority_v2(
    payload: Mapping[str, Any] | None,
) -> None:
    """Fail closed until the real cross-registry authority is implemented."""

    del payload
    raise ValueError(
        "real A1 training is fail-closed: immutable post-freeze target-join "
        "authority/registry validator is not implemented"
    )


@dataclass(frozen=True)
class BAIEGA1CompleteRecordRosterEntryV2:
    patient_uid: str
    recording_id: str
    source_container_sha256: str
    exact_container_equivalence_id: str
    candidate_status: str
    expected_unique_occurrence_count: int
    expected_qualified_unique_occurrence_count: int
    model_split: str = "source_train"

    def __post_init__(self) -> None:
        _identifier(self.patient_uid, "patient UID")
        _identifier(self.recording_id, "recording ID")
        container = _sha256(self.source_container_sha256, "source container")
        if self.exact_container_equivalence_id != "TUSZ-EDF-CONTAINER-" + container:
            raise ValueError("A1 exact-container equivalence does not replay")
        if self.model_split != "source_train":
            raise ValueError("A1 complete-patient roster is source_train-only")
        if self.candidate_status not in BA_IEG_A1_RECORD_STATUSES_V2:
            raise ValueError("A1 complete record status is unsupported")
        total = self.expected_unique_occurrence_count
        qualified = self.expected_qualified_unique_occurrence_count
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (total, qualified)):
            raise TypeError("A1 occurrence counts must be integers")
        if self.candidate_status == "qualified_event_candidates_present":
            if not total >= qualified >= 1:
                raise ValueError("qualified A1 record has invalid occurrence counts")
        elif self.candidate_status == "event_candidates_present_none_qualified":
            if total < 1 or qualified != 0:
                raise ValueError("unqualified A1 record needs candidates and no qualified event")
        elif total != 0 or qualified != 0:
            raise ValueError("zero/failure/alias A1 record cannot claim occurrences")


@dataclass(frozen=True)
class BAIEGA1CompletePatientRecordRosterV2:
    identity_binding_sha256: str
    prediction_roster_id: str
    prediction_roster_receipt_sha256: str
    acquisition_support_lineage_receipt_sha256: str
    stable_origin_registry_receipt_sha256: str
    target_independent_candidate_roster_receipt_sha256: str
    synthetic_training_authority_receipt_sha256: str
    records: tuple[BAIEGA1CompleteRecordRosterEntryV2, ...]
    navigation_arm: str = "A1_detector_frozen"
    model_split: str = "source_train"
    real_training_authorized: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "identity_binding_sha256",
            "prediction_roster_receipt_sha256",
            "acquisition_support_lineage_receipt_sha256",
            "stable_origin_registry_receipt_sha256",
            "target_independent_candidate_roster_receipt_sha256",
            "synthetic_training_authority_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        _identifier(self.prediction_roster_id, "prediction roster ID")
        if (
            self.navigation_arm != "A1_detector_frozen"
            or self.model_split != "source_train"
            or self.real_training_authorized is not False
        ):
            raise ValueError("A1 synthetic roster authority/split drifted")
        rows = tuple(self.records)
        if not rows or not all(isinstance(row, BAIEGA1CompleteRecordRosterEntryV2) for row in rows):
            raise TypeError("A1 complete patient roster requires typed records")
        if len({row.recording_id for row in rows}) != len(rows):
            raise ValueError("A1 complete patient roster repeats a recording")
        by_equivalence: dict[str, list[BAIEGA1CompleteRecordRosterEntryV2]] = {}
        for row in rows:
            by_equivalence.setdefault(row.exact_container_equivalence_id, []).append(row)
        for members in by_equivalence.values():
            representatives = [
                row for row in members
                if row.candidate_status != "exact_signal_alias_excluded"
            ]
            if len(representatives) != 1 or len({row.patient_uid for row in members}) != 1:
                raise ValueError("A1 exact-container group needs one same-patient representative")
        object.__setattr__(self, "records", rows)
        object.__setattr__(
            self,
            "receipt_sha256",
            _canonical_sha256(
                {
                    "schema": "ba_ieg_a1_complete_patient_record_roster_v2",
                    "identity_binding_sha256": self.identity_binding_sha256,
                    "prediction_roster_id": self.prediction_roster_id,
                    "prediction_roster_receipt_sha256": self.prediction_roster_receipt_sha256,
                    "acquisition_support_lineage_receipt_sha256": self.acquisition_support_lineage_receipt_sha256,
                    "stable_origin_registry_receipt_sha256": self.stable_origin_registry_receipt_sha256,
                    "target_independent_candidate_roster_receipt_sha256": self.target_independent_candidate_roster_receipt_sha256,
                    "synthetic_training_authority_receipt_sha256": self.synthetic_training_authority_receipt_sha256,
                    "records": [row.__dict__ for row in rows],
                    "navigation_arm": self.navigation_arm,
                    "model_split": self.model_split,
                    "real_training_authorized": False,
                    "zero_failure_records_retained": True,
                }
            ),
        )


def build_ba_ieg_a1_complete_patient_record_roster_v2(
    *,
    acquisition_support_lineage: Mapping[str, Any],
    identity_binding_sha256: str,
    target_independent_candidate_roster_receipt_sha256: str,
    synthetic_authority: BAIEGA1SyntheticTrainingAuthorityV2 | None,
    records: Sequence[BAIEGA1CompleteRecordRosterEntryV2],
) -> BAIEGA1CompletePatientRecordRosterV2:
    """Bind a complete record denominator to one validated A1 support ledger."""

    lineage = validate_ba_ieg_g0_a1_acquisition_support_lineage_v1(
        dict(acquisition_support_lineage)
    )
    stable = lineage["stable_origin_registry_receipt_sha256"]
    if stable is None:
        raise ValueError("A1 complete patient roster requires a stable-origin receipt")
    if synthetic_authority is None:
        require_real_a1_postfreeze_training_authority_v2(None)
    if not isinstance(synthetic_authority, BAIEGA1SyntheticTrainingAuthorityV2):
        raise TypeError("A1 synthetic route requires typed software authority")
    if (
        synthetic_authority.prediction_roster_receipt_sha256
        != lineage["prediction_roster_receipt_sha256"]
        or synthetic_authority.acquisition_support_lineage_receipt_sha256
        != lineage["receipt_sha256"]
        or synthetic_authority.stable_origin_registry_receipt_sha256 != stable
        or synthetic_authority.target_independent_candidate_roster_receipt_sha256
        != target_independent_candidate_roster_receipt_sha256
    ):
        raise ValueError("A1 synthetic training authority crosses lineage/target freeze")
    typed_rows = tuple(records)
    lineage_by_record = {row["recording_id"]: row for row in lineage["records"]}
    if {row.recording_id for row in typed_rows} != set(lineage_by_record):
        raise ValueError("A1 complete roster must equal the acquisition record denominator")
    for row in typed_rows:
        source = lineage_by_record[row.recording_id]
        if row.patient_uid != source["patient_uid"]:
            raise ValueError("A1 complete roster crosses patient identity")
        expected_status = {
            "completed_zero_candidate": "zero_detector_candidate",
            "technical_failure": "detector_technical_failure_not_evaluable",
            "partial_coverage": "detector_partial_coverage_not_evaluable",
        }.get(source["prediction_outcome"])
        if expected_status is not None and row.candidate_status != expected_status:
            raise ValueError("A1 record status disagrees with prediction outcome")
    return BAIEGA1CompletePatientRecordRosterV2(
        identity_binding_sha256=_sha256(identity_binding_sha256, "identity binding"),
        prediction_roster_id=lineage["prediction_roster_id"],
        prediction_roster_receipt_sha256=lineage["prediction_roster_receipt_sha256"],
        acquisition_support_lineage_receipt_sha256=lineage["receipt_sha256"],
        stable_origin_registry_receipt_sha256=stable,
        target_independent_candidate_roster_receipt_sha256=_sha256(
            target_independent_candidate_roster_receipt_sha256,
            "target-independent candidate roster",
        ),
        synthetic_training_authority_receipt_sha256=(
            synthetic_authority.receipt_sha256
        ),
        records=typed_rows,
    )


class BAIEGA1CompletePatientCappedLogMeanExpV2(nn.Module):
    """A1 counterpart of the frozen complete-record patient aggregator."""

    implementation_id: Final[str] = BA_IEG_A1_COMPLETE_PATIENT_BRIDGE_ID_V2

    def __init__(self) -> None:
        super().__init__()
        self.temperature = 1.0
        self.symmetric_record_logit_clip = 12.0

    def forward(
        self,
        record_evidence: BAIEGPhysicalRecordEvidenceBatchV1 | None,
        roster: BAIEGA1CompletePatientRecordRosterV2,
        *,
        empty_dtype: torch.dtype = torch.float32,
        empty_device: torch.device | str = "cpu",
    ) -> BAIEGCompletePatientAggregationOutputV1:
        if not isinstance(roster, BAIEGA1CompletePatientRecordRosterV2):
            raise TypeError("A1 patient aggregation requires its typed complete roster")
        if record_evidence is not None and not isinstance(record_evidence, BAIEGPhysicalRecordEvidenceBatchV1):
            raise TypeError("A1 patient aggregation accepts physical evidence only")
        if record_evidence is None:
            dtype, device = empty_dtype, torch.device(empty_device)
            evidence_by_record: dict[str, int] = {}
        else:
            dtype = record_evidence.physical_electrode_record_logits.dtype
            device = record_evidence.physical_electrode_record_logits.device
            evidence_by_record = {
                value: index for index, value in enumerate(record_evidence.recording_ids)
            }
        candidate_statuses = {
            "qualified_event_candidates_present",
            "event_candidates_present_none_qualified",
        }
        expected_evidence = {
            row.recording_id for row in roster.records
            if row.candidate_status in candidate_statuses
        }
        if set(evidence_by_record) != expected_evidence:
            raise ValueError("A1 record evidence does not equal candidate-bearing roster")

        groups: dict[str, list[BAIEGA1CompleteRecordRosterEntryV2]] = {}
        for row in roster.records:
            groups.setdefault(row.exact_container_equivalence_id, []).append(row)
        group_rows = []
        for equivalence_id in sorted(groups):
            members = tuple(groups[equivalence_id])
            representatives = [row for row in members if row.candidate_status != "exact_signal_alias_excluded"]
            if len(representatives) != 1:
                raise RuntimeError("validated A1 roster lost its representative")
            group_rows.append((equivalence_id, representatives[0], members))
        patients = tuple(sorted({row.patient_uid for row in roster.records}))
        patient_lookup = {value: index for index, value in enumerate(patients)}
        record_count = len(group_rows)
        shape = (record_count, len(STANDARD_19))
        record_logits = torch.zeros(shape, dtype=dtype, device=device)
        record_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        candidate_present = torch.zeros(record_count, dtype=torch.bool, device=device)
        qualified_present = torch.zeros_like(candidate_present)
        zero_candidate = torch.zeros_like(candidate_present)
        unique = torch.zeros(record_count, dtype=torch.long, device=device)
        qualified = torch.zeros_like(unique)
        uncertain = torch.zeros_like(unique)
        not_evaluable = torch.zeros_like(unique)
        copied = torch.zeros_like(unique)
        record_patient_index = torch.tensor(
            [patient_lookup[row.patient_uid] for _, row, _ in group_rows],
            dtype=torch.long,
            device=device,
        )
        for index, (_, row, members) in enumerate(group_rows):
            alias_count = len(members) - 1
            if row.candidate_status in candidate_statuses:
                evidence_index = evidence_by_record[row.recording_id]
                assert record_evidence is not None
                observed = int(record_evidence.unique_occurrence_count[evidence_index])
                observed_qualified = int(record_evidence.qualified_unique_occurrence_count[evidence_index])
                if (observed, observed_qualified) != (
                    row.expected_unique_occurrence_count,
                    row.expected_qualified_unique_occurrence_count,
                ):
                    raise ValueError("A1 record evidence lost/duplicated a frozen occurrence")
                record_logits[index] = record_evidence.physical_electrode_record_logits[evidence_index]
                record_mask[index] = record_evidence.physical_electrode_mask[evidence_index]
                unique[index] = observed
                qualified[index] = observed_qualified
                uncertain[index] = record_evidence.uncertain_unique_occurrence_count[evidence_index]
                not_evaluable[index] = record_evidence.not_evaluable_unique_occurrence_count[evidence_index]
                copied[index] = record_evidence.copied_event_count[evidence_index] + alias_count
                candidate_present[index] = True
                qualified_present[index] = observed_qualified > 0
            elif row.candidate_status == "zero_detector_candidate":
                zero_candidate[index] = True
                copied[index] = alias_count
            else:
                copied[index] = alias_count

        patient_shape = (len(patients), len(STANDARD_19))
        patient_logits = torch.zeros(patient_shape, dtype=dtype, device=device)
        patient_mask = torch.zeros(patient_shape, dtype=torch.bool, device=device)
        patient_unique = torch.zeros(len(patients), dtype=torch.long, device=device)
        patient_candidate = torch.zeros_like(patient_unique)
        patient_zero = torch.zeros_like(patient_unique)
        patient_evaluable = torch.zeros_like(patient_unique)
        clipped = record_logits.clamp(-12.0, 12.0)
        for patient_index in range(len(patients)):
            selected_records = record_patient_index == patient_index
            patient_unique[patient_index] = selected_records.sum()
            patient_candidate[patient_index] = (selected_records & candidate_present).sum()
            patient_zero[patient_index] = (selected_records & zero_candidate).sum()
            patient_evaluable[patient_index] = (selected_records & record_mask.any(dim=1)).sum()
            for electrode_index in range(len(STANDARD_19)):
                selected = selected_records & record_mask[:, electrode_index]
                count = int(selected.sum())
                if count:
                    patient_logits[patient_index, electrode_index] = (
                        torch.logsumexp(clipped[selected, electrode_index], dim=0)
                        - math.log(count)
                    )
                    patient_mask[patient_index, electrode_index] = True
        return BAIEGCompletePatientAggregationOutputV1(
            implementation_id=self.implementation_id,
            identity_binding_sha256=roster.identity_binding_sha256,
            complete_roster_receipt_sha256=roster.receipt_sha256,
            navigation_arm=roster.navigation_arm,
            patient_uids=patients,
            exact_container_equivalence_ids=tuple(row[0] for row in group_rows),
            representative_recording_ids=tuple(row[1].recording_id for row in group_rows),
            record_patient_index=record_patient_index,
            complete_record_logits=record_logits,
            complete_record_mask=record_mask,
            record_candidate_present_mask=candidate_present,
            record_qualified_candidate_present_mask=qualified_present,
            zero_candidate_record_mask=zero_candidate,
            unique_occurrence_count=unique,
            qualified_unique_occurrence_count=qualified,
            uncertain_unique_occurrence_count=uncertain,
            not_evaluable_unique_occurrence_count=not_evaluable,
            copied_event_count=copied,
            patient_logits=patient_logits,
            patient_electrode_mask=patient_mask,
            patient_unique_record_count=patient_unique,
            patient_candidate_record_count=patient_candidate,
            patient_zero_candidate_record_count=patient_zero,
            patient_evaluable_record_count=patient_evaluable,
        )


__all__ = [
    "BA_IEG_A1_COMPLETE_PATIENT_BRIDGE_ID_V2",
    "BA_IEG_A1_RECORD_STATUSES_V2",
    "BA_IEG_A1_SYNTHETIC_TRAINING_AUTHORITY_SCHEMA_V2",
    "BAIEGA1CompletePatientCappedLogMeanExpV2",
    "BAIEGA1CompletePatientRecordRosterV2",
    "BAIEGA1CompleteRecordRosterEntryV2",
    "BAIEGA1SyntheticTrainingAuthorityV2",
    "build_ba_ieg_a1_complete_patient_record_roster_v2",
    "require_real_a1_postfreeze_training_authority_v2",
]
