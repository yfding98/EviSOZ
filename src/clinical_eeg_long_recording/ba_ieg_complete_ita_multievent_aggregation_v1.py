"""Complete, EEG-only, multi-occurrence ITA aggregation for BA-IEG.

This additive module closes a deliberately different estimand from
``ba_ieg_capped_log_mean_exp_event_bag_v1``.  The older bag is a
``qualified_ictal``-only evidence-grade secondary analysis.  Here the primary
track is intention-to-analyze (ITA): every EEG-evaluable physical occurrence
contributes exactly one event distribution, whether its target-free event
qualification is ``qualified_ictal`` or ``uncertain``.

Important invariants are implemented rather than left as prose:

* a complete record roster represents zero-candidate records explicitly;
* detector fragments are collapsed by a content-addressed, reference-free
  occurrence receipt frozen before typed-unit logits are read;
* one canonical fragment supplies evidence and an arbitrary number of aliases
  cannot increase an occurrence's weight;
* complete, partial, failure and not-evaluable outcomes remain in the record
  denominator;
* an evaluable event without a locked typed-unit hypothesis contributes one
  unit of explicit ``unresolved`` mass instead of disappearing or receiving a
  fabricated uniform channel distribution;
* ITA and qualified-only results have separate memberships, normalization,
  roles and authorization flags;
* event order, repeated identical model outputs and microbatch partitioning do
  not affect the numerical result;
* detector score, event duration, late amplitude, token count and detector
  fragment count are absent from the weighting API.

The output is a research ranking over scalp-visible typed onset units.  A
bipolar derivation remains one whole lead and is never split across endpoints.
Pairwise distribution distances and earliest-field overlap are emitted only
as a target-free heterogeneity *shadow*; this module has no clinical seizure
type or cortical SOZ/EZ naming surface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Iterable, Mapping, Sequence

import torch

from src.soz.geometry import STANDARD_19

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
)
from .ba_ieg_target_free_event_qualification_v1 import (
    BAIEGTargetFreeEventQualificationReceiptV1,
)


BA_IEG_COMPLETE_ITA_MULTIEVENT_AGGREGATION_ID_V1: Final[
    str
] = "ba_ieg_complete_ita_multievent_aggregation_v1"
BA_IEG_REFERENCE_FREE_OCCURRENCE_DEDUP_SCHEMA_V1: Final[
    str
] = "ba_ieg_reference_free_occurrence_dedup_receipt_v1"
BA_IEG_COMPLETE_ITA_MANIFEST_SCHEMA_V1: Final[
    str
] = "ba_ieg_complete_ita_record_roster_manifest_v1"
BA_IEG_ITA_PRIMARY_TRACK_ID_V1: Final[
    str
] = "ita_primary_all_eeg_evaluable_occurrences_v1"
BA_IEG_QUALIFIED_ONLY_SECONDARY_TRACK_ID_V1: Final[
    str
] = "qualified_only_secondary_not_complete_record_substitute_v1"
BA_IEG_PROCESSING_STATUSES_V1: Final[tuple[str, ...]] = (
    "complete",
    "partial",
    "failure",
    "not_evaluable",
)
BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1: Final[
    str
] = "unresolved:no_locked_typed_onset_distribution"

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOL: Final[float] = 1e-9
_FROZEN_LOGIT_CLIP: Final[float] = 12.0
_ELECTRODE_KIND_INDEX: Final[int] = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index(
    "physical_electrode"
)
_LEAD_KIND_INDEX: Final[int] = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("bipolar_lead")

_EEG_ONLY_INFERENCE_SCOPE: Final[dict[str, bool]] = {
    "canonical_eeg_samples_used": True,
    "typed_causal_onset_output_used": True,
    "target_free_event_qualification_used": True,
    "reference_free_occurrence_receipts_used": True,
    "edf_annotations_used": False,
    "spreadsheet_or_excel_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_or_patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "activation_or_provocation_used": False,
    "ecg_emg_eog_used": False,
    "llm_or_knowledge_base_used": False,
}

_RANK_WEIGHT_POLICY: Final[dict[str, object]] = {
    "aggregation": "equal_occurrence_probability_mixture_after_deduplication",
    "event_presence_gate": "q_e_equals_one_sensitivity_analysis",
    "detector_score_weight": False,
    "event_duration_weight": False,
    "late_amplitude_weight": False,
    "late_recruitment_or_spread_weight": False,
    "token_or_selected_second_weight": False,
    "detector_fragment_count_weight": False,
    "reference_copy_weight": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _content_receipt(payload: Mapping[str, Any], field_name: str) -> str:
    candidate = deepcopy(dict(payload))
    candidate[field_name] = "CONTENT-ADDRESS-PENDING"
    return _canonical_sha256(candidate)


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BAIEGReferenceFreeOccurrenceDedupReceiptV1:
    """Freeze one physical occurrence before any typed-unit rank is read.

    Fragment identities and their source receipts are sorted as aligned pairs,
    so caller order cannot change the receipt.  The temporal envelope may be
    used for target-free grouping, but it is never exposed to the rank-weight
    function.
    """

    recording_id: str
    occurrence_id: str
    fragment_event_ids: tuple[str, ...]
    fragment_source_event_receipt_sha256s: tuple[str, ...]
    canonical_event_id: str
    reference_free_temporal_envelope_seconds: tuple[float, float]
    complete_candidate_roster_receipt_sha256: str
    schema_version: str = BA_IEG_REFERENCE_FREE_OCCURRENCE_DEDUP_SCHEMA_V1
    grouping_policy_id: str = (
        "reference_free_temporal_envelope_boundary_consistency_provider_lineage_v1"
    )
    reference_view_read: bool = False
    typed_unit_rank_read: bool = False
    localization_target_read: bool = False
    detector_score_read: bool = False
    clinical_or_annotation_source_read: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recording_id", _identifier(self.recording_id, "recording_id")
        )
        object.__setattr__(
            self, "occurrence_id", _identifier(self.occurrence_id, "occurrence_id")
        )
        event_ids = tuple(
            _identifier(value, "fragment_event_id") for value in self.fragment_event_ids
        )
        source_receipts = tuple(
            _sha256(value, "fragment_source_event_receipt_sha256")
            for value in self.fragment_source_event_receipt_sha256s
        )
        if not event_ids or len(event_ids) != len(source_receipts):
            raise ValueError("fragment event and source-receipt rosters must align")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("one occurrence receipt repeats a fragment event ID")
        aligned = tuple(sorted(zip(event_ids, source_receipts)))
        object.__setattr__(self, "fragment_event_ids", tuple(row[0] for row in aligned))
        object.__setattr__(
            self,
            "fragment_source_event_receipt_sha256s",
            tuple(row[1] for row in aligned),
        )
        canonical = _identifier(self.canonical_event_id, "canonical_event_id")
        if canonical not in event_ids:
            raise ValueError("canonical event must be one fragment in its occurrence")
        object.__setattr__(self, "canonical_event_id", canonical)
        if (
            not isinstance(self.reference_free_temporal_envelope_seconds, tuple)
            or len(self.reference_free_temporal_envelope_seconds) != 2
        ):
            raise TypeError(
                "reference-free temporal envelope must be a two-value tuple"
            )
        start = _finite(
            self.reference_free_temporal_envelope_seconds[0],
            "reference_free_temporal_envelope_seconds[0]",
        )
        stop = _finite(
            self.reference_free_temporal_envelope_seconds[1],
            "reference_free_temporal_envelope_seconds[1]",
        )
        if start < 0.0 or stop < start:
            raise ValueError("reference-free temporal envelope is reversed or negative")
        object.__setattr__(
            self, "reference_free_temporal_envelope_seconds", (start, stop)
        )
        object.__setattr__(
            self,
            "complete_candidate_roster_receipt_sha256",
            _sha256(
                self.complete_candidate_roster_receipt_sha256,
                "complete_candidate_roster_receipt_sha256",
            ),
        )
        if self.schema_version != BA_IEG_REFERENCE_FREE_OCCURRENCE_DEDUP_SCHEMA_V1:
            raise ValueError("occurrence dedup receipt schema drifted")
        if self.grouping_policy_id != (
            "reference_free_temporal_envelope_boundary_consistency_provider_lineage_v1"
        ):
            raise ValueError("occurrence grouping policy drifted")
        forbidden_flags = (
            self.reference_view_read,
            self.typed_unit_rank_read,
            self.localization_target_read,
            self.detector_score_read,
            self.clinical_or_annotation_source_read,
        )
        if any(value is not False for value in forbidden_flags):
            raise ValueError(
                "occurrence deduplication must be reference-, rank-, target- and clinical-free"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _content_receipt(self.to_dict(include_receipt=False), "receipt_sha256"),
        )

    @property
    def canonical_source_event_receipt_sha256(self) -> str:
        index = self.fragment_event_ids.index(self.canonical_event_id)
        return self.fragment_source_event_receipt_sha256s[index]

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "grouping_policy_id": self.grouping_policy_id,
            "recording_id": self.recording_id,
            "occurrence_id": self.occurrence_id,
            "fragment_event_ids": list(self.fragment_event_ids),
            "fragment_source_event_receipt_sha256s": list(
                self.fragment_source_event_receipt_sha256s
            ),
            "canonical_event_id": self.canonical_event_id,
            "reference_free_temporal_envelope_seconds": list(
                self.reference_free_temporal_envelope_seconds
            ),
            "complete_candidate_roster_receipt_sha256": (
                self.complete_candidate_roster_receipt_sha256
            ),
            "reference_view_read": self.reference_view_read,
            "typed_unit_rank_read": self.typed_unit_rank_read,
            "localization_target_read": self.localization_target_read,
            "detector_score_read": self.detector_score_read,
            "clinical_or_annotation_source_read": (
                self.clinical_or_annotation_source_read
            ),
        }
        if include_receipt:
            result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True)
class BAIEGCompleteITAOccurrenceEntryV1:
    """One deduplicated occurrence plus its target-free disposition."""

    dedup_receipt: BAIEGReferenceFreeOccurrenceDedupReceiptV1
    processing_status: str
    processing_receipt_sha256: str
    qualification_receipt: BAIEGTargetFreeEventQualificationReceiptV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.dedup_receipt, BAIEGReferenceFreeOccurrenceDedupReceiptV1
        ):
            raise TypeError("occurrence entry needs a typed dedup receipt")
        if self.processing_status not in BA_IEG_PROCESSING_STATUSES_V1:
            raise ValueError("occurrence processing status is unsupported")
        object.__setattr__(
            self,
            "processing_receipt_sha256",
            _sha256(self.processing_receipt_sha256, "processing_receipt_sha256"),
        )
        qualification = self.qualification_receipt
        if qualification is not None:
            if not isinstance(
                qualification, BAIEGTargetFreeEventQualificationReceiptV1
            ):
                raise TypeError("qualification receipt must be a typed provider output")
            observation = qualification.observation
            expected = (
                self.dedup_receipt.canonical_event_id,
                self.dedup_receipt.recording_id,
                self.dedup_receipt.canonical_source_event_receipt_sha256,
                self.dedup_receipt.occurrence_id,
            )
            observed = (
                observation.event_id,
                observation.recording_id,
                observation.source_event_receipt_sha256,
                qualification.occurrence_equivalence_id,
            )
            if observed != expected:
                raise ValueError(
                    "qualification and reference-free occurrence identities drifted"
                )
        if self.processing_status in {"complete", "partial"}:
            if qualification is None:
                raise ValueError(
                    "an EEG-evaluable occurrence needs qualification evidence"
                )
            if qualification.event_aggregation_status == "not_evaluable":
                raise ValueError(
                    "complete/partial processing cannot contradict not-evaluable qualification"
                )
        elif qualification is not None and qualification.event_aggregation_status != (
            "not_evaluable"
        ):
            raise ValueError(
                "failure/not-evaluable processing cannot carry an evaluable qualification"
            )

    @property
    def eeg_evaluable(self) -> bool:
        return self.processing_status in {"complete", "partial"}

    @property
    def qualification_status(self) -> str:
        if self.qualification_receipt is None:
            return "not_evaluable"
        return self.qualification_receipt.event_aggregation_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "dedup_receipt": self.dedup_receipt.to_dict(),
            "processing_status": self.processing_status,
            "processing_receipt_sha256": self.processing_receipt_sha256,
            "qualification_receipt_sha256": (
                None
                if self.qualification_receipt is None
                else self.qualification_receipt.receipt_sha256
            ),
            "qualification_status": self.qualification_status,
            "eeg_evaluable": self.eeg_evaluable,
        }


@dataclass(frozen=True)
class BAIEGCompleteITARecordRosterManifestV1:
    """Complete record/candidate roster, including records with zero candidates."""

    recording_ids: tuple[str, ...]
    candidate_fragment_counts: tuple[int, ...]
    occurrence_entries: tuple[BAIEGCompleteITAOccurrenceEntryV1, ...]
    complete_candidate_roster_receipt_sha256: str
    schema_version: str = BA_IEG_COMPLETE_ITA_MANIFEST_SCHEMA_V1
    top_n_truncation_applied: bool = False
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        records = tuple(
            _identifier(value, "recording_id") for value in self.recording_ids
        )
        counts = tuple(self.candidate_fragment_counts)
        if not records or len(records) != len(counts):
            raise ValueError(
                "record and candidate-count rosters must align and be non-empty"
            )
        if len(set(records)) != len(records):
            raise ValueError("complete record roster repeats a recording ID")
        for value in counts:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "candidate fragment counts must be non-negative integers"
                )
        aligned_records = tuple(sorted(zip(records, counts)))
        object.__setattr__(
            self, "recording_ids", tuple(row[0] for row in aligned_records)
        )
        object.__setattr__(
            self, "candidate_fragment_counts", tuple(row[1] for row in aligned_records)
        )
        roster_receipt = _sha256(
            self.complete_candidate_roster_receipt_sha256,
            "complete_candidate_roster_receipt_sha256",
        )
        object.__setattr__(
            self, "complete_candidate_roster_receipt_sha256", roster_receipt
        )
        if self.schema_version != BA_IEG_COMPLETE_ITA_MANIFEST_SCHEMA_V1:
            raise ValueError("complete ITA manifest schema drifted")
        if self.top_n_truncation_applied is not False:
            raise ValueError("complete ITA forbids top-N candidate truncation")

        entries = tuple(self.occurrence_entries)
        if any(
            not isinstance(entry, BAIEGCompleteITAOccurrenceEntryV1)
            for entry in entries
        ):
            raise TypeError("manifest occurrence entries must be typed")
        entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.dedup_receipt.recording_id,
                    entry.dedup_receipt.reference_free_temporal_envelope_seconds,
                    entry.dedup_receipt.occurrence_id,
                ),
            )
        )
        object.__setattr__(self, "occurrence_entries", entries)
        record_set = set(self.recording_ids)
        occurrence_ids: set[str] = set()
        fragment_ids: set[str] = set()
        observed_counts = {record: 0 for record in self.recording_ids}
        for entry in entries:
            receipt = entry.dedup_receipt
            if receipt.recording_id not in record_set:
                raise ValueError(
                    "occurrence belongs to a record outside the complete roster"
                )
            if receipt.complete_candidate_roster_receipt_sha256 != roster_receipt:
                raise ValueError(
                    "occurrence dedup receipt is bound to another candidate roster"
                )
            if receipt.occurrence_id in occurrence_ids:
                raise ValueError("manifest repeats an occurrence ID")
            occurrence_ids.add(receipt.occurrence_id)
            overlap = fragment_ids.intersection(receipt.fragment_event_ids)
            if overlap:
                raise ValueError(
                    "one detector fragment belongs to multiple occurrences"
                )
            fragment_ids.update(receipt.fragment_event_ids)
            observed_counts[receipt.recording_id] += len(receipt.fragment_event_ids)
        declared_counts = dict(zip(self.recording_ids, self.candidate_fragment_counts))
        if observed_counts != declared_counts:
            raise ValueError(
                "complete candidate fragment counts disagree with occurrence receipts"
            )
        object.__setattr__(
            self,
            "manifest_sha256",
            _content_receipt(self.to_dict(include_manifest=False), "manifest_sha256"),
        )

    def to_dict(self, *, include_manifest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "recording_ids": list(self.recording_ids),
            "candidate_fragment_counts": list(self.candidate_fragment_counts),
            "occurrence_entries": [
                entry.to_dict() for entry in self.occurrence_entries
            ],
            "complete_candidate_roster_receipt_sha256": (
                self.complete_candidate_roster_receipt_sha256
            ),
            "top_n_truncation_applied": self.top_n_truncation_applied,
            "inference_scope": deepcopy(_EEG_ONLY_INFERENCE_SCOPE),
        }
        if include_manifest:
            result["manifest_sha256"] = self.manifest_sha256
        return result


@dataclass(frozen=True)
class BAIEGTypedUnitProbabilityV1:
    unit_key: str
    unit_kind: str
    probability: float
    electrode_id: str | None = None
    whole_bipolar_lead: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_key": self.unit_key,
            "unit_kind": self.unit_kind,
            "electrode_id": self.electrode_id,
            "whole_bipolar_lead": (
                None
                if self.whole_bipolar_lead is None
                else list(self.whole_bipolar_lead)
            ),
            "probability": self.probability,
        }


@dataclass(frozen=True)
class BAIEGPerOccurrenceDistributionV1:
    recording_id: str
    occurrence_id: str
    canonical_event_id: str
    dedup_receipt_sha256: str
    processing_status: str
    qualification_status: str
    qualification_receipt_sha256: str
    available_resolution_kinds: tuple[str, ...]
    typed_unit_probabilities: tuple[BAIEGTypedUnitProbabilityV1, ...]
    unresolved_probability: float
    earliest_typed_unit_keys: tuple[str, ...]
    top_distribution_key: str
    distribution_sha256: str

    def probability_map(self) -> dict[str, float]:
        result = {
            row.unit_key: float(row.probability)
            for row in self.typed_unit_probabilities
        }
        result[BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1] = float(self.unresolved_probability)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "occurrence_id": self.occurrence_id,
            "canonical_event_id": self.canonical_event_id,
            "dedup_receipt_sha256": self.dedup_receipt_sha256,
            "processing_status": self.processing_status,
            "qualification_status": self.qualification_status,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "available_resolution_kinds": list(self.available_resolution_kinds),
            "typed_unit_probabilities": [
                row.to_dict() for row in self.typed_unit_probabilities
            ],
            "unresolved_probability": self.unresolved_probability,
            "earliest_typed_unit_keys": list(self.earliest_typed_unit_keys),
            "top_distribution_key": self.top_distribution_key,
            "distribution_sha256": self.distribution_sha256,
            "probability_semantics": (
                "within_event_uncalibrated_softmax_or_explicit_unresolved_mass"
            ),
        }


@dataclass(frozen=True)
class BAIEGLeaveOneOccurrenceOutResultV1:
    omitted_occurrence_id: str
    status: str
    typed_unit_probabilities: tuple[BAIEGTypedUnitProbabilityV1, ...]
    unresolved_probability: float | None
    top_distribution_key: str | None
    full_vs_loeo_total_variation: float | None
    top1_changed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "omitted_occurrence_id": self.omitted_occurrence_id,
            "status": self.status,
            "typed_unit_probabilities": [
                row.to_dict() for row in self.typed_unit_probabilities
            ],
            "unresolved_probability": self.unresolved_probability,
            "top_distribution_key": self.top_distribution_key,
            "full_vs_loeo_total_variation": self.full_vs_loeo_total_variation,
            "top1_changed": self.top1_changed,
        }


@dataclass(frozen=True)
class BAIEGTargetFreePairwiseHeterogeneityV1:
    left_occurrence_id: str
    right_occurrence_id: str
    total_variation_distance: float
    top1_discordant: bool
    earliest_field_overlap_status: str
    earliest_field_jaccard: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_occurrence_id": self.left_occurrence_id,
            "right_occurrence_id": self.right_occurrence_id,
            "distribution_distance_metric": "total_variation",
            "total_variation_distance": self.total_variation_distance,
            "top1_discordant": self.top1_discordant,
            "earliest_field_overlap_status": self.earliest_field_overlap_status,
            "earliest_field_jaccard": self.earliest_field_jaccard,
            "semantics": (
                "target_free_heterogeneity_shadow_not_a_clinical_seizure_type"
            ),
        }


@dataclass(frozen=True)
class BAIEGAggregationTrackV1:
    track_id: str
    role: str
    aggregation_status: str
    included_occurrence_ids: tuple[str, ...]
    excluded_occurrence_ids: tuple[str, ...]
    per_occurrence_distributions: tuple[BAIEGPerOccurrenceDistributionV1, ...]
    equal_occurrence_mixture: tuple[BAIEGTypedUnitProbabilityV1, ...]
    unresolved_probability: float | None
    top_distribution_key: str | None
    leave_one_occurrence_out: tuple[BAIEGLeaveOneOccurrenceOutResultV1, ...]
    pairwise_heterogeneity_shadow: tuple[BAIEGTargetFreePairwiseHeterogeneityV1, ...]
    concordant_top1_occurrence_count: int
    discordant_top1_occurrence_count: int
    discordant_event_pair_count: int
    single_occurrence_dependence: float | None
    single_occurrence_dependence_status: str
    production_qualification_authorized: bool
    complete_record_denominator_authorized: bool
    report_claim_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "role": self.role,
            "aggregation_status": self.aggregation_status,
            "included_occurrence_ids": list(self.included_occurrence_ids),
            "excluded_occurrence_ids": list(self.excluded_occurrence_ids),
            "per_occurrence_distributions": [
                row.to_dict() for row in self.per_occurrence_distributions
            ],
            "equal_occurrence_mixture": [
                row.to_dict() for row in self.equal_occurrence_mixture
            ],
            "unresolved_probability": self.unresolved_probability,
            "top_distribution_key": self.top_distribution_key,
            "leave_one_occurrence_out": [
                row.to_dict() for row in self.leave_one_occurrence_out
            ],
            "pairwise_heterogeneity_shadow": [
                row.to_dict() for row in self.pairwise_heterogeneity_shadow
            ],
            "concordant_top1_occurrence_count": (self.concordant_top1_occurrence_count),
            "discordant_top1_occurrence_count": (self.discordant_top1_occurrence_count),
            "discordant_event_pair_count": self.discordant_event_pair_count,
            "single_occurrence_dependence": self.single_occurrence_dependence,
            "single_occurrence_dependence_status": (
                self.single_occurrence_dependence_status
            ),
            "production_qualification_authorized": (
                self.production_qualification_authorized
            ),
            "complete_record_denominator_authorized": (
                self.complete_record_denominator_authorized
            ),
            "report_claim_authorized": self.report_claim_authorized,
            "rank_weight_policy": deepcopy(_RANK_WEIGHT_POLICY),
            "heterogeneity_semantics": (
                "target_free_shadow_without_clinical_mode_or_seizure_type_names"
            ),
        }


@dataclass(frozen=True)
class BAIEGRecordITADenominatorV1:
    recording_id: str
    candidate_fragment_count: int
    unique_occurrence_count: int
    duplicate_fragment_count: int
    complete_occurrence_count: int
    partial_occurrence_count: int
    failure_occurrence_count: int
    not_evaluable_occurrence_count: int
    eeg_evaluable_occurrence_count: int
    qualified_evaluable_occurrence_count: int
    uncertain_evaluable_occurrence_count: int
    resolved_typed_distribution_count: int
    unresolved_typed_distribution_count: int
    nonproduction_qualified_occurrence_count: int
    zero_candidate_record: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class BAIEGCompleteITARecordOutputV1:
    recording_id: str
    denominator: BAIEGRecordITADenominatorV1
    occurrence_dedup_receipt_sha256s: tuple[str, ...]
    ita_primary: BAIEGAggregationTrackV1
    qualified_only_secondary: BAIEGAggregationTrackV1
    record_output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "denominator": self.denominator.to_dict(),
            "occurrence_dedup_receipt_sha256s": list(
                self.occurrence_dedup_receipt_sha256s
            ),
            "ita_primary": self.ita_primary.to_dict(),
            "qualified_only_secondary": self.qualified_only_secondary.to_dict(),
            "record_output_sha256": self.record_output_sha256,
            "tracks_are_separate_estimands": True,
        }


@dataclass(frozen=True)
class BAIEGCompleteITAMultiEventAggregationOutputV1:
    implementation_id: str
    source_manifest_sha256: str
    records: tuple[BAIEGCompleteITARecordOutputV1, ...]
    output_sha256: str
    output_semantics: str = (
        "complete_record_ita_scalp_visible_typed_onset_research_distribution_"
        "not_cortical_soz_or_ez_not_clinical_diagnosis"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "records": [record.to_dict() for record in self.records],
            "inference_scope": deepcopy(_EEG_ONLY_INFERENCE_SCOPE),
            "output_semantics": self.output_semantics,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class _TypedUnitDescriptor:
    unit_key: str
    unit_kind: str
    electrode_id: str | None
    whole_bipolar_lead: tuple[str, str] | None


@dataclass(frozen=True)
class _ProjectedEventEvidence:
    event_id: str
    recording_id: str
    source_event_receipt_sha256: str
    source_input_batch_sha256s: tuple[str, ...]
    available_resolution_kinds: tuple[str, ...]
    logits: tuple[tuple[_TypedUnitDescriptor, float], ...]
    earliest_typed_unit_keys: tuple[str, ...]

    def invariant_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "source_event_receipt_sha256": self.source_event_receipt_sha256,
            "available_resolution_kinds": list(self.available_resolution_kinds),
            "logits": [
                {
                    "unit_key": descriptor.unit_key,
                    "unit_kind": descriptor.unit_kind,
                    "electrode_id": descriptor.electrode_id,
                    "whole_bipolar_lead": descriptor.whole_bipolar_lead,
                    "logit": logit,
                }
                for descriptor, logit in self.logits
            ],
            "earliest_typed_unit_keys": list(self.earliest_typed_unit_keys),
        }


def _unit_descriptor(
    kind_index: int,
    electrode_index: int,
    lead_endpoint_indices: tuple[int, int],
) -> _TypedUnitDescriptor:
    if kind_index == _ELECTRODE_KIND_INDEX:
        if electrode_index < 0 or electrode_index >= len(STANDARD_19):
            raise ValueError("physical typed unit has an invalid electrode index")
        electrode = STANDARD_19[electrode_index]
        return _TypedUnitDescriptor(
            unit_key=f"physical_electrode:{electrode}",
            unit_kind="physical_electrode",
            electrode_id=electrode,
            whole_bipolar_lead=None,
        )
    if kind_index == _LEAD_KIND_INDEX:
        first, second = lead_endpoint_indices
        if (
            first < 0
            or second < 0
            or first >= len(STANDARD_19)
            or second >= len(STANDARD_19)
            or first >= second
        ):
            raise ValueError("bipolar typed unit has invalid whole-lead endpoints")
        endpoints = (STANDARD_19[first], STANDARD_19[second])
        return _TypedUnitDescriptor(
            unit_key=f"whole_bipolar_lead:{endpoints[0]}--{endpoints[1]}",
            unit_kind="whole_bipolar_lead",
            electrode_id=None,
            whole_bipolar_lead=endpoints,
        )
    raise ValueError("typed output contains an unsupported unit kind")


def _validate_head_shapes(output: BAIEGShallowCausalTypedUnitHeadOutput) -> None:
    if not isinstance(output, BAIEGShallowCausalTypedUnitHeadOutput):
        raise TypeError("complete ITA accepts only typed-unit head outputs")
    event_count = len(output.event_ids)
    if not (
        len(output.recording_ids)
        == len(output.source_event_receipt_sha256s)
        == event_count
    ):
        raise ValueError("typed-head event identity rosters do not align")
    if len(set(output.event_ids)) != event_count:
        raise ValueError("one typed-head microbatch repeats an event ID")
    logits = output.typed_unit_event_logits
    if logits.ndim != 2 or int(logits.shape[0]) != event_count:
        raise ValueError("typed event logits do not align with event identities")
    typed_count = int(logits.shape[1])
    expected = (event_count, typed_count)
    for name, value in (
        ("typed_unit_mask", output.typed_unit_mask),
        ("typed_unit_inventory_mask", output.typed_unit_inventory_mask),
        ("typed_unit_kind_index", output.typed_unit_kind_index),
        ("typed_unit_electrode_index", output.typed_unit_electrode_index),
        (
            "typed_unit_candidate_boundary_mask",
            output.typed_unit_candidate_boundary_mask,
        ),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} does not align with typed event logits")
    if tuple(output.typed_unit_lead_endpoint_index.shape) != (
        event_count,
        typed_count,
        2,
    ):
        raise ValueError("typed whole-lead endpoint identities do not align")
    if tuple(output.typed_unit_candidate_boundary_interval_seconds.shape) != (
        event_count,
        typed_count,
        2,
    ):
        raise ValueError("typed boundary candidate intervals do not align")
    if (
        output.typed_unit_mask.dtype != torch.bool
        or output.typed_unit_inventory_mask.dtype != torch.bool
        or output.typed_unit_candidate_boundary_mask.dtype != torch.bool
    ):
        raise ValueError("typed opportunity and boundary masks must be boolean")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("typed event logits must be finite floating point")


def _project_one_head_event(
    output: BAIEGShallowCausalTypedUnitHeadOutput,
    event_index: int,
) -> _ProjectedEventEvidence:
    inventory = output.typed_unit_inventory_mask[event_index]
    rank_mask = output.typed_unit_mask[event_index]
    if bool((rank_mask & ~inventory).any().item()):
        raise ValueError("typed rank mask exceeds the declared inventory")
    descriptors: dict[int, _TypedUnitDescriptor] = {}
    seen_keys: set[str] = set()
    for typed_index_tensor in torch.nonzero(inventory, as_tuple=False).flatten():
        typed_index = int(typed_index_tensor)
        descriptor = _unit_descriptor(
            int(output.typed_unit_kind_index[event_index, typed_index]),
            int(output.typed_unit_electrode_index[event_index, typed_index]),
            tuple(
                int(value)
                for value in output.typed_unit_lead_endpoint_index[
                    event_index, typed_index
                ]
            ),
        )
        if descriptor.unit_key in seen_keys:
            raise ValueError("one event repeats a typed-unit identity")
        descriptors[typed_index] = descriptor
        seen_keys.add(descriptor.unit_key)

    logits = tuple(
        sorted(
            (
                (
                    descriptors[typed_index],
                    float(
                        output.typed_unit_event_logits[event_index, typed_index].item()
                    ),
                )
                for typed_index in descriptors
                if bool(rank_mask[typed_index].item())
            ),
            key=lambda row: row[0].unit_key,
        )
    )

    boundary_rows: list[tuple[str, float, float]] = []
    boundary_mask = output.typed_unit_candidate_boundary_mask[event_index]
    for typed_index_tensor in torch.nonzero(
        boundary_mask & inventory, as_tuple=False
    ).flatten():
        typed_index = int(typed_index_tensor)
        lower, upper = (
            float(value)
            for value in output.typed_unit_candidate_boundary_interval_seconds[
                event_index, typed_index
            ]
        )
        if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            raise ValueError("typed boundary candidate interval is invalid")
        boundary_rows.append((descriptors[typed_index].unit_key, lower, upper))
    if boundary_rows:
        earliest_upper = min(row[2] for row in boundary_rows)
        earliest_keys = tuple(
            sorted(row[0] for row in boundary_rows if row[1] <= earliest_upper + _TOL)
        )
    else:
        earliest_keys = ()
    return _ProjectedEventEvidence(
        event_id=_identifier(output.event_ids[event_index], "event_id"),
        recording_id=_identifier(output.recording_ids[event_index], "recording_id"),
        source_event_receipt_sha256=_sha256(
            output.source_event_receipt_sha256s[event_index],
            "source_event_receipt_sha256",
        ),
        source_input_batch_sha256s=(
            _sha256(output.source_input_batch_sha256, "source_input_batch_sha256"),
        ),
        available_resolution_kinds=tuple(
            sorted({descriptor.unit_kind for descriptor in descriptors.values()})
        ),
        logits=logits,
        earliest_typed_unit_keys=earliest_keys,
    )


def _project_microbatches(
    outputs: Iterable[BAIEGShallowCausalTypedUnitHeadOutput],
) -> dict[str, _ProjectedEventEvidence]:
    projected: dict[str, _ProjectedEventEvidence] = {}
    for output in tuple(outputs):
        _validate_head_shapes(output)
        for event_index in range(len(output.event_ids)):
            row = _project_one_head_event(output, event_index)
            previous = projected.get(row.event_id)
            if previous is None:
                projected[row.event_id] = row
                continue
            if previous.invariant_payload() != row.invariant_payload():
                raise ValueError(
                    "the same event ID has conflicting evidence across microbatches"
                )
            projected[row.event_id] = _ProjectedEventEvidence(
                event_id=previous.event_id,
                recording_id=previous.recording_id,
                source_event_receipt_sha256=previous.source_event_receipt_sha256,
                source_input_batch_sha256s=tuple(
                    sorted(
                        set(previous.source_input_batch_sha256s).union(
                            row.source_input_batch_sha256s
                        )
                    )
                ),
                available_resolution_kinds=previous.available_resolution_kinds,
                logits=previous.logits,
                earliest_typed_unit_keys=previous.earliest_typed_unit_keys,
            )
    return projected


def _probability_rows(
    mapping: Mapping[str, float],
    descriptors: Mapping[str, _TypedUnitDescriptor],
) -> tuple[BAIEGTypedUnitProbabilityV1, ...]:
    rows: list[BAIEGTypedUnitProbabilityV1] = []
    for key, probability in mapping.items():
        if key == BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1:
            continue
        descriptor = descriptors[key]
        rows.append(
            BAIEGTypedUnitProbabilityV1(
                unit_key=key,
                unit_kind=descriptor.unit_kind,
                electrode_id=descriptor.electrode_id,
                whole_bipolar_lead=descriptor.whole_bipolar_lead,
                probability=float(probability),
            )
        )
    return tuple(sorted(rows, key=lambda row: (-row.probability, row.unit_key)))


def _top_key(mapping: Mapping[str, float]) -> str:
    return min(mapping, key=lambda key: (-float(mapping[key]), key))


def _event_distribution(
    entry: BAIEGCompleteITAOccurrenceEntryV1,
    evidence: _ProjectedEventEvidence,
) -> tuple[BAIEGPerOccurrenceDistributionV1, dict[str, _TypedUnitDescriptor]]:
    receipt = entry.dedup_receipt
    qualification = entry.qualification_receipt
    if qualification is None:
        raise AssertionError("evaluable event unexpectedly lacks qualification")
    if evidence.recording_id != receipt.recording_id or (
        evidence.source_event_receipt_sha256
        != receipt.canonical_source_event_receipt_sha256
    ):
        raise ValueError("canonical model evidence is bound to another event/record")
    if qualification.observation.source_input_batch_sha256 not in (
        evidence.source_input_batch_sha256s
    ):
        raise ValueError(
            "qualification receipt is not bound to a supplied model microbatch"
        )

    descriptor_map = {
        descriptor.unit_key: descriptor for descriptor, _ in evidence.logits
    }
    if evidence.logits:
        clipped = {
            descriptor.unit_key: max(
                -_FROZEN_LOGIT_CLIP, min(_FROZEN_LOGIT_CLIP, logit)
            )
            for descriptor, logit in evidence.logits
        }
        maximum = max(clipped.values())
        exponentials = {
            key: math.exp(value - maximum) for key, value in clipped.items()
        }
        denominator = math.fsum(exponentials[key] for key in sorted(exponentials))
        probabilities = {
            key: exponentials[key] / denominator for key in sorted(exponentials)
        }
        unresolved = 0.0
    else:
        probabilities = {}
        unresolved = 1.0
    full_map = dict(probabilities)
    full_map[BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1] = unresolved
    rows = _probability_rows(probabilities, descriptor_map)
    body = {
        "recording_id": receipt.recording_id,
        "occurrence_id": receipt.occurrence_id,
        "canonical_event_id": receipt.canonical_event_id,
        "dedup_receipt_sha256": receipt.receipt_sha256,
        "processing_status": entry.processing_status,
        "qualification_status": qualification.event_aggregation_status,
        "qualification_receipt_sha256": qualification.receipt_sha256,
        "available_resolution_kinds": list(evidence.available_resolution_kinds),
        "typed_unit_probabilities": [row.to_dict() for row in rows],
        "unresolved_probability": unresolved,
        "earliest_typed_unit_keys": list(evidence.earliest_typed_unit_keys),
        "top_distribution_key": _top_key(full_map),
    }
    distribution_sha = _canonical_sha256(body)
    return (
        BAIEGPerOccurrenceDistributionV1(
            recording_id=receipt.recording_id,
            occurrence_id=receipt.occurrence_id,
            canonical_event_id=receipt.canonical_event_id,
            dedup_receipt_sha256=receipt.receipt_sha256,
            processing_status=entry.processing_status,
            qualification_status=qualification.event_aggregation_status,
            qualification_receipt_sha256=qualification.receipt_sha256,
            available_resolution_kinds=evidence.available_resolution_kinds,
            typed_unit_probabilities=rows,
            unresolved_probability=unresolved,
            earliest_typed_unit_keys=evidence.earliest_typed_unit_keys,
            top_distribution_key=body["top_distribution_key"],
            distribution_sha256=distribution_sha,
        ),
        descriptor_map,
    )


def _mixture_map(
    distributions: Sequence[BAIEGPerOccurrenceDistributionV1],
) -> dict[str, float]:
    if not distributions:
        return {}
    maps = [row.probability_map() for row in distributions]
    keys = sorted(set().union(*(mapping.keys() for mapping in maps)))
    denominator = float(len(maps))
    result = {
        key: math.fsum(mapping.get(key, 0.0) for mapping in maps) / denominator
        for key in keys
    }
    if abs(math.fsum(result.values()) - 1.0) > 1e-7:
        raise AssertionError("equal-occurrence mixture did not preserve unit mass")
    return result


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left).union(right))
    return 0.5 * math.fsum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    )


def _build_track(
    *,
    track_id: str,
    role: str,
    selected: Sequence[BAIEGPerOccurrenceDistributionV1],
    all_evaluable_ids: Sequence[str],
    descriptors: Mapping[str, _TypedUnitDescriptor],
    aggregation_status_when_empty: str,
    production_qualification_authorized: bool,
    complete_record_denominator_authorized: bool,
) -> BAIEGAggregationTrackV1:
    selected = tuple(sorted(selected, key=lambda row: row.occurrence_id))
    selected_ids = tuple(row.occurrence_id for row in selected)
    excluded_ids = tuple(sorted(set(all_evaluable_ids) - set(selected_ids)))
    if not selected:
        return BAIEGAggregationTrackV1(
            track_id=track_id,
            role=role,
            aggregation_status=aggregation_status_when_empty,
            included_occurrence_ids=(),
            excluded_occurrence_ids=excluded_ids,
            per_occurrence_distributions=(),
            equal_occurrence_mixture=(),
            unresolved_probability=None,
            top_distribution_key=None,
            leave_one_occurrence_out=(),
            pairwise_heterogeneity_shadow=(),
            concordant_top1_occurrence_count=0,
            discordant_top1_occurrence_count=0,
            discordant_event_pair_count=0,
            single_occurrence_dependence=None,
            single_occurrence_dependence_status="not_applicable_no_included_occurrence",
            production_qualification_authorized=production_qualification_authorized,
            complete_record_denominator_authorized=(
                complete_record_denominator_authorized
            ),
        )

    full_map = _mixture_map(selected)
    mixture_rows = _probability_rows(full_map, descriptors)
    unresolved = full_map.get(BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1, 0.0)
    full_top = _top_key(full_map)
    loeo_rows: list[BAIEGLeaveOneOccurrenceOutResultV1] = []
    dependence_values: list[float] = []
    if len(selected) == 1:
        loeo_rows.append(
            BAIEGLeaveOneOccurrenceOutResultV1(
                omitted_occurrence_id=selected[0].occurrence_id,
                status="not_evaluable_single_included_occurrence",
                typed_unit_probabilities=(),
                unresolved_probability=None,
                top_distribution_key=None,
                full_vs_loeo_total_variation=None,
                top1_changed=None,
            )
        )
        dependence = 1.0
        dependence_status = "complete_dependence_on_single_included_occurrence"
    else:
        for omitted in selected:
            retained = tuple(
                row for row in selected if row.occurrence_id != omitted.occurrence_id
            )
            loeo_map = _mixture_map(retained)
            distance = _total_variation(full_map, loeo_map)
            dependence_values.append(distance)
            loeo_top = _top_key(loeo_map)
            loeo_rows.append(
                BAIEGLeaveOneOccurrenceOutResultV1(
                    omitted_occurrence_id=omitted.occurrence_id,
                    status="evaluated",
                    typed_unit_probabilities=_probability_rows(loeo_map, descriptors),
                    unresolved_probability=loeo_map.get(
                        BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1, 0.0
                    ),
                    top_distribution_key=loeo_top,
                    full_vs_loeo_total_variation=distance,
                    top1_changed=loeo_top != full_top,
                )
            )
        dependence = max(dependence_values)
        dependence_status = "maximum_full_vs_leave_one_occurrence_out_tv"

    pairwise: list[BAIEGTargetFreePairwiseHeterogeneityV1] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            left_earliest = set(left.earliest_typed_unit_keys)
            right_earliest = set(right.earliest_typed_unit_keys)
            if left_earliest and right_earliest:
                union = left_earliest.union(right_earliest)
                jaccard = len(left_earliest.intersection(right_earliest)) / len(union)
                overlap_status = "evaluated"
            else:
                jaccard = None
                overlap_status = "not_evaluable_missing_earliest_field"
            pairwise.append(
                BAIEGTargetFreePairwiseHeterogeneityV1(
                    left_occurrence_id=left.occurrence_id,
                    right_occurrence_id=right.occurrence_id,
                    total_variation_distance=_total_variation(
                        left.probability_map(), right.probability_map()
                    ),
                    top1_discordant=(
                        left.top_distribution_key != right.top_distribution_key
                    ),
                    earliest_field_overlap_status=overlap_status,
                    earliest_field_jaccard=jaccard,
                )
            )
    concordant = sum(row.top_distribution_key == full_top for row in selected)
    return BAIEGAggregationTrackV1(
        track_id=track_id,
        role=role,
        aggregation_status=(
            "evaluable_without_resolved_typed_distribution"
            if unresolved >= 1.0 - _TOL
            else "aggregated_equal_occurrence_mixture"
        ),
        included_occurrence_ids=selected_ids,
        excluded_occurrence_ids=excluded_ids,
        per_occurrence_distributions=selected,
        equal_occurrence_mixture=mixture_rows,
        unresolved_probability=unresolved,
        top_distribution_key=full_top,
        leave_one_occurrence_out=tuple(loeo_rows),
        pairwise_heterogeneity_shadow=tuple(pairwise),
        concordant_top1_occurrence_count=concordant,
        discordant_top1_occurrence_count=len(selected) - concordant,
        discordant_event_pair_count=sum(row.top1_discordant for row in pairwise),
        single_occurrence_dependence=dependence,
        single_occurrence_dependence_status=dependence_status,
        production_qualification_authorized=production_qualification_authorized,
        complete_record_denominator_authorized=complete_record_denominator_authorized,
    )


class BAIEGCompleteITAMultiEventAggregatorV1:
    """Reassemble arbitrary typed-head microbatches into complete record bags."""

    implementation_id: Final[str] = BA_IEG_COMPLETE_ITA_MULTIEVENT_AGGREGATION_ID_V1

    def __init__(self, *, allow_component_test_qualification: bool = False) -> None:
        if not isinstance(allow_component_test_qualification, bool):
            raise TypeError("component-test qualification gate must be boolean")
        self.allow_component_test_qualification = allow_component_test_qualification

    def __call__(
        self,
        event_outputs: Sequence[BAIEGShallowCausalTypedUnitHeadOutput],
        manifest: BAIEGCompleteITARecordRosterManifestV1,
    ) -> BAIEGCompleteITAMultiEventAggregationOutputV1:
        return self.aggregate(event_outputs=event_outputs, manifest=manifest)

    def aggregate(
        self,
        *,
        event_outputs: Sequence[BAIEGShallowCausalTypedUnitHeadOutput],
        manifest: BAIEGCompleteITARecordRosterManifestV1,
    ) -> BAIEGCompleteITAMultiEventAggregationOutputV1:
        if not isinstance(manifest, BAIEGCompleteITARecordRosterManifestV1):
            raise TypeError("complete ITA requires a complete typed record manifest")
        if isinstance(event_outputs, BAIEGShallowCausalTypedUnitHeadOutput):
            raise TypeError("event_outputs must be a sequence of microbatch outputs")
        projected = _project_microbatches(tuple(event_outputs))
        fragment_registry: dict[str, tuple[BAIEGCompleteITAOccurrenceEntryV1, str]] = {}
        for entry in manifest.occurrence_entries:
            receipt = entry.dedup_receipt
            for event_id, source_receipt in zip(
                receipt.fragment_event_ids,
                receipt.fragment_source_event_receipt_sha256s,
            ):
                fragment_registry[event_id] = (entry, source_receipt)
        unexpected = sorted(set(projected) - set(fragment_registry))
        if unexpected:
            raise ValueError(
                f"model outputs contain events outside the complete roster: {unexpected}"
            )
        for event_id, evidence in projected.items():
            entry, expected_source_receipt = fragment_registry[event_id]
            if evidence.recording_id != entry.dedup_receipt.recording_id or (
                evidence.source_event_receipt_sha256 != expected_source_receipt
            ):
                raise ValueError(
                    "model event identity disagrees with the complete roster"
                )

        record_outputs: list[BAIEGCompleteITARecordOutputV1] = []
        count_by_record = dict(
            zip(manifest.recording_ids, manifest.candidate_fragment_counts)
        )
        for recording_id in manifest.recording_ids:
            entries = tuple(
                entry
                for entry in manifest.occurrence_entries
                if entry.dedup_receipt.recording_id == recording_id
            )
            distributions: list[BAIEGPerOccurrenceDistributionV1] = []
            descriptors: dict[str, _TypedUnitDescriptor] = {}
            for entry in entries:
                if not entry.eeg_evaluable:
                    continue
                canonical_id = entry.dedup_receipt.canonical_event_id
                evidence = projected.get(canonical_id)
                if evidence is None:
                    raise ValueError(
                        "an EEG-evaluable canonical occurrence is missing model evidence"
                    )
                distribution, event_descriptors = _event_distribution(entry, evidence)
                for key, descriptor in event_descriptors.items():
                    previous = descriptors.setdefault(key, descriptor)
                    if previous != descriptor:
                        raise AssertionError(
                            "typed-unit key semantics drifted across events"
                        )
                distributions.append(distribution)
            distributions = sorted(distributions, key=lambda row: row.occurrence_id)
            all_evaluable_ids = tuple(row.occurrence_id for row in distributions)
            qualified_distributions = tuple(
                row
                for row in distributions
                if row.qualification_status == "qualified_ictal"
            )
            qualified_entries = tuple(
                entry
                for entry in entries
                if entry.eeg_evaluable
                and entry.qualification_status == "qualified_ictal"
            )
            qualified_production_authorized = bool(
                qualified_entries
                and all(
                    entry.qualification_receipt is not None
                    and entry.qualification_receipt.evidence_grade_qualification_authorized
                    for entry in qualified_entries
                )
            )
            qualified_component_only = bool(
                qualified_entries
                and not qualified_production_authorized
                and all(
                    entry.qualification_receipt is not None
                    and entry.qualification_receipt.component_test_only
                    for entry in qualified_entries
                )
            )
            if qualified_production_authorized or (
                qualified_component_only and self.allow_component_test_qualification
            ):
                secondary_selected = qualified_distributions
                empty_secondary_status = "no_qualified_occurrence_secondary_empty"
            else:
                secondary_selected = ()
                empty_secondary_status = (
                    "nonproduction_qualification_not_authorized_secondary_empty"
                    if qualified_distributions
                    else "no_qualified_occurrence_secondary_empty"
                )

            ita = _build_track(
                track_id=BA_IEG_ITA_PRIMARY_TRACK_ID_V1,
                role="complete_record_intention_to_analyze_primary_research_rank",
                selected=distributions,
                all_evaluable_ids=all_evaluable_ids,
                descriptors=descriptors,
                aggregation_status_when_empty=(
                    "zero_candidate_record"
                    if not entries
                    else "no_eeg_evaluable_occurrence"
                ),
                production_qualification_authorized=False,
                complete_record_denominator_authorized=True,
            )
            secondary = _build_track(
                track_id=BA_IEG_QUALIFIED_ONLY_SECONDARY_TRACK_ID_V1,
                role=(
                    "qualified_only_safety_secondary_not_a_complete_record_substitute"
                ),
                selected=secondary_selected,
                all_evaluable_ids=all_evaluable_ids,
                descriptors=descriptors,
                aggregation_status_when_empty=empty_secondary_status,
                production_qualification_authorized=qualified_production_authorized,
                complete_record_denominator_authorized=False,
            )
            status_counts = {
                status: sum(entry.processing_status == status for entry in entries)
                for status in BA_IEG_PROCESSING_STATUSES_V1
            }
            qualified_count = sum(
                entry.eeg_evaluable and entry.qualification_status == "qualified_ictal"
                for entry in entries
            )
            uncertain_count = sum(
                entry.eeg_evaluable and entry.qualification_status == "uncertain"
                for entry in entries
            )
            nonproduction_qualified = sum(
                entry.eeg_evaluable
                and entry.qualification_status == "qualified_ictal"
                and entry.qualification_receipt is not None
                and not entry.qualification_receipt.evidence_grade_qualification_authorized
                for entry in entries
            )
            resolved_count = sum(
                row.unresolved_probability < 1.0 - _TOL for row in distributions
            )
            candidate_count = count_by_record[recording_id]
            denominator = BAIEGRecordITADenominatorV1(
                recording_id=recording_id,
                candidate_fragment_count=candidate_count,
                unique_occurrence_count=len(entries),
                duplicate_fragment_count=candidate_count - len(entries),
                complete_occurrence_count=status_counts["complete"],
                partial_occurrence_count=status_counts["partial"],
                failure_occurrence_count=status_counts["failure"],
                not_evaluable_occurrence_count=status_counts["not_evaluable"],
                eeg_evaluable_occurrence_count=len(distributions),
                qualified_evaluable_occurrence_count=qualified_count,
                uncertain_evaluable_occurrence_count=uncertain_count,
                resolved_typed_distribution_count=resolved_count,
                unresolved_typed_distribution_count=len(distributions) - resolved_count,
                nonproduction_qualified_occurrence_count=nonproduction_qualified,
                zero_candidate_record=candidate_count == 0,
            )
            record_body = {
                "recording_id": recording_id,
                "denominator": denominator.to_dict(),
                "occurrence_dedup_receipt_sha256s": [
                    entry.dedup_receipt.receipt_sha256 for entry in entries
                ],
                "ita_primary": ita.to_dict(),
                "qualified_only_secondary": secondary.to_dict(),
                "tracks_are_separate_estimands": True,
            }
            record_sha = _canonical_sha256(record_body)
            record_outputs.append(
                BAIEGCompleteITARecordOutputV1(
                    recording_id=recording_id,
                    denominator=denominator,
                    occurrence_dedup_receipt_sha256s=tuple(
                        entry.dedup_receipt.receipt_sha256 for entry in entries
                    ),
                    ita_primary=ita,
                    qualified_only_secondary=secondary,
                    record_output_sha256=record_sha,
                )
            )

        output_body = {
            "implementation_id": self.implementation_id,
            "source_manifest_sha256": manifest.manifest_sha256,
            "records": [record.to_dict() for record in record_outputs],
            "inference_scope": deepcopy(_EEG_ONLY_INFERENCE_SCOPE),
            "output_semantics": (
                "complete_record_ita_scalp_visible_typed_onset_research_distribution_"
                "not_cortical_soz_or_ez_not_clinical_diagnosis"
            ),
        }
        return BAIEGCompleteITAMultiEventAggregationOutputV1(
            implementation_id=self.implementation_id,
            source_manifest_sha256=manifest.manifest_sha256,
            records=tuple(record_outputs),
            output_sha256=_canonical_sha256(output_body),
        )


__all__ = [
    "BA_IEG_COMPLETE_ITA_MULTIEVENT_AGGREGATION_ID_V1",
    "BA_IEG_REFERENCE_FREE_OCCURRENCE_DEDUP_SCHEMA_V1",
    "BA_IEG_COMPLETE_ITA_MANIFEST_SCHEMA_V1",
    "BA_IEG_ITA_PRIMARY_TRACK_ID_V1",
    "BA_IEG_QUALIFIED_ONLY_SECONDARY_TRACK_ID_V1",
    "BA_IEG_PROCESSING_STATUSES_V1",
    "BA_IEG_UNRESOLVED_TYPED_UNIT_KEY_V1",
    "BAIEGReferenceFreeOccurrenceDedupReceiptV1",
    "BAIEGCompleteITAOccurrenceEntryV1",
    "BAIEGCompleteITARecordRosterManifestV1",
    "BAIEGTypedUnitProbabilityV1",
    "BAIEGPerOccurrenceDistributionV1",
    "BAIEGLeaveOneOccurrenceOutResultV1",
    "BAIEGTargetFreePairwiseHeterogeneityV1",
    "BAIEGAggregationTrackV1",
    "BAIEGRecordITADenominatorV1",
    "BAIEGCompleteITARecordOutputV1",
    "BAIEGCompleteITAMultiEventAggregationOutputV1",
    "BAIEGCompleteITAMultiEventAggregatorV1",
]
