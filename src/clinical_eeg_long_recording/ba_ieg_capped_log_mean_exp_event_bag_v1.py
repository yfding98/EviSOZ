"""Deduplicated event-to-record aggregation for the BA-IEG v1 core.

The aggregator is deliberately label-free and record-local.  Exact event
copies must carry one target-independent occurrence-equivalence identifier;
they are checked for identical masks/logits and contribute exactly once.
Distinct evaluable occurrences then receive equal weight under the frozen
clip[-12,12], temperature-1 capped log-mean-exp rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Final

import torch
from torch import nn

from src.soz.geometry import STANDARD_19

from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
    ba_ieg_event_identity_roster_sha256,
)
from .ba_ieg_training_contract import BAIEGCollatedEventBatch
from .ba_ieg_target_free_event_qualification_v1 import (
    BA_IEG_EVENT_QUALIFICATION_STATUSES,
    BA_IEG_LEGACY_CALLER_STATUS_AUTHORITY,
    BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1,
    BAIEGTargetFreeEventQualificationReceiptV1,
)


BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID: Final[str] = (
    "ba_ieg_capped_log_mean_exp_event_bag_v1"
)
BA_IEG_EVENT_AGGREGATION_STATUSES: Final[tuple[str, ...]] = (
    BA_IEG_EVENT_QUALIFICATION_STATUSES
)


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


@dataclass(frozen=True)
class BAIEGRecordEventBagManifest:
    """Target-independent event/occurrence roster aligned to one model batch."""

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    source_event_receipt_sha256s: tuple[str, ...]
    occurrence_equivalence_ids: tuple[str, ...]
    event_aggregation_statuses: tuple[str, ...]
    event_qualification_receipts: tuple[
        BAIEGTargetFreeEventQualificationReceiptV1, ...
    ] = ()
    equivalence_authority: str = (
        "target_free_detector_candidate_occurrence_receipt_v1"
    )
    event_qualification_authority: str = BA_IEG_LEGACY_CALLER_STATUS_AUTHORITY
    target_conditioned_deduplication: bool = False
    identity_roster_sha256: str = field(init=False)
    event_qualification_receipt_sha256s: tuple[str, ...] = field(init=False)
    qualification_provider_validated: bool = field(init=False)
    evidence_grade_aggregation_authorized: bool = field(init=False)
    component_test_qualification_present: bool = field(init=False)

    def __post_init__(self) -> None:
        if len(self.source_input_batch_sha256) != 64:
            raise ValueError("record event-bag manifest needs a batch SHA-256")
        events = tuple(_identifier(value, "event_id") for value in self.event_ids)
        records = tuple(
            _identifier(value, "recording_id") for value in self.recording_ids
        )
        occurrences = tuple(
            _identifier(value, "occurrence_equivalence_id")
            for value in self.occurrence_equivalence_ids
        )
        statuses = tuple(str(value) for value in self.event_aggregation_statuses)
        if not events or len(set(events)) != len(events):
            raise ValueError("event-bag event IDs must be non-empty and unique")
        receipts = tuple(self.source_event_receipt_sha256s)
        if (
            len(records) != len(events)
            or len(receipts) != len(events)
            or len(occurrences) != len(events)
            or len(statuses) != len(events)
        ):
            raise ValueError(
                "event/record/receipt/occurrence/status rosters must align"
            )
        if any(value not in BA_IEG_EVENT_AGGREGATION_STATUSES for value in statuses):
            raise ValueError("event aggregation status is unsupported")
        if any(len(value) != 64 for value in receipts):
            raise ValueError("event-bag source event receipts must be SHA-256 values")
        qualification_receipts = tuple(self.event_qualification_receipts)
        provider_validated = bool(qualification_receipts)
        if provider_validated:
            if len(qualification_receipts) != len(events):
                raise ValueError(
                    "event qualification receipts must align with the event roster"
                )
            if self.event_qualification_authority != (
                BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
            ):
                raise ValueError(
                    "content-bound qualification receipts require their typed provider"
                )
            for index, qualification_receipt in enumerate(
                qualification_receipts
            ):
                if not isinstance(
                    qualification_receipt,
                    BAIEGTargetFreeEventQualificationReceiptV1,
                ):
                    raise TypeError(
                        "event qualification receipts must be typed provider outputs"
                    )
                observation = qualification_receipt.observation
                expected_identity = (
                    events[index],
                    records[index],
                    receipts[index],
                    occurrences[index],
                    self.source_input_batch_sha256,
                )
                observed_identity = (
                    observation.event_id,
                    observation.recording_id,
                    observation.source_event_receipt_sha256,
                    qualification_receipt.occurrence_equivalence_id,
                    observation.source_input_batch_sha256,
                )
                if observed_identity != expected_identity:
                    raise ValueError(
                        "event qualification receipt identity/order roster drifted"
                    )
                if qualification_receipt.event_aggregation_status != statuses[index]:
                    raise ValueError(
                        "caller status disagrees with content-bound qualification receipt"
                    )
        elif self.event_qualification_authority != (
            BA_IEG_LEGACY_CALLER_STATUS_AUTHORITY
        ):
            raise ValueError(
                "old caller-status path cannot claim a legitimate qualification provider"
            )
        occurrence_record: dict[str, str] = {}
        occurrence_status: dict[str, str] = {}
        for occurrence, record, status in zip(occurrences, records, statuses):
            previous = occurrence_record.setdefault(occurrence, record)
            if previous != record:
                raise ValueError("one occurrence equivalence class crosses records")
            previous_status = occurrence_status.setdefault(occurrence, status)
            if previous_status != status:
                raise ValueError(
                    "one occurrence equivalence class has conflicting aggregation status"
                )
        if self.equivalence_authority != (
            "target_free_detector_candidate_occurrence_receipt_v1"
        ) or self.target_conditioned_deduplication is not False:
            raise ValueError("event deduplication must remain target-independent")
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "recording_ids", records)
        object.__setattr__(self, "source_event_receipt_sha256s", receipts)
        object.__setattr__(self, "occurrence_equivalence_ids", occurrences)
        object.__setattr__(self, "event_aggregation_statuses", statuses)
        object.__setattr__(
            self, "event_qualification_receipts", qualification_receipts
        )
        object.__setattr__(
            self,
            "event_qualification_receipt_sha256s",
            tuple(
                receipt.receipt_sha256 for receipt in qualification_receipts
            ),
        )
        object.__setattr__(
            self, "qualification_provider_validated", provider_validated
        )
        evidence_grade_authorized = bool(
            provider_validated
            and all(
                receipt.evidence_grade_qualification_authorized
                for receipt in qualification_receipts
            )
        )
        object.__setattr__(
            self,
            "evidence_grade_aggregation_authorized",
            evidence_grade_authorized,
        )
        object.__setattr__(
            self,
            "component_test_qualification_present",
            bool(
                provider_validated
                and any(
                    receipt.component_test_only
                    for receipt in qualification_receipts
                )
            ),
        )
        object.__setattr__(
            self,
            "identity_roster_sha256",
            ba_ieg_event_identity_roster_sha256(
                source_input_batch_sha256=self.source_input_batch_sha256,
                event_ids=events,
                recording_ids=records,
                source_event_receipt_sha256s=receipts,
            ),
        )

    @classmethod
    def from_batch(
        cls,
        batch: BAIEGCollatedEventBatch,
        *,
        occurrence_equivalence_ids: tuple[str, ...],
        event_aggregation_statuses: tuple[str, ...],
    ) -> "BAIEGRecordEventBagManifest":
        if not isinstance(batch, BAIEGCollatedEventBatch):
            raise TypeError("record event-bag manifest requires a registered batch")
        return cls(
            source_input_batch_sha256=batch.input_batch_sha256,
            event_ids=batch.event_ids,
            recording_ids=batch.recording_ids,
            source_event_receipt_sha256s=batch.input_event_receipt_sha256s,
            occurrence_equivalence_ids=occurrence_equivalence_ids,
            event_aggregation_statuses=event_aggregation_statuses,
        )

    @classmethod
    def from_target_free_qualification_receipts(
        cls,
        batch: BAIEGCollatedEventBatch,
        *,
        occurrence_equivalence_ids: tuple[str, ...],
        event_qualification_receipts: tuple[
            BAIEGTargetFreeEventQualificationReceiptV1, ...
        ],
    ) -> "BAIEGRecordEventBagManifest":
        """Build the only evidence-grade manifest path.

        Statuses are projected from content-bound provider receipts.  There is
        no status argument for a caller to fill.
        """

        if not isinstance(batch, BAIEGCollatedEventBatch):
            raise TypeError("record event-bag manifest requires a registered batch")
        receipts = tuple(event_qualification_receipts)
        return cls(
            source_input_batch_sha256=batch.input_batch_sha256,
            event_ids=batch.event_ids,
            recording_ids=batch.recording_ids,
            source_event_receipt_sha256s=batch.input_event_receipt_sha256s,
            occurrence_equivalence_ids=occurrence_equivalence_ids,
            event_aggregation_statuses=tuple(
                receipt.event_aggregation_status for receipt in receipts
            ),
            event_qualification_receipts=receipts,
            event_qualification_authority=(
                BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
            ),
        )


@dataclass(frozen=True)
class BAIEGCappedLogMeanExpEventBagOutput:
    source_input_batch_sha256: str
    implementation_id: str
    recording_ids: tuple[str, ...]
    typed_unit_record_logits: torch.Tensor
    typed_unit_mask: torch.Tensor
    typed_unit_kind_index: torch.Tensor
    typed_unit_electrode_index: torch.Tensor
    typed_unit_lead_endpoint_index: torch.Tensor
    physical_electrode_record_logits: torch.Tensor
    physical_electrode_mask: torch.Tensor
    unique_occurrence_count: torch.Tensor
    qualified_unique_occurrence_count: torch.Tensor
    uncertain_unique_occurrence_count: torch.Tensor
    not_evaluable_unique_occurrence_count: torch.Tensor
    copied_event_count: torch.Tensor
    temperature: float
    symmetric_event_logit_clip: float
    qualification_provider_validated: bool
    evidence_grade_aggregation_authorized: bool
    event_qualification_receipt_sha256s: tuple[str, ...]
    aggregation_authority: str
    output_semantics: str = (
        "record_level_scalp_visible_onset_candidate_logits_not_clinical_probabilities"
    )
    region_laterality_projection_status: str = (
        "not_implemented_requires_deterministic_conservative_projection"
    )


class BAIEGCappedLogMeanExpEventBag(nn.Module):
    """Frozen equal-occurrence, permutation-invariant record aggregator."""

    implementation_id: Final[str] = BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        symmetric_event_logit_clip: float = 12.0,
        allow_legacy_component_test_only: bool = False,
        allow_component_test_qualification: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("event-bag temperature must be positive")
        if (
            not math.isfinite(symmetric_event_logit_clip)
            or symmetric_event_logit_clip <= 0.0
        ):
            raise ValueError("event-bag logit clip must be positive")
        # v1 is frozen; alternate values belong in a separately named ablation.
        if temperature != 1.0 or symmetric_event_logit_clip != 12.0:
            raise ValueError("BA-IEG v1 event-bag policy is frozen at tau=1, clip=12")
        self.temperature = float(temperature)
        self.symmetric_event_logit_clip = float(symmetric_event_logit_clip)
        if not isinstance(allow_legacy_component_test_only, bool):
            raise TypeError("legacy component-test gate must be boolean")
        if not isinstance(allow_component_test_qualification, bool):
            raise TypeError("component-test qualification gate must be boolean")
        self.allow_legacy_component_test_only = allow_legacy_component_test_only
        self.allow_component_test_qualification = allow_component_test_qualification

    def forward(
        self,
        event_output: BAIEGShallowCausalTypedUnitHeadOutput,
        manifest: BAIEGRecordEventBagManifest,
    ) -> BAIEGCappedLogMeanExpEventBagOutput:
        if not isinstance(event_output, BAIEGShallowCausalTypedUnitHeadOutput):
            raise TypeError("event-bag aggregation requires typed-unit head output")
        if not isinstance(manifest, BAIEGRecordEventBagManifest):
            raise TypeError("event-bag aggregation requires a frozen manifest")
        if event_output.source_input_batch_sha256 != manifest.source_input_batch_sha256:
            raise ValueError("event-bag manifest is bound to another model batch")
        if (
            event_output.event_ids != manifest.event_ids
            or event_output.recording_ids != manifest.recording_ids
            or event_output.source_event_receipt_sha256s
            != manifest.source_event_receipt_sha256s
            or event_output.identity_roster_sha256
            != manifest.identity_roster_sha256
        ):
            raise ValueError("event-bag manifest identity/order roster drifted")
        if (
            not manifest.qualification_provider_validated
            and not self.allow_legacy_component_test_only
        ):
            raise ValueError(
                "evidence-grade event bag requires content-bound target-free "
                "qualification receipts"
            )
        if (
            manifest.qualification_provider_validated
            and not manifest.evidence_grade_aggregation_authorized
            and not self.allow_component_test_qualification
        ):
            raise ValueError(
                "non-production qualification receipts cannot enter the "
                "evidence-grade event bag"
            )
        physical_logits = event_output.physical_electrode_event_logits
        physical_mask = event_output.physical_electrode_mask
        typed_logits = event_output.typed_unit_event_logits
        typed_mask = event_output.typed_unit_mask
        typed_kind = event_output.typed_unit_kind_index
        typed_electrode = event_output.typed_unit_electrode_index
        typed_lead = event_output.typed_unit_lead_endpoint_index
        event_count = len(manifest.event_ids)
        if tuple(physical_logits.shape) != (
            event_count,
            len(STANDARD_19),
        ) or tuple(physical_mask.shape) != tuple(physical_logits.shape):
            raise ValueError("event electrode logits/mask do not align with manifest")
        if typed_logits.ndim != 2 or int(typed_logits.shape[0]) != event_count:
            raise ValueError("event typed-unit logits do not align with manifest")
        maximum_event_typed = int(typed_logits.shape[1])
        if (
            tuple(typed_mask.shape) != tuple(typed_logits.shape)
            or tuple(typed_kind.shape) != tuple(typed_logits.shape)
            or tuple(typed_electrode.shape) != tuple(typed_logits.shape)
            or tuple(typed_lead.shape) != (event_count, maximum_event_typed, 2)
        ):
            raise ValueError("event typed-unit identity tensors are not aligned")
        if not typed_logits.is_floating_point() or not torch.isfinite(
            typed_logits
        ).all() or not torch.isfinite(physical_logits).all():
            raise ValueError("event electrode logits must be finite floating point")
        if typed_mask.dtype != torch.bool or physical_mask.dtype != torch.bool:
            raise ValueError("event opportunity masks must be boolean")

        electrode_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index(
            "physical_electrode"
        )
        lead_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("bipolar_lead")

        def typed_key(event_index: int, typed_index: int) -> tuple[int, int, int]:
            kind = int(typed_kind[event_index, typed_index])
            if kind == electrode_kind:
                electrode = int(typed_electrode[event_index, typed_index])
                if electrode < 0 or electrode >= len(STANDARD_19):
                    raise ValueError("physical typed unit has an invalid electrode")
                return (kind, electrode, -1)
            if kind == lead_kind:
                first, second = (
                    int(value) for value in typed_lead[event_index, typed_index]
                )
                if (
                    first < 0
                    or second < 0
                    or first >= len(STANDARD_19)
                    or second >= len(STANDARD_19)
                    or first >= second
                ):
                    raise ValueError("bipolar typed unit has invalid endpoints")
                return (kind, first, second)
            raise ValueError("event output contains an unsupported typed-unit kind")

        for event_index in range(event_count):
            seen: set[tuple[int, int, int]] = set()
            reconstructed_mask = torch.zeros(
                len(STANDARD_19), dtype=torch.bool, device=typed_logits.device
            )
            reconstructed_logits = torch.zeros(
                len(STANDARD_19),
                dtype=typed_logits.dtype,
                device=typed_logits.device,
            )
            for typed_index in torch.nonzero(
                typed_mask[event_index], as_tuple=False
            ).flatten():
                local_index = int(typed_index)
                key = typed_key(event_index, local_index)
                if key in seen:
                    raise ValueError("event output repeats one typed-unit identity")
                seen.add(key)
                if key[0] == electrode_kind:
                    reconstructed_mask[key[1]] = True
                    reconstructed_logits[key[1]] = typed_logits[
                        event_index, local_index
                    ]
            if not torch.equal(reconstructed_mask, physical_mask[event_index]) or not (
                torch.equal(
                    reconstructed_logits[reconstructed_mask].detach(),
                    physical_logits[event_index, reconstructed_mask].detach(),
                )
            ):
                raise ValueError("event physical projection disagrees with typed units")

        records = tuple(sorted(set(manifest.recording_ids)))
        record_key_sets: list[set[tuple[int, int, int]]] = []
        for recording_id in records:
            keys: set[tuple[int, int, int]] = set()
            for event_index, record in enumerate(manifest.recording_ids):
                if record != recording_id or manifest.event_aggregation_statuses[
                    event_index
                ] != "qualified_ictal":
                    continue
                for typed_index in torch.nonzero(
                    typed_mask[event_index], as_tuple=False
                ).flatten():
                    keys.add(typed_key(event_index, int(typed_index)))
            record_key_sets.append(keys)
        maximum_record_typed = max(1, *(len(keys) for keys in record_key_sets))
        record_typed_logits = torch.zeros(
            (len(records), maximum_record_typed),
            dtype=typed_logits.dtype,
            device=typed_logits.device,
        )
        record_typed_mask = torch.zeros_like(record_typed_logits, dtype=torch.bool)
        record_typed_kind = torch.full(
            (len(records), maximum_record_typed),
            -1,
            dtype=torch.long,
            device=typed_logits.device,
        )
        record_typed_electrode = torch.full_like(record_typed_kind, -1)
        record_typed_lead = torch.full(
            (len(records), maximum_record_typed, 2),
            -1,
            dtype=torch.long,
            device=typed_logits.device,
        )
        record_logits = torch.zeros(
            (len(records), len(STANDARD_19)),
            dtype=typed_logits.dtype,
            device=typed_logits.device,
        )
        record_mask = torch.zeros_like(record_logits, dtype=torch.bool)
        unique_counts = torch.zeros(
            len(records), dtype=torch.long, device=typed_logits.device
        )
        qualified_counts = torch.zeros_like(unique_counts)
        uncertain_counts = torch.zeros_like(unique_counts)
        not_evaluable_counts = torch.zeros_like(unique_counts)
        copied_counts = torch.zeros_like(unique_counts)

        for record_index, recording_id in enumerate(records):
            equivalence_groups: dict[str, list[int]] = {}
            for event_index, (record, occurrence) in enumerate(
                zip(
                    manifest.recording_ids,
                    manifest.occurrence_equivalence_ids,
                )
            ):
                if record == recording_id:
                    equivalence_groups.setdefault(occurrence, []).append(event_index)
            representatives: list[int] = []
            for occurrence in sorted(equivalence_groups):
                indices = sorted(
                    equivalence_groups[occurrence],
                    key=lambda index: manifest.event_ids[index],
                )
                representative = indices[0]
                for duplicate in indices[1:]:
                    if (
                        not torch.equal(
                            typed_mask[duplicate], typed_mask[representative]
                        )
                        or not torch.equal(
                            typed_kind[duplicate], typed_kind[representative]
                        )
                        or not torch.equal(
                            typed_electrode[duplicate],
                            typed_electrode[representative],
                        )
                        or not torch.equal(
                            typed_lead[duplicate], typed_lead[representative]
                        )
                        or not (
                        torch.equal(
                            typed_logits[duplicate].detach(),
                            typed_logits[representative].detach(),
                        )
                        )
                    ):
                        raise ValueError(
                            "one occurrence equivalence class has non-identical model evidence"
                        )
                representatives.append(representative)
                copied_counts[record_index] += len(indices) - 1
            unique_counts[record_index] = len(representatives)
            status_counts = {
                status: sum(
                    manifest.event_aggregation_statuses[index] == status
                    for index in representatives
                )
                for status in BA_IEG_EVENT_AGGREGATION_STATUSES
            }
            qualified_counts[record_index] = status_counts["qualified_ictal"]
            uncertain_counts[record_index] = status_counts["uncertain"]
            not_evaluable_counts[record_index] = status_counts["not_evaluable"]
            representatives = [
                index
                for index in representatives
                if manifest.event_aggregation_statuses[index]
                == "qualified_ictal"
            ]
            if not representatives:
                continue
            representative_index = torch.tensor(
                representatives, dtype=torch.long, device=typed_logits.device
            )
            values_by_key: dict[tuple[int, int, int], list[torch.Tensor]] = {}
            for event_index_tensor in representative_index:
                event_index = int(event_index_tensor)
                for typed_index_tensor in torch.nonzero(
                    typed_mask[event_index], as_tuple=False
                ).flatten():
                    typed_index = int(typed_index_tensor)
                    key = typed_key(event_index, typed_index)
                    values_by_key.setdefault(key, []).append(
                        typed_logits[event_index, typed_index]
                    )
            for record_typed_index, key in enumerate(sorted(values_by_key)):
                values = torch.stack(values_by_key[key]).clamp(
                    min=-self.symmetric_event_logit_clip,
                    max=self.symmetric_event_logit_clip,
                ) / self.temperature
                aggregate = self.temperature * (
                    torch.logsumexp(values, dim=0) - math.log(len(values))
                )
                record_typed_logits[record_index, record_typed_index] = aggregate
                record_typed_mask[record_index, record_typed_index] = True
                record_typed_kind[record_index, record_typed_index] = key[0]
                if key[0] == electrode_kind:
                    record_typed_electrode[record_index, record_typed_index] = key[1]
                    record_logits[record_index, key[1]] = aggregate
                    record_mask[record_index, key[1]] = True
                else:
                    record_typed_lead[record_index, record_typed_index] = torch.tensor(
                        key[1:], dtype=torch.long, device=typed_logits.device
                    )

        return BAIEGCappedLogMeanExpEventBagOutput(
            source_input_batch_sha256=manifest.source_input_batch_sha256,
            implementation_id=self.implementation_id,
            recording_ids=records,
            typed_unit_record_logits=record_typed_logits,
            typed_unit_mask=record_typed_mask,
            typed_unit_kind_index=record_typed_kind,
            typed_unit_electrode_index=record_typed_electrode,
            typed_unit_lead_endpoint_index=record_typed_lead,
            physical_electrode_record_logits=record_logits,
            physical_electrode_mask=record_mask,
            unique_occurrence_count=unique_counts,
            qualified_unique_occurrence_count=qualified_counts,
            uncertain_unique_occurrence_count=uncertain_counts,
            not_evaluable_unique_occurrence_count=not_evaluable_counts,
            copied_event_count=copied_counts,
            temperature=self.temperature,
            symmetric_event_logit_clip=self.symmetric_event_logit_clip,
            qualification_provider_validated=(
                manifest.qualification_provider_validated
            ),
            evidence_grade_aggregation_authorized=(
                manifest.evidence_grade_aggregation_authorized
            ),
            event_qualification_receipt_sha256s=(
                manifest.event_qualification_receipt_sha256s
            ),
            aggregation_authority=(
                "content_bound_target_free_event_qualification_receipts"
                if manifest.evidence_grade_aggregation_authorized
                else "component_test_only_non_production_qualification_receipts"
                if manifest.qualification_provider_validated
                else "legacy_component_test_only_unverified_status_vector"
            ),
        )


__all__ = [
    "BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID",
    "BA_IEG_EVENT_AGGREGATION_STATUSES",
    "BAIEGRecordEventBagManifest",
    "BAIEGCappedLogMeanExpEventBagOutput",
    "BAIEGCappedLogMeanExpEventBag",
]
