"""Complete-patient collation for evidence-only SOZ supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from ..aggregation import PatientAggregation, aggregate_patient_logits
from ..evidence import EvidenceBatch
from .deepsoz import DeepSOZReferenceRegistry, normalize_patient_id
from .provenance import (
    EvidenceCacheReceipt,
    EventInputRegistry,
    evidence_batch_sha256,
)


@dataclass(frozen=True)
class EvidenceEvent:
    """One detached evidence tensor bound to a registry-issued event receipt.

    Patient identity, source, and split are intentionally absent. They are
    resolved from :class:`EventInputRegistry` during collation.
    """

    event_id: str
    evidence: EvidenceBatch
    cache_receipt: EvidenceCacheReceipt

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id cannot be empty")
        object.__setattr__(self, "event_id", event_id)
        if self.evidence.batch_size != 1:
            raise ValueError("Each EvidenceEvent must contain exactly one event")
        if self.evidence.node.requires_grad or self.evidence.edge.requires_grad:
            raise ValueError("Cached evidence must be detached before registration")
        if self.cache_receipt.event_id != event_id:
            raise ValueError("Evidence event ID does not match its cache receipt")
        if self.cache_receipt.evidence_sha256 != evidence_batch_sha256(self.evidence):
            raise ValueError("Evidence tensor content does not match its cache receipt")


@dataclass(frozen=True)
class PatientEvidenceBatch:
    """Flat events plus exactly one target and complete-event count per patient."""

    evidence: EvidenceBatch
    event_patient_index: torch.Tensor
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    expected_event_counts: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    event_registry_sha256: str

    def __post_init__(self) -> None:
        n_events = self.evidence.batch_size
        n_patients = len(self.patient_ids)
        if tuple(self.event_patient_index.shape) != (n_events,):
            raise ValueError("event_patient_index must have shape [E]")
        if self.event_patient_index.dtype != torch.long:
            raise TypeError("event_patient_index must be torch.long")
        if len(self.event_ids) != n_events:
            raise ValueError("event_ids must align with event evidence")
        if tuple(self.expected_event_counts.shape) != (n_patients,):
            raise ValueError("expected_event_counts must have shape [P]")
        if self.expected_event_counts.dtype != torch.long:
            raise TypeError("expected_event_counts must be torch.long")
        if tuple(self.targets.shape) != (n_patients, 19):
            raise ValueError("targets must have one [19] row per patient")
        if tuple(self.target_mask.shape) != (n_patients, 19):
            raise ValueError("target_mask must have one [19] row per patient")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("target_mask must be torch.bool")
        if n_patients < 1 or self.event_patient_index.min() < 0:
            raise ValueError("Patient batch cannot be empty or use negative indices")
        if self.event_patient_index.max() >= n_patients:
            raise ValueError("event_patient_index refers to an absent patient")
        observed_counts = torch.bincount(
            self.event_patient_index, minlength=n_patients
        ).to(device=self.expected_event_counts.device)
        if not torch.equal(observed_counts, self.expected_event_counts):
            raise ValueError("Batch does not contain the registered complete patient bags")

    def aggregate(self, event_logits: torch.Tensor) -> PatientAggregation:
        if event_logits.ndim != 2 or tuple(event_logits.shape) != (
            self.evidence.batch_size,
            19,
        ):
            raise ValueError("event_logits must align with the complete event bag")
        if event_logits.device != self.event_patient_index.device:
            raise ValueError("event logits and patient/event masks must share a device")

        # A fully invalid phase row produces the channel prior and no EEG
        # contribution in the reasoner. Averaging such a row would shrink a
        # patient's actual evidence merely because more unusable events were
        # registered. Keep the event in the complete cache/bag audit, but do
        # not let it enter the equal-evidence-event mean.
        aggregation_event_mask = self.evidence.ictal_phase_mask.any(dim=1)
        usable_patient_index = self.event_patient_index[aggregation_event_mask]
        usable_counts = torch.bincount(
            usable_patient_index,
            minlength=len(self.patient_ids),
        )
        if (usable_counts == 0).any():
            bad = (usable_counts == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every patient requires at least one event with valid phase evidence; "
                f"missing patient rows={bad}"
            )
        aggregation = aggregate_patient_logits(
            event_logits[aggregation_event_mask],
            usable_patient_index,
        )
        expected_ids = torch.arange(
            len(self.patient_ids), device=event_logits.device, dtype=torch.long
        )
        if not torch.equal(aggregation.patient_ids, expected_ids):
            raise RuntimeError("Patient aggregation lost or reordered a target row")
        if not torch.equal(aggregation.event_counts, usable_counts):
            raise RuntimeError("Patient aggregation lost a phase-valid event")
        return aggregation

    def to(self, device: torch.device | str) -> "PatientEvidenceBatch":
        """Move numeric batch state without changing its frozen registry receipt."""

        return PatientEvidenceBatch(
            evidence=self.evidence.to(device=device),
            event_patient_index=self.event_patient_index.to(device=device),
            patient_ids=self.patient_ids,
            event_ids=self.event_ids,
            expected_event_counts=self.expected_event_counts.to(device=device),
            targets=self.targets.to(device=device),
            target_mask=self.target_mask.to(device=device),
            event_registry_sha256=self.event_registry_sha256,
        )


def _concatenate_evidence(events: Sequence[EvidenceEvent]) -> EvidenceBatch:
    first = events[0].evidence
    for event in events[1:]:
        current = event.evidence
        if current.n_tiles != first.n_tiles:
            raise ValueError("All events in a batch must share the evidence time grid")
        if current.node.dtype != first.node.dtype or current.edge.dtype != first.edge.dtype:
            raise TypeError("All events in a batch must share evidence dtypes")
        if current.node.device != first.node.device or current.edge.device != first.edge.device:
            raise ValueError("All events in a batch must share a device")
    return EvidenceBatch(
        node=torch.cat([event.evidence.node for event in events], dim=0),
        edge=torch.cat([event.evidence.edge for event in events], dim=0),
        node_mask=torch.cat([event.evidence.node_mask for event in events], dim=0),
        edge_mask=torch.cat([event.evidence.edge_mask for event in events], dim=0),
        physical_signal_mask=torch.cat(
            [event.evidence.physical_signal_mask for event in events], dim=0
        ),
        ictal_phase_mask=torch.cat(
            [event.evidence.ictal_phase_mask for event in events], dim=0
        ),
        morphology_mask=torch.cat(
            [event.evidence.morphology_mask for event in events], dim=0
        ),
        morphology_context_mask=torch.cat(
            [event.evidence.morphology_context_mask for event in events], dim=0
        ),
        ictal_mask=torch.cat(
            [event.evidence.ictal_mask for event in events], dim=0
        ),
    )


def collate_patient_evidence(
    events: Sequence[EvidenceEvent],
    references: DeepSOZReferenceRegistry,
    event_registry: EventInputRegistry,
    *,
    expected_model_split: str,
) -> PatientEvidenceBatch:
    """Join complete patient bags to targets under verified cache provenance."""

    if not events:
        raise ValueError("Cannot collate an empty event sequence")
    event_ids = tuple(event.event_id for event in events)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Duplicate event IDs are not allowed in a patient batch")

    records = tuple(event_registry.get(event.event_id) for event in events)
    patient_ids_by_event = tuple(record.patient_id for record in records)
    unique_patient_ids = tuple(sorted(set(patient_ids_by_event)))
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(unique_patient_ids)
    }

    for event, record in zip(events, records):
        if record.model_split != expected_model_split:
            raise ValueError(
                f"Registered event {record.event_id} has split {record.model_split}, "
                f"expected {expected_model_split}"
            )
        reference = references.get(record.patient_id)
        if reference.model_split != expected_model_split:
            raise ValueError(
                f"Patient {record.patient_id} registry split is {reference.model_split}, "
                f"expected {expected_model_split}"
            )
        if not reference.eligible_for_localization:
            raise ValueError(
                f"Patient {record.patient_id} is not eligible for localization"
            )
        event_registry.validate_cache_receipt(event.cache_receipt, references)

    expected_counts: list[int] = []
    for patient_id in unique_patient_ids:
        provided = {
            event_id
            for event_id, event_patient_id in zip(event_ids, patient_ids_by_event)
            if event_patient_id == patient_id
        }
        registered = {
            record.event_id for record in event_registry.events_for_patient(patient_id)
        }
        if provided != registered:
            missing = sorted(registered - provided)
            extra = sorted(provided - registered)
            raise ValueError(
                f"Patient {patient_id} requires a complete event bag; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        expected_counts.append(len(registered))

    evidence = _concatenate_evidence(events)
    target_batch = references.target_batch(
        unique_patient_ids, device=evidence.node.device
    )
    event_patient_index = torch.tensor(
        [patient_to_index[patient_id] for patient_id in patient_ids_by_event],
        dtype=torch.long,
        device=evidence.node.device,
    )
    return PatientEvidenceBatch(
        evidence=evidence,
        event_patient_index=event_patient_index,
        patient_ids=unique_patient_ids,
        event_ids=event_ids,
        expected_event_counts=torch.tensor(
            expected_counts, dtype=torch.long, device=evidence.node.device
        ),
        targets=target_batch.values,
        target_mask=target_batch.mask,
        event_registry_sha256=event_registry.manifest_sha256,
    )


class PatientBagDataset(Sequence[PatientEvidenceBatch]):
    """One index per patient, with all registered primary events loaded.

    Construction requires a complete cache for the selected split. The
    provided ``iter_epoch`` accepts only a permutation of the patient roster,
    preventing a patient from silently receiving two losses in one epoch.
    """

    def __init__(
        self,
        events: Sequence[EvidenceEvent],
        references: DeepSOZReferenceRegistry,
        event_registry: EventInputRegistry,
        *,
        expected_model_split: str,
    ) -> None:
        by_event = {event.event_id: event for event in events}
        if len(by_event) != len(events):
            raise ValueError("PatientBagDataset received duplicate event IDs")
        expected_records = tuple(
            record
            for record in event_registry
            if record.model_split == expected_model_split
        )
        expected_ids = {record.event_id for record in expected_records}
        provided_ids = set(by_event)
        if provided_ids != expected_ids:
            raise ValueError(
                "PatientBagDataset requires the complete registered split cache; "
                f"missing={sorted(expected_ids - provided_ids)[:5]}, "
                f"extra={sorted(provided_ids - expected_ids)[:5]}"
            )
        patient_ids = tuple(
            sorted({record.patient_id for record in expected_records})
        )
        if not patient_ids:
            raise ValueError("Selected split has no registered patient events")

        by_patient: dict[str, tuple[EvidenceEvent, ...]] = {}
        extractor_hash_by_key: dict[tuple[str, int | None], str] = {}
        for patient_id in patient_ids:
            patient_events = tuple(
                by_event[record.event_id]
                for record in event_registry.events_for_patient(patient_id)
            )
            # Validate completeness, target join, and every receipt at dataset
            # construction rather than discovering leakage during optimization.
            collate_patient_evidence(
                patient_events,
                references,
                event_registry,
                expected_model_split=expected_model_split,
            )
            family_sets = {
                tuple(
                    extractor.concept_family
                    for extractor in event.cache_receipt.extractors
                )
                for event in patient_events
            }
            if len(family_sets) != 1:
                raise ValueError("One patient bag changes its active-family set")
            for family in next(iter(family_sets)):
                extractors = tuple(
                    event.cache_receipt.extractor(family)
                    for event in patient_events
                )
                hashes = {extractor.receipt_sha256 for extractor in extractors}
                if len(hashes) != 1:
                    raise ValueError(
                        f"One patient bag uses multiple {family} extractors"
                    )
                extractor = extractors[0]
                key = (family, extractor.oof_fold)
                receipt_hash = next(iter(hashes))
                previous = extractor_hash_by_key.setdefault(key, receipt_hash)
                if previous != receipt_hash:
                    raise ValueError(
                        f"One OOF fold/split uses multiple {family} extractors"
                    )
            by_patient[patient_id] = patient_events

        self._references = references
        self._event_registry = event_registry
        self._expected_model_split = expected_model_split
        self._patient_ids = patient_ids
        self._by_patient = by_patient

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return self._patient_ids

    @property
    def model_split(self) -> str:
        return self._expected_model_split

    def __len__(self) -> int:
        return len(self._patient_ids)

    def __getitem__(self, index: int) -> PatientEvidenceBatch:
        patient_id = self._patient_ids[index]
        return collate_patient_evidence(
            self._by_patient[patient_id],
            self._references,
            self._event_registry,
            expected_model_split=self._expected_model_split,
        )

    def iter_epoch(
        self, patient_order: Sequence[object] | None = None
    ) -> Iterator[PatientEvidenceBatch]:
        if patient_order is None:
            order = self._patient_ids
        else:
            order = tuple(normalize_patient_id(value) for value in patient_order)
            if len(order) != len(self._patient_ids) or set(order) != set(
                self._patient_ids
            ):
                raise ValueError(
                    "Epoch order must contain every registered patient exactly once"
                )
        for patient_id in order:
            yield collate_patient_evidence(
                self._by_patient[patient_id],
                self._references,
                self._event_registry,
                expected_model_split=self._expected_model_split,
            )
