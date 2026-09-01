"""Target-free same-event C-CAR19/C-REF19 phenotype consistency.

This module compares two independently materialized outputs of the frozen
event-phenotype producer.  It does not read EEG, labels, or localization
scores, and it never changes a patient SOZ ranking.  A positive result is a
descriptive robustness fact about *scalp-visible bipolar event evidence*;
because common-reference subtraction cancels in a bipolar derivation, it is
also a preprocessing/extraction consistency check rather than independent
cortical SOZ confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Final

from .clinical_reporting import (
    ClinicalReportFactsV2,
    EventReferenceConsistencyReceipt,
    EventScalpPhenotypeAbstention,
    EventScalpPhenotypeEvidence,
    LATERALITY_GROUPS,
)
from .event_phenotype_producer import EventPhenotypeProductionResult


PRIMARY_EVENT_REFERENCE_ARM: Final[str] = "C-CAR19"
SENSITIVITY_EVENT_REFERENCE_ARM: Final[str] = "C-REF19"
EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC: Final[float] = 0.25


@dataclass(frozen=True)
class EventReferenceConsistencyResult:
    """Primary event facts plus a target-free paired-reference receipt."""

    event: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention
    receipt: EventReferenceConsistencyReceipt

    def __post_init__(self) -> None:
        if not isinstance(
            self.event,
            (EventScalpPhenotypeEvidence, EventScalpPhenotypeAbstention),
        ):
            raise TypeError("event must be typed event evidence or abstention")
        if not isinstance(self.receipt, EventReferenceConsistencyReceipt):
            raise TypeError("receipt must be EventReferenceConsistencyReceipt")
        event_receipt = self.event.receipt
        expected = (
            (event_receipt.patient_pseudonym, self.receipt.patient_pseudonym),
            (event_receipt.event_pseudonym, self.receipt.event_pseudonym),
            (
                event_receipt.signal_artifact_sha256,
                self.receipt.signal_artifact_sha256,
            ),
            (
                event_receipt.evidence_artifact_sha256,
                self.receipt.primary_evidence_artifact_sha256,
            ),
        )
        if any(actual != wanted for actual, wanted in expected):
            raise ValueError("Paired-reference result identity disagrees with receipt")
        for arm_id in (
            self.receipt.primary_arm_id,
            self.receipt.sensitivity_arm_id,
        ):
            if arm_id not in event_receipt.montages:
                raise ValueError("Paired-reference arm is absent from event provenance")
        event_stability = (
            self.event.montage_stability
            if isinstance(self.event, EventScalpPhenotypeEvidence)
            else None
        )
        if event_stability != self.receipt.montage_stability:
            raise ValueError("Paired-reference result stability disagrees with receipt")
        expected_primary_status = (
            "reportable"
            if isinstance(self.event, EventScalpPhenotypeEvidence)
            else "abstained"
        )
        if self.receipt.primary_result_status != expected_primary_status:
            raise ValueError("Paired-reference result primary status disagrees")


def _event_receipt(result: EventPhenotypeProductionResult):
    if not isinstance(result, EventPhenotypeProductionResult):
        raise TypeError("Reference comparison inputs must be producer results")
    return result.report_event.receipt


def _require_arm(
    result: EventPhenotypeProductionResult,
    *,
    expected_arm: str,
) -> None:
    receipt = _event_receipt(result)
    if receipt.montages != (expected_arm,):
        raise ValueError(
            f"Event phenotype must be produced solely from {expected_arm}"
        )
    if receipt.reference_disagreement_receipt_sha256 is not None:
        raise ValueError("Event reference comparison must precede patient audit binding")


def _unilateral_support(
    derivations: tuple[str, ...],
) -> str | None:
    channel_side = {
        channel: side
        for side, channels in LATERALITY_GROUPS.items()
        for channel in channels
    }
    lateral: set[str] = set()
    for derivation in derivations:
        left, right = derivation.split("-")
        for channel in (left, right):
            side = channel_side[channel]
            if side in {"left", "right"}:
                lateral.add(side)
    if len(lateral) == 1:
        return next(iter(lateral))
    return None


def assess_event_reference_consistency(
    primary: EventPhenotypeProductionResult,
    sensitivity: EventPhenotypeProductionResult,
    *,
    temporal_alignment_tolerance_sec: float = (
        EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC
    ),
) -> EventReferenceConsistencyResult:
    """Compare paired target-free event phenotypes and bind the primary facts.

    Exact edge-set agreement yields ``consistent``.  Different exact edges
    can yield ``partially_consistent`` only when both sets are unambiguously
    confined to the same hemisphere.  Any stability state additionally
    requires onset-time agreement within the frozen tolerance.  Otherwise
    the slot remains absent with an explicit reason code.
    """

    if (
        isinstance(temporal_alignment_tolerance_sec, bool)
        or not isinstance(temporal_alignment_tolerance_sec, (int, float))
        or not math.isfinite(float(temporal_alignment_tolerance_sec))
        or temporal_alignment_tolerance_sec < 0
    ):
        raise ValueError("temporal_alignment_tolerance_sec must be non-negative")
    _require_arm(primary, expected_arm=PRIMARY_EVENT_REFERENCE_ARM)
    _require_arm(sensitivity, expected_arm=SENSITIVITY_EVENT_REFERENCE_ARM)
    primary_event = primary.report_event
    sensitivity_event = sensitivity.report_event
    primary_receipt = primary_event.receipt
    sensitivity_receipt = sensitivity_event.receipt
    identity_pairs = (
        (
            primary_receipt.patient_pseudonym,
            sensitivity_receipt.patient_pseudonym,
        ),
        (primary_receipt.event_pseudonym, sensitivity_receipt.event_pseudonym),
        (
            primary_receipt.signal_artifact_sha256,
            sensitivity_receipt.signal_artifact_sha256,
        ),
        (
            primary_receipt.time_coordinate_semantics,
            sensitivity_receipt.time_coordinate_semantics,
        ),
        (
            primary_receipt.causal_prefix_safe,
            sensitivity_receipt.causal_prefix_safe,
        ),
        (
            primary_receipt.extractor_model_id,
            sensitivity_receipt.extractor_model_id,
        ),
        (
            primary_receipt.extractor_model_version,
            sensitivity_receipt.extractor_model_version,
        ),
        (
            primary_receipt.evidence_generation_policy,
            sensitivity_receipt.evidence_generation_policy,
        ),
    )
    if any(left != right for left, right in identity_pairs):
        raise ValueError(
            "C-CAR19/C-REF19 event phenotypes do not share exact identity/lineage"
        )

    primary_edges = (
        primary_event.first_visible_derivations
        if isinstance(primary_event, EventScalpPhenotypeEvidence)
        else ()
    )
    sensitivity_edges = (
        sensitivity_event.first_visible_derivations
        if isinstance(sensitivity_event, EventScalpPhenotypeEvidence)
        else ()
    )
    onset_delta: float | None = None
    stability: str | None = None
    reasons: tuple[str, ...] = ()
    if not isinstance(primary_event, EventScalpPhenotypeEvidence):
        reasons = ("primary_reference_abstained",)
    elif not isinstance(sensitivity_event, EventScalpPhenotypeEvidence):
        reasons = ("sensitivity_reference_abstained",)
    else:
        onset_delta = abs(
            float(primary_event.onset_start_sec)
            - float(sensitivity_event.onset_start_sec)
        )
        if onset_delta > float(temporal_alignment_tolerance_sec) + 1e-12:
            reasons = ("reference_onset_timing_mismatch",)
        elif frozenset(primary_edges) == frozenset(sensitivity_edges):
            stability = "consistent"
        else:
            primary_side = _unilateral_support(primary_edges)
            sensitivity_side = _unilateral_support(sensitivity_edges)
            if primary_side is not None and primary_side == sensitivity_side:
                stability = "partially_consistent"
            else:
                stability = "inconsistent"

    paired_receipt = replace(
        primary_receipt,
        montages=(PRIMARY_EVENT_REFERENCE_ARM, SENSITIVITY_EVENT_REFERENCE_ARM),
    )
    if isinstance(primary_event, EventScalpPhenotypeEvidence):
        bound_primary = replace(
            primary_event,
            receipt=paired_receipt,
            montage_stability=stability,
        )
    else:
        bound_primary = replace(primary_event, receipt=paired_receipt)
    receipt = EventReferenceConsistencyReceipt(
        patient_pseudonym=primary_receipt.patient_pseudonym,
        event_pseudonym=primary_receipt.event_pseudonym,
        signal_artifact_sha256=primary_receipt.signal_artifact_sha256,
        primary_evidence_artifact_sha256=(
            primary_receipt.evidence_artifact_sha256
        ),
        sensitivity_evidence_artifact_sha256=(
            sensitivity_receipt.evidence_artifact_sha256
        ),
        primary_arm_id=PRIMARY_EVENT_REFERENCE_ARM,
        sensitivity_arm_id=SENSITIVITY_EVENT_REFERENCE_ARM,
        primary_result_status=primary.status,
        sensitivity_result_status=sensitivity.status,
        temporal_alignment_tolerance_sec=float(
            temporal_alignment_tolerance_sec
        ),
        onset_start_delta_sec=onset_delta,
        primary_first_visible_derivations=primary_edges,
        sensitivity_first_visible_derivations=sensitivity_edges,
        montage_stability=stability,
        reason_codes=reasons,
        target_labels_used=False,
        private_data_used=False,
        localization_scores_used=False,
        training_performed=False,
    )
    return EventReferenceConsistencyResult(event=bound_primary, receipt=receipt)


def attach_event_reference_consistency_to_clinical_facts(
    facts: ClinicalReportFactsV2,
    consistency: EventReferenceConsistencyResult,
) -> ClinicalReportFactsV2:
    """Atomically replace only event facts; patient scores remain untouched."""

    if not isinstance(facts, ClinicalReportFactsV2):
        raise TypeError("facts must be ClinicalReportFactsV2")
    if not isinstance(consistency, EventReferenceConsistencyResult):
        raise TypeError("consistency must be EventReferenceConsistencyResult")
    if facts.event_reference_consistency_receipt is not None:
        raise ValueError("Clinical facts already contain event reference consistency")
    if facts.reference_disagreement_receipt is not None:
        raise ValueError(
            "Event reference consistency must be attached before the patient-level "
            "reference-disagreement receipt"
        )
    current = facts.event_phenotype
    current_receipt = current.receipt
    paired_receipt = consistency.event.receipt
    expected = (
        (current_receipt.patient_pseudonym, paired_receipt.patient_pseudonym),
        (current_receipt.event_pseudonym, paired_receipt.event_pseudonym),
        (
            current_receipt.signal_artifact_sha256,
            paired_receipt.signal_artifact_sha256,
        ),
        (
            current_receipt.evidence_artifact_sha256,
            paired_receipt.evidence_artifact_sha256,
        ),
    )
    if any(left != right for left, right in expected):
        raise ValueError("Clinical event and paired-reference primary event mismatch")
    if isinstance(current, EventScalpPhenotypeEvidence):
        if not isinstance(consistency.event, EventScalpPhenotypeEvidence):
            raise ValueError("Paired-reference result changed primary event status")
        if replace(current, receipt=consistency.event.receipt) != replace(
            consistency.event,
            montage_stability=current.montage_stability,
        ):
            raise ValueError("Paired-reference result changed non-reference event facts")
    else:
        if not isinstance(consistency.event, EventScalpPhenotypeAbstention):
            raise ValueError("Paired-reference result changed primary event status")
        if replace(current, receipt=consistency.event.receipt) != consistency.event:
            raise ValueError("Paired-reference result changed primary abstention facts")
    return replace(
        facts,
        event_phenotype=consistency.event,
        event_reference_consistency_receipt=consistency.receipt,
    )


__all__ = [
    "EVENT_REFERENCE_TEMPORAL_TOLERANCE_SEC",
    "EventReferenceConsistencyResult",
    "PRIMARY_EVENT_REFERENCE_ARM",
    "SENSITIVITY_EVENT_REFERENCE_ARM",
    "assess_event_reference_consistency",
    "attach_event_reference_consistency_to_clinical_facts",
]
