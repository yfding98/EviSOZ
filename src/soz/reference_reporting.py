"""Atomic adapter from target-free reference receipts to report facts."""

from __future__ import annotations

from dataclasses import replace

from .clinical_reporting import ClinicalReportFactsV2
from .reference_disagreement import ReferenceDisagreementReceipt


def attach_reference_disagreement_to_clinical_facts(
    facts: ClinicalReportFactsV2,
    reference: ReferenceDisagreementReceipt,
) -> ClinicalReportFactsV2:
    """Atomically bind reference disagreement to provenance and uncertainty.

    The adapter changes neither localization scores nor event evidence.  It
    only attaches the already-computed target-free robustness measurement to
    the event provenance and the patient uncertainty decomposition.  The
    exact event roster must match the patient ranking, so a one-event audit
    cannot be laundered into a multi-event patient-level uncertainty claim.
    """

    if not isinstance(facts, ClinicalReportFactsV2):
        raise TypeError("facts must be ClinicalReportFactsV2")
    if not isinstance(reference, ReferenceDisagreementReceipt):
        raise TypeError("reference must be ReferenceDisagreementReceipt")
    if facts.reference_disagreement_receipt is not None:
        raise ValueError("Clinical facts already contain reference disagreement")

    event = facts.event_phenotype
    provenance = event.receipt
    ranking = facts.patient_ranking
    uncertainty = ranking.uncertainty
    if provenance.reference_disagreement_receipt_sha256 is not None:
        raise ValueError("Event provenance already contains reference disagreement")
    if uncertainty.montage_disagreement is not None:
        raise ValueError("montage_disagreement is already populated")
    if reference.patient_pseudonym != provenance.patient_pseudonym:
        raise ValueError("Reference receipt and clinical facts patient mismatch")
    if reference.aggregation_event_ids != ranking.aggregation_event_ids:
        raise ValueError(
            "Reference receipt and patient ranking aggregation roster mismatch"
        )
    signal_sha = reference.signal_artifact_sha256_for_event(
        provenance.event_pseudonym
    )
    if signal_sha != provenance.signal_artifact_sha256:
        raise ValueError("Reference receipt and event provenance signal mismatch")

    montages = list(provenance.montages)
    for arm_id in (reference.primary_arm_id, reference.sensitivity_arm_id):
        if arm_id not in montages:
            montages.append(arm_id)
    bound_provenance = replace(
        provenance,
        montages=tuple(montages),
        reference_pair_schema_version=reference.reference_pair_schema_version,
        reference_pair_role=reference.reference_pair_role,
        reference_primary_arm_id=reference.primary_arm_id,
        reference_sensitivity_arm_id=reference.sensitivity_arm_id,
        reference_disagreement_metric_id=reference.metric_id,
        reference_disagreement_receipt_sha256=reference.receipt_sha256,
    )
    bound_event = replace(event, receipt=bound_provenance)
    bound_uncertainty = replace(
        uncertainty,
        montage_disagreement=reference.montage_disagreement,
    )
    bound_ranking = replace(ranking, uncertainty=bound_uncertainty)
    return replace(
        facts,
        event_phenotype=bound_event,
        patient_ranking=bound_ranking,
        reference_disagreement_receipt=reference,
    )


__all__ = ["attach_reference_disagreement_to_clinical_facts"]
