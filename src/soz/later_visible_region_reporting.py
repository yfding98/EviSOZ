"""Fail-closed adapter for a receipted later-visible region fact."""

from __future__ import annotations

from dataclasses import replace

from .clinical_reporting import (
    ClinicalReportFactsV2,
    EventScalpPhenotypeEvidence,
)
from .later_visible_region_producer import (
    LaterVisibleRegionReceipt,
    event_evidence_core_sha256,
)


def attach_later_visible_region_to_clinical_facts(
    facts: ClinicalReportFactsV2,
    region_receipt: LaterVisibleRegionReceipt,
) -> ClinicalReportFactsV2:
    """Populate only the later-visible region slot and preserve SOZ ranking."""

    if not isinstance(facts, ClinicalReportFactsV2):
        raise TypeError("facts must be ClinicalReportFactsV2")
    if not isinstance(region_receipt, LaterVisibleRegionReceipt):
        raise TypeError("region_receipt must be LaterVisibleRegionReceipt")
    if facts.later_visible_region_receipt is not None:
        raise ValueError("Clinical facts already contain a later-visible receipt")
    event = facts.event_phenotype
    if not isinstance(event, EventScalpPhenotypeEvidence):
        raise ValueError("An abstained event cannot receive a later-visible region")
    if event.later_visible_region_zh is not None:
        raise ValueError("later_visible_region_zh is already populated")
    event_receipt = event.receipt
    expected = (
        (region_receipt.patient_pseudonym, event_receipt.patient_pseudonym),
        (region_receipt.event_pseudonym, event_receipt.event_pseudonym),
        (
            region_receipt.evidence_artifact_sha256,
            event_receipt.evidence_artifact_sha256,
        ),
        (
            region_receipt.source_event_receipt_sha256,
            event_evidence_core_sha256(event_receipt),
        ),
        (region_receipt.observed_derivations, event.later_visible_derivations),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise ValueError("Later-visible region receipt and event evidence mismatch")

    preserved_ranking = facts.patient_ranking
    preserved_spatial_report = preserved_ranking.spatial_report
    bound_event = replace(
        event,
        later_visible_region_zh=region_receipt.later_visible_region_zh,
    )
    result = replace(
        facts,
        event_phenotype=bound_event,
        later_visible_region_receipt=region_receipt,
    )
    if result.patient_ranking is not preserved_ranking or (
        result.patient_ranking.spatial_report is not preserved_spatial_report
    ):
        raise RuntimeError("Later-visible report adapter changed the SOZ ranking")
    return result


__all__ = ["attach_later_visible_region_to_clinical_facts"]
