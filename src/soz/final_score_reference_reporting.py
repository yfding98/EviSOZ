"""Score-preserving report adapter for final-score reference sensitivity."""

from __future__ import annotations

from dataclasses import replace

from .clinical_reporting import ClinicalReportFactsV2
from .final_score_reference_disagreement import (
    FinalScoreReferenceDisagreementReceipt,
)


def attach_final_score_reference_disagreement_to_clinical_facts(
    facts: ClinicalReportFactsV2,
    reference: FinalScoreReferenceDisagreementReceipt,
) -> ClinicalReportFactsV2:
    """Attach one independently typed final-score sensitivity fact.

    Only the separately named uncertainty scalar and its typed receipt are
    added.  The existing C-CAR19 spatial report, top channels, score, model
    identity, aggregation identity, and all event facts remain unchanged.
    A block-9 representation receipt is intentionally rejected by the type
    boundary and cannot populate this slot.
    """

    if not isinstance(facts, ClinicalReportFactsV2):
        raise TypeError("facts must be ClinicalReportFactsV2")
    if not isinstance(reference, FinalScoreReferenceDisagreementReceipt):
        raise TypeError(
            "reference must be FinalScoreReferenceDisagreementReceipt"
        )
    if facts.final_score_reference_disagreement_receipt is not None:
        raise ValueError("Clinical facts already contain final-score sensitivity")

    ranking = facts.patient_ranking
    uncertainty = ranking.uncertainty
    if uncertainty.final_score_reference_disagreement is not None:
        raise ValueError("final_score_reference_disagreement is already populated")
    if reference.patient_pseudonym != ranking.patient_pseudonym:
        raise ValueError("Final-score receipt and clinical facts patient mismatch")
    if reference.aggregation_event_ids != ranking.aggregation_event_ids:
        raise ValueError(
            "Final-score receipt and patient ranking aggregation roster mismatch"
        )
    if reference.primary_top1_channel not in ranking.spatial_report.top_channels:
        raise ValueError(
            "Final-score receipt primary Top-1 disagrees with preserved CAR ranking"
        )

    preserved_spatial_report = ranking.spatial_report
    preserved_ranking_identity = (
        ranking.patient_pseudonym,
        ranking.model_id,
        ranking.model_version,
        ranking.model_checkpoint_sha256,
        ranking.aggregation_method,
        ranking.aggregation_event_count,
        ranking.aggregation_event_ids,
        ranking.aggregation_receipt_sha256,
        ranking.ranking_granularity,
    )
    bound_uncertainty = replace(
        uncertainty,
        final_score_reference_disagreement=(
            reference.final_score_reference_disagreement
        ),
    )
    bound_ranking = replace(ranking, uncertainty=bound_uncertainty)
    result = replace(
        facts,
        patient_ranking=bound_ranking,
        final_score_reference_disagreement_receipt=reference,
    )

    result_ranking = result.patient_ranking
    result_ranking_identity = (
        result_ranking.patient_pseudonym,
        result_ranking.model_id,
        result_ranking.model_version,
        result_ranking.model_checkpoint_sha256,
        result_ranking.aggregation_method,
        result_ranking.aggregation_event_count,
        result_ranking.aggregation_event_ids,
        result_ranking.aggregation_receipt_sha256,
        result_ranking.ranking_granularity,
    )
    if result_ranking.spatial_report is not preserved_spatial_report or (
        result_ranking_identity != preserved_ranking_identity
    ):
        raise RuntimeError("Final-score report adapter changed the CAR ranking")
    return result


__all__ = ["attach_final_score_reference_disagreement_to_clinical_facts"]
