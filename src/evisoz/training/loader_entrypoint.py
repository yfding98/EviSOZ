"""Authorized training-data entry point for the EviSOZ route.

The order here is part of the safety contract: the aggregate Stage-0 guard is
called before the bound-evidence loader is constructed.  A ``NO_GO`` or
evaluator-only gate therefore cannot even open a training iterator.
"""

from __future__ import annotations

from typing import Iterator, Mapping

from src.evisoz.data.bound_evidence_loader import (
    BoundEvidenceRecord,
    iter_bound_evidence_records,
)

from .stage0_guard import require_stage0_training_authorized


def open_authorized_training_records(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    requested_actions: tuple[str, ...],
    bound_evidence_root: str,
    private_examples_root: str,
    findings_claim_report_root: str,
    private_cohort_root: str,
    split_roster_path: str,
    evisoz_role: str = "development_cv",
) -> tuple[dict[str, object], Iterator[BoundEvidenceRecord]]:
    """Return authorization receipt and a loader-backed training iterator.

    This function does not weaken the Stage-0 policy.  With the current
    shadow gate it always raises ``Stage0TrainingBlocked`` before any source
    event or tensor is opened.
    """

    authorization = require_stage0_training_authorized(
        gate,
        pipeline_config=pipeline_config,
        requested_actions=requested_actions,
    )
    records = iter_bound_evidence_records(
        bound_evidence_root=bound_evidence_root,
        private_examples_root=private_examples_root,
        findings_claim_report_root=findings_claim_report_root,
        private_cohort_root=private_cohort_root,
        split_roster_path=split_roster_path,
        evisoz_role=evisoz_role,
    )
    return authorization, records


__all__ = ["open_authorized_training_records"]
