"""One-shot, grounding-gated report feedback for EviSOZ.

Report text is never a localization label.  The only permitted feedback is a
non-negative residual over an already frozen-v29 node distribution, and only
for claims that can be replayed against an explicit structured evidence view.
The gate is useful for an eventual V4 experiment; it returns a diagnostic
under Stage-0 ``NO_GO`` and refuses to apply any update until the aggregate
training/Qwen authorization is open.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from torch import Tensor

from src.evisoz.evaluation.report_factuality import evaluate_evisoz_report_factuality

from .stage0_guard import require_stage0_training_authorized


_CERTAINTY_RANK = {
    "not_assessable": 0,
    "uncertain": 1,
    "possible": 1,
    "probable": 2,
    "supported": 3,
    "definite": 4,
}


def _claim_supports_evidence(claim: Mapping[str, object], evidence: Mapping[str, object]) -> tuple[bool, str]:
    evidence_ids = {str(item) for item in evidence.get("evidence_ids", [])}
    supports = claim.get("evidence_ids", [])
    if not isinstance(supports, list) or not supports or not set(map(str, supports)).issubset(evidence_ids):
        return False, "missing_or_unknown_evidence_ids"
    claim_type = claim.get("claim_type")
    units = {str(item) for item in claim.get("units", [])}
    if claim_type == "onset" and not units.issubset(set(evidence.get("onset_channels", []))):
        return False, "onset_units_not_grounded"
    if claim_type == "spread" and not units.issubset(set(evidence.get("spread_channels", []))):
        return False, "spread_units_not_grounded"
    if claim_type in {"onset", "spread"}:
        order = claim.get("order", [])
        allowed_order = evidence.get("onset_order" if claim_type == "onset" else "spread_order", [])
        if not isinstance(order, list) or not set(order).issubset(set(allowed_order)):
            return False, "recruitment_order_not_grounded"
    if claim_type == "morphology" and not units.issubset(set(evidence.get("morphologies", []))):
        return False, "morphology_not_grounded"
    if claim_type == "region" and not units.issubset(set(evidence.get("regions", []))):
        return False, "region_not_grounded"
    if claim_type == "laterality" and units != {str(evidence.get("laterality"))}:
        return False, "laterality_not_grounded"
    if _CERTAINTY_RANK.get(str(claim.get("certainty", "uncertain")), 0) > _CERTAINTY_RANK.get(str(evidence.get("certainty", "uncertain")), 0):
        return False, "certainty_promoted"
    return True, "grounded"


def evaluate_grounding_gate(
    report: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, Any]:
    """Return per-claim grounding decisions without changing model outputs."""

    factuality = evaluate_evisoz_report_factuality(report, evidence)
    claims = report.get("claims")
    if not isinstance(claims, list):
        raise ValueError("report.claims must be a list")
    decisions: list[dict[str, object]] = []
    approved = 0
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise ValueError("report claims must be objects")
        ok, reason = _claim_supports_evidence(claim, evidence)
        row = {
            "claim_id": str(claim.get("claim_id", f"claim-{index}")),
            "approved": bool(ok),
            "reason": reason,
        }
        approved += int(ok)
        decisions.append(row)
    return {
        "status": "grounding_gate_evaluated",
        "approved_claim_count": approved,
        "claim_count": len(decisions),
        "all_claims_grounded": approved == len(decisions),
        "claim_decisions": decisions,
        "factuality": factuality,
        "feedback_enabled": False,
    }


def apply_one_shot_grounded_feedback(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    baseline_logits: Tensor,
    report_delta: Tensor,
    candidate_mask: Tensor,
    feedback_eligible: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Apply one non-negative report residual after the full grounding gate.

    ``feedback_eligible`` is a caller-supplied per-event boolean produced by
    the structured claim gate.  This function does not infer eligibility from
    free text and does not permit a report to alter the candidate mask.
    """

    if not isinstance(baseline_logits, Tensor) or baseline_logits.ndim != 2 or baseline_logits.shape[-1] != 19:
        raise ValueError("baseline_logits must have shape [B,19]")
    if not isinstance(report_delta, Tensor) or tuple(report_delta.shape) != tuple(baseline_logits.shape):
        raise ValueError("report_delta must have shape [B,19]")
    if not baseline_logits.is_floating_point() or not report_delta.is_floating_point() or not torch.isfinite(baseline_logits).all() or not torch.isfinite(report_delta).all():
        raise ValueError("baseline_logits and report_delta must be finite floating point")
    batch = baseline_logits.shape[0]
    for name, value, shape in (
        ("candidate_mask", candidate_mask, (batch, 19)),
        ("feedback_eligible", feedback_eligible, (batch,)),
    ):
        if not isinstance(value, Tensor) or value.dtype is not torch.bool or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be bool with shape {shape}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(float(alpha))) or float(alpha) <= 0 or float(alpha) > 1:
        raise ValueError("alpha must be in (0,1]")
    authorization = require_stage0_training_authorized(
        gate,
        pipeline_config=pipeline_config,
        requested_actions=("qwen_sft_or_eeg_to_qwen_alignment", "nonzero_residual_gate"),
    )
    # The residual is deliberately clipped to non-negative candidate support;
    # report feedback can never subtract from or create an unobserved node.
    delta = report_delta.clamp_min(0) * candidate_mask.to(report_delta.dtype)
    applied_gate = feedback_eligible.to(report_delta.dtype).unsqueeze(-1)
    applied = float(alpha) * applied_gate * delta
    updated = baseline_logits + applied
    return updated, applied, applied_gate, {
        "authorization": deepcopy(authorization),
        "feedback_applied_once": True,
        "report_can_create_patient_fact": False,
    }


__all__ = ["apply_one_shot_grounded_feedback", "evaluate_grounding_gate"]
