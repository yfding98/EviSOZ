"""Authorized residual-localization forward for the EviSOZ route."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline
from src.evisoz.training.stage0_guard import require_stage0_training_authorized


def run_authorized_residual_forward(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    pipeline: EviSOZEvidencePipeline,
    node_tokens: Tensor,
    node_mask: Tensor,
    edge_tokens: Tensor,
    edge_mask: Tensor,
    baseline_logits: Tensor,
    candidate_mask: Tensor,
    residual_mode_eligible: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run evidence decoding plus a gated residual only after Stage-0 GO.

    The authorization check is deliberately before the residual forward.  A
    caller may construct a model for an isolated shape smoke, but it cannot
    obtain a non-zero residual without the aggregate gate and an explicitly
    opened pipeline configuration.
    """

    if not isinstance(pipeline, EviSOZEvidencePipeline):
        raise TypeError("pipeline must be EviSOZEvidencePipeline")
    authorization = require_stage0_training_authorized(
        gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
    )
    del authorization
    if alpha <= 0:
        raise ValueError("authorized residual forward requires alpha > 0")
    output, z1, delta, residual_gate = pipeline(
        node_tokens,
        node_mask,
        edge_tokens=edge_tokens,
        edge_mask=edge_mask,
        z0_node=baseline_logits,
        candidate_mask=candidate_mask,
        residual_mode_eligible=residual_mode_eligible,
        residual_enabled=True,
        alpha=alpha,
        stage0_status="GO",
    )
    if z1 is None or delta is None or residual_gate is None:
        raise RuntimeError("authorized residual forward did not return residual outputs")
    return z1, delta, residual_gate, output.evidence_tokens


__all__ = ["run_authorized_residual_forward"]
