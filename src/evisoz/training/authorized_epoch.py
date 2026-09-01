"""Generic authorized EviSOZ training loop.

This module intentionally contains no private-data or Qwen-specific loader.
It provides the final guard boundary that a future Stage-1/Stage-2 trainer
must use: authorization is checked, then (and only then) the model, optimizer
and bound-evidence iterator are opened.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord

from .loader_entrypoint import open_authorized_training_records


ModelFactory = Callable[[], nn.Module]
OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
TrainingStep = Callable[[nn.Module, BoundEvidenceRecord], Tensor]


def run_authorized_evidence_epoch(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    requested_actions: tuple[str, ...],
    bound_evidence_root: str,
    private_examples_root: str,
    findings_claim_report_root: str,
    private_cohort_root: str,
    split_roster_path: str,
    model_factory: ModelFactory,
    optimizer_factory: OptimizerFactory,
    training_step: TrainingStep,
    evisoz_role: str = "development_cv",
) -> dict[str, object]:
    """Run one optimizer epoch only after the aggregate Stage-0 gate passes."""

    authorization, records = open_authorized_training_records(
        gate=gate,
        pipeline_config=pipeline_config,
        requested_actions=requested_actions,
        bound_evidence_root=bound_evidence_root,
        private_examples_root=private_examples_root,
        findings_claim_report_root=findings_claim_report_root,
        private_cohort_root=private_cohort_root,
        split_roster_path=split_roster_path,
        evisoz_role=evisoz_role,
    )
    # These factories are deliberately below the guard call.  Under NO_GO no
    # model weights, optimizer state or data record is opened.
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    optimizer = optimizer_factory(model)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_factory must return a torch optimizer")
    model.train()
    losses: list[float] = []
    for record in records:
        optimizer.zero_grad(set_to_none=True)
        loss = training_step(model, record)
        if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("training_step must return a finite scalar tensor")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("authorized training iterator yielded no records")
    return {
        "status": "authorized_evidence_epoch_complete",
        "authorization": authorization,
        "event_count": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "role": evisoz_role,
    }


__all__ = ["run_authorized_evidence_epoch"]
