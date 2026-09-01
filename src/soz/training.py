"""Minimal framework-agnostic training steps for the staged SOZ pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .data.batching import PatientBagDataset, PatientEvidenceBatch
from .losses import PatientLevelSOZObjective, SOZLossOutput
from .models.reasoner import AdditiveEvidenceReasoner, ReasonerOutput
from .reasoner_calibration import uncalibrated_training_probabilities


@dataclass(frozen=True)
class ReasonerStepOutput:
    reasoner: ReasonerOutput
    patient_logits: torch.Tensor
    patient_probabilities: torch.Tensor
    event_counts: torch.Tensor
    loss: SOZLossOutput


@dataclass(frozen=True)
class ReasonerEpochOutput:
    """Patient-macro epoch receipt and detached channel-level predictions."""

    mean_total_loss: float
    mean_bce_loss: float
    mean_ranking_loss: float
    n_patients: int
    n_events: int
    patient_ids: tuple[str, ...]
    patient_logits: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor

    @property
    def patient_probabilities(self) -> torch.Tensor:
        """Uncalibrated probabilities used before the post-freeze dev stage."""

        return uncalibrated_training_probabilities(self.patient_logits)


def reasoner_training_step(
    model: AdditiveEvidenceReasoner,
    batch: PatientEvidenceBatch,
    objective: PatientLevelSOZObjective,
) -> ReasonerStepOutput:
    """Run the complete event-evidence → one-loss-per-patient computation."""

    reasoner_output = model(batch.evidence)
    aggregation = batch.aggregate(reasoner_output.event_logits)
    loss = objective(
        aggregation.logits,
        batch.targets,
        batch.target_mask,
    )
    return ReasonerStepOutput(
        reasoner=reasoner_output,
        patient_logits=aggregation.logits,
        patient_probabilities=uncalibrated_training_probabilities(
            aggregation.logits
        ),
        event_counts=aggregation.event_counts,
        loss=loss,
    )


def _model_device(model: AdditiveEvidenceReasoner) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - model always has parameters
        raise ValueError("Reasoner has no trainable parameters") from exc


def _validate_reasoner_optimizer(
    model: AdditiveEvidenceReasoner,
    optimizer: torch.optim.Optimizer,
) -> None:
    model_parameters = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != model_parameters:
        raise ValueError(
            "Reasoner optimizer parameters must exactly equal reasoner trainable parameters"
        )


def _epoch_output(
    *,
    totals: list[float],
    bces: list[float],
    rankings: list[float],
    patient_ids: list[str],
    event_count: int,
    logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> ReasonerEpochOutput:
    if not totals:
        raise ValueError("Reasoner epoch produced no patient losses")
    return ReasonerEpochOutput(
        mean_total_loss=sum(totals) / len(totals),
        mean_bce_loss=sum(bces) / len(bces),
        mean_ranking_loss=sum(rankings) / len(rankings),
        n_patients=len(patient_ids),
        n_events=event_count,
        patient_ids=tuple(patient_ids),
        patient_logits=torch.cat(logits, dim=0),
        targets=torch.cat(targets, dim=0),
        target_mask=torch.cat(masks, dim=0),
    )


def train_reasoner_epoch(
    model: AdditiveEvidenceReasoner,
    dataset: PatientBagDataset,
    optimizer: torch.optim.Optimizer,
    objective: PatientLevelSOZObjective,
    *,
    patient_order: Sequence[object] | None = None,
    max_grad_norm: float | None = 1.0,
) -> ReasonerEpochOutput:
    """Fit one source-train epoch with one update per complete OOF patient bag.

    Returned training logits are the pre-update predictions seen at each
    patient step and must not be reported as final training performance.
    """

    if dataset.model_split != "source_train":
        raise ValueError("Reasoner fitting is restricted to source_train OOF evidence")
    if max_grad_norm is not None and (
        not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0
    ):
        raise ValueError("max_grad_norm must be positive or None")
    _validate_reasoner_optimizer(model, optimizer)
    model.train()
    device = _model_device(model)
    totals: list[float] = []
    bces: list[float] = []
    rankings: list[float] = []
    patient_ids: list[str] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    n_events = 0
    for raw_batch in dataset.iter_epoch(patient_order):
        if len(raw_batch.patient_ids) != 1:
            raise RuntimeError("PatientBagDataset must yield exactly one complete patient")
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        step = reasoner_training_step(model, batch, objective)
        step.loss.total.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        totals.append(float(step.loss.total.detach().cpu()))
        bces.append(float(step.loss.bce.detach().cpu()))
        rankings.append(float(step.loss.ranking.detach().cpu()))
        patient_ids.extend(batch.patient_ids)
        logits.append(step.patient_logits.detach().cpu())
        targets.append(batch.targets.detach().cpu())
        masks.append(batch.target_mask.detach().cpu())
        n_events += int(step.event_counts.sum().item())
    return _epoch_output(
        totals=totals,
        bces=bces,
        rankings=rankings,
        patient_ids=patient_ids,
        event_count=n_events,
        logits=logits,
        targets=targets,
        masks=masks,
    )


def train_formal_reasoner_epoch(
    model: AdditiveEvidenceReasoner,
    dataset: PatientBagDataset,
    optimizer: torch.optim.Optimizer,
    objective: PatientLevelSOZObjective,
    *,
    patient_order: Sequence[object] | None = None,
    max_grad_norm: float | None = 1.0,
) -> ReasonerEpochOutput:
    """Formal entry point: reject caches lacking independent OOF authority."""

    from .evidence_authorization import AuthorizedPatientBagDataset

    if not isinstance(dataset, AuthorizedPatientBagDataset):
        raise TypeError(
            "Formal reasoner fitting requires AuthorizedPatientBagDataset"
        )
    return train_reasoner_epoch(
        model,
        dataset,
        optimizer,
        objective,
        patient_order=patient_order,
        max_grad_norm=max_grad_norm,
    )


@torch.no_grad()
def evaluate_reasoner_epoch(
    model: AdditiveEvidenceReasoner,
    dataset: PatientBagDataset,
    objective: PatientLevelSOZObjective,
    *,
    patient_order: Sequence[object] | None = None,
) -> ReasonerEpochOutput:
    """Evaluate a complete public split once per patient without mutation."""

    if dataset.model_split not in {"source_train", "source_dev", "source_eval"}:
        raise ValueError("Unsupported reasoner evaluation split")
    model.eval()
    device = _model_device(model)
    totals: list[float] = []
    bces: list[float] = []
    rankings: list[float] = []
    patient_ids: list[str] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    n_events = 0
    for raw_batch in dataset.iter_epoch(patient_order):
        if len(raw_batch.patient_ids) != 1:
            raise RuntimeError("PatientBagDataset must yield exactly one complete patient")
        batch = raw_batch.to(device)
        step = reasoner_training_step(model, batch, objective)
        totals.append(float(step.loss.total.detach().cpu()))
        bces.append(float(step.loss.bce.detach().cpu()))
        rankings.append(float(step.loss.ranking.detach().cpu()))
        patient_ids.extend(batch.patient_ids)
        logits.append(step.patient_logits.detach().cpu())
        targets.append(batch.targets.detach().cpu())
        masks.append(batch.target_mask.detach().cpu())
        n_events += int(step.event_counts.sum().item())
    return _epoch_output(
        totals=totals,
        bces=bces,
        rankings=rankings,
        patient_ids=patient_ids,
        event_count=n_events,
        logits=logits,
        targets=targets,
        masks=masks,
    )


@torch.no_grad()
def evaluate_formal_reasoner_epoch(
    model: AdditiveEvidenceReasoner,
    dataset: PatientBagDataset,
    objective: PatientLevelSOZObjective,
    *,
    patient_order: Sequence[object] | None = None,
) -> ReasonerEpochOutput:
    """Formal evaluation entry point with the same authorization firewall."""

    from .evidence_authorization import AuthorizedPatientBagDataset

    if not isinstance(dataset, AuthorizedPatientBagDataset):
        raise TypeError(
            "Formal reasoner evaluation requires AuthorizedPatientBagDataset"
        )
    return evaluate_reasoner_epoch(
        model,
        dataset,
        objective,
        patient_order=patient_order,
    )
