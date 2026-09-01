"""Patient-balanced objectives for the DeepSOZ reference task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import N_STANDARD_CHANNELS


def _validate_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    require_positive: bool,
) -> None:
    expected = (logits.shape[0], N_STANDARD_CHANNELS)
    if logits.ndim != 2 or tuple(logits.shape) != expected:
        raise ValueError(f"logits must have shape [P,19], got {tuple(logits.shape)}")
    if tuple(targets.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError("targets and target_mask must have the same [P,19] shape")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be torch.bool")
    if not targets.is_floating_point() or not logits.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
    if logits.device != targets.device or logits.device != target_mask.device:
        raise ValueError("logits, targets, and target_mask must share a device")
    if logits.shape[0] < 1:
        raise ValueError("At least one patient is required")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not torch.isfinite(targets[target_mask]).all():
        raise ValueError("Observed targets must be finite")
    observed = targets[target_mask]
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Observed targets must be binary")
    if not target_mask.any(dim=1).all():
        bad = (~target_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Patients without observed targets are not eligible: {bad}")
    if require_positive:
        known_positive = ((targets == 1) & target_mask).any(dim=1)
        if not known_positive.all():
            bad = (~known_positive).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Patients without an in-head observed positive must be routed out "
                f"of the localization loss: {bad}"
            )


def masked_patient_balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    require_positive: bool = True,
) -> torch.Tensor:
    """Compute one class-balanced BCE contribution per patient.

    Positive and dataset-complement-negative channels each receive half of a
    patient's weight when both are present.  Repeated events never enter this
    function and therefore cannot increase a patient's statistical weight.

    ``binary_cross_entropy_with_logits`` is the numerically stable evaluation
    of BCE on ``sigmoid(raw_patient_logits)``.  No temperature, affine bias, or
    development-fitted calibrator is accepted by this training objective.
    """

    _validate_targets(
        logits, targets, target_mask, require_positive=require_positive
    )
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    patient_losses: list[torch.Tensor] = []
    for patient_index in range(logits.shape[0]):
        observed = target_mask[patient_index]
        positive = observed & (targets[patient_index] == 1)
        negative = observed & (targets[patient_index] == 0)
        class_terms: list[torch.Tensor] = []
        if positive.any():
            class_terms.append(elementwise[patient_index][positive].mean())
        if negative.any():
            class_terms.append(elementwise[patient_index][negative].mean())
        patient_losses.append(torch.stack(class_terms).mean())
    return torch.stack(patient_losses).mean()


def masked_pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    margin: float = 0.0,
    require_positive: bool = True,
) -> torch.Tensor:
    """Rank each observed positive above each operational complement."""

    _validate_targets(
        logits, targets, target_mask, require_positive=require_positive
    )
    patient_losses: list[torch.Tensor] = []
    for patient_index in range(logits.shape[0]):
        observed = target_mask[patient_index]
        positive_logits = logits[patient_index][observed & (targets[patient_index] == 1)]
        negative_logits = logits[patient_index][observed & (targets[patient_index] == 0)]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue
        differences = positive_logits[:, None] - negative_logits[None, :]
        patient_losses.append(F.softplus(float(margin) - differences).mean())
    if not patient_losses:
        return logits.sum() * 0.0
    return torch.stack(patient_losses).mean()


@dataclass(frozen=True)
class SOZLossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    ranking: torch.Tensor


class PatientLevelSOZObjective(nn.Module):
    """Masked patient-balanced BCE plus pairwise channel ranking."""

    def __init__(
        self,
        *,
        ranking_weight: float = 0.25,
        ranking_margin: float = 0.0,
        require_positive: bool = True,
    ) -> None:
        super().__init__()
        if ranking_weight < 0:
            raise ValueError("ranking_weight must be non-negative")
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)
        self.require_positive = bool(require_positive)

    def forward(
        self,
        patient_logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> SOZLossOutput:
        bce = masked_patient_balanced_bce_with_logits(
            patient_logits,
            targets,
            target_mask,
            require_positive=self.require_positive,
        )
        ranking = masked_pairwise_ranking_loss(
            patient_logits,
            targets,
            target_mask,
            margin=self.ranking_margin,
            require_positive=self.require_positive,
        )
        total = bce + self.ranking_weight * ranking
        return SOZLossOutput(total=total, bce=bce, ranking=ranking)
