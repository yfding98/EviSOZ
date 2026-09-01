"""Source-native, patient-balanced losses for the evidence heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models.concept_heads import EvolutionHeadOutput


def _validate_patient_ids(patient_ids: torch.Tensor, n_examples: int, device: torch.device) -> None:
    if patient_ids.ndim != 1 or patient_ids.shape[0] != n_examples:
        raise ValueError("patient_ids must have shape [B]")
    if patient_ids.dtype != torch.long:
        raise TypeError("patient_ids must be torch.long")
    if patient_ids.device != device:
        raise ValueError("patient_ids and losses must share a device")


def _patient_balanced_masked_mean(
    element_loss: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
    *,
    allow_empty: bool = False,
) -> torch.Tensor:
    if tuple(element_loss.shape) != tuple(mask.shape):
        raise ValueError("element_loss and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be torch.bool")
    _validate_patient_ids(patient_ids, element_loss.shape[0], element_loss.device)
    patient_losses: list[torch.Tensor] = []
    for patient_id in torch.unique(patient_ids, sorted=True):
        example_mask = patient_ids == patient_id
        observed = mask[example_mask]
        if observed.any():
            patient_losses.append(element_loss[example_mask][observed].mean())
    if not patient_losses:
        if allow_empty:
            return element_loss.sum() * 0.0
        raise ValueError("No observed labels are available for this concept loss")
    return torch.stack(patient_losses).mean()


def morphology_ce6_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    patient_ids: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patient-balanced CE over native TUEV CE6 edge-time labels."""

    if logits.ndim != 4 or logits.shape[1] != 20 or logits.shape[-1] != 6:
        raise ValueError("Morphology logits must have shape [B,20,T,6]")
    expected = tuple(logits.shape[:-1])
    if tuple(labels.shape) != expected or tuple(label_mask.shape) != expected:
        raise ValueError("Morphology labels/mask must have shape [B,20,T]")
    if labels.dtype != torch.long or label_mask.dtype != torch.bool:
        raise TypeError("Morphology labels must be long and mask must be bool")
    observed_labels = labels[label_mask]
    if observed_labels.numel() and not torch.all((observed_labels >= 0) & (observed_labels < 6)):
        raise ValueError("Observed morphology labels must be in [0,5]")
    safe_labels = torch.where(label_mask, labels, 0)
    element_loss = F.cross_entropy(
        logits.movedim(-1, 1),
        safe_labels,
        weight=class_weights,
        reduction="none",
    )
    return _patient_balanced_masked_mean(
        element_loss, label_mask, patient_ids
    )


def ictal_involvement_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: torch.Tensor,
    *,
    positive_weight: float | None = None,
) -> torch.Tensor:
    """Patient-balanced masked BCE on explicit TUSZ edge-time coverage."""

    if logits.ndim != 4 or logits.shape[1] != 20 or logits.shape[-1] != 1:
        raise ValueError("Ictal logits must have shape [B,20,T,1]")
    squeezed = logits.squeeze(-1)
    if tuple(targets.shape) != tuple(squeezed.shape) or tuple(target_mask.shape) != tuple(squeezed.shape):
        raise ValueError("Ictal targets/mask must have shape [B,20,T]")
    if not targets.is_floating_point() or target_mask.dtype != torch.bool:
        raise TypeError("Ictal targets must be float and mask must be bool")
    observed = targets[target_mask]
    if not torch.isfinite(observed).all():
        raise ValueError("Observed ictal targets must be finite")
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Observed ictal targets must be binary")
    pos_weight = None
    if positive_weight is not None:
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        pos_weight = squeezed.new_tensor(float(positive_weight))
    safe_targets = torch.where(target_mask, targets, 0.0)
    element_loss = F.binary_cross_entropy_with_logits(
        squeezed, safe_targets, reduction="none", pos_weight=pos_weight
    )
    return _patient_balanced_masked_mean(
        element_loss, target_mask, patient_ids
    )


@dataclass(frozen=True)
class EvolutionLossOutput:
    total: torch.Tensor
    descriptor: torch.Tensor
    future_change: torch.Tensor


def temporal_evolution_loss(
    output: EvolutionHeadOutput,
    descriptor_targets: torch.Tensor,
    descriptor_mask: torch.Tensor,
    patient_ids: torch.Tensor,
    *,
    future_weight: float = 0.25,
) -> EvolutionLossOutput:
    """Smooth-L1 descriptors plus the single cross-call future-change loss."""

    if future_weight < 0:
        raise ValueError("future_weight must be non-negative")
    if tuple(descriptor_targets.shape) != tuple(output.descriptors.shape):
        raise ValueError("Descriptor targets must match [B,19,Q,6] predictions")
    if tuple(descriptor_mask.shape) != tuple(output.descriptors.shape[:-1]):
        raise ValueError("descriptor_mask must have shape [B,19,Q]")
    if descriptor_mask.dtype != torch.bool:
        raise TypeError("descriptor_mask must be torch.bool")
    observed_descriptors = descriptor_targets[descriptor_mask]
    if not torch.isfinite(observed_descriptors).all():
        raise ValueError("Observed descriptor targets must be finite")
    safe_descriptor_targets = torch.where(
        descriptor_mask.unsqueeze(-1), descriptor_targets, 0.0
    )
    descriptor_element = F.smooth_l1_loss(
        output.descriptors, safe_descriptor_targets, reduction="none"
    ).mean(dim=-1)
    descriptor = _patient_balanced_masked_mean(
        descriptor_element, descriptor_mask, patient_ids
    )

    source = output.future_source_tiles
    target = output.future_target_tiles
    future_targets = safe_descriptor_targets.index_select(2, target) - safe_descriptor_targets.index_select(2, source)
    future_mask = descriptor_mask.index_select(2, target) & descriptor_mask.index_select(2, source)
    if tuple(output.future_change.shape) != tuple(future_targets.shape):
        raise ValueError("Future-change predictions do not match cross-call boundaries")
    future_element = F.smooth_l1_loss(
        output.future_change, future_targets, reduction="none"
    ).mean(dim=-1)
    future_change = _patient_balanced_masked_mean(
        future_element,
        future_mask,
        patient_ids,
        allow_empty=True,
    )
    total = descriptor + float(future_weight) * future_change
    return EvolutionLossOutput(
        total=total, descriptor=descriptor, future_change=future_change
    )
