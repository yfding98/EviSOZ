"""Threshold-free, patient-macro fidelity metrics for learned concepts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IctalConceptMetrics:
    """Patient-macro metrics on observed TUSZ involvement labels only.

    AUROC and average precision are averaged over patients containing both an
    explicit positive and an explicit negative.  They are ``None`` when no
    patient is evaluable; no missing label is imputed as background.
    """

    patient_macro_bce: float
    patient_macro_brier: float
    patient_macro_auroc: float | None
    patient_macro_average_precision: float | None
    n_patients: int
    n_discrimination_patients: int
    n_observed_labels: int
    n_positive_labels: int
    n_negative_labels: int

    def __post_init__(self) -> None:
        if self.n_patients < 1 or self.n_observed_labels < 1:
            raise ValueError("Ictal concept metrics require observed patient labels")
        if not 0 <= self.n_discrimination_patients <= self.n_patients:
            raise ValueError("Invalid discrimination-patient count")
        if self.n_positive_labels < 0 or self.n_negative_labels < 0:
            raise ValueError("Ictal label counts cannot be negative")
        if self.n_positive_labels + self.n_negative_labels != self.n_observed_labels:
            raise ValueError("Positive/negative counts do not match observed labels")
        finite_metrics = (self.patient_macro_bce, self.patient_macro_brier)
        if any(not math.isfinite(value) for value in finite_metrics):
            raise ValueError("BCE and Brier metrics must be finite")
        if not 0.0 <= self.patient_macro_brier <= 1.0:
            raise ValueError("Brier score must lie in [0,1]")
        for name in ("patient_macro_auroc", "patient_macro_average_precision"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be None or lie in [0,1]")
        if self.n_discrimination_patients == 0:
            if self.patient_macro_auroc is not None or self.patient_macro_average_precision is not None:
                raise ValueError("Undefined discrimination metrics must be represented by None")
        elif self.patient_macro_auroc is None or self.patient_macro_average_precision is None:
            raise ValueError("Evaluable discrimination metrics cannot be missing")


def _binary_auroc_with_ties(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    ranks = torch.empty_like(sorted_scores, dtype=torch.float64)
    start = 0
    n_values = int(sorted_scores.numel())
    while start < n_values:
        stop = start + 1
        while stop < n_values and bool(sorted_scores[stop] == sorted_scores[start]):
            stop += 1
        # Rank positions are one-based; all tied observations get their mean.
        ranks[start:stop] = ((start + 1) + stop) / 2.0
        start = stop
    positives = int(sorted_targets.sum().item())
    negatives = n_values - positives
    positive_rank_sum = ranks[sorted_targets].sum()
    value = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return float(value.item())


def _binary_average_precision_with_ties(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Average precision evaluated at distinct-score thresholds."""

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    positives = int(sorted_targets.sum().item())
    cumulative_true = 0
    cumulative_total = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    n_values = int(sorted_scores.numel())
    while start < n_values:
        stop = start + 1
        while stop < n_values and bool(sorted_scores[stop] == sorted_scores[start]):
            stop += 1
        group_true = int(sorted_targets[start:stop].sum().item())
        cumulative_true += group_true
        cumulative_total += stop - start
        recall = cumulative_true / positives
        precision = cumulative_true / cumulative_total
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(average_precision)


def patient_macro_ictal_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> IctalConceptMetrics:
    """Evaluate edge-time involvement without coercing unknown bins to zero."""

    if logits.ndim != 4 or logits.shape[1] != 20 or logits.shape[-1] != 1:
        raise ValueError("Ictal logits must have shape [E,20,T,1]")
    squeezed = logits.squeeze(-1)
    if tuple(targets.shape) != tuple(squeezed.shape) or tuple(target_mask.shape) != tuple(
        squeezed.shape
    ):
        raise ValueError("Ictal targets/mask must have shape [E,20,T]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("Ictal logits and targets must be floating-point")
    if target_mask.dtype != torch.bool:
        raise TypeError("Ictal target_mask must be bool")
    if patient_ids.ndim != 1 or patient_ids.shape[0] != logits.shape[0]:
        raise ValueError("patient_ids must have shape [E]")
    if patient_ids.dtype != torch.long:
        raise TypeError("patient_ids must be torch.long")
    devices = {logits.device, targets.device, target_mask.device, patient_ids.device}
    if len(devices) != 1:
        raise ValueError("Metric tensors must share one device")
    if not torch.isfinite(logits).all():
        raise ValueError("Ictal logits must be finite")
    observed_targets = targets[target_mask]
    if not torch.isfinite(observed_targets).all() or (
        observed_targets.numel()
        and not torch.all((observed_targets == 0) | (observed_targets == 1))
    ):
        raise ValueError("Observed ictal targets must be finite and binary")

    bces: list[float] = []
    briers: list[float] = []
    aurocs: list[float] = []
    average_precisions: list[float] = []
    positive_count = 0
    negative_count = 0
    for patient_id in torch.unique(patient_ids, sorted=True):
        patient_examples = patient_ids == patient_id
        patient_mask = target_mask[patient_examples]
        if not patient_mask.any():
            continue
        patient_logits = squeezed[patient_examples][patient_mask]
        patient_targets = targets[patient_examples][patient_mask]
        bces.append(
            float(
                F.binary_cross_entropy_with_logits(
                    patient_logits, patient_targets, reduction="mean"
                )
                .detach()
                .cpu()
            )
        )
        probabilities = patient_logits.sigmoid()
        briers.append(
            float(((probabilities - patient_targets) ** 2).mean().detach().cpu())
        )
        patient_positive = int(patient_targets.sum().item())
        patient_negative = int(patient_targets.numel()) - patient_positive
        positive_count += patient_positive
        negative_count += patient_negative
        if patient_positive and patient_negative:
            cpu_scores = patient_logits.detach().cpu().to(torch.float64)
            cpu_targets = patient_targets.detach().cpu().to(torch.bool)
            aurocs.append(_binary_auroc_with_ties(cpu_scores, cpu_targets))
            average_precisions.append(
                _binary_average_precision_with_ties(cpu_scores, cpu_targets)
            )
    if not bces:
        raise ValueError("No observed ictal labels are available for evaluation")
    return IctalConceptMetrics(
        patient_macro_bce=sum(bces) / len(bces),
        patient_macro_brier=sum(briers) / len(briers),
        patient_macro_auroc=None if not aurocs else sum(aurocs) / len(aurocs),
        patient_macro_average_precision=(
            None
            if not average_precisions
            else sum(average_precisions) / len(average_precisions)
        ),
        n_patients=len(bces),
        n_discrimination_patients=len(aurocs),
        n_observed_labels=positive_count + negative_count,
        n_positive_labels=positive_count,
        n_negative_labels=negative_count,
    )


__all__ = ["IctalConceptMetrics", "patient_macro_ictal_metrics"]
