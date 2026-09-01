"""Guarded residual-localization objectives and epoch wiring.

The canonical v29 H/D prediction remains the identity reference.  This module
only supplies the *training boundary* for a future residual head: all direct
node targets are explicit ``[B,19]`` masks, while the preservation term keeps
the new distribution close to v29 on examples where the frozen baseline was
already correct.  A non-zero residual epoch is opened only after the
aggregate Stage-0 guard has passed; with the current real gate the guard fires
before a loader, model, optimizer, or record is constructed.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord

from .loader_entrypoint import open_authorized_training_records
from .stage0_guard import require_stage0_training_authorized


def _finite_logits(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[-1] != 19:
        raise ValueError(f"{name} must have shape [B,19]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


def _bool_mask(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")


def _masked_mean(values: Tensor, mask: Tensor, *, name: str) -> Tensor:
    _bool_mask(mask, name=f"{name}_mask", shape=tuple(values.shape))
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        # An absent target is not an all-negative target.  A graph zero lets a
        # caller combine optional terms and record the no-evaluable count.
        return values.sum() * 0.0
    return selected.mean()


def residual_node_localization_loss(
    updated_logits: Tensor,
    target: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Masked BCE for direct Standard19 node labels.

    ``target_mask`` is mandatory.  For incomplete-positive labels it should
    select only known positives; unspecified nodes therefore never become
    implicit negatives.  TCP22 edge channels are intentionally not accepted.
    """

    _finite_logits(updated_logits, name="updated_logits")
    if not isinstance(target, Tensor) or not target.is_floating_point() or tuple(target.shape) != tuple(updated_logits.shape):
        raise ValueError("target must be floating point with shape [B,19]")
    if not torch.isfinite(target).all() or bool((target < 0).any()) or bool((target > 1).any()):
        raise ValueError("target must be finite and in [0,1]")
    _bool_mask(target_mask, name="target_mask", shape=tuple(target.shape))
    per_node = F.binary_cross_entropy_with_logits(updated_logits, target.detach(), reduction="none")
    return _masked_mean(per_node, target_mask, name="residual_node_localization")


def baseline_preservation_kl_loss(
    baseline_logits: Tensor,
    updated_logits: Tensor,
    preserve_mask: Tensor,
) -> Tensor:
    """Keep updated probabilities close to frozen v29 on protected events."""

    _finite_logits(baseline_logits, name="baseline_logits")
    _finite_logits(updated_logits, name="updated_logits")
    if tuple(baseline_logits.shape) != tuple(updated_logits.shape):
        raise ValueError("baseline and updated logits must have identical shapes")
    _bool_mask(preserve_mask, name="preserve_mask", shape=(baseline_logits.shape[0],))
    baseline_prob = F.softmax(baseline_logits.detach(), dim=-1)
    per_event = F.kl_div(
        F.log_softmax(updated_logits, dim=-1),
        baseline_prob,
        reduction="none",
    ).sum(dim=-1)
    return _masked_mean(per_event, preserve_mask, name="baseline_preservation")


def compute_residual_localization_objective(
    *,
    baseline_logits: Tensor,
    updated_logits: Tensor,
    target: Tensor | None = None,
    target_mask: Tensor | None = None,
    preserve_mask: Tensor | None = None,
    localization_weight: float = 1.0,
    preservation_weight: float = 0.1,
) -> dict[str, object]:
    """Combine explicit residual localization and baseline-preservation terms."""

    _finite_logits(baseline_logits, name="baseline_logits")
    _finite_logits(updated_logits, name="updated_logits")
    if tuple(baseline_logits.shape) != tuple(updated_logits.shape):
        raise ValueError("baseline and updated logits must have identical shapes")
    for name, value in (("localization_weight", localization_weight), ("preservation_weight", preservation_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"{name} must be finite")
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    localization = updated_logits.sum() * 0.0
    if target is None or target_mask is None:
        if target is not None or target_mask is not None:
            raise ValueError("target and target_mask must be supplied together")
    elif float(localization_weight) > 0:
        localization = residual_node_localization_loss(updated_logits, target, target_mask)
    else:
        localization = updated_logits.sum() * 0.0
    if preserve_mask is None:
        preservation = updated_logits.sum() * 0.0
    else:
        preservation = baseline_preservation_kl_loss(baseline_logits, updated_logits, preserve_mask)
    weighted_localization = float(localization_weight) * localization
    weighted_preservation = float(preservation_weight) * preservation
    total = weighted_localization + weighted_preservation
    if not torch.isfinite(total):
        raise ValueError("residual localization objective must be finite")
    return {
        "total": total,
        "by_objective": {
            "localization": localization,
            "baseline_preservation": preservation,
        },
        "weights": {
            "localization": float(localization_weight),
            "baseline_preservation": float(preservation_weight),
        },
    }


ModelFactory = Callable[[], nn.Module]
OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
ResidualTrainingStep = Callable[[nn.Module, BoundEvidenceRecord], Tensor]


def run_authorized_residual_epoch(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    bound_evidence_root: str,
    private_examples_root: str,
    findings_claim_report_root: str,
    private_cohort_root: str,
    split_roster_path: str,
    model_factory: ModelFactory,
    optimizer_factory: OptimizerFactory,
    training_step: ResidualTrainingStep,
    alpha: float,
    evisoz_role: str = "development_cv",
) -> dict[str, object]:
    """Run a residual epoch only after aggregate Stage-0 authorization.

    The guard is deliberately first.  In particular, an invalid source path
    cannot be used to probe or open private records while Stage 0 is closed.
    """

    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not torch.isfinite(torch.tensor(float(alpha))):
        raise ValueError("alpha must be finite")
    if float(alpha) <= 0:
        raise ValueError("formal residual training requires alpha > 0")
    authorization = require_stage0_training_authorized(
        gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
    )
    # Keep loader construction after the guard.  This is part of the privacy
    # and holdout contract, not merely an optimization.
    records = open_authorized_training_records(
        gate=gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
        bound_evidence_root=bound_evidence_root,
        private_examples_root=private_examples_root,
        findings_claim_report_root=findings_claim_report_root,
        private_cohort_root=private_cohort_root,
        split_roster_path=split_roster_path,
        evisoz_role=evisoz_role,
    )[1]
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    optimizer = optimizer_factory(model)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_factory must return torch.optim.Optimizer")
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
        raise ValueError("authorized residual iterator yielded no records")
    return {
        "status": "authorized_residual_epoch_complete",
        "authorization": authorization,
        "event_count": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "alpha": float(alpha),
        "baseline_preservation_required": True,
        "evisoz_role": evisoz_role,
    }


__all__ = [
    "baseline_preservation_kl_loss",
    "compute_residual_localization_objective",
    "residual_node_localization_loss",
    "run_authorized_residual_epoch",
]
