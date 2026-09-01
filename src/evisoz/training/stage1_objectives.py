"""Audited Stage-1 evidence objectives for EviSOZ-LM.

The functions in this module are objective *ports*, not a training launcher.
They operate on already materialized tensors and never infer labels from text,
knowledge cards, or TCP22 edge endpoints.  The public ``compute_authorized_*``
wrapper checks the aggregate Stage-0 gate before evaluating any objective;
the lower-level ``compute_stage1_objective_bundle`` is provided for isolated
unit tests and for a future guarded trainer after the gate is closed.

All masks are explicit.  A missing channel/time cell is excluded from a loss,
never treated as a negative or reconstructed signal.  Teacher tensors are
detached at this boundary so a cached teacher cannot receive gradients.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .stage0_guard import require_stage0_training_authorized


def _finite_float(value: Tensor, *, name: str, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _mask(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")


def _masked_mean(values: Tensor, mask: Tensor, *, name: str) -> Tensor:
    _mask(mask, name=f"{name}_mask", shape=tuple(values.shape))
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        # An absent field is not a zero-valued negative.  Returning a graph
        # zero lets a caller combine sparse optional objectives safely while
        # retaining an auditable no-evaluable count at the caller.
        return values.sum() * 0.0
    return selected.mean()


def masked_latent_reconstruction_loss(
    student_tokens: Tensor,
    teacher_tokens: Tensor,
    reconstruction_mask: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """MSE on explicitly masked node/time cells against a frozen teacher.

    ``student_tokens`` and ``teacher_tokens`` have shape ``[B,C,T,D]``;
    ``reconstruction_mask`` and optional ``valid_mask`` have shape ``[B,C,T]``.
    The teacher is detached and is never used to create a channel label.
    """

    _finite_float(student_tokens, name="student_tokens", ndim=4)
    _finite_float(teacher_tokens, name="teacher_tokens", ndim=4)
    if tuple(student_tokens.shape) != tuple(teacher_tokens.shape):
        raise ValueError("student_tokens and teacher_tokens must have identical shapes")
    b, c, t, _ = student_tokens.shape
    _mask(reconstruction_mask, name="reconstruction_mask", shape=(b, c, t))
    effective = reconstruction_mask
    if valid_mask is not None:
        _mask(valid_mask, name="valid_mask", shape=(b, c, t))
        effective = effective & valid_mask
    per_cell = F.mse_loss(student_tokens, teacher_tokens.detach(), reduction="none").mean(dim=-1)
    return _masked_mean(per_cell, effective, name="masked_latent")


def motif_soft_target_loss(
    motif_logits: Tensor,
    target_probabilities: Tensor,
    cell_mask: Tensor,
) -> Tensor:
    """BCE for programmatic/derived motif probabilities.

    The target is a soft probability tensor ``[B,C,T,M]``.  It must be in
    ``[0,1]`` and is never interpreted as a hard SOZ label.  ``cell_mask``
    selects evaluable channel/time cells and has shape ``[B,C,T]``.
    """

    _finite_float(motif_logits, name="motif_logits", ndim=4)
    _finite_float(target_probabilities, name="target_probabilities", ndim=4)
    if tuple(motif_logits.shape) != tuple(target_probabilities.shape):
        raise ValueError("motif logits and target probabilities must match")
    if bool((target_probabilities < 0).any()) or bool((target_probabilities > 1).any()):
        raise ValueError("motif target probabilities must be in [0,1]")
    b, c, t, _ = motif_logits.shape
    _mask(cell_mask, name="cell_mask", shape=(b, c, t))
    per_cell = F.binary_cross_entropy_with_logits(
        motif_logits, target_probabilities.detach(), reduction="none"
    ).mean(dim=-1)
    return _masked_mean(per_cell, cell_mask, name="motif_soft_target")


def motif_teacher_kl_loss(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    cell_mask: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Distributional KL from a cached motif teacher to the student.

    A teacher may emit multi-label probabilities.  They are normalized over
    the motif axis solely for a stable KL objective; this does not promote a
    teacher event to a clinical fact or node target.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _finite_float(student_logits, name="student_logits", ndim=4)
    _finite_float(teacher_probabilities, name="teacher_probabilities", ndim=4)
    if tuple(student_logits.shape) != tuple(teacher_probabilities.shape):
        raise ValueError("student and teacher motif shapes must match")
    if bool((teacher_probabilities < 0).any()) or bool((teacher_probabilities > 1).any()):
        raise ValueError("teacher probabilities must be in [0,1]")
    b, c, t, _ = student_logits.shape
    _mask(cell_mask, name="cell_mask", shape=(b, c, t))
    teacher = teacher_probabilities.detach().clamp_min(0)
    denominator = teacher.sum(dim=-1, keepdim=True)
    # A zero teacher row has no information and is masked from the objective.
    informative = denominator.squeeze(-1) > epsilon
    teacher = teacher / denominator.clamp_min(epsilon)
    log_student = F.log_softmax(student_logits, dim=-1)
    per_cell = F.kl_div(log_student, teacher, reduction="none").sum(dim=-1)
    return _masked_mean(per_cell, cell_mask & informative, name="motif_teacher_kl")


def montage_consistency_loss(
    standard_tokens: Tensor,
    tcp22_projected_tokens: Tensor,
    standard_mask: Tensor,
    tcp22_mask: Tensor,
) -> Tensor:
    """Consistency on a shared node representation from two signal views.

    The TCP22 branch must already be projected to the 19-node semantic space;
    this function does not expand an edge into endpoint labels.  The caller
    supplies the intersection mask, so unavailable nodes remain absent.
    """

    _finite_float(standard_tokens, name="standard_tokens", ndim=4)
    _finite_float(tcp22_projected_tokens, name="tcp22_projected_tokens", ndim=4)
    if tuple(standard_tokens.shape) != tuple(tcp22_projected_tokens.shape):
        raise ValueError("montage consistency tensors must have identical shapes")
    b, c, t, _ = standard_tokens.shape
    _mask(standard_mask, name="standard_mask", shape=(b, c, t))
    _mask(tcp22_mask, name="tcp22_mask", shape=(b, c, t))
    per_cell = F.smooth_l1_loss(
        standard_tokens, tcp22_projected_tokens.detach(), reduction="none"
    ).mean(dim=-1)
    return _masked_mean(per_cell, standard_mask & tcp22_mask, name="montage_consistency")


def channel_edge_dropout_consistency_loss(
    reference_logits: Tensor,
    dropped_logits: Tensor,
    *,
    valid_channel_mask: Tensor | None = None,
) -> Tensor:
    """Keep node predictions stable under channel/edge dropout.

    Logits are ``[B,19]``.  The dropped view is detached because it is a
    stochastic target for the reference view; a future trainer may evaluate a
    symmetric variant explicitly, but must not silently change this contract.
    """

    _finite_float(reference_logits, name="reference_logits", ndim=2)
    _finite_float(dropped_logits, name="dropped_logits", ndim=2)
    if tuple(reference_logits.shape) != tuple(dropped_logits.shape):
        raise ValueError("dropout logits must have identical shapes")
    if reference_logits.shape[-1] != 19:
        raise ValueError("dropout consistency expects 19 Standard19 node logits")
    valid = torch.ones_like(reference_logits, dtype=torch.bool)
    if valid_channel_mask is not None:
        _mask(valid_channel_mask, name="valid_channel_mask", shape=tuple(reference_logits.shape))
        valid = valid_channel_mask
    per_node = F.mse_loss(
        torch.sigmoid(reference_logits), torch.sigmoid(dropped_logits.detach()), reduction="none"
    )
    return _masked_mean(per_node, valid, name="dropout_consistency")


def compute_stage1_objective_bundle(
    *,
    masked_latent: tuple[Tensor, Tensor, Tensor] | None = None,
    motif_soft: tuple[Tensor, Tensor, Tensor] | None = None,
    motif_teacher_kl: tuple[Tensor, Tensor, Tensor] | None = None,
    montage_consistency: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    dropout_consistency: tuple[Tensor, Tensor] | None = None,
    dropout_valid_mask: Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute explicitly supplied Stage-1 objective components.

    No component is synthesized when its inputs are absent.  This is the
    implementation-level counterpart of sparse field releases: callers can
    report ``by_objective`` and evaluable masks without turning missing data
    into negatives.  At least one component must be supplied.
    """

    provided = {
        "masked_latent": masked_latent,
        "motif_soft": motif_soft,
        "motif_teacher_kl": motif_teacher_kl,
        "montage_consistency": montage_consistency,
        "dropout_consistency": dropout_consistency,
    }
    if not any(value is not None for value in provided.values()):
        raise ValueError("at least one Stage-1 objective component is required")
    coefficients = {
        "masked_latent": 1.0,
        "motif_soft": 1.0,
        "motif_teacher_kl": 1.0,
        "montage_consistency": 1.0,
        "dropout_consistency": 1.0,
    }
    if weights is not None:
        unknown = set(weights).difference(coefficients)
        if unknown:
            raise ValueError(f"unknown Stage-1 objective weights: {sorted(unknown)}")
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"weight {name} must be finite")
            if float(value) < 0:
                raise ValueError(f"weight {name} must be non-negative")
            coefficients[name] = float(value)

    by_objective: dict[str, Tensor] = {}
    if masked_latent is not None:
        by_objective["masked_latent"] = masked_latent_reconstruction_loss(*masked_latent)
    if motif_soft is not None:
        by_objective["motif_soft"] = motif_soft_target_loss(*motif_soft)
    if motif_teacher_kl is not None:
        by_objective["motif_teacher_kl"] = motif_teacher_kl_loss(*motif_teacher_kl)
    if montage_consistency is not None:
        by_objective["montage_consistency"] = montage_consistency_loss(*montage_consistency)
    if dropout_consistency is not None:
        reference, dropped = dropout_consistency
        by_objective["dropout_consistency"] = channel_edge_dropout_consistency_loss(
            reference, dropped, valid_channel_mask=dropout_valid_mask
        )
    weighted = [coefficients[name] * value for name, value in by_objective.items() if coefficients[name] > 0]
    if not weighted:
        raise ValueError("all supplied Stage-1 objective weights are zero")
    total = torch.stack(weighted).sum()
    if not torch.isfinite(total):
        raise ValueError("Stage-1 objective total must be finite")
    return {
        "total": total,
        "by_objective": by_objective,
        "weights": {name: coefficients[name] for name in by_objective},
    }


def compute_authorized_stage1_objective(
    *,
    stage0_gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    **components: Any,
) -> dict[str, Any]:
    """Guard Stage-1 objective evaluation behind an aggregate Stage-0 GO."""

    authorization = require_stage0_training_authorized(
        stage0_gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
    )
    result = compute_stage1_objective_bundle(**components)
    result["authorization"] = authorization
    return result


__all__ = [
    "channel_edge_dropout_consistency_loss",
    "compute_authorized_stage1_objective",
    "compute_stage1_objective_bundle",
    "masked_latent_reconstruction_loss",
    "montage_consistency_loss",
    "motif_soft_target_loss",
    "motif_teacher_kl_loss",
]
