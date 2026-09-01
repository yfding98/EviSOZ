"""Differentiable patient-bag bridge for the single v11-B LaBraM PEFT trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .geometry import N_STANDARD_CHANNELS
from .v11_reasoner import (
    FoldFeatureTransform,
    SharedPositiveSetReasoner,
    TransformedPatientFeatures,
    V11_CANDIDATE_MASK,
    V11_FINE_DIM,
    V11_H_PCA_DIM,
    V11_H_RAW_DIM,
    positive_set_mass_loss,
)


V11B_CALLS_PER_EVENT = 15
V11B_SECONDS_PER_CALL = 4
V11B_TOKEN_DIM = 200


def differentiable_suffix_phase_contrasts(node_tokens: torch.Tensor) -> torch.Tensor:
    """Convert post-block11 tokens to the frozen three-phase H carrier.

    Unlike the v11-A cache extractor, this function intentionally preserves
    autograd so an event-level upstream gradient can be replayed through the
    two trainable LoRA-parametrized suffix blocks.
    """

    if not isinstance(node_tokens, torch.Tensor) or not node_tokens.is_floating_point():
        raise TypeError("post-suffix node tokens must be floating point")
    expected_tail = (
        N_STANDARD_CHANNELS,
        V11B_CALLS_PER_EVENT,
        V11B_SECONDS_PER_CALL,
        V11B_TOKEN_DIM,
    )
    if node_tokens.ndim != 5 or tuple(node_tokens.shape[1:]) != expected_tail:
        raise ValueError("post-suffix node tokens must have shape [E,19,15,4,200]")
    if node_tokens.shape[0] < 1 or not torch.isfinite(node_tokens).all():
        raise ValueError("post-suffix node tokens must be non-empty and finite")
    seconds = node_tokens.reshape(
        node_tokens.shape[0], N_STANDARD_CHANNELS, 60, V11B_TOKEN_DIM
    )
    baseline = seconds[:, :, 0:12].mean(dim=2)
    onset = seconds[:, :, 12:20].mean(dim=2)
    early = seconds[:, :, 12:28].mean(dim=2)
    late = seconds[:, :, 28:52].mean(dim=2)
    result = torch.cat((onset - baseline, early - baseline, late - early), dim=-1)
    if tuple(result.shape) != (
        node_tokens.shape[0],
        N_STANDARD_CHANNELS,
        V11_H_RAW_DIM,
    ) or not torch.isfinite(result).all():
        raise RuntimeError("v11-B suffix phase contrast carrier drifted")
    return result.contiguous()


def differentiable_pool_complete_patient_bags(
    event_features: torch.Tensor,
    event_patient_index: torch.Tensor,
    n_patients: int,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable counterpart of v11 reliability/winsor pooling."""

    if not isinstance(event_features, torch.Tensor) or not event_features.is_floating_point():
        raise TypeError("event_features must be floating point")
    if event_features.ndim != 3 or event_features.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("event_features must have shape [E,19,D]")
    events = int(event_features.shape[0])
    if events < 1 or not torch.isfinite(event_features).all():
        raise ValueError("event_features must be non-empty and finite")
    if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if tuple(event_patient_index.shape) != (events,) or event_patient_index.dtype != torch.long:
        raise TypeError("event_patient_index must be long [E]")
    if event_patient_index.device != event_features.device:
        raise ValueError("event_patient_index and event features must share a device")
    if events < n_patients or int(event_patient_index.min()) != 0 or (
        int(event_patient_index.max()) != n_patients - 1
    ) or torch.unique(event_patient_index).numel() != n_patients:
        raise ValueError("event_patient_index must be a contiguous complete roster")
    if not isinstance(reliability, torch.Tensor) or not reliability.is_floating_point():
        raise TypeError("reliability must be floating point")
    if tuple(reliability.shape) != (events, N_STANDARD_CHANNELS) or (
        reliability.device != event_features.device
    ):
        raise ValueError("reliability must align as [E,19]")
    if reliability.requires_grad:
        raise ValueError("target-free reliability must remain detached")
    if not torch.isfinite(reliability).all() or bool(
        ((reliability < 0) | (reliability > 1)).any()
    ):
        raise ValueError("reliability must be finite in [0,1]")

    weights = reliability.to(event_features.dtype).clamp_min(0.1)
    pooled: list[torch.Tensor] = []
    for patient in range(n_patients):
        selector = event_patient_index == patient
        values = event_features[selector]
        patient_weights = weights[selector]
        if values.shape[0] >= 3:
            lower = torch.quantile(values, 0.1, dim=0)
            upper = torch.quantile(values, 0.9, dim=0)
            values = torch.minimum(torch.maximum(values, lower), upper)
        denominator = patient_weights.sum(dim=0).clamp_min(1e-6)
        pooled.append(
            (values * patient_weights.unsqueeze(-1)).sum(dim=0)
            / denominator.unsqueeze(-1)
        )
    result = torch.stack(pooled).contiguous()
    if tuple(result.shape) != (
        n_patients,
        N_STANDARD_CHANNELS,
        event_features.shape[2],
    ) or not torch.isfinite(result).all():
        raise RuntimeError("v11-B differentiable patient pooling failed")
    return result


def apply_fold_transform_differentiable(
    h_patient: torch.Tensor,
    fine_patient: torch.Tensor,
    transform: FoldFeatureTransform,
) -> TransformedPatientFeatures:
    """Apply an already fitted outer-train transform without detaching H."""

    if not isinstance(transform, FoldFeatureTransform):
        raise TypeError("transform must be a FoldFeatureTransform")
    if not isinstance(h_patient, torch.Tensor) or not h_patient.is_floating_point():
        raise TypeError("patient H must be floating point")
    if not isinstance(fine_patient, torch.Tensor) or not fine_patient.is_floating_point():
        raise TypeError("patient fine features must be floating point")
    if tuple(h_patient.shape[1:]) != (N_STANDARD_CHANNELS, V11_H_RAW_DIM) or (
        tuple(fine_patient.shape[1:]) != (N_STANDARD_CHANNELS, V11_FINE_DIM)
    ):
        raise ValueError("v11-B patient features must be [P,19,600] and [P,19,20]")
    if h_patient.shape[0] != fine_patient.shape[0] or h_patient.device != fine_patient.device:
        raise ValueError("v11-B H/fine patient carriers must align")
    if not torch.isfinite(h_patient).all() or not torch.isfinite(fine_patient).all():
        raise ValueError("v11-B patient features must be finite")

    device = h_patient.device
    h_dtype = h_patient.dtype
    fine_dtype = fine_patient.dtype
    h_center = transform.h_center.to(device=device, dtype=h_dtype)
    h_scale = transform.h_scale.to(device=device, dtype=h_dtype)
    h_mean = transform.h_pca_mean.to(device=device, dtype=h_dtype)
    components = transform.h_components.to(device=device, dtype=h_dtype)
    fine_center = transform.fine_center.to(device=device, dtype=fine_dtype)
    fine_scale = transform.fine_scale.to(device=device, dtype=fine_dtype)
    h = torch.matmul((h_patient - h_center) / h_scale - h_mean, components)
    fine = (fine_patient - fine_center) / fine_scale
    if tuple(h.shape[1:]) != (N_STANDARD_CHANNELS, V11_H_PCA_DIM):
        raise RuntimeError("v11-B differentiable PCA carrier drifted")
    return TransformedPatientFeatures(h=h.contiguous(), fine=fine.contiguous())


def clone_reasoner_pair(
    initial_state: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> tuple[SharedPositiveSetReasoner, SharedPositiveSetReasoner]:
    """Create storage-independent matched and PEFT heads from one fold state."""

    if not isinstance(initial_state, Mapping) or "prior_logits" not in initial_state:
        raise ValueError("initial reasoner state must include prior_logits")
    if "h_weight" not in initial_state or "fine_weight" not in initial_state:
        raise ValueError("v11-B requires both H and fine reasoner weights")
    if "candidate_mask" not in initial_state or not torch.equal(
        initial_state["candidate_mask"].cpu(), V11_CANDIDATE_MASK
    ):
        raise ValueError("initial reasoner state lost the fixed candidate mask")
    result = []
    for _ in range(2):
        model = SharedPositiveSetReasoner(
            initial_state["prior_logits"], use_h=True, use_fine=True
        )
        model.load_state_dict(
            {name: value.detach().cpu().clone() for name, value in initial_state.items()},
            strict=True,
        )
        result.append(model.to(device))
    first, second = result
    if any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(first.parameters(), second.parameters())
    ):
        raise RuntimeError("v11-B reasoner clones unexpectedly share parameter storage")
    return first, second


@dataclass(frozen=True)
class V11BPatientObjective:
    loss: torch.Tensor
    patient_logits: torch.Tensor
    event_h_upstream: torch.Tensor
    pooled_h: torch.Tensor


def patient_loss_and_h_upstream(
    event_h: torch.Tensor,
    event_patient_index: torch.Tensor,
    reliability: torch.Tensor,
    fine_patient: torch.Tensor,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> V11BPatientObjective:
    """Backpropagate the complete patient objective to an event-H leaf."""

    if not isinstance(reasoner, SharedPositiveSetReasoner):
        raise TypeError("reasoner must be SharedPositiveSetReasoner")
    if reasoner.h_weight is None or not bool(torch.count_nonzero(reasoner.h_weight.detach())):
        raise ValueError("v11-B requires a nonzero fold-local H-head warm start")
    patients = int(fine_patient.shape[0])
    leaf = event_h.detach().clone().requires_grad_(True)
    pooled = differentiable_pool_complete_patient_bags(
        leaf, event_patient_index, patients, reliability
    )
    transformed = apply_fold_transform_differentiable(pooled, fine_patient, transform)
    logits = reasoner(transformed).logits
    loss = positive_set_mass_loss(logits, targets, target_mask)
    if not torch.isfinite(loss):
        raise RuntimeError("v11-B patient objective became non-finite")
    loss.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all() or not bool(
        torch.count_nonzero(leaf.grad)
    ):
        raise RuntimeError("v11-B event-H gradient bridge is absent")
    return V11BPatientObjective(
        loss=loss.detach(),
        patient_logits=logits.detach(),
        event_h_upstream=leaf.grad.detach(),
        pooled_h=pooled.detach(),
    )


__all__ = [
    "V11BPatientObjective",
    "V11B_CALLS_PER_EVENT",
    "V11B_SECONDS_PER_CALL",
    "V11B_TOKEN_DIM",
    "apply_fold_transform_differentiable",
    "clone_reasoner_pair",
    "differentiable_pool_complete_patient_bags",
    "differentiable_suffix_phase_contrasts",
    "patient_loss_and_h_upstream",
]
