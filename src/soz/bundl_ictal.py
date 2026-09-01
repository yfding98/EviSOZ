"""Minimal BUNDL objective for source-native ictal edge-time labels.

This module implements Eqs. (2)--(4) from Shama and Venkataraman,
PLOS ONE 2026 (doi:10.1371/journal.pone.0352191).  It is deliberately
independent of every dataset and SOZ target loader: its only supervision is a
caller-supplied binary ``[E,20,T]`` edge-time target and observation mask.

The paper's asymmetric over-segmentation setting is frozen here: positive
label unreliability is estimated with ten Monte-Carlo dropout samples, while
negative-label unreliability is ``0.001``.  The MCD teacher and the resulting
clean-label posterior are stop-gradient quantities.  Consequently, gradients
flow only through the current ``[E,20,T,1]`` logits in Eq. (3).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.concept_heads import LongContextTemporalResidualIctalInvolvementHead


BUNDL_MC_SAMPLES = 10
BUNDL_PROBABILITY_FLOOR = 0.001
BUNDL_PROBABILITY_CEILING = 0.999
BUNDL_NEGATIVE_LABEL_UNRELIABILITY = 0.001
BUNDL_DROPOUT_PROBABILITY = 0.2
BUNDL_CLEAN_CORE_RADIUS_SECONDS = 2
N_ICTAL_EDGES = 20


@dataclass(frozen=True)
class BundlMCDStatistics:
    """Stop-gradient teacher mean and Eq. (4) normalized entropy."""

    mean_probability: torch.Tensor
    label_unreliability: torch.Tensor


@dataclass(frozen=True)
class BundlIctalLossOutput:
    """BUNDL loss and auditable stop-gradient intermediate tensors."""

    total: torch.Tensor
    clean_posterior: torch.Tensor
    noisy_label_probability: torch.Tensor
    teacher_probability: torch.Tensor
    positive_label_unreliability: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        """Alias for callers that use a generic ``output.loss`` contract."""

        return self.total


class BundlDropoutK31IctalHead(LongContextTemporalResidualIctalInvolvementHead):
    """Parameter-matched k31 ictal head with the paper's fixed 20% dropout.

    Dropout is applied to the normalized temporal representation immediately
    before the unchanged one-logit classifier.  It adds no trainable
    parameters, so BCE and BUNDL arms can use the exact same architecture.
    Symmetric k31 padding remains retrospective and must not be called causal.
    """

    dropout_probability = BUNDL_DROPOUT_PROBABILITY

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__(token_dim=token_dim, hidden_dim=hidden_dim)
        self.bundl_dropout = nn.Dropout(p=self.dropout_probability)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.adapter(self.edge_tokens(tokens))
        batch, edges, seconds, dimensions = hidden.shape
        temporal_input = hidden.reshape(
            batch * edges, seconds, dimensions
        ).transpose(1, 2)
        temporal_delta = self.temporal_depthwise(temporal_input).transpose(1, 2)
        temporal_hidden = self.temporal_norm(
            self.temporal_activation(
                hidden.reshape(batch * edges, seconds, dimensions)
                + temporal_delta
            )
        )
        dropped = self.bundl_dropout(temporal_hidden)
        return self.classifier(
            dropped.reshape(batch, edges, seconds, dimensions)
        )


def _validate_edge_time_targets(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    if not isinstance(targets, torch.Tensor) or not isinstance(
        target_mask, torch.Tensor
    ):
        raise TypeError("targets and target_mask must be torch tensors")
    if targets.ndim != 3 or targets.shape[1] != N_ICTAL_EDGES:
        raise ValueError("targets must have shape [E,20,T]")
    shape = tuple(int(value) for value in targets.shape)
    if shape[0] < 1 or shape[2] < 1:
        raise ValueError("targets require at least one event and one time bin")
    if tuple(target_mask.shape) != shape:
        raise ValueError("target_mask must have the same [E,20,T] shape")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"targets/mask must have shape {expected_shape}, got {shape}"
        )
    if not targets.is_floating_point():
        raise TypeError("targets must be floating-point")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be torch.bool")
    if targets.device != target_mask.device:
        raise ValueError("targets and target_mask must share a device")
    observed = targets[target_mask]
    if not torch.isfinite(observed).all():
        raise ValueError("observed targets must be finite")
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("observed targets must be binary")
    return shape


def _validate_current_logits(logits: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch tensor")
    if (
        logits.ndim != 4
        or logits.shape[1] != N_ICTAL_EDGES
        or logits.shape[-1] != 1
    ):
        raise ValueError("logits must have shape [E,20,T,1]")
    shape = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    if shape[0] < 1 or shape[2] < 1:
        raise ValueError("logits require at least one event and one time bin")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating-point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return shape


def _validate_patient_ids(
    patient_ids: torch.Tensor,
    *,
    n_events: int,
    device: torch.device,
) -> None:
    if not isinstance(patient_ids, torch.Tensor):
        raise TypeError("patient_ids must be a torch tensor")
    if patient_ids.ndim != 1 or patient_ids.shape[0] != n_events:
        raise ValueError("patient_ids must have shape [E]")
    if patient_ids.dtype != torch.long:
        raise TypeError("patient_ids must be torch.long")
    if patient_ids.device != device:
        raise ValueError("patient_ids must share the logits device")


def _probability_tensor(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
    field: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"{field} tensor must match [E,20,T]")
        if not value.is_floating_point():
            raise TypeError(f"{field} tensor must be floating-point")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"{field} tensor must share target device and dtype")
        tensor = value.detach()
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric or a tensor")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{field} must be finite")
        tensor = reference.new_full(reference.shape, scalar)
    if not torch.isfinite(tensor).all() or not torch.all(
        (tensor >= 0) & (tensor <= 1)
    ):
        raise ValueError(f"{field} must lie in [0,1]")
    return tensor


def bundl_clean_core_mask(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    radius_seconds: int = BUNDL_CLEAN_CORE_RADIUS_SECONDS,
) -> torch.Tensor:
    """Return centers whose complete local window is observed and constant.

    For the frozen ±2-second warm-up core, a center ``t`` is eligible exactly
    when all five bins ``[t-2,t+2]`` exist, are observed, and equal ``y[t]``.
    Thus the first/last two record bins are ineligible, as are centers whose
    window crosses either a label transition or an annotation gap.
    """

    _validate_edge_time_targets(targets, target_mask)
    if (
        isinstance(radius_seconds, bool)
        or not isinstance(radius_seconds, Integral)
        or int(radius_seconds) < 0
    ):
        raise ValueError("radius_seconds must be a non-negative integer")
    radius = int(radius_seconds)
    core = torch.zeros_like(target_mask)
    n_seconds = int(targets.shape[-1])
    width = 2 * radius + 1
    if n_seconds < width:
        return core
    center_start = radius
    center_stop = n_seconds - radius
    center_targets = targets[..., center_start:center_stop]
    eligible = torch.ones_like(center_targets, dtype=torch.bool)
    for offset in range(-radius, radius + 1):
        neighbor = slice(center_start + offset, center_stop + offset)
        eligible &= target_mask[..., neighbor]
        eligible &= targets[..., neighbor] == center_targets
    core[..., center_start:center_stop] = eligible
    return core


def bundl_mcd_statistics(mc_logits: torch.Tensor) -> BundlMCDStatistics:
    """Compute the detached MCD teacher and Eq. (4) uncertainty.

    ``mc_logits`` must contain exactly ten stochastic passes with shape
    ``[10,E,20,T,1]``.  Eq. (4) averages the entropy of each Bernoulli sample;
    it is intentionally *not* the entropy of the averaged probability.
    """

    if not isinstance(mc_logits, torch.Tensor):
        raise TypeError("mc_logits must be a torch tensor")
    if (
        mc_logits.ndim != 5
        or mc_logits.shape[0] != BUNDL_MC_SAMPLES
        or mc_logits.shape[2] != N_ICTAL_EDGES
        or mc_logits.shape[-1] != 1
    ):
        raise ValueError("mc_logits must have shape [10,E,20,T,1]")
    if mc_logits.shape[1] < 1 or mc_logits.shape[3] < 1:
        raise ValueError("mc_logits require at least one event and time bin")
    if not mc_logits.is_floating_point():
        raise TypeError("mc_logits must be floating-point")
    if not torch.isfinite(mc_logits).all():
        raise ValueError("mc_logits must be finite")
    probabilities = mc_logits.detach().sigmoid().clamp(
        BUNDL_PROBABILITY_FLOOR, BUNDL_PROBABILITY_CEILING
    )
    sample_entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * torch.log1p(-probabilities)
    )
    uncertainty = (
        sample_entropy.mean(dim=0).squeeze(-1) / math.log(2.0)
    ).clamp(0.0, 1.0)
    teacher = probabilities.mean(dim=0).squeeze(-1).clamp(
        BUNDL_PROBABILITY_FLOOR, BUNDL_PROBABILITY_CEILING
    )
    return BundlMCDStatistics(
        mean_probability=teacher.detach(),
        label_unreliability=uncertainty.detach(),
    )


def bundl_eq2_clean_posterior(
    targets: torch.Tensor,
    teacher_probability: torch.Tensor,
    *,
    positive_unreliability: float | torch.Tensor,
    negative_unreliability: float | torch.Tensor = (
        BUNDL_NEGATIVE_LABEL_UNRELIABILITY
    ),
) -> torch.Tensor:
    """Evaluate Eq. (2) and return a stop-gradient clean-label posterior.

    Binary labels are converted to the paper's numerical-tolerance values
    ``0.001`` and ``0.999``.  Passing zero unreliability therefore trusts that
    tolerant label, while passing one for both classes returns the detached
    teacher prior.
    """

    if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
        raise ValueError("targets must have shape [E,20,T]")
    dummy_mask = torch.ones_like(targets, dtype=torch.bool)
    shape = _validate_edge_time_targets(targets, dummy_mask)
    if not isinstance(teacher_probability, torch.Tensor):
        raise TypeError("teacher_probability must be a torch tensor")
    if tuple(teacher_probability.shape) != shape:
        raise ValueError("teacher_probability must have shape [E,20,T]")
    if not teacher_probability.is_floating_point():
        raise TypeError("teacher_probability must be floating-point")
    if (
        teacher_probability.device != targets.device
        or teacher_probability.dtype != targets.dtype
    ):
        raise ValueError("teacher_probability must share target device and dtype")
    if not torch.isfinite(teacher_probability).all() or not torch.all(
        (teacher_probability >= 0) & (teacher_probability <= 1)
    ):
        raise ValueError("teacher_probability must be finite and lie in [0,1]")

    z1 = _probability_tensor(
        positive_unreliability,
        reference=targets,
        field="positive_unreliability",
    )
    z0 = _probability_tensor(
        negative_unreliability,
        reference=targets,
        field="negative_unreliability",
    )
    noisy = targets.detach() * (
        BUNDL_PROBABILITY_CEILING - BUNDL_PROBABILITY_FLOOR
    ) + BUNDL_PROBABILITY_FLOOR
    teacher = teacher_probability.detach().clamp(
        BUNDL_PROBABILITY_FLOOR, BUNDL_PROBABILITY_CEILING
    )
    clean = noisy * (z1 * teacher + noisy * (1.0 - z1)) + (
        1.0 - noisy
    ) * (z0 * teacher + noisy * (1.0 - z0))
    return clean.clamp(
        BUNDL_PROBABILITY_FLOOR, BUNDL_PROBABILITY_CEILING
    ).detach()


def patient_macro_masked_mean(
    element_loss: torch.Tensor,
    mask: torch.Tensor,
    patient_ids: torch.Tensor,
) -> torch.Tensor:
    """Average observed cells within patient, then patients with equal weight."""

    if not isinstance(element_loss, torch.Tensor) or element_loss.ndim != 3:
        raise ValueError("element_loss must have shape [E,20,T]")
    if element_loss.shape[1] != N_ICTAL_EDGES:
        raise ValueError("element_loss must have shape [E,20,T]")
    if tuple(mask.shape) != tuple(element_loss.shape):
        raise ValueError("mask must match element_loss [E,20,T]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be torch.bool")
    if not element_loss.is_floating_point() or not torch.isfinite(element_loss).all():
        raise ValueError("element_loss must be finite floating-point")
    if element_loss.device != mask.device:
        raise ValueError("element_loss and mask must share a device")
    _validate_patient_ids(
        patient_ids,
        n_events=int(element_loss.shape[0]),
        device=element_loss.device,
    )
    patient_losses: list[torch.Tensor] = []
    for patient_id in torch.unique(patient_ids, sorted=True):
        patient_events = patient_ids == patient_id
        patient_mask = mask[patient_events]
        if patient_mask.any():
            patient_losses.append(
                element_loss[patient_events][patient_mask].mean()
            )
    if not patient_losses:
        raise ValueError("At least one observed edge-time label is required")
    return torch.stack(patient_losses).mean()


def bundl_ictal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: torch.Tensor,
    mc_logits: torch.Tensor,
) -> BundlIctalLossOutput:
    """Patient-macro masked BUNDL objective for ``[E,20,T,1]`` logits."""

    shape = _validate_current_logits(logits)
    _validate_edge_time_targets(
        targets, target_mask, expected_shape=shape
    )
    _validate_patient_ids(
        patient_ids, n_events=shape[0], device=logits.device
    )
    if not isinstance(mc_logits, torch.Tensor):
        raise TypeError("mc_logits must be a torch tensor")
    devices = {logits.device, targets.device, target_mask.device, mc_logits.device}
    if len(devices) != 1:
        raise ValueError("All BUNDL tensors must share one device")
    if targets.dtype != logits.dtype:
        raise ValueError("targets and logits must share a floating dtype")
    statistics = bundl_mcd_statistics(mc_logits)
    if tuple(statistics.mean_probability.shape) != shape:
        raise ValueError("mc_logits event/time shape must match current logits")
    if mc_logits.dtype != logits.dtype:
        raise ValueError("mc_logits and current logits must share a dtype")

    # Unknown cells may contain NaN or arbitrary sentinels.  They receive a
    # finite internal placeholder solely so vectorized Eq. (2)/BCE evaluation
    # stays well-defined; target_mask is the only authority for supervision.
    safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
    clean_posterior = bundl_eq2_clean_posterior(
        safe_targets,
        statistics.mean_probability,
        positive_unreliability=statistics.label_unreliability,
        negative_unreliability=BUNDL_NEGATIVE_LABEL_UNRELIABILITY,
    )
    element_loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), clean_posterior, reduction="none"
    )
    total = patient_macro_masked_mean(element_loss, target_mask, patient_ids)
    noisy_probability = safe_targets.detach() * (
        BUNDL_PROBABILITY_CEILING - BUNDL_PROBABILITY_FLOOR
    ) + BUNDL_PROBABILITY_FLOOR
    return BundlIctalLossOutput(
        total=total,
        clean_posterior=clean_posterior,
        noisy_label_probability=noisy_probability.detach(),
        teacher_probability=statistics.mean_probability,
        positive_label_unreliability=statistics.label_unreliability,
    )


@torch.no_grad()
def sample_mcd_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    seed: int,
    n_samples: int = BUNDL_MC_SAMPLES,
) -> torch.Tensor:
    """Deterministically draw ten dropout logits while keeping BN in eval mode.

    The helper forks the PyTorch RNG, enables only dropout modules, restores
    every module's prior training flag even on failure, and returns detached
    logits with shape ``[10,E,20,T,1]``.  A fixed seed therefore gives an
    exact replay without contaminating the caller's global RNG stream.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch tensor")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("inputs must be finite floating-point")
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, Integral)
        or int(n_samples) != BUNDL_MC_SAMPLES
    ):
        raise ValueError("BUNDL requires exactly 10 MCD samples")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or int(seed) < 0
        or int(seed) >= 2**63
    ):
        raise ValueError("seed must be an integer in [0,2**63)")

    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
    dropouts = [module for module in model.modules() if isinstance(module, dropout_types)]
    if not dropouts:
        raise ValueError("MCD model must contain at least one dropout module")
    prior_modes = [(module, bool(module.training)) for module in model.modules()]
    fork_devices: list[int] = []
    if inputs.device.type == "cuda":
        fork_devices = [
            torch.cuda.current_device()
            if inputs.device.index is None
            else int(inputs.device.index)
        ]
    outputs: list[torch.Tensor] = []
    try:
        model.eval()
        for dropout in dropouts:
            dropout.train(True)
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(int(seed))
            if inputs.device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            for _ in range(BUNDL_MC_SAMPLES):
                output = model(inputs)
                if not isinstance(output, torch.Tensor):
                    raise TypeError("MCD model must return one logits tensor")
                _validate_current_logits(output)
                outputs.append(output.detach())
    finally:
        for module, training in prior_modes:
            module.training = training
    stacked = torch.stack(outputs, dim=0)
    if not torch.isfinite(stacked).all():
        raise ValueError("MCD model produced non-finite logits")
    return stacked


# Explicit alias used by process-isolated runners to make replay intent clear.
deterministic_mcd_logits = sample_mcd_logits


__all__ = (
    "BUNDL_CLEAN_CORE_RADIUS_SECONDS",
    "BUNDL_DROPOUT_PROBABILITY",
    "BUNDL_MC_SAMPLES",
    "BUNDL_NEGATIVE_LABEL_UNRELIABILITY",
    "BUNDL_PROBABILITY_CEILING",
    "BUNDL_PROBABILITY_FLOOR",
    "BundlDropoutK31IctalHead",
    "BundlIctalLossOutput",
    "BundlMCDStatistics",
    "bundl_clean_core_mask",
    "bundl_eq2_clean_posterior",
    "bundl_ictal_loss",
    "bundl_mcd_statistics",
    "deterministic_mcd_logits",
    "patient_macro_masked_mean",
    "sample_mcd_logits",
)
