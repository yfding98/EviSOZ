"""Global-I temporal gating with V-only channel localization.

This development-only recovery head removes the spatial edge-to-node route
used by the preceding temporal-MIL candidate.  Bipolar ictal evidence is
collapsed across *all* valid edges before it enters the model, so it can only
change a time weight shared by every physical channel.  The frozen V
descriptors are the sole event-dependent source of channel differences.

The attention weights are discriminative pooling weights.  They are not
physiological seizure-onset or propagation estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import N_NODE_FEATURES, N_STANDARD_CHANNELS, N_TIME_TILES
from .losses import masked_pairwise_ranking_loss
from .temporal_mil_recovery import TemporalMILPatientBatch


GLOBAL_I_V_RECOVERY_SCHEMA = "soz_labram_global_i_gate_v_localizer_recovery_v2"
PatientPoolingMode = Literal["equal_probability_mean", "aq_probability_mean"]


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse-softplus input must be positive")
    return math.log(math.expm1(value))


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("masked softmax requires aligned values and bool mask")
    if values.ndim < 1 or not torch.isfinite(values).all():
        raise ValueError("masked softmax values must be finite")
    available = mask.any(dim=-1, keepdim=True)
    masked = values.masked_fill(~mask, -torch.inf)
    safe = torch.where(available, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=-1)
    return torch.where(available & mask, weights, torch.zeros_like(weights))


def _positive_only_reliability_gate(
    contribution: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    if contribution.shape != reliability.shape:
        raise ValueError("contribution and reliability shapes differ")
    if not torch.isfinite(contribution).all() or not torch.isfinite(reliability).all():
        raise ValueError("quality gate requires finite tensors")
    if torch.any((reliability < 0) | (reliability > 1)):
        raise ValueError("reliability must lie in [0,1]")
    return contribution.clamp_max(0) + reliability * contribution.clamp_min(0)


@dataclass(frozen=True)
class GlobalIVEventOutput:
    """Auditable event output with no channel-wise I contribution."""

    event_logits: torch.Tensor
    event_probabilities: torch.Tensor
    channel_prior: torch.Tensor
    evolution_contribution: torch.Tensor
    temporal_weights: torch.Tensor
    global_ictal_support: torch.Tensor
    global_ictal_mask: torch.Tensor
    evolution_tile_score: torch.Tensor
    channel_available: torch.Tensor
    prior_only_event: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return self.channel_prior + self.evolution_contribution


class GlobalITemporalGateVLocalizer(nn.Module):
    """Low-capacity evidence head whose I path is spatially invariant."""

    def __init__(self, prior_logits: torch.Tensor, *, hidden_dim: int = 8) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        if not prior_logits.is_floating_point() or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite floating-point values")
        if hidden_dim != 8:
            raise ValueError("global-I/V hidden_dim is frozen at 8")

        self.evolution_scorer = nn.Sequential(
            nn.Linear(2 * N_NODE_FEATURES, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.ictal_feature_logits = nn.Parameter(torch.zeros(2))
        self.raw_temporal_gate_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.temporal_bias = nn.Parameter(torch.zeros(N_TIME_TILES))
        self.raw_evolution_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.register_buffer(
            "channel_prior_logits", prior_logits.detach().float().contiguous()
        )
        if self.n_trainable_parameters >= 500:
            raise ValueError("global-I/V head exceeds its capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, evidence: DevelopmentIVEvidenceBatch) -> GlobalIVEventOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("global-I/V reasoner accepts evidence batches only")
        evidence.validate()
        phase = evidence.phase_mask
        evolution_mask = evidence.evolution_mask & phase.unsqueeze(1)
        ictal_mask = evidence.ictal_mask & phase.unsqueeze(1)

        # Crucial identifiability constraint: the complete edge dimension is
        # reduced here.  No incidence matrix or endpoint identity is present.
        feature_weights = self.ictal_feature_logits.softmax(dim=0).to(
            evidence.ictal.dtype
        )
        edge_support = (evidence.ictal * feature_weights).sum(dim=-1)
        edge_support = torch.where(ictal_mask, edge_support, 0.0)
        edge_count = ictal_mask.sum(dim=1)
        global_ictal_mask = edge_count > 0
        global_ictal_support = edge_support.sum(dim=1) / edge_count.clamp_min(1)
        global_ictal_support = torch.where(
            global_ictal_mask, global_ictal_support, 0.0
        )

        current = torch.where(
            evolution_mask.unsqueeze(-1), evidence.evolution, 0.0
        )
        previous = torch.roll(current, shifts=1, dims=2)
        previous_mask = torch.roll(evolution_mask, shifts=1, dims=2)
        previous_mask[:, :, 0] = False
        delta_mask = evolution_mask & previous_mask
        delta = torch.where(
            delta_mask.unsqueeze(-1), current - previous, torch.zeros_like(current)
        )
        evolution_features = torch.cat((current, delta), dim=-1)
        evolution_tile_score = self.evolution_scorer(evolution_features).squeeze(-1)
        evolution_tile_score = torch.where(
            evolution_mask, evolution_tile_score, 0.0
        )

        centered_bias = self.temporal_bias - self.temporal_bias.mean()
        gate_scale = F.softplus(self.raw_temporal_gate_scale).to(
            global_ictal_support.dtype
        )
        global_energy = (
            gate_scale * global_ictal_support
            + centered_bias.to(global_ictal_support.dtype).unsqueeze(0)
        )
        # The energy is exactly shared across all 19 channels.  Per-channel
        # weights can differ only because a V tile is unavailable.
        temporal_weights = _masked_softmax(
            global_energy.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1),
            evolution_mask,
        )

        reliability = torch.where(
            evolution_mask, evidence.reliability, torch.zeros_like(evidence.reliability)
        )
        gated_evolution = _positive_only_reliability_gate(
            evolution_tile_score, reliability
        )
        gain = F.softplus(self.raw_evolution_gain).to(evolution_tile_score.dtype)
        evolution_contribution = gain * (
            temporal_weights * gated_evolution
        ).sum(dim=-1)

        prior = self.channel_prior_logits.to(evolution_contribution.dtype).unsqueeze(0)
        prior = prior.expand(evidence.batch_size, -1)
        event_logits = prior + evolution_contribution
        channel_available = evolution_mask.any(dim=-1)
        prior_only_event = ~channel_available.any(dim=1)
        # A completely V-missing event is retained as an auditable no-signal
        # observation.  It contributes only the fold-local prior; its fixed AQ
        # weight is zero, while equal pooling keeps it without inventing data.
        softmax_mask = channel_available | prior_only_event.unsqueeze(1)
        event_probabilities = _masked_softmax(event_logits, softmax_mask)

        output = GlobalIVEventOutput(
            event_logits=event_logits,
            event_probabilities=event_probabilities,
            channel_prior=prior,
            evolution_contribution=evolution_contribution,
            temporal_weights=temporal_weights,
            global_ictal_support=global_ictal_support,
            global_ictal_mask=global_ictal_mask,
            evolution_tile_score=evolution_tile_score,
            channel_available=channel_available,
            prior_only_event=prior_only_event,
        )
        if not torch.allclose(
            output.reconstructed_logits(), event_logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("global-I/V contribution decomposition drifted")
        if not torch.allclose(
            event_probabilities.sum(dim=-1),
            torch.ones(evidence.batch_size, device=event_probabilities.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("event channel probabilities are not normalized")
        return output


def target_free_event_aq_weight(evidence: DevelopmentIVEvidenceBatch) -> torch.Tensor:
    """Fixed event reliability/coverage score with no label access."""

    evidence.validate()
    valid = evidence.evolution_mask & evidence.phase_mask.unsqueeze(1)
    denominator = (
        evidence.phase_mask.sum(dim=1).to(evidence.reliability.dtype)
        * float(N_STANDARD_CHANNELS)
    )
    numerator = torch.where(
        valid, evidence.reliability, torch.zeros_like(evidence.reliability)
    ).sum(dim=(1, 2))
    weights = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )
    if not torch.isfinite(weights).all() or torch.any((weights < 0) | (weights > 1)):
        raise RuntimeError("target-free AQ event weights left [0,1]")
    return weights


@dataclass(frozen=True)
class PatientProbabilityAggregation:
    probabilities: torch.Tensor
    ranking_logits: torch.Tensor
    event_normalized_weights: torch.Tensor
    event_counts: torch.Tensor


def aggregate_patient_event_probabilities(
    event_probabilities: torch.Tensor,
    event_patient_index: torch.Tensor,
    *,
    mode: PatientPoolingMode,
    aq_event_weight: torch.Tensor | None = None,
) -> PatientProbabilityAggregation:
    """Average normalized event maps, optionally using fixed target-free AQ."""

    if event_probabilities.ndim != 2 or event_probabilities.shape[1] != 19:
        raise ValueError("event_probabilities must have shape [E,19]")
    events = int(event_probabilities.shape[0])
    if events < 1 or tuple(event_patient_index.shape) != (events,):
        raise ValueError("event_patient_index must have shape [E]")
    if event_patient_index.dtype != torch.long:
        raise TypeError("event_patient_index must be torch.long")
    if event_patient_index.device != event_probabilities.device:
        raise ValueError("event probabilities and patient indices must share a device")
    if not torch.isfinite(event_probabilities).all() or torch.any(
        event_probabilities < 0
    ):
        raise ValueError("event probabilities must be finite and non-negative")
    if not torch.allclose(
        event_probabilities.sum(dim=1),
        torch.ones(events, device=event_probabilities.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each event probability row must sum to one")
    if event_patient_index.min().item() != 0:
        raise ValueError("event_patient_index must start at zero")
    patient_count = int(event_patient_index.max().item()) + 1
    if torch.unique(event_patient_index).numel() != patient_count:
        raise ValueError("event_patient_index must be contiguous")
    if mode not in {"equal_probability_mean", "aq_probability_mean"}:
        raise ValueError("unknown patient pooling mode")
    if mode == "equal_probability_mean":
        raw_weight = torch.ones(
            events,
            dtype=event_probabilities.dtype,
            device=event_probabilities.device,
        )
    else:
        if aq_event_weight is None or tuple(aq_event_weight.shape) != (events,):
            raise ValueError("AQ pooling requires aq_event_weight [E]")
        if aq_event_weight.device != event_probabilities.device:
            raise ValueError("AQ weights and event probabilities must share a device")
        if not torch.isfinite(aq_event_weight).all() or torch.any(aq_event_weight < 0):
            raise ValueError("AQ weights must be finite and non-negative")
        raw_weight = aq_event_weight.to(event_probabilities.dtype)

    normalized = torch.zeros_like(raw_weight)
    patient_probabilities = event_probabilities.new_zeros((patient_count, 19))
    event_counts = torch.bincount(event_patient_index, minlength=patient_count)
    for patient in range(patient_count):
        selected = event_patient_index == patient
        weights = raw_weight[selected]
        if float(weights.sum().detach().cpu()) <= 0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        normalized[selected] = weights
        patient_probabilities[patient] = (
            event_probabilities[selected] * weights.unsqueeze(1)
        ).sum(dim=0)

    if not torch.allclose(
        patient_probabilities.sum(dim=1),
        torch.ones(patient_count, device=patient_probabilities.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("patient probability maps are not normalized")
    eps = torch.finfo(patient_probabilities.dtype).eps
    ranking_logits = torch.logit(patient_probabilities.clamp(eps, 1.0 - eps))
    return PatientProbabilityAggregation(
        probabilities=patient_probabilities,
        ranking_logits=ranking_logits,
        event_normalized_weights=normalized,
        event_counts=event_counts,
    )


def exact_positive_probability_mass_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Maximize probability mass assigned to the exact observed SOZ set."""

    if probabilities.ndim != 2 or probabilities.shape != targets.shape or (
        probabilities.shape != target_mask.shape
    ):
        raise ValueError("set-mass inputs must share [P,19]")
    if target_mask.dtype != torch.bool or not targets.is_floating_point():
        raise TypeError("targets must be floating point and target_mask bool")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("patient probabilities must be finite and non-negative")
    rows = []
    eps = torch.finfo(probabilities.dtype).tiny
    for patient in range(probabilities.shape[0]):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        if not observed.any() or not positive.any():
            raise ValueError("set-mass loss requires an observed exact positive")
        denominator = probabilities[patient][observed].sum()
        numerator = probabilities[patient][positive].sum()
        rows.append(-torch.log((numerator / denominator.clamp_min(eps)).clamp_min(eps)))
    return torch.stack(rows).mean()


@dataclass(frozen=True)
class GlobalIVObjectiveOutput:
    total: torch.Tensor
    exact_set_mass: torch.Tensor
    pairwise: torch.Tensor
    aggregation: PatientProbabilityAggregation


def global_i_v_objective(
    event_probabilities: torch.Tensor,
    batch: TemporalMILPatientBatch,
    *,
    mode: PatientPoolingMode,
) -> GlobalIVObjectiveOutput:
    aq = None
    if mode == "aq_probability_mean":
        aq = target_free_event_aq_weight(batch.evidence)
    aggregation = aggregate_patient_event_probabilities(
        event_probabilities,
        batch.event_patient_index,
        mode=mode,
        aq_event_weight=aq,
    )
    exact = exact_positive_probability_mass_loss(
        aggregation.probabilities, batch.targets, batch.target_mask
    )
    pairwise = masked_pairwise_ranking_loss(
        aggregation.ranking_logits, batch.targets, batch.target_mask
    )
    return GlobalIVObjectiveOutput(
        total=exact + 0.25 * pairwise,
        exact_set_mass=exact,
        pairwise=pairwise,
        aggregation=aggregation,
    )


__all__ = [
    "GLOBAL_I_V_RECOVERY_SCHEMA",
    "GlobalITemporalGateVLocalizer",
    "GlobalIVEventOutput",
    "GlobalIVObjectiveOutput",
    "PatientProbabilityAggregation",
    "PatientPoolingMode",
    "aggregate_patient_event_probabilities",
    "exact_positive_probability_mass_loss",
    "global_i_v_objective",
    "target_free_event_aq_weight",
]
