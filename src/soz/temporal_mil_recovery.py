"""Low-capacity temporal-MIL recovery head for LaBraM-derived I/V evidence.

The model deliberately consumes :class:`DevelopmentIVEvidenceBatch` rather
than raw EEG or foundation-model latents.  Its temporal weights are
discriminative pooling weights; they are not physiological onset or
propagation estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import aggregate_patient_logits
from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import (
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    N_TIME_TILES,
    unsigned_incidence_matrix,
)
from .losses import (
    masked_pairwise_ranking_loss,
    masked_patient_balanced_bce_with_logits,
)
from .metrics import DEEPSOZ_STANDARD19_NEIGHBORS


TEMPORAL_MIL_RECOVERY_SCHEMA = "soz_labram_evidence_temporal_mil_recovery_v1"


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
class TemporalMILPatientBatch:
    """Complete event bags and patient-level DeepSOZ targets."""

    evidence: DevelopmentIVEvidenceBatch
    event_patient_index: torch.Tensor
    patient_ids: tuple[str, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor

    def __post_init__(self) -> None:
        events = self.evidence.batch_size
        patients = len(self.patient_ids)
        if patients < 1 or len(set(self.patient_ids)) != patients:
            raise ValueError("patient_ids must be non-empty and unique")
        if (
            self.event_patient_index.dtype != torch.long
            or tuple(self.event_patient_index.shape) != (events,)
        ):
            raise TypeError("event_patient_index must be long [E]")
        if events < patients or self.event_patient_index.min().item() != 0 or (
            self.event_patient_index.max().item() != patients - 1
        ):
            raise ValueError("every patient must own at least one complete event bag")
        if torch.unique(self.event_patient_index).numel() != patients:
            raise ValueError("event_patient_index skips a patient")
        if tuple(self.targets.shape) != (patients, N_STANDARD_CHANNELS) or (
            tuple(self.target_mask.shape) != (patients, N_STANDARD_CHANNELS)
        ):
            raise ValueError("targets and target_mask must have shape [P,19]")
        if self.target_mask.dtype != torch.bool or not self.targets.is_floating_point():
            raise TypeError("targets must be floating point and target_mask bool")
        devices = {
            self.evidence.evolution.device,
            self.event_patient_index.device,
            self.targets.device,
            self.target_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("all patient-batch tensors must share a device")
        observed = self.targets[self.target_mask]
        if not torch.isfinite(observed).all() or (
            observed.numel() and not torch.all((observed == 0) | (observed == 1))
        ):
            raise ValueError("observed targets must be finite binary values")
        if not (((self.targets == 1) & self.target_mask).any(dim=1)).all():
            raise ValueError("every patient requires an observed in-head SOZ positive")

    def to(self, device: str | torch.device) -> "TemporalMILPatientBatch":
        return TemporalMILPatientBatch(
            evidence=self.evidence.to(device),
            event_patient_index=self.event_patient_index.to(device=device),
            patient_ids=self.patient_ids,
            targets=self.targets.to(device=device),
            target_mask=self.target_mask.to(device=device),
        )


def subset_patient_batch(
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    patient_ids: Sequence[str],
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_indices: Sequence[int],
) -> TemporalMILPatientBatch:
    """Select complete patient bags without changing event weights."""

    selected = tuple(int(value) for value in patient_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("patient_indices must be non-empty and unique")
    if any(value < 0 or value >= len(patient_ids) for value in selected):
        raise IndexError("patient index is outside the full batch")
    device = event_patient_index.device
    lookup = torch.full(
        (len(patient_ids),), -1, dtype=torch.long, device=device
    )
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
    lookup[selected_tensor] = torch.arange(len(selected), device=device)
    remapped = lookup[event_patient_index]
    event_indices = torch.nonzero(remapped >= 0, as_tuple=False).flatten()
    return TemporalMILPatientBatch(
        evidence=evidence.index_select(event_indices),
        event_patient_index=remapped[event_indices],
        patient_ids=tuple(str(patient_ids[index]) for index in selected),
        targets=targets.index_select(0, selected_tensor),
        target_mask=target_mask.index_select(0, selected_tensor),
    )


def jeffreys_channel_prior_logits(batch: TemporalMILPatientBatch) -> torch.Tensor:
    """Compute a fold-local Jeffreys prevalence prior without held-out labels."""

    positive = ((batch.targets == 1) & batch.target_mask).sum(dim=0).float()
    observed = batch.target_mask.sum(dim=0).float()
    prevalence = (positive + 0.5) / (observed + 1.0)
    return torch.logit(prevalence.clamp(1e-4, 1 - 1e-4))


@dataclass(frozen=True)
class TemporalMILEvidenceOutput:
    event_logits: torch.Tensor
    channel_prior: torch.Tensor
    ictal_contribution: torch.Tensor
    evolution_contribution: torch.Tensor
    temporal_weights: torch.Tensor
    ictal_node_support: torch.Tensor
    ictal_node_mask: torch.Tensor
    evolution_tile_score: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return (
            self.channel_prior
            + self.ictal_contribution
            + self.evolution_contribution
        )


class TemporalMILEvidenceReasoner(nn.Module):
    """Evidence-only temporal pooling with a fixed channel prevalence prior."""

    def __init__(self, prior_logits: torch.Tensor, *, hidden_dim: int = 8) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        if not prior_logits.is_floating_point() or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite floating-point values")
        if hidden_dim != 8:
            raise ValueError("temporal-MIL hidden_dim is frozen at 8")
        self.evolution_scorer = nn.Sequential(
            nn.Linear(2 * N_NODE_FEATURES, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.ictal_feature_logits = nn.Parameter(torch.zeros(2))
        self.raw_attention_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.temporal_bias = nn.Parameter(torch.zeros(N_TIME_TILES))
        self.raw_ictal_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.raw_evolution_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.register_buffer(
            "channel_prior_logits", prior_logits.detach().float().contiguous()
        )
        self.register_buffer("incidence", unsigned_incidence_matrix(), persistent=True)
        if self.n_trainable_parameters >= 500:
            raise ValueError("temporal-MIL head exceeds its capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _route_ictal(
        self,
        evidence: DevelopmentIVEvidenceBatch,
        ictal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.ictal_feature_logits.softmax(dim=0).to(evidence.ictal.dtype)
        edge_support = (evidence.ictal * weights).sum(dim=-1)
        edge_support = torch.where(ictal_mask, edge_support, 0.0)
        incidence = self.incidence.to(
            device=edge_support.device, dtype=edge_support.dtype
        )
        valid = ictal_mask.to(edge_support.dtype)
        degree = torch.einsum("ce,bet->bct", incidence, valid)
        support = torch.einsum("ce,bet->bct", incidence, edge_support)
        support = support / degree.clamp_min(1.0)
        node_mask = degree > 0
        return torch.where(node_mask, support, 0.0), node_mask

    def forward(
        self, evidence: DevelopmentIVEvidenceBatch
    ) -> TemporalMILEvidenceOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("temporal-MIL reasoner accepts evidence batches only")
        evidence.validate()
        phase = evidence.phase_mask.unsqueeze(1)
        evolution_mask = evidence.evolution_mask & phase
        ictal_mask = evidence.ictal_mask & phase

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
        evolution_tile = self.evolution_scorer(evolution_features).squeeze(-1)
        evolution_tile = torch.where(evolution_mask, evolution_tile, 0.0)

        ictal_node, ictal_node_mask = self._route_ictal(evidence, ictal_mask)
        attention_scale = F.softplus(self.raw_attention_scale).to(ictal_node.dtype)
        centered_bias = self.temporal_bias - self.temporal_bias.mean()
        energy = attention_scale * ictal_node + centered_bias.view(1, 1, -1)
        temporal_weights = _masked_softmax(energy, evolution_mask)

        reliability = torch.where(
            evolution_mask, evidence.reliability, torch.zeros_like(evidence.reliability)
        )
        gated_evolution = _positive_only_reliability_gate(
            evolution_tile, reliability
        )
        gated_ictal = ictal_node * reliability * ictal_node_mask.to(ictal_node.dtype)
        evolution_gain = F.softplus(self.raw_evolution_gain).to(evolution_tile.dtype)
        ictal_gain = F.softplus(self.raw_ictal_gain).to(ictal_node.dtype)
        evolution_contribution = evolution_gain * (
            temporal_weights * gated_evolution
        ).sum(dim=-1)
        ictal_contribution = ictal_gain * (
            temporal_weights * gated_ictal
        ).sum(dim=-1)
        prior = self.channel_prior_logits.to(evolution_tile.dtype).unsqueeze(0)
        prior = prior.expand(evidence.batch_size, -1)
        logits = prior + evolution_contribution + ictal_contribution
        output = TemporalMILEvidenceOutput(
            event_logits=logits,
            channel_prior=prior,
            ictal_contribution=ictal_contribution,
            evolution_contribution=evolution_contribution,
            temporal_weights=temporal_weights,
            ictal_node_support=ictal_node,
            ictal_node_mask=ictal_node_mask,
            evolution_tile_score=evolution_tile,
        )
        if not torch.allclose(
            output.reconstructed_logits(), logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("temporal-MIL contribution decomposition drifted")
        return output


def exact_positive_set_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Place categorical ranking mass on the exact observed positive set."""

    if logits.ndim != 2 or tuple(logits.shape) != tuple(targets.shape) or (
        tuple(logits.shape) != tuple(target_mask.shape)
    ):
        raise ValueError("set-mass inputs must share [P,19]")
    rows = []
    for patient in range(logits.shape[0]):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        if not observed.any() or not positive.any():
            raise ValueError("set-mass loss requires observed exact positives")
        rows.append(
            torch.logsumexp(logits[patient][observed], dim=0)
            - torch.logsumexp(logits[patient][positive], dim=0)
        )
    return torch.stack(rows).mean()


def neighbor_sensitivity_set_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    max_positive_for_neighbor: int = 4,
) -> torch.Tensor:
    """Low-weight DeepSOZ-neighbour sensitivity auxiliary.

    This does not alter exact targets and must never replace the strict loss.
    """

    rows = []
    for patient in range(logits.shape[0]):
        observed = target_mask[patient]
        positive = observed & (targets[patient] == 1)
        accepted = positive.clone()
        if int(positive.sum().item()) <= max_positive_for_neighbor:
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
        accepted &= observed
        rows.append(
            torch.logsumexp(logits[patient][observed], dim=0)
            - torch.logsumexp(logits[patient][accepted], dim=0)
        )
    return torch.stack(rows).mean()


def event_consistency_loss(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean within-patient Jensen-Shannon dispersion across seizure events."""

    rows = []
    for patient in range(target_mask.shape[0]):
        selected = event_logits[event_patient_index == patient]
        if selected.shape[0] < 2:
            continue
        observed = target_mask[patient]
        log_probability = F.log_softmax(selected[:, observed], dim=-1)
        probability = log_probability.exp()
        mixture = probability.mean(dim=0).clamp_min(1e-8)
        rows.append(
            (probability * (log_probability - mixture.log())).sum(dim=-1).mean()
        )
    if not rows:
        return event_logits.sum() * 0.0
    return torch.stack(rows).mean()


@dataclass(frozen=True)
class TemporalMILObjectiveOutput:
    total: torch.Tensor
    exact_set_mass: torch.Tensor
    pairwise: torch.Tensor
    bce: torch.Tensor
    consistency: torch.Tensor
    neighbor_auxiliary: torch.Tensor


def temporal_mil_objective(
    event_logits: torch.Tensor,
    batch: TemporalMILPatientBatch,
    *,
    neighbor_weight: float,
) -> TemporalMILObjectiveOutput:
    if neighbor_weight not in {0.0, 0.05}:
        raise ValueError("neighbor_weight must be one of the frozen candidates")
    aggregation = aggregate_patient_logits(
        event_logits, batch.event_patient_index
    )
    patient_logits = aggregation.logits
    exact = exact_positive_set_mass_loss(
        patient_logits, batch.targets, batch.target_mask
    )
    pairwise = masked_pairwise_ranking_loss(
        patient_logits, batch.targets, batch.target_mask
    )
    bce = masked_patient_balanced_bce_with_logits(
        patient_logits, batch.targets, batch.target_mask
    )
    consistency = event_consistency_loss(
        event_logits, batch.event_patient_index, batch.target_mask
    )
    neighbor = neighbor_sensitivity_set_mass_loss(
        patient_logits, batch.targets, batch.target_mask
    )
    total = (
        exact
        + 0.50 * pairwise
        + 0.25 * bce
        + 0.05 * consistency
        + float(neighbor_weight) * neighbor
    )
    return TemporalMILObjectiveOutput(
        total=total,
        exact_set_mass=exact,
        pairwise=pairwise,
        bce=bce,
        consistency=consistency,
        neighbor_auxiliary=neighbor,
    )
