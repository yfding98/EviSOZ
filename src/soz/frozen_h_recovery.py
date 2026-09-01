"""Frozen node-indexed LaBraM-token recovery heads for SOZ development.

The token tensor is a matched direct foundation latent, not a named concept.
It is accepted only by this development module and cannot satisfy the formal
evidence-bottleneck APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .development_reasoner import DevelopmentIVEvidenceBatch
from .geometry import N_NODE_FEATURES, N_STANDARD_CHANNELS, N_TIME_TILES
from .global_i_v_recovery import (
    GlobalIVObjectiveOutput,
    _masked_softmax,
    _positive_only_reliability_gate,
    global_i_v_objective,
)
from .temporal_mil_recovery import (
    TemporalMILPatientBatch,
    subset_patient_batch,
)


FROZEN_H_RECOVERY_SCHEMA = "soz_labram_frozen_node_token_recovery_v3"
FROZEN_H_TOKEN_DIM = 200
FROZEN_H_SECONDS_PER_TILE = 4
FrozenHCandidate = Literal[
    "frozen_h_uniform",
    "frozen_h_v_uniform",
    "frozen_h_v_global_i",
]
FROZEN_H_CANDIDATES: tuple[FrozenHCandidate, ...] = (
    "frozen_h_uniform",
    "frozen_h_v_uniform",
    "frozen_h_v_global_i",
)


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse-softplus input must be positive")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class FrozenHStandardization:
    mean: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        expected = (FROZEN_H_TOKEN_DIM,)
        if tuple(self.mean.shape) != expected or tuple(self.scale.shape) != expected:
            raise ValueError("H standardization tensors must have shape [200]")
        if not self.mean.is_floating_point() or not self.scale.is_floating_point():
            raise TypeError("H standardization tensors must be floating point")
        if self.mean.device != self.scale.device:
            raise ValueError("H standardization tensors must share a device")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("H standardization tensors must be finite")
        if torch.any(self.scale < 1e-5):
            raise ValueError("H standardization scale must be at least 1e-5")


@dataclass(frozen=True)
class FrozenHPatientBatch:
    base: TemporalMILPatientBatch
    node_tokens: torch.Tensor

    def __post_init__(self) -> None:
        expected = (
            self.base.evidence.batch_size,
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
            FROZEN_H_SECONDS_PER_TILE,
            FROZEN_H_TOKEN_DIM,
        )
        if tuple(self.node_tokens.shape) != expected:
            raise ValueError(
                "frozen LaBraM node tokens must have shape [E,19,15,4,200]"
            )
        if not self.node_tokens.is_floating_point():
            raise TypeError("frozen LaBraM node tokens must be floating point")
        if self.node_tokens.requires_grad:
            raise ValueError("frozen LaBraM node tokens must be detached")
        if self.node_tokens.device != self.base.evidence.evolution.device:
            raise ValueError("H tokens and base evidence must share a device")
        if not torch.isfinite(self.node_tokens).all():
            raise ValueError("frozen LaBraM node tokens must be finite")

    def to(self, device: str | torch.device) -> "FrozenHPatientBatch":
        return FrozenHPatientBatch(
            base=self.base.to(device),
            node_tokens=self.node_tokens.to(device=device),
        )


def subset_frozen_h_patient_batch(
    full: FrozenHPatientBatch,
    patient_indices: Sequence[int],
) -> FrozenHPatientBatch:
    base = subset_patient_batch(
        full.base.evidence,
        full.base.event_patient_index,
        full.base.patient_ids,
        full.base.targets,
        full.base.target_mask,
        patient_indices,
    )
    selected_patient = torch.full(
        (len(full.base.patient_ids),),
        False,
        dtype=torch.bool,
        device=full.base.event_patient_index.device,
    )
    selected_patient[
        torch.tensor(
            tuple(int(value) for value in patient_indices),
            dtype=torch.long,
            device=selected_patient.device,
        )
    ] = True
    event_indices = torch.nonzero(
        selected_patient[full.base.event_patient_index], as_tuple=False
    ).flatten()
    return FrozenHPatientBatch(
        base=base,
        node_tokens=full.node_tokens.index_select(0, event_indices),
    )


def fit_frozen_h_standardization(batch: FrozenHPatientBatch) -> FrozenHStandardization:
    """Fit target-free 200-D moments on this training patient subset only."""

    tile_tokens = batch.node_tokens.mean(dim=3)
    valid = batch.base.evidence.phase_mask.unsqueeze(1).expand(
        -1, N_STANDARD_CHANNELS, -1
    )
    selected = tile_tokens[valid]
    if selected.ndim != 2 or selected.shape[0] < 2:
        raise ValueError("H standardization requires at least two valid token rows")
    mean = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False).clamp_min(1e-5)
    return FrozenHStandardization(mean=mean.detach(), scale=scale.detach())


@dataclass(frozen=True)
class FrozenHEventOutput:
    event_logits: torch.Tensor
    event_probabilities: torch.Tensor
    channel_prior: torch.Tensor
    h_contribution: torch.Tensor
    v_contribution: torch.Tensor
    h_temporal_weights: torch.Tensor
    v_temporal_weights: torch.Tensor
    global_ictal_support: torch.Tensor
    global_ictal_mask: torch.Tensor
    h_tile_score: torch.Tensor
    v_tile_score: torch.Tensor
    prior_only_event: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return self.channel_prior + self.h_contribution + self.v_contribution


class FrozenHNodeLocalizer(nn.Module):
    """Shared low-capacity probe over frozen physical-node LaBraM tokens."""

    def __init__(
        self,
        prior_logits: torch.Tensor,
        standardization: FrozenHStandardization,
        *,
        candidate: FrozenHCandidate,
    ) -> None:
        super().__init__()
        if candidate not in FROZEN_H_CANDIDATES:
            raise ValueError("unknown frozen-H candidate")
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        if not prior_logits.is_floating_point() or not torch.isfinite(prior_logits).all():
            raise ValueError("prior_logits must be finite floating-point values")
        if not isinstance(standardization, FrozenHStandardization):
            raise TypeError("standardization must be FrozenHStandardization")
        self.candidate = candidate
        self.use_v = candidate != "frozen_h_uniform"
        self.use_global_i = candidate == "frozen_h_v_global_i"

        # No bias: otherwise a constant H path followed by channel-varying Q
        # could masquerade as foundation-token localization.
        self.h_scorer = nn.Linear(FROZEN_H_TOKEN_DIM, 1, bias=False)
        self.raw_h_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        if self.use_v:
            self.v_scorer: nn.Module | None = nn.Sequential(
                nn.Linear(2 * N_NODE_FEATURES, 8),
                nn.Tanh(),
                nn.Linear(8, 1, bias=False),
            )
            self.raw_v_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
            )
        else:
            self.v_scorer = None
            self.register_parameter("raw_v_gain", None)

        if self.use_global_i:
            self.ictal_feature_logits: nn.Parameter | None = nn.Parameter(
                torch.zeros(2)
            )
            self.raw_temporal_gate_scale: nn.Parameter | None = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
            )
        else:
            self.register_parameter("ictal_feature_logits", None)
            self.register_parameter("raw_temporal_gate_scale", None)

        self.register_buffer(
            "channel_prior_logits", prior_logits.detach().float().contiguous()
        )
        self.register_buffer(
            "h_mean", standardization.mean.detach().float().contiguous()
        )
        self.register_buffer(
            "h_scale", standardization.scale.detach().float().contiguous()
        )
        if self.n_trainable_parameters >= 500:
            raise ValueError("frozen-H head exceeds its capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _global_energy(
        self,
        evidence: DevelopmentIVEvidenceBatch,
        ictal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_global_i:
            zeros = evidence.evolution.new_zeros((evidence.batch_size, N_TIME_TILES))
            return zeros, zeros, torch.zeros_like(zeros, dtype=torch.bool)
        assert self.ictal_feature_logits is not None
        assert self.raw_temporal_gate_scale is not None
        feature_weights = self.ictal_feature_logits.softmax(dim=0).to(
            evidence.ictal.dtype
        )
        edge_support = (evidence.ictal * feature_weights).sum(dim=-1)
        edge_support = torch.where(ictal_mask, edge_support, 0.0)
        count = ictal_mask.sum(dim=1)
        mask = count > 0
        support = edge_support.sum(dim=1) / count.clamp_min(1)
        support = torch.where(mask, support, 0.0)
        # No free tile bias: otherwise the global-I candidate would be
        # confounded with a learned absolute-time prior absent from the
        # uniform comparator.
        energy = F.softplus(self.raw_temporal_gate_scale).to(support.dtype) * support
        return energy, support, mask

    def forward(
        self,
        node_tokens: torch.Tensor,
        evidence: DevelopmentIVEvidenceBatch,
    ) -> FrozenHEventOutput:
        evidence.validate()
        expected = (
            evidence.batch_size,
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
            FROZEN_H_SECONDS_PER_TILE,
            FROZEN_H_TOKEN_DIM,
        )
        if tuple(node_tokens.shape) != expected:
            raise ValueError("node_tokens must have shape [E,19,15,4,200]")
        if not node_tokens.is_floating_point() or not torch.isfinite(node_tokens).all():
            raise ValueError("node_tokens must be finite floating point")
        if node_tokens.requires_grad:
            raise ValueError("frozen-H model accepts detached tokens only")
        if node_tokens.device != evidence.evolution.device:
            raise ValueError("node tokens and evidence must share a device")

        phase = evidence.phase_mask
        h_mask = phase.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1)
        v_mask = evidence.evolution_mask & phase.unsqueeze(1)
        ictal_mask = evidence.ictal_mask & phase.unsqueeze(1)
        global_energy, global_support, global_mask = self._global_energy(
            evidence, ictal_mask
        )

        h_tile = node_tokens.mean(dim=3)
        standardized = (
            h_tile - self.h_mean.to(h_tile.dtype)
        ) / self.h_scale.to(h_tile.dtype)
        h_tile_score = self.h_scorer(standardized).squeeze(-1)
        h_tile_score = torch.where(h_mask, h_tile_score, 0.0)

        if self.use_global_i:
            shared_energy = global_energy
        else:
            shared_energy = torch.zeros_like(global_energy)
        h_temporal_weights = _masked_softmax(
            shared_energy.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1),
            h_mask,
        )
        h_reliability = torch.where(
            h_mask, evidence.reliability, torch.zeros_like(evidence.reliability)
        )
        gated_h = _positive_only_reliability_gate(h_tile_score, h_reliability)
        h_contribution = F.softplus(self.raw_h_gain).to(h_tile_score.dtype) * (
            h_temporal_weights * gated_h
        ).sum(dim=-1)

        if self.use_v:
            assert self.v_scorer is not None and self.raw_v_gain is not None
            current = torch.where(v_mask.unsqueeze(-1), evidence.evolution, 0.0)
            previous = torch.roll(current, shifts=1, dims=2)
            previous_mask = torch.roll(v_mask, shifts=1, dims=2)
            previous_mask[:, :, 0] = False
            delta = torch.where(
                (v_mask & previous_mask).unsqueeze(-1),
                current - previous,
                torch.zeros_like(current),
            )
            v_tile_score = self.v_scorer(torch.cat((current, delta), dim=-1)).squeeze(-1)
            v_tile_score = torch.where(v_mask, v_tile_score, 0.0)
            v_temporal_weights = _masked_softmax(
                shared_energy.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1),
                v_mask,
            )
            v_reliability = torch.where(
                v_mask, evidence.reliability, torch.zeros_like(evidence.reliability)
            )
            gated_v = _positive_only_reliability_gate(v_tile_score, v_reliability)
            v_contribution = F.softplus(self.raw_v_gain).to(v_tile_score.dtype) * (
                v_temporal_weights * gated_v
            ).sum(dim=-1)
        else:
            v_tile_score = evidence.evolution.new_zeros(
                (evidence.batch_size, N_STANDARD_CHANNELS, N_TIME_TILES)
            )
            v_temporal_weights = torch.zeros_like(v_tile_score)
            v_contribution = evidence.evolution.new_zeros(
                (evidence.batch_size, N_STANDARD_CHANNELS)
            )

        prior = self.channel_prior_logits.to(h_contribution.dtype).unsqueeze(0)
        prior = prior.expand(evidence.batch_size, -1)
        event_logits = prior + h_contribution + v_contribution
        prior_only_event = ~phase.any(dim=1)
        # H is complete for every physical channel whenever the phase is valid;
        # no per-channel missingness is inferred from V.
        channel_mask = torch.ones_like(event_logits, dtype=torch.bool)
        event_probabilities = _masked_softmax(event_logits, channel_mask)
        output = FrozenHEventOutput(
            event_logits=event_logits,
            event_probabilities=event_probabilities,
            channel_prior=prior,
            h_contribution=h_contribution,
            v_contribution=v_contribution,
            h_temporal_weights=h_temporal_weights,
            v_temporal_weights=v_temporal_weights,
            global_ictal_support=global_support,
            global_ictal_mask=global_mask,
            h_tile_score=h_tile_score,
            v_tile_score=v_tile_score,
            prior_only_event=prior_only_event,
        )
        if not torch.allclose(
            output.reconstructed_logits(), event_logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("frozen-H contribution decomposition drifted")
        if not torch.allclose(
            event_probabilities.sum(dim=1),
            torch.ones(evidence.batch_size, device=event_probabilities.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("frozen-H event probabilities are not normalized")
        return output


def frozen_h_objective(
    event_probabilities: torch.Tensor,
    batch: FrozenHPatientBatch,
) -> GlobalIVObjectiveOutput:
    return global_i_v_objective(
        event_probabilities,
        batch.base,
        mode="equal_probability_mean",
    )


__all__ = [
    "FROZEN_H_CANDIDATES",
    "FROZEN_H_RECOVERY_SCHEMA",
    "FROZEN_H_SECONDS_PER_TILE",
    "FROZEN_H_TOKEN_DIM",
    "FrozenHCandidate",
    "FrozenHEventOutput",
    "FrozenHNodeLocalizer",
    "FrozenHPatientBatch",
    "FrozenHStandardization",
    "fit_frozen_h_standardization",
    "frozen_h_objective",
    "subset_frozen_h_patient_batch",
]
