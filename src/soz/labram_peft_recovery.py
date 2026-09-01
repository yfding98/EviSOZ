"""Patient-level endpoint-aligned LaBraM PEFT recovery primitives.

The only trainable foundation parameters are supplied by
``OfficialLaBraMMinimalPEFTSuffix``.  This module keeps the downstream
full-phase H+V computation mathematically matched to the frozen v6/v7 control,
while accepting differentiable H tokens.  It deliberately exposes only the
exact positive-set objective; neighbour-expanded, event-level SOZ and
propagation targets are not part of this recovery experiment.
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
from .frozen_h_recovery import FrozenHStandardization
from .geometry import N_NODE_FEATURES, N_STANDARD_CHANNELS, N_TIME_TILES
from .global_i_v_recovery import _positive_only_reliability_gate
from .models.labram_peft import (
    LABRAM_PEFT_PREFIX_TOKENS,
    LABRAM_PEFT_SECONDS_PER_CALL,
    LABRAM_PEFT_TOKEN_DIM,
    OfficialLaBraMMinimalPEFTSuffix,
)
from .onset_contrast_recovery import ScalpOnsetContrastNodeLocalizer
from .temporal_mil_recovery import (
    TemporalMILPatientBatch,
    exact_positive_set_mass_loss,
    subset_patient_batch,
)


LABRAM_PEFT_RECOVERY_SCHEMA = "soz_labram_endpoint_aligned_peft_recovery_v8"
LABRAM_PEFT_EVENT_TILES = 15
LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS = 314
LABRAM_PEFT_MATCHED_HEAD_CANDIDATE = "full_phase_h_v_matched"


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse-softplus input must be positive")
    return math.log(math.expm1(value))


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[: mask.ndim] != mask.shape or mask.dtype != torch.bool:
        raise TypeError("masked mean requires an aligned bool mask")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    count = mask.sum(dim=dim)
    total = torch.where(expanded, values, torch.zeros_like(values)).sum(dim=dim)
    denominator = count.clamp_min(1).to(values.dtype)
    while denominator.ndim < total.ndim:
        denominator = denominator.unsqueeze(-1)
    available = count > 0
    mean = total / denominator
    available_expanded = available
    while available_expanded.ndim < mean.ndim:
        available_expanded = available_expanded.unsqueeze(-1)
    return torch.where(available_expanded, mean, torch.zeros_like(mean)), available


def _masked_min(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("masked min requires aligned values and bool mask")
    available = mask.any(dim=dim)
    minimum = values.masked_fill(~mask, torch.inf).amin(dim=dim)
    return torch.where(available, minimum, torch.zeros_like(minimum)), available


@dataclass(frozen=True)
class LaBraMPEFTPatientBatch:
    """Complete patient bags plus detached block-10 prefix activations."""

    base: TemporalMILPatientBatch
    prefix_tokens: torch.Tensor

    def __post_init__(self) -> None:
        expected = (
            self.base.evidence.batch_size,
            LABRAM_PEFT_EVENT_TILES,
            LABRAM_PEFT_PREFIX_TOKENS,
            LABRAM_PEFT_TOKEN_DIM,
        )
        if tuple(self.prefix_tokens.shape) != expected:
            raise ValueError(
                "LaBraM prefix cache must have shape [E,15,77,200], got "
                f"{tuple(self.prefix_tokens.shape)}"
            )
        if not self.prefix_tokens.is_floating_point():
            raise TypeError("LaBraM prefix cache must be floating point")
        if self.prefix_tokens.requires_grad:
            raise ValueError("LaBraM prefix cache must be detached")
        if not torch.isfinite(self.prefix_tokens).all():
            raise ValueError("LaBraM prefix cache contains non-finite values")
        if self.prefix_tokens.device != self.base.evidence.evolution.device:
            raise ValueError("LaBraM prefix cache and evidence must share a device")

    def to(self, device: str | torch.device) -> "LaBraMPEFTPatientBatch":
        return LaBraMPEFTPatientBatch(
            base=self.base.to(device),
            prefix_tokens=self.prefix_tokens.to(device=device),
        )


def subset_labram_peft_patient_batch(
    full: LaBraMPEFTPatientBatch,
    patient_indices: Sequence[int],
) -> LaBraMPEFTPatientBatch:
    """Select complete patient bags without treating seizures as samples."""

    selected = tuple(int(value) for value in patient_indices)
    base = subset_patient_batch(
        full.base.evidence,
        full.base.event_patient_index,
        full.base.patient_ids,
        full.base.targets,
        full.base.target_mask,
        selected,
    )
    patient_mask = torch.zeros(
        len(full.base.patient_ids),
        dtype=torch.bool,
        device=full.base.event_patient_index.device,
    )
    patient_mask[
        torch.tensor(selected, dtype=torch.long, device=patient_mask.device)
    ] = True
    event_indices = torch.nonzero(
        patient_mask[full.base.event_patient_index], as_tuple=False
    ).flatten()
    return LaBraMPEFTPatientBatch(
        base=base,
        prefix_tokens=full.prefix_tokens.index_select(0, event_indices),
    )


def suffix_node_tokens(
    suffix: OfficialLaBraMMinimalPEFTSuffix,
    prefix_tokens: torch.Tensor,
) -> torch.Tensor:
    """Run differentiable blocks 10--11 and restore the event tile carrier."""

    if not isinstance(suffix, OfficialLaBraMMinimalPEFTSuffix):
        raise TypeError("suffix must be OfficialLaBraMMinimalPEFTSuffix")
    if prefix_tokens.ndim != 4 or tuple(prefix_tokens.shape[1:]) != (
        LABRAM_PEFT_EVENT_TILES,
        LABRAM_PEFT_PREFIX_TOKENS,
        LABRAM_PEFT_TOKEN_DIM,
    ):
        raise ValueError("prefix_tokens must have shape [E,15,77,200]")
    if prefix_tokens.shape[0] < 1:
        raise ValueError("at least one event prefix is required")
    if prefix_tokens.requires_grad:
        raise ValueError("prefix cache must remain detached")
    flat = prefix_tokens.reshape(
        prefix_tokens.shape[0] * LABRAM_PEFT_EVENT_TILES,
        LABRAM_PEFT_PREFIX_TOKENS,
        LABRAM_PEFT_TOKEN_DIM,
    )
    encoded = suffix(flat)
    output = (
        encoded.reshape(
            prefix_tokens.shape[0],
            LABRAM_PEFT_EVENT_TILES,
            N_STANDARD_CHANNELS,
            LABRAM_PEFT_SECONDS_PER_CALL,
            LABRAM_PEFT_TOKEN_DIM,
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    expected = (
        prefix_tokens.shape[0],
        N_STANDARD_CHANNELS,
        LABRAM_PEFT_EVENT_TILES,
        LABRAM_PEFT_SECONDS_PER_CALL,
        LABRAM_PEFT_TOKEN_DIM,
    )
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise RuntimeError("LaBraM PEFT event token carrier is invalid")
    return output


@dataclass(frozen=True)
class PEFTFullPhaseEventOutput:
    event_logits: torch.Tensor
    event_probabilities: torch.Tensor
    channel_prior: torch.Tensor
    main_raw_score: torch.Tensor
    main_contribution: torch.Tensor
    h_main_score: torch.Tensor
    v_main_score: torch.Tensor
    onset_reliability: torch.Tensor
    matched_event: torch.Tensor
    prior_only_event: torch.Tensor
    v_main_valid: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return self.channel_prior + self.main_contribution


class DifferentiableFullPhaseHVHead(nn.Module):
    """The v6/v7 full-phase H+V head with a differentiable H input."""

    def __init__(
        self,
        prior_logits: torch.Tensor,
        standardization: FrozenHStandardization,
    ) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        if not prior_logits.is_floating_point() or not torch.isfinite(
            prior_logits
        ).all():
            raise ValueError("prior_logits must be finite floating point")
        if not isinstance(standardization, FrozenHStandardization):
            raise TypeError("standardization must be FrozenHStandardization")

        # Preserve the exact construction order of
        # ScalpOnsetContrastNodeLocalizer(full_phase_h_v_matched).
        self.v_scorer = nn.Sequential(
            nn.Linear(2 * N_NODE_FEATURES, 8),
            nn.Tanh(),
            nn.Linear(8, 1, bias=False),
        )
        self.raw_v_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.h_scorer = nn.Linear(LABRAM_PEFT_TOKEN_DIM, 1, bias=False)
        self.raw_h_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.register_buffer(
            "channel_prior_logits", prior_logits.detach().float().contiguous()
        )
        self.register_buffer(
            "h_mean", standardization.mean.detach().float().contiguous()
        )
        self.register_buffer(
            "h_scale", standardization.scale.detach().float().contiguous()
        )
        if self.n_trainable_parameters != LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS:
            raise RuntimeError("full-phase H+V head parameter count changed")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        node_tokens: torch.Tensor,
        evidence: DevelopmentIVEvidenceBatch,
    ) -> PEFTFullPhaseEventOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("PEFT head requires DevelopmentIVEvidenceBatch")
        evidence.validate()
        expected = (
            evidence.batch_size,
            N_STANDARD_CHANNELS,
            N_TIME_TILES,
            LABRAM_PEFT_SECONDS_PER_CALL,
            LABRAM_PEFT_TOKEN_DIM,
        )
        if tuple(node_tokens.shape) != expected:
            raise ValueError("node_tokens must have shape [E,19,15,4,200]")
        if not node_tokens.is_floating_point() or not torch.isfinite(
            node_tokens
        ).all():
            raise ValueError("node_tokens must be finite floating point")
        if node_tokens.device != evidence.evolution.device:
            raise ValueError("node tokens and evidence must share a device")

        phase = evidence.phase_mask
        matched = phase[:, :3].all(dim=1) & phase[:, 3:6].all(dim=1)
        event_channel = matched.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS)
        phase_channel = phase.unsqueeze(1).expand(
            -1, N_STANDARD_CHANNELS, -1
        )

        h_tile = node_tokens.mean(dim=3)
        h_tile = (
            h_tile - self.h_mean.to(device=h_tile.device, dtype=h_tile.dtype)
        ) / self.h_scale.to(device=h_tile.device, dtype=h_tile.dtype)
        h_full, _ = _masked_mean(h_tile, phase_channel, dim=2)
        h_gain = F.softplus(self.raw_h_gain).to(h_tile.dtype)
        h_main = h_gain * self.h_scorer(h_full).squeeze(-1)
        h_main = torch.where(event_channel, h_main, torch.zeros_like(h_main))

        v_valid = evidence.evolution_mask & phase.unsqueeze(1)
        current = torch.where(
            v_valid.unsqueeze(-1),
            evidence.evolution,
            torch.zeros_like(evidence.evolution),
        )
        previous = torch.roll(current, shifts=1, dims=2)
        previous_valid = torch.roll(v_valid, shifts=1, dims=2)
        previous_valid[:, :, 0] = False
        delta = torch.where(
            (v_valid & previous_valid).unsqueeze(-1),
            current - previous,
            torch.zeros_like(current),
        )
        v_tile = torch.cat((current, delta), dim=-1)
        v_full, v_full_any = _masked_mean(v_tile, v_valid, dim=2)
        v_main_valid = event_channel & v_full_any
        v_gain = F.softplus(self.raw_v_gain).to(v_tile.dtype)
        v_main = v_gain * self.v_scorer(v_full).squeeze(-1)
        v_main = torch.where(v_main_valid, v_main, torch.zeros_like(v_main))

        reliability, _ = _masked_min(
            evidence.reliability, phase_channel, dim=2
        )
        reliability = torch.where(
            event_channel, reliability, torch.zeros_like(reliability)
        )
        main_raw = h_main + v_main
        main = _positive_only_reliability_gate(main_raw, reliability)
        main = torch.where(event_channel, main, torch.zeros_like(main))

        prior = self.channel_prior_logits.to(
            device=node_tokens.device, dtype=main.dtype
        ).unsqueeze(0).expand(evidence.batch_size, -1)
        logits = prior + main
        probabilities = torch.softmax(logits, dim=1)
        output = PEFTFullPhaseEventOutput(
            event_logits=logits,
            event_probabilities=probabilities,
            channel_prior=prior,
            main_raw_score=main_raw,
            main_contribution=main,
            h_main_score=h_main,
            v_main_score=v_main,
            onset_reliability=reliability,
            matched_event=matched,
            prior_only_event=~matched,
            v_main_valid=v_main_valid,
        )
        if not torch.allclose(
            output.reconstructed_logits(), logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("PEFT full-phase contribution decomposition drifted")
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(evidence.batch_size, device=probabilities.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("PEFT event probabilities are not normalized")
        return output


def seeded_differentiable_full_phase_head(
    prior_logits: torch.Tensor,
    standardization: FrozenHStandardization,
    *,
    seed: int,
    device: str | torch.device,
) -> DifferentiableFullPhaseHVHead:
    """Reproduce the frozen-control head initialization exactly."""

    execution_device = torch.device(device)
    fork_devices: list[int] = []
    if execution_device.type == "cuda":
        fork_devices = [
            execution_device.index if execution_device.index is not None else 0
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        reference = ScalpOnsetContrastNodeLocalizer(
            prior_logits,
            standardization,
            candidate=LABRAM_PEFT_MATCHED_HEAD_CANDIDATE,
        )
    head = DifferentiableFullPhaseHVHead(prior_logits, standardization)
    head.load_state_dict(reference.state_dict(), strict=True)
    return head.to(execution_device)


def exact_patient_set_objective(
    event_logits: torch.Tensor,
    batch: TemporalMILPatientBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean events first, then apply the one preregistered SOZ set loss."""

    if tuple(event_logits.shape) != (batch.evidence.batch_size, N_STANDARD_CHANNELS):
        raise ValueError("event_logits must align with the complete patient bag")
    aggregation = aggregate_patient_logits(
        event_logits, batch.event_patient_index
    )
    expected_ids = torch.arange(
        len(batch.patient_ids),
        device=event_logits.device,
        dtype=torch.long,
    )
    if not torch.equal(aggregation.patient_ids, expected_ids):
        raise RuntimeError("patient aggregation carrier changed")
    loss = exact_positive_set_mass_loss(
        aggregation.logits, batch.targets, batch.target_mask
    )
    return loss, aggregation.logits


__all__ = [
    "DifferentiableFullPhaseHVHead",
    "LABRAM_PEFT_EVENT_TILES",
    "LABRAM_PEFT_HEAD_TRAINABLE_PARAMETERS",
    "LABRAM_PEFT_MATCHED_HEAD_CANDIDATE",
    "LABRAM_PEFT_RECOVERY_SCHEMA",
    "LaBraMPEFTPatientBatch",
    "PEFTFullPhaseEventOutput",
    "exact_patient_set_objective",
    "seeded_differentiable_full_phase_head",
    "subset_labram_peft_patient_batch",
    "suffix_node_tokens",
]
