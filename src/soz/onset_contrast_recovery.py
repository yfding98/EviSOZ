"""Frozen-LaBraM scalp-visible onset-contrast recovery heads.

This module is a deliberately small source-train experiment.  It contrasts
the fixed ``[-12, 0)`` and ``[0, 12)`` scalp-visible phases; it does not claim
that the contrast is a seizure-onset-zone label or a propagation label.  The
late phase can only corroborate a positive early contrast and can never add a
channel score by itself.

The only learned inputs are detached, physical-node-indexed LaBraM tokens
(``H``) and the observable evolution descriptor (``V``).  ``I`` is absent by
construction.  All projections are shared across physical channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .development_reasoner import DevelopmentIVEvidenceBatch
from .frozen_h_recovery import (
    FROZEN_H_SECONDS_PER_TILE,
    FROZEN_H_TOKEN_DIM,
    FrozenHPatientBatch,
    FrozenHStandardization,
    subset_frozen_h_patient_batch,
)
from .geometry import N_NODE_FEATURES, N_STANDARD_CHANNELS, N_TIME_TILES
from .temporal_mil_recovery import (
    TemporalMILObjectiveOutput,
    _positive_only_reliability_gate,
    temporal_mil_objective,
)


ONSET_CONTRAST_RECOVERY_SCHEMA = (
    "soz_labram_scalp_visible_onset_contrast_recovery_v6"
)
OnsetContrastCandidate = Literal[
    "onset_contrast_v_only",
    "onset_contrast_h_v",
    "full_phase_h_v_matched",
]
ONSET_CONTRAST_CANDIDATES: tuple[OnsetContrastCandidate, ...] = (
    "onset_contrast_v_only",
    "onset_contrast_h_v",
    "full_phase_h_v_matched",
)

PRE_TILES = slice(0, 3)
EARLY_TILES = slice(3, 6)
LATE_TILES = slice(6, N_TIME_TILES)
LATE_CORROBORATION_SCALE = 0.25


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse-softplus input must be positive")
    return math.log(math.expm1(value))


def _matched_event_mask(phase_mask: torch.Tensor) -> torch.Tensor:
    """Events with all three pre and all three early phase tiles."""

    if phase_mask.dtype != torch.bool or tuple(phase_mask.shape[1:]) != (
        N_TIME_TILES,
    ):
        raise TypeError("phase_mask must be bool [E,15]")
    return phase_mask[:, PRE_TILES].all(dim=1) & phase_mask[:, EARLY_TILES].all(
        dim=1
    )


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite masked mean and an availability mask with ``dim`` removed."""

    if mask.dtype != torch.bool or value.shape[: mask.ndim] != mask.shape:
        raise TypeError("masked mean requires a bool prefix mask")
    expanded = mask
    for _ in range(value.ndim - mask.ndim):
        expanded = expanded.unsqueeze(-1)
    count = mask.sum(dim=dim)
    summed = torch.where(expanded, value, torch.zeros_like(value)).sum(dim=dim)
    denominator = count.clamp_min(1).to(value.dtype)
    for _ in range(summed.ndim - denominator.ndim):
        denominator = denominator.unsqueeze(-1)
    mean = summed / denominator
    available = count > 0
    available_expanded = available
    for _ in range(mean.ndim - available.ndim):
        available_expanded = available_expanded.unsqueeze(-1)
    return torch.where(available_expanded, mean, torch.zeros_like(mean)), available


def _masked_min(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked conservative minimum, returning zero when nothing is valid."""

    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("masked min requires aligned values and bool mask")
    available = mask.any(dim=dim)
    minimum = value.masked_fill(~mask, torch.inf).amin(dim=dim)
    return torch.where(available, minimum, torch.zeros_like(minimum)), available


def fit_onset_contrast_standardization(
    batch: FrozenHPatientBatch,
) -> FrozenHStandardization:
    """Fit target-free H moments on matched events in this train fold only.

    Prior-only events are intentionally excluded: they cannot influence either
    learned contributions or the transform applied to another event.  Within
    a matched event, every phase-valid tile is eligible, including available
    late tiles, because the same transform serves the primary and matched
    full-phase control.
    """

    if not isinstance(batch, FrozenHPatientBatch):
        raise TypeError("batch must be FrozenHPatientBatch")
    evidence = batch.base.evidence
    matched = _matched_event_mask(evidence.phase_mask)
    tile_tokens = batch.node_tokens.mean(dim=3)
    valid = (evidence.phase_mask & matched.unsqueeze(1)).unsqueeze(1).expand(
        -1, N_STANDARD_CHANNELS, -1
    )
    selected = tile_tokens[valid]
    if selected.ndim != 2 or selected.shape[0] < 2:
        raise ValueError(
            "onset-contrast standardization requires matched train-fold H rows"
        )
    mean = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False).clamp_min(1e-5)
    return FrozenHStandardization(mean=mean.detach(), scale=scale.detach())


def subset_onset_contrast_patient_batch(
    full: FrozenHPatientBatch,
    patient_indices: Sequence[int],
) -> FrozenHPatientBatch:
    """Select complete patient bags; events, including prior-only ones, stay."""

    return subset_frozen_h_patient_batch(full, patient_indices)


@dataclass(frozen=True)
class OnsetContrastEventOutput:
    """Event logits plus a complete audit decomposition."""

    event_logits: torch.Tensor
    event_probabilities: torch.Tensor
    channel_prior: torch.Tensor
    main_raw_score: torch.Tensor
    main_contribution: torch.Tensor
    h_main_score: torch.Tensor
    v_main_score: torch.Tensor
    late_raw_score: torch.Tensor
    late_gated_score: torch.Tensor
    h_late_score: torch.Tensor
    v_late_score: torch.Tensor
    late_corroboration: torch.Tensor
    onset_reliability: torch.Tensor
    late_reliability: torch.Tensor
    matched_event: torch.Tensor
    prior_only_event: torch.Tensor
    pre_phase_mask: torch.Tensor
    early_phase_mask: torch.Tensor
    late_phase_mask: torch.Tensor
    v_main_valid: torch.Tensor
    v_late_valid: torch.Tensor
    late_phase_available: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        """Rebuild logits; late scores are audit-only except corroboration."""

        return self.channel_prior + self.main_contribution + self.late_corroboration


class ScalpOnsetContrastNodeLocalizer(nn.Module):
    """Low-capacity, shared-channel frozen-H/V localization probe."""

    def __init__(
        self,
        prior_logits: torch.Tensor,
        standardization: FrozenHStandardization,
        *,
        candidate: OnsetContrastCandidate,
    ) -> None:
        super().__init__()
        if candidate not in ONSET_CONTRAST_CANDIDATES:
            raise ValueError("unknown onset-contrast candidate")
        if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("prior_logits must have shape [19]")
        if not prior_logits.is_floating_point() or not torch.isfinite(
            prior_logits
        ).all():
            raise ValueError("prior_logits must be finite floating-point values")
        if not isinstance(standardization, FrozenHStandardization):
            raise TypeError("standardization must be FrozenHStandardization")

        self.candidate = candidate
        self.use_h = candidate != "onset_contrast_v_only"
        self.use_onset_contrast = candidate != "full_phase_h_v_matched"

        # Instantiate V first in every candidate.  Re-seeding before candidate
        # construction therefore gives the same V initialization, while the
        # two H+V candidates also receive the same subsequent H initialization.
        self.v_scorer = nn.Sequential(
            nn.Linear(2 * N_NODE_FEATURES, 8),
            nn.Tanh(),
            nn.Linear(8, 1, bias=False),
        )
        self.raw_v_gain = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        if self.use_h:
            self.h_scorer: nn.Linear | None = nn.Linear(
                FROZEN_H_TOKEN_DIM, 1, bias=False
            )
            self.raw_h_gain: nn.Parameter | None = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
            )
        else:
            self.h_scorer = None
            self.register_parameter("raw_h_gain", None)

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
            raise ValueError("onset-contrast head exceeds its capacity gate")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _h_tiles(self, node_tokens: torch.Tensor) -> torch.Tensor:
        tile = node_tokens.mean(dim=3)
        return (
            tile - self.h_mean.to(device=tile.device, dtype=tile.dtype)
        ) / self.h_scale.to(device=tile.device, dtype=tile.dtype)

    @staticmethod
    def _v_tiles(
        evidence: DevelopmentIVEvidenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = evidence.phase_mask.unsqueeze(1)
        valid = evidence.evolution_mask & phase
        current = torch.where(
            valid.unsqueeze(-1), evidence.evolution, torch.zeros_like(evidence.evolution)
        )
        previous = torch.roll(current, shifts=1, dims=2)
        previous_valid = torch.roll(valid, shifts=1, dims=2)
        previous_valid[:, :, 0] = False
        delta_valid = valid & previous_valid
        delta = torch.where(
            delta_valid.unsqueeze(-1),
            current - previous,
            torch.zeros_like(current),
        )
        return torch.cat((current, delta), dim=-1), valid

    def forward(
        self,
        node_tokens: torch.Tensor,
        evidence: DevelopmentIVEvidenceBatch,
    ) -> OnsetContrastEventOutput:
        if not isinstance(evidence, DevelopmentIVEvidenceBatch):
            raise TypeError("onset-contrast localizer accepts development evidence")
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
        if not node_tokens.is_floating_point() or not torch.isfinite(
            node_tokens
        ).all():
            raise ValueError("node_tokens must be finite floating point")
        if node_tokens.requires_grad:
            raise ValueError("onset-contrast localizer accepts detached H tokens only")
        if node_tokens.device != evidence.evolution.device:
            raise ValueError("node tokens and evidence must share a device")

        events = evidence.batch_size
        matched = _matched_event_mask(evidence.phase_mask)
        event_channel = matched.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS)
        pre_phase = evidence.phase_mask[:, PRE_TILES]
        early_phase = evidence.phase_mask[:, EARLY_TILES]
        late_phase = evidence.phase_mask[:, LATE_TILES]
        late_available = late_phase.any(dim=1) & matched

        h_tile = self._h_tiles(node_tokens)
        v_tile, v_mask = self._v_tiles(evidence)
        zeros = node_tokens.new_zeros((events, N_STANDARD_CHANNELS))

        # The six-tile gate is formed once and applied only after H/V have
        # produced a single signed onset contrast.
        onset_reliability = evidence.reliability[:, :, :6].amin(dim=2)
        onset_reliability = torch.where(
            event_channel, onset_reliability, torch.zeros_like(onset_reliability)
        )

        h_main = zeros
        v_main = zeros
        h_late = zeros
        v_late = zeros
        v_late_valid = torch.zeros_like(event_channel)
        late_reliability = zeros

        if self.use_onset_contrast:
            h_pre = h_tile[:, :, PRE_TILES].mean(dim=2)
            h_early = h_tile[:, :, EARLY_TILES].mean(dim=2)
            if self.use_h:
                assert self.h_scorer is not None and self.raw_h_gain is not None
                h_gain = F.softplus(self.raw_h_gain).to(h_tile.dtype)
                h_main = h_gain * self.h_scorer(h_early - h_pre).squeeze(-1)
                h_main = torch.where(event_channel, h_main, torch.zeros_like(h_main))

            v_pre = v_tile[:, :, PRE_TILES].mean(dim=2)
            v_early = v_tile[:, :, EARLY_TILES].mean(dim=2)
            v_pre_valid = v_mask[:, :, PRE_TILES].all(dim=2)
            v_early_valid = v_mask[:, :, EARLY_TILES].all(dim=2)
            v_main_valid = event_channel & v_pre_valid & v_early_valid
            v_gain = F.softplus(self.raw_v_gain).to(v_tile.dtype)
            v_main = v_gain * self.v_scorer(v_early - v_pre).squeeze(-1)
            v_main = torch.where(v_main_valid, v_main, torch.zeros_like(v_main))

            late_h_mean, _ = _masked_mean(
                h_tile[:, :, LATE_TILES],
                late_phase.unsqueeze(1).expand(-1, N_STANDARD_CHANNELS, -1),
                dim=2,
            )
            if self.use_h:
                assert self.h_scorer is not None and self.raw_h_gain is not None
                h_gain = F.softplus(self.raw_h_gain).to(h_tile.dtype)
                h_late = h_gain * self.h_scorer(late_h_mean - h_pre).squeeze(-1)
                h_late = torch.where(
                    late_available.unsqueeze(1), h_late, torch.zeros_like(h_late)
                )

            late_v_mean, late_v_any = _masked_mean(
                v_tile[:, :, LATE_TILES], v_mask[:, :, LATE_TILES], dim=2
            )
            v_late_valid = event_channel & v_pre_valid & late_v_any
            v_late = v_gain * self.v_scorer(late_v_mean - v_pre).squeeze(-1)
            v_late = torch.where(v_late_valid, v_late, torch.zeros_like(v_late))

            reliability_mask = evidence.phase_mask.unsqueeze(1).expand(
                -1, N_STANDARD_CHANNELS, -1
            ).clone()
            reliability_mask[:, :, EARLY_TILES] = False
            reliability_mask &= (
                matched.unsqueeze(1).unsqueeze(2)
                & (
                    torch.arange(N_TIME_TILES, device=node_tokens.device)
                    .view(1, 1, -1)
                    .lt(3)
                    | torch.arange(N_TIME_TILES, device=node_tokens.device)
                    .view(1, 1, -1)
                    .ge(6)
                )
            )
            late_reliability, _ = _masked_min(
                evidence.reliability, reliability_mask, dim=2
            )
            late_reliability = torch.where(
                late_available.unsqueeze(1),
                late_reliability,
                torch.zeros_like(late_reliability),
            )
        else:
            phase_mask = evidence.phase_mask.unsqueeze(1).expand(
                -1, N_STANDARD_CHANNELS, -1
            )
            h_full, _ = _masked_mean(h_tile, phase_mask, dim=2)
            assert self.h_scorer is not None and self.raw_h_gain is not None
            h_gain = F.softplus(self.raw_h_gain).to(h_tile.dtype)
            h_main = h_gain * self.h_scorer(h_full).squeeze(-1)
            h_main = torch.where(event_channel, h_main, torch.zeros_like(h_main))

            v_full, v_full_any = _masked_mean(v_tile, v_mask, dim=2)
            v_main_valid = event_channel & v_full_any
            v_gain = F.softplus(self.raw_v_gain).to(v_tile.dtype)
            v_main = v_gain * self.v_scorer(v_full).squeeze(-1)
            v_main = torch.where(v_main_valid, v_main, torch.zeros_like(v_main))

            onset_reliability, _ = _masked_min(
                evidence.reliability, phase_mask, dim=2
            )
            onset_reliability = torch.where(
                event_channel,
                onset_reliability,
                torch.zeros_like(onset_reliability),
            )

        main_raw = h_main + v_main
        main = _positive_only_reliability_gate(main_raw, onset_reliability)
        main = torch.where(event_channel, main, torch.zeros_like(main))

        if self.use_onset_contrast:
            late_raw = h_late + v_late
            late_gated = _positive_only_reliability_gate(
                late_raw, late_reliability
            )
            late_gated = torch.where(
                late_available.unsqueeze(1),
                late_gated,
                torch.zeros_like(late_gated),
            )
            corroboration = (
                LATE_CORROBORATION_SCALE
                * F.relu(main)
                * torch.tanh(F.relu(late_gated))
            )
        else:
            late_raw = zeros
            late_gated = zeros
            corroboration = zeros

        prior = self.channel_prior_logits.to(
            device=node_tokens.device, dtype=main.dtype
        ).unsqueeze(0).expand(events, -1)
        logits = prior + main + corroboration
        probabilities = torch.softmax(logits, dim=1)
        output = OnsetContrastEventOutput(
            event_logits=logits,
            event_probabilities=probabilities,
            channel_prior=prior,
            main_raw_score=main_raw,
            main_contribution=main,
            h_main_score=h_main,
            v_main_score=v_main,
            late_raw_score=late_raw,
            late_gated_score=late_gated,
            h_late_score=h_late,
            v_late_score=v_late,
            late_corroboration=corroboration,
            onset_reliability=onset_reliability,
            late_reliability=late_reliability,
            matched_event=matched,
            prior_only_event=~matched,
            pre_phase_mask=pre_phase,
            early_phase_mask=early_phase,
            late_phase_mask=late_phase,
            v_main_valid=v_main_valid,
            v_late_valid=v_late_valid,
            late_phase_available=late_available,
        )
        if not torch.allclose(
            output.reconstructed_logits(), logits, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("onset-contrast contribution decomposition drifted")
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(events, device=probabilities.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("onset-contrast event probabilities are not normalized")
        return output


def onset_contrast_objective(
    event_logits: torch.Tensor,
    batch: FrozenHPatientBatch,
) -> TemporalMILObjectiveOutput:
    """Frozen exact-target objective; neighbour targets are never introduced."""

    if not isinstance(batch, FrozenHPatientBatch):
        raise TypeError("batch must be FrozenHPatientBatch")
    return temporal_mil_objective(
        event_logits,
        batch.base,
        neighbor_weight=0.0,
    )


__all__ = [
    "EARLY_TILES",
    "LATE_CORROBORATION_SCALE",
    "LATE_TILES",
    "ONSET_CONTRAST_CANDIDATES",
    "ONSET_CONTRAST_RECOVERY_SCHEMA",
    "OnsetContrastCandidate",
    "OnsetContrastEventOutput",
    "PRE_TILES",
    "ScalpOnsetContrastNodeLocalizer",
    "fit_onset_contrast_standardization",
    "onset_contrast_objective",
    "subset_onset_contrast_patient_batch",
]
