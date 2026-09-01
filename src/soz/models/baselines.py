"""Minimal scientific controls for the evidence-bottleneck hypothesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..geometry import N_STANDARD_CHANNELS


N_MATCHED_DIRECT_TILES = 15
MATCHED_DIRECT_PHASE_COMPONENTS = (
    "pre",
    "early",
    "late",
    "early_minus_pre",
    "late_minus_early",
)
_MATCHED_DIRECT_PHASE_BOUNDS = ((0, 3), (3, 6), (6, 15))


def _matched_direct_phase_components(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != mask.shape or values.ndim != 3:
        raise ValueError("Direct tile values/mask must share [B,19,15]")
    if values.shape[-1] != N_MATCHED_DIRECT_TILES:
        raise ValueError("Matched direct baseline requires exactly 15 tiles")
    phase_values: list[torch.Tensor] = []
    phase_masks: list[torch.Tensor] = []
    for start, stop in _MATCHED_DIRECT_PHASE_BOUNDS:
        phase_mask = mask[..., start:stop]
        valid = phase_mask.any(dim=-1)
        numerator = (
            values[..., start:stop] * phase_mask.to(dtype=values.dtype)
        ).sum(dim=-1)
        denominator = phase_mask.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
        phase_values.append(torch.where(valid, numerator / denominator, 0.0))
        phase_masks.append(valid)
    pre, early, late = phase_values
    pre_valid, early_valid, late_valid = phase_masks
    early_pre_valid = early_valid & pre_valid
    late_early_valid = late_valid & early_valid
    components = torch.stack(
        (
            pre,
            early,
            late,
            torch.where(early_pre_valid, early - pre, 0.0),
            torch.where(late_early_valid, late - early, 0.0),
        ),
        dim=-1,
    )
    component_mask = torch.stack(
        (pre_valid, early_valid, late_valid, early_pre_valid, late_early_valid),
        dim=-1,
    )
    return components, component_mask


class ElectrodePrevalencePrior(nn.Module):
    """Train-only smoothed channel prevalence with no EEG input."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        if tuple(logits.shape) != (N_STANDARD_CHANNELS,):
            raise ValueError("Prevalence logits must have shape [19]")
        self.register_buffer("logits", logits.detach().clone(), persistent=True)

    @classmethod
    def fit(
        cls,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        alpha: float = 1.0,
    ) -> "ElectrodePrevalencePrior":
        if tuple(targets.shape) != tuple(target_mask.shape) or targets.ndim != 2 or targets.shape[1] != 19:
            raise ValueError("targets and target_mask must have shape [P,19]")
        if target_mask.dtype != torch.bool:
            raise TypeError("target_mask must be torch.bool")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        observed_targets = targets[target_mask]
        if observed_targets.numel() and not torch.all(
            (observed_targets == 0) | (observed_targets == 1)
        ):
            raise ValueError("Observed targets must be binary")
        positives = (targets * target_mask.to(targets.dtype)).sum(dim=0)
        observed = target_mask.sum(dim=0).to(targets.dtype)
        probability = (positives + float(alpha)) / (observed + 2.0 * float(alpha))
        logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return cls(logits)

    def forward(self, n_patients: int) -> torch.Tensor:
        if n_patients < 1:
            raise ValueError("n_patients must be positive")
        return self.logits.unsqueeze(0).expand(int(n_patients), -1)


class ChannelIdentityBaseline(nn.Module):
    """Learned channel bias only; controls for label prevalence and identity."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_logits = nn.Parameter(torch.zeros(N_STANDARD_CHANNELS))

    def forward(self, n_patients: int) -> torch.Tensor:
        if n_patients < 1:
            raise ValueError("n_patients must be positive")
        return self.channel_logits.unsqueeze(0).expand(int(n_patients), -1)


class DirectFrozenTokenHead(nn.Module):
    """Legacy time-mean diagnostic; not the matched direct comparator."""

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 32) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.shared_head = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.channel_bias = nn.Parameter(torch.zeros(N_STANDARD_CHANNELS))

    def forward(self, frozen_tokens: torch.Tensor) -> torch.Tensor:
        if frozen_tokens.ndim != 4 or frozen_tokens.shape[1] != N_STANDARD_CHANNELS or frozen_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"frozen_tokens must have shape [B,19,T,{self.token_dim}]"
            )
        if frozen_tokens.requires_grad:
            raise ValueError("Direct baseline requires detached frozen foundation tokens")
        pooled = frozen_tokens.mean(dim=2)
        return self.shared_head(pooled).squeeze(-1) + self.channel_bias.unsqueeze(0)


@dataclass(frozen=True)
class MatchedDirectOutput:
    """Event logits and the same five temporal components as the reasoner."""

    event_logits: torch.Tensor
    channel_prior: torch.Tensor
    tile_scores: torch.Tensor
    phase_components: torch.Tensor
    phase_component_mask: torch.Tensor
    phase_contributions: torch.Tensor

    def reconstructed_logits(self) -> torch.Tensor:
        return self.channel_prior + self.phase_contributions.sum(dim=-1)


class MatchedDirectFrozenTokenHead(nn.Module):
    """Capacity- and time-structure-matched detached-LaBraM comparator.

    The primary input is ``[B,19,15,4,200]``: fifteen independent four-second
    LaBraM calls with four read slots.  Read slots are averaged inside each
    tile, then a single shared linear scorer and the same pre/early/late plus
    two-contrast structure used by the evidence reasoner produce event logits.
    It uses no concept labels and is excluded from the clinical architecture,
    but must share patient aggregation, SOZ objective and event window with the
    proposed model during scientific comparison.
    """

    def __init__(self, *, token_dim: int = 200) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        if self.token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.tile_scorer = nn.Linear(self.token_dim, 1)
        self.phase_weights = nn.Parameter(
            torch.full((len(MATCHED_DIRECT_PHASE_COMPONENTS),), 0.2)
        )
        self.channel_bias = nn.Parameter(torch.zeros(N_STANDARD_CHANNELS))

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        frozen_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        ictal_phase_mask: torch.Tensor,
    ) -> MatchedDirectOutput:
        if frozen_tokens.ndim != 5 or tuple(frozen_tokens.shape[1:4]) != (
            N_STANDARD_CHANNELS,
            N_MATCHED_DIRECT_TILES,
            4,
        ) or frozen_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "frozen_tokens must have shape "
                f"[B,19,15,4,{self.token_dim}]"
            )
        batch_size = frozen_tokens.shape[0]
        if tuple(token_mask.shape) != (
            batch_size,
            N_STANDARD_CHANNELS,
            N_MATCHED_DIRECT_TILES,
        ):
            raise ValueError("token_mask must have shape [B,19,15]")
        if tuple(ictal_phase_mask.shape) != (
            batch_size,
            N_MATCHED_DIRECT_TILES,
        ):
            raise ValueError("ictal_phase_mask must have shape [B,15]")
        if token_mask.dtype != torch.bool or ictal_phase_mask.dtype != torch.bool:
            raise TypeError("Matched direct masks must be bool")
        if frozen_tokens.requires_grad:
            raise ValueError(
                "Matched direct baseline requires detached frozen foundation tokens"
            )
        if not frozen_tokens.is_floating_point() or not torch.isfinite(
            frozen_tokens
        ).all():
            raise ValueError("Matched direct tokens must be finite floating point")
        if (
            frozen_tokens.device != token_mask.device
            or token_mask.device != ictal_phase_mask.device
        ):
            raise ValueError("Matched direct tokens and masks must share a device")

        effective_mask = token_mask & ictal_phase_mask.unsqueeze(1)
        pooled_slots = frozen_tokens.mean(dim=3)
        safe_tokens = torch.where(
            effective_mask.unsqueeze(-1), pooled_slots, 0.0
        )
        tile_scores = self.tile_scorer(safe_tokens).squeeze(-1)
        tile_scores = tile_scores * effective_mask.to(dtype=tile_scores.dtype)
        phase_components, phase_component_mask = _matched_direct_phase_components(
            tile_scores, effective_mask
        )
        phase_contributions = (
            phase_components
            * phase_component_mask.to(dtype=phase_components.dtype)
            * self.phase_weights.to(dtype=phase_components.dtype)
        )
        channel_prior = self.channel_bias.to(dtype=frozen_tokens.dtype).unsqueeze(0)
        channel_prior = channel_prior.expand(batch_size, -1)
        event_logits = channel_prior + phase_contributions.sum(dim=-1)
        return MatchedDirectOutput(
            event_logits=event_logits,
            channel_prior=channel_prior,
            tile_scores=tile_scores,
            phase_components=phase_components,
            phase_component_mask=phase_component_mask,
            phase_contributions=phase_contributions,
        )
