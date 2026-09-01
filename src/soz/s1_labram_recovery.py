"""Frozen LaBraM A0/A1/A2 primitives for the label-fresh TUSZ S1 cohort.

The module is intentionally small.  A shared linear projection scores every
channel from official LaBraM token features; there are no free per-channel
biases, graph parameters, patient embeddings, or concept pseudo-labels.
Patient targets are used only after equal-event aggregation.  A2 adds a
deterministic hierarchy loss derived from the same channel logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .aggregation import aggregate_patient_logits
from .clinical_reporting import CLINICAL_SCALP_REGIONS, LATERALITY_GROUPS
from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19
from .positive_set_endpoint_residual import positive_set_mass_loss


S1_LABRAM_RECOVERY_SCHEMA = "tusz_eeg_only_s1_labram_a0_a1_a2_v1"
S1_EVENT_TILES = 15
S1_SECONDS_PER_TILE = 4
S1_TOKEN_DIM = 200
S1_PREFIX_TOKENS = 77
S1_PROJECTOR_TRAINABLE_PARAMETERS = 200
S1_HIERARCHY_WEIGHT = 0.25


class S1SharedChannelProjector(nn.Module):
    """One shared 200-to-1 map applied to every channel-time token."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(S1_TOKEN_DIM, 1, bias=False)
        if self.n_trainable_parameters != S1_PROJECTOR_TRAINABLE_PARAMETERS:
            raise RuntimeError("S1 shared projector parameter count changed")

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        if channel_tokens.ndim != 4 or tuple(channel_tokens.shape[1:]) != (
            N_STANDARD_CHANNELS,
            S1_SECONDS_PER_TILE,
            S1_TOKEN_DIM,
        ):
            raise ValueError(
                "S1 channel tokens must have shape [BT,19,4,200]"
            )
        if not channel_tokens.is_floating_point() or not torch.isfinite(
            channel_tokens
        ).all():
            raise ValueError("S1 channel tokens must be finite floating point")
        scores = self.projection(channel_tokens).squeeze(-1)
        return scores.mean(dim=2)


def event_logits_from_prefix(
    suffix: nn.Module,
    projector: S1SharedChannelProjector,
    prefix_tokens: torch.Tensor,
    channel_prior_logits: torch.Tensor,
) -> torch.Tensor:
    """Score complete 60-s events while retaining the 15 official calls."""

    if not isinstance(suffix, nn.Module):
        raise TypeError("suffix must be an nn.Module")
    if not isinstance(projector, S1SharedChannelProjector):
        raise TypeError("projector must be S1SharedChannelProjector")
    if prefix_tokens.ndim != 4 or tuple(prefix_tokens.shape[1:]) != (
        S1_EVENT_TILES,
        S1_PREFIX_TOKENS,
        S1_TOKEN_DIM,
    ):
        raise ValueError("S1 prefix must have shape [E,15,77,200]")
    if prefix_tokens.shape[0] < 1:
        raise ValueError("S1 event batch must be non-empty")
    if prefix_tokens.requires_grad:
        raise ValueError("S1 block-9 prefix cache must remain detached")
    if not prefix_tokens.is_floating_point() or not torch.isfinite(
        prefix_tokens
    ).all():
        raise ValueError("S1 prefix must be finite floating point")
    if tuple(channel_prior_logits.shape) != (N_STANDARD_CHANNELS,):
        raise ValueError("S1 channel prior must have shape [19]")
    if not channel_prior_logits.is_floating_point() or not torch.isfinite(
        channel_prior_logits
    ).all():
        raise ValueError("S1 channel prior must be finite floating point")
    if channel_prior_logits.device != prefix_tokens.device:
        raise ValueError("S1 channel prior and prefix must share a device")

    events = prefix_tokens.shape[0]
    flat = prefix_tokens.reshape(
        events * S1_EVENT_TILES, S1_PREFIX_TOKENS, S1_TOKEN_DIM
    )
    channel_tokens = suffix(flat)
    expected = (
        events * S1_EVENT_TILES,
        N_STANDARD_CHANNELS,
        S1_SECONDS_PER_TILE,
        S1_TOKEN_DIM,
    )
    if tuple(channel_tokens.shape) != expected or not torch.isfinite(
        channel_tokens
    ).all():
        raise RuntimeError("S1 LaBraM suffix output contract changed")
    call_logits = projector(channel_tokens).reshape(
        events, S1_EVENT_TILES, N_STANDARD_CHANNELS
    )
    event_logits = call_logits.mean(dim=1) + channel_prior_logits.unsqueeze(0)
    if tuple(event_logits.shape) != (events, N_STANDARD_CHANNELS) or not (
        torch.isfinite(event_logits).all()
    ):
        raise RuntimeError("S1 event logit carrier is invalid")
    return event_logits


def fold_channel_prior_logits(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Fold-train-only Jeffreys prior; unknown/spread-masked cells are absent."""

    if targets.ndim != 2 or targets.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("S1 targets must have shape [P,19]")
    if tuple(target_mask.shape) != tuple(targets.shape) or (
        target_mask.dtype != torch.bool
    ):
        raise TypeError("S1 target_mask must be bool [P,19]")
    observed = targets[target_mask]
    if not targets.is_floating_point() or not torch.isfinite(observed).all() or (
        observed.numel() and not torch.all((observed == 0) | (observed == 1))
    ):
        raise ValueError("S1 observed targets must be finite binary values")
    positive = ((targets == 1) & target_mask).sum(dim=0).float()
    count = target_mask.sum(dim=0).float()
    prevalence = (positive + 0.5) / (count + 1.0)
    return torch.logit(prevalence.clamp(1e-4, 1.0 - 1e-4))


def aggregate_complete_patient_bags(
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    patient_count: int,
) -> torch.Tensor:
    """Equal-mean every patient's complete event bag exactly once."""

    if event_logits.ndim != 2 or event_logits.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("S1 event logits must have shape [E,19]")
    if event_patient_index.dtype != torch.long or tuple(
        event_patient_index.shape
    ) != (event_logits.shape[0],):
        raise TypeError("S1 event_patient_index must be long [E]")
    if type(patient_count) is not int or patient_count < 1:
        raise ValueError("S1 patient_count must be positive")
    if event_patient_index.numel() < patient_count or (
        int(event_patient_index.min()) != 0
        or int(event_patient_index.max()) != patient_count - 1
        or int(torch.unique(event_patient_index).numel()) != patient_count
    ):
        raise ValueError("Every S1 patient must own a complete non-empty event bag")
    aggregation = aggregate_patient_logits(event_logits, event_patient_index)
    expected = torch.arange(
        patient_count, dtype=torch.long, device=event_patient_index.device
    )
    if not torch.equal(aggregation.patient_ids, expected):
        raise RuntimeError("S1 patient aggregation identity changed")
    expected_counts = torch.bincount(
        event_patient_index, minlength=patient_count
    )
    if not torch.equal(aggregation.event_counts, expected_counts):
        raise RuntimeError("S1 patient event bags were truncated or reweighted")
    return aggregation.logits


def _group_set_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    groups: Mapping[str, Sequence[str]],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for patient in range(logits.shape[0]):
        observed_group_masses: list[torch.Tensor] = []
        positive_group_masses: list[torch.Tensor] = []
        for members in groups.values():
            indices = torch.tensor(
                [CHANNEL_INDEX[channel] for channel in members],
                dtype=torch.long,
                device=logits.device,
            )
            observed = target_mask[patient].index_select(0, indices)
            if not bool(observed.any()):
                continue
            observed_indices = indices[observed]
            group_mass = torch.logsumexp(
                logits[patient].index_select(0, observed_indices), dim=0
            )
            observed_group_masses.append(group_mass)
            positive = (targets[patient] == 1) & target_mask[patient]
            if bool(positive.index_select(0, indices).any()):
                positive_group_masses.append(group_mass)
        if not observed_group_masses or not positive_group_masses:
            raise ValueError("Every S1 patient needs an observed positive hierarchy group")
        rows.append(
            torch.logsumexp(torch.stack(observed_group_masses), dim=0)
            - torch.logsumexp(torch.stack(positive_group_masses), dim=0)
        )
    return torch.stack(rows).mean()


def hierarchy_set_mass_loss(
    patient_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean frozen region and laterality mass losses from channel scores."""

    # The exact channel loss performs the complete input validation first.
    positive_set_mass_loss(patient_logits, targets, target_mask)
    region = _group_set_mass_loss(
        patient_logits, targets, target_mask, CLINICAL_SCALP_REGIONS
    )
    laterality = _group_set_mass_loss(
        patient_logits, targets, target_mask, LATERALITY_GROUPS
    )
    return 0.5 * (region + laterality)


@dataclass(frozen=True)
class S1ObjectiveOutput:
    total: torch.Tensor
    exact_set_mass: torch.Tensor
    hierarchy_set_mass: torch.Tensor


def s1_patient_objective(
    patient_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    arm: str,
) -> S1ObjectiveOutput:
    """Closed objective: only A2 receives the fixed 0.25 hierarchy term."""

    if arm not in {"A0", "A1", "A2"}:
        raise ValueError("S1 arm must be A0, A1, or A2")
    exact = positive_set_mass_loss(patient_logits, targets, target_mask)
    if arm == "A2":
        hierarchy = hierarchy_set_mass_loss(
            patient_logits, targets, target_mask
        )
        total = exact + S1_HIERARCHY_WEIGHT * hierarchy
    else:
        hierarchy = exact.detach() * 0.0
        total = exact
    return S1ObjectiveOutput(
        total=total,
        exact_set_mass=exact,
        hierarchy_set_mass=hierarchy,
    )


def validate_protocol_partitions(
    regions: Mapping[str, Sequence[str]],
    lateralities: Mapping[str, Sequence[str]],
) -> None:
    """Bind a serialized preregistration to the frozen report partitions."""

    normalized_regions = {
        str(name): tuple(str(value) for value in members)
        for name, members in regions.items()
    }
    normalized_lateralities = {
        str(name): tuple(str(value) for value in members)
        for name, members in lateralities.items()
    }
    if normalized_regions != CLINICAL_SCALP_REGIONS:
        raise ValueError("S1 protocol region partition changed")
    if normalized_lateralities != LATERALITY_GROUPS:
        raise ValueError("S1 protocol laterality partition changed")
    for name, groups in (
        ("regions", normalized_regions),
        ("lateralities", normalized_lateralities),
    ):
        members = tuple(channel for group in groups.values() for channel in group)
        if len(members) != len(STANDARD_19) or set(members) != set(STANDARD_19):
            raise RuntimeError(f"S1 {name} must be a standard-19 partition")


__all__ = [
    "S1_EVENT_TILES",
    "S1_HIERARCHY_WEIGHT",
    "S1_LABRAM_RECOVERY_SCHEMA",
    "S1_PROJECTOR_TRAINABLE_PARAMETERS",
    "S1SharedChannelProjector",
    "S1ObjectiveOutput",
    "aggregate_complete_patient_bags",
    "event_logits_from_prefix",
    "fold_channel_prior_logits",
    "hierarchy_set_mass_loss",
    "s1_patient_objective",
    "validate_protocol_partitions",
]
