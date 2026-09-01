"""Target-isolated shortcut controls for the frozen LaBraM-H recovery head.

These utilities deliberately operate on an already materialized source-train
``FrozenHPatientBatch``.  They never load another split and they never inspect
held-patient targets.  The controls answer a narrow question: does the v3
``H+V uniform`` result use event-specific H content, or can static physical
position/channel signatures and the fold-local prevalence prior explain it?
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from .frozen_h_recovery import FrozenHPatientBatch, FrozenHStandardization
from .geometry import N_STANDARD_CHANNELS, N_TIME_TILES
from .temporal_mil_recovery import jeffreys_channel_prior_logits


FROZEN_H_SHORTCUT_CONTROL_SCHEMA = "soz_labram_frozen_h_shortcut_controls_v3_1"
POSITION_PROTOTYPE_SEMANTICS = (
    "fold-local target-free mean H at each physical-channel x tile x second "
    "position; preserves static channel/position and corpus-average signatures "
    "but removes patient/event-specific H content"
)
EVENT_TIME_SHUFFLE_SEMANTICS = (
    "fold-contained cross-patient bijection of H events followed by a nonzero "
    "circular tile shift; preserves channel identity and the global H multiset "
    "but breaks H-to-patient, H-to-V, and H-to-time alignment"
)
Q_ONLY_SEMANTICS = (
    "fold-local Jeffreys channel-prevalence logits only; H and V contributions "
    "are exactly absent"
)
ZERO_H_V_SEMANTICS = (
    "matched H+V-uniform head with every H second-token replaced by the "
    "original training-fold H mean; standardized H is exactly zero while V "
    "and the shared fold-local channel prior remain available"
)


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
    digest.update(len(metadata).to_bytes(4, "little"))
    digest.update(metadata)
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenHPositionPrototype:
    """Training-fold H mean indexed by physical channel and absolute position."""

    tokens: torch.Tensor
    valid_event_count_by_tile: torch.Tensor
    training_event_count: int

    def __post_init__(self) -> None:
        if tuple(self.tokens.shape) != (N_STANDARD_CHANNELS, N_TIME_TILES, 4, 200):
            raise ValueError("position prototype must have shape [19,15,4,200]")
        if not self.tokens.is_floating_point() or not torch.isfinite(self.tokens).all():
            raise ValueError("position prototype tokens must be finite floating point")
        if self.tokens.requires_grad:
            raise ValueError("position prototype must be detached")
        if (
            self.valid_event_count_by_tile.dtype != torch.long
            or tuple(self.valid_event_count_by_tile.shape) != (N_TIME_TILES,)
        ):
            raise TypeError("prototype tile counts must be long [15]")
        if self.valid_event_count_by_tile.device != self.tokens.device:
            raise ValueError("prototype tokens and counts must share a device")
        if torch.any(self.valid_event_count_by_tile < 0):
            raise ValueError("prototype tile counts cannot be negative")
        if self.training_event_count < 1:
            raise ValueError("prototype requires at least one training event")

    def receipt(self) -> dict[str, object]:
        return {
            "semantics": POSITION_PROTOTYPE_SEMANTICS,
            "training_event_count": self.training_event_count,
            "valid_event_count_by_tile": [
                int(value) for value in self.valid_event_count_by_tile.cpu().tolist()
            ],
            "prototype_tensor_sha256": _tensor_sha256(
                "position_prototype", self.tokens
            ),
        }


def fit_fold_local_position_prototype(
    train: FrozenHPatientBatch,
) -> FrozenHPositionPrototype:
    """Fit a target-free channel x tile x second H prototype on one train fold.

    Only ``node_tokens`` and ``phase_mask`` are read.  A tile unsupported in the
    training fold falls back to that channel/second's mean over supported tiles;
    held data are never used to fill it.
    """

    tokens = train.node_tokens
    phase = train.base.evidence.phase_mask
    if tokens.shape[0] < 1 or not phase.any():
        raise ValueError("position prototype requires valid training-fold H")
    weights = phase.to(tokens.dtype).view(tokens.shape[0], 1, N_TIME_TILES, 1, 1)
    tile_count = phase.sum(dim=0)
    tile_sum = (tokens * weights).sum(dim=0)
    prototype = tile_sum / tile_count.clamp_min(1).to(tokens.dtype).view(
        1, N_TIME_TILES, 1, 1
    )

    # A train-fold-only fallback makes the transformation total without
    # borrowing a held event when a phase tile happens to be absent.
    supported_sum = tile_sum.sum(dim=1)
    supported_count = tile_count.sum().clamp_min(1).to(tokens.dtype)
    channel_second_fallback = supported_sum / supported_count
    unsupported = tile_count == 0
    if unsupported.any():
        prototype[:, unsupported] = channel_second_fallback.unsqueeze(1)

    return FrozenHPositionPrototype(
        tokens=prototype.detach().contiguous(),
        valid_event_count_by_tile=tile_count.detach().long().contiguous(),
        training_event_count=int(tokens.shape[0]),
    )


def replace_h_with_position_prototype(
    batch: FrozenHPatientBatch,
    prototype: FrozenHPositionPrototype,
) -> FrozenHPatientBatch:
    """Replace every event H by a prototype fitted outside this function."""

    if prototype.tokens.device != batch.node_tokens.device:
        raise ValueError("prototype and destination batch must share a device")
    expanded = prototype.tokens.unsqueeze(0).expand(batch.node_tokens.shape[0], -1, -1, -1, -1)
    return FrozenHPatientBatch(base=batch.base, node_tokens=expanded.detach())


def replace_h_with_standardization_mean(
    batch: FrozenHPatientBatch,
    standardization: FrozenHStandardization,
) -> FrozenHPatientBatch:
    """Make the H route exactly zero under a fixed original-fold transform."""

    if standardization.mean.device != batch.node_tokens.device:
        raise ValueError("H standardization and destination batch must share a device")
    tokens = standardization.mean.view(1, 1, 1, 1, 200).expand(
        batch.node_tokens.shape[0], N_STANDARD_CHANNELS, N_TIME_TILES, 4, -1
    )
    return FrozenHPatientBatch(base=batch.base, node_tokens=tokens.detach())


@dataclass(frozen=True)
class FrozenHEventTimeShufflePlan:
    """A reproducible, fold-contained H source assignment and tile rotation."""

    event_source_index: torch.Tensor
    time_shift_by_event: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        if self.event_source_index.dtype != torch.long or self.event_source_index.ndim != 1:
            raise TypeError("event_source_index must be long [E]")
        if (
            self.time_shift_by_event.dtype != torch.long
            or self.time_shift_by_event.shape != self.event_source_index.shape
        ):
            raise TypeError("time_shift_by_event must be long [E]")
        if self.event_source_index.device.type != "cpu" or self.time_shift_by_event.device.type != "cpu":
            raise ValueError("shuffle plans must remain on CPU")
        events = int(self.event_source_index.numel())
        if events < 2 or not torch.equal(
            torch.sort(self.event_source_index).values, torch.arange(events)
        ):
            raise ValueError("event_source_index must be a permutation")
        if torch.any((self.time_shift_by_event <= 0) | (self.time_shift_by_event >= N_TIME_TILES)):
            raise ValueError("every temporal shift must be nonzero and below 15")

    def receipt(self, event_patient_index: torch.Tensor) -> dict[str, object]:
        owner = event_patient_index.detach().cpu().long()
        if owner.shape != self.event_source_index.shape:
            raise ValueError("shuffle receipt owner vector has the wrong shape")
        same_patient = owner == owner.index_select(0, self.event_source_index)
        fixed = self.event_source_index == torch.arange(owner.numel())
        return {
            "semantics": EVENT_TIME_SHUFFLE_SEMANTICS,
            "seed": self.seed,
            "event_count": int(owner.numel()),
            "event_source_index_sha256": _tensor_sha256(
                "event_source_index", self.event_source_index
            ),
            "time_shift_by_event_sha256": _tensor_sha256(
                "time_shift_by_event", self.time_shift_by_event
            ),
            "event_fixed_point_count": int(fixed.sum().item()),
            "same_patient_assignment_count": int(same_patient.sum().item()),
            "nonzero_time_shift_count": int((self.time_shift_by_event != 0).sum().item()),
            "bijection": True,
        }


def cross_patient_event_bijection_feasibility(
    event_patient_index: torch.Tensor,
) -> dict[str, object]:
    """Check the ownership-only necessary/sufficient condition for C2."""

    owner = event_patient_index.detach().cpu().long()
    if owner.ndim != 1 or owner.numel() < 1:
        raise ValueError("event_patient_index must be a non-empty vector")
    unique, counts = torch.unique(owner, sorted=True, return_counts=True)
    events = int(owner.numel())
    maximum = int(counts.max().item())
    enough_patients = unique.numel() >= 2
    no_majority_owner = maximum * 2 <= events
    feasible = bool(enough_patients and no_majority_owner)
    if not enough_patients:
        reason = "fewer_than_two_patients"
    elif not no_majority_owner:
        reason = "one_patient_owns_more_than_half_of_events"
    else:
        reason = "feasible"
    return {
        "feasible": feasible,
        "reason": reason,
        "patient_count": int(unique.numel()),
        "event_count": events,
        "maximum_patient_event_count": maximum,
    }


def make_cross_patient_event_time_shuffle_plan(
    batch: FrozenHPatientBatch,
    *,
    seed: int,
) -> FrozenHEventTimeShufflePlan:
    """Create a bijection in which every H event comes from another patient.

    Events are grouped by owner in a seeded random group order and then the
    concatenated roster is rotated by the largest patient event count.  This is
    a guaranteed cross-owner derangement whenever no patient owns more than
    half of the fold's events, the necessary feasibility condition.
    """

    owner = batch.base.event_patient_index.detach().cpu().long()
    events = int(owner.numel())
    unique, counts = torch.unique(owner, sorted=True, return_counts=True)
    feasibility = cross_patient_event_bijection_feasibility(owner)
    if not feasibility["feasible"]:
        raise ValueError(
            "cross-patient event bijection is infeasible: "
            f"{feasibility['reason']}"
        )
    maximum = int(counts.max().item())

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    patient_order = unique.index_select(
        0, torch.randperm(unique.numel(), generator=generator)
    )
    groups = []
    for patient in patient_order.tolist():
        indices = torch.nonzero(owner == int(patient), as_tuple=False).flatten()
        indices = indices.index_select(
            0, torch.randperm(indices.numel(), generator=generator)
        )
        groups.append(indices)
    ordered = torch.cat(groups)
    rotated = torch.roll(ordered, shifts=-maximum)
    source = torch.empty(events, dtype=torch.long)
    source[ordered] = rotated
    if torch.any(owner == owner.index_select(0, source)):
        raise RuntimeError("constructed event shuffle did not cross patients")

    time_shift = torch.randint(
        1,
        N_TIME_TILES,
        (events,),
        generator=generator,
        dtype=torch.long,
    )
    return FrozenHEventTimeShufflePlan(
        event_source_index=source,
        time_shift_by_event=time_shift,
        seed=int(seed),
    )


def apply_event_time_shuffle(
    batch: FrozenHPatientBatch,
    plan: FrozenHEventTimeShufflePlan,
) -> FrozenHPatientBatch:
    """Apply a plan without altering V, masks, patient bags, or targets."""

    events = batch.node_tokens.shape[0]
    if plan.event_source_index.numel() != events:
        raise ValueError("shuffle plan and batch event counts differ")
    source_index = plan.event_source_index.to(batch.node_tokens.device)
    shifted = batch.node_tokens.index_select(0, source_index)
    output = torch.empty_like(shifted)
    shifts = plan.time_shift_by_event.to(batch.node_tokens.device)
    for shift in range(1, N_TIME_TILES):
        selected = torch.nonzero(shifts == shift, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        rows = shifted.index_select(0, selected)
        output.index_copy_(0, selected, torch.roll(rows, shifts=-shift, dims=2))
    return FrozenHPatientBatch(base=batch.base, node_tokens=output.detach())


def zero_h_q_only_patient_outputs(
    train: FrozenHPatientBatch,
    *,
    held_patient_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q-only patient scores using train-fold targets and no held labels."""

    if held_patient_count < 1:
        raise ValueError("held_patient_count must be positive")
    prior = jeffreys_channel_prior_logits(train.base).detach().cpu()
    scores = prior.unsqueeze(0).expand(held_patient_count, -1).clone()
    probabilities = torch.softmax(prior, dim=0).unsqueeze(0).expand(
        held_patient_count, -1
    ).clone()
    return scores, probabilities, prior


__all__ = [
    "EVENT_TIME_SHUFFLE_SEMANTICS",
    "FROZEN_H_SHORTCUT_CONTROL_SCHEMA",
    "FrozenHEventTimeShufflePlan",
    "FrozenHPositionPrototype",
    "POSITION_PROTOTYPE_SEMANTICS",
    "Q_ONLY_SEMANTICS",
    "ZERO_H_V_SEMANTICS",
    "apply_event_time_shuffle",
    "cross_patient_event_bijection_feasibility",
    "fit_fold_local_position_prototype",
    "make_cross_patient_event_time_shuffle_plan",
    "replace_h_with_position_prototype",
    "replace_h_with_standardization_mean",
    "zero_h_q_only_patient_outputs",
]
