"""Minimal patient-balanced optimization for the first TUSZ concept stage."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterator, Sequence

import torch

from .concept_losses import ictal_involvement_loss
from .concept_metrics import IctalConceptMetrics, patient_macro_ictal_metrics
from .geometry import N_STANDARD_CHANNELS, N_TCP_EDGES
from .models.concept_heads import IctalInvolvementHead
from .models.foundation import TiledFoundationEncoder


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class IctalPatientBag:
    """All frozen-manifest ictal-training events for exactly one patient.

    ``target_mask`` is retained as an API field for compatibility, but its
    only semantics are TUSZ ``source_target_mask``.  It may enter the Stage-3I
    loss/native metrics and has no route to the SOZ reasoner.
    """

    patient_id: str
    event_ids: tuple[str, ...]
    expected_event_ids: tuple[str, ...]
    source_manifest_sha256: str
    eeg_volts: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor

    def __post_init__(self) -> None:
        patient_id = str(self.patient_id).strip()
        if not patient_id:
            raise ValueError("Ictal patient_id cannot be empty")
        object.__setattr__(self, "patient_id", patient_id)
        if not _SHA256_PATTERN.fullmatch(str(self.source_manifest_sha256)):
            raise ValueError("source_manifest_sha256 must be a lowercase SHA256")
        if not self.event_ids or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("Ictal patient bag requires unique non-empty event IDs")
        if set(self.event_ids) != set(self.expected_event_ids) or len(
            self.event_ids
        ) != len(self.expected_event_ids):
            raise ValueError("Ictal patient bag is incomplete relative to its manifest")
        n_events = len(self.event_ids)
        if tuple(self.eeg_volts.shape) != (n_events, N_STANDARD_CHANNELS, 12_000):
            raise ValueError("eeg_volts must have shape [E,19,12000]")
        expected_target = (n_events, N_TCP_EDGES, 60)
        if tuple(self.targets.shape) != expected_target or tuple(
            self.target_mask.shape
        ) != expected_target:
            raise ValueError("Ictal targets/mask must have shape [E,20,60]")
        if not self.eeg_volts.is_floating_point() or not self.targets.is_floating_point():
            raise TypeError("Ictal EEG and targets must be floating-point tensors")
        if self.eeg_volts.dtype != torch.float32 or self.targets.dtype != torch.float32:
            raise TypeError("Ictal EEG and targets must use float32")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("Ictal target_mask must be torch.bool")
        if not torch.isfinite(self.eeg_volts).all():
            raise ValueError("Ictal EEG must be finite")
        observed = self.targets[self.target_mask]
        if not torch.isfinite(observed).all() or (
            observed.numel() and not torch.all((observed == 0) | (observed == 1))
        ):
            raise ValueError("Observed ictal targets must be finite binary values")
        if not self.target_mask.any():
            raise ValueError("Ictal patient bag contains no observed edge-time labels")
        devices = {
            self.eeg_volts.device,
            self.targets.device,
            self.target_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("Ictal patient-bag tensors must share a device")

    @property
    def source_target_mask(self) -> torch.Tensor:
        return self.target_mask


@dataclass(frozen=True)
class IctalStepOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    observed_labels: int


@dataclass(frozen=True)
class IctalEpochOutput:
    mean_patient_loss: float
    n_patients: int
    n_events: int
    n_observed_labels: int


DEFAULT_EVENT_MICROBATCH_SIZE = 4


def _event_slices(
    n_events: int,
    event_microbatch_size: int | None,
) -> Iterator[slice]:
    if isinstance(n_events, bool) or not isinstance(n_events, int) or n_events < 1:
        raise ValueError("n_events must be a positive integer")
    if event_microbatch_size is None:
        size = n_events
    else:
        if (
            isinstance(event_microbatch_size, bool)
            or not isinstance(event_microbatch_size, int)
            or event_microbatch_size < 1
        ):
            raise ValueError("event_microbatch_size must be a positive integer or None")
        size = event_microbatch_size
    for start in range(0, n_events, size):
        yield slice(start, min(start + size, n_events))


def _forward_event_slice(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    bag: IctalPatientBag,
    event_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    head_devices = {parameter.device for parameter in head.parameters()}
    encoder_devices = {parameter.device for parameter in encoder.parameters()}
    if len(head_devices) != 1 or len(encoder_devices) != 1:
        raise ValueError("Ictal head and foundation must each occupy exactly one device")
    device = next(iter(head_devices))
    if next(iter(encoder_devices)) != device:
        raise ValueError("Ictal head and frozen foundation must share one device")
    eeg = bag.eeg_volts[event_slice].to(device=device, non_blocking=True)
    targets = bag.targets[event_slice].to(device=device, non_blocking=True)
    mask = bag.target_mask[event_slice].to(device=device, non_blocking=True)
    observed = int(mask.sum().item())
    tokens = encoder(eeg)
    if tokens.requires_grad:
        raise RuntimeError("Frozen foundation tokens unexpectedly require gradients")
    logits = head(tokens.detach())
    if observed < 1:
        return logits, logits.sum() * 0.0, 0
    patient_ids = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = ictal_involvement_loss(logits, targets, mask, patient_ids)
    return logits, loss, observed


def _validate_frozen_encoder(encoder: TiledFoundationEncoder) -> None:
    trainable = [name for name, value in encoder.named_parameters() if value.requires_grad]
    if trainable:
        raise ValueError(f"Foundation encoder must remain frozen, got {trainable[:5]}")


def _validate_head_optimizer(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    optimizer: torch.optim.Optimizer,
) -> None:
    _validate_frozen_encoder(encoder)
    head_parameters = {id(parameter) for parameter in head.parameters() if parameter.requires_grad}
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    foundation_parameters = {id(parameter) for parameter in encoder.parameters()}
    if optimizer_parameters & foundation_parameters:
        raise ValueError("Optimizer must not contain frozen foundation parameters")
    if optimizer_parameters != head_parameters:
        raise ValueError("Optimizer parameters must exactly equal trainable ictal-head parameters")


def ictal_training_step(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    bag: IctalPatientBag,
    *,
    event_microbatch_size: int | None = None,
) -> IctalStepOutput:
    """Forward one complete patient bag with no foundation gradient path.

    This convenience API returns every event logit and therefore retains all
    head graphs until its loss is consumed.  The epoch trainer below uses
    weighted microbatch backward passes instead, so large patient bags never
    need to coexist in LaBraM memory.
    """

    _validate_frozen_encoder(encoder)
    encoder.eval()
    chunks: list[torch.Tensor] = []
    weighted_losses: list[torch.Tensor] = []
    total_observed = int(bag.target_mask.sum().item())
    for event_slice in _event_slices(len(bag.event_ids), event_microbatch_size):
        logits, loss, observed = _forward_event_slice(encoder, head, bag, event_slice)
        chunks.append(logits)
        if observed:
            weighted_losses.append(loss * (observed / total_observed))
    return IctalStepOutput(
        logits=torch.cat(chunks, dim=0),
        loss=torch.stack(weighted_losses).sum(),
        observed_labels=total_observed,
    )


def train_ictal_epoch(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    patient_bags: Sequence[IctalPatientBag],
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float | None = 1.0,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> IctalEpochOutput:
    """Update only the ictal head once per complete, unique patient bag.

    Events are forwarded in bounded microbatches, but their gradients are
    weighted by their number of observed edge-time labels and accumulated
    before exactly one optimizer step.  This is algebraically the same masked
    patient mean as a monolithic bag, without its foundation-memory cost.
    """

    if not patient_bags:
        raise ValueError("Ictal epoch requires at least one patient bag")
    patient_ids = tuple(bag.patient_id for bag in patient_bags)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("An ictal epoch may contain each patient exactly once")
    manifest_hashes = {bag.source_manifest_sha256 for bag in patient_bags}
    if len(manifest_hashes) != 1:
        raise ValueError("One ictal epoch must use one frozen source manifest")
    if max_grad_norm is not None and (
        not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0
    ):
        raise ValueError("max_grad_norm must be positive or None")
    tuple(_event_slices(1, event_microbatch_size))
    _validate_head_optimizer(encoder, head, optimizer)

    encoder.eval()
    head.train()
    losses: list[float] = []
    n_events = 0
    n_observed = 0
    for bag in patient_bags:
        optimizer.zero_grad(set_to_none=True)
        patient_observed = int(bag.target_mask.sum().item())
        patient_loss = 0.0
        for event_slice in _event_slices(
            len(bag.event_ids), event_microbatch_size
        ):
            _, micro_loss, micro_observed = _forward_event_slice(
                encoder, head, bag, event_slice
            )
            if not micro_observed:
                continue
            weight = micro_observed / patient_observed
            (micro_loss * weight).backward()
            patient_loss += float(micro_loss.detach().cpu()) * weight
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(max_grad_norm))
        optimizer.step()
        losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_observed += patient_observed
    return IctalEpochOutput(
        mean_patient_loss=sum(losses) / len(losses),
        n_patients=len(patient_bags),
        n_events=n_events,
        n_observed_labels=n_observed,
    )


@torch.no_grad()
def evaluate_ictal_bags(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    patient_bags: Sequence[IctalPatientBag],
    *,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> IctalEpochOutput:
    """Evaluate patient-macro concept loss without mutating either model."""

    if not patient_bags:
        raise ValueError("Ictal evaluation requires at least one patient bag")
    patient_ids = tuple(bag.patient_id for bag in patient_bags)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Ictal evaluation may contain each patient exactly once")
    if len({bag.source_manifest_sha256 for bag in patient_bags}) != 1:
        raise ValueError("One ictal evaluation must use one frozen source manifest")
    tuple(_event_slices(1, event_microbatch_size))
    _validate_frozen_encoder(encoder)
    encoder.eval()
    head.eval()
    losses: list[float] = []
    n_events = 0
    n_observed = 0
    for bag in patient_bags:
        patient_observed = int(bag.target_mask.sum().item())
        patient_loss = 0.0
        for event_slice in _event_slices(
            len(bag.event_ids), event_microbatch_size
        ):
            _, micro_loss, micro_observed = _forward_event_slice(
                encoder, head, bag, event_slice
            )
            if not micro_observed:
                continue
            patient_loss += (
                float(micro_loss.detach().cpu()) * micro_observed / patient_observed
            )
        losses.append(patient_loss)
        n_events += len(bag.event_ids)
        n_observed += patient_observed
    return IctalEpochOutput(
        mean_patient_loss=sum(losses) / len(losses),
        n_patients=len(patient_bags),
        n_events=n_events,
        n_observed_labels=n_observed,
    )


@torch.no_grad()
def evaluate_ictal_bag_metrics(
    encoder: TiledFoundationEncoder,
    head: IctalInvolvementHead,
    patient_bags: Sequence[IctalPatientBag],
    *,
    event_microbatch_size: int | None = DEFAULT_EVENT_MICROBATCH_SIZE,
) -> IctalConceptMetrics:
    """Compute threshold-free patient-macro fidelity on one frozen cohort."""

    if not patient_bags:
        raise ValueError("Ictal metric evaluation requires at least one patient bag")
    patient_names = tuple(bag.patient_id for bag in patient_bags)
    if len(set(patient_names)) != len(patient_names):
        raise ValueError("Ictal metric evaluation may contain each patient exactly once")
    if len({bag.source_manifest_sha256 for bag in patient_bags}) != 1:
        raise ValueError("One ictal metric evaluation must use one frozen source manifest")
    tuple(_event_slices(1, event_microbatch_size))
    _validate_frozen_encoder(encoder)
    encoder.eval()
    head.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_patient_indices: list[torch.Tensor] = []
    for patient_index, bag in enumerate(patient_bags):
        for event_slice in _event_slices(
            len(bag.event_ids), event_microbatch_size
        ):
            logits, _, _ = _forward_event_slice(encoder, head, bag, event_slice)
            count = logits.shape[0]
            all_logits.append(logits.detach().cpu())
            all_targets.append(bag.targets[event_slice].detach().cpu())
            all_masks.append(bag.target_mask[event_slice].detach().cpu())
            all_patient_indices.append(
                torch.full((count,), patient_index, dtype=torch.long)
            )
    return patient_macro_ictal_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_masks, dim=0),
        torch.cat(all_patient_indices, dim=0),
    )


__all__ = [
    "DEFAULT_EVENT_MICROBATCH_SIZE",
    "IctalEpochOutput",
    "IctalPatientBag",
    "IctalStepOutput",
    "evaluate_ictal_bags",
    "evaluate_ictal_bag_metrics",
    "ictal_training_step",
    "train_ictal_epoch",
]
