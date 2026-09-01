"""Strict mask contracts for ictal evidence and offset-aware reasoning phases.

This module keeps two distinctions that are easy to lose in a dense evidence
pipeline:

* TUSZ annotation coverage is a *source supervision* mask.  It is never a
  deployment-availability mask and must not cross the evidence firewall.
* A numerically available 4-second tile is not necessarily valid ictal-phase
  evidence.  Tiles crossing a trustworthy seizure offset and tiles wholly
  after it are excluded from the primary phase mask.

The functions here are target- and model-free.  In particular, deployment
availability is derived only from physical signal availability and an
optional producer-output mask; ``source_target_mask`` is deliberately absent
from that computation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, Sequence

import torch

from .geometry import (
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    N_TIME_TILES,
    edge_endpoint_indices,
)


ICTAL_THREE_MASK_SCHEMA: Final[str] = "ictal_three_mask_contract_v1"
OFFSET_AWARE_PHASE_MASK_SCHEMA: Final[str] = "offset_aware_phase_mask_v1"
PRIMARY_WINDOW_START_SEC: Final[float] = -12.0
PRIMARY_WINDOW_STOP_SEC: Final[float] = 48.0
REASONING_TILE_SECONDS: Final[float] = 4.0
OFFSET_TIME_TOLERANCE_SEC: Final[float] = 1e-6

REASONING_TILE_EDGES_SEC: Final[tuple[float, ...]] = tuple(
    PRIMARY_WINDOW_START_SEC + REASONING_TILE_SECONDS * index
    for index in range(N_TIME_TILES + 1)
)
PRE_TILE_INDICES: Final[tuple[int, ...]] = (0, 1, 2)
EARLY_TILE_INDICES: Final[tuple[int, ...]] = (3, 4, 5)
LATE_TILE_INDICES: Final[tuple[int, ...]] = tuple(range(6, N_TIME_TILES))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_OFFSET_POLICY_PAYLOAD: Final[dict[str, object]] = {
    "schema": OFFSET_AWARE_PHASE_MASK_SCHEMA,
    "window_sec": [PRIMARY_WINDOW_START_SEC, PRIMARY_WINDOW_STOP_SEC],
    "tile_edges_sec": REASONING_TILE_EDGES_SEC,
    "pre_tiles": PRE_TILE_INDICES,
    "early_tiles": EARLY_TILE_INDICES,
    "late_tiles": LATE_TILE_INDICES,
    "trusted_offset": {
        "wholly_before_or_ending_at_offset": "primary_ictal_phase_valid",
        "crosses_offset": "transition_excluded",
        "starts_at_or_after_offset": "postictal_excluded",
    },
    "untrusted_offset": {
        "early": "primary_operational_evidence_valid",
        "late": "primary_invalid_sensitivity_only",
    },
    "pre_anchor_context": {
        "name": "pre_anchor_context_not_assumed_interictal_baseline",
        "trusted_no_previous_or_gap_ge_12s": "primary_context_valid",
        "trusted_previous_gap_lt_12s": "all_pre_tiles_invalid",
        "untrusted_previous_timeline": "all_pre_tiles_invalid",
        "washout_30s_60s": "reported_sensitivity_only_not_hard_mask",
    },
    "tolerance_sec": OFFSET_TIME_TOLERANCE_SEC,
    "window_stop_is_not_seizure_offset": True,
}
OFFSET_AWARE_PHASE_POLICY_SHA256: Final[str] = _canonical_sha256(
    _OFFSET_POLICY_PAYLOAD
)


def _require_bool_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    return value


def physical_node_to_edge_mask(physical_signal_mask: torch.Tensor) -> torch.Tensor:
    """Map physical node availability to TCP20 edge availability.

    Both physical endpoints must be available.  This is a geometry operation,
    not endpoint label expansion: no source target or probability is read.
    """

    _require_bool_tensor(physical_signal_mask, name="physical_signal_mask")
    if physical_signal_mask.ndim != 3 or physical_signal_mask.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("physical_signal_mask must have shape [B,19,S]")
    endpoints = edge_endpoint_indices(device=physical_signal_mask.device)
    return (
        physical_signal_mask[:, endpoints[:, 0], :]
        & physical_signal_mask[:, endpoints[:, 1], :]
    )


@dataclass(frozen=True)
class IctalDeploymentMasks:
    """Reasoner-safe mask view with no TUSZ annotation coverage.

    ``physical_signal_mask`` is node-time signal validity ``[B,19,S]``.
    ``deployment_prediction_mask`` is edge-time producer availability
    ``[B,20,S]`` and must be a subset of physical endpoint availability.
    """

    deployment_prediction_mask: torch.Tensor
    physical_signal_mask: torch.Tensor
    schema_version: str = ICTAL_THREE_MASK_SCHEMA

    def __post_init__(self) -> None:
        _require_bool_tensor(
            self.deployment_prediction_mask,
            name="deployment_prediction_mask",
        )
        _require_bool_tensor(self.physical_signal_mask, name="physical_signal_mask")
        if self.deployment_prediction_mask.ndim != 3:
            raise ValueError("deployment_prediction_mask must have shape [B,20,S]")
        batch_size, n_edges, n_seconds = self.deployment_prediction_mask.shape
        if n_edges != N_TCP_EDGES:
            raise ValueError("deployment_prediction_mask must have shape [B,20,S]")
        if tuple(self.physical_signal_mask.shape) != (
            batch_size,
            N_STANDARD_CHANNELS,
            n_seconds,
        ):
            raise ValueError("physical_signal_mask must have shape [B,19,S]")
        if self.deployment_prediction_mask.device != self.physical_signal_mask.device:
            raise ValueError("Ictal deployment masks must share one device")
        if self.schema_version != ICTAL_THREE_MASK_SCHEMA:
            raise ValueError("Unsupported ictal three-mask schema")
        physical_edges = physical_node_to_edge_mask(self.physical_signal_mask)
        if (self.deployment_prediction_mask & ~physical_edges).any():
            raise ValueError(
                "deployment_prediction_mask cannot mark an edge whose physical "
                "endpoint signal is unavailable"
            )

    @property
    def physical_edge_mask(self) -> torch.Tensor:
        return physical_node_to_edge_mask(self.physical_signal_mask)

    def tile_masks(
        self,
        *,
        seconds_per_tile: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return strict all-seconds physical-node and deployment-edge tiles."""

        if (
            isinstance(seconds_per_tile, bool)
            or not isinstance(seconds_per_tile, int)
            or seconds_per_tile < 1
        ):
            raise ValueError("seconds_per_tile must be a positive integer")
        n_seconds = self.physical_signal_mask.shape[-1]
        if n_seconds % seconds_per_tile:
            raise ValueError("Ictal masks must form complete fixed-size tiles")
        n_tiles = n_seconds // seconds_per_tile
        physical_tiles = self.physical_signal_mask.reshape(
            self.physical_signal_mask.shape[0],
            N_STANDARD_CHANNELS,
            n_tiles,
            seconds_per_tile,
        ).all(dim=-1)
        deployment_tiles = self.deployment_prediction_mask.reshape(
            self.deployment_prediction_mask.shape[0],
            N_TCP_EDGES,
            n_tiles,
            seconds_per_tile,
        ).all(dim=-1)
        return physical_tiles, deployment_tiles


@dataclass(frozen=True)
class IctalMaskBundle:
    """The three distinct masks at the supervised ictal-concept boundary."""

    source_target_mask: torch.Tensor
    deployment_prediction_mask: torch.Tensor
    physical_signal_mask: torch.Tensor
    schema_version: str = ICTAL_THREE_MASK_SCHEMA

    def __post_init__(self) -> None:
        _require_bool_tensor(self.source_target_mask, name="source_target_mask")
        deployment = IctalDeploymentMasks(
            deployment_prediction_mask=self.deployment_prediction_mask,
            physical_signal_mask=self.physical_signal_mask,
            schema_version=self.schema_version,
        )
        if tuple(self.source_target_mask.shape) != tuple(
            self.deployment_prediction_mask.shape
        ):
            raise ValueError("source_target_mask must have shape [B,20,S]")
        if self.source_target_mask.device != self.deployment_prediction_mask.device:
            raise ValueError("All three ictal masks must share one device")
        if (self.source_target_mask & ~deployment.physical_edge_mask).any():
            raise ValueError(
                "Source supervision cannot be used where the physical edge signal "
                "is unavailable"
            )

    def reasoner_view(self) -> IctalDeploymentMasks:
        """Drop source annotation coverage at the evidence firewall."""

        return IctalDeploymentMasks(
            deployment_prediction_mask=self.deployment_prediction_mask,
            physical_signal_mask=self.physical_signal_mask,
            schema_version=self.schema_version,
        )


def build_ictal_mask_bundle(
    source_target_mask: torch.Tensor,
    physical_signal_mask: torch.Tensor,
    *,
    producer_prediction_mask: torch.Tensor | None = None,
) -> IctalMaskBundle:
    """Build three masks without any annotation-to-deployment information path.

    The deployment mask is the conjunction of physical endpoint availability
    and the optional producer-output availability mask.  Crucially,
    ``source_target_mask`` is validated and retained for concept loss/native
    evaluation only; it is never read while constructing deployment
    availability.
    """

    _require_bool_tensor(source_target_mask, name="source_target_mask")
    _require_bool_tensor(physical_signal_mask, name="physical_signal_mask")
    if source_target_mask.ndim != 3 or source_target_mask.shape[1] != N_TCP_EDGES:
        raise ValueError("source_target_mask must have shape [B,20,S]")
    batch_size, _, n_seconds = source_target_mask.shape
    if tuple(physical_signal_mask.shape) != (
        batch_size,
        N_STANDARD_CHANNELS,
        n_seconds,
    ):
        raise ValueError("physical_signal_mask must have shape [B,19,S]")
    if source_target_mask.device != physical_signal_mask.device:
        raise ValueError("Source and physical masks must share one device")
    physical_edges = physical_node_to_edge_mask(physical_signal_mask)
    if producer_prediction_mask is None:
        producer = torch.ones_like(physical_edges)
    else:
        producer = _require_bool_tensor(
            producer_prediction_mask,
            name="producer_prediction_mask",
            shape=tuple(physical_edges.shape),
        )
        if producer.device != physical_edges.device:
            raise ValueError("Producer and physical masks must share one device")
    deployment = physical_edges & producer
    return IctalMaskBundle(
        source_target_mask=source_target_mask,
        deployment_prediction_mask=deployment,
        physical_signal_mask=physical_signal_mask,
    )


@dataclass(frozen=True)
class OffsetAwarePhaseMasks:
    """Auditable phase states for one batch of fixed 60-second events.

    ``ictal_phase_mask`` is the mask intersected with the reasoner's relative
    pre/early/late partitions.  The pre-onset tiles are merely anchor context,
    not assumed interictal baseline: all three are invalid when a trustworthy
    previous seizure overlaps ``[-12,0)`` or when the previous timeline is
    untrustworthy.  For a trustworthy current offset, post-onset validity
    includes only tiles wholly inside ``[t0, offset)``.  For an untrustworthy
    current offset, the operational early window remains valid but all late
    tiles are excluded from the primary analysis.
    """

    ictal_phase_mask: torch.Tensor
    pre_anchor_context_mask: torch.Tensor
    pre_previous_seizure_overlap_mask: torch.Tensor
    pre_unknown_context_mask: torch.Tensor
    within_trusted_ictal_mask: torch.Tensor
    transition_mask: torch.Tensor
    postictal_mask: torch.Tensor
    unknown_offset_mask: torch.Tensor
    offset_trustworthy: torch.Tensor
    seizure_duration_sec: torch.Tensor
    previous_timeline_trustworthy: torch.Tensor
    has_previous_seizure: torch.Tensor
    previous_seizure_overlap: torch.Tensor
    previous_seizure_gap_sec: torch.Tensor
    policy_sha256: str = OFFSET_AWARE_PHASE_POLICY_SHA256
    schema_version: str = OFFSET_AWARE_PHASE_MASK_SCHEMA

    def __post_init__(self) -> None:
        masks = (
            ("ictal_phase_mask", self.ictal_phase_mask),
            ("pre_anchor_context_mask", self.pre_anchor_context_mask),
            (
                "pre_previous_seizure_overlap_mask",
                self.pre_previous_seizure_overlap_mask,
            ),
            ("pre_unknown_context_mask", self.pre_unknown_context_mask),
            ("within_trusted_ictal_mask", self.within_trusted_ictal_mask),
            ("transition_mask", self.transition_mask),
            ("postictal_mask", self.postictal_mask),
            ("unknown_offset_mask", self.unknown_offset_mask),
        )
        if self.ictal_phase_mask.ndim != 2:
            raise ValueError("ictal_phase_mask must have shape [B,15]")
        batch_size = self.ictal_phase_mask.shape[0]
        expected = (batch_size, N_TIME_TILES)
        for name, mask in masks:
            _require_bool_tensor(mask, name=name, shape=expected)
        _require_bool_tensor(
            self.offset_trustworthy,
            name="offset_trustworthy",
            shape=(batch_size,),
        )
        for name, value in (
            ("previous_timeline_trustworthy", self.previous_timeline_trustworthy),
            ("has_previous_seizure", self.has_previous_seizure),
            ("previous_seizure_overlap", self.previous_seizure_overlap),
        ):
            _require_bool_tensor(value, name=name, shape=(batch_size,))
        if not isinstance(self.seizure_duration_sec, torch.Tensor):
            raise TypeError("seizure_duration_sec must be a torch.Tensor")
        if tuple(self.seizure_duration_sec.shape) != (batch_size,):
            raise ValueError("seizure_duration_sec must have shape [B]")
        if self.seizure_duration_sec.dtype != torch.float64:
            raise TypeError("seizure_duration_sec must use torch.float64")
        if self.seizure_duration_sec.requires_grad:
            raise ValueError("seizure_duration_sec must be detached")
        if not isinstance(self.previous_seizure_gap_sec, torch.Tensor):
            raise TypeError("previous_seizure_gap_sec must be a torch.Tensor")
        if tuple(self.previous_seizure_gap_sec.shape) != (batch_size,):
            raise ValueError("previous_seizure_gap_sec must have shape [B]")
        if self.previous_seizure_gap_sec.dtype != torch.float64:
            raise TypeError("previous_seizure_gap_sec must use torch.float64")
        if self.previous_seizure_gap_sec.requires_grad:
            raise ValueError("previous_seizure_gap_sec must be detached")
        devices = {
            mask.device for _, mask in masks
        } | {
            self.offset_trustworthy.device,
            self.seizure_duration_sec.device,
            self.previous_timeline_trustworthy.device,
            self.has_previous_seizure.device,
            self.previous_seizure_overlap.device,
            self.previous_seizure_gap_sec.device,
        }
        if len(devices) != 1:
            raise ValueError("Offset-aware phase tensors must share one device")
        if not torch.isfinite(self.seizure_duration_sec).all():
            raise ValueError("seizure_duration_sec must use finite zero fill")
        if not torch.isfinite(self.previous_seizure_gap_sec).all():
            raise ValueError("previous_seizure_gap_sec must use finite zero fill")
        if torch.any(
            self.has_previous_seizure & (self.previous_seizure_gap_sec < 0)
        ):
            raise ValueError("Previous seizure gap cannot be negative")
        if (self.previous_seizure_overlap & ~self.has_previous_seizure).any():
            raise ValueError("Previous overlap requires a previous seizure")
        if (
            self.previous_seizure_overlap
            & ~self.previous_timeline_trustworthy
        ).any():
            raise ValueError("Previous overlap requires a trustworthy timeline")
        if torch.any(self.seizure_duration_sec < 0):
            raise ValueError("seizure_duration_sec cannot be negative")
        if torch.any(
            self.offset_trustworthy & (self.seizure_duration_sec <= 0)
        ):
            raise ValueError("Trustworthy seizure offsets require positive duration")
        if self.policy_sha256 != OFFSET_AWARE_PHASE_POLICY_SHA256:
            raise ValueError("Offset-aware phase policy SHA mismatch")
        if self.schema_version != OFFSET_AWARE_PHASE_MASK_SCHEMA:
            raise ValueError("Unsupported offset-aware phase-mask schema")

        state_masks = (
            self.pre_anchor_context_mask,
            self.pre_previous_seizure_overlap_mask,
            self.pre_unknown_context_mask,
            self.within_trusted_ictal_mask,
            self.transition_mask,
            self.postictal_mask,
            self.unknown_offset_mask,
        )
        state_count = sum(mask.to(torch.int8) for mask in state_masks)
        if not torch.equal(state_count, torch.ones_like(state_count)):
            raise ValueError("Every tile must have exactly one offset-aware phase state")
        early_grid = torch.zeros_like(self.ictal_phase_mask)
        early_grid[:, EARLY_TILE_INDICES] = True
        expected_primary = (
            self.pre_anchor_context_mask
            | self.within_trusted_ictal_mask
            | (self.unknown_offset_mask & early_grid)
        )
        if not torch.equal(self.ictal_phase_mask, expected_primary):
            raise ValueError("ictal_phase_mask disagrees with the frozen primary policy")
        if self.transition_mask.any(dim=1).logical_and(
            ~self.offset_trustworthy
        ).any():
            raise ValueError("Untrusted offsets cannot create a transition tile")
        if (self.postictal_mask.any(dim=1) & ~self.offset_trustworthy).any():
            raise ValueError("Untrusted offsets cannot create a postictal tile")
        expected_overlap = (
            self.has_previous_seizure
            & self.previous_timeline_trustworthy
            & (
                self.previous_seizure_gap_sec
                < abs(PRIMARY_WINDOW_START_SEC) - OFFSET_TIME_TOLERANCE_SEC
            )
        )
        if not torch.equal(self.previous_seizure_overlap, expected_overlap):
            raise ValueError("Previous-seizure overlap state disagrees with its gap")
        pre_grid = torch.zeros_like(self.ictal_phase_mask)
        pre_grid[:, PRE_TILE_INDICES] = True
        if not torch.equal(
            self.pre_previous_seizure_overlap_mask,
            pre_grid & self.previous_seizure_overlap[:, None],
        ):
            raise ValueError("Previous-overlap pre mask disagrees with event state")
        if not torch.equal(
            self.pre_unknown_context_mask,
            pre_grid & ~self.previous_timeline_trustworthy[:, None],
        ):
            raise ValueError("Unknown-context pre mask disagrees with timeline trust")

    @property
    def late_primary_mask(self) -> torch.Tensor:
        late = torch.zeros_like(self.ictal_phase_mask)
        late[:, LATE_TILE_INDICES] = True
        return self.ictal_phase_mask & late

    @property
    def late_unmasked_sensitivity_mask(self) -> torch.Tensor:
        """Prespecified numerical sensitivity, never primary phase evidence."""

        late = torch.zeros_like(self.ictal_phase_mask)
        late[:, LATE_TILE_INDICES] = True
        return late


def build_offset_aware_phase_masks(
    seizure_duration_sec: Sequence[float | None],
    *,
    offset_trustworthy: Sequence[bool],
    previous_seizure_gap_sec: Sequence[float | None],
    previous_timeline_trustworthy: Sequence[bool],
) -> OffsetAwarePhaseMasks:
    """Classify every reasoning tile relative to a seizure offset.

    Parameters
    ----------
    seizure_duration_sec:
        Global seizure stop minus global seizure start, in seconds.  This is
        *not* the 60-second crop's ``window_stop_sec``.  Use ``None`` when no
        usable offset exists.
    offset_trustworthy:
        One explicit decision per event.  A finite duration with ``False`` is
        treated as unreliable; it cannot create ictal-late, transition, or
        postictal semantics in the primary analysis.
    previous_seizure_gap_sec:
        Current global t0 minus the previous global seizure stop. ``None``
        means that an audited timeline contains no previous seizure. A gap
        below 12 seconds means the previous seizure overlaps the pre-anchor
        context; all three pre tiles are then invalid. Values below 30/60
        seconds are retained for sensitivity strata only and do not alter the
        primary mask beyond the exact 12-second overlap rule.
    previous_timeline_trustworthy:
        Explicit timeline trust per event. If false, the pre-anchor context is
        fail-closed to invalid because overlap status is unknown.
    """

    durations = tuple(seizure_duration_sec)
    trustworthy = tuple(offset_trustworthy)
    previous_gaps = tuple(previous_seizure_gap_sec)
    previous_trustworthy = tuple(previous_timeline_trustworthy)
    if not durations or not (
        len(durations)
        == len(trustworthy)
        == len(previous_gaps)
        == len(previous_trustworthy)
    ):
        raise ValueError("Current and previous seizure timing inputs must align")
    if any(not isinstance(value, bool) for value in trustworthy):
        raise TypeError("offset_trustworthy must contain booleans")
    if any(not isinstance(value, bool) for value in previous_trustworthy):
        raise TypeError("previous_timeline_trustworthy must contain booleans")

    normalized: list[float] = []
    for index, (value, trusted) in enumerate(zip(durations, trustworthy)):
        if value is None:
            if trusted:
                raise ValueError(
                    f"Event {index}: trustworthy offset requires a seizure duration"
                )
            normalized.append(0.0)
            continue
        if isinstance(value, bool):
            raise TypeError("seizure_duration_sec cannot contain booleans")
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Provided seizure durations must be finite and positive")
        normalized.append(duration)

    normalized_previous_gaps: list[float] = []
    has_previous: list[bool] = []
    for value in previous_gaps:
        if value is None:
            normalized_previous_gaps.append(0.0)
            has_previous.append(False)
            continue
        if isinstance(value, bool):
            raise TypeError("previous_seizure_gap_sec cannot contain booleans")
        gap = float(value)
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("Previous seizure gaps must be finite and non-negative")
        normalized_previous_gaps.append(gap)
        has_previous.append(True)

    batch_size = len(normalized)
    shape = (batch_size, N_TIME_TILES)
    pre_context = torch.zeros(shape, dtype=torch.bool)
    pre_overlap = torch.zeros(shape, dtype=torch.bool)
    pre_unknown = torch.zeros(shape, dtype=torch.bool)
    within = torch.zeros(shape, dtype=torch.bool)
    transition = torch.zeros(shape, dtype=torch.bool)
    postictal = torch.zeros(shape, dtype=torch.bool)
    unknown = torch.zeros(shape, dtype=torch.bool)

    previous_overlap = tuple(
        has_previous_event
        and timeline_trusted
        and gap
        < abs(PRIMARY_WINDOW_START_SEC) - OFFSET_TIME_TOLERANCE_SEC
        for has_previous_event, timeline_trusted, gap in zip(
            has_previous, previous_trustworthy, normalized_previous_gaps
        )
    )
    for event_index, (duration, trusted) in enumerate(
        zip(normalized, trustworthy)
    ):
        for tile_index, (start, stop) in enumerate(
            zip(REASONING_TILE_EDGES_SEC, REASONING_TILE_EDGES_SEC[1:])
        ):
            if stop <= 0.0 + OFFSET_TIME_TOLERANCE_SEC:
                if not previous_trustworthy[event_index]:
                    pre_unknown[event_index, tile_index] = True
                elif previous_overlap[event_index]:
                    pre_overlap[event_index, tile_index] = True
                else:
                    pre_context[event_index, tile_index] = True
                continue
            if not trusted:
                unknown[event_index, tile_index] = True
                continue
            if stop <= duration + OFFSET_TIME_TOLERANCE_SEC:
                within[event_index, tile_index] = True
            elif start >= duration - OFFSET_TIME_TOLERANCE_SEC:
                postictal[event_index, tile_index] = True
            else:
                transition[event_index, tile_index] = True

    early_grid = torch.zeros(shape, dtype=torch.bool)
    early_grid[:, EARLY_TILE_INDICES] = True
    primary = pre_context | within | (unknown & early_grid)
    return OffsetAwarePhaseMasks(
        ictal_phase_mask=primary,
        pre_anchor_context_mask=pre_context,
        pre_previous_seizure_overlap_mask=pre_overlap,
        pre_unknown_context_mask=pre_unknown,
        within_trusted_ictal_mask=within,
        transition_mask=transition,
        postictal_mask=postictal,
        unknown_offset_mask=unknown,
        offset_trustworthy=torch.tensor(trustworthy, dtype=torch.bool),
        seizure_duration_sec=torch.tensor(normalized, dtype=torch.float64),
        previous_timeline_trustworthy=torch.tensor(
            previous_trustworthy, dtype=torch.bool
        ),
        has_previous_seizure=torch.tensor(has_previous, dtype=torch.bool),
        previous_seizure_overlap=torch.tensor(
            previous_overlap, dtype=torch.bool
        ),
        previous_seizure_gap_sec=torch.tensor(
            normalized_previous_gaps, dtype=torch.float64
        ),
    )


__all__ = [
    "EARLY_TILE_INDICES",
    "ICTAL_THREE_MASK_SCHEMA",
    "IctalDeploymentMasks",
    "IctalMaskBundle",
    "LATE_TILE_INDICES",
    "OFFSET_AWARE_PHASE_MASK_SCHEMA",
    "OFFSET_AWARE_PHASE_POLICY_SHA256",
    "OffsetAwarePhaseMasks",
    "PRE_TILE_INDICES",
    "PRIMARY_WINDOW_START_SEC",
    "PRIMARY_WINDOW_STOP_SEC",
    "REASONING_TILE_EDGES_SEC",
    "REASONING_TILE_SECONDS",
    "build_ictal_mask_bundle",
    "build_offset_aware_phase_masks",
    "physical_node_to_edge_mask",
]
