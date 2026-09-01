"""Permission-split, censoring-aware segmental state model for BA-IEG.

This module is the trainable boundary/state primitive that intentionally sits
*before* clinical Findings and record-level scalp-onset reasoning.  It fixes a
specific limitation of the earlier retained-K marginalizer: an event is not
forced to expose a complete ``S0 -> S1 -> S2 -> S3`` path inside the queried
support.  Left-censored, right-censored, short and quality-gapped events remain
explicitly representable.

Two computational lanes are kept separate by construction:

* the onset lane reads only :meth:`BAIEGCollatedEventBatch.onset_causal_inputs`
  and produces the only time-local potentials permitted for ``S0 -> S1``;
* the offline lane may read complete signal context, but can produce only
  course, sustain, offset and review potentials.  The adaptive-search phase
  hint is zeroed in the main arm and is available only in an explicitly named
  heuristic-prior ablation.

The offline lane receives a detached copy of the causal hidden state.  This is
the sole cross-lane edge.  Consequently an offline/course loss can regularise
itself against an already available causal trace, but it cannot update or
create a positive onset trace.  The public forward method accepts no labels,
annotations, spreadsheets, clinical text or doctor channels.

The decoder operates on recording-relative physical-time cells.  Token row
order is a storage detail; rows are grouped and sorted by actual support end.
It computes an exact finite log partition and exact ``full_*``
forward--backward marginals over the registered segmental topology, plus a
retained top-K set for auditable inspection.  The legacy
``retained_state_marginals`` remain conditional on that retained set and must
not be confused with the full posterior.  None of these uncalibrated
quantities is a clinical probability.  This shadow research module does not
qualify seizure, cortical SOZ/EZ, or a report term and is not connected to
private or production reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Final, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .ba_ieg_training_contract import (
    BA_IEG_EVIDENCE_FAMILIES,
    BA_IEG_PHASE_STATES,
    BA_IEG_REFERENCE_FAMILIES,
    BA_IEG_TOKEN_SCALES,
    BAIEGCollatedEventBatch,
)
from .ba_ieg_segmental_dp_kernel_v1 import SegmentalPotentialsV1
from .ba_ieg_segmental_forward_backward_v1 import (
    ExactSegmentalForwardBackwardOutputV1,
    build_lognormal_segment_duration_log_scores_v1,
    run_exact_segmental_forward_backward_v1,
)


__all__ = [
    "BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID",
    "BA_IEG_SEGMENTAL_CONTEXT_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_TRANSITIONS",
    "BA_IEG_SEGMENTAL_TRANSITION_EDGES",
    "BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES",
    "BA_IEG_SEGMENTAL_CENSOR_REASONS",
    "BA_IEG_SEGMENTAL_TARGET_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_TARGET_STATUSES",
    "BA_IEG_SEGMENTAL_TARGET_AUTHORITIES",
    "BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID",
    "BA_IEG_CAUSAL_TYPED_UNIT_KINDS",
    "ba_ieg_event_identity_roster_sha256",
    "BAIEGSegmentalBoundaryContext",
    "BAIEGCausalTypedUnitTrace",
    "BAIEGPermissionSplitSegmentalStateOutput",
    "BAIEGPermissionSplitSegmentalStateModel",
    "build_ba_ieg_segmental_boundary_context",
    "segmental_log_partition_from_potentials",
]


BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID: Final[str] = (
    "ba_ieg_permission_split_censoring_aware_segmental_state_model_v1"
)
BA_IEG_SEGMENTAL_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_segmental_boundary_context_v1"
)
BA_IEG_SEGMENTAL_TARGET_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_segmental_boundary_targets_v1"
)
BA_IEG_SEGMENTAL_TRANSITIONS: Final[tuple[str, ...]] = (
    "S0_to_S1",
    "S1_to_S2",
    "S2_to_S3",
    "S0_to_S2_short_event",
    "S1_to_S3_short_return",
    "S3_to_S0_clean_return",
    "S3_to_S1_reentry",
    "S3_to_S2_reentry_short",
)
BA_IEG_SEGMENTAL_TRANSITION_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 2),
    (1, 3),
    (3, 0),
    (3, 1),
    (3, 2),
)
BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES: Final[frozenset[int]] = frozenset(
    {0, 3, 6, 7}
)
BA_IEG_SEGMENTAL_CENSOR_REASONS: Final[frozenset[str]] = frozenset(
    {
        "none",
        "recording_edge",
        "search_cap",
        "neighbor_event_protection",
        "quality_gap_at_boundary",
        "detector_navigation_uncertainty",
    }
)
BA_IEG_SEGMENTAL_TARGET_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "observed_interval",
        "left_censored",
        "right_censored",
        "not_observed",
        "not_evaluable",
    }
)
BA_IEG_SEGMENTAL_TARGET_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "synthetic_signal_injection",
        "public_seizure_interval",
        "source_development_eeg_expert_atomic_boundary",
    }
)
BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID: Final[str] = (
    "ba_ieg_segmental_causal_typed_unit_trace_v1"
)
BA_IEG_CAUSAL_TYPED_UNIT_KINDS: Final[tuple[str, ...]] = (
    "physical_electrode",
    "bipolar_lead",
)


def _sha256_text(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ba_ieg_event_identity_roster_sha256(
    *,
    source_input_batch_sha256: str,
    event_ids: Sequence[str],
    recording_ids: Sequence[str],
    source_event_receipt_sha256s: Sequence[str],
) -> str:
    """Bind the positional event identity roster to one registered batch."""

    return _canonical_sha256(
        {
            "schema": "ba_ieg_event_identity_roster_v1",
            "source_input_batch_sha256": str(source_input_batch_sha256),
            "event_ids": list(event_ids),
            "recording_ids": list(recording_ids),
            "source_event_receipt_sha256s": list(
                source_event_receipt_sha256s
            ),
        }
    )


def _training_contract_tensor_sha256(value: torch.Tensor) -> str:
    """Replay ``ba_ieg_training_contract`` tensor hashing without importing a private API."""

    tensor = value.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _collated_batch_input_sha256(batch: BAIEGCollatedEventBatch) -> str:
    """Replay the registered v3 model-input hash before neural execution."""

    tensor_names = (
        "token_values",
        "token_feature_mask",
        "token_row_mask",
        "token_signal_mask",
        "token_time_bounds_seconds",
        "token_unit_index",
        "token_view_index",
        "token_scale_index",
        "token_family_mask",
        "phase_posterior",
        "token_future_sample_access",
        "token_onset_evidence_mask",
        "token_positive_onset_mask",
        "token_phase_context_mask",
        "view_row_mask",
        "view_temporal_role_index",
        "view_dependency_policy_index",
        "view_reference_family_index",
        "view_future_sample_access",
        "view_onset_evidence_authorized",
        "unit_row_mask",
        "unit_view_index",
        "unit_reference_matrix",
        "unit_evidence_mask",
        "unit_family_mask",
        "physical_xyz",
        "physical_xyz_mask",
        "physical_evidence_mask",
    )
    payload = {
        "schema": "ba_ieg_collated_event_model_inputs_v3",
        "model_split": batch.model_split,
        "event_input_receipts": list(batch.input_event_receipt_sha256s),
        "view_temporal_evidence_sha256s": [
            list(row) for row in batch.view_temporal_evidence_sha256s
        ],
        "tensor_sha256": {
            name: _training_contract_tensor_sha256(getattr(batch, name))
            for name in tensor_names
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frozen_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype | None = None,
    floating: bool | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if floating is True and not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if floating is False and value.is_floating_point():
        raise TypeError(f"{name} must not be floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached from autograd")
    result = value.detach().clone().contiguous()
    if result.is_floating_point() and not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _intervals_are_canonical(
    intervals: torch.Tensor,
    mask: torch.Tensor,
    *,
    tolerance_seconds: float = 1e-8,
) -> bool:
    for batch_index in range(int(intervals.shape[0])):
        previous_stop: float | None = None
        seen_padding = False
        for row_index in range(int(intervals.shape[1])):
            active = bool(mask[batch_index, row_index])
            if not active:
                seen_padding = True
                if torch.any(intervals[batch_index, row_index] != 0):
                    return False
                continue
            if seen_padding:
                return False
            start = float(intervals[batch_index, row_index, 0])
            stop = float(intervals[batch_index, row_index, 1])
            if stop <= start:
                return False
            if previous_stop is not None and start < previous_stop - tolerance_seconds:
                return False
            previous_stop = stop
    return True


@dataclass(frozen=True)
class BAIEGSegmentalBoundaryContext:
    """EEG-only queried-support and censoring context for one collated batch.

    ``observed_support_intervals_seconds`` is the union of support that was
    actually queried and is potentially usable.  ``quality_gap`` intervals
    remain separate so gaps are not silently convex-hulled.  Censor reasons
    describe the acquisition/search boundary and are not seizure labels.
    """

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    source_event_receipt_sha256s: tuple[str, ...]
    adaptive_acquisition_receipt_sha256s: tuple[str, ...]
    observed_support_intervals_seconds: torch.Tensor
    observed_support_mask: torch.Tensor
    quality_gap_intervals_seconds: torch.Tensor
    quality_gap_mask: torch.Tensor
    left_censor_reason_codes: tuple[str, ...]
    right_censor_reason_codes: tuple[str, ...]
    source_authority: str = "eeg_signal_adaptive_search_only"
    schema_version: str = BA_IEG_SEGMENTAL_CONTEXT_SCHEMA_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256_text(self.source_input_batch_sha256, "source_input_batch_sha256")
        if self.schema_version != BA_IEG_SEGMENTAL_CONTEXT_SCHEMA_VERSION:
            raise ValueError("segmental boundary context schema drifted")
        if self.source_authority != "eeg_signal_adaptive_search_only":
            raise ValueError("segmental context authority is not EEG-only")
        support = _frozen_tensor(
            self.observed_support_intervals_seconds,
            name="observed_support_intervals_seconds",
            ndim=3,
            floating=True,
        )
        support_mask = _frozen_tensor(
            self.observed_support_mask,
            name="observed_support_mask",
            ndim=2,
            dtype=torch.bool,
        )
        gaps = _frozen_tensor(
            self.quality_gap_intervals_seconds,
            name="quality_gap_intervals_seconds",
            ndim=3,
            floating=True,
        )
        gap_mask = _frozen_tensor(
            self.quality_gap_mask,
            name="quality_gap_mask",
            ndim=2,
            dtype=torch.bool,
        )
        if support.shape[-1] != 2 or tuple(support.shape[:2]) != tuple(support_mask.shape):
            raise ValueError("observed support intervals and mask do not align")
        if gaps.shape[-1] != 2 or tuple(gaps.shape[:2]) != tuple(gap_mask.shape):
            raise ValueError("quality-gap intervals and mask do not align")
        batch_size = int(support.shape[0])
        if batch_size < 1 or int(support.shape[1]) < 1 or int(gaps.shape[1]) < 1:
            raise ValueError("segmental context requires padded non-empty tensor axes")
        metadata_lengths = (
            len(self.event_ids),
            len(self.source_event_receipt_sha256s),
            len(self.adaptive_acquisition_receipt_sha256s),
            len(self.left_censor_reason_codes),
            len(self.right_censor_reason_codes),
        )
        if any(length != batch_size for length in metadata_lengths):
            raise ValueError("segmental context event metadata does not align")
        if len(set(self.event_ids)) != len(self.event_ids) or any(
            not isinstance(event_id, str) or not event_id for event_id in self.event_ids
        ):
            raise ValueError("segmental context event IDs must be unique identifiers")
        if not support_mask.any(dim=1).all():
            raise ValueError("every event needs at least one observed support interval")
        if not _intervals_are_canonical(support, support_mask):
            raise ValueError("observed support intervals must be sorted, disjoint and padded")
        if not _intervals_are_canonical(gaps, gap_mask):
            raise ValueError("quality gaps must be sorted, disjoint and padded")
        for digest in self.source_event_receipt_sha256s:
            _sha256_text(digest, "source_event_receipt_sha256")
        for digest in self.adaptive_acquisition_receipt_sha256s:
            _sha256_text(digest, "adaptive_acquisition_receipt_sha256")
        for side, codes in (
            ("left", self.left_censor_reason_codes),
            ("right", self.right_censor_reason_codes),
        ):
            if any(code not in BA_IEG_SEGMENTAL_CENSOR_REASONS for code in codes):
                raise ValueError(f"unsupported {side} censor reason code")
        for batch_index in range(batch_size):
            local_support = support[batch_index, support_mask[batch_index]]
            local_gaps = gaps[batch_index, gap_mask[batch_index]]
            if local_gaps.numel():
                support_start = float(local_support[:, 0].min())
                support_stop = float(local_support[:, 1].max())
                if torch.any(local_gaps[:, 0] < support_start) or torch.any(
                    local_gaps[:, 1] > support_stop
                ):
                    raise ValueError("quality gaps must lie inside the support envelope")
        object.__setattr__(self, "observed_support_intervals_seconds", support)
        object.__setattr__(self, "observed_support_mask", support_mask)
        object.__setattr__(self, "quality_gap_intervals_seconds", gaps)
        object.__setattr__(self, "quality_gap_mask", gap_mask)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    @property
    def left_censoring_possible(self) -> torch.Tensor:
        return torch.tensor(
            [code != "none" for code in self.left_censor_reason_codes],
            dtype=torch.bool,
            device=self.observed_support_intervals_seconds.device,
        )

    @property
    def right_censoring_possible(self) -> torch.Tensor:
        return torch.tensor(
            [code != "none" for code in self.right_censor_reason_codes],
            dtype=torch.bool,
            device=self.observed_support_intervals_seconds.device,
        )

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "event_ids": list(self.event_ids),
                "source_event_receipt_sha256s": list(self.source_event_receipt_sha256s),
                "adaptive_acquisition_receipt_sha256s": list(
                    self.adaptive_acquisition_receipt_sha256s
                ),
                "source_authority": self.source_authority,
                "left_censor_reason_codes": list(self.left_censor_reason_codes),
                "right_censor_reason_codes": list(self.right_censor_reason_codes),
                "tensor_sha256": {
                    "observed_support_intervals_seconds": _tensor_sha256(
                        self.observed_support_intervals_seconds
                    ),
                    "observed_support_mask": _tensor_sha256(self.observed_support_mask),
                    "quality_gap_intervals_seconds": _tensor_sha256(
                        self.quality_gap_intervals_seconds
                    ),
                    "quality_gap_mask": _tensor_sha256(self.quality_gap_mask),
                },
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("segmental boundary context changed after registration")


def _merge_intervals(
    intervals: Iterable[tuple[float, float]], *, tolerance_seconds: float = 1e-8
) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(stop)) for start, stop in intervals)
    merged: list[list[float]] = []
    for start, stop in ordered:
        if stop <= start:
            continue
        if not merged or start > merged[-1][1] + tolerance_seconds:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def build_ba_ieg_segmental_boundary_context(
    batch: BAIEGCollatedEventBatch,
    *,
    adaptive_acquisition_receipt_sha256s: Sequence[str],
    quality_gap_intervals_by_event: Sequence[Sequence[tuple[float, float]]] | None = None,
    left_censor_reason_codes: Sequence[str] | None = None,
    right_censor_reason_codes: Sequence[str] | None = None,
) -> BAIEGSegmentalBoundaryContext:
    """Build a target-free context from the registered EEG batch.

    The default observed support is the gap-preserving union of active token
    supports.  Explicit gap intervals and censor reasons must come from the
    EEG-only adaptive-search/QC path; this function has no parameter for labels
    or clinical metadata.
    """

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("segmental context requires a registered collated batch")
    batch_size = len(batch.event_ids)
    if len(adaptive_acquisition_receipt_sha256s) != batch_size:
        raise ValueError("adaptive acquisition receipts must align with events")
    for digest in adaptive_acquisition_receipt_sha256s:
        _sha256_text(digest, "adaptive_acquisition_receipt_sha256")
    active = batch.token_row_mask & batch.token_signal_mask
    support_rows: list[list[tuple[float, float]]] = []
    for batch_index in range(batch_size):
        indices = torch.nonzero(active[batch_index], as_tuple=False).flatten()
        if not bool(indices.numel()):
            raise ValueError("cannot build segmental context without active EEG support")
        local = [
            (
                float(batch.token_time_bounds_seconds[batch_index, index, 0]),
                float(batch.token_time_bounds_seconds[batch_index, index, 1]),
            )
            for index in indices.detach().cpu()
        ]
        support_rows.append(_merge_intervals(local))
    maximum_support = max(len(rows) for rows in support_rows)
    support = torch.zeros(
        (batch_size, maximum_support, 2),
        dtype=batch.token_time_bounds_seconds.dtype,
        device=batch.token_time_bounds_seconds.device,
    )
    support_mask = torch.zeros(
        (batch_size, maximum_support),
        dtype=torch.bool,
        device=batch.token_time_bounds_seconds.device,
    )
    for batch_index, rows in enumerate(support_rows):
        support[batch_index, : len(rows)] = torch.tensor(
            rows, dtype=support.dtype, device=support.device
        )
        support_mask[batch_index, : len(rows)] = True

    if quality_gap_intervals_by_event is None:
        gap_rows: list[list[tuple[float, float]]] = [[] for _ in range(batch_size)]
    else:
        if len(quality_gap_intervals_by_event) != batch_size:
            raise ValueError("quality gap rows must align with events")
        gap_rows = [_merge_intervals(rows) for rows in quality_gap_intervals_by_event]
    maximum_gaps = max(1, *(len(rows) for rows in gap_rows))
    gaps = torch.zeros(
        (batch_size, maximum_gaps, 2), dtype=support.dtype, device=support.device
    )
    gap_mask = torch.zeros(
        (batch_size, maximum_gaps), dtype=torch.bool, device=support.device
    )
    for batch_index, rows in enumerate(gap_rows):
        if rows:
            gaps[batch_index, : len(rows)] = torch.tensor(
                rows, dtype=gaps.dtype, device=gaps.device
            )
            gap_mask[batch_index, : len(rows)] = True

    left_codes = tuple(left_censor_reason_codes or ("none",) * batch_size)
    right_codes = tuple(right_censor_reason_codes or ("none",) * batch_size)
    if len(left_codes) != batch_size or len(right_codes) != batch_size:
        raise ValueError("censor reason rows must align with events")
    return BAIEGSegmentalBoundaryContext(
        source_input_batch_sha256=batch.input_batch_sha256,
        event_ids=batch.event_ids,
        source_event_receipt_sha256s=batch.input_event_receipt_sha256s,
        adaptive_acquisition_receipt_sha256s=tuple(
            adaptive_acquisition_receipt_sha256s
        ),
        observed_support_intervals_seconds=support,
        observed_support_mask=support_mask,
        quality_gap_intervals_seconds=gaps,
        quality_gap_mask=gap_mask,
        left_censor_reason_codes=left_codes,
        right_censor_reason_codes=right_codes,
    )


@dataclass(frozen=True)
class _SegmentalPath:
    score: torch.Tensor
    start_state: int
    end_state: int
    transition_indices: tuple[int, ...]


def _select_top_k_paths(
    entries: Sequence[_SegmentalPath], maximum_paths: int
) -> list[_SegmentalPath]:
    if not entries:
        return []
    scores = torch.stack([entry.score for entry in entries])
    ordering = torch.argsort(scores, descending=True, stable=True)
    return [entries[int(index)] for index in ordering[:maximum_paths]]


def _lognormal_duration_score(
    duration_seconds: torch.Tensor,
    *,
    location: torch.Tensor,
    scale: torch.Tensor,
    censored: bool,
) -> torch.Tensor:
    duration = duration_seconds.clamp_min(torch.finfo(duration_seconds.dtype).eps)
    log_duration = torch.log(duration)
    z = (log_duration - location) / scale
    if censored:
        survival = 0.5 * torch.erfc(z / math.sqrt(2.0))
        return torch.log(survival.clamp_min(torch.finfo(duration.dtype).tiny))
    return (
        -0.5 * z.square()
        - torch.log(scale)
        - log_duration
        - 0.5 * math.log(2.0 * math.pi)
    )


def _segment_score(
    *,
    state: int,
    start_index: int,
    end_index: int,
    emission_prefix: torch.Tensor,
    duration_prefix: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    censored: bool,
) -> torch.Tensor | None:
    if end_index < start_index:
        return None
    emission = emission_prefix[end_index + 1, state] - emission_prefix[start_index, state]
    duration = duration_prefix[end_index + 1] - duration_prefix[start_index]
    if float(duration.detach().cpu()) + 1e-8 < float(
        minimum_duration_seconds[state].detach().cpu()
    ):
        return None
    return emission + _lognormal_duration_score(
        duration,
        location=duration_location[state],
        scale=duration_scale[state],
        censored=censored,
    )


def _allowed_start_end_pairs(
    *, left_censoring_possible: bool, right_censoring_possible: bool
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = [(0, 0)]
    starts = (0, 1, 2) if left_censoring_possible else (0,)
    for start in starts:
        minimum_end = max(start, 1)
        ends = range(minimum_end, 4) if right_censoring_possible else (3,)
        for end in ends:
            if start == 0 and end == 0:
                continue
            pairs.append((start, int(end)))
    return tuple(dict.fromkeys(pairs))


def _fixed_pair_log_partition(
    *,
    start_state: int,
    end_state: int,
    emission_prefix: torch.Tensor,
    duration_prefix: torch.Tensor,
    transition_log_scores: torch.Tensor,
    transition_mask: torch.Tensor,
    start_log_scores: torch.Tensor,
    end_log_scores: torch.Tensor,
    event_log_score: torch.Tensor,
    no_event_log_score: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
) -> torch.Tensor | None:
    grid_count = int(transition_log_scores.shape[0])
    if grid_count < 1:
        return None
    if start_state == 0 and end_state == 0:
        segment = _segment_score(
            state=0,
            start_index=0,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=False,
        )
        return None if segment is None else no_event_log_score + segment
    if end_state < start_state:
        return None
    left_censored = start_state > 0
    right_censored = end_state < len(BA_IEG_PHASE_STATES) - 1
    if start_state == end_state:
        segment = _segment_score(
            state=start_state,
            start_index=0,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=left_censored or right_censored,
        )
        if segment is None:
            return None
        return event_log_score + start_log_scores[start_state] + end_log_scores[end_state] + segment

    # Scores after a segment in ``state`` has ended and its transition to the
    # next state has been taken at the corresponding physical grid boundary.
    prefix: list[torch.Tensor | None] = [None] * grid_count
    first_transition = start_state
    for boundary_index in range(grid_count - 1):
        if not bool(transition_mask[boundary_index, first_transition]):
            continue
        segment = _segment_score(
            state=start_state,
            start_index=0,
            end_index=boundary_index,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=left_censored,
        )
        if segment is not None:
            prefix[boundary_index] = (
                start_log_scores[start_state]
                + segment
                + transition_log_scores[boundary_index, first_transition]
            )

    for state in range(start_state + 1, end_state):
        transition_index = state
        updated: list[torch.Tensor | None] = [None] * grid_count
        for boundary_index in range(grid_count - 1):
            if not bool(transition_mask[boundary_index, transition_index]):
                continue
            candidates: list[torch.Tensor] = []
            for previous_boundary in range(boundary_index):
                if prefix[previous_boundary] is None:
                    continue
                segment = _segment_score(
                    state=state,
                    start_index=previous_boundary + 1,
                    end_index=boundary_index,
                    emission_prefix=emission_prefix,
                    duration_prefix=duration_prefix,
                    duration_location=duration_location,
                    duration_scale=duration_scale,
                    minimum_duration_seconds=minimum_duration_seconds,
                    censored=False,
                )
                if segment is not None:
                    candidates.append(
                        prefix[previous_boundary]
                        + segment
                        + transition_log_scores[boundary_index, transition_index]
                    )
            if candidates:
                updated[boundary_index] = torch.logsumexp(torch.stack(candidates), dim=0)
        prefix = updated

    final_candidates: list[torch.Tensor] = []
    for previous_boundary in range(grid_count - 1):
        if prefix[previous_boundary] is None:
            continue
        segment = _segment_score(
            state=end_state,
            start_index=previous_boundary + 1,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=right_censored,
        )
        if segment is not None:
            final_candidates.append(prefix[previous_boundary] + segment)
    if not final_candidates:
        return None
    return (
        event_log_score
        + end_log_scores[end_state]
        + torch.logsumexp(torch.stack(final_candidates), dim=0)
    )


def _monotone_segmental_log_partition_from_potentials(
    *,
    state_emission_log_prob: torch.Tensor,
    opportunity_duration_seconds: torch.Tensor,
    transition_log_scores: torch.Tensor,
    transition_mask: torch.Tensor,
    start_log_scores: torch.Tensor,
    end_log_scores: torch.Tensor,
    event_log_score: torch.Tensor,
    no_event_log_score: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    left_censoring_possible: bool,
    right_censoring_possible: bool,
    allowed_start_end_pairs: Sequence[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Exact differentiable partition for one physical-time event lattice."""

    grid_count = int(state_emission_log_prob.shape[0])
    if tuple(state_emission_log_prob.shape) != (grid_count, len(BA_IEG_PHASE_STATES)):
        raise ValueError("state emission potentials have invalid shape")
    if tuple(opportunity_duration_seconds.shape) != (grid_count,):
        raise ValueError("opportunity durations do not align with the grid")
    if tuple(transition_log_scores.shape) != (
        grid_count,
        len(BA_IEG_SEGMENTAL_TRANSITIONS),
    ) or tuple(transition_mask.shape) != tuple(transition_log_scores.shape):
        raise ValueError("transition potentials do not align with the grid")
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must be boolean")
    weighted_emission = state_emission_log_prob * opportunity_duration_seconds.unsqueeze(-1)
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, len(BA_IEG_PHASE_STATES)),
                dtype=weighted_emission.dtype,
                device=weighted_emission.device,
            ),
            torch.cumsum(weighted_emission, dim=0),
        ),
        dim=0,
    )
    duration_prefix = torch.cat(
        (
            torch.zeros(
                1,
                dtype=opportunity_duration_seconds.dtype,
                device=opportunity_duration_seconds.device,
            ),
            torch.cumsum(opportunity_duration_seconds, dim=0),
        )
    )
    pairs = tuple(allowed_start_end_pairs) if allowed_start_end_pairs is not None else _allowed_start_end_pairs(
        left_censoring_possible=left_censoring_possible,
        right_censoring_possible=right_censoring_possible,
    )
    legal_pairs = set(
        _allowed_start_end_pairs(
            left_censoring_possible=left_censoring_possible,
            right_censoring_possible=right_censoring_possible,
        )
    )
    if not pairs or any(pair not in legal_pairs for pair in pairs):
        raise ValueError("requested start/end constraints are outside the legal lattice")
    scores: list[torch.Tensor] = []
    for start_state, end_state in pairs:
        score = _fixed_pair_log_partition(
            start_state=start_state,
            end_state=end_state,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            transition_log_scores=transition_log_scores,
            transition_mask=transition_mask,
            start_log_scores=start_log_scores,
            end_log_scores=end_log_scores,
            event_log_score=event_log_score,
            no_event_log_score=no_event_log_score,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
        )
        if score is not None:
            scores.append(score)
    if not scores:
        return torch.full(
            (),
            -torch.inf,
            dtype=state_emission_log_prob.dtype,
            device=state_emission_log_prob.device,
        )
    return torch.logsumexp(torch.stack(scores), dim=0)


def _top_k_fixed_pair_paths(
    *,
    start_state: int,
    end_state: int,
    emission_prefix: torch.Tensor,
    duration_prefix: torch.Tensor,
    transition_log_scores: torch.Tensor,
    transition_mask: torch.Tensor,
    start_log_scores: torch.Tensor,
    end_log_scores: torch.Tensor,
    event_log_score: torch.Tensor,
    no_event_log_score: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    maximum_paths: int,
) -> list[_SegmentalPath]:
    grid_count = int(transition_log_scores.shape[0])
    if start_state == 0 and end_state == 0:
        segment = _segment_score(
            state=0,
            start_index=0,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=False,
        )
        return [] if segment is None else [
            _SegmentalPath(no_event_log_score + segment, 0, 0, ())
        ]
    if end_state < start_state:
        return []
    left_censored = start_state > 0
    right_censored = end_state < len(BA_IEG_PHASE_STATES) - 1
    if start_state == end_state:
        segment = _segment_score(
            state=start_state,
            start_index=0,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=left_censored or right_censored,
        )
        return [] if segment is None else [
            _SegmentalPath(
                event_log_score
                + start_log_scores[start_state]
                + end_log_scores[end_state]
                + segment,
                start_state,
                end_state,
                (),
            )
        ]

    # Prefix entries carry transition boundary indices in physical order.
    prefix: list[list[_SegmentalPath]] = [[] for _ in range(grid_count)]
    first_transition = start_state
    for boundary_index in range(grid_count - 1):
        if not bool(transition_mask[boundary_index, first_transition]):
            continue
        segment = _segment_score(
            state=start_state,
            start_index=0,
            end_index=boundary_index,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=left_censored,
        )
        if segment is not None:
            prefix[boundary_index] = [
                _SegmentalPath(
                    start_log_scores[start_state]
                    + segment
                    + transition_log_scores[boundary_index, first_transition],
                    start_state,
                    end_state,
                    (boundary_index,),
                )
            ]

    for state in range(start_state + 1, end_state):
        transition_index = state
        updated: list[list[_SegmentalPath]] = [[] for _ in range(grid_count)]
        for boundary_index in range(grid_count - 1):
            if not bool(transition_mask[boundary_index, transition_index]):
                continue
            candidates: list[_SegmentalPath] = []
            for previous_boundary in range(boundary_index):
                segment = _segment_score(
                    state=state,
                    start_index=previous_boundary + 1,
                    end_index=boundary_index,
                    emission_prefix=emission_prefix,
                    duration_prefix=duration_prefix,
                    duration_location=duration_location,
                    duration_scale=duration_scale,
                    minimum_duration_seconds=minimum_duration_seconds,
                    censored=False,
                )
                if segment is None:
                    continue
                for entry in prefix[previous_boundary]:
                    candidates.append(
                        _SegmentalPath(
                            entry.score
                            + segment
                            + transition_log_scores[
                                boundary_index, transition_index
                            ],
                            start_state,
                            end_state,
                            entry.transition_indices + (boundary_index,),
                        )
                    )
            updated[boundary_index] = _select_top_k_paths(
                candidates, maximum_paths
            )
        prefix = updated

    completed: list[_SegmentalPath] = []
    for previous_boundary in range(grid_count - 1):
        segment = _segment_score(
            state=end_state,
            start_index=previous_boundary + 1,
            end_index=grid_count - 1,
            emission_prefix=emission_prefix,
            duration_prefix=duration_prefix,
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration_seconds,
            censored=right_censored,
        )
        if segment is None:
            continue
        for entry in prefix[previous_boundary]:
            completed.append(
                _SegmentalPath(
                    event_log_score
                    + end_log_scores[end_state]
                    + entry.score
                    + segment,
                    start_state,
                    end_state,
                    entry.transition_indices,
                )
            )
    return _select_top_k_paths(completed, maximum_paths)


def _physical_footprint_key(
    reference_row: torch.Tensor,
    *,
    relative_threshold: float,
    maximum_footprint: int,
    fallback_unit_index: int,
) -> tuple[int, ...]:
    magnitude = reference_row.detach().abs()
    maximum = float(magnitude.max()) if magnitude.numel() else 0.0
    if maximum <= 0.0:
        return (-(fallback_unit_index + 1),)
    selected = tuple(
        int(index)
        for index in torch.nonzero(
            magnitude >= maximum * relative_threshold, as_tuple=False
        ).flatten().detach().cpu()
    )
    if not selected or len(selected) > maximum_footprint:
        return (-(fallback_unit_index + 1),)
    return selected


def _hierarchical_pool_rows(
    token_hidden: torch.Tensor,
    token_indices: Sequence[int],
    *,
    token_unit_index: torch.Tensor,
    token_scale_index: torch.Tensor,
    unit_view_index: torch.Tensor,
    view_reference_family_index: torch.Tensor,
    unit_reference_matrix: torch.Tensor,
    row_weights: torch.Tensor | None,
    relative_reference_threshold: float,
    maximum_local_footprint: int,
) -> torch.Tensor:
    """Pool reference replicas within a physical footprint before averaging.

    The key retains a bipolar footprint as a pair and never attributes the
    row to either endpoint.  Replicas are first averaged inside one
    footprint/scale/reference-family cell and only then are independent cells
    averaged, so copying one reference family cannot outvote another.
    """

    groups: dict[tuple[tuple[int, ...], int, int], list[int]] = {}
    for token_index in token_indices:
        unit_index = int(token_unit_index[token_index])
        scale_index = int(token_scale_index[token_index])
        view_index = int(unit_view_index[unit_index])
        reference_family_index = int(
            view_reference_family_index[view_index]
        )
        if reference_family_index < 0:
            raise RuntimeError("active token has no registered reference family")
        footprint = _physical_footprint_key(
            unit_reference_matrix[unit_index],
            relative_threshold=relative_reference_threshold,
            maximum_footprint=maximum_local_footprint,
            fallback_unit_index=unit_index,
        )
        groups.setdefault(
            (footprint, scale_index, reference_family_index), []
        ).append(token_index)
    pooled_groups: list[torch.Tensor] = []
    for indices in groups.values():
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=token_hidden.device
        )
        local = token_hidden[index_tensor]
        if row_weights is None:
            pooled_groups.append(local.mean(dim=0))
        else:
            weights = row_weights[index_tensor].clamp_min(0.0)
            pooled_groups.append(
                torch.sum(local * weights.unsqueeze(-1), dim=0)
                / weights.sum().clamp_min(torch.finfo(local.dtype).eps)
            )
    if not pooled_groups:
        return torch.zeros(
            token_hidden.shape[-1],
            dtype=token_hidden.dtype,
            device=token_hidden.device,
        )
    return torch.stack(pooled_groups).mean(dim=0)


def _typed_unit_key(
    reference_row: torch.Tensor,
    *,
    reference_family_index: int,
    relative_reference_threshold: float,
    maximum_local_footprint: int,
) -> tuple[int, int, int] | None:
    """Project a constructive reference row to an onset-safe typed identity.

    Referential/common-average rows may represent one physical electrode only
    when their dominant physical footprint is a singleton.  A bipolar row is
    retained as one unordered lead identity; it is never expanded into either
    endpoint.  Laplacian and arbitrary linear rows intentionally fail closed.
    """

    if reference_family_index < 0 or reference_family_index >= len(
        BA_IEG_REFERENCE_FAMILIES
    ):
        return None
    family = BA_IEG_REFERENCE_FAMILIES[reference_family_index]
    footprint = _physical_footprint_key(
        reference_row,
        relative_threshold=relative_reference_threshold,
        maximum_footprint=maximum_local_footprint,
        fallback_unit_index=0,
    )
    if any(index < 0 for index in footprint):
        return None
    if family in {"referential", "common_average"} and len(footprint) == 1:
        return (BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("physical_electrode"), footprint[0], -1)
    if family == "bipolar" and len(footprint) == 2:
        first, second = sorted(footprint)
        if float(reference_row[first] * reference_row[second]) >= 0.0:
            return None
        return (BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("bipolar_lead"), first, second)
    return None


def _event_typed_unit_inventory(
    batch: BAIEGCollatedEventBatch,
    batch_index: int,
    *,
    relative_reference_threshold: float,
    maximum_local_footprint: int,
) -> tuple[tuple[tuple[int, int, int], ...], dict[tuple[int, int, int], tuple[int, ...]]]:
    """Return deterministic typed keys and their analysis-unit aliases."""

    spatial_index = BA_IEG_EVIDENCE_FAMILIES.index("spatial_field")
    aliases: dict[tuple[int, int, int], list[int]] = {}
    maximum_units = int(batch.unit_row_mask.shape[1])
    for unit_index in range(maximum_units):
        if not bool(batch.unit_row_mask[batch_index, unit_index]):
            continue
        view_index = int(batch.unit_view_index[batch_index, unit_index])
        if view_index < 0 or not bool(
            batch.view_row_mask[batch_index, view_index]
        ):
            continue
        eligible = bool(batch.unit_evidence_mask[batch_index, unit_index]) and bool(
            batch.unit_family_mask[batch_index, unit_index, spatial_index]
        )
        onset_safe = bool(
            batch.view_onset_evidence_authorized[batch_index, view_index]
        ) and not bool(batch.view_future_sample_access[batch_index, view_index])
        has_positive_token = bool(
            (
                batch.token_positive_onset_mask[batch_index]
                & (batch.token_unit_index[batch_index] == unit_index)
            ).any()
        )
        if not (eligible and onset_safe and has_positive_token):
            continue
        reference_family_index = int(
            batch.view_reference_family_index[batch_index, view_index]
        )
        key = _typed_unit_key(
            batch.unit_reference_matrix[batch_index, unit_index],
            reference_family_index=reference_family_index,
            relative_reference_threshold=relative_reference_threshold,
            maximum_local_footprint=maximum_local_footprint,
        )
        if key is not None:
            aliases.setdefault(key, []).append(unit_index)
    keys = tuple(sorted(aliases))
    return keys, {key: tuple(aliases[key]) for key in keys}


def _unique_sorted_times(
    values: Sequence[float], *, tolerance_seconds: float
) -> list[float]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not result or abs(value - result[-1]) > tolerance_seconds:
            result.append(value)
        else:
            result[-1] = 0.5 * (result[-1] + value)
    return result


def _overlap_length(
    start: float, stop: float, intervals: Sequence[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(stop, right) - max(start, left))
        for left, right in intervals
    )


def _event_physical_lattice(
    *,
    token_bounds: torch.Tensor,
    active_indices: Sequence[int],
    support_intervals: Sequence[tuple[float, float]],
    quality_gaps: Sequence[tuple[float, float]],
    tolerance_seconds: float,
) -> tuple[list[tuple[float, float]], list[float]]:
    edges = [coordinate for interval in support_intervals for coordinate in interval]
    edges.extend(coordinate for interval in quality_gaps for coordinate in interval)
    for token_index in active_indices:
        edges.extend(
            (
                float(token_bounds[token_index, 0]),
                float(token_bounds[token_index, 1]),
            )
        )
    unique = _unique_sorted_times(edges, tolerance_seconds=tolerance_seconds)
    cells: list[tuple[float, float]] = []
    durations: list[float] = []
    for start, stop in zip(unique[:-1], unique[1:]):
        if stop - start <= tolerance_seconds:
            continue
        support = _overlap_length(start, stop, support_intervals)
        gap = _overlap_length(start, stop, quality_gaps)
        opportunity = max(0.0, support - gap)
        cells.append((start, stop))
        durations.append(opportunity)
    return cells, durations


def _discrete_hazard_distribution(
    logits: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return event-time mass and survival mass on a masked physical clock."""

    if logits.ndim != 1 or tuple(mask.shape) != tuple(logits.shape):
        raise ValueError("hazard logits and mask must be aligned vectors")
    hazards = torch.where(mask, torch.sigmoid(logits), torch.zeros_like(logits))
    masses: list[torch.Tensor] = []
    survival = torch.ones((), dtype=logits.dtype, device=logits.device)
    for index in range(int(logits.numel())):
        if bool(mask[index]):
            mass = survival * hazards[index]
            survival = survival * (1.0 - hazards[index])
        else:
            mass = torch.zeros((), dtype=logits.dtype, device=logits.device)
        masses.append(mass)
    return torch.stack(masses) if masses else torch.zeros_like(logits), survival


@dataclass(frozen=True)
class BAIEGCausalTypedUnitTrace:
    """Future-free causal seam shared by the segmental and spatial heads.

    The global trace is the actual post-prefix-GRU state used by the
    segmental onset lane.  ``typed_unit_local_hidden`` contains only the
    corresponding pre-GRU projected causal rows after alias-safe hierarchical
    pooling.  No offline hidden, phase posterior or full-path marginal is
    present in this type, so a downstream typed-unit head cannot accidentally
    consume future context through its public API.
    """

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    source_event_receipt_sha256s: tuple[str, ...]
    identity_roster_sha256: str
    implementation_id: str
    group_times_seconds: torch.Tensor
    group_boundary_bounds_seconds: torch.Tensor
    group_mask: torch.Tensor
    global_onset_boundary_mass: torch.Tensor
    global_left_censor_state_mass: torch.Tensor
    global_no_onset_within_support_mass: torch.Tensor
    global_group_hidden: torch.Tensor
    typed_unit_local_hidden: torch.Tensor
    typed_unit_time_mask: torch.Tensor
    typed_unit_mask: torch.Tensor
    typed_unit_kind_index: torch.Tensor
    typed_unit_electrode_index: torch.Tensor
    typed_unit_lead_endpoint_index: torch.Tensor
    typed_unit_source_analysis_unit_mask: torch.Tensor
    typed_unit_reference_family_mask: torch.Tensor
    future_sample_access: bool = False
    temporal_permission: str = "past_and_present_only"

    def verify_shapes(self) -> None:
        if self.implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID:
            raise ValueError("causal typed-unit trace implementation drifted")
        if self.future_sample_access is not False or self.temporal_permission != (
            "past_and_present_only"
        ):
            raise ValueError("causal typed-unit trace permission drifted")
        if self.group_times_seconds.ndim != 2:
            raise ValueError("causal group times must have shape [B,G]")
        batch_size, group_count = self.group_times_seconds.shape
        if (
            len(self.event_ids) != batch_size
            or len(self.recording_ids) != batch_size
            or len(self.source_event_receipt_sha256s) != batch_size
            or len(set(self.event_ids)) != batch_size
        ):
            raise ValueError("causal trace event identity roster is not aligned")
        if self.identity_roster_sha256 != ba_ieg_event_identity_roster_sha256(
            source_input_batch_sha256=self.source_input_batch_sha256,
            event_ids=self.event_ids,
            recording_ids=self.recording_ids,
            source_event_receipt_sha256s=self.source_event_receipt_sha256s,
        ):
            raise ValueError("causal trace event identity receipt drifted")
        if tuple(self.group_boundary_bounds_seconds.shape) != (
            batch_size,
            group_count,
            2,
        ) or tuple(self.group_mask.shape) != (batch_size, group_count):
            raise ValueError("causal group coordinates/mask are not aligned")
        if (
            tuple(self.global_onset_boundary_mass.shape)
            != (batch_size, group_count)
            or tuple(self.global_left_censor_state_mass.shape) != (batch_size, 2)
            or tuple(self.global_no_onset_within_support_mass.shape)
            != (batch_size,)
        ):
            raise ValueError("global causal onset status masses are not aligned")
        for name, value in (
            ("global_onset_boundary_mass", self.global_onset_boundary_mass),
            ("global_left_censor_state_mass", self.global_left_censor_state_mass),
            (
                "global_no_onset_within_support_mass",
                self.global_no_onset_within_support_mass,
            ),
        ):
            if (
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or torch.any(value < 0)
            ):
                raise ValueError(f"{name} must contain finite non-negative mass")
        if torch.any(
            (self.global_onset_boundary_mass > 0) & ~self.group_mask
        ):
            raise ValueError("global onset mass exceeds causal opportunity")
        global_status_total = (
            self.global_onset_boundary_mass.sum(dim=1)
            + self.global_left_censor_state_mass.sum(dim=1)
            + self.global_no_onset_within_support_mass
        )
        if not torch.allclose(
            global_status_total,
            torch.ones_like(global_status_total),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("global causal onset status masses are not normalized")
        if self.global_group_hidden.ndim != 3 or tuple(
            self.global_group_hidden.shape[:2]
        ) != (batch_size, group_count):
            raise ValueError("global causal hidden must have shape [B,G,H]")
        if self.typed_unit_local_hidden.ndim != 4 or tuple(
            self.typed_unit_local_hidden.shape[:2]
        ) != (batch_size, group_count):
            raise ValueError("typed-unit causal hidden must have shape [B,G,K,H]")
        typed_count = int(self.typed_unit_local_hidden.shape[2])
        hidden_dim = int(self.global_group_hidden.shape[-1])
        if int(self.typed_unit_local_hidden.shape[-1]) != hidden_dim:
            raise ValueError("global/local causal hidden dimensions disagree")
        if tuple(self.typed_unit_time_mask.shape) != (
            batch_size,
            group_count,
            typed_count,
        ) or tuple(self.typed_unit_mask.shape) != (batch_size, typed_count):
            raise ValueError("typed-unit masks are not aligned")
        if tuple(self.typed_unit_kind_index.shape) != (
            batch_size,
            typed_count,
        ) or tuple(self.typed_unit_electrode_index.shape) != (
            batch_size,
            typed_count,
        ) or tuple(self.typed_unit_lead_endpoint_index.shape) != (
            batch_size,
            typed_count,
            2,
        ):
            raise ValueError("typed-unit identity tensors are not aligned")
        if self.typed_unit_source_analysis_unit_mask.ndim != 3 or tuple(
            self.typed_unit_source_analysis_unit_mask.shape[:2]
        ) != (batch_size, typed_count):
            raise ValueError("typed-unit alias map is not aligned")
        if tuple(self.typed_unit_reference_family_mask.shape) != (
            batch_size,
            typed_count,
            len(BA_IEG_REFERENCE_FAMILIES),
        ):
            raise ValueError("typed-unit reference-family mask is not aligned")
        if self.group_mask.dtype != torch.bool or self.typed_unit_time_mask.dtype != (
            torch.bool
        ) or self.typed_unit_mask.dtype != torch.bool:
            raise ValueError("causal trace masks must be boolean")
        if torch.any(self.typed_unit_time_mask & ~self.group_mask.unsqueeze(-1)):
            raise ValueError("typed-unit time opportunity exceeds causal groups")
        if torch.any(
            self.typed_unit_time_mask & ~self.typed_unit_mask.unsqueeze(1)
        ):
            raise ValueError("typed-unit time opportunity exceeds unit inventory")


@dataclass(frozen=True)
class BAIEGPermissionSplitSegmentalStateOutput:
    """Uncalibrated EEG-only potentials, exact partition and retained paths."""

    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    implementation_id: str
    heuristic_phase_posterior_used: bool
    causal_typed_unit_trace: BAIEGCausalTypedUnitTrace
    causal_candidate_times_seconds: torch.Tensor
    causal_candidate_mask: torch.Tensor
    causal_onset_hazard_logits: torch.Tensor
    causal_onset_boundary_mass: torch.Tensor
    causal_left_censor_state_mass: torch.Tensor
    causal_no_onset_within_support_mass: torch.Tensor
    lattice_cell_bounds_seconds: torch.Tensor
    lattice_cell_mask: torch.Tensor
    lattice_opportunity_duration_seconds: torch.Tensor
    lattice_physical_duration_seconds: torch.Tensor
    quality_gap_cell_mask: torch.Tensor
    offline_state_emission_logits: torch.Tensor
    offline_state_emission_log_prob: torch.Tensor
    offline_offset_hazard_logits: torch.Tensor
    offline_offset_boundary_mass: torch.Tensor
    offline_right_censor_state_mass: torch.Tensor
    offline_no_offset_within_support_mass: torch.Tensor
    split_or_reentry_review_logits: torch.Tensor
    transition_log_scores: torch.Tensor
    transition_mask: torch.Tensor
    start_state_log_scores: torch.Tensor
    end_state_log_scores: torch.Tensor
    event_presence_log_scores: torch.Tensor
    exact_path_log_partition: torch.Tensor
    full_state_marginals: torch.Tensor
    full_transition_marginals: torch.Tensor
    full_start_state_marginals: torch.Tensor
    full_end_state_marginals: torch.Tensor
    full_segment_count_marginals: torch.Tensor
    exact_onset_boundary_mass: torch.Tensor
    exact_primary_onset_boundary_mass: torch.Tensor
    exact_secondary_onset_boundary_mass: torch.Tensor
    exact_s0_onset_boundary_mass: torch.Tensor
    exact_s3_reentry_onset_boundary_mass: torch.Tensor
    exact_offset_boundary_mass: torch.Tensor
    exact_event_mass: torch.Tensor
    exact_null_mass: torch.Tensor
    exact_left_censor_mass: torch.Tensor
    exact_right_censor_mass: torch.Tensor
    exact_both_censor_mass: torch.Tensor
    exact_recurrent_event_mass: torch.Tensor
    exact_expected_event_bout_count: torch.Tensor
    exact_expected_recurrent_bout_count: torch.Tensor
    path_start_state_index: torch.Tensor
    path_end_state_index: torch.Tensor
    path_transition_times_seconds: torch.Tensor
    path_transition_mask: torch.Tensor
    path_transition_type_index: torch.Tensor
    path_segment_bounds_seconds: torch.Tensor
    path_segment_state_index: torch.Tensor
    path_segment_mask: torch.Tensor
    path_recurrent_cycle_count: torch.Tensor
    path_log_scores: torch.Tensor
    path_posterior_mass: torch.Tensor
    path_weights_conditional_on_retained: torch.Tensor
    path_mask: torch.Tensor
    retained_path_mass_fraction: torch.Tensor
    residual_path_mass_fraction: torch.Tensor
    retained_state_marginals: torch.Tensor
    one_way_consistency_loss_per_event: torch.Tensor
    event_evaluable_mask: torch.Tensor
    duration_location_log_seconds: torch.Tensor
    duration_scale_log_seconds: torch.Tensor
    minimum_state_duration_seconds: torch.Tensor
    calibration_status: str = "uncalibrated_source_development_shadow"
    state_marginal_semantics: str = "retained_k_conditional_not_full_posterior"
    full_marginal_semantics: str = (
        "exact_full_finite_path_posterior_not_calibrated_clinical_probability"
    )


def _general_segment_score(
    *,
    state: int,
    start_index: int,
    end_index: int,
    emission_prefix: torch.Tensor,
    physical_duration_prefix: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    censored: bool,
) -> torch.Tensor | None:
    if end_index < start_index:
        return None
    emission = emission_prefix[end_index + 1, state] - emission_prefix[start_index, state]
    physical_duration = (
        physical_duration_prefix[end_index + 1]
        - physical_duration_prefix[start_index]
    )
    if float(physical_duration.detach().cpu()) + 1e-8 < float(
        minimum_duration_seconds[state].detach().cpu()
    ):
        return None
    return emission + _lognormal_duration_score(
        physical_duration,
        location=duration_location[state],
        scale=duration_scale[state],
        censored=censored,
    )


def _logadd(existing: torch.Tensor | None, candidate: torch.Tensor) -> torch.Tensor:
    if existing is None:
        return candidate
    return torch.logaddexp(existing, candidate)


def segmental_log_partition_from_potentials(
    *,
    state_emission_log_prob: torch.Tensor,
    opportunity_duration_seconds: torch.Tensor,
    physical_duration_seconds: torch.Tensor,
    transition_log_scores: torch.Tensor,
    transition_mask: torch.Tensor,
    start_log_scores: torch.Tensor,
    end_log_scores: torch.Tensor,
    event_log_score: torch.Tensor,
    no_event_log_score: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    left_censoring_possible: bool,
    right_censoring_possible: bool,
    maximum_segments: int,
    allowed_start_states: Sequence[int] | None = None,
    allowed_end_states: Sequence[int] | None = None,
    allowed_transition_mask: torch.Tensor | None = None,
    require_event: bool | None = None,
) -> torch.Tensor:
    """Exact finite segmental-CRF partition on an irregular physical grid.

    Cycles are finite because ``maximum_segments`` is explicit.  Emissions are
    integrated over EEG opportunity seconds, while duration priors use actual
    physical cell duration, including an unobserved quality gap.  The latter
    prevents a ten-second gap from becoming a zero-duration state; the former
    prevents the gap from contributing evidence for any state.
    """

    state_count = len(BA_IEG_PHASE_STATES)
    edge_count = len(BA_IEG_SEGMENTAL_TRANSITIONS)
    grid_count = int(state_emission_log_prob.shape[0])
    if maximum_segments < 1:
        raise ValueError("maximum_segments must be positive")
    if tuple(state_emission_log_prob.shape) != (grid_count, state_count):
        raise ValueError("state emission potentials have invalid shape")
    if tuple(opportunity_duration_seconds.shape) != (grid_count,) or tuple(
        physical_duration_seconds.shape
    ) != (grid_count,):
        raise ValueError("physical and opportunity durations must align with the grid")
    if torch.any(opportunity_duration_seconds < 0) or torch.any(
        physical_duration_seconds <= 0
    ) or torch.any(opportunity_duration_seconds > physical_duration_seconds + 1e-7):
        raise ValueError("cell opportunity/physical durations are invalid")
    if tuple(transition_log_scores.shape) != (grid_count, edge_count) or tuple(
        transition_mask.shape
    ) != tuple(transition_log_scores.shape):
        raise ValueError("transition potentials do not align with the topology")
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must be boolean")
    if allowed_transition_mask is not None:
        if allowed_transition_mask.dtype != torch.bool or tuple(
            allowed_transition_mask.shape
        ) != tuple(transition_mask.shape):
            raise ValueError("allowed transition mask does not align")
        transition_mask = transition_mask & allowed_transition_mask
    for name, value in (
        ("start_log_scores", start_log_scores),
        ("end_log_scores", end_log_scores),
        ("duration_location", duration_location),
        ("duration_scale", duration_scale),
        ("minimum_duration_seconds", minimum_duration_seconds),
    ):
        if tuple(value.shape) != (state_count,):
            raise ValueError(f"{name} must have one value per state")

    default_starts = {0, 1, 2} if left_censoring_possible else {0}
    default_ends = {0, 1, 2, 3} if right_censoring_possible else {0, 3}
    starts = set(default_starts if allowed_start_states is None else allowed_start_states)
    ends = set(default_ends if allowed_end_states is None else allowed_end_states)
    if not starts or not ends or not starts <= default_starts or not ends <= default_ends:
        raise ValueError("start/end constraints violate censor permissions")

    weighted_emission = state_emission_log_prob * opportunity_duration_seconds.unsqueeze(-1)
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, state_count),
                dtype=weighted_emission.dtype,
                device=weighted_emission.device,
            ),
            torch.cumsum(weighted_emission, dim=0),
        ),
        dim=0,
    )
    physical_prefix = torch.cat(
        (
            torch.zeros(
                1,
                dtype=physical_duration_seconds.dtype,
                device=physical_duration_seconds.device,
            ),
            torch.cumsum(physical_duration_seconds, dim=0),
        )
    )

    # key = (current_state, current_segment_start_cell, has_event, initial_state)
    current: dict[tuple[int, int, bool, int], torch.Tensor] = {
        (state, 0, state > 0, state): start_log_scores[state]
        for state in sorted(starts)
    }
    completed: list[torch.Tensor] = []
    for segment_count in range(1, maximum_segments + 1):
        for (state, segment_start, has_event, initial_state), prefix_score in current.items():
            if state not in ends:
                continue
            if require_event is not None and has_event is not require_event:
                continue
            right_censored = right_censoring_possible and state in (1, 2)
            left_censored_same_segment = (
                segment_count == 1 and initial_state > 0
            )
            segment = _general_segment_score(
                state=state,
                start_index=segment_start,
                end_index=grid_count - 1,
                emission_prefix=emission_prefix,
                physical_duration_prefix=physical_prefix,
                duration_location=duration_location,
                duration_scale=duration_scale,
                minimum_duration_seconds=minimum_duration_seconds,
                censored=right_censored or left_censored_same_segment,
            )
            if segment is not None:
                completed.append(
                    prefix_score
                    + segment
                    + end_log_scores[state]
                    + (event_log_score if has_event else no_event_log_score)
                )
        if segment_count == maximum_segments:
            break
        updated: dict[tuple[int, int, bool, int], torch.Tensor] = {}
        for (state, segment_start, has_event, initial_state), prefix_score in current.items():
            outgoing = [
                (edge_index, target)
                for edge_index, (source, target) in enumerate(
                    BA_IEG_SEGMENTAL_TRANSITION_EDGES
                )
                if source == state
            ]
            for boundary_index in range(segment_start, grid_count - 1):
                left_censored = segment_count == 1 and initial_state > 0
                segment = _general_segment_score(
                    state=state,
                    start_index=segment_start,
                    end_index=boundary_index,
                    emission_prefix=emission_prefix,
                    physical_duration_prefix=physical_prefix,
                    duration_location=duration_location,
                    duration_scale=duration_scale,
                    minimum_duration_seconds=minimum_duration_seconds,
                    censored=left_censored,
                )
                if segment is None:
                    continue
                for edge_index, target in outgoing:
                    if not bool(transition_mask[boundary_index, edge_index]):
                        continue
                    new_has_event = has_event or edge_index in (
                        BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES
                    )
                    key = (target, boundary_index + 1, new_has_event, initial_state)
                    candidate = (
                        prefix_score
                        + segment
                        + transition_log_scores[boundary_index, edge_index]
                    )
                    updated[key] = _logadd(updated.get(key), candidate)
        current = updated
        if not current:
            break
    if not completed:
        return torch.full(
            (),
            -torch.inf,
            dtype=state_emission_log_prob.dtype,
            device=state_emission_log_prob.device,
        )
    return torch.logsumexp(torch.stack(completed), dim=0)


@dataclass(frozen=True)
class _GeneralPartialPath:
    score: torch.Tensor
    current_state: int
    current_segment_start: int
    start_state: int
    state_sequence: tuple[int, ...]
    transition_indices: tuple[int, ...]
    transition_edge_indices: tuple[int, ...]
    has_event: bool


@dataclass(frozen=True)
class _GeneralCompletedPath:
    score: torch.Tensor
    start_state: int
    end_state: int
    state_sequence: tuple[int, ...]
    transition_indices: tuple[int, ...]
    transition_edge_indices: tuple[int, ...]


def _select_top_k_general(
    entries: Sequence[_GeneralPartialPath] | Sequence[_GeneralCompletedPath],
    maximum_paths: int,
) -> list[_GeneralPartialPath] | list[_GeneralCompletedPath]:
    if not entries:
        return []
    # Stable semantic tie order is independent of token row/enumeration order.
    ordered = sorted(
        entries,
        key=lambda entry: (
            tuple(entry.transition_indices),
            tuple(entry.state_sequence),
            int(entry.start_state),
        ),
    )
    scores = torch.stack([entry.score for entry in ordered])
    ranking = torch.argsort(scores, descending=True, stable=True)
    return [ordered[int(index)] for index in ranking[:maximum_paths]]


def _top_k_general_segmental_paths(
    *,
    state_emission_log_prob: torch.Tensor,
    opportunity_duration_seconds: torch.Tensor,
    physical_duration_seconds: torch.Tensor,
    transition_log_scores: torch.Tensor,
    transition_mask: torch.Tensor,
    start_log_scores: torch.Tensor,
    end_log_scores: torch.Tensor,
    event_log_score: torch.Tensor,
    no_event_log_score: torch.Tensor,
    duration_location: torch.Tensor,
    duration_scale: torch.Tensor,
    minimum_duration_seconds: torch.Tensor,
    left_censoring_possible: bool,
    right_censoring_possible: bool,
    maximum_segments: int,
    maximum_paths: int,
) -> list[_GeneralCompletedPath]:
    state_count = len(BA_IEG_PHASE_STATES)
    grid_count = int(state_emission_log_prob.shape[0])
    weighted_emission = state_emission_log_prob * opportunity_duration_seconds.unsqueeze(-1)
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, state_count),
                dtype=weighted_emission.dtype,
                device=weighted_emission.device,
            ),
            torch.cumsum(weighted_emission, dim=0),
        ),
        dim=0,
    )
    physical_prefix = torch.cat(
        (
            torch.zeros(
                1,
                dtype=physical_duration_seconds.dtype,
                device=physical_duration_seconds.device,
            ),
            torch.cumsum(physical_duration_seconds, dim=0),
        )
    )
    starts = (0, 1, 2) if left_censoring_possible else (0,)
    ends = {0, 1, 2, 3} if right_censoring_possible else {0, 3}
    current: list[_GeneralPartialPath] = [
        _GeneralPartialPath(
            score=start_log_scores[state],
            current_state=state,
            current_segment_start=0,
            start_state=state,
            state_sequence=(state,),
            transition_indices=(),
            transition_edge_indices=(),
            has_event=state > 0,
        )
        for state in starts
    ]
    completed: list[_GeneralCompletedPath] = []
    for segment_count in range(1, maximum_segments + 1):
        for entry in current:
            state = entry.current_state
            if state in ends:
                segment = _general_segment_score(
                    state=state,
                    start_index=entry.current_segment_start,
                    end_index=grid_count - 1,
                    emission_prefix=emission_prefix,
                    physical_duration_prefix=physical_prefix,
                    duration_location=duration_location,
                    duration_scale=duration_scale,
                    minimum_duration_seconds=minimum_duration_seconds,
                    censored=(
                        (right_censoring_possible and state in (1, 2))
                        or (segment_count == 1 and entry.start_state > 0)
                    ),
                )
                if segment is not None:
                    completed.append(
                        _GeneralCompletedPath(
                            score=(
                                entry.score
                                + segment
                                + end_log_scores[state]
                                + (
                                    event_log_score
                                    if entry.has_event
                                    else no_event_log_score
                                )
                            ),
                            start_state=entry.start_state,
                            end_state=state,
                            state_sequence=entry.state_sequence,
                            transition_indices=entry.transition_indices,
                            transition_edge_indices=entry.transition_edge_indices,
                        )
                    )
        if segment_count == maximum_segments:
            break
        buckets: dict[
            tuple[int, int, int, bool], list[_GeneralPartialPath]
        ] = {}
        for entry in current:
            state = entry.current_state
            outgoing = [
                (edge_index, target)
                for edge_index, (source, target) in enumerate(
                    BA_IEG_SEGMENTAL_TRANSITION_EDGES
                )
                if source == state
            ]
            for boundary_index in range(entry.current_segment_start, grid_count - 1):
                segment = _general_segment_score(
                    state=state,
                    start_index=entry.current_segment_start,
                    end_index=boundary_index,
                    emission_prefix=emission_prefix,
                    physical_duration_prefix=physical_prefix,
                    duration_location=duration_location,
                    duration_scale=duration_scale,
                    minimum_duration_seconds=minimum_duration_seconds,
                    censored=segment_count == 1 and entry.start_state > 0,
                )
                if segment is None:
                    continue
                for edge_index, target in outgoing:
                    if not bool(transition_mask[boundary_index, edge_index]):
                        continue
                    new_entry = _GeneralPartialPath(
                        score=(
                            entry.score
                            + segment
                            + transition_log_scores[boundary_index, edge_index]
                        ),
                        current_state=target,
                        current_segment_start=boundary_index + 1,
                        start_state=entry.start_state,
                        state_sequence=entry.state_sequence + (target,),
                        transition_indices=entry.transition_indices
                        + (boundary_index,),
                        transition_edge_indices=entry.transition_edge_indices
                        + (edge_index,),
                        has_event=(
                            entry.has_event
                            or edge_index
                            in BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES
                        ),
                    )
                    key = (
                        target,
                        boundary_index + 1,
                        entry.start_state,
                        new_entry.has_event,
                    )
                    buckets.setdefault(key, []).append(new_entry)
        current = []
        for entries in buckets.values():
            current.extend(_select_top_k_general(entries, maximum_paths))
        if not current:
            break
    return list(_select_top_k_general(completed, maximum_paths))


def _token_time_features(bounds_seconds: torch.Tensor) -> torch.Tensor:
    start = bounds_seconds[..., 0]
    stop = bounds_seconds[..., 1]
    midpoint = 0.5 * (start + stop)
    duration = (stop - start).clamp_min(0.0)
    return torch.stack(
        (
            torch.asinh(start / 60.0),
            torch.asinh(stop / 60.0),
            torch.asinh(midpoint / 60.0),
            torch.log1p(duration),
        ),
        dim=-1,
    )


class BAIEGPermissionSplitSegmentalStateModel(nn.Module):
    """Trainable dual-lane state model with an exact finite segmental CRF."""

    implementation_id: Final[str] = BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
        maximum_segments: int = 8,
        maximum_paths: int = 8,
        minimum_state_duration_seconds: tuple[float, float, float, float] = (
            0.0,
            0.25,
            0.25,
            0.0,
        ),
        duration_location_seconds: tuple[float, float, float, float] = (
            8.0,
            2.0,
            12.0,
            8.0,
        ),
        duration_scale_initializer: float = 0.75,
        dropout: float = 0.0,
        time_tolerance_seconds: float = 1e-6,
        relative_reference_threshold: float = 0.2,
        maximum_local_footprint: int = 6,
        allow_heuristic_phase_posterior: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if maximum_segments < 2 or maximum_paths <= 0:
            raise ValueError("maximum_segments/maximum_paths are invalid")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if (
            not math.isfinite(time_tolerance_seconds)
            or time_tolerance_seconds < 0.0
            or not math.isfinite(relative_reference_threshold)
            or relative_reference_threshold <= 0.0
            or relative_reference_threshold > 1.0
            or maximum_local_footprint <= 0
        ):
            raise ValueError("physical grouping policy is invalid")
        if len(minimum_state_duration_seconds) != len(BA_IEG_PHASE_STATES) or any(
            not math.isfinite(value) or value < 0.0
            for value in minimum_state_duration_seconds
        ):
            raise ValueError("minimum state durations are invalid")
        if len(duration_location_seconds) != len(BA_IEG_PHASE_STATES) or any(
            not math.isfinite(value) or value <= 0.0
            for value in duration_location_seconds
        ):
            raise ValueError("duration location initializers are invalid")
        if not math.isfinite(duration_scale_initializer) or duration_scale_initializer <= 0:
            raise ValueError("duration scale initializer is invalid")
        if type(allow_heuristic_phase_posterior) is not bool:
            raise TypeError("allow_heuristic_phase_posterior must be boolean")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_segments = int(maximum_segments)
        self.maximum_paths = int(maximum_paths)
        self.time_tolerance_seconds = float(time_tolerance_seconds)
        self.relative_reference_threshold = float(relative_reference_threshold)
        self.maximum_local_footprint = int(maximum_local_footprint)
        self.allow_heuristic_phase_posterior = allow_heuristic_phase_posterior

        # No trainable parameter is shared across the two lanes.
        self.causal_value_projection = nn.Linear(2 * feature_dim + 4, hidden_dim)
        self.causal_scale_embedding = nn.Embedding(
            len(BA_IEG_TOKEN_SCALES), hidden_dim
        )
        self.causal_input_norm = nn.LayerNorm(hidden_dim)
        self.causal_prefix_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.causal_onset_hazard_head = nn.Linear(hidden_dim, 1)
        self.causal_left_state_head = nn.Linear(hidden_dim, 3)
        self.causal_transition_type_head = nn.Linear(
            hidden_dim, len(BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES)
        )
        self.causal_empty_left_logits = nn.Parameter(torch.zeros(3))

        # Phase has a fixed input slot so the main and explicitly labelled
        # heuristic-prior ablation have checkpoint-compatible dimensions.  In
        # the main arm this slot is an input-independent zero tensor.
        self.offline_value_projection = nn.Linear(
            2 * feature_dim + 4 + len(BA_IEG_PHASE_STATES) + 1,
            hidden_dim,
        )
        self.offline_scale_embedding = nn.Embedding(
            len(BA_IEG_TOKEN_SCALES), hidden_dim
        )
        self.offline_input_norm = nn.LayerNorm(hidden_dim)
        self.causal_to_offline_projection = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self.offline_forward_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.offline_backward_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.offline_bidirectional_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.offline_state_head = nn.Linear(hidden_dim, len(BA_IEG_PHASE_STATES))
        self.offline_course_transition_head = nn.Linear(hidden_dim, 4)
        self.offline_offset_hazard_head = nn.Linear(hidden_dim, 1)
        self.offline_right_status_head = nn.Linear(hidden_dim, 3)
        self.offline_event_presence_head = nn.Linear(hidden_dim, 2)
        self.offline_split_reentry_review_head = nn.Linear(hidden_dim, 1)
        self.offline_empty_right_logits = nn.Parameter(torch.zeros(3))
        self.offline_empty_event_logits = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(dropout)

        self.duration_location_log_seconds = nn.Parameter(
            torch.log(torch.tensor(duration_location_seconds, dtype=torch.float32))
        )
        inverse_softplus = math.log(math.expm1(duration_scale_initializer))
        self.duration_scale_unconstrained = nn.Parameter(
            torch.full((len(BA_IEG_PHASE_STATES),), inverse_softplus)
        )
        self.register_buffer(
            "minimum_state_duration_seconds",
            torch.tensor(minimum_state_duration_seconds, dtype=torch.float32),
            persistent=True,
        )

    def _project_token_lanes(
        self, batch: BAIEGCollatedEventBatch
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        causal = batch.onset_causal_inputs()
        full = batch.model_inputs()
        causal_active = causal["token_row_mask"] & causal["token_signal_mask"]
        full_active = full["token_row_mask"] & full["token_signal_mask"]
        if torch.any(causal["token_future_sample_access"] & causal_active):
            raise RuntimeError("future-dependent token reached the causal state lane")

        values = batch.token_values
        feature_mask = batch.token_feature_mask
        masked_values = torch.where(feature_mask, values, torch.zeros_like(values))
        time_features = _token_time_features(batch.token_time_bounds_seconds).to(
            dtype=values.dtype
        )
        scale_index = batch.token_scale_index.clamp(
            min=0, max=len(BA_IEG_TOKEN_SCALES) - 1
        )

        causal_input = torch.cat(
            (masked_values, feature_mask.to(dtype=values.dtype), time_features),
            dim=-1,
        )
        causal_hidden = self.causal_value_projection(causal_input) + self.causal_scale_embedding(
            scale_index
        )
        causal_hidden = torch.where(
            causal_active.unsqueeze(-1),
            self.causal_input_norm(causal_hidden),
            torch.zeros_like(causal_hidden),
        )

        if self.allow_heuristic_phase_posterior:
            phase = torch.where(
                batch.token_phase_context_mask.unsqueeze(-1),
                batch.phase_posterior,
                torch.zeros_like(batch.phase_posterior),
            )
        else:
            phase = torch.zeros(
                (*values.shape[:2], len(BA_IEG_PHASE_STATES)),
                dtype=values.dtype,
                device=values.device,
            )
        offline_input = torch.cat(
            (
                masked_values,
                feature_mask.to(dtype=values.dtype),
                time_features,
                phase.to(dtype=values.dtype),
                batch.token_future_sample_access.unsqueeze(-1).to(dtype=values.dtype),
            ),
            dim=-1,
        )
        offline_hidden = self.offline_value_projection(
            offline_input
        ) + self.offline_scale_embedding(scale_index)
        offline_hidden = torch.where(
            full_active.unsqueeze(-1),
            self.offline_input_norm(offline_hidden),
            torch.zeros_like(offline_hidden),
        )
        return full, causal_hidden, offline_hidden, causal_active, full_active

    def _validate_context(
        self,
        batch: BAIEGCollatedEventBatch,
        context: BAIEGSegmentalBoundaryContext,
    ) -> None:
        if not isinstance(batch, BAIEGCollatedEventBatch):
            raise TypeError("segmental state model requires a registered event batch")
        if not isinstance(context, BAIEGSegmentalBoundaryContext):
            raise TypeError("segmental state model requires a registered boundary context")
        if batch.input_batch_sha256 != _collated_batch_input_sha256(batch):
            raise ValueError("BA-IEG collated model inputs changed after registration")
        context.verify_integrity()
        if context.source_input_batch_sha256 != batch.input_batch_sha256:
            raise ValueError("segmental context is bound to another input batch")
        if context.event_ids != batch.event_ids or context.source_event_receipt_sha256s != batch.input_event_receipt_sha256s:
            raise ValueError("segmental context event identity/order drifted")

    def forward(
        self,
        batch: BAIEGCollatedEventBatch,
        context: BAIEGSegmentalBoundaryContext,
    ) -> BAIEGPermissionSplitSegmentalStateOutput:
        self._validate_context(batch, context)
        full, causal_token_hidden, offline_token_hidden, causal_active, full_active = self._project_token_lanes(
            batch
        )
        batch_size = len(batch.event_ids)
        device = batch.token_values.device
        dtype = batch.token_values.dtype
        time_dtype = batch.token_time_bounds_seconds.dtype
        duration_location = self.duration_location_log_seconds.to(
            device=device, dtype=dtype
        )
        duration_scale = (
            F.softplus(self.duration_scale_unconstrained).to(device=device, dtype=dtype)
            + 1e-4
        )
        minimum_duration = self.minimum_state_duration_seconds.to(
            device=device, dtype=dtype
        )

        event_results: list[dict[str, object]] = []
        for batch_index in range(batch_size):
            support = [
                (float(row[0]), float(row[1]))
                for row in context.observed_support_intervals_seconds[
                    batch_index, context.observed_support_mask[batch_index]
                ].detach().cpu()
            ]
            gaps = [
                (float(row[0]), float(row[1]))
                for row in context.quality_gap_intervals_seconds[
                    batch_index, context.quality_gap_mask[batch_index]
                ].detach().cpu()
            ]
            full_indices = [
                int(index)
                for index in torch.nonzero(
                    full_active[batch_index], as_tuple=False
                ).flatten().detach().cpu()
            ]
            causal_indices = [
                int(index)
                for index in torch.nonzero(
                    causal_active[batch_index], as_tuple=False
                ).flatten().detach().cpu()
            ]
            cells, declared_opportunity = _event_physical_lattice(
                token_bounds=batch.token_time_bounds_seconds[batch_index],
                active_indices=full_indices,
                support_intervals=support,
                quality_gaps=gaps,
                tolerance_seconds=self.time_tolerance_seconds,
            )
            if not cells:
                raise ValueError("segmental physical lattice has no positive-width cell")

            causal_times = _unique_sorted_times(
                [
                    float(batch.token_time_bounds_seconds[batch_index, index, 1])
                    for index in causal_indices
                ],
                tolerance_seconds=self.time_tolerance_seconds,
            )
            causal_group_hidden: list[torch.Tensor] = []
            for time in causal_times:
                selected = [
                    index
                    for index in causal_indices
                    if abs(
                        float(batch.token_time_bounds_seconds[batch_index, index, 1])
                        - time
                    )
                    <= self.time_tolerance_seconds
                ]
                causal_group_hidden.append(
                    _hierarchical_pool_rows(
                        causal_token_hidden[batch_index],
                        selected,
                        token_unit_index=batch.token_unit_index[batch_index],
                        token_scale_index=batch.token_scale_index[batch_index],
                        unit_view_index=batch.unit_view_index[batch_index],
                        view_reference_family_index=(
                            batch.view_reference_family_index[batch_index]
                        ),
                        unit_reference_matrix=batch.unit_reference_matrix[batch_index],
                        row_weights=None,
                        relative_reference_threshold=self.relative_reference_threshold,
                        maximum_local_footprint=self.maximum_local_footprint,
                    )
                )
            if causal_group_hidden:
                causal_sequence = torch.stack(causal_group_hidden).unsqueeze(0)
                causal_sequence, _ = self.causal_prefix_gru(
                    self.dropout(causal_sequence)
                )
                causal_sequence = causal_sequence.squeeze(0)
                causal_logits = self.causal_onset_hazard_head(
                    causal_sequence
                ).squeeze(-1)
                left_logits = self.causal_left_state_head(causal_sequence[0])
                causal_type_modifiers = self.causal_transition_type_head(
                    causal_sequence
                )
            else:
                causal_sequence = torch.zeros(
                    (0, self.hidden_dim), dtype=dtype, device=device
                )
                causal_logits = torch.zeros(0, dtype=dtype, device=device)
                left_logits = self.causal_empty_left_logits.to(dtype=dtype)
                causal_type_modifiers = torch.zeros(
                    (0, len(BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES)),
                    dtype=dtype,
                    device=device,
                )

            causal_opportunity = torch.tensor(
                [
                    any(
                        stop - self.time_tolerance_seconds <= time <= stop + self.time_tolerance_seconds
                        and _overlap_length(start, stop, support)
                        - _overlap_length(start, stop, gaps)
                        > self.time_tolerance_seconds
                        for start, stop in cells
                    )
                    for time in causal_times
                ],
                dtype=torch.bool,
                device=device,
            )
            typed_keys, typed_aliases = _event_typed_unit_inventory(
                batch,
                batch_index,
                relative_reference_threshold=self.relative_reference_threshold,
                maximum_local_footprint=self.maximum_local_footprint,
            )
            typed_source_unit_mask = torch.zeros(
                (len(typed_keys), int(batch.unit_row_mask.shape[1])),
                dtype=torch.bool,
                device=device,
            )
            typed_reference_family_mask = torch.zeros(
                (len(typed_keys), len(BA_IEG_REFERENCE_FAMILIES)),
                dtype=torch.bool,
                device=device,
            )
            for typed_index, key in enumerate(typed_keys):
                for unit_index in typed_aliases[key]:
                    typed_source_unit_mask[typed_index, unit_index] = True
                    view_index = int(
                        batch.unit_view_index[batch_index, unit_index]
                    )
                    reference_family_index = int(
                        batch.view_reference_family_index[
                            batch_index, view_index
                        ]
                    )
                    typed_reference_family_mask[
                        typed_index, reference_family_index
                    ] = True

            typed_local_by_time: list[torch.Tensor] = []
            typed_mask_by_time: list[torch.Tensor] = []
            causal_boundary_bounds: list[tuple[float, float]] = []
            for causal_index, time in enumerate(causal_times):
                matching_bounds = [
                    (start, stop)
                    for start, stop in cells
                    if abs(stop - time) <= self.time_tolerance_seconds
                ]
                if len(matching_bounds) != 1:
                    raise RuntimeError(
                        "causal time does not map to exactly one physical lattice cell"
                    )
                causal_boundary_bounds.append(matching_bounds[0])
                local_rows: list[torch.Tensor] = []
                local_mask = torch.zeros(
                    len(typed_keys), dtype=torch.bool, device=device
                )
                for typed_index, key in enumerate(typed_keys):
                    aliases = frozenset(typed_aliases[key])
                    selected = [
                        index
                        for index in causal_indices
                        if int(batch.token_unit_index[batch_index, index])
                        in aliases
                        and bool(
                            batch.token_positive_onset_mask[
                                batch_index, index
                            ]
                        )
                        and abs(
                            float(
                                batch.token_time_bounds_seconds[
                                    batch_index, index, 1
                                ]
                            )
                            - time
                        )
                        <= self.time_tolerance_seconds
                    ]
                    local_mask[typed_index] = bool(selected) and bool(
                        causal_opportunity[causal_index]
                    )
                    local_rows.append(
                        _hierarchical_pool_rows(
                            causal_token_hidden[batch_index],
                            selected if bool(local_mask[typed_index]) else [],
                            token_unit_index=batch.token_unit_index[batch_index],
                            token_scale_index=batch.token_scale_index[batch_index],
                            unit_view_index=batch.unit_view_index[batch_index],
                            view_reference_family_index=(
                                batch.view_reference_family_index[batch_index]
                            ),
                            unit_reference_matrix=(
                                batch.unit_reference_matrix[batch_index]
                            ),
                            row_weights=None,
                            relative_reference_threshold=(
                                self.relative_reference_threshold
                            ),
                            maximum_local_footprint=(
                                self.maximum_local_footprint
                            ),
                        )
                    )
                if local_rows:
                    typed_local_by_time.append(torch.stack(local_rows))
                else:
                    typed_local_by_time.append(
                        torch.zeros(
                            (0, self.hidden_dim), dtype=dtype, device=device
                        )
                    )
                typed_mask_by_time.append(local_mask)
            if causal_times:
                typed_local_hidden = torch.stack(typed_local_by_time)
                typed_time_mask = torch.stack(typed_mask_by_time)
            else:
                typed_local_hidden = torch.zeros(
                    (0, len(typed_keys), self.hidden_dim),
                    dtype=dtype,
                    device=device,
                )
                typed_time_mask = torch.zeros(
                    (0, len(typed_keys)), dtype=torch.bool, device=device
                )
            hazard_mass, hazard_survival = _discrete_hazard_distribution(
                causal_logits, causal_opportunity
            )
            if bool(context.left_censoring_possible[batch_index]):
                left_state_probability = torch.softmax(left_logits, dim=0)
            else:
                left_state_probability = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=dtype, device=device
                )
            observed_onset_mass = left_state_probability[0] * hazard_mass
            left_censor_mass = left_state_probability[1:]
            no_onset_mass = left_state_probability[0] * hazard_survival

            cell_hidden: list[torch.Tensor] = []
            cell_has_evidence: list[bool] = []
            effective_opportunity: list[float] = []
            for (cell_start, cell_stop), declared in zip(cells, declared_opportunity):
                overlap_weights = torch.zeros(
                    int(batch.token_values.shape[1]), dtype=dtype, device=device
                )
                selected: list[int] = []
                for index in full_indices:
                    token_start = float(
                        batch.token_time_bounds_seconds[batch_index, index, 0]
                    )
                    token_stop = float(
                        batch.token_time_bounds_seconds[batch_index, index, 1]
                    )
                    overlap = max(
                        0.0,
                        min(cell_stop, token_stop) - max(cell_start, token_start),
                    )
                    if overlap > self.time_tolerance_seconds:
                        selected.append(index)
                        overlap_weights[index] = overlap
                has_evidence = bool(selected) and declared > self.time_tolerance_seconds
                cell_has_evidence.append(has_evidence)
                effective_opportunity.append(declared if has_evidence else 0.0)
                cell_hidden.append(
                    _hierarchical_pool_rows(
                        offline_token_hidden[batch_index],
                        selected if has_evidence else [],
                        token_unit_index=batch.token_unit_index[batch_index],
                        token_scale_index=batch.token_scale_index[batch_index],
                        unit_view_index=batch.unit_view_index[batch_index],
                        view_reference_family_index=(
                            batch.view_reference_family_index[batch_index]
                        ),
                        unit_reference_matrix=batch.unit_reference_matrix[batch_index],
                        row_weights=overlap_weights,
                        relative_reference_threshold=self.relative_reference_threshold,
                        maximum_local_footprint=self.maximum_local_footprint,
                    )
                )
            offline_cell_input = torch.stack(cell_hidden)
            # The only cross-lane edge is explicitly detached.
            for causal_index, time in enumerate(causal_times):
                matching_cells = [
                    index
                    for index, (_, stop) in enumerate(cells)
                    if abs(stop - time) <= self.time_tolerance_seconds
                ]
                for cell_index in matching_cells:
                    offline_cell_input[cell_index] = (
                        offline_cell_input[cell_index]
                        + self.causal_to_offline_projection(
                            causal_sequence[causal_index].detach()
                        )
                    )

            active_cells = [
                index for index, active in enumerate(cell_has_evidence) if active
            ]
            offline_hidden = torch.zeros(
                (len(cells), self.hidden_dim), dtype=dtype, device=device
            )
            if active_cells:
                active_index_tensor = torch.tensor(
                    active_cells, dtype=torch.long, device=device
                )
                active_sequence = offline_cell_input[active_index_tensor].unsqueeze(0)
                forward_hidden, _ = self.offline_forward_gru(
                    self.dropout(active_sequence)
                )
                reversed_hidden, _ = self.offline_backward_gru(
                    self.dropout(torch.flip(active_sequence, dims=(1,)))
                )
                backward_hidden = torch.flip(reversed_hidden, dims=(1,))
                combined = self.offline_bidirectional_projection(
                    torch.cat((forward_hidden, backward_hidden), dim=-1)
                ).squeeze(0)
                offline_hidden[active_index_tensor] = combined

            cell_mask = torch.tensor(
                cell_has_evidence, dtype=torch.bool, device=device
            )
            opportunity_duration = torch.tensor(
                effective_opportunity, dtype=dtype, device=device
            )
            physical_duration = torch.tensor(
                [stop - start for start, stop in cells],
                dtype=dtype,
                device=device,
            )
            quality_gap_mask = opportunity_duration <= self.time_tolerance_seconds
            state_logits = self.offline_state_head(offline_hidden)
            state_logits = torch.where(
                cell_mask.unsqueeze(-1), state_logits, torch.zeros_like(state_logits)
            )
            state_log_prob = torch.log_softmax(state_logits, dim=-1)
            course_logits = self.offline_course_transition_head(offline_hidden)
            offset_logits = self.offline_offset_hazard_head(
                offline_hidden
            ).squeeze(-1)
            offset_hazard_mass, offset_survival = _discrete_hazard_distribution(
                offset_logits, cell_mask
            )
            if active_cells:
                final_hidden = offline_hidden[active_cells[-1]]
                right_logits = self.offline_right_status_head(final_hidden)
                event_logits = self.offline_event_presence_head(final_hidden)
            else:
                right_logits = self.offline_empty_right_logits.to(dtype=dtype)
                event_logits = self.offline_empty_event_logits.to(dtype=dtype)
            if bool(context.right_censoring_possible[batch_index]):
                right_probability = torch.softmax(right_logits, dim=0)
            else:
                right_probability = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=dtype, device=device
                )
            observed_offset_mass = right_probability[0] * offset_hazard_mass
            right_censor_mass = right_probability[1:]
            no_offset_mass = right_probability[0] * offset_survival
            event_log_probability = torch.log_softmax(event_logits, dim=0)

            invalid = torch.tensor(-1e9, dtype=dtype, device=device)
            if bool(context.left_censoring_possible[batch_index]):
                start_scores = torch.stack(
                    (
                        torch.log(left_state_probability[0].clamp_min(1e-12)),
                        torch.log(left_state_probability[1].clamp_min(1e-12)),
                        torch.log(left_state_probability[2].clamp_min(1e-12)),
                        invalid,
                    )
                )
            else:
                start_scores = torch.stack(
                    (torch.zeros((), dtype=dtype, device=device), invalid, invalid, invalid)
                )
            if bool(context.right_censoring_possible[batch_index]):
                end_scores = torch.stack(
                    (
                        torch.log(right_probability[0].clamp_min(1e-12)),
                        torch.log(right_probability[1].clamp_min(1e-12)),
                        torch.log(right_probability[2].clamp_min(1e-12)),
                        torch.log(right_probability[0].clamp_min(1e-12)),
                    )
                )
            else:
                end_scores = torch.stack(
                    (torch.zeros((), dtype=dtype, device=device), invalid, invalid, torch.zeros((), dtype=dtype, device=device))
                )

            transition_scores = torch.zeros(
                (len(cells), len(BA_IEG_SEGMENTAL_TRANSITIONS)),
                dtype=dtype,
                device=device,
            )
            transition_mask = torch.zeros_like(transition_scores, dtype=torch.bool)
            causal_edge_indices = sorted(BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES)
            for causal_index, time in enumerate(causal_times):
                matching = [
                    index
                    for index, (_, stop) in enumerate(cells)
                    if abs(stop - time) <= self.time_tolerance_seconds
                ]
                if not bool(causal_opportunity[causal_index]):
                    continue
                for cell_index in matching:
                    for local_index, edge_index in enumerate(causal_edge_indices):
                        transition_scores[cell_index, edge_index] = (
                            torch.log(observed_onset_mass[causal_index].clamp_min(1e-12))
                            + causal_type_modifiers[causal_index, local_index]
                        )
                        transition_mask[cell_index, edge_index] = True
            offline_edge_indices = (1, 2, 4, 5)
            for local_index, edge_index in enumerate(offline_edge_indices):
                transition_scores[:, edge_index] = F.logsigmoid(
                    course_logits[:, local_index]
                )
                transition_mask[:, edge_index] = cell_mask
            transition_scores[:, 2] = torch.log(
                observed_offset_mass.clamp_min(1e-12)
            ) + F.logsigmoid(course_logits[:, 1])
            transition_scores[:, 4] = torch.log(
                observed_offset_mass.clamp_min(1e-12)
            ) + F.logsigmoid(course_logits[:, 2])

            effective_maximum_segments = min(self.maximum_segments, len(cells))
            duration_log_scores = build_lognormal_segment_duration_log_scores_v1(
                physical_duration=physical_duration,
                duration_location_log_seconds=duration_location,
                duration_scale_log_seconds=duration_scale,
                minimum_state_duration_seconds=minimum_duration,
            )
            exact_potentials = SegmentalPotentialsV1(
                emission_log_density=state_log_prob,
                opportunity_duration=opportunity_duration,
                physical_duration=physical_duration,
                transition_log_scores=transition_scores,
                transition_mask=transition_mask,
                start_log_scores=start_scores,
                end_log_scores=end_scores,
                event_log_score=event_log_probability[1],
                no_event_log_score=event_log_probability[0],
                segment_duration_log_scores=duration_log_scores,
                maximum_segments=effective_maximum_segments,
                left_censoring_possible=bool(
                    context.left_censoring_possible[batch_index]
                ),
                right_censoring_possible=bool(
                    context.right_censoring_possible[batch_index]
                ),
            )
            exact_posterior = run_exact_segmental_forward_backward_v1(
                exact_potentials
            )
            log_partition = exact_posterior.exact_log_partition
            paths = _top_k_general_segmental_paths(
                state_emission_log_prob=state_log_prob,
                opportunity_duration_seconds=opportunity_duration,
                physical_duration_seconds=physical_duration,
                transition_log_scores=transition_scores,
                transition_mask=transition_mask,
                start_log_scores=start_scores,
                end_log_scores=end_scores,
                event_log_score=event_log_probability[1],
                no_event_log_score=event_log_probability[0],
                duration_location=duration_location,
                duration_scale=duration_scale,
                minimum_duration_seconds=minimum_duration,
                left_censoring_possible=bool(
                    context.left_censoring_possible[batch_index]
                ),
                right_censoring_possible=bool(
                    context.right_censoring_possible[batch_index]
                ),
                maximum_segments=effective_maximum_segments,
                maximum_paths=self.maximum_paths,
            )
            if paths:
                path_scores = torch.stack([path.score for path in paths])
                path_weights = torch.softmax(path_scores, dim=0)
                path_posterior_mass = torch.exp(path_scores - log_partition)
                retained_mass = path_posterior_mass.sum().clamp(0.0, 1.0)
            else:
                path_scores = torch.zeros(0, dtype=dtype, device=device)
                path_weights = torch.zeros(0, dtype=dtype, device=device)
                path_posterior_mass = torch.zeros(0, dtype=dtype, device=device)
                retained_mass = torch.zeros((), dtype=dtype, device=device)

            causal_started = torch.zeros(len(cells), dtype=dtype, device=device)
            for cell_index, (_, stop) in enumerate(cells):
                earlier = [
                    index
                    for index, time in enumerate(causal_times)
                    if time <= stop + self.time_tolerance_seconds
                ]
                causal_started[cell_index] = left_censor_mass.sum()
                if earlier:
                    causal_started[cell_index] = (
                        causal_started[cell_index]
                        + observed_onset_mass[
                            torch.tensor(earlier, dtype=torch.long, device=device)
                        ].sum()
                    )
            offline_non_background = 1.0 - torch.softmax(state_logits, dim=-1)[:, 0]
            if bool(cell_mask.any()):
                consistency = F.binary_cross_entropy(
                    offline_non_background[cell_mask],
                    causal_started.detach()[cell_mask].clamp(0.0, 1.0),
                    reduction="mean",
                )
            else:
                consistency = torch.zeros((), dtype=dtype, device=device)

            event_results.append(
                {
                    "cells": cells,
                    "causal_times": causal_times,
                    "causal_boundary_bounds": causal_boundary_bounds,
                    "causal_mask": causal_opportunity,
                    "causal_group_hidden": causal_sequence,
                    "typed_keys": typed_keys,
                    "typed_local_hidden": typed_local_hidden,
                    "typed_time_mask": typed_time_mask,
                    "typed_source_unit_mask": typed_source_unit_mask,
                    "typed_reference_family_mask": typed_reference_family_mask,
                    "causal_logits": causal_logits,
                    "causal_mass": observed_onset_mass,
                    "left_mass": left_censor_mass,
                    "no_onset_mass": no_onset_mass,
                    "cell_mask": cell_mask,
                    "opportunity_duration": opportunity_duration,
                    "physical_duration": physical_duration,
                    "quality_gap_mask": quality_gap_mask,
                    "state_logits": state_logits,
                    "state_log_prob": state_log_prob,
                    "offset_logits": offset_logits,
                    "offset_mass": observed_offset_mass,
                    "right_mass": right_censor_mass,
                    "no_offset_mass": no_offset_mass,
                    "split_logits": self.offline_split_reentry_review_head(
                        offline_hidden
                    ).squeeze(-1),
                    "transition_scores": transition_scores,
                    "transition_mask": transition_mask,
                    "start_scores": start_scores,
                    "end_scores": end_scores,
                    "event_scores": event_log_probability,
                    "log_partition": log_partition,
                    "exact_posterior": exact_posterior,
                    "paths": paths,
                    "path_scores": path_scores,
                    "path_posterior_mass": path_posterior_mass,
                    "path_weights": path_weights,
                    "retained_mass": retained_mass,
                    "consistency": consistency,
                }
            )

        maximum_causal = max(1, *(len(result["causal_times"]) for result in event_results))
        maximum_typed_units = max(
            1, *(len(result["typed_keys"]) for result in event_results)
        )
        maximum_analysis_units = int(batch.unit_row_mask.shape[1])
        maximum_cells = max(len(result["cells"]) for result in event_results)
        maximum_transitions = self.maximum_segments - 1
        causal_times_out = torch.zeros((batch_size, maximum_causal), dtype=time_dtype, device=device)
        causal_mask_out = torch.zeros((batch_size, maximum_causal), dtype=torch.bool, device=device)
        causal_logits_out = torch.zeros((batch_size, maximum_causal), dtype=dtype, device=device)
        causal_mass_out = torch.zeros_like(causal_logits_out)
        causal_boundary_bounds_out = torch.zeros(
            (batch_size, maximum_causal, 2),
            dtype=time_dtype,
            device=device,
        )
        causal_group_hidden_out = torch.zeros(
            (batch_size, maximum_causal, self.hidden_dim),
            dtype=dtype,
            device=device,
        )
        typed_local_hidden_out = torch.zeros(
            (
                batch_size,
                maximum_causal,
                maximum_typed_units,
                self.hidden_dim,
            ),
            dtype=dtype,
            device=device,
        )
        typed_time_mask_out = torch.zeros(
            (batch_size, maximum_causal, maximum_typed_units),
            dtype=torch.bool,
            device=device,
        )
        typed_unit_mask_out = torch.zeros(
            (batch_size, maximum_typed_units),
            dtype=torch.bool,
            device=device,
        )
        typed_kind_out = torch.full(
            (batch_size, maximum_typed_units),
            -1,
            dtype=torch.long,
            device=device,
        )
        typed_electrode_out = torch.full_like(typed_kind_out, -1)
        typed_lead_endpoint_out = torch.full(
            (batch_size, maximum_typed_units, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        typed_source_unit_mask_out = torch.zeros(
            (batch_size, maximum_typed_units, maximum_analysis_units),
            dtype=torch.bool,
            device=device,
        )
        typed_reference_family_mask_out = torch.zeros(
            (
                batch_size,
                maximum_typed_units,
                len(BA_IEG_REFERENCE_FAMILIES),
            ),
            dtype=torch.bool,
            device=device,
        )
        left_mass_out = torch.zeros((batch_size, 2), dtype=dtype, device=device)
        no_onset_out = torch.zeros(batch_size, dtype=dtype, device=device)
        cell_bounds_out = torch.zeros((batch_size, maximum_cells, 2), dtype=time_dtype, device=device)
        cell_mask_out = torch.zeros((batch_size, maximum_cells), dtype=torch.bool, device=device)
        opportunity_out = torch.zeros((batch_size, maximum_cells), dtype=dtype, device=device)
        physical_duration_out = torch.zeros_like(opportunity_out)
        quality_gap_out = torch.zeros_like(cell_mask_out)
        state_logits_out = torch.zeros((batch_size, maximum_cells, len(BA_IEG_PHASE_STATES)), dtype=dtype, device=device)
        state_log_prob_out = torch.zeros_like(state_logits_out)
        offset_logits_out = torch.zeros((batch_size, maximum_cells), dtype=dtype, device=device)
        offset_mass_out = torch.zeros_like(offset_logits_out)
        right_mass_out = torch.zeros((batch_size, 2), dtype=dtype, device=device)
        no_offset_out = torch.zeros(batch_size, dtype=dtype, device=device)
        split_logits_out = torch.zeros_like(offset_logits_out)
        transition_scores_out = torch.zeros((batch_size, maximum_cells, len(BA_IEG_SEGMENTAL_TRANSITIONS)), dtype=dtype, device=device)
        transition_mask_out = torch.zeros_like(transition_scores_out, dtype=torch.bool)
        start_scores_out = torch.zeros((batch_size, len(BA_IEG_PHASE_STATES)), dtype=dtype, device=device)
        end_scores_out = torch.zeros_like(start_scores_out)
        event_scores_out = torch.zeros((batch_size, 2), dtype=dtype, device=device)
        log_partition_out = torch.full((batch_size,), -torch.inf, dtype=dtype, device=device)
        exact_state_out = torch.zeros(
            (batch_size, maximum_cells, len(BA_IEG_PHASE_STATES)),
            dtype=dtype,
            device=device,
        )
        exact_transition_out = torch.zeros(
            (batch_size, maximum_cells, len(BA_IEG_SEGMENTAL_TRANSITIONS)),
            dtype=dtype,
            device=device,
        )
        exact_start_out = torch.zeros(
            (batch_size, len(BA_IEG_PHASE_STATES)), dtype=dtype, device=device
        )
        exact_end_out = torch.zeros_like(exact_start_out)
        exact_segment_count_out = torch.zeros(
            (batch_size, self.maximum_segments), dtype=dtype, device=device
        )
        exact_onset_out = torch.zeros(
            (batch_size, maximum_cells), dtype=dtype, device=device
        )
        exact_primary_onset_out = torch.zeros_like(exact_onset_out)
        exact_secondary_onset_out = torch.zeros_like(exact_onset_out)
        exact_s0_onset_out = torch.zeros_like(exact_onset_out)
        exact_s3_reentry_onset_out = torch.zeros_like(exact_onset_out)
        exact_offset_out = torch.zeros_like(exact_onset_out)
        exact_event_out = torch.zeros(batch_size, dtype=dtype, device=device)
        exact_null_out = torch.zeros_like(exact_event_out)
        exact_left_censor_out = torch.zeros_like(exact_event_out)
        exact_right_censor_out = torch.zeros_like(exact_event_out)
        exact_both_censor_out = torch.zeros_like(exact_event_out)
        exact_recurrent_event_out = torch.zeros_like(exact_event_out)
        exact_expected_bout_out = torch.zeros_like(exact_event_out)
        exact_expected_recurrent_bout_out = torch.zeros_like(exact_event_out)
        path_start_out = torch.zeros((batch_size, self.maximum_paths), dtype=torch.long, device=device)
        path_end_out = torch.zeros_like(path_start_out)
        path_transition_times_out = torch.zeros((batch_size, self.maximum_paths, maximum_transitions), dtype=time_dtype, device=device)
        path_transition_mask_out = torch.zeros((batch_size, self.maximum_paths, maximum_transitions), dtype=torch.bool, device=device)
        path_transition_type_out = torch.full((batch_size, self.maximum_paths, maximum_transitions), -1, dtype=torch.long, device=device)
        path_segment_bounds_out = torch.zeros((batch_size, self.maximum_paths, self.maximum_segments, 2), dtype=time_dtype, device=device)
        path_segment_state_out = torch.full((batch_size, self.maximum_paths, self.maximum_segments), -1, dtype=torch.long, device=device)
        path_segment_mask_out = torch.zeros((batch_size, self.maximum_paths, self.maximum_segments), dtype=torch.bool, device=device)
        path_cycle_count_out = torch.zeros((batch_size, self.maximum_paths), dtype=torch.long, device=device)
        path_scores_out = torch.zeros((batch_size, self.maximum_paths), dtype=dtype, device=device)
        path_posterior_mass_out = torch.zeros_like(path_scores_out)
        path_weights_out = torch.zeros_like(path_scores_out)
        path_mask_out = torch.zeros_like(path_scores_out, dtype=torch.bool)
        retained_mass_out = torch.zeros(batch_size, dtype=dtype, device=device)
        state_marginals_out = torch.zeros((batch_size, maximum_cells, len(BA_IEG_PHASE_STATES)), dtype=dtype, device=device)
        consistency_out = torch.zeros(batch_size, dtype=dtype, device=device)

        for batch_index, result in enumerate(event_results):
            causal_count = len(result["causal_times"])
            if causal_count:
                causal_times_out[batch_index, :causal_count] = torch.tensor(result["causal_times"], dtype=time_dtype, device=device)
                causal_mask_out[batch_index, :causal_count] = result["causal_mask"]
                causal_logits_out[batch_index, :causal_count] = result["causal_logits"]
                causal_mass_out[batch_index, :causal_count] = result["causal_mass"]
                causal_boundary_bounds_out[batch_index, :causal_count] = torch.tensor(
                    result["causal_boundary_bounds"],
                    dtype=time_dtype,
                    device=device,
                )
                causal_group_hidden_out[batch_index, :causal_count] = torch.where(
                    result["causal_mask"].unsqueeze(-1),
                    result["causal_group_hidden"],
                    torch.zeros_like(result["causal_group_hidden"]),
                )
            typed_count = len(result["typed_keys"])
            if typed_count:
                typed_time_mask_out[
                    batch_index, :causal_count, :typed_count
                ] = result["typed_time_mask"]
                typed_local_hidden_out[
                    batch_index, :causal_count, :typed_count
                ] = torch.where(
                    result["typed_time_mask"].unsqueeze(-1),
                    result["typed_local_hidden"],
                    torch.zeros_like(result["typed_local_hidden"]),
                )
                typed_unit_mask_out[batch_index, :typed_count] = result[
                    "typed_time_mask"
                ].any(dim=0)
                typed_source_unit_mask_out[
                    batch_index, :typed_count
                ] = result["typed_source_unit_mask"]
                typed_reference_family_mask_out[
                    batch_index, :typed_count
                ] = result["typed_reference_family_mask"]
                for typed_index, key in enumerate(result["typed_keys"]):
                    kind_index, first, second = key
                    typed_kind_out[batch_index, typed_index] = kind_index
                    if kind_index == BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index(
                        "physical_electrode"
                    ):
                        typed_electrode_out[batch_index, typed_index] = first
                    else:
                        typed_lead_endpoint_out[
                            batch_index, typed_index
                        ] = torch.tensor(
                            (first, second), dtype=torch.long, device=device
                        )
            left_mass_out[batch_index] = result["left_mass"]
            no_onset_out[batch_index] = result["no_onset_mass"]
            cell_count = len(result["cells"])
            cell_bounds_out[batch_index, :cell_count] = torch.tensor(result["cells"], dtype=time_dtype, device=device)
            for name, target in (
                ("cell_mask", cell_mask_out),
                ("opportunity_duration", opportunity_out),
                ("physical_duration", physical_duration_out),
                ("quality_gap_mask", quality_gap_out),
                ("state_logits", state_logits_out),
                ("state_log_prob", state_log_prob_out),
                ("offset_logits", offset_logits_out),
                ("offset_mass", offset_mass_out),
                ("split_logits", split_logits_out),
                ("transition_scores", transition_scores_out),
                ("transition_mask", transition_mask_out),
            ):
                target[batch_index, :cell_count] = result[name]
            right_mass_out[batch_index] = result["right_mass"]
            no_offset_out[batch_index] = result["no_offset_mass"]
            start_scores_out[batch_index] = result["start_scores"]
            end_scores_out[batch_index] = result["end_scores"]
            event_scores_out[batch_index] = result["event_scores"]
            log_partition_out[batch_index] = result["log_partition"]
            exact_posterior: ExactSegmentalForwardBackwardOutputV1 = result[
                "exact_posterior"
            ]
            exact_state_out[batch_index, :cell_count] = (
                exact_posterior.state_marginal
            )
            exact_transition_out[batch_index, :cell_count] = (
                exact_posterior.transition_marginal
            )
            exact_start_out[batch_index] = exact_posterior.start_state_marginal
            exact_end_out[batch_index] = exact_posterior.end_state_marginal
            segment_count = int(exact_posterior.segment_count_marginal.numel())
            exact_segment_count_out[batch_index, :segment_count] = (
                exact_posterior.segment_count_marginal
            )
            exact_onset_out[batch_index, :cell_count] = (
                exact_posterior.onset_boundary_mass
            )
            exact_primary_onset_out[batch_index, :cell_count] = (
                exact_posterior.primary_onset_boundary_mass
            )
            exact_secondary_onset_out[batch_index, :cell_count] = (
                exact_posterior.secondary_onset_boundary_mass
            )
            exact_s0_onset_out[batch_index, :cell_count] = (
                exact_posterior.s0_onset_boundary_mass
            )
            exact_s3_reentry_onset_out[batch_index, :cell_count] = (
                exact_posterior.s3_reentry_onset_boundary_mass
            )
            exact_offset_out[batch_index, :cell_count] = (
                exact_posterior.offset_boundary_mass
            )
            exact_event_out[batch_index] = exact_posterior.event_mass
            exact_null_out[batch_index] = exact_posterior.null_mass
            exact_left_censor_out[batch_index] = exact_posterior.left_censor_mass
            exact_right_censor_out[batch_index] = (
                exact_posterior.right_censor_mass
            )
            exact_both_censor_out[batch_index] = exact_posterior.both_censor_mass
            exact_recurrent_event_out[batch_index] = (
                exact_posterior.recurrent_event_mass
            )
            exact_expected_bout_out[batch_index] = (
                exact_posterior.expected_event_bout_count
            )
            exact_expected_recurrent_bout_out[batch_index] = (
                exact_posterior.expected_recurrent_bout_count
            )
            retained_mass_out[batch_index] = result["retained_mass"]
            consistency_out[batch_index] = result["consistency"]
            paths: list[_GeneralCompletedPath] = result["paths"]
            count = len(paths)
            if count:
                path_mask_out[batch_index, :count] = True
                path_scores_out[batch_index, :count] = result["path_scores"]
                path_posterior_mass_out[batch_index, :count] = result[
                    "path_posterior_mass"
                ]
                path_weights_out[batch_index, :count] = result["path_weights"]
            for path_index, path in enumerate(paths):
                path_start_out[batch_index, path_index] = path.start_state
                path_end_out[batch_index, path_index] = path.end_state
                transition_count = len(path.transition_indices)
                if transition_count:
                    boundary_indices = torch.tensor(path.transition_indices, dtype=torch.long, device=device)
                    path_transition_times_out[batch_index, path_index, :transition_count] = cell_bounds_out[batch_index, boundary_indices, 1]
                    path_transition_mask_out[batch_index, path_index, :transition_count] = True
                    path_transition_type_out[batch_index, path_index, :transition_count] = torch.tensor(path.transition_edge_indices, dtype=torch.long, device=device)
                segment_starts = (0,) + tuple(index + 1 for index in path.transition_indices)
                segment_ends = tuple(path.transition_indices) + (cell_count - 1,)
                for segment_index, (state, start, end) in enumerate(zip(path.state_sequence, segment_starts, segment_ends)):
                    path_segment_bounds_out[batch_index, path_index, segment_index, 0] = cell_bounds_out[batch_index, start, 0]
                    path_segment_bounds_out[batch_index, path_index, segment_index, 1] = cell_bounds_out[batch_index, end, 1]
                    path_segment_state_out[batch_index, path_index, segment_index] = state
                    path_segment_mask_out[batch_index, path_index, segment_index] = True
                    state_marginals_out[batch_index, start : end + 1, state] += result["path_weights"][path_index]
                event_bout_count = int(path.start_state in (1, 2)) + sum(
                    edge in BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES
                    for edge in path.transition_edge_indices
                )
                path_cycle_count_out[batch_index, path_index] = max(
                    0, event_bout_count - 1
                )

        event_evaluable = torch.isfinite(log_partition_out) & path_mask_out.any(dim=1) & cell_mask_out.any(dim=1)
        causal_trace = BAIEGCausalTypedUnitTrace(
            source_input_batch_sha256=batch.input_batch_sha256,
            event_ids=batch.event_ids,
            recording_ids=batch.recording_ids,
            source_event_receipt_sha256s=batch.input_event_receipt_sha256s,
            identity_roster_sha256=ba_ieg_event_identity_roster_sha256(
                source_input_batch_sha256=batch.input_batch_sha256,
                event_ids=batch.event_ids,
                recording_ids=batch.recording_ids,
                source_event_receipt_sha256s=(
                    batch.input_event_receipt_sha256s
                ),
            ),
            implementation_id=BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
            group_times_seconds=causal_times_out,
            group_boundary_bounds_seconds=causal_boundary_bounds_out,
            group_mask=causal_mask_out,
            global_onset_boundary_mass=causal_mass_out,
            global_left_censor_state_mass=left_mass_out,
            global_no_onset_within_support_mass=no_onset_out,
            global_group_hidden=causal_group_hidden_out,
            typed_unit_local_hidden=typed_local_hidden_out,
            typed_unit_time_mask=typed_time_mask_out,
            typed_unit_mask=typed_unit_mask_out,
            typed_unit_kind_index=typed_kind_out,
            typed_unit_electrode_index=typed_electrode_out,
            typed_unit_lead_endpoint_index=typed_lead_endpoint_out,
            typed_unit_source_analysis_unit_mask=typed_source_unit_mask_out,
            typed_unit_reference_family_mask=typed_reference_family_mask_out,
        )
        causal_trace.verify_shapes()
        return BAIEGPermissionSplitSegmentalStateOutput(
            source_input_batch_sha256=batch.input_batch_sha256,
            source_context_receipt_sha256=context.receipt_sha256,
            implementation_id=self.implementation_id,
            heuristic_phase_posterior_used=self.allow_heuristic_phase_posterior,
            causal_typed_unit_trace=causal_trace,
            causal_candidate_times_seconds=causal_times_out,
            causal_candidate_mask=causal_mask_out,
            causal_onset_hazard_logits=causal_logits_out,
            causal_onset_boundary_mass=causal_mass_out,
            causal_left_censor_state_mass=left_mass_out,
            causal_no_onset_within_support_mass=no_onset_out,
            lattice_cell_bounds_seconds=cell_bounds_out,
            lattice_cell_mask=cell_mask_out,
            lattice_opportunity_duration_seconds=opportunity_out,
            lattice_physical_duration_seconds=physical_duration_out,
            quality_gap_cell_mask=quality_gap_out,
            offline_state_emission_logits=state_logits_out,
            offline_state_emission_log_prob=state_log_prob_out,
            offline_offset_hazard_logits=offset_logits_out,
            offline_offset_boundary_mass=offset_mass_out,
            offline_right_censor_state_mass=right_mass_out,
            offline_no_offset_within_support_mass=no_offset_out,
            split_or_reentry_review_logits=split_logits_out,
            transition_log_scores=transition_scores_out,
            transition_mask=transition_mask_out,
            start_state_log_scores=start_scores_out,
            end_state_log_scores=end_scores_out,
            event_presence_log_scores=event_scores_out,
            exact_path_log_partition=log_partition_out,
            full_state_marginals=exact_state_out,
            full_transition_marginals=exact_transition_out,
            full_start_state_marginals=exact_start_out,
            full_end_state_marginals=exact_end_out,
            full_segment_count_marginals=exact_segment_count_out,
            exact_onset_boundary_mass=exact_onset_out,
            exact_primary_onset_boundary_mass=exact_primary_onset_out,
            exact_secondary_onset_boundary_mass=exact_secondary_onset_out,
            exact_s0_onset_boundary_mass=exact_s0_onset_out,
            exact_s3_reentry_onset_boundary_mass=exact_s3_reentry_onset_out,
            exact_offset_boundary_mass=exact_offset_out,
            exact_event_mass=exact_event_out,
            exact_null_mass=exact_null_out,
            exact_left_censor_mass=exact_left_censor_out,
            exact_right_censor_mass=exact_right_censor_out,
            exact_both_censor_mass=exact_both_censor_out,
            exact_recurrent_event_mass=exact_recurrent_event_out,
            exact_expected_event_bout_count=exact_expected_bout_out,
            exact_expected_recurrent_bout_count=(
                exact_expected_recurrent_bout_out
            ),
            path_start_state_index=path_start_out,
            path_end_state_index=path_end_out,
            path_transition_times_seconds=path_transition_times_out,
            path_transition_mask=path_transition_mask_out,
            path_transition_type_index=path_transition_type_out,
            path_segment_bounds_seconds=path_segment_bounds_out,
            path_segment_state_index=path_segment_state_out,
            path_segment_mask=path_segment_mask_out,
            path_recurrent_cycle_count=path_cycle_count_out,
            path_log_scores=path_scores_out,
            path_posterior_mass=path_posterior_mass_out,
            path_weights_conditional_on_retained=path_weights_out,
            path_mask=path_mask_out,
            retained_path_mass_fraction=retained_mass_out,
            residual_path_mass_fraction=(1.0 - retained_mass_out).clamp(0.0, 1.0),
            retained_state_marginals=state_marginals_out,
            one_way_consistency_loss_per_event=consistency_out,
            event_evaluable_mask=event_evaluable,
            duration_location_log_seconds=duration_location,
            duration_scale_log_seconds=torch.log(duration_scale),
            minimum_state_duration_seconds=minimum_duration,
        )
