"""Public event-level MIL supervision for the shallow causal typed-unit head.

The public seizure interval says *when an event starts*, not which electrode or
lead is positive.  This module therefore projects that single event-level fact
onto the already frozen future-free causal axis and optimizes only the bag
statement

    at least one eligible typed unit places boundary mass in the onset support.

The latent typed-unit identity is marginalized with a noisy-OR MIL objective;
the event label is never copied to every unit.  No channel target, patient
positive set, spread label, offline representation, private source, EDF
annotation, spreadsheet, doctor text or report field is accepted by either
public function.  Unit identity remains supervised only after
event -> record -> patient aggregation by the separate complete-patient
positive-set bridge.

These values are uncalibrated source-training objectives.  They are not
clinical probabilities, Findings, seizure qualification or localization
claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Final, Mapping, Sequence

import torch

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    BAIEGCausalTypedUnitTrace,
)
from .ba_ieg_permission_split_segmental_supervision_v1 import (
    BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID,
    BAIEGSegmentalEventTargetV1,
    BAIEGSegmentalLatticeTargetProjectionV1,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
    BAIEGShallowCausalTypedUnitHeadOutput,
    ba_ieg_causal_typed_unit_axis_receipt_sha256,
)


BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID: Final[str] = (
    "ba_ieg_shallow_causal_typed_unit_supervision_v1"
)
BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_TARGET_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1"
)
BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_shallow_causal_typed_unit_mil_boundary_loss_v1"
)

_TIME_TOLERANCE_SECONDS: Final[float] = 1e-6
_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PUBLIC_ONSET_AUTHORITY: Final[str] = "public_seizure_interval"

_LOSS_POLICY: Final[Mapping[str, object]] = {
    "schema_version": BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_SCHEMA_VERSION,
    "optimization_split": "source_train",
    "target_authority": _PUBLIC_ONSET_AUTHORITY,
    "target_fact": "event_level_observed_onset_interval_only",
    "temporal_axis": "future_free_causal_group_axis",
    "objective": (
        "negative_log_noisy_or_mass_that_at_least_one_eligible_typed_unit_"
        "places_a_boundary_in_projected_onset_support"
    ),
    "event_positive_copied_to_each_typed_unit": False,
    "typed_unit_event_or_rank_logits_used": False,
    "patient_positive_set_or_channel_target_used": False,
    "offline_lane_or_spread_used": False,
    "private_doctor_annotation_clinical_text_or_report_used": False,
    "not_evaluable_used_as_negative": False,
    "clinical_probability_or_localization_claim": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256: Final[str] = (
    _canonical_sha256(_LOSS_POLICY)
)


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or set(text) - _SHA256_ALPHABET:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _interval_hits_right_closed_support(
    interval: tuple[float, float], support: tuple[float, float]
) -> bool:
    start, stop = interval
    support_start, support_stop = support
    if stop > start:
        return min(stop, support_stop) - max(start, support_start) > 1e-12
    return (
        support_start + _TIME_TOLERANCE_SECONDS
        < start
        <= support_stop + _TIME_TOLERANCE_SECONDS
    )


def _effective_trace_masks(
    trace: BAIEGCausalTypedUnitTrace,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_mask = trace.typed_unit_time_mask & trace.group_mask.unsqueeze(-1)
    unit_mask = trace.typed_unit_mask & time_mask.any(dim=1)
    return time_mask, unit_mask


def _trace_axis_receipt(trace: BAIEGCausalTypedUnitTrace) -> str:
    time_mask, unit_mask = _effective_trace_masks(trace)
    return ba_ieg_causal_typed_unit_axis_receipt_sha256(
        source_input_batch_sha256=trace.source_input_batch_sha256,
        event_ids=trace.event_ids,
        recording_ids=trace.recording_ids,
        source_event_receipt_sha256s=trace.source_event_receipt_sha256s,
        identity_roster_sha256=trace.identity_roster_sha256,
        source_trace_implementation_id=trace.implementation_id,
        group_times_seconds=trace.group_times_seconds,
        group_boundary_bounds_seconds=trace.group_boundary_bounds_seconds,
        group_mask=trace.group_mask,
        typed_unit_time_mask=time_mask,
        typed_unit_mask=unit_mask,
        typed_unit_kind_index=trace.typed_unit_kind_index,
        typed_unit_electrode_index=trace.typed_unit_electrode_index,
        typed_unit_lead_endpoint_index=trace.typed_unit_lead_endpoint_index,
    )


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitMILEventTargetV1:
    """One event-level temporal bag target; it contains no unit label."""

    event_id: str
    recording_id: str
    source_event_receipt_sha256: str
    source_segmental_target_receipt_sha256: str
    source_lattice_projection_receipt_sha256: str
    causal_group_target_mask: tuple[bool, ...]
    event_evaluable: bool
    non_evaluable_reason: str
    source_authority: str = _PUBLIC_ONSET_AUTHORITY
    model_split: str = "source_train"
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        event_id = _identifier(self.event_id, "event_id")
        recording_id = _identifier(self.recording_id, "recording_id")
        for name in (
            "source_event_receipt_sha256",
            "source_segmental_target_receipt_sha256",
            "source_lattice_projection_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        mask = tuple(self.causal_group_target_mask)
        if not mask or any(type(value) is not bool for value in mask):
            raise TypeError("causal group target mask must be a non-empty bool tuple")
        if type(self.event_evaluable) is not bool:
            raise TypeError("event_evaluable must be boolean")
        reason = _identifier(self.non_evaluable_reason, "non_evaluable_reason")
        if self.event_evaluable:
            if not any(mask) or reason != "none":
                raise ValueError("evaluable MIL target needs temporal support and no reason")
        elif reason == "none":
            raise ValueError("non-evaluable MIL target needs an explicit reason")
        if self.source_authority != _PUBLIC_ONSET_AUTHORITY:
            raise ValueError("typed-unit MIL target must use public seizure intervals")
        if self.model_split != "source_train":
            raise ValueError("typed-unit MIL target is source_train-only")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "causal_group_target_mask", mask)
        object.__setattr__(self, "non_evaluable_reason", reason)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "ba_ieg_shallow_causal_typed_unit_mil_event_target_v1",
                "event_id": self.event_id,
                "recording_id": self.recording_id,
                "source_event_receipt_sha256": self.source_event_receipt_sha256,
                "source_segmental_target_receipt_sha256": (
                    self.source_segmental_target_receipt_sha256
                ),
                "source_lattice_projection_receipt_sha256": (
                    self.source_lattice_projection_receipt_sha256
                ),
                "causal_group_target_mask": list(
                    self.causal_group_target_mask
                ),
                "event_evaluable": self.event_evaluable,
                "non_evaluable_reason": self.non_evaluable_reason,
                "source_authority": self.source_authority,
                "model_split": self.model_split,
                "loss_contract_sha256": (
                    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256
                ),
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("typed-unit MIL event target changed after registration")


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitMILTargetBundleV1:
    """A source-train event bag roster aligned to one causal typed-unit axis."""

    source_input_batch_sha256: str
    identity_roster_sha256: str
    causal_typed_unit_axis_receipt_sha256: str
    source_context_receipt_sha256: str
    target_independent_candidate_roster_receipt_sha256: str
    event_targets: tuple[BAIEGShallowCausalTypedUnitMILEventTargetV1, ...]
    optimization_role: str = "optimize"
    model_split: str = "source_train"
    source_authority: str = _PUBLIC_ONSET_AUTHORITY
    schema_version: str = (
        BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_TARGET_SCHEMA_VERSION
    )
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_input_batch_sha256",
            "identity_roster_sha256",
            "causal_typed_unit_axis_receipt_sha256",
            "source_context_receipt_sha256",
            "target_independent_candidate_roster_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.schema_version != (
            BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_TARGET_SCHEMA_VERSION
        ):
            raise ValueError("typed-unit MIL target schema drifted")
        if self.optimization_role != "optimize" or self.model_split != "source_train":
            raise ValueError("typed-unit MIL optimization is source_train-only")
        if self.source_authority != _PUBLIC_ONSET_AUTHORITY:
            raise ValueError("typed-unit MIL requires public seizure intervals")
        rows = tuple(self.event_targets)
        if not rows or not all(
            isinstance(row, BAIEGShallowCausalTypedUnitMILEventTargetV1)
            for row in rows
        ):
            raise TypeError("typed-unit MIL target bundle requires typed event rows")
        for row in rows:
            row.verify_integrity()
            if (
                row.model_split != self.model_split
                or row.source_authority != self.source_authority
            ):
                raise ValueError("typed-unit MIL event target permission drifted")
        if len({row.event_id for row in rows}) != len(rows):
            raise ValueError("typed-unit MIL target bundle repeats an event")
        object.__setattr__(self, "event_targets", rows)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "identity_roster_sha256": self.identity_roster_sha256,
                "causal_typed_unit_axis_receipt_sha256": (
                    self.causal_typed_unit_axis_receipt_sha256
                ),
                "source_context_receipt_sha256": (
                    self.source_context_receipt_sha256
                ),
                "target_independent_candidate_roster_receipt_sha256": (
                    self.target_independent_candidate_roster_receipt_sha256
                ),
                "event_target_receipt_sha256s": [
                    row.receipt_sha256 for row in self.event_targets
                ],
                "optimization_role": self.optimization_role,
                "model_split": self.model_split,
                "source_authority": self.source_authority,
                "loss_contract_sha256": (
                    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256
                ),
            }
        )

    def verify_integrity(self) -> None:
        for row in self.event_targets:
            row.verify_integrity()
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("typed-unit MIL target bundle changed after registration")


def _validate_trace_coordinates(trace: BAIEGCausalTypedUnitTrace) -> None:
    times = trace.group_times_seconds.detach().cpu()
    bounds = trace.group_boundary_bounds_seconds.detach().cpu()
    group_mask = trace.group_mask.detach().cpu()
    if not torch.isfinite(times).all() or not torch.isfinite(bounds).all():
        raise ValueError("causal typed-unit axis contains non-finite coordinates")
    coordinate_mask = bounds[..., 1] > bounds[..., 0]
    if torch.any(group_mask & ~coordinate_mask):
        raise ValueError("causal opportunity has no physical boundary support")
    if torch.any(~coordinate_mask & (bounds != 0).any(dim=-1)):
        raise ValueError("padded causal boundary coordinates are non-canonical")
    if torch.any(~coordinate_mask & (times != 0)):
        raise ValueError("padded causal times are non-canonical")
    for event_index in range(int(times.shape[0])):
        real = torch.nonzero(coordinate_mask[event_index], as_tuple=False).flatten()
        if not bool(real.numel()):
            continue
        local_times = times[event_index, real]
        local_bounds = bounds[event_index, real]
        if not torch.allclose(
            local_times,
            local_bounds[:, 1],
            atol=_TIME_TOLERANCE_SECONDS,
            rtol=0.0,
        ):
            raise ValueError("causal group time is not its physical cell boundary")
        if bool((local_times[1:] <= local_times[:-1]).any()):
            raise ValueError("causal group times are not strictly ordered")


def _projected_group_mask_for_event(
    *,
    trace: BAIEGCausalTypedUnitTrace,
    event_index: int,
    target: BAIEGSegmentalEventTargetV1,
    projection: BAIEGSegmentalLatticeTargetProjectionV1,
) -> tuple[tuple[bool, ...], str]:
    group_count = int(trace.group_times_seconds.shape[1])
    if projection.causal_axis_length != group_count:
        raise ValueError("segmental projection and typed-unit causal axes disagree")

    if (
        target.event_status == "not_evaluable"
        and target.onset_status == "not_evaluable"
    ):
        if (
            projection.effective_onset_status != "not_evaluable"
            or projection.causal_onset_candidate_mask is not None
            or projection.onset_selected_support_rows
        ):
            raise ValueError("not-evaluable onset projection carried positive support")
        return (False,) * group_count, "source_event_onset_not_evaluable"

    if target.event_status != "present" or target.onset_status != "observed_interval":
        raise ValueError(
            "typed-unit MIL accepts only present observed-onset events or "
            "explicitly not-evaluable events"
        )
    if (
        projection.effective_onset_status != "observed_interval"
        or projection.onset_projection_status
        != "mapped_to_frozen_causal_support"
        or projection.causal_onset_candidate_mask is None
        or target.onset_interval_seconds is None
    ):
        raise ValueError("observed event has no frozen causal onset projection")
    projected_mask = tuple(projection.causal_onset_candidate_mask)
    if len(projected_mask) != group_count:
        raise ValueError("projected onset mask has the wrong causal-axis length")
    selected_indices = {index for index, value in enumerate(projected_mask) if value}
    row_indices = {row[0] for row in projection.onset_selected_support_rows}
    if selected_indices != row_indices or len(row_indices) != len(
        projection.onset_selected_support_rows
    ):
        raise ValueError("projected onset support rows and mask disagree")

    group_times = trace.group_times_seconds[event_index].detach().cpu()
    group_bounds = trace.group_boundary_bounds_seconds[event_index].detach().cpu()
    group_mask = trace.group_mask[event_index].detach().cpu()
    active_indices = [
        int(index)
        for index in torch.nonzero(group_mask, as_tuple=False).flatten()
    ]
    previous_active_time: dict[int, float | None] = {}
    previous: float | None = None
    for index in active_indices:
        previous_active_time[index] = previous
        previous = float(group_times[index])

    for index, support_start, support_stop, represented_boundary in (
        projection.onset_selected_support_rows
    ):
        if not 0 <= index < group_count or not bool(group_mask[index]):
            raise ValueError("projected onset selected a masked causal group")
        group_time = float(group_times[index])
        cell_start = float(group_bounds[index, 0])
        cell_stop = float(group_bounds[index, 1])
        coordinates = (support_start, support_stop, represented_boundary)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("projected onset support contains non-finite coordinates")
        if (
            support_stop <= support_start
            or abs(support_stop - group_time) > _TIME_TOLERANCE_SECONDS
            or abs(represented_boundary - group_time)
            > _TIME_TOLERANCE_SECONDS
            or abs(cell_stop - group_time) > _TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("projected onset support is not bound to the causal group")
        lower = previous_active_time[index]
        if lower is None:
            if abs(support_start - cell_start) > _TIME_TOLERANCE_SECONDS:
                raise ValueError("first causal support bin has an invalid left edge")
        elif (
            support_start < lower - _TIME_TOLERANCE_SECONDS
            or support_start > cell_start + _TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("causal support bin crosses its frozen neighboring bounds")
        if not _interval_hits_right_closed_support(
            target.onset_interval_seconds,
            (float(support_start), float(support_stop)),
        ):
            raise ValueError("projected support does not intersect the raw onset interval")
    return projected_mask, "none"


def build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1(
    trace: BAIEGCausalTypedUnitTrace,
    targets: Sequence[BAIEGSegmentalEventTargetV1],
    projections: Sequence[BAIEGSegmentalLatticeTargetProjectionV1],
) -> BAIEGShallowCausalTypedUnitMILTargetBundleV1:
    """Bind public onset projections to the exact future-free typed-unit axis.

    This builder accepts no head logits and performs no target-conditioned
    candidate selection.  ``projections`` must already have been created by
    the registered post-forward frozen-lattice projector.
    """

    if not isinstance(trace, BAIEGCausalTypedUnitTrace):
        raise TypeError("typed-unit MIL target builder requires a causal trace")
    trace.verify_shapes()
    if trace.implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID:
        raise ValueError("typed-unit MIL trace implementation drifted")
    _validate_trace_coordinates(trace)
    target_rows = tuple(targets)
    projection_rows = tuple(projections)
    event_count = len(trace.event_ids)
    if len(target_rows) != event_count or len(projection_rows) != event_count:
        raise ValueError("typed-unit MIL targets/projections must cover the event roster")
    if not all(isinstance(row, BAIEGSegmentalEventTargetV1) for row in target_rows):
        raise TypeError("typed-unit MIL requires typed segmental event targets")
    if not all(
        isinstance(row, BAIEGSegmentalLatticeTargetProjectionV1)
        for row in projection_rows
    ):
        raise TypeError("typed-unit MIL requires typed lattice projections")

    context_receipts: set[str] = set()
    candidate_roster_receipts: set[str] = set()
    event_targets: list[BAIEGShallowCausalTypedUnitMILEventTargetV1] = []
    effective_time_mask, _ = _effective_trace_masks(trace)
    for event_index, (target, projection) in enumerate(
        zip(target_rows, projection_rows)
    ):
        target.verify_integrity()
        projection.verify_integrity()
        if target.model_split != "source_train":
            raise ValueError("typed-unit MIL supervision is source_train-only")
        if target.authority != _PUBLIC_ONSET_AUTHORITY:
            raise ValueError("typed-unit MIL supervision requires public onset intervals")
        if (
            target.offset_status != "not_evaluable"
            or target.offset_interval_seconds is not None
            or projection.effective_offset_status != "not_evaluable"
            or projection.raw_offset_interval_seconds is not None
        ):
            raise ValueError(
                "typed-unit MIL supervision accepts no offset/offline target"
            )
        if projection.method_id != (
            BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID
        ):
            raise ValueError("typed-unit MIL lattice projection method drifted")
        if (
            target.event_id != trace.event_ids[event_index]
            or target.recording_id != trace.recording_ids[event_index]
            or target.source_event_receipt_sha256
            != trace.source_event_receipt_sha256s[event_index]
        ):
            raise ValueError("typed-unit MIL target identity/order drifted")
        if (
            projection.event_id != target.event_id
            or projection.source_target_receipt_sha256
            != target.receipt_sha256
            or projection.source_input_batch_sha256
            != trace.source_input_batch_sha256
            or projection.raw_onset_interval_seconds
            != target.onset_interval_seconds
        ):
            raise ValueError("typed-unit MIL target/projection binding drifted")
        projected_mask, reason = _projected_group_mask_for_event(
            trace=trace,
            event_index=event_index,
            target=target,
            projection=projection,
        )
        target_mask_tensor = torch.tensor(
            projected_mask,
            dtype=torch.bool,
            device=effective_time_mask.device,
        )
        has_typed_opportunity = bool(
            (
                effective_time_mask[event_index]
                & target_mask_tensor.unsqueeze(-1)
            ).any()
        )
        event_evaluable = reason == "none" and has_typed_opportunity
        if reason == "none" and not has_typed_opportunity:
            reason = "no_typed_unit_opportunity_in_projected_onset_support"
        event_targets.append(
            BAIEGShallowCausalTypedUnitMILEventTargetV1(
                event_id=target.event_id,
                recording_id=target.recording_id,
                source_event_receipt_sha256=(
                    target.source_event_receipt_sha256
                ),
                source_segmental_target_receipt_sha256=(
                    target.receipt_sha256
                ),
                source_lattice_projection_receipt_sha256=(
                    projection.receipt_sha256
                ),
                causal_group_target_mask=projected_mask,
                event_evaluable=event_evaluable,
                non_evaluable_reason=reason,
            )
        )
        context_receipts.add(projection.source_context_receipt_sha256)
        candidate_roster_receipts.add(
            target.target_independent_candidate_roster_receipt_sha256
        )
    if len(context_receipts) != 1:
        raise ValueError("typed-unit MIL projection roster crosses boundary contexts")
    if len(candidate_roster_receipts) != 1:
        raise ValueError("typed-unit MIL target roster crosses candidate freezes")
    return BAIEGShallowCausalTypedUnitMILTargetBundleV1(
        source_input_batch_sha256=trace.source_input_batch_sha256,
        identity_roster_sha256=trace.identity_roster_sha256,
        causal_typed_unit_axis_receipt_sha256=_trace_axis_receipt(trace),
        source_context_receipt_sha256=next(iter(context_receipts)),
        target_independent_candidate_roster_receipt_sha256=next(
            iter(candidate_roster_receipts)
        ),
        event_targets=tuple(event_targets),
    )


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1:
    total_loss: torch.Tensor
    nll_per_event: torch.Tensor
    event_loss_mask: torch.Tensor
    event_explanation_mass: torch.Tensor
    eligible_typed_unit_count: torch.Tensor
    source_input_batch_sha256: str
    target_bundle_receipt_sha256: str
    causal_typed_unit_axis_receipt_sha256: str
    loss_contract_sha256: str = (
        BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256
    )
    objective_semantics: str = (
        "event_level_at_least_one_latent_typed_unit_boundary_noisy_or_mil"
    )
    output_semantics: str = (
        "source_training_objective_not_a_clinical_probability_finding_or_localization_claim"
    )


def _validate_head_output_for_mil(
    output: BAIEGShallowCausalTypedUnitHeadOutput,
) -> None:
    if output.implementation_id != BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID:
        raise ValueError("typed-unit head implementation drifted")
    if output.source_trace_implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID:
        raise ValueError("typed-unit head is not bound to the registered causal trace")
    event_count = len(output.event_ids)
    if (
        event_count < 1
        or len(output.recording_ids) != event_count
        or len(output.source_event_receipt_sha256s) != event_count
    ):
        raise ValueError("typed-unit head event roster is not aligned")
    mass = output.typed_unit_boundary_mass
    no_boundary = output.typed_unit_no_boundary_mass
    time_mask = output.typed_unit_time_mask
    unit_mask = output.typed_unit_inventory_mask
    if mass.ndim != 3:
        raise ValueError("typed-unit boundary mass must have shape [B,G,K]")
    batch_size, group_count, typed_count = mass.shape
    if batch_size != event_count:
        raise ValueError("typed-unit boundary mass event axis drifted")
    if (
        tuple(no_boundary.shape) != (batch_size, typed_count)
        or tuple(time_mask.shape) != tuple(mass.shape)
        or tuple(unit_mask.shape) != (batch_size, typed_count)
        or tuple(output.typed_unit_mask.shape) != (batch_size, typed_count)
        or tuple(output.causal_group_times_seconds.shape)
        != (batch_size, group_count)
        or tuple(output.causal_group_boundary_bounds_seconds.shape)
        != (batch_size, group_count, 2)
        or tuple(output.causal_group_mask.shape) != (batch_size, group_count)
        or tuple(output.typed_unit_kind_index.shape)
        != (batch_size, typed_count)
        or tuple(output.typed_unit_electrode_index.shape)
        != (batch_size, typed_count)
        or tuple(output.typed_unit_lead_endpoint_index.shape)
        != (batch_size, typed_count, 2)
    ):
        raise ValueError("typed-unit MIL axes are not aligned")
    if (
        not mass.is_floating_point()
        or not no_boundary.is_floating_point()
        or not torch.isfinite(mass).all()
        or not torch.isfinite(no_boundary).all()
        or torch.any(mass < 0)
        or torch.any(no_boundary < 0)
    ):
        raise ValueError("typed-unit boundary masses must be finite and non-negative")
    if (
        time_mask.dtype != torch.bool
        or unit_mask.dtype != torch.bool
        or output.typed_unit_mask.dtype != torch.bool
        or output.causal_group_mask.dtype != torch.bool
        or mass.device != no_boundary.device
        or mass.device != time_mask.device
        or mass.device != unit_mask.device
    ):
        raise ValueError("typed-unit MIL masses and masks have incompatible types/devices")
    if torch.any(output.typed_unit_mask & ~unit_mask):
        raise ValueError("onset-associated typed units exceed the causal inventory")
    if torch.any(time_mask & ~output.causal_group_mask.unsqueeze(-1)) or torch.any(
        time_mask & ~unit_mask.unsqueeze(1)
    ):
        raise ValueError("typed-unit opportunity exceeds its registered causal axis")
    tolerance = 2e-5
    if bool((mass.masked_select(~time_mask).abs() > tolerance).any()):
        raise ValueError("masked typed-unit times carry boundary mass")
    total_mass = mass.sum(dim=1) + no_boundary
    if bool(
        (
            torch.abs(total_mass[unit_mask] - 1.0) > tolerance
        ).any()
    ) or bool((total_mass[~unit_mask].abs() > tolerance).any()):
        raise ValueError("typed-unit boundary/null masses are not normalized")
    replayed_axis = ba_ieg_causal_typed_unit_axis_receipt_sha256(
        source_input_batch_sha256=output.source_input_batch_sha256,
        event_ids=output.event_ids,
        recording_ids=output.recording_ids,
        source_event_receipt_sha256s=output.source_event_receipt_sha256s,
        identity_roster_sha256=output.identity_roster_sha256,
        source_trace_implementation_id=output.source_trace_implementation_id,
        group_times_seconds=output.causal_group_times_seconds,
        group_boundary_bounds_seconds=(
            output.causal_group_boundary_bounds_seconds
        ),
        group_mask=output.causal_group_mask,
        typed_unit_time_mask=output.typed_unit_time_mask,
        typed_unit_mask=output.typed_unit_inventory_mask,
        typed_unit_kind_index=output.typed_unit_kind_index,
        typed_unit_electrode_index=output.typed_unit_electrode_index,
        typed_unit_lead_endpoint_index=(
            output.typed_unit_lead_endpoint_index
        ),
    )
    if replayed_axis != output.causal_typed_unit_axis_receipt_sha256:
        raise ValueError("typed-unit head causal-axis receipt drifted")


def shallow_causal_typed_unit_mil_boundary_loss_v1(
    output: BAIEGShallowCausalTypedUnitHeadOutput,
    target_bundle: BAIEGShallowCausalTypedUnitMILTargetBundleV1,
) -> BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1:
    """Optimize an event-level positive bag without per-unit target copying."""

    if not isinstance(output, BAIEGShallowCausalTypedUnitHeadOutput):
        raise TypeError("typed-unit MIL loss requires a shallow causal head output")
    if not isinstance(
        target_bundle, BAIEGShallowCausalTypedUnitMILTargetBundleV1
    ):
        raise TypeError("typed-unit MIL loss requires its typed target bundle")
    _validate_head_output_for_mil(output)
    target_bundle.verify_integrity()
    if (
        target_bundle.optimization_role != "optimize"
        or target_bundle.model_split != "source_train"
        or target_bundle.source_authority != _PUBLIC_ONSET_AUTHORITY
    ):
        raise ValueError("gradient-bearing typed-unit MIL loss is public source_train-only")
    if (
        output.source_input_batch_sha256
        != target_bundle.source_input_batch_sha256
        or output.identity_roster_sha256
        != target_bundle.identity_roster_sha256
        or output.causal_typed_unit_axis_receipt_sha256
        != target_bundle.causal_typed_unit_axis_receipt_sha256
    ):
        raise ValueError("typed-unit head and MIL target content addresses disagree")
    if (
        tuple(row.event_id for row in target_bundle.event_targets)
        != output.event_ids
        or tuple(row.recording_id for row in target_bundle.event_targets)
        != output.recording_ids
        or tuple(
            row.source_event_receipt_sha256
            for row in target_bundle.event_targets
        )
        != output.source_event_receipt_sha256s
    ):
        raise ValueError("typed-unit MIL target event identity/order drifted")

    mass = output.typed_unit_boundary_mass
    event_count, group_count, _ = mass.shape
    nll_rows = mass.new_zeros(event_count)
    explanation_rows = mass.new_zeros(event_count)
    loss_mask = torch.zeros(event_count, dtype=torch.bool, device=mass.device)
    eligible_counts = torch.zeros(
        event_count, dtype=torch.long, device=mass.device
    )
    for event_index, target in enumerate(target_bundle.event_targets):
        target.verify_integrity()
        if len(target.causal_group_target_mask) != group_count:
            raise ValueError("typed-unit MIL temporal target axis drifted")
        target_mask = torch.tensor(
            target.causal_group_target_mask,
            dtype=torch.bool,
            device=mass.device,
        )
        eligible_unit_mask = output.typed_unit_inventory_mask[event_index] & (
            output.typed_unit_time_mask[event_index]
            & target_mask.unsqueeze(-1)
        ).any(dim=0)
        eligible_count = int(eligible_unit_mask.sum().detach().cpu())
        eligible_counts[event_index] = eligible_count
        expected_evaluable = bool(target_mask.any()) and eligible_count > 0
        if expected_evaluable != target.event_evaluable:
            raise ValueError("typed-unit MIL evaluability drifted from causal opportunity")
        if not target.event_evaluable:
            continue

        selected_mass = (
            mass[event_index]
            * target_mask.unsqueeze(-1).to(dtype=mass.dtype)
            * output.typed_unit_time_mask[event_index].to(dtype=mass.dtype)
        ).sum(dim=0)[eligible_unit_mask]
        epsilon = torch.finfo(mass.dtype).eps
        log_no_explaining_unit = torch.log1p(
            -selected_mass.clamp(min=0.0, max=1.0 - epsilon)
        ).sum()
        explanation_mass = -torch.expm1(log_no_explaining_unit)
        nll = -torch.log(
            explanation_mass.clamp_min(torch.finfo(mass.dtype).tiny)
        )
        explanation_rows[event_index] = explanation_mass
        nll_rows[event_index] = nll
        loss_mask[event_index] = True

    if not bool(loss_mask.any()):
        raise ValueError("typed-unit MIL target bundle has no evaluable event")
    total_loss = nll_rows[loss_mask].mean()
    return BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1(
        total_loss=total_loss,
        nll_per_event=nll_rows,
        event_loss_mask=loss_mask,
        event_explanation_mass=explanation_rows,
        eligible_typed_unit_count=eligible_counts,
        source_input_batch_sha256=output.source_input_batch_sha256,
        target_bundle_receipt_sha256=target_bundle.receipt_sha256,
        causal_typed_unit_axis_receipt_sha256=(
            output.causal_typed_unit_axis_receipt_sha256
        ),
    )


__all__ = [
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_CONTRACT_SHA256",
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_LOSS_SCHEMA_VERSION",
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_MIL_TARGET_SCHEMA_VERSION",
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID",
    "BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1",
    "BAIEGShallowCausalTypedUnitMILEventTargetV1",
    "BAIEGShallowCausalTypedUnitMILTargetBundleV1",
    "build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1",
    "shallow_causal_typed_unit_mil_boundary_loss_v1",
]
