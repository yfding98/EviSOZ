"""Shallow future-free typed-unit onset head for the BA-IEG v1 core.

The head accepts only :class:`BAIEGCausalTypedUnitTrace`.  It therefore reuses
the exact causal projection and prefix-GRU execution already performed by the
permission-split segmental model, without constructing a second temporal
backbone and without exposing the offline lane to positive onset scoring.

Physical-electrode candidates are constructive referential/common-average
identities.  Bipolar candidates remain whole lead identities and are never
scattered to either endpoint.  All quantities are uncalibrated research
candidates; none is a clinical SOZ probability or report-qualified Finding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.soz.geometry import STANDARD_19

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
    BAIEGCausalTypedUnitTrace,
)


BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID: Final[str] = (
    "ba_ieg_shallow_causal_typed_unit_onset_head_v1"
)
BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS: Final[float] = 0.95
_ONSET_IDENTITY_POLICY_V1: Final[dict[str, object]] = {
    "schema": "ba_ieg_causal_onset_identity_association_policy_v1",
    "global_gate_source": "future_free_segmental_global_onset_boundary_mass",
    "global_gate_target_conditioned": False,
    "global_gate_patient_identity_gradient_allowed": False,
    "typed_boundary_patient_identity_gradient_allowed": False,
    "causal_trace_patient_identity_gradient_allowed": False,
    "global_support": "equal_tail_central_observed_onset_mass",
    "global_support_mass": BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS,
    "global_status_rule": "observed_onset_mass_gt_left_censor_plus_no_onset_mass",
    "typed_boundary_source": "shallow_causal_typed_unit_boundary_mass",
    "identity_mass": (
        "sum_over_causal_groups_of_detached_global_gate_times_typed_boundary_"
        "mass_times_sigmoid_cell_rank_logit"
    ),
    "event_logit": "logit_of_clamped_joint_onset_identity_mass",
    "all_causal_group_rank_logmeanexp_allowed": False,
    "late_only_unit_without_global_support_overlap_evaluable": False,
    "left_or_no_onset_unresolved_event_evaluable": False,
    "public_target_mask_accepted_by_forward": False,
    "output_is_clinical_probability_or_localization_claim": False,
}
BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_V1: Final[Mapping[str, object]] = (
    MappingProxyType(_ONSET_IDENTITY_POLICY_V1)
)
BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_SHA256: Final[str] = hashlib.sha256(
    json.dumps(
        _ONSET_IDENTITY_POLICY_V1,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _axis_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def ba_ieg_causal_typed_unit_axis_receipt_sha256(
    *,
    source_input_batch_sha256: str,
    event_ids: Sequence[str],
    recording_ids: Sequence[str],
    source_event_receipt_sha256s: Sequence[str],
    identity_roster_sha256: str,
    source_trace_implementation_id: str,
    group_times_seconds: torch.Tensor,
    group_boundary_bounds_seconds: torch.Tensor,
    group_mask: torch.Tensor,
    typed_unit_time_mask: torch.Tensor,
    typed_unit_mask: torch.Tensor,
    typed_unit_kind_index: torch.Tensor,
    typed_unit_electrode_index: torch.Tensor,
    typed_unit_lead_endpoint_index: torch.Tensor,
) -> str:
    """Content-address the exact future-free time/unit supervision axes.

    Hidden values and learned logits are intentionally excluded: this receipt
    binds only the immutable event order, physical causal grid, opportunity
    masks and typed-unit identities needed to align an event-level boundary
    target.  It therefore cannot encode a patient positive set or a channel
    label.
    """

    payload = {
        "schema": "ba_ieg_causal_typed_unit_supervision_axis_v1",
        "source_input_batch_sha256": str(source_input_batch_sha256),
        "event_ids": list(event_ids),
        "recording_ids": list(recording_ids),
        "source_event_receipt_sha256s": list(source_event_receipt_sha256s),
        "identity_roster_sha256": str(identity_roster_sha256),
        "source_trace_implementation_id": str(source_trace_implementation_id),
        "tensor_sha256": {
            "group_times_seconds": _axis_tensor_sha256(group_times_seconds),
            "group_boundary_bounds_seconds": _axis_tensor_sha256(
                group_boundary_bounds_seconds
            ),
            "group_mask": _axis_tensor_sha256(group_mask),
            "typed_unit_time_mask": _axis_tensor_sha256(typed_unit_time_mask),
            "typed_unit_mask": _axis_tensor_sha256(typed_unit_mask),
            "typed_unit_kind_index": _axis_tensor_sha256(
                typed_unit_kind_index
            ),
            "typed_unit_electrode_index": _axis_tensor_sha256(
                typed_unit_electrode_index
            ),
            "typed_unit_lead_endpoint_index": _axis_tensor_sha256(
                typed_unit_lead_endpoint_index
            ),
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitHeadOutput:
    """Masked event-local ranking and boundary candidates."""

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    source_event_receipt_sha256s: tuple[str, ...]
    identity_roster_sha256: str
    source_trace_implementation_id: str
    causal_typed_unit_axis_receipt_sha256: str
    implementation_id: str
    causal_group_times_seconds: torch.Tensor
    causal_group_boundary_bounds_seconds: torch.Tensor
    causal_group_mask: torch.Tensor
    causal_global_onset_boundary_mass: torch.Tensor
    causal_global_left_censor_state_mass: torch.Tensor
    causal_global_no_onset_within_support_mass: torch.Tensor
    causal_global_onset_support_mask: torch.Tensor
    causal_global_onset_resolved_mask: torch.Tensor
    typed_unit_cell_rank_logits: torch.Tensor
    typed_unit_boundary_logits: torch.Tensor
    typed_unit_boundary_mass: torch.Tensor
    typed_unit_no_boundary_mass: torch.Tensor
    typed_unit_event_logits: torch.Tensor
    typed_unit_candidate_boundary_interval_seconds: torch.Tensor
    typed_unit_candidate_boundary_mask: torch.Tensor
    typed_unit_rank_within_kind: torch.Tensor
    typed_unit_time_mask: torch.Tensor
    typed_unit_inventory_mask: torch.Tensor
    typed_unit_mask: torch.Tensor
    typed_unit_onset_association_mass: torch.Tensor
    typed_unit_onset_identity_mass: torch.Tensor
    typed_unit_kind_index: torch.Tensor
    typed_unit_electrode_index: torch.Tensor
    typed_unit_lead_endpoint_index: torch.Tensor
    physical_electrode_event_logits: torch.Tensor
    physical_electrode_mask: torch.Tensor
    calibration_status: str = "uncalibrated_source_development_shadow"
    output_semantics: str = (
        "scalp_visible_onset_association_candidate_not_cortical_soz_or_clinical_probability"
    )
    global_onset_support_policy: str = (
        "target_free_equal_tail_95pct_central_support_with_observed_status_dominance"
    )


def _target_free_global_onset_support(
    trace: BAIEGCausalTypedUnitTrace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a hard target-free support around the global causal onset mass.

    The support is the discrete equal-tail 95% central interval of the
    *observed-onset* mass.  It is empty unless observed onset has more mass
    than the combined left-censored/no-onset alternatives.  The returned gate
    is detached so a downstream patient channel loss cannot move the global
    onset detector.
    """

    observed = torch.where(
        trace.group_mask,
        trace.global_onset_boundary_mass,
        torch.zeros_like(trace.global_onset_boundary_mass),
    ).detach()
    observed_total = observed.sum(dim=1)
    unresolved = (
        trace.global_left_censor_state_mass.sum(dim=1)
        + trace.global_no_onset_within_support_mass
    ).detach()
    resolved = observed_total > unresolved
    safe_total = observed_total.clamp_min(torch.finfo(observed.dtype).tiny)
    normalized = observed / safe_total.unsqueeze(-1)
    cumulative = torch.cumsum(normalized, dim=1)
    cumulative_before = cumulative - normalized
    tail = 0.5 * (1.0 - BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS)
    support = (
        trace.group_mask
        & resolved.unsqueeze(-1)
        & (normalized > 0)
        & (cumulative > tail)
        & (cumulative_before < 1.0 - tail)
    )
    gate = torch.where(support, observed, torch.zeros_like(observed))
    return support, resolved, gate


class BAIEGShallowCausalTypedUnitOnsetHead(nn.Module):
    """One-hidden-layer scoring/interval head over the shared causal trace."""

    implementation_id: Final[str] = BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden and bottleneck dimensions must be positive")
        if not math.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.fusion = nn.Linear(
            2 * hidden_dim + len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS),
            bottleneck_dim,
        )
        self.identity_fusion = nn.Linear(
            2 * hidden_dim + len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS),
            bottleneck_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.cell_rank_head = nn.Linear(bottleneck_dim, 1)
        self.boundary_head = nn.Linear(bottleneck_dim, 1)
        self.no_boundary_logit_by_kind = nn.Embedding(
            len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS), 1
        )

    def forward(
        self, trace: BAIEGCausalTypedUnitTrace
    ) -> BAIEGShallowCausalTypedUnitHeadOutput:
        if not isinstance(trace, BAIEGCausalTypedUnitTrace):
            raise TypeError("typed-unit head accepts only a causal trace")
        trace.verify_shapes()
        global_hidden = trace.global_group_hidden
        local_hidden = trace.typed_unit_local_hidden
        if int(global_hidden.shape[-1]) != self.hidden_dim:
            raise ValueError("causal trace hidden dimension does not match the head")
        if global_hidden.device != local_hidden.device or global_hidden.dtype != (
            local_hidden.dtype
        ):
            raise ValueError("global/local causal hidden tensors must share device/dtype")

        batch_size, group_count, typed_count, _ = local_hidden.shape
        time_mask = trace.typed_unit_time_mask & trace.group_mask.unsqueeze(-1)
        inventory_mask = trace.typed_unit_mask & time_mask.any(dim=1)
        safe_kind = trace.typed_unit_kind_index.clamp(
            min=0, max=len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS) - 1
        )
        kind_one_hot = F.one_hot(
            safe_kind, num_classes=len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS)
        ).to(dtype=local_hidden.dtype)
        kind_one_hot = kind_one_hot.unsqueeze(1).expand(
            -1, group_count, -1, -1
        )
        expanded_global = global_hidden.unsqueeze(2).expand(
            -1, -1, typed_count, -1
        )
        fused_input = torch.cat(
            (expanded_global, local_hidden, kind_one_hot), dim=-1
        )
        boundary_fused = self.dropout(F.gelu(self.fusion(fused_input)))
        boundary_fused = torch.where(
            time_mask.unsqueeze(-1),
            boundary_fused,
            torch.zeros_like(boundary_fused),
        )
        # Patient positive-set gradients are permitted to train the identity
        # adapter/rank head, but not the shared causal trace or boundary head.
        identity_fused = self.dropout(
            F.gelu(self.identity_fusion(fused_input.detach()))
        )
        identity_fused = torch.where(
            time_mask.unsqueeze(-1),
            identity_fused,
            torch.zeros_like(identity_fused),
        )

        cell_rank_logits = self.cell_rank_head(identity_fused).squeeze(-1)
        boundary_logits = self.boundary_head(boundary_fused).squeeze(-1)
        cell_rank_logits = torch.where(
            time_mask, cell_rank_logits, torch.zeros_like(cell_rank_logits)
        )
        boundary_logits = torch.where(
            time_mask, boundary_logits, torch.zeros_like(boundary_logits)
        )

        negative_infinity = torch.finfo(boundary_logits.dtype).min
        masked_boundary = boundary_logits.masked_fill(
            ~time_mask, negative_infinity
        ).transpose(1, 2)
        null_logits = self.no_boundary_logit_by_kind(safe_kind).squeeze(-1)
        all_boundary_logits = torch.cat(
            (masked_boundary, null_logits.unsqueeze(-1)), dim=-1
        )
        all_boundary_mass = torch.softmax(all_boundary_logits, dim=-1)
        all_boundary_mass = torch.where(
            inventory_mask.unsqueeze(-1),
            all_boundary_mass,
            torch.zeros_like(all_boundary_mass),
        )
        boundary_mass = all_boundary_mass[..., :group_count].transpose(1, 2)
        no_boundary_mass = all_boundary_mass[..., group_count]

        (
            global_onset_support_mask,
            global_onset_resolved_mask,
            detached_global_onset_gate,
        ) = _target_free_global_onset_support(trace)
        joint_boundary_mass = (
            boundary_mass * detached_global_onset_gate.unsqueeze(-1)
        )
        onset_association_mass = joint_boundary_mass.sum(dim=1)
        rank_probability = torch.sigmoid(cell_rank_logits)
        onset_identity_mass = (
            joint_boundary_mass.detach() * rank_probability
        ).sum(dim=1)
        association_mask = (
            inventory_mask
            & global_onset_resolved_mask.unsqueeze(-1)
            & (onset_association_mass > torch.finfo(boundary_mass.dtype).tiny)
        )
        clamped_identity_mass = onset_identity_mass.clamp(
            min=torch.finfo(onset_identity_mass.dtype).tiny,
            max=1.0 - torch.finfo(onset_identity_mass.dtype).eps,
        )
        event_logits = torch.logit(clamped_identity_mass)
        event_logits = torch.where(
            association_mask, event_logits, torch.zeros_like(event_logits)
        )

        best_group = joint_boundary_mass.transpose(1, 2).argmax(dim=-1)
        gather_index = best_group.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, 2
        )
        expanded_bounds = trace.group_boundary_bounds_seconds.unsqueeze(1).expand(
            -1, typed_count, -1, -1
        )
        candidate_interval = torch.gather(
            expanded_bounds, dim=2, index=gather_index
        ).squeeze(2)
        candidate_mask = association_mask
        candidate_interval = torch.where(
            candidate_mask.unsqueeze(-1),
            candidate_interval,
            torch.zeros_like(candidate_interval),
        )

        rank_within_kind = torch.zeros(
            (batch_size, typed_count),
            dtype=torch.long,
            device=event_logits.device,
        )
        for batch_index in range(batch_size):
            for kind_index in range(len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS)):
                selected = torch.nonzero(
                    association_mask[batch_index]
                    & (trace.typed_unit_kind_index[batch_index] == kind_index),
                    as_tuple=False,
                ).flatten()
                if not bool(selected.numel()):
                    continue
                ordered = selected[
                    torch.argsort(
                        event_logits[batch_index, selected],
                        descending=True,
                        stable=True,
                    )
                ]
                rank_within_kind[batch_index, ordered] = torch.arange(
                    1,
                    int(ordered.numel()) + 1,
                    dtype=torch.long,
                    device=event_logits.device,
                )

        physical_logits = torch.zeros(
            (batch_size, len(STANDARD_19)),
            dtype=event_logits.dtype,
            device=event_logits.device,
        )
        physical_mask = torch.zeros(
            (batch_size, len(STANDARD_19)),
            dtype=torch.bool,
            device=event_logits.device,
        )
        electrode_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index(
            "physical_electrode"
        )
        for batch_index in range(batch_size):
            selected = torch.nonzero(
                association_mask[batch_index]
                & (trace.typed_unit_kind_index[batch_index] == electrode_kind),
                as_tuple=False,
            ).flatten()
            for typed_index_tensor in selected:
                typed_index = int(typed_index_tensor)
                electrode_index = int(
                    trace.typed_unit_electrode_index[batch_index, typed_index]
                )
                if electrode_index < 0 or electrode_index >= len(STANDARD_19):
                    raise RuntimeError("physical typed unit has no valid electrode")
                if bool(physical_mask[batch_index, electrode_index]):
                    raise RuntimeError("physical typed-unit alias collapse failed")
                physical_logits[batch_index, electrode_index] = event_logits[
                    batch_index, typed_index
                ]
                physical_mask[batch_index, electrode_index] = True

        return BAIEGShallowCausalTypedUnitHeadOutput(
            source_input_batch_sha256=trace.source_input_batch_sha256,
            event_ids=trace.event_ids,
            recording_ids=trace.recording_ids,
            source_event_receipt_sha256s=trace.source_event_receipt_sha256s,
            identity_roster_sha256=trace.identity_roster_sha256,
            source_trace_implementation_id=trace.implementation_id,
            causal_typed_unit_axis_receipt_sha256=(
                ba_ieg_causal_typed_unit_axis_receipt_sha256(
                    source_input_batch_sha256=(
                        trace.source_input_batch_sha256
                    ),
                    event_ids=trace.event_ids,
                    recording_ids=trace.recording_ids,
                    source_event_receipt_sha256s=(
                        trace.source_event_receipt_sha256s
                    ),
                    identity_roster_sha256=trace.identity_roster_sha256,
                    source_trace_implementation_id=trace.implementation_id,
                    group_times_seconds=trace.group_times_seconds,
                    group_boundary_bounds_seconds=(
                        trace.group_boundary_bounds_seconds
                    ),
                    group_mask=trace.group_mask,
                    typed_unit_time_mask=time_mask,
                    typed_unit_mask=inventory_mask,
                    typed_unit_kind_index=trace.typed_unit_kind_index,
                    typed_unit_electrode_index=(
                        trace.typed_unit_electrode_index
                    ),
                    typed_unit_lead_endpoint_index=(
                        trace.typed_unit_lead_endpoint_index
                    ),
                )
            ),
            implementation_id=self.implementation_id,
            causal_group_times_seconds=trace.group_times_seconds,
            causal_group_boundary_bounds_seconds=(
                trace.group_boundary_bounds_seconds
            ),
            causal_group_mask=trace.group_mask,
            causal_global_onset_boundary_mass=(
                trace.global_onset_boundary_mass
            ),
            causal_global_left_censor_state_mass=(
                trace.global_left_censor_state_mass
            ),
            causal_global_no_onset_within_support_mass=(
                trace.global_no_onset_within_support_mass
            ),
            causal_global_onset_support_mask=global_onset_support_mask,
            causal_global_onset_resolved_mask=global_onset_resolved_mask,
            typed_unit_cell_rank_logits=cell_rank_logits,
            typed_unit_boundary_logits=boundary_logits,
            typed_unit_boundary_mass=boundary_mass,
            typed_unit_no_boundary_mass=no_boundary_mass,
            typed_unit_event_logits=event_logits,
            typed_unit_candidate_boundary_interval_seconds=candidate_interval,
            typed_unit_candidate_boundary_mask=candidate_mask,
            typed_unit_rank_within_kind=rank_within_kind,
            typed_unit_time_mask=time_mask,
            typed_unit_inventory_mask=inventory_mask,
            typed_unit_mask=association_mask,
            typed_unit_onset_association_mass=onset_association_mass,
            typed_unit_onset_identity_mass=onset_identity_mass,
            typed_unit_kind_index=trace.typed_unit_kind_index,
            typed_unit_electrode_index=trace.typed_unit_electrode_index,
            typed_unit_lead_endpoint_index=trace.typed_unit_lead_endpoint_index,
            physical_electrode_event_logits=physical_logits,
            physical_electrode_mask=physical_mask,
        )


__all__ = [
    "BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS",
    "BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_SHA256",
    "BA_IEG_CAUSAL_ONSET_IDENTITY_POLICY_V1",
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID",
    "BAIEGShallowCausalTypedUnitHeadOutput",
    "BAIEGShallowCausalTypedUnitOnsetHead",
    "ba_ieg_causal_typed_unit_axis_receipt_sha256",
]
