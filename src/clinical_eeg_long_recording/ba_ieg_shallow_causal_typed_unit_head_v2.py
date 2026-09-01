"""Clock-restricted typed-unit head for the BA-IEG G1 v2 candidate.

The v1 head intentionally remains untouched.  This v2 head makes the K3 lock
an explicit input type and separates two numerical paths:

* boundary proposals may use global causal hidden plus signal-only local hidden;
* identity/rank may use only detached signal-only local hidden, canonical unit
  kind, and the detached locked global onset mass.

Consequently recording/candidate/support geometry and shared boundary hidden
cannot reach the identity logits, even indirectly through the boundary head.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.soz.geometry import STANDARD_19

from .ba_ieg_earliest_prefix_k3_gate_v1 import (
    BAIEGEarliestPrefixK3GateResultV1,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
)
from .ba_ieg_permission_split_segmental_state_model_v2 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
    _target_free_global_onset_support,
    ba_ieg_causal_typed_unit_axis_receipt_sha256,
)


BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID_V2: Final[str] = (
    "ba_ieg_locked_signal_only_typed_unit_onset_head_v2"
)


class BAIEGShallowCausalTypedUnitOnsetHeadV2(nn.Module):
    """Separate boundary proposal from locked signal-only identity ranking."""

    implementation_id: Final[str] = BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID_V2

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
        self.boundary_fusion = nn.Linear(
            2 * hidden_dim + len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS),
            bottleneck_dim,
        )
        # Deliberately no global hidden, time coordinate or geometry slot.
        self.identity_fusion = nn.Linear(
            hidden_dim + len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS),
            bottleneck_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.cell_rank_head = nn.Linear(bottleneck_dim, 1)
        self.boundary_head = nn.Linear(bottleneck_dim, 1)
        self.no_boundary_logit_by_kind = nn.Embedding(
            len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS), 1
        )

    def forward(
        self, gate: BAIEGEarliestPrefixK3GateResultV1
    ) -> BAIEGShallowCausalTypedUnitHeadOutput:
        if not isinstance(gate, BAIEGEarliestPrefixK3GateResultV1):
            raise TypeError("v2 typed identity head requires an earliest-prefix/K3 gate")
        gate.verify()
        trace = gate.gated_trace
        if trace.implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2:
            raise ValueError("v2 typed identity head requires a signal-only v2 trace")
        global_hidden = trace.global_group_hidden
        local_hidden = trace.typed_unit_local_hidden
        if int(global_hidden.shape[-1]) != self.hidden_dim:
            raise ValueError("v2 causal trace hidden dimension does not match the head")
        if global_hidden.device != local_hidden.device or global_hidden.dtype != (
            local_hidden.dtype
        ):
            raise ValueError("v2 global/local hidden tensors must share device/dtype")

        batch_size, group_count, typed_count, _ = local_hidden.shape
        time_mask = trace.typed_unit_time_mask & trace.group_mask.unsqueeze(-1)
        inventory_mask = trace.typed_unit_mask & time_mask.any(dim=1)
        safe_kind = trace.typed_unit_kind_index.clamp(
            min=0, max=len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS) - 1
        )
        kind_one_hot = F.one_hot(
            safe_kind, num_classes=len(BA_IEG_CAUSAL_TYPED_UNIT_KINDS)
        ).to(dtype=local_hidden.dtype)
        expanded_kind = kind_one_hot.unsqueeze(1).expand(
            -1, group_count, -1, -1
        )
        # Global causal hidden is a proposal context only.  In particular,
        # L_typed_boundary_MIL must not use this branch to update the causal
        # onset model; only the signal-only local projection and the boundary
        # proposal adapter are authorized by the v1.3 permission split.
        expanded_global = global_hidden.detach().unsqueeze(2).expand(
            -1, -1, typed_count, -1
        )

        boundary_input = torch.cat(
            (expanded_global, local_hidden, expanded_kind), dim=-1
        )
        boundary_fused = self.dropout(
            F.gelu(self.boundary_fusion(boundary_input))
        )
        boundary_fused = torch.where(
            time_mask.unsqueeze(-1),
            boundary_fused,
            torch.zeros_like(boundary_fused),
        )
        boundary_logits = self.boundary_head(boundary_fused).squeeze(-1)
        boundary_logits = torch.where(
            time_mask, boundary_logits, torch.zeros_like(boundary_logits)
        )

        # The identity path is numerically and gradient-isolated from global
        # hidden and boundary proposal logits.  Locked global onset mass is
        # introduced later only as a detached scalar gate per causal group.
        identity_input = torch.cat((local_hidden, expanded_kind), dim=-1).detach()
        identity_fused = self.dropout(F.gelu(self.identity_fusion(identity_input)))
        identity_fused = torch.where(
            time_mask.unsqueeze(-1),
            identity_fused,
            torch.zeros_like(identity_fused),
        )
        cell_rank_logits = self.cell_rank_head(identity_fused).squeeze(-1)
        cell_rank_logits = torch.where(
            time_mask, cell_rank_logits, torch.zeros_like(cell_rank_logits)
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
            detached_locked_global_onset_gate,
        ) = _target_free_global_onset_support(trace)
        detached_locked_global_onset_gate = (
            detached_locked_global_onset_gate
            * gate.locked_event_mask.to(
                dtype=detached_locked_global_onset_gate.dtype
            ).unsqueeze(-1)
        ).detach()

        # Boundary association remains an auditable proposal only.
        boundary_joint = (
            boundary_mass * detached_locked_global_onset_gate.unsqueeze(-1)
        )
        onset_association_mass = boundary_joint.sum(dim=1)

        # Identity logits do not consume boundary mass/global hidden.  Their
        # only global quantity is the detached, prefix-locked onset mass.
        rank_probability = torch.sigmoid(cell_rank_logits)
        identity_joint = (
            detached_locked_global_onset_gate.unsqueeze(-1) * rank_probability
        )
        onset_identity_mass = identity_joint.sum(dim=1)
        identity_mask = (
            inventory_mask
            & gate.locked_event_mask.unsqueeze(-1)
            & global_onset_resolved_mask.unsqueeze(-1)
            & (onset_identity_mass > torch.finfo(boundary_mass.dtype).tiny)
        )
        clamped_identity_mass = onset_identity_mass.clamp(
            min=torch.finfo(onset_identity_mass.dtype).tiny,
            max=1.0 - torch.finfo(onset_identity_mass.dtype).eps,
        )
        event_logits = torch.logit(clamped_identity_mass)
        event_logits = torch.where(
            identity_mask, event_logits, torch.zeros_like(event_logits)
        )

        boundary_candidate_mask = (
            inventory_mask
            & gate.locked_event_mask.unsqueeze(-1)
            & (onset_association_mass > torch.finfo(boundary_mass.dtype).tiny)
        )
        best_group = boundary_joint.transpose(1, 2).argmax(dim=-1)
        gather_index = best_group.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, 2
        )
        expanded_bounds = trace.group_boundary_bounds_seconds.unsqueeze(1).expand(
            -1, typed_count, -1, -1
        )
        candidate_interval = torch.gather(
            expanded_bounds, dim=2, index=gather_index
        ).squeeze(2)
        candidate_interval = torch.where(
            boundary_candidate_mask.unsqueeze(-1),
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
                    identity_mask[batch_index]
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
                identity_mask[batch_index]
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
                    source_input_batch_sha256=trace.source_input_batch_sha256,
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
            typed_unit_candidate_boundary_mask=boundary_candidate_mask,
            typed_unit_rank_within_kind=rank_within_kind,
            typed_unit_time_mask=time_mask,
            typed_unit_inventory_mask=inventory_mask,
            typed_unit_mask=identity_mask,
            typed_unit_onset_association_mass=onset_association_mass,
            typed_unit_onset_identity_mass=onset_identity_mass,
            typed_unit_kind_index=trace.typed_unit_kind_index,
            typed_unit_electrode_index=trace.typed_unit_electrode_index,
            typed_unit_lead_endpoint_index=trace.typed_unit_lead_endpoint_index,
            physical_electrode_event_logits=physical_logits,
            physical_electrode_mask=physical_mask,
        )


__all__ = [
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID_V2",
    "BAIEGShallowCausalTypedUnitOnsetHeadV2",
]
