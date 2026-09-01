"""Permission-preserving typed-boundary MIL bridge for BA-IEG v2.

The frozen v1 supervision implementation validates the v1 trace/head identity
literally.  The v2 model intentionally has different identities because its
typed local trace is signal-only and its boundary head detaches global causal
hidden.  This additive bridge reuses the already tested v1 target projection
and noisy-or arithmetic on an internal identity-only compatibility view while
binding the public result to the original v2 axis.

No tensor is copied, detached, or numerically changed by the compatibility
view.  Consequently gradients keep their v2 authority: typed-boundary MIL may
reach the signal-only typed projection and boundary proposal adapter, but it
cannot reach the causal/offline segmental lanes or the identity/rank adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Final, Sequence

import torch

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    BAIEGCausalTypedUnitTrace,
)
from .ba_ieg_permission_split_segmental_state_model_v2 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2,
)
from .ba_ieg_permission_split_segmental_supervision_v1 import (
    BAIEGSegmentalEventTargetV1,
    BAIEGSegmentalLatticeTargetProjectionV1,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
    BAIEGShallowCausalTypedUnitHeadOutput,
    ba_ieg_causal_typed_unit_axis_receipt_sha256,
)
from .ba_ieg_shallow_causal_typed_unit_head_v2 import (
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID_V2,
)
from .ba_ieg_shallow_causal_typed_unit_supervision_v1 import (
    BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1,
    BAIEGShallowCausalTypedUnitMILTargetBundleV1,
    build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1,
    shallow_causal_typed_unit_mil_boundary_loss_v1,
)


BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID_V2: Final[str] = (
    "ba_ieg_signal_only_k3_typed_unit_boundary_mil_supervision_v2"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _axis_receipt_from_trace(trace: BAIEGCausalTypedUnitTrace) -> str:
    time_mask = trace.typed_unit_time_mask & trace.group_mask.unsqueeze(-1)
    unit_mask = trace.typed_unit_mask & time_mask.any(dim=1)
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


def _axis_receipt_from_output(
    output: BAIEGShallowCausalTypedUnitHeadOutput,
    *,
    source_trace_implementation_id: str,
) -> str:
    return ba_ieg_causal_typed_unit_axis_receipt_sha256(
        source_input_batch_sha256=output.source_input_batch_sha256,
        event_ids=output.event_ids,
        recording_ids=output.recording_ids,
        source_event_receipt_sha256s=output.source_event_receipt_sha256s,
        identity_roster_sha256=output.identity_roster_sha256,
        source_trace_implementation_id=source_trace_implementation_id,
        group_times_seconds=output.causal_group_times_seconds,
        group_boundary_bounds_seconds=(
            output.causal_group_boundary_bounds_seconds
        ),
        group_mask=output.causal_group_mask,
        typed_unit_time_mask=output.typed_unit_time_mask,
        typed_unit_mask=output.typed_unit_inventory_mask,
        typed_unit_kind_index=output.typed_unit_kind_index,
        typed_unit_electrode_index=output.typed_unit_electrode_index,
        typed_unit_lead_endpoint_index=output.typed_unit_lead_endpoint_index,
    )


def _identity_only_v1_trace(
    trace: BAIEGCausalTypedUnitTrace,
) -> BAIEGCausalTypedUnitTrace:
    payload = {item.name: getattr(trace, item.name) for item in fields(trace)}
    payload["implementation_id"] = BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID
    view = BAIEGCausalTypedUnitTrace(**payload)
    view.verify_shapes()
    return view


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitMILTargetBundleV2:
    """V1-compatible temporal targets bound to the signal-only v2 axis."""

    target_bundle: BAIEGShallowCausalTypedUnitMILTargetBundleV1
    source_trace_implementation_id: str
    causal_typed_unit_axis_receipt_sha256: str
    v1_compatibility_axis_receipt_sha256: str
    implementation_id: str = BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID_V2
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.target_bundle, BAIEGShallowCausalTypedUnitMILTargetBundleV1
        ):
            raise TypeError("v2 typed MIL requires a registered target bundle")
        self.target_bundle.verify_integrity()
        if self.source_trace_implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2:
            raise ValueError("v2 typed MIL target has the wrong trace identity")
        if (
            self.target_bundle.causal_typed_unit_axis_receipt_sha256
            != self.causal_typed_unit_axis_receipt_sha256
        ):
            raise ValueError("v2 typed MIL target axis binding drifted")
        expected = _canonical_sha256(
            {
                "implementation_id": self.implementation_id,
                "target_bundle_receipt_sha256": self.target_bundle.receipt_sha256,
                "source_trace_implementation_id": self.source_trace_implementation_id,
                "causal_typed_unit_axis_receipt_sha256": (
                    self.causal_typed_unit_axis_receipt_sha256
                ),
                "v1_compatibility_axis_receipt_sha256": (
                    self.v1_compatibility_axis_receipt_sha256
                ),
                "numeric_target_or_loss_changed_from_v1": False,
                "k3_mask_is_part_of_axis_receipt": True,
            }
        )
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("v2 typed MIL target receipt does not replay")
        object.__setattr__(self, "receipt_sha256", expected)


@dataclass(frozen=True)
class BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2:
    total_loss: torch.Tensor
    v1_equivalent: BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV1
    source_target_bundle_receipt_sha256: str
    causal_typed_unit_axis_receipt_sha256: str
    implementation_id: str = BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID_V2
    gradient_authority: str = (
        "typed_signal_projection_and_boundary_adapter_only_global_hidden_detached"
    )


def build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v2(
    trace: BAIEGCausalTypedUnitTrace,
    targets: Sequence[BAIEGSegmentalEventTargetV1],
    projections: Sequence[BAIEGSegmentalLatticeTargetProjectionV1],
) -> BAIEGShallowCausalTypedUnitMILTargetBundleV2:
    """Reuse target projection arithmetic while retaining the exact v2 axis."""

    if not isinstance(trace, BAIEGCausalTypedUnitTrace):
        raise TypeError("v2 typed MIL target builder requires a causal trace")
    trace.verify_shapes()
    if trace.implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2:
        raise ValueError("v2 typed MIL requires the signal-only v2 trace")
    v2_axis = _axis_receipt_from_trace(trace)
    legacy_trace = _identity_only_v1_trace(trace)
    legacy = build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v1(
        legacy_trace, targets, projections
    )
    # The event rows are invariant to implementation identity.  Only the
    # causal typed-unit axis receipt changes between the two permission lanes.
    rebound = BAIEGShallowCausalTypedUnitMILTargetBundleV1(
        source_input_batch_sha256=legacy.source_input_batch_sha256,
        identity_roster_sha256=legacy.identity_roster_sha256,
        causal_typed_unit_axis_receipt_sha256=v2_axis,
        source_context_receipt_sha256=legacy.source_context_receipt_sha256,
        target_independent_candidate_roster_receipt_sha256=(
            legacy.target_independent_candidate_roster_receipt_sha256
        ),
        event_targets=legacy.event_targets,
    )
    return BAIEGShallowCausalTypedUnitMILTargetBundleV2(
        target_bundle=rebound,
        source_trace_implementation_id=trace.implementation_id,
        causal_typed_unit_axis_receipt_sha256=v2_axis,
        v1_compatibility_axis_receipt_sha256=(
            legacy.causal_typed_unit_axis_receipt_sha256
        ),
    )


def shallow_causal_typed_unit_mil_boundary_loss_v2(
    output: BAIEGShallowCausalTypedUnitHeadOutput,
    target_bundle: BAIEGShallowCausalTypedUnitMILTargetBundleV2,
) -> BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2:
    """Run the frozen noisy-or objective without granting v1 neural identity."""

    if not isinstance(output, BAIEGShallowCausalTypedUnitHeadOutput):
        raise TypeError("v2 typed MIL loss requires a typed-unit head output")
    if not isinstance(
        target_bundle, BAIEGShallowCausalTypedUnitMILTargetBundleV2
    ):
        raise TypeError("v2 typed MIL loss requires its registered target bundle")
    if (
        output.implementation_id != BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID_V2
        or output.source_trace_implementation_id
        != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2
    ):
        raise ValueError("v2 typed MIL loss rejected a non-v2 head/trace")
    v2_axis = _axis_receipt_from_output(
        output,
        source_trace_implementation_id=BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2,
    )
    if (
        output.causal_typed_unit_axis_receipt_sha256 != v2_axis
        or target_bundle.causal_typed_unit_axis_receipt_sha256 != v2_axis
    ):
        raise ValueError("v2 typed MIL output/target axis receipt drifted")

    legacy_axis = _axis_receipt_from_output(
        output,
        source_trace_implementation_id=BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    )
    if legacy_axis != target_bundle.v1_compatibility_axis_receipt_sha256:
        raise ValueError("v2 typed MIL compatibility axis does not replay")
    output_payload = {
        item.name: getattr(output, item.name) for item in fields(output)
    }
    output_payload.update(
        implementation_id=BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
        source_trace_implementation_id=BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
        causal_typed_unit_axis_receipt_sha256=legacy_axis,
    )
    legacy_output = BAIEGShallowCausalTypedUnitHeadOutput(**output_payload)
    source = target_bundle.target_bundle
    legacy_target = BAIEGShallowCausalTypedUnitMILTargetBundleV1(
        source_input_batch_sha256=source.source_input_batch_sha256,
        identity_roster_sha256=source.identity_roster_sha256,
        causal_typed_unit_axis_receipt_sha256=legacy_axis,
        source_context_receipt_sha256=source.source_context_receipt_sha256,
        target_independent_candidate_roster_receipt_sha256=(
            source.target_independent_candidate_roster_receipt_sha256
        ),
        event_targets=source.event_targets,
    )
    loss = shallow_causal_typed_unit_mil_boundary_loss_v1(
        legacy_output, legacy_target
    )
    return BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2(
        total_loss=loss.total_loss,
        v1_equivalent=loss,
        source_target_bundle_receipt_sha256=target_bundle.receipt_sha256,
        causal_typed_unit_axis_receipt_sha256=v2_axis,
    )


__all__ = [
    "BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_SUPERVISION_ID_V2",
    "BAIEGShallowCausalTypedUnitMILBoundaryLossOutputV2",
    "BAIEGShallowCausalTypedUnitMILTargetBundleV2",
    "build_ba_ieg_shallow_causal_typed_unit_mil_target_bundle_v2",
    "shallow_causal_typed_unit_mil_boundary_loss_v2",
]
