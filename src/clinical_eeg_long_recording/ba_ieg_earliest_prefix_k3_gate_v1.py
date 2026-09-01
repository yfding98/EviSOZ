"""Target-free earliest-prefix lock and K3 typed-unit gate for BA-IEG v1.3.

The v1.1 typed-unit head is causal at every cell, but its central onset
support and per-unit boundary softmax are computed over the complete acquired
event.  Consequently, a late hazard can still change which early cells enter
the normalized support.  This additive v1.3 development layer removes that
remaining whole-course normalization route without mutating the frozen v1.1
implementation.

For every event, the layer scans the already causal discrete onset masses in
physical-time order.  It locks the *first* prefix that satisfies a
source-development threshold and a maximum central-interval width.  The
locked prefix is never revised.  Typed-unit opportunity is then restricted to
cells overlapping the locked interval through three seconds after its right
edge (K3).  Later onset mass is converted back to prefix survival mass before
the frozen v1.1 head is called, so values after the K3 boundary cannot enter a
normalizer used by the primary rank.

This module implements a development shadow, not a calibrated threshold or an
admitted checkpoint.  The threshold receipt must be produced on source-dev;
the module never reads a public interval, channel target, EDF annotation,
spreadsheet, clinical text, private label, or report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Final

import torch
from torch import nn

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BAIEGCausalTypedUnitTrace,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS,
    BAIEGShallowCausalTypedUnitHeadOutput,
    BAIEGShallowCausalTypedUnitOnsetHead,
)


BA_IEG_EARLIEST_PREFIX_K3_GATE_ID: Final[str] = (
    "ba_ieg_earliest_prefix_locked_k3_typed_unit_gate_v1"
)
BA_IEG_PRIMARY_K3_HORIZON_SECONDS: Final[float] = 3.0
BA_IEG_EARLIEST_PREFIX_K3_PRIMARY_ADMITTED: Final[bool] = False
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BAIEGEarliestPrefixK3DevelopmentPolicyV1:
    """A content-bound, source-dev-only lock policy.

    Supplying a syntactically valid receipt does not admit a threshold.  The
    surrounding v1.3 machine contract remains the authority for promotion.
    """

    observed_onset_mass_threshold: float
    maximum_central_interval_width_seconds: float
    threshold_selection_receipt_sha256: str
    central_support_mass: float = BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS
    k3_horizon_seconds: float = BA_IEG_PRIMARY_K3_HORIZON_SECONDS
    threshold_selection_split: str = "source_dev"
    primary_admitted: bool = BA_IEG_EARLIEST_PREFIX_K3_PRIMARY_ADMITTED

    def __post_init__(self) -> None:
        threshold = float(self.observed_onset_mass_threshold)
        width = float(self.maximum_central_interval_width_seconds)
        if not math.isfinite(threshold) or not 0.5 < threshold < 1.0:
            raise ValueError("earliest-prefix threshold must lie strictly in (0.5,1)")
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("maximum central interval width must be positive")
        if self.central_support_mass != BA_IEG_CAUSAL_ONSET_CENTRAL_SUPPORT_MASS:
            raise ValueError("earliest-prefix central support must remain 0.95")
        if self.k3_horizon_seconds != BA_IEG_PRIMARY_K3_HORIZON_SECONDS:
            raise ValueError("primary typed-unit horizon must remain K3=3 seconds")
        if self.threshold_selection_split != "source_dev":
            raise ValueError("earliest-prefix thresholds must be selected on source-dev")
        if self.primary_admitted is not False:
            raise ValueError("v1.3 earliest-prefix/K3 primary is not admitted")
        if _SHA256.fullmatch(self.threshold_selection_receipt_sha256) is None:
            raise ValueError("threshold selection receipt must be a lowercase SHA-256")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "ba_ieg_earliest_prefix_k3_development_policy_v1",
                "implementation_id": BA_IEG_EARLIEST_PREFIX_K3_GATE_ID,
                "observed_onset_mass_threshold": float(
                    self.observed_onset_mass_threshold
                ),
                "maximum_central_interval_width_seconds": float(
                    self.maximum_central_interval_width_seconds
                ),
                "threshold_selection_receipt_sha256": (
                    self.threshold_selection_receipt_sha256
                ),
                "central_support_mass": float(self.central_support_mass),
                "k3_horizon_seconds": float(self.k3_horizon_seconds),
                "threshold_selection_split": self.threshold_selection_split,
                "primary_admitted": self.primary_admitted,
                "public_interval_or_localization_target_used_in_forward": False,
                "late_revision_of_locked_prefix_allowed": False,
            }
        )


@dataclass(frozen=True, slots=True)
class BAIEGEarliestPrefixK3GateResultV1:
    """The target-free lock decision and the trace visible to the v1.1 head."""

    policy_receipt_sha256: str
    locked_event_mask: torch.Tensor
    locked_prefix_group_index: torch.Tensor
    locked_prefix_time_seconds: torch.Tensor
    locked_onset_interval_seconds: torch.Tensor
    k3_interval_seconds: torch.Tensor
    locked_prefix_group_mask: torch.Tensor
    k3_group_mask: torch.Tensor
    k3_typed_unit_time_mask: torch.Tensor
    gated_trace: BAIEGCausalTypedUnitTrace
    implementation_id: str = BA_IEG_EARLIEST_PREFIX_K3_GATE_ID
    primary_admitted: bool = BA_IEG_EARLIEST_PREFIX_K3_PRIMARY_ADMITTED
    output_semantics: str = (
        "uncalibrated_source_development_shadow_not_primary_soz_probability"
    )

    def verify(self) -> None:
        self.gated_trace.verify_shapes()
        batch_size, group_count = self.gated_trace.group_mask.shape
        typed_count = int(self.gated_trace.typed_unit_mask.shape[1])
        expected = {
            "locked_event_mask": (self.locked_event_mask, (batch_size,), torch.bool),
            "locked_prefix_group_index": (
                self.locked_prefix_group_index,
                (batch_size,),
                torch.long,
            ),
            "locked_prefix_time_seconds": (
                self.locked_prefix_time_seconds,
                (batch_size,),
                None,
            ),
            "locked_onset_interval_seconds": (
                self.locked_onset_interval_seconds,
                (batch_size, 2),
                None,
            ),
            "k3_interval_seconds": (
                self.k3_interval_seconds,
                (batch_size, 2),
                None,
            ),
            "locked_prefix_group_mask": (
                self.locked_prefix_group_mask,
                (batch_size, group_count),
                torch.bool,
            ),
            "k3_group_mask": (
                self.k3_group_mask,
                (batch_size, group_count),
                torch.bool,
            ),
            "k3_typed_unit_time_mask": (
                self.k3_typed_unit_time_mask,
                (batch_size, group_count, typed_count),
                torch.bool,
            ),
        }
        for name, (value, shape, dtype) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape drifted")
            if dtype is not None and value.dtype != dtype:
                raise TypeError(f"{name} dtype drifted")
        if self.implementation_id != BA_IEG_EARLIEST_PREFIX_K3_GATE_ID:
            raise ValueError("earliest-prefix/K3 implementation identity drifted")
        if self.primary_admitted is not False:
            raise ValueError("earliest-prefix/K3 shadow was promoted")
        if _SHA256.fullmatch(self.policy_receipt_sha256) is None:
            raise ValueError("earliest-prefix/K3 policy receipt is invalid")
        unlocked = ~self.locked_event_mask
        if torch.any(self.locked_prefix_group_index[unlocked] != -1):
            raise ValueError("unlocked event has a prefix index")
        if torch.any(self.k3_group_mask & ~self.gated_trace.group_mask):
            raise ValueError("K3 opportunity exceeds the physical causal grid")
        if torch.any(
            self.k3_typed_unit_time_mask
            & ~self.gated_trace.typed_unit_time_mask
        ):
            raise ValueError("K3 result exceeds gated typed-unit opportunity")
        if torch.any(
            self.gated_trace.typed_unit_time_mask
            & ~self.k3_group_mask.unsqueeze(-1)
        ):
            raise ValueError("gated typed-unit trace exceeds K3")
        if torch.any(self.k3_group_mask[unlocked]) or torch.any(
            self.gated_trace.typed_unit_mask[unlocked]
        ):
            raise ValueError("unlocked event exposed positive K3 identity opportunity")


def _central_prefix_interval(
    masses: torch.Tensor,
    bounds: torch.Tensor,
    valid_prefix: torch.Tensor,
    *,
    central_support_mass: float,
) -> tuple[float, float] | None:
    selected_mass = torch.where(
        valid_prefix, masses.detach(), torch.zeros_like(masses)
    )
    total = selected_mass.sum()
    if not bool(total > 0):
        return None
    normalized = selected_mass / total
    cumulative = torch.cumsum(normalized, dim=0)
    cumulative_before = cumulative - normalized
    tail = 0.5 * (1.0 - central_support_mass)
    support = (
        valid_prefix
        & (normalized > 0)
        & (cumulative > tail)
        & (cumulative_before < 1.0 - tail)
    )
    selected = torch.nonzero(support, as_tuple=False).flatten()
    if not bool(selected.numel()):
        return None
    start = float(bounds[int(selected[0]), 0])
    stop = float(bounds[int(selected[-1]), 1])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        return None
    return start, stop


def build_ba_ieg_earliest_prefix_k3_gate_v1(
    trace: BAIEGCausalTypedUnitTrace,
    policy: BAIEGEarliestPrefixK3DevelopmentPolicyV1,
) -> BAIEGEarliestPrefixK3GateResultV1:
    """Lock the first admissible causal prefix and remove all later rank paths."""

    if not isinstance(trace, BAIEGCausalTypedUnitTrace):
        raise TypeError("earliest-prefix/K3 gate accepts only a causal trace")
    if not isinstance(policy, BAIEGEarliestPrefixK3DevelopmentPolicyV1):
        raise TypeError("earliest-prefix/K3 gate requires a validated policy")
    trace.verify_shapes()
    batch_size, group_count = trace.group_mask.shape
    device = trace.group_mask.device
    time_dtype = trace.group_boundary_bounds_seconds.dtype

    locked = torch.zeros(batch_size, dtype=torch.bool, device=device)
    prefix_index = torch.full(
        (batch_size,), -1, dtype=torch.long, device=device
    )
    prefix_time = torch.zeros(batch_size, dtype=time_dtype, device=device)
    onset_interval = torch.zeros(
        (batch_size, 2), dtype=time_dtype, device=device
    )
    k3_interval = torch.zeros_like(onset_interval)
    prefix_group_mask = torch.zeros_like(trace.group_mask)
    k3_group_mask = torch.zeros_like(trace.group_mask)

    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(
            trace.group_mask[batch_index], as_tuple=False
        ).flatten()
        if not bool(valid_indices.numel()):
            continue
        running_mass = 0.0
        for index_tensor in valid_indices:
            index = int(index_tensor)
            running_mass += float(
                trace.global_onset_boundary_mass[batch_index, index].detach()
            )
            if running_mass < policy.observed_onset_mass_threshold:
                continue
            current_prefix = trace.group_mask[batch_index].clone()
            current_prefix &= torch.arange(group_count, device=device) <= index
            interval = _central_prefix_interval(
                trace.global_onset_boundary_mass[batch_index],
                trace.group_boundary_bounds_seconds[batch_index],
                current_prefix,
                central_support_mass=policy.central_support_mass,
            )
            if interval is None:
                continue
            start, stop = interval
            if stop - start > policy.maximum_central_interval_width_seconds:
                continue
            locked[batch_index] = True
            prefix_index[batch_index] = index
            prefix_time[batch_index] = trace.group_times_seconds[
                batch_index, index
            ]
            onset_interval[batch_index] = torch.tensor(
                (start, stop), dtype=time_dtype, device=device
            )
            k3_stop = stop + policy.k3_horizon_seconds
            k3_interval[batch_index] = torch.tensor(
                (start, k3_stop), dtype=time_dtype, device=device
            )
            prefix_group_mask[batch_index] = current_prefix
            bounds = trace.group_boundary_bounds_seconds[batch_index]
            positive_overlap = (bounds[:, 1] > start) & (bounds[:, 0] < k3_stop)
            k3_group_mask[batch_index] = (
                trace.group_mask[batch_index] & positive_overlap
            )
            break

    gated_onset_mass = torch.where(
        prefix_group_mask,
        trace.global_onset_boundary_mass,
        torch.zeros_like(trace.global_onset_boundary_mass),
    )
    # Reconstruct prefix survival from prefix-stable mass only.  The final
    # whole-course no-onset value and every later hazard are deliberately
    # ignored, which is the key future-perturbation invariant.
    gated_no_onset = (
        torch.ones_like(trace.global_no_onset_within_support_mass)
        - trace.global_left_censor_state_mass.sum(dim=1)
        - gated_onset_mass.sum(dim=1)
    ).clamp_min(0.0)
    k3_typed_time_mask = (
        trace.typed_unit_time_mask & k3_group_mask.unsqueeze(-1)
    )
    gated_trace = replace(
        trace,
        global_onset_boundary_mass=gated_onset_mass,
        global_no_onset_within_support_mass=gated_no_onset,
        typed_unit_time_mask=k3_typed_time_mask,
        typed_unit_mask=k3_typed_time_mask.any(dim=1),
    )
    gated_trace.verify_shapes()
    result = BAIEGEarliestPrefixK3GateResultV1(
        policy_receipt_sha256=policy.receipt_sha256,
        locked_event_mask=locked,
        locked_prefix_group_index=prefix_index,
        locked_prefix_time_seconds=prefix_time,
        locked_onset_interval_seconds=onset_interval,
        k3_interval_seconds=k3_interval,
        locked_prefix_group_mask=prefix_group_mask,
        k3_group_mask=k3_group_mask,
        k3_typed_unit_time_mask=k3_typed_time_mask,
        gated_trace=gated_trace,
    )
    result.verify()
    return result


@dataclass(frozen=True, slots=True)
class BAIEGEarliestPrefixK3TypedUnitHeadOutputV1:
    gate: BAIEGEarliestPrefixK3GateResultV1
    typed_unit: BAIEGShallowCausalTypedUnitHeadOutput
    implementation_id: str = BA_IEG_EARLIEST_PREFIX_K3_GATE_ID
    primary_admitted: bool = BA_IEG_EARLIEST_PREFIX_K3_PRIMARY_ADMITTED


class BAIEGEarliestPrefixK3TypedUnitOnsetHeadV1(nn.Module):
    """Composition that makes the K3 gate mandatory before v1.1 scoring."""

    def __init__(
        self,
        *,
        policy: BAIEGEarliestPrefixK3DevelopmentPolicyV1,
        hidden_dim: int = 64,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(policy, BAIEGEarliestPrefixK3DevelopmentPolicyV1):
            raise TypeError("K3 typed-unit head requires a validated lock policy")
        self.policy = policy
        self.typed_unit_head = BAIEGShallowCausalTypedUnitOnsetHead(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

    def forward(
        self, trace: BAIEGCausalTypedUnitTrace
    ) -> BAIEGEarliestPrefixK3TypedUnitHeadOutputV1:
        gate = build_ba_ieg_earliest_prefix_k3_gate_v1(trace, self.policy)
        typed_unit = self.typed_unit_head(gate.gated_trace)
        return BAIEGEarliestPrefixK3TypedUnitHeadOutputV1(
            gate=gate,
            typed_unit=typed_unit,
        )


__all__ = [
    "BA_IEG_EARLIEST_PREFIX_K3_GATE_ID",
    "BA_IEG_EARLIEST_PREFIX_K3_PRIMARY_ADMITTED",
    "BA_IEG_PRIMARY_K3_HORIZON_SECONDS",
    "BAIEGEarliestPrefixK3DevelopmentPolicyV1",
    "BAIEGEarliestPrefixK3GateResultV1",
    "BAIEGEarliestPrefixK3TypedUnitHeadOutputV1",
    "BAIEGEarliestPrefixK3TypedUnitOnsetHeadV1",
    "build_ba_ieg_earliest_prefix_k3_gate_v1",
]
