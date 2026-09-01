"""Target-free conservative scalp-observability producer.

The producer accepts four pre-qualified reliability components on the
standard-19 carrier and projects to the DeepSOZ C18 endpoint.  It never reads
SOZ labels, localization scores, patient identity, clinical text, or private
data.  Missing required evidence fails closed and PZ never affects C18
completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

import torch

from .geometry import N_STANDARD_CHANNELS, N_TIME_TILES, STANDARD_19
from .v11_reasoner import V11_CANDIDATE_INDICES


SCALP_OBSERVABILITY_SCHEMA: Final[str] = "soz_scalp_observability_v1"
SCALP_OBSERVABILITY_COMPONENTS: Final[tuple[str, ...]] = (
    "signal_quality",
    "reference_stability",
    "channel_dropout_stability",
    "visibility_strength",
)
SCALP_OBSERVABILITY_AGGREGATION: Final[str] = (
    "fixed_unweighted_minimum_over_required_available_components_and_c18_tiles"
)
SCALP_OBSERVABILITY_USE_POLICY: Final[str] = (
    "target_free_reliability_downweight_or_abstain_only_never_positive_soz_support"
)
SCALP_OBSERVABILITY_CANDIDATE_CHANNELS: Final[tuple[str, ...]] = tuple(
    STANDARD_19[index] for index in V11_CANDIDATE_INDICES
)

_REASON_RE = re.compile(r"[a-z][a-z0-9_]*")
_MISSING_COMPONENT_REASON: Final[dict[str, str]] = {
    component: f"observability_{component}_missing"
    for component in SCALP_OBSERVABILITY_COMPONENTS
}


def _require_float_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached from all training losses")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _require_bool_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a torch.bool tensor")


def _project_c18(value: torch.Tensor) -> torch.Tensor:
    indices = torch.tensor(
        V11_CANDIDATE_INDICES, dtype=torch.long, device=value.device
    )
    return value.index_select(1, indices)


def _replay_reliability(
    components: torch.Tensor,
    available: torch.Tensor,
    required: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_available = available.all(dim=-1)
    node_tile = torch.where(
        required & all_available,
        components.amin(dim=-1),
        torch.zeros_like(components[..., 0]),
    )
    channel_has_required = required.any(dim=-1)
    channel = torch.where(
        required, node_tile, torch.ones_like(node_tile)
    ).amin(dim=-1)
    channel = torch.where(
        channel_has_required, channel, torch.zeros_like(channel)
    )
    event = channel.amin(dim=-1)
    return node_tile, channel, event


@dataclass(frozen=True)
class ScalpObservabilityResult:
    """Auditable, target-free C18 reliability receipt."""

    component_reliability: torch.Tensor
    component_available: torch.Tensor
    required_mask: torch.Tensor
    node_tile_reliability: torch.Tensor
    channel_reliability: torch.Tensor
    event_reliability: torch.Tensor
    abstain: torch.Tensor
    reason_codes: tuple[tuple[str, ...], ...]
    component_names: tuple[str, ...] = SCALP_OBSERVABILITY_COMPONENTS
    candidate_channels: tuple[str, ...] = SCALP_OBSERVABILITY_CANDIDATE_CHANNELS
    aggregation: str = SCALP_OBSERVABILITY_AGGREGATION
    calibrated: bool = False
    target_labels_used: bool = False
    private_data_used: bool = False
    localization_scores_used: bool = False
    training_performed: bool = False
    use_policy: str = SCALP_OBSERVABILITY_USE_POLICY
    schema_version: str = SCALP_OBSERVABILITY_SCHEMA

    def __post_init__(self) -> None:
        _require_float_tensor(
            self.component_reliability, name="component_reliability"
        )
        _require_bool_tensor(self.component_available, name="component_available")
        _require_bool_tensor(self.required_mask, name="required_mask")
        _require_float_tensor(
            self.node_tile_reliability, name="node_tile_reliability"
        )
        _require_float_tensor(
            self.channel_reliability, name="channel_reliability"
        )
        _require_float_tensor(self.event_reliability, name="event_reliability")
        _require_bool_tensor(self.abstain, name="abstain")

        if self.component_reliability.ndim != 4:
            raise ValueError("component_reliability must have shape [B,18,15,4]")
        batch, channels, tiles, components = self.component_reliability.shape
        expected = (
            batch,
            len(SCALP_OBSERVABILITY_CANDIDATE_CHANNELS),
            N_TIME_TILES,
            len(SCALP_OBSERVABILITY_COMPONENTS),
        )
        if tuple(self.component_reliability.shape) != expected:
            raise ValueError("component_reliability must have shape [B,18,15,4]")
        if tuple(self.component_available.shape) != expected:
            raise ValueError("component_available must have shape [B,18,15,4]")
        if tuple(self.required_mask.shape) != expected[:3]:
            raise ValueError("required_mask must have shape [B,18,15]")
        if tuple(self.node_tile_reliability.shape) != expected[:3]:
            raise ValueError("node_tile_reliability must have shape [B,18,15]")
        if tuple(self.channel_reliability.shape) != expected[:2]:
            raise ValueError("channel_reliability must have shape [B,18]")
        if tuple(self.event_reliability.shape) != (batch,):
            raise ValueError("event_reliability must have shape [B]")
        if tuple(self.abstain.shape) != (batch,):
            raise ValueError("abstain must have shape [B]")
        devices = {
            self.component_reliability.device,
            self.component_available.device,
            self.required_mask.device,
            self.node_tile_reliability.device,
            self.channel_reliability.device,
            self.event_reliability.device,
            self.abstain.device,
        }
        if len(devices) != 1:
            raise ValueError("all observability tensors must share one device")
        for name, value in (
            ("component_reliability", self.component_reliability),
            ("node_tile_reliability", self.node_tile_reliability),
            ("channel_reliability", self.channel_reliability),
            ("event_reliability", self.event_reliability),
        ):
            if torch.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must lie in [0,1]")

        replay = _replay_reliability(
            self.component_reliability,
            self.component_available,
            self.required_mask,
        )
        observed = (
            self.node_tile_reliability,
            self.channel_reliability,
            self.event_reliability,
        )
        if any(not torch.equal(left, right) for left, right in zip(replay, observed)):
            raise ValueError("observability reliability does not replay exactly")
        if self.component_names != SCALP_OBSERVABILITY_COMPONENTS:
            raise ValueError("observability component order is frozen")
        if self.candidate_channels != SCALP_OBSERVABILITY_CANDIDATE_CHANNELS:
            raise ValueError("observability candidate-channel order is frozen")
        if self.aggregation != SCALP_OBSERVABILITY_AGGREGATION:
            raise ValueError("observability aggregation is frozen")
        if self.use_policy != SCALP_OBSERVABILITY_USE_POLICY:
            raise ValueError("unsupported observability use policy")
        if self.schema_version != SCALP_OBSERVABILITY_SCHEMA:
            raise ValueError("unsupported observability schema")
        if type(self.calibrated) is not bool or self.calibrated:
            raise ValueError("observability is not independently calibrated")
        for name in (
            "target_labels_used",
            "private_data_used",
            "localization_scores_used",
            "training_performed",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise ValueError(
                    "observability must remain target/private/score/training free"
                )
        if not bool(self.abstain.all()):
            raise ValueError("uncalibrated observability must always abstain")
        if len(self.reason_codes) != batch:
            raise ValueError("reason_codes must contain one tuple per event")
        for reasons in self.reason_codes:
            if (
                not isinstance(reasons, tuple)
                or not reasons
                or len(set(reasons)) != len(reasons)
                or any(_REASON_RE.fullmatch(code) is None for code in reasons)
            ):
                raise ValueError("reason_codes must be non-empty unique stable tokens")
            if "observability_threshold_undefined" not in reasons:
                raise ValueError(
                    "uncalibrated observability requires threshold-undefined reason"
                )


def produce_scalp_observability(
    component_reliability: torch.Tensor,
    component_available: torch.Tensor,
    required_mask: torch.Tensor,
) -> ScalpObservabilityResult:
    """Produce fixed-minimum C18 reliability from ``[B,19,15,*]`` inputs."""

    _require_float_tensor(component_reliability, name="component_reliability")
    _require_bool_tensor(component_available, name="component_available")
    _require_bool_tensor(required_mask, name="required_mask")
    if component_reliability.ndim != 4:
        raise ValueError("component_reliability must have shape [B,19,15,4]")
    batch, channels, tiles, components = component_reliability.shape
    if (
        batch < 1
        or tiles != N_TIME_TILES
        or channels != N_STANDARD_CHANNELS
        or components != len(SCALP_OBSERVABILITY_COMPONENTS)
    ):
        raise ValueError("component_reliability must have shape [B,19,15,4]")
    if tuple(component_available.shape) != tuple(component_reliability.shape):
        raise ValueError("component_available must have shape [B,19,15,4]")
    if tuple(required_mask.shape) != (batch, N_STANDARD_CHANNELS, tiles):
        raise ValueError("required_mask must have shape [B,19,15]")
    if len(
        {
            component_reliability.device,
            component_available.device,
            required_mask.device,
        }
    ) != 1:
        raise ValueError("all observability inputs must share one device")
    if torch.any((component_reliability < 0.0) | (component_reliability > 1.0)):
        raise ValueError("component reliability must lie in [0,1]")

    projected_components = _project_c18(component_reliability).detach().contiguous()
    projected_available = _project_c18(component_available).detach().contiguous()
    projected_required = _project_c18(required_mask).detach().contiguous()
    node_tile, channel, event = _replay_reliability(
        projected_components, projected_available, projected_required
    )

    reason_rows: list[tuple[str, ...]] = []
    for event_index in range(batch):
        reasons: list[str] = []
        event_required = projected_required[event_index]
        if not bool(event_required.any()):
            reasons.append("observability_no_required_c18_observation")
        if bool((~event_required.any(dim=-1)).any()):
            reasons.append("observability_required_c18_channel_missing")
        for component_index, component_name in enumerate(
            SCALP_OBSERVABILITY_COMPONENTS
        ):
            missing = event_required & ~projected_available[
                event_index, ..., component_index
            ]
            if bool(missing.any()):
                reasons.append(_MISSING_COMPONENT_REASON[component_name])
        reasons.append("observability_threshold_undefined")
        reason_rows.append(tuple(reasons))

    return ScalpObservabilityResult(
        component_reliability=projected_components,
        component_available=projected_available,
        required_mask=projected_required,
        node_tile_reliability=node_tile.detach().contiguous(),
        channel_reliability=channel.detach().contiguous(),
        event_reliability=event.detach().contiguous(),
        abstain=torch.ones(batch, dtype=torch.bool, device=event.device),
        reason_codes=tuple(reason_rows),
    )


__all__ = [
    "SCALP_OBSERVABILITY_AGGREGATION",
    "SCALP_OBSERVABILITY_CANDIDATE_CHANNELS",
    "SCALP_OBSERVABILITY_COMPONENTS",
    "SCALP_OBSERVABILITY_SCHEMA",
    "SCALP_OBSERVABILITY_USE_POLICY",
    "ScalpObservabilityResult",
    "produce_scalp_observability",
]
