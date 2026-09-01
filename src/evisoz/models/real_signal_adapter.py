"""Real dual-montage signal-to-token adapters for the EviSOZ route.

The node path is the audited official LaBraM base encoder and therefore keeps
the frozen Standard19/CAR reference semantics.  TCP22 is deliberately a
separate edge encoder: its 22 signed bipolar channels remain edges throughout
the adapter and are never expanded into endpoint-node labels.

The default ``projection_mode='fixed_shadow'`` is deterministic and has no
trainable parameters.  ``projection_mode='learnable'`` is provided for a
future authorized Stage-1 run, but constructing it does not authorize
training; callers must still pass the aggregate Stage-0 guard first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.soz.models.labram import OfficialLaBraMEncoder


STANDARD19_COUNT = 19
TCP22_COUNT = 22
SAMPLE_RATE_HZ = 200
PATCH_SECONDS = 4
PATCH_SAMPLES = SAMPLE_RATE_HZ * PATCH_SECONDS
NODE_INPUT_DIM = 200
TOKEN_DIM = 128


@dataclass(frozen=True)
class RealSignalAdapterReceipt:
    """Machine-readable contract for one adapter construction."""

    source_interval_seconds: tuple[float, float]
    onset_start_seconds: float
    sample_rate_hz: int
    patch_seconds: int
    node_units: int
    edge_units: int
    node_encoder: str
    edge_encoder: str
    token_dim: int
    node_projection: str
    edge_projection: str
    edge_endpoint_expansion: bool
    trainable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_interval_seconds": list(self.source_interval_seconds),
            "onset_start_seconds": self.onset_start_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "patch_seconds": self.patch_seconds,
            "node_units": self.node_units,
            "edge_units": self.edge_units,
            "node_encoder": self.node_encoder,
            "edge_encoder": self.edge_encoder,
            "token_dim": self.token_dim,
            "node_projection": self.node_projection,
            "edge_projection": self.edge_projection,
            "edge_endpoint_expansion": self.edge_endpoint_expansion,
            "trainable": self.trainable,
        }


def _validate_waveform(value: Tensor, *, units: int, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,{units},N]")
    if value.shape[1] != units or value.shape[2] < PATCH_SAMPLES:
        raise ValueError(f"{name} must have shape [B,{units},>=800]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


def _validate_mask(value: Tensor, *, batch: int, units: int, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean [B,{units}]")
    if tuple(value.shape) != (batch, units):
        raise ValueError(f"{name} must be boolean [B,{units}]")
    return value


def project_token_dimension(value: Tensor, *, output_dim: int = TOKEN_DIM, mode: str = "fixed_shadow") -> Tensor:
    """Project LaBraM/patch features to the EviSOZ token dimension.

    ``fixed_shadow`` uses adaptive average pooling and is deterministic.  The
    learnable mode is intentionally a pure functional helper for callers that
    own a separately constructed ``nn.Linear``; it is rejected here to avoid
    silently creating unregistered trainable state.
    """

    if not isinstance(value, Tensor) or value.ndim < 1 or not value.is_floating_point():
        raise ValueError("token features must be floating point")
    if value.shape[-1] != NODE_INPUT_DIM:
        raise ValueError(f"token features must end in {NODE_INPUT_DIM}")
    if mode != "fixed_shadow":
        raise ValueError("functional projection only supports fixed_shadow")
    pooled = F.adaptive_avg_pool1d(value.reshape(-1, 1, NODE_INPUT_DIM), output_dim)
    return pooled.reshape(*value.shape[:-1], output_dim)


class RealDualMontageTokenAdapter(nn.Module):
    """Encode one real CAR19/TCP22 dual view with explicit masks."""

    def __init__(
        self,
        *,
        modeling_path: str,
        checkpoint_path: str,
        onset_start_seconds: float = 12.0,
        source_interval_seconds: tuple[float, float] = (-12.0, 48.0),
        projection_mode: str = "fixed_shadow",
    ) -> None:
        super().__init__()
        if projection_mode not in {"fixed_shadow", "learnable"}:
            raise ValueError("projection_mode must be fixed_shadow or learnable")
        if onset_start_seconds < 0:
            raise ValueError("onset_start_seconds must be non-negative")
        self.onset_start_seconds = float(onset_start_seconds)
        self.onset_start_samples = int(round(self.onset_start_seconds * SAMPLE_RATE_HZ))
        self.source_interval_seconds = tuple(float(x) for x in source_interval_seconds)
        self.projection_mode = projection_mode
        self.labram = OfficialLaBraMEncoder(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            tile_seconds=PATCH_SECONDS,
        )
        if projection_mode == "learnable":
            self.node_projection = nn.Linear(NODE_INPUT_DIM, TOKEN_DIM)
            self.edge_projection = nn.Linear(NODE_INPUT_DIM, TOKEN_DIM)
        else:
            self.node_projection = None
            self.edge_projection = None
        self.receipt = RealSignalAdapterReceipt(
            source_interval_seconds=self.source_interval_seconds,
            onset_start_seconds=self.onset_start_seconds,
            sample_rate_hz=SAMPLE_RATE_HZ,
            patch_seconds=PATCH_SECONDS,
            node_units=STANDARD19_COUNT,
            edge_units=TCP22_COUNT,
            node_encoder="official_labram_base_patch200_200",
            edge_encoder="signed_tcp22_independent_temporal_patch_encoder",
            token_dim=TOKEN_DIM,
            node_projection=("learnable_linear" if projection_mode == "learnable" else "fixed_adaptive_average_pool"),
            edge_projection=("learnable_linear" if projection_mode == "learnable" else "fixed_adaptive_average_pool"),
            edge_endpoint_expansion=False,
            trainable=projection_mode == "learnable",
        )

    def _crop(self, waveform: Tensor, *, units: int, name: str) -> Tensor:
        _validate_waveform(waveform, units=units, name=name)
        stop = self.onset_start_samples + PATCH_SAMPLES
        if waveform.shape[2] < stop:
            raise ValueError(f"{name} does not contain the configured onset window")
        return waveform[:, :, self.onset_start_samples : stop]

    def _project(self, value: Tensor, *, edge: bool) -> Tensor:
        projection = self.edge_projection if edge else self.node_projection
        if projection is None:
            return project_token_dimension(value)
        return projection(value)

    def encode_node_view(self, waveform: Tensor) -> Tensor:
        """Encode a batched Standard19/CAR waveform as node tokens."""

        node_window = self._crop(waveform, units=STANDARD19_COUNT, name="node_waveform")
        batch = node_window.shape[0]
        node_patches = node_window.reshape(batch, STANDARD19_COUNT, PATCH_SECONDS, SAMPLE_RATE_HZ)
        return self._project(self.labram(node_patches), edge=False)

    def encode_edge_view(self, waveform: Tensor) -> Tensor:
        """Encode a batched signed TCP22 waveform as independent edge tokens."""

        edge_window = self._crop(waveform, units=TCP22_COUNT, name="edge_waveform")
        batch = edge_window.shape[0]
        edge_patches = edge_window.reshape(batch, TCP22_COUNT, PATCH_SECONDS, SAMPLE_RATE_HZ)
        return self._project(edge_patches, edge=True)

    def forward(
        self,
        node_waveform: Tensor,
        edge_waveform: Tensor,
        node_observed_mask: Tensor,
        edge_observed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return node tokens, edge tokens and token-level masks."""

        _validate_waveform(node_waveform, units=STANDARD19_COUNT, name="node_waveform")
        _validate_waveform(edge_waveform, units=TCP22_COUNT, name="edge_waveform")
        batch = node_waveform.shape[0]
        if edge_waveform.shape[0] != batch:
            raise ValueError("node and edge batch sizes differ")
        node_observed_mask = _validate_mask(
            node_observed_mask, batch=batch, units=STANDARD19_COUNT, name="node_observed_mask"
        )
        edge_observed_mask = _validate_mask(
            edge_observed_mask, batch=batch, units=TCP22_COUNT, name="edge_observed_mask"
        )
        node_tokens = self.encode_node_view(node_waveform)
        edge_tokens = self.encode_edge_view(edge_waveform)
        # The edge route remains an edge route.  This shape is [B,22,4,128],
        # and no endpoint scatter or node-label projection is performed.
        node_mask = node_observed_mask.unsqueeze(-1).expand(batch, STANDARD19_COUNT, PATCH_SECONDS)
        edge_mask = edge_observed_mask.unsqueeze(-1).expand(batch, TCP22_COUNT, PATCH_SECONDS)
        return node_tokens, edge_tokens, node_mask, edge_mask


__all__ = [
    "NODE_INPUT_DIM",
    "PATCH_SAMPLES",
    "PATCH_SECONDS",
    "RealDualMontageTokenAdapter",
    "RealSignalAdapterReceipt",
    "SAMPLE_RATE_HZ",
    "STANDARD19_COUNT",
    "TCP22_COUNT",
    "TOKEN_DIM",
    "project_token_dimension",
]
