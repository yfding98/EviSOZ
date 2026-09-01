"""TFM-motif mixers for leakage-safe standard-19 SOZ localization.

The module deliberately separates three roles:

* a frozen, label-free TFM front end extracts one motif sequence per channel;
* one interchangeable mixer is selected from ``head_only``, ``timefilter``,
  or ``cbramod``;
* a shared channel head produces finite standard-19 logits while deployment
  ranking always removes canonical PZ through the fixed C18 candidate mask.

TimeFilter is adapted as a sparse relation filter rather than copied from the
forecasting implementation.  Its temporal, spatial, and spatial--temporal
branches operate on the same ``[B,19,T,D]`` motif carrier.  Spatial support is
the frozen 34-edge standard 10--20 graph; learned edges are an explicit number
of *additional* edges, not a misleading total top-k cap.  Optional wPLI,
Granger, and transfer-entropy features enter only as zero-initialized edge
biases.  The CBraMod arm uses the same carrier, depth, width, physical support,
and output head, changing only the mixer to masked criss-cross attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..aggregation import PatientAggregation, aggregate_patient_logits
from ..evolution import STANDARD19_NEIGHBOR_EDGES
from ..geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19
from ..v11_reasoner import V11_CANDIDATE_MASK, positive_set_mass_loss


C18_INDICES: Final[tuple[int, ...]] = tuple(
    index for index, channel in enumerate(STANDARD_19) if channel != "PZ"
)
C18_CHANNELS: Final[tuple[str, ...]] = tuple(STANDARD_19[index] for index in C18_INDICES)
DEFAULT_CONNECTIVITY_KINDS: Final[tuple[str, ...]] = ("wpli", "granger", "te")
VALID_ENCODERS: Final[tuple[str, ...]] = ("head_only", "timefilter", "cbramod")


def standard19_physical_support(*, include_self: bool = True) -> torch.Tensor:
    """Return the frozen 34-edge standard 10--20 physical support.

    This must not be replaced by the denser DeepSOZ relaxed-evaluation
    neighbourhood.  The latter is an outcome-scoring tolerance, not a
    label-independent anatomical training graph.
    """

    support = torch.eye(N_STANDARD_CHANNELS, dtype=torch.bool) if include_self else torch.zeros(
        (N_STANDARD_CHANNELS, N_STANDARD_CHANNELS), dtype=torch.bool
    )
    for left, right in STANDARD19_NEIGHBOR_EDGES:
        left_index = CHANNEL_INDEX[left]
        right_index = CHANNEL_INDEX[right]
        support[left_index, right_index] = True
        support[right_index, left_index] = True
    return support


@dataclass(frozen=True)
class TFMMotifTokens:
    """Frozen TFM output aligned to the standard-19 carrier."""

    token_ids: torch.Tensor
    embeddings: torch.Tensor
    channel_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 3 or self.token_ids.shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("TFM token IDs must have shape [B,19,T]")
        batch, channels, n_time = self.token_ids.shape
        if self.token_ids.dtype != torch.long:
            raise TypeError("TFM token IDs must be torch.long")
        if self.embeddings.ndim != 4 or tuple(self.embeddings.shape[:3]) != (
            batch,
            channels,
            n_time,
        ):
            raise ValueError("TFM embeddings must have shape [B,19,T,D]")
        if not self.embeddings.is_floating_point() or not torch.isfinite(
            self.embeddings
        ).all():
            raise ValueError("TFM embeddings must be finite floating point")
        if tuple(self.channel_mask.shape) != (batch, channels) or (
            self.channel_mask.dtype != torch.bool
        ):
            raise TypeError("TFM channel_mask must be bool [B,19]")
        devices = {
            self.token_ids.device,
            self.embeddings.device,
            self.channel_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("TFM motif tensors must share a device")


class FrozenTFMMotifExtractor(nn.Module):
    """Adapt a trained ``TFMTokenizer`` to standard-19 raw waveforms.

    The wrapped tokenizer is always frozen and kept in evaluation mode.  A
    60-second, 200-Hz input produces 119 half-overlapping motif tokens with
    the historical TFM settings (200-sample window, 100-sample hop).
    """

    def __init__(self, tokenizer: nn.Module, *, sample_rate_hz: int = 200) -> None:
        super().__init__()
        if isinstance(sample_rate_hz, bool) or int(sample_rate_hz) != 200:
            raise ValueError("the audited TFM tokenizer contract requires 200 Hz")
        for attribute in ("tokenize", "code_book_size"):
            if not hasattr(tokenizer, attribute):
                raise TypeError(f"TFM tokenizer is missing {attribute!r}")
        self.tokenizer = tokenizer
        self.sample_rate_hz = int(sample_rate_hz)
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

    @property
    def code_book_size(self) -> int:
        return int(getattr(self.tokenizer, "code_book_size"))

    def train(self, mode: bool = True) -> FrozenTFMMotifExtractor:
        super().train(mode)
        self.tokenizer.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        *,
        channel_mask: torch.Tensor | None = None,
    ) -> TFMMotifTokens:
        if not waveform.is_floating_point() or waveform.ndim != 3 or (
            waveform.shape[1] != N_STANDARD_CHANNELS
        ):
            raise ValueError("TFM waveform must be floating point [B,19,S]")
        batch, channels, _ = waveform.shape
        if channel_mask is None:
            valid_channels = torch.ones(
                (batch, channels), dtype=torch.bool, device=waveform.device
            )
        else:
            if tuple(channel_mask.shape) != (batch, channels) or (
                channel_mask.dtype != torch.bool
            ):
                raise TypeError("channel_mask must be bool [B,19]")
            if channel_mask.device != waveform.device:
                raise ValueError("waveform and channel_mask must share a device")
            valid_channels = channel_mask
        if not bool(valid_channels.any(dim=1).all()):
            raise ValueError("every waveform must contain at least one signal channel")
        valid_samples = valid_channels.unsqueeze(-1).expand_as(waveform)
        if not torch.isfinite(waveform[valid_samples]).all():
            raise ValueError("available waveform channels must be finite")
        signal = torch.where(valid_samples, waveform, torch.zeros_like(waveform))

        flat = signal.reshape(batch * channels, signal.shape[-1])
        window = torch.hann_window(
            self.sample_rate_hz, dtype=flat.dtype, device=flat.device
        )
        spectrum = torch.stft(
            flat,
            n_fft=self.sample_rate_hz,
            hop_length=self.sample_rate_hz // 2,
            onesided=True,
            return_complex=True,
            center=False,
            window=window,
        ).abs()[:, : self.sample_rate_hz // 2, :]
        quantized, indices, _ = self.tokenizer.tokenize(spectrum, flat)
        if indices.ndim != 2 or quantized.ndim != 3 or (
            tuple(indices.shape) != tuple(quantized.shape[:2])
        ):
            raise RuntimeError("TFM tokenizer returned an unsupported motif shape")
        n_time = int(indices.shape[1])
        token_ids = indices.reshape(batch, channels, n_time).long()
        embeddings = quantized.reshape(
            batch, channels, n_time, quantized.shape[-1]
        ).contiguous()
        token_ids = token_ids.masked_fill(
            ~valid_channels.unsqueeze(-1), self.code_book_size
        )
        embeddings = embeddings * valid_channels[:, :, None, None].to(
            embeddings.dtype
        )
        return TFMMotifTokens(
            token_ids=token_ids.contiguous(),
            embeddings=embeddings,
            channel_mask=valid_channels,
        )


@dataclass(frozen=True)
class CanonicalConnectivity:
    """Connectivity values after mask-aware semantic normalization."""

    values: torch.Tensor
    mask: torch.Tensor
    kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[2:4] != (
            N_STANDARD_CHANNELS,
            N_STANDARD_CHANNELS,
        ):
            raise ValueError("connectivity values must be [B,T,19,19,K]")
        if tuple(self.mask.shape) != tuple(self.values.shape) or (
            self.mask.dtype != torch.bool
        ):
            raise TypeError("connectivity mask must be bool and match values")
        if self.values.shape[-1] != len(self.kinds):
            raise ValueError("connectivity kind count does not match feature axis")
        if not self.values.is_floating_point() or not torch.isfinite(self.values).all():
            raise ValueError("canonical connectivity must be finite floating point")


def _expand_connectivity_time(value: torch.Tensor, n_time: int) -> torch.Tensor:
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5:
        raise ValueError("connectivity must be [B,19,19,K] or [B,T,19,19,K]")
    if value.shape[1] == 1 and n_time != 1:
        value = value.expand(-1, n_time, -1, -1, -1)
    elif value.shape[1] != n_time:
        raise ValueError("connectivity time axis must be 1 or match motif T")
    return value


def canonicalize_connectivity(
    connectivity: torch.Tensor,
    *,
    n_time: int,
    channel_mask: torch.Tensor,
    time_mask: torch.Tensor,
    kinds: Sequence[str] = DEFAULT_CONNECTIVITY_KINDS,
    connectivity_mask: torch.Tensor | None = None,
) -> CanonicalConnectivity:
    """Normalize connectivity without destroying directed feature semantics.

    wPLI is made mask-aware symmetric.  Granger and transfer entropy retain
    their destination/source orientation.  Missing or non-finite *masked*
    cells become zero and never contribute an edge bias; an available
    non-finite cell is rejected.
    """

    if not connectivity.is_floating_point():
        raise TypeError("connectivity must be floating point")
    normalized_kinds = tuple(str(value).strip().lower() for value in kinds)
    if not normalized_kinds or len(set(normalized_kinds)) != len(normalized_kinds):
        raise ValueError("connectivity kinds must be non-empty and unique")
    unknown = set(normalized_kinds).difference(DEFAULT_CONNECTIVITY_KINDS)
    if unknown:
        raise ValueError(f"unsupported connectivity kinds: {sorted(unknown)}")
    values = _expand_connectivity_time(connectivity, n_time)
    batch = int(values.shape[0])
    expected_tail = (N_STANDARD_CHANNELS, N_STANDARD_CHANNELS, len(normalized_kinds))
    if tuple(values.shape[2:]) != expected_tail:
        raise ValueError(f"connectivity must end in {expected_tail}")
    if tuple(channel_mask.shape) != (batch, N_STANDARD_CHANNELS) or (
        channel_mask.dtype != torch.bool
    ):
        raise TypeError("channel_mask must be bool [B,19]")
    if tuple(time_mask.shape) != (batch, n_time) or time_mask.dtype != torch.bool:
        raise TypeError("time_mask must be bool [B,T]")
    if values.device != channel_mask.device or values.device != time_mask.device:
        raise ValueError("connectivity and signal masks must share a device")

    if connectivity_mask is None:
        available = torch.isfinite(values)
    else:
        expanded_mask = _expand_connectivity_time(connectivity_mask, n_time)
        if expanded_mask.dtype != torch.bool or tuple(expanded_mask.shape) != tuple(
            values.shape
        ):
            raise TypeError("connectivity_mask must be bool and match connectivity")
        if expanded_mask.device != values.device:
            raise ValueError("connectivity and connectivity_mask must share a device")
        if not torch.isfinite(values[expanded_mask]).all():
            raise ValueError("available connectivity cells must be finite")
        available = expanded_mask
    values = torch.where(available, values, torch.zeros_like(values))

    output_values: list[torch.Tensor] = []
    output_masks: list[torch.Tensor] = []
    for feature_index, kind in enumerate(normalized_kinds):
        feature = values[..., feature_index]
        feature_mask = available[..., feature_index]
        if kind == "wpli":
            transposed_feature = feature.transpose(-1, -2)
            transposed_mask = feature_mask.transpose(-1, -2)
            count = feature_mask.to(feature.dtype) + transposed_mask.to(feature.dtype)
            feature = (
                feature * feature_mask.to(feature.dtype)
                + transposed_feature * transposed_mask.to(feature.dtype)
            ) / count.clamp_min(1.0)
            feature_mask = count > 0
        output_values.append(feature)
        output_masks.append(feature_mask)
    canonical_values = torch.stack(output_values, dim=-1)
    canonical_mask = torch.stack(output_masks, dim=-1)

    node_pair = channel_mask[:, None, :, None] & channel_mask[:, None, None, :]
    valid = node_pair.unsqueeze(-1) & time_mask[:, :, None, None, None]
    off_diagonal = ~torch.eye(
        N_STANDARD_CHANNELS, dtype=torch.bool, device=values.device
    ).view(1, 1, N_STANDARD_CHANNELS, N_STANDARD_CHANNELS, 1)
    canonical_mask = canonical_mask & valid & off_diagonal
    canonical_values = torch.where(
        canonical_mask, canonical_values, torch.zeros_like(canonical_values)
    )
    return CanonicalConnectivity(
        values=canonical_values.contiguous(),
        mask=canonical_mask.contiguous(),
        kinds=normalized_kinds,
    )


def _validate_signal_masks(
    *,
    batch: int,
    n_time: int,
    device: torch.device,
    channel_mask: torch.Tensor | None,
    time_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channel_mask is None:
        valid_channels = torch.ones(
            (batch, N_STANDARD_CHANNELS), dtype=torch.bool, device=device
        )
    else:
        if tuple(channel_mask.shape) != (batch, N_STANDARD_CHANNELS) or (
            channel_mask.dtype != torch.bool
        ):
            raise TypeError("signal_channel_mask must be bool [B,19]")
        if channel_mask.device != device:
            raise ValueError("motifs and signal_channel_mask must share a device")
        valid_channels = channel_mask
    if time_mask is None:
        valid_time = torch.ones((batch, n_time), dtype=torch.bool, device=device)
    else:
        if tuple(time_mask.shape) != (batch, n_time) or time_mask.dtype != torch.bool:
            raise TypeError("time_mask must be bool [B,T]")
        if time_mask.device != device:
            raise ValueError("motifs and time_mask must share a device")
        valid_time = time_mask
    if not bool(valid_channels.any(dim=1).all()):
        raise ValueError("every example must contain at least one signal channel")
    if not bool(valid_time.any(dim=1).all()):
        raise ValueError("every example must contain at least one motif token")
    return valid_channels, valid_time


def _sinusoidal_time_encoding(
    n_time: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(n_time, device=device, dtype=torch.float32).unsqueeze(1)
    divider = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros((n_time, dimension), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divider)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * divider[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).view(1, 1, n_time, dimension)


class HeadOnlyMotifEncoder(nn.Module):
    """Identity control for the shared TFM motif carrier."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del channel_mask, time_mask, connectivity
        return x, {}


class TimeFilterMotifLayer(nn.Module):
    """Sparse temporal/spatial/ST relation filter for one motif layer."""

    def __init__(
        self,
        *,
        dimension: int,
        num_heads: int,
        physical_support: torch.Tensor,
        dynamic_topk_extra: int,
        connectivity_kinds: Sequence[str],
        connectivity_allow_nonphysical: bool,
        dropout: float,
        residual_init: float,
        ff_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if dynamic_topk_extra < 0 or dynamic_topk_extra >= N_STANDARD_CHANNELS:
            raise ValueError("dynamic_topk_extra must lie in [0,18]")
        self.dimension = int(dimension)
        self.dynamic_topk_extra = int(dynamic_topk_extra)
        graph_dimension = max(8, dimension // 4)
        self.input_norm = nn.LayerNorm(dimension)
        self.temporal_attention = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.graph_query = nn.Linear(dimension, graph_dimension, bias=False)
        self.graph_key = nn.Linear(dimension, graph_dimension, bias=False)
        self.graph_value = nn.Linear(dimension, dimension, bias=False)
        self.graph_projection = nn.Linear(dimension, dimension)
        self.cross_temporal = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=3,
            padding=1,
            groups=dimension,
        )
        self.cross_projection = nn.Linear(dimension, dimension)
        router_hidden = max(16, dimension // 2)
        self.router = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 3),
        )
        self.branch_projection = nn.Linear(dimension, dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        hidden = dimension * int(ff_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        normalized_kinds = tuple(str(value).strip().lower() for value in connectivity_kinds)
        self.connectivity_kinds = normalized_kinds
        self.connectivity_allow_nonphysical = bool(connectivity_allow_nonphysical)
        self.connectivity_scales = nn.Parameter(torch.zeros(len(normalized_kinds)))
        self.register_buffer(
            "physical_support", physical_support.to(dtype=torch.bool).clone(), persistent=True
        )

    def _spatial_branch(
        self,
        z: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, n_time, _ = z.shape
        graph_features = z.permute(0, 2, 1, 3)
        query = self.graph_query(graph_features)
        key = self.graph_key(graph_features)
        scores = torch.einsum("btid,btjd->btij", query, key) / math.sqrt(
            float(query.shape[-1])
        )
        has_connectivity_signal = connectivity is not None and bool(
            (connectivity.mask & connectivity.values.ne(0)).any()
        )
        if has_connectivity_signal:
            assert connectivity is not None
            if connectivity.kinds != self.connectivity_kinds:
                raise ValueError("connectivity kinds do not match TimeFilter configuration")
            edge_values = torch.tanh(connectivity.values)
            edge_weights = torch.tanh(self.connectivity_scales).view(1, 1, 1, 1, -1)
            edge_bias = (
                edge_values
                * connectivity.mask.to(edge_values.dtype)
                * edge_weights
            ).sum(dim=-1)
            if not self.connectivity_allow_nonphysical:
                edge_bias = edge_bias * self.physical_support.view(
                    1, 1, self.physical_support.shape[0], self.physical_support.shape[1]
                ).to(edge_bias.dtype)
            scores = scores + edge_bias

        valid_pair = (
            channel_mask[:, None, :, None]
            & channel_mask[:, None, None, :]
            & time_mask[:, :, None, None]
        )
        physical = self.physical_support.view(1, 1, channels, channels) & valid_pair
        dynamic = torch.zeros_like(valid_pair)
        if self.dynamic_topk_extra > 0:
            eligible = valid_pair & ~physical
            selection_scores = scores.masked_fill(
                ~eligible, -torch.finfo(scores.dtype).max
            )
            indices = selection_scores.topk(
                k=self.dynamic_topk_extra, dim=-1
            ).indices
            dynamic = torch.zeros_like(scores, dtype=torch.bool)
            dynamic.scatter_(-1, indices, True)
            dynamic = dynamic & eligible
        candidates = physical | dynamic
        masked_scores = scores.masked_fill(
            ~candidates, -torch.finfo(scores.dtype).max
        )
        adjacency = torch.softmax(masked_scores, dim=-1)
        adjacency = adjacency * candidates.to(adjacency.dtype)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        values = self.graph_value(graph_features)
        spatial = torch.einsum("btij,btjd->btid", adjacency, values)
        spatial = F.gelu(self.graph_projection(spatial)).permute(0, 2, 1, 3)
        valid_tokens = (
            channel_mask[:, :, None, None] & time_mask[:, None, :, None]
        )
        spatial = spatial * valid_tokens.to(spatial.dtype)
        physical_density = physical.to(scores.dtype).sum(dim=(-1, -2))
        dynamic_density = dynamic.to(scores.dtype).sum(dim=(-1, -2))
        return spatial, physical_density, dynamic_density

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, n_time, dimension = x.shape
        if channels != N_STANDARD_CHANNELS or dimension != self.dimension:
            raise ValueError("TimeFilter input must be [B,19,T,D]")
        valid_tokens = channel_mask[:, :, None, None] & time_mask[:, None, :, None]
        z = self.input_norm(x) * valid_tokens.to(x.dtype)

        temporal_sequence = z.reshape(batch * channels, n_time, dimension)
        temporal_padding = (
            ~time_mask[:, None, :]
            .expand(batch, channels, n_time)
            .reshape(batch * channels, n_time)
        )
        temporal = self.temporal_attention(
            temporal_sequence,
            temporal_sequence,
            temporal_sequence,
            key_padding_mask=temporal_padding,
            need_weights=False,
        )[0].reshape(batch, channels, n_time, dimension)
        temporal = temporal * valid_tokens.to(temporal.dtype)

        spatial, physical_edges, dynamic_edges = self._spatial_branch(
            z,
            channel_mask=channel_mask,
            time_mask=time_mask,
            connectivity=connectivity,
        )
        cross = self.cross_temporal(
            spatial.reshape(batch * channels, n_time, dimension).transpose(1, 2)
        ).transpose(1, 2)
        cross = F.gelu(self.cross_projection(cross)).reshape(
            batch, channels, n_time, dimension
        )
        cross = cross * valid_tokens.to(cross.dtype)

        router_weights = torch.softmax(
            self.router(torch.cat((z, spatial), dim=-1)), dim=-1
        )
        branches = torch.stack((temporal, spatial, cross), dim=-2)
        mixed = (branches * router_weights.unsqueeze(-1)).sum(dim=-2)
        projected = self.branch_projection(mixed)
        correction = projected + self.ffn(self.ffn_norm(z + projected))
        correction = correction * valid_tokens.to(correction.dtype)
        gate = torch.tanh(self.residual_scale)
        output = x + gate * self.dropout(correction)
        output = output * valid_tokens.to(output.dtype)
        diagnostics = {
            "residual_scale": gate,
            "router_mean": (
                router_weights * valid_tokens.to(router_weights.dtype)
            ).sum(dim=(1, 2))
            / valid_tokens.to(router_weights.dtype).sum(dim=(1, 2)).clamp_min(1.0),
            "physical_edge_count": physical_edges,
            "dynamic_edge_count": dynamic_edges,
            "connectivity_scales": torch.tanh(self.connectivity_scales),
        }
        return output, diagnostics


class TimeFilterMotifEncoder(nn.Module):
    def __init__(
        self,
        *,
        dimension: int,
        num_heads: int,
        depth: int,
        physical_support: torch.Tensor,
        dynamic_topk_extra: int,
        connectivity_kinds: Sequence[str],
        connectivity_allow_nonphysical: bool,
        dropout: float,
        residual_init: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TimeFilterMotifLayer(
                    dimension=dimension,
                    num_heads=num_heads,
                    physical_support=physical_support,
                    dynamic_topk_extra=dynamic_topk_extra,
                    connectivity_kinds=connectivity_kinds,
                    connectivity_allow_nonphysical=connectivity_allow_nonphysical,
                    dropout=dropout,
                    residual_init=residual_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        statistics: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            x, layer_statistics = layer(
                x,
                channel_mask=channel_mask,
                time_mask=time_mask,
                connectivity=connectivity,
            )
            statistics.update(
                {f"layer{index}.{name}": value for name, value in layer_statistics.items()}
            )
        return x, statistics


class PhysicalAdaptivePositionalEncoding(nn.Module):
    """CBraMod-style ACPE without pretending channel indices are distances."""

    def __init__(
        self, dimension: int, physical_support: torch.Tensor, temporal_kernel: int = 7
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            dimension,
            dimension,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            groups=dimension,
        )
        self.graph_scale = nn.Parameter(torch.zeros(dimension))
        adjacency = physical_support.float()
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("normalized_support", adjacency, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        graph = torch.einsum("ij,bjte->bite", self.normalized_support, x)
        graph = graph * torch.tanh(self.graph_scale).view(1, 1, 1, -1)
        return x + temporal + graph


class MaskedCrissCrossLayer(nn.Module):
    """Capacity-controlled CBraMod criss-cross comparison layer."""

    def __init__(
        self,
        *,
        dimension: int,
        num_heads: int,
        physical_support: torch.Tensor,
        connectivity_kinds: Sequence[str],
        dropout: float,
        residual_init: float,
        ff_multiplier: int = 5,
    ) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("CBraMod dimension must be even")
        half_dimension = dimension // 2
        half_heads = max(1, num_heads // 2)
        if num_heads % 2:
            raise ValueError("CBraMod num_heads must be even")
        if half_dimension % half_heads:
            raise ValueError("half CBraMod dimension must divide half head count")
        self.dimension = int(dimension)
        self.half_heads = int(half_heads)
        self.connectivity_kinds = tuple(
            str(value).strip().lower() for value in connectivity_kinds
        )
        self.connectivity_scales = nn.Parameter(
            torch.zeros(len(self.connectivity_kinds))
        )
        self.input_norm = nn.LayerNorm(dimension)
        self.spatial_attention = nn.MultiheadAttention(
            half_dimension, half_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            half_dimension, half_heads, dropout=dropout, batch_first=True
        )
        self.branch_projection = nn.Linear(dimension, dimension)
        self.ffn_norm = nn.LayerNorm(dimension)
        hidden = dimension * int(ff_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.register_buffer(
            "spatial_attention_mask", ~physical_support.to(dtype=torch.bool), persistent=True
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None = None,
        residual_base: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, n_time, dimension = x.shape
        base = x if residual_base is None else residual_base
        if tuple(base.shape) != tuple(x.shape):
            raise ValueError("CBraMod residual_base must match its motif input")
        valid_tokens = channel_mask[:, :, None, None] & time_mask[:, None, :, None]
        z = self.input_norm(x) * valid_tokens.to(x.dtype)
        spatial_half, temporal_half = z.chunk(2, dim=-1)

        spatial_sequence = spatial_half.permute(0, 2, 1, 3).reshape(
            batch * n_time, channels, dimension // 2
        )
        spatial_padding = (
            ~channel_mask[:, None, :]
            .expand(batch, n_time, channels)
            .reshape(batch * n_time, channels)
        )
        additive_support = torch.zeros(
            (channels, channels), dtype=x.dtype, device=x.device
        ).masked_fill(self.spatial_attention_mask, -torch.inf)
        edge_bias = torch.zeros(
            (batch, n_time, channels, channels), dtype=x.dtype, device=x.device
        )
        has_connectivity_signal = connectivity is not None and bool(
            (connectivity.mask & connectivity.values.ne(0)).any()
        )
        if has_connectivity_signal:
            assert connectivity is not None
            if connectivity.kinds != self.connectivity_kinds:
                raise ValueError("connectivity kinds do not match CBraMod configuration")
            edge_values = torch.tanh(connectivity.values)
            edge_weights = torch.tanh(self.connectivity_scales).view(
                1, 1, 1, 1, -1
            )
            edge_bias = (
                edge_values
                * connectivity.mask.to(edge_values.dtype)
                * edge_weights
            ).sum(dim=-1)
        spatial_attention_mask = (
            edge_bias.unsqueeze(2)
            .expand(batch, n_time, self.half_heads, channels, channels)
            .reshape(batch * n_time * self.half_heads, channels, channels)
            + additive_support.view(1, channels, channels)
        )
        spatial_key_padding = torch.zeros(
            (batch * n_time, channels), dtype=x.dtype, device=x.device
        ).masked_fill(spatial_padding, -torch.inf)
        spatial = self.spatial_attention(
            spatial_sequence,
            spatial_sequence,
            spatial_sequence,
            attn_mask=spatial_attention_mask,
            key_padding_mask=spatial_key_padding,
            need_weights=False,
        )[0].reshape(batch, n_time, channels, dimension // 2).permute(0, 2, 1, 3)

        temporal_sequence = temporal_half.reshape(
            batch * channels, n_time, dimension // 2
        )
        temporal_padding = (
            ~time_mask[:, None, :]
            .expand(batch, channels, n_time)
            .reshape(batch * channels, n_time)
        )
        temporal = self.temporal_attention(
            temporal_sequence,
            temporal_sequence,
            temporal_sequence,
            key_padding_mask=temporal_padding,
            need_weights=False,
        )[0].reshape(batch, channels, n_time, dimension // 2)

        projected = self.branch_projection(torch.cat((spatial, temporal), dim=-1))
        correction = (x - base) + projected + self.ffn(self.ffn_norm(z + projected))
        correction = correction * valid_tokens.to(correction.dtype)
        gate = torch.tanh(self.residual_scale)
        output = base + gate * self.dropout(correction)
        output = output * valid_tokens.to(output.dtype)
        return output, {
            "residual_scale": gate,
            "connectivity_scales": torch.tanh(self.connectivity_scales),
        }


class CBraModMotifEncoder(nn.Module):
    def __init__(
        self,
        *,
        dimension: int,
        num_heads: int,
        depth: int,
        physical_support: torch.Tensor,
        connectivity_kinds: Sequence[str],
        dropout: float,
        residual_init: float,
    ) -> None:
        super().__init__()
        self.position = PhysicalAdaptivePositionalEncoding(
            dimension, physical_support
        )
        self.layers = nn.ModuleList(
            [
                MaskedCrissCrossLayer(
                    dimension=dimension,
                    num_heads=num_heads,
                    physical_support=physical_support,
                    connectivity_kinds=connectivity_kinds,
                    dropout=dropout,
                    residual_init=residual_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        connectivity: CanonicalConnectivity | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_tokens = channel_mask[:, :, None, None] & time_mask[:, None, :, None]
        positioned = self.position(x) * valid_tokens.to(x.dtype)
        # Position encoding is part of the first gated correction, so setting
        # residual_init=0 remains an exact head-only identity.
        statistics: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            layer_input = positioned if index == 0 else x
            filtered, layer_statistics = layer(
                layer_input,
                channel_mask=channel_mask,
                time_mask=time_mask,
                connectivity=connectivity,
                residual_base=x if index == 0 else None,
            )
            x = filtered
            statistics.update(
                {f"layer{index}.{name}": value for name, value in layer_statistics.items()}
            )
        return x, statistics


@dataclass(frozen=True)
class Standard19MotifOutput:
    """Model output with an explicit internal-C19/deployment-C18 boundary."""

    standard19_logits: torch.Tensor
    c18_logits: torch.Tensor
    candidate_logits: torch.Tensor
    candidate_probabilities: torch.Tensor
    motif_embeddings: torch.Tensor
    encoded_embeddings: torch.Tensor
    signal_channel_mask: torch.Tensor
    c18_mask: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]


class Standard19MotifModel(nn.Module):
    """Shared TFM motif head with interchangeable relation mixers."""

    def __init__(
        self,
        *,
        encoder_type: str,
        code_book_size: int = 512,
        dimension: int = 64,
        motif_input_dimension: int | None = None,
        num_heads: int = 4,
        depth: int = 2,
        dynamic_topk_extra: int = 2,
        connectivity_kinds: Sequence[str] = DEFAULT_CONNECTIVITY_KINDS,
        connectivity_allow_nonphysical: bool = False,
        dropout: float = 0.1,
        residual_init: float = 1e-3,
        prior_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if encoder_type not in VALID_ENCODERS:
            raise ValueError(f"encoder_type must be one of {VALID_ENCODERS}")
        if dimension < 8 or dimension % num_heads or dimension % 2:
            raise ValueError("dimension must be even, >=8, and divisible by num_heads")
        if depth < 1:
            raise ValueError("depth must be positive")
        if code_book_size < 2:
            raise ValueError("code_book_size must be >=2")
        if not math.isfinite(float(residual_init)):
            raise ValueError("residual_init must be finite")
        self.encoder_type = str(encoder_type)
        self.code_book_size = int(code_book_size)
        self.dimension = int(dimension)
        self.connectivity_kinds = tuple(
            str(value).strip().lower() for value in connectivity_kinds
        )
        if len(set(self.connectivity_kinds)) != len(self.connectivity_kinds):
            raise ValueError("connectivity_kinds must be unique")

        self.token_embedding = nn.Embedding(
            self.code_book_size + 1,
            dimension,
            padding_idx=self.code_book_size,
        )
        input_dimension = dimension if motif_input_dimension is None else int(
            motif_input_dimension
        )
        self.continuous_projection: nn.Module
        if input_dimension == dimension:
            self.continuous_projection = nn.Identity()
        else:
            self.continuous_projection = nn.Linear(input_dimension, dimension)
        self.input_norm = nn.LayerNorm(dimension)
        self.channel_embedding = nn.Embedding(N_STANDARD_CHANNELS, dimension)
        self.input_dropout = nn.Dropout(dropout)
        support = standard19_physical_support()
        self.register_buffer("physical_support", support.clone(), persistent=True)
        self.register_buffer("candidate_mask", V11_CANDIDATE_MASK.clone(), persistent=True)
        self.register_buffer(
            "c18_indices", torch.tensor(C18_INDICES, dtype=torch.long), persistent=True
        )
        if prior_logits is None:
            prior = torch.zeros(N_STANDARD_CHANNELS, dtype=torch.float32)
        else:
            if tuple(prior_logits.shape) != (N_STANDARD_CHANNELS,) or (
                not prior_logits.is_floating_point()
            ) or not torch.isfinite(prior_logits).all():
                raise ValueError("prior_logits must be finite floating point [19]")
            prior = prior_logits.detach().float().clone()
        self.register_buffer("prior_logits", prior, persistent=True)

        # Instantiate every shared trainable component before the selectable
        # mixer.  With the same seed, all three arms therefore start from
        # byte-identical TFM embeddings and output heads.
        self.temporal_pool = nn.Linear(dimension, 1, bias=False)
        self.head_norm = nn.LayerNorm(dimension)
        self.channel_head = nn.Linear(dimension, 1)

        if self.encoder_type == "head_only":
            self.encoder: nn.Module = HeadOnlyMotifEncoder()
        elif self.encoder_type == "timefilter":
            self.encoder = TimeFilterMotifEncoder(
                dimension=dimension,
                num_heads=num_heads,
                depth=depth,
                physical_support=support,
                dynamic_topk_extra=dynamic_topk_extra,
                connectivity_kinds=self.connectivity_kinds,
                connectivity_allow_nonphysical=connectivity_allow_nonphysical,
                dropout=dropout,
                residual_init=residual_init,
            )
        else:
            self.encoder = CBraModMotifEncoder(
                dimension=dimension,
                num_heads=num_heads,
                depth=depth,
                physical_support=support,
                connectivity_kinds=self.connectivity_kinds,
                dropout=dropout,
                residual_init=residual_init,
            )

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def n_mixer_parameters(self) -> int:
        return sum(value.numel() for value in self.encoder.parameters())

    def initialize_token_embedding_from_tfm(self, codebook: torch.Tensor) -> None:
        """Copy a frozen TFM codebook into the classifier lookup table."""

        expected = (self.code_book_size, self.dimension)
        if tuple(codebook.shape) != expected or not codebook.is_floating_point() or (
            not torch.isfinite(codebook).all()
        ):
            raise ValueError(f"TFM codebook must be finite floating point {expected}")
        with torch.no_grad():
            self.token_embedding.weight[: self.code_book_size].copy_(
                codebook.to(
                    device=self.token_embedding.weight.device,
                    dtype=self.token_embedding.weight.dtype,
                )
            )
            self.token_embedding.weight[self.code_book_size].zero_()

    def _embed_motifs(
        self,
        token_ids: torch.Tensor | None,
        motif_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if (token_ids is None) == (motif_embeddings is None):
            raise ValueError("provide exactly one of token_ids or motif_embeddings")
        if token_ids is not None:
            if token_ids.ndim != 3 or token_ids.shape[1] != N_STANDARD_CHANNELS or (
                token_ids.dtype != torch.long
            ):
                raise TypeError("token_ids must be long [B,19,T]")
            if bool(((token_ids < 0) | (token_ids > self.code_book_size)).any()):
                raise ValueError("token_ids lie outside the TFM codebook")
            return self.token_embedding(token_ids)
        assert motif_embeddings is not None
        if motif_embeddings.ndim != 4 or motif_embeddings.shape[1] != N_STANDARD_CHANNELS:
            raise ValueError("motif_embeddings must be [B,19,T,D_in]")
        if not motif_embeddings.is_floating_point() or not torch.isfinite(
            motif_embeddings
        ).all():
            raise ValueError("motif_embeddings must be finite floating point")
        return self.continuous_projection(motif_embeddings)

    def forward(
        self,
        token_ids: torch.Tensor | None = None,
        *,
        motif_embeddings: torch.Tensor | None = None,
        signal_channel_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        connectivity: torch.Tensor | None = None,
        connectivity_mask: torch.Tensor | None = None,
    ) -> Standard19MotifOutput:
        motifs = self._embed_motifs(token_ids, motif_embeddings)
        batch, channels, n_time, dimension = motifs.shape
        if channels != N_STANDARD_CHANNELS or dimension != self.dimension:
            raise RuntimeError("embedded motif carrier changed shape")
        valid_channels, valid_time = _validate_signal_masks(
            batch=batch,
            n_time=n_time,
            device=motifs.device,
            channel_mask=signal_channel_mask,
            time_mask=time_mask,
        )
        valid_tokens = valid_channels[:, :, None, None] & valid_time[:, None, :, None]
        motifs = motifs * valid_tokens.to(motifs.dtype)
        channel_ids = torch.arange(channels, device=motifs.device)
        channel_position = self.channel_embedding(channel_ids).view(1, channels, 1, dimension)
        time_position = _sinusoidal_time_encoding(
            n_time,
            dimension,
            device=motifs.device,
            dtype=motifs.dtype,
        )
        encoded_input = self.input_dropout(
            self.input_norm(motifs) + channel_position + time_position
        )
        encoded_input = encoded_input * valid_tokens.to(encoded_input.dtype)

        canonical_connectivity = None
        if connectivity is not None:
            canonical_connectivity = canonicalize_connectivity(
                connectivity,
                n_time=n_time,
                channel_mask=valid_channels,
                time_mask=valid_time,
                kinds=self.connectivity_kinds,
                connectivity_mask=connectivity_mask,
            )
        elif connectivity_mask is not None:
            raise ValueError("connectivity_mask requires connectivity values")
        encoded, diagnostics = self.encoder(
            encoded_input,
            channel_mask=valid_channels,
            time_mask=valid_time,
            connectivity=canonical_connectivity,
        )

        pooling_scores = self.temporal_pool(encoded).squeeze(-1)
        pooling_scores = pooling_scores.masked_fill(
            ~valid_time[:, None, :], -torch.finfo(pooling_scores.dtype).max
        )
        pooling_weights = torch.softmax(pooling_scores, dim=-1)
        pooled = (encoded * pooling_weights.unsqueeze(-1)).sum(dim=2)
        pooled = pooled * valid_channels.unsqueeze(-1).to(pooled.dtype)
        standard19_logits = self.channel_head(self.head_norm(pooled)).squeeze(-1)
        standard19_logits = standard19_logits + self.prior_logits.to(
            dtype=standard19_logits.dtype
        ).view(1, -1)
        if not torch.isfinite(standard19_logits).all():
            raise RuntimeError("standard19 motif head produced non-finite logits")

        ranking_mask = valid_channels & self.candidate_mask.view(1, -1)
        if not bool(ranking_mask.any(dim=1).all()):
            raise ValueError("every example needs at least one available C18 candidate")
        candidate_logits = standard19_logits.masked_fill(~ranking_mask, -torch.inf)
        probabilities = torch.softmax(candidate_logits, dim=-1)
        c18_logits = standard19_logits.index_select(1, self.c18_indices)
        c18_mask = valid_channels.index_select(1, self.c18_indices)
        extended_diagnostics = dict(diagnostics)
        extended_diagnostics["connectivity_used"] = torch.tensor(
            canonical_connectivity is not None and self.encoder_type != "head_only",
            dtype=torch.bool,
            device=motifs.device,
        )
        return Standard19MotifOutput(
            standard19_logits=standard19_logits,
            c18_logits=c18_logits,
            candidate_logits=candidate_logits,
            candidate_probabilities=probabilities,
            motif_embeddings=motifs,
            encoded_embeddings=encoded,
            signal_channel_mask=valid_channels,
            c18_mask=c18_mask,
            diagnostics=extended_diagnostics,
        )


def patient_bag_positive_set_mass_loss(
    event_logits: torch.Tensor,
    event_patient_ids: torch.Tensor,
    patient_targets: torch.Tensor,
    patient_target_mask: torch.Tensor,
    *,
    allow_candidate_subset: bool = False,
) -> tuple[torch.Tensor, PatientAggregation]:
    """Train on complete patient bags with equal weight per patient.

    ``patient_targets`` is indexed by the integer IDs carried in
    ``event_patient_ids``.  A batch may contain a subset of patients, but it
    must contain the complete event bag for every included patient; otherwise
    train/inference aggregation would no longer match.
    """

    aggregation = aggregate_patient_logits(event_logits, event_patient_ids)
    if patient_targets.ndim != 2 or patient_targets.shape[1] != N_STANDARD_CHANNELS:
        raise ValueError("patient_targets must be [P,19]")
    if tuple(patient_target_mask.shape) != tuple(patient_targets.shape):
        raise ValueError("patient_target_mask must match patient_targets")
    if aggregation.patient_ids.min().item() < 0 or aggregation.patient_ids.max().item() >= (
        patient_targets.shape[0]
    ):
        raise ValueError("event_patient_ids lie outside the patient target table")
    selected_targets = patient_targets.index_select(0, aggregation.patient_ids)
    selected_mask = patient_target_mask.index_select(0, aggregation.patient_ids)
    loss = positive_set_mass_loss(
        aggregation.logits,
        selected_targets,
        selected_mask,
        allow_candidate_subset=allow_candidate_subset,
    )
    return loss, aggregation


__all__ = [
    "C18_CHANNELS",
    "C18_INDICES",
    "CBraModMotifEncoder",
    "CanonicalConnectivity",
    "DEFAULT_CONNECTIVITY_KINDS",
    "FrozenTFMMotifExtractor",
    "HeadOnlyMotifEncoder",
    "Standard19MotifModel",
    "Standard19MotifOutput",
    "TFMMotifTokens",
    "TimeFilterMotifEncoder",
    "TimeFilterMotifLayer",
    "VALID_ENCODERS",
    "canonicalize_connectivity",
    "patient_bag_positive_set_mass_loss",
    "standard19_physical_support",
]
