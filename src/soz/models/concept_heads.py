"""Independent lightweight heads for the three evidence families."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..geometry import (
    N_MORPHOLOGY_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    edge_endpoint_indices,
)


def _validate_tokens(tokens: torch.Tensor, token_dim: int) -> None:
    if tokens.ndim != 4:
        raise ValueError(f"tokens must have shape [B,19,T,D], got {tuple(tokens.shape)}")
    if tokens.shape[1] != N_STANDARD_CHANNELS or tokens.shape[-1] != token_dim:
        raise ValueError(
            f"tokens must have channel/dim [{N_STANDARD_CHANNELS},{token_dim}], "
            f"got [{tokens.shape[1]},{tokens.shape[-1]}]"
        )
    if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError("tokens must be finite floating-point values")


class NodeToEdgeTokens(nn.Module):
    """Build ordered edge features for predicting bipolar native labels."""

    def __init__(self, *, token_dim: int = 200) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.register_buffer("endpoints", edge_endpoint_indices(), persistent=True)

    @property
    def output_dim(self) -> int:
        return self.token_dim * 3

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(tokens, self.token_dim)
        left = tokens[:, self.endpoints[:, 0]]
        right = tokens[:, self.endpoints[:, 1]]
        return torch.cat([left, right, left - right], dim=-1)


class MorphologyEvidenceHead(nn.Module):
    """Predict native TUEV CE6 logits on each bipolar edge and second."""

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__()
        self.edge_tokens = NodeToEdgeTokens(token_dim=token_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.edge_tokens.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, 6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(self.edge_tokens(tokens)))

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != N_TCP_EDGES or logits.shape[-1] != 6:
            raise ValueError("Morphology logits must have shape [B,20,T,6]")
        return logits.softmax(dim=-1)


class IctalInvolvementHead(nn.Module):
    """Predict TUSZ bipolar edge-time ictal-involvement logits."""

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__()
        self.edge_tokens = NodeToEdgeTokens(token_dim=token_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.edge_tokens.output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(self.edge_tokens(tokens)))

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != N_TCP_EDGES or logits.shape[-1] != 1:
            raise ValueError("Ictal logits must have shape [B,20,T,1]")
        return logits.sigmoid()


class CapacityMatchedChannelResidualIctalInvolvementHead(IctalInvolvementHead):
    """No-time-mixing control with the same extra capacity as the k31 block.

    The grouped ``1 x 1`` operation acts only on the 128 hidden features of
    one edge-second at a time.  It therefore adds exactly 4,096 weights; the
    following LayerNorm adds 256 affine parameters.  The 4,352-parameter
    residual block matches the k31 temporal block's extra capacity without
    allowing information to cross relative seconds.
    """

    groups = 4

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        if hidden_dim % self.groups:
            raise ValueError("hidden_dim must be divisible by four groups")
        super().__init__(token_dim=token_dim, hidden_dim=hidden_dim)
        self.channel_grouped = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            groups=self.groups,
            bias=False,
        )
        self.channel_activation = nn.GELU()
        self.channel_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.adapter(self.edge_tokens(tokens))
        batch, edges, seconds, dimensions = hidden.shape
        independent = hidden.reshape(batch * edges * seconds, dimensions, 1)
        channel_delta = self.channel_grouped(independent).squeeze(-1).reshape_as(
            hidden
        )
        channel_hidden = self.channel_norm(
            self.channel_activation(hidden + channel_delta)
        )
        return self.classifier(channel_hidden)


class TemporalResidualIctalInvolvementHead(IctalInvolvementHead):
    """V5 ictal head with one local residual temporal operation.

    The source-native target is still bipolar edge-time involvement.  The
    module deliberately keeps the V4 node-to-edge adapter and adds only one
    depthwise five-second convolution.  It receives neither annotation masks
    nor patient/fold identity and therefore cannot use label availability as
    an input shortcut.
    """

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__(token_dim=token_dim, hidden_dim=hidden_dim)
        self.temporal_depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2,
            groups=hidden_dim,
        )
        self.temporal_activation = nn.GELU()
        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.adapter(self.edge_tokens(tokens))
        batch, edges, seconds, dimensions = hidden.shape
        temporal_input = hidden.reshape(
            batch * edges, seconds, dimensions
        ).transpose(1, 2)
        temporal_delta = self.temporal_depthwise(temporal_input).transpose(1, 2)
        temporal_hidden = self.temporal_norm(
            self.temporal_activation(
                hidden.reshape(batch * edges, seconds, dimensions)
                + temporal_delta
            )
        )
        return self.classifier(
            temporal_hidden.reshape(batch, edges, seconds, dimensions)
        )


class LongContextTemporalResidualIctalInvolvementHead(IctalInvolvementHead):
    """Development-only long-context residual head for frozen LaBraM tokens.

    This is the single prespecified LaBraM-only recovery candidate.  It keeps
    the native edge/second target and the V4 node-to-edge adapter unchanged,
    but expands the depthwise temporal receptive field from five to 31
    seconds.  Padding is symmetric, so its output is retrospective
    involvement evidence and must never be described as causal onset evidence.
    """

    context_seconds = 31

    def __init__(self, *, token_dim: int = 200, hidden_dim: int = 128) -> None:
        super().__init__(token_dim=token_dim, hidden_dim=hidden_dim)
        self.temporal_depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=self.context_seconds,
            padding=self.context_seconds // 2,
            groups=hidden_dim,
        )
        self.temporal_activation = nn.GELU()
        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.adapter(self.edge_tokens(tokens))
        batch, edges, seconds, dimensions = hidden.shape
        temporal_input = hidden.reshape(
            batch * edges, seconds, dimensions
        ).transpose(1, 2)
        temporal_delta = self.temporal_depthwise(temporal_input).transpose(1, 2)
        temporal_hidden = self.temporal_norm(
            self.temporal_activation(
                hidden.reshape(batch * edges, seconds, dimensions)
                + temporal_delta
            )
        )
        return self.classifier(
            temporal_hidden.reshape(batch, edges, seconds, dimensions)
        )


@dataclass(frozen=True)
class EvolutionHeadOutput:
    descriptors: torch.Tensor
    future_change: torch.Tensor
    future_source_tiles: torch.Tensor
    future_target_tiles: torch.Tensor


class TemporalEvolutionHead(nn.Module):
    """Predict six four-second descriptors and one cross-call future change."""

    def __init__(
        self,
        *,
        token_dim: int = 200,
        tokens_per_tile: int = 4,
        tokens_per_call: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if tokens_per_call % tokens_per_tile != 0:
            raise ValueError("tokens_per_call must be divisible by tokens_per_tile")
        self.token_dim = int(token_dim)
        self.tokens_per_tile = int(tokens_per_tile)
        self.tokens_per_call = int(tokens_per_call)
        tile_dim = self.token_dim * self.tokens_per_tile
        self.tile_adapter = nn.Sequential(
            nn.Linear(tile_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.descriptor_head = nn.Linear(hidden_dim, N_NODE_FEATURES)
        self.future_head = nn.Linear(hidden_dim, N_NODE_FEATURES)

    def forward(self, tokens: torch.Tensor) -> EvolutionHeadOutput:
        _validate_tokens(tokens, self.token_dim)
        n_tokens = int(tokens.shape[2])
        if n_tokens % self.tokens_per_call != 0 or n_tokens % self.tokens_per_tile != 0:
            raise ValueError(
                "Token count must contain complete independent calls and four-second tiles"
            )
        n_tiles = n_tokens // self.tokens_per_tile
        tile_tokens = tokens.reshape(
            tokens.shape[0],
            N_STANDARD_CHANNELS,
            n_tiles,
            self.tokens_per_tile * self.token_dim,
        )
        hidden = self.tile_adapter(tile_tokens)
        descriptors = self.descriptor_head(hidden)

        tiles_per_call = self.tokens_per_call // self.tokens_per_tile
        source_tiles = torch.arange(
            tiles_per_call - 1,
            n_tiles - 1,
            tiles_per_call,
            device=tokens.device,
        )
        target_tiles = source_tiles + 1
        future_change = self.future_head(hidden.index_select(2, source_tiles))
        return EvolutionHeadOutput(
            descriptors=descriptors,
            future_change=future_change,
            future_source_tiles=source_tiles,
            future_target_tiles=target_tiles,
        )
