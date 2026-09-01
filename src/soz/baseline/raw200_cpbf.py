"""Controlled CPBF graph refinement for the unified C18 Raw200 benchmark."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the historical implementation through ``tfm_soz`` rather than the
# repository's top-level ``code`` package name, which can collide with
# Python's standard-library ``code`` module after Torch initializes.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_CODE = _REPO_ROOT / "code"
if str(_LEGACY_CODE) not in sys.path:
    sys.path.insert(0, str(_LEGACY_CODE))

from tfm_soz.model import CPBFSparseGraphBlock  # noqa: E402
from src.soz.baseline.raw200_shallow import Raw200ChannelShallowNet


# This is the same standard-19 one-hop graph used by the frozen DeepSOZ
# endpoint implementation.  It is used here only as label-independent physical
# support for graph message passing; neighborhood scoring remains a separate
# evaluation operation.
STANDARD19_NEIGHBORS: Final[dict[int, tuple[int, ...]]] = {
    0: (1, 2, 3, 4),
    1: (0, 4, 5, 6),
    2: (0, 3, 4, 7, 8),
    3: (0, 2, 4, 8, 9),
    4: (0, 1, 3, 5, 9),
    5: (1, 4, 6, 9, 10),
    6: (1, 4, 5, 10, 11),
    7: (2, 8, 12, 13, 17),
    8: (2, 3, 4, 7, 9, 12, 13, 14),
    9: (3, 4, 5, 8, 10, 13, 14, 15),
    10: (4, 5, 6, 9, 11, 14, 15, 16),
    11: (6, 10, 15, 16, 18),
    12: (7, 8, 13, 17),
    13: (7, 8, 9, 12, 14, 17),
    14: (8, 9, 10, 13, 15, 17, 18),
    15: (9, 10, 11, 14, 16, 18),
    16: (10, 11, 15, 18),
    17: (7, 12, 13, 14, 18),
    18: (11, 14, 15, 16, 17),
}


def standard19_candidate_mask() -> torch.Tensor:
    """Return the symmetric one-hop physical support including self edges."""

    mask = torch.eye(19, dtype=torch.bool)
    for destination, sources in STANDARD19_NEIGHBORS.items():
        for source in sources:
            mask[destination, source] = True
            mask[source, destination] = True
    return mask


class Raw200CPBFRefinement(nn.Module):
    """Refine frozen Raw200 phase tokens with the historical CPBF graph block.

    ``head_only`` performs the same second-stage channel-head optimization but
    bypasses CPBF.  ``cpbf`` inserts the original context-conditioned sparse
    graph block over six phase/statistic tokens per physical electrode.  The
    frozen temporal convolution and fold-local prior are inherited from an
    already fitted Raw200-Shallow fold.
    """

    VALID_VARIANTS = ("head_only", "cpbf")
    N_PHASE_TOKENS = 6
    EMBEDDING_DIM = Raw200ChannelShallowNet.TEMPORAL_FILTERS
    SEGMENT_IDS = (0, 0, 1, 1, 2, 2)

    def __init__(
        self,
        base: Raw200ChannelShallowNet,
        *,
        variant: str,
        cpbf_topk: int = 6,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown Raw200 refinement variant {variant!r}")
        self.variant = str(variant)
        self.temporal = base.temporal
        self.channel_scorer = base.channel_scorer
        self.register_buffer(
            "prior_logits", base.prior_logits.detach().float().contiguous()
        )
        self.register_buffer(
            "segment_ids",
            torch.tensor(self.SEGMENT_IDS, dtype=torch.long),
            persistent=True,
        )
        if self.variant == "cpbf":
            self.cpbf_graph = CPBFSparseGraphBlock(
                emb_size=self.EMBEDDING_DIM,
                n_channels=Raw200ChannelShallowNet.N_CHANNELS,
                topk=int(cpbf_topk),
                mode="context_graph",
                residual_init=0.0,
                candidate_policy="compact",
                use_region_bias=False,
                residual_policy="signed_scalar",
                use_temporal_adapter=False,
                dropout=0.1,
            )
            # CPBFSparseGraphBlock defaults to identity-only static support for
            # non-TCP channel counts.  Replace that fallback with the frozen
            # standard-19 physical graph required by this benchmark.
            self.cpbf_graph.static_candidate_mask.copy_(standard19_candidate_mask())
        for parameter in self.temporal.parameters():
            parameter.requires_grad_(False)
        for parameter in self.channel_scorer.parameters():
            parameter.requires_grad_(True)

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            value.numel() for value in self.parameters() if value.requires_grad
        )

    def extract_phase_tokens(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert ``[B,19,12000]`` waveforms to ``[B,19,6,32]`` tokens."""

        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (
            Raw200ChannelShallowNet.N_CHANNELS,
            Raw200ChannelShallowNet.N_SAMPLES,
        ):
            raise ValueError("Raw200 CPBF input must be [B,19,12000]")
        if not torch.isfinite(waveform).all():
            raise ValueError("Raw200 CPBF input must be finite")
        batch = len(waveform)
        filtered = self.temporal(
            waveform.reshape(
                batch * Raw200ChannelShallowNet.N_CHANNELS,
                1,
                Raw200ChannelShallowNet.N_SAMPLES,
            )
        )
        power = F.avg_pool1d(filtered.square(), kernel_size=50, stride=12)
        if tuple(power.shape[1:]) != (
            self.EMBEDDING_DIM,
            Raw200ChannelShallowNet.TEMPORAL_GRID,
        ):
            raise RuntimeError("Raw200 CPBF temporal grid changed")
        log_power = power.clamp_min(1e-8).log()
        tokens = []
        for phase in Raw200ChannelShallowNet.PHASE_SLICES:
            value = log_power[:, :, phase]
            tokens.extend((value.mean(dim=2), value.std(dim=2, unbiased=False)))
        stacked = torch.stack(tokens, dim=1)
        return stacked.reshape(
            batch,
            Raw200ChannelShallowNet.N_CHANNELS,
            self.N_PHASE_TOKENS,
            self.EMBEDDING_DIM,
        ).contiguous()

    def forward_tokens(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (
            Raw200ChannelShallowNet.N_CHANNELS,
            self.N_PHASE_TOKENS,
            self.EMBEDDING_DIM,
        )
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"Raw200 phase tokens must be [B,{expected}]")
        if not torch.isfinite(tokens).all():
            raise ValueError("Raw200 phase tokens must be finite")
        stats: dict[str, torch.Tensor] = {}
        refined = tokens
        if self.variant == "cpbf":
            channel_mask = torch.ones(
                len(tokens),
                Raw200ChannelShallowNet.N_CHANNELS,
                dtype=torch.bool,
                device=tokens.device,
            )
            refined, stats = self.cpbf_graph(
                tokens,
                self.segment_ids,
                channel_mask=channel_mask,
            )
        pooled = refined.reshape(
            len(tokens) * Raw200ChannelShallowNet.N_CHANNELS,
            self.N_PHASE_TOKENS * self.EMBEDDING_DIM,
        )
        logits = self.channel_scorer(pooled).reshape(
            len(tokens), Raw200ChannelShallowNet.N_CHANNELS
        )
        return logits + self.prior_logits.view(1, -1), stats

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward_tokens(self.extract_phase_tokens(waveform))


__all__ = [
    "Raw200CPBFRefinement",
    "STANDARD19_NEIGHBORS",
    "standard19_candidate_mask",
]
