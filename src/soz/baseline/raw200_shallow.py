"""Full-bandwidth channel-local raw-EEG comparator for the v60 audit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Raw200ChannelShallowNet(nn.Module):
    """Shared ShallowConvNet-style temporal-power scorer over 19 electrodes.

    The network deliberately preserves the physical-electrode axis and emits
    one logit per electrode.  It is a task-adapted comparator, not an exact
    reproduction of canonical EEGNet or the original ShallowConvNet.
    """

    N_CHANNELS = 19
    N_SAMPLES = 12_000
    TEMPORAL_FILTERS = 32
    TEMPORAL_GRID = 246
    PHASE_SLICES = (slice(0, 49), slice(49, 98), slice(98, 246))

    def __init__(self, prior_logits: torch.Tensor) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (self.N_CHANNELS,) or not torch.isfinite(
            prior_logits
        ).all():
            raise ValueError("raw200 prior_logits must be finite [19]")
        self.temporal = nn.Conv1d(
            1,
            self.TEMPORAL_FILTERS,
            kernel_size=101,
            stride=4,
            padding=50,
            bias=False,
        )
        self.channel_scorer = nn.Linear(self.TEMPORAL_FILTERS * 6, 1)
        self.register_buffer(
            "prior_logits", prior_logits.detach().float().contiguous()
        )

    @property
    def n_trainable_parameters(self) -> int:
        return sum(
            value.numel() for value in self.parameters() if value.requires_grad
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (
            self.N_CHANNELS,
            self.N_SAMPLES,
        ):
            raise ValueError("raw200 input must be [B,19,12000]")
        if not torch.isfinite(waveform).all():
            raise ValueError("raw200 input must be finite")
        batch = len(waveform)
        filtered = self.temporal(
            waveform.reshape(batch * self.N_CHANNELS, 1, self.N_SAMPLES)
        )
        power = F.avg_pool1d(filtered.square(), kernel_size=50, stride=12)
        if tuple(power.shape[1:]) != (
            self.TEMPORAL_FILTERS,
            self.TEMPORAL_GRID,
        ):
            raise RuntimeError("raw200 temporal grid changed")
        log_power = power.clamp_min(1e-8).log()
        features = []
        for phase in self.PHASE_SLICES:
            value = log_power[:, :, phase]
            features.extend((value.mean(dim=2), value.std(dim=2, unbiased=False)))
        pooled = torch.cat(features, dim=1)
        if tuple(pooled.shape) != (
            batch * self.N_CHANNELS,
            self.TEMPORAL_FILTERS * 6,
        ):
            raise RuntimeError("raw200 phase feature shape changed")
        score = self.channel_scorer(pooled).reshape(batch, self.N_CHANNELS)
        return score + self.prior_logits.view(1, self.N_CHANNELS)


__all__ = ["Raw200ChannelShallowNet"]
