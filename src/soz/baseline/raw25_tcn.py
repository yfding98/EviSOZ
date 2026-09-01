"""Low-capacity channel-local raw-waveform TCN for the v54 baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Raw25ChannelTCN(nn.Module):
    """Shared channel-local TCN with three fixed event-relative phase pools."""

    def __init__(self, prior_logits: torch.Tensor) -> None:
        super().__init__()
        if tuple(prior_logits.shape) != (19,) or not torch.isfinite(prior_logits).all():
            raise ValueError("raw25 prior_logits must be finite [19]")
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=25, stride=5, padding=12),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=5, padding=4),
            nn.GELU(),
        )
        self.channel_scorer = nn.Linear(48, 1)
        self.register_buffer("prior_logits", prior_logits.detach().float().contiguous())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (19, 1_500):
            raise ValueError("raw25 TCN input must be [B,19,1500]")
        if not torch.isfinite(waveform).all():
            raise ValueError("raw25 TCN input must be finite")
        batch = len(waveform)
        feature = self.temporal(waveform.reshape(batch * 19, 1, 1_500))
        if tuple(feature.shape[1:]) != (16, 60):
            raise RuntimeError("raw25 TCN temporal grid changed")
        pooled = torch.cat(
            (
                feature[:, :, 0:12].mean(dim=2),
                feature[:, :, 12:24].mean(dim=2),
                feature[:, :, 24:60].mean(dim=2),
            ),
            dim=1,
        )
        score = self.channel_scorer(pooled).reshape(batch, 19)
        return score + self.prior_logits.view(1, 19)


__all__ = ["Raw25ChannelTCN"]
