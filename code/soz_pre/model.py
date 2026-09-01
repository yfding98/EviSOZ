#!/usr/bin/env python3
"""Multi-task SOZ model for heterogeneous public/private supervision."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOZPreNet(nn.Module):
    """DeepSOZ-like temporal encoder with SOZ/region/hemisphere heads.

    Unlike a plain DeepSOZ reproduction, this model exposes separate heads for
    channel SOZ, region SOZ, hemisphere, temporal seizure detection, and
    propagation. Extra input channels such as SPHL/SPHR can participate in the
    shared encoder while the channel SOZ head is restricted to the canonical
    22 TCP label channels.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_label_channels: int,
        window_samples: int,
        n_regions: int = 5,
        n_hemisphere_classes: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        transformer_layers: int = 2,
        dim_feedforward: int = 128,
        lstm_hidden_dim: int = 64,
        dropout: float = 0.15,
        attention_temperature: float = 1.0,
    ):
        super().__init__()
        self.n_input_channels = int(n_input_channels)
        self.n_label_channels = int(n_label_channels)
        self.window_samples = int(window_samples)
        self.n_regions = int(n_regions)
        self.attention_temperature = float(attention_temperature)
        self.d_model = int(d_model)

        self.window_projection = nn.Sequential(
            nn.Linear(self.window_samples, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.channel_embedding = nn.Embedding(self.n_input_channels, self.d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.transformer_norm = nn.LayerNorm(self.d_model)
        self.temporal_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=int(lstm_hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = int(lstm_hidden_dim) * 2
        self.seizure_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(temporal_dim, 1),
        )
        self.channel_head = nn.Sequential(
            nn.LayerNorm(self.d_model + temporal_dim),
            nn.Linear(self.d_model + temporal_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )
        global_dim = temporal_dim + self.d_model
        self.region_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.n_regions),
        )
        self.propagation_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.n_regions),
        )
        self.hemisphere_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, int(n_hemisphere_classes)),
        )
        self.register_buffer("channel_ids", torch.arange(self.n_input_channels, dtype=torch.long), persistent=False)
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,C,W], got {tuple(x.shape)}")
        bsz, n_windows, n_channels, window_samples = x.shape
        if n_channels != self.n_input_channels or window_samples != self.window_samples:
            raise ValueError(
                f"Expected channels/window [{self.n_input_channels},{self.window_samples}], "
                f"got [{n_channels},{window_samples}]"
            )
        tokens = self.window_projection(x.reshape(bsz * n_windows * n_channels, window_samples))
        tokens = tokens.reshape(bsz * n_windows, n_channels, self.d_model)
        tokens = tokens + self.channel_embedding(self.channel_ids).unsqueeze(0)
        if input_mask is not None:
            mask = input_mask.reshape(bsz, 1, n_channels, 1).expand(-1, n_windows, -1, self.d_model)
            tokens = tokens.reshape(bsz, n_windows, n_channels, self.d_model) * mask
            tokens = tokens.reshape(bsz * n_windows, n_channels, self.d_model)
        global_token = self.global_token.expand(bsz * n_windows, -1, -1)
        encoded = self.transformer_norm(self.transformer(torch.cat([global_token, tokens], dim=1)))
        global_features = encoded[:, 0].reshape(bsz, n_windows, self.d_model)
        channel_features = encoded[:, 1:].reshape(bsz, n_windows, n_channels, self.d_model)
        temporal_features, _ = self.temporal_lstm(global_features)
        seizure_logits = self.seizure_head(temporal_features).squeeze(-1)

        label_channel_features = channel_features[:, :, : self.n_label_channels, :]
        temporal_for_channels = temporal_features.unsqueeze(2).expand(-1, -1, self.n_label_channels, -1)
        channel_context = torch.cat([label_channel_features, temporal_for_channels], dim=-1)
        window_channel_logits = self.channel_head(channel_context).squeeze(-1)

        temperature = max(float(self.attention_temperature), 1e-6)
        attention = F.softmax(seizure_logits / temperature, dim=1)
        channel_logits = (window_channel_logits * attention.unsqueeze(-1)).sum(dim=1)
        pooled_temporal = (temporal_features * attention.unsqueeze(-1)).sum(dim=1)
        pooled_global = (global_features * attention.unsqueeze(-1)).sum(dim=1)
        global_context = torch.cat([pooled_temporal, pooled_global], dim=-1)
        return {
            "channel_logits": channel_logits,
            "window_channel_logits": window_channel_logits,
            "seizure_logits": seizure_logits,
            "attention": attention,
            "region_logits": self.region_head(global_context),
            "propagation_logits": self.propagation_head(global_context),
            "hemisphere_logits": self.hemisphere_head(global_context),
            "global_context": global_context,
        }


class EEGNetSOZNet(nn.Module):
    """EEGNet-style baseline with SOZ-compatible multi-task heads.

    The encoder follows the EEGNet pattern of temporal convolution, depthwise
    spatial convolution, and separable temporal convolution. Channel SOZ logits
    are produced before the spatial channel collapse so that localization still
    has a per-bipolar-channel output.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_label_channels: int,
        window_samples: int,
        n_windows: int,
        n_regions: int = 5,
        n_hemisphere_classes: int = 4,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        pointwise_filters: int = 32,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
        attention_temperature: float = 1.0,
    ):
        super().__init__()
        self.n_input_channels = int(n_input_channels)
        self.n_label_channels = int(n_label_channels)
        self.window_samples = int(window_samples)
        self.n_windows = int(n_windows)
        self.n_regions = int(n_regions)
        self.temporal_filters = int(temporal_filters)
        self.attention_temperature = float(attention_temperature)

        f1 = max(1, int(temporal_filters))
        depth = max(1, int(depth_multiplier))
        f2 = max(1, int(pointwise_filters))
        k1 = max(1, int(kernel_length))
        k2 = max(1, int(separable_kernel_length))
        if k1 % 2 == 0:
            k1 += 1
        if k2 % 2 == 0:
            k2 += 1
        pool1 = max(1, int(pool1))
        pool2 = max(1, int(pool2))

        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f1 * depth, kernel_size=(self.n_input_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1), stride=(1, pool1)),
            nn.Dropout(float(dropout)),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth,
                f1 * depth,
                kernel_size=(1, k2),
                padding=(0, k2 // 2),
                groups=f1 * depth,
                bias=False,
            ),
            nn.Conv2d(f1 * depth, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2), stride=(1, pool2)),
            nn.Dropout(float(dropout)),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_head = nn.Sequential(
            nn.LayerNorm(f1),
            nn.Linear(f1, 1),
        )
        self.seizure_head = nn.Sequential(
            nn.LayerNorm(f1),
            nn.Dropout(float(dropout)),
            nn.Linear(f1, 1),
        )
        self.region_head = nn.Sequential(
            nn.LayerNorm(f2),
            nn.Dropout(float(dropout)),
            nn.Linear(f2, self.n_regions),
        )
        self.propagation_head = nn.Sequential(
            nn.LayerNorm(f2),
            nn.Dropout(float(dropout)),
            nn.Linear(f2, self.n_regions),
        )
        self.hemisphere_head = nn.Sequential(
            nn.LayerNorm(f2),
            nn.Dropout(float(dropout)),
            nn.Linear(f2, int(n_hemisphere_classes)),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,T,C,W], got {tuple(x.shape)}")
        bsz, n_windows, n_channels, window_samples = x.shape
        if n_channels != self.n_input_channels or window_samples != self.window_samples:
            raise ValueError(
                f"Expected channels/window [{self.n_input_channels},{self.window_samples}], "
                f"got [{n_channels},{window_samples}]"
            )
        if n_windows != self.n_windows:
            raise ValueError(f"Expected {self.n_windows} windows, got {n_windows}")

        if input_mask is not None:
            x = x * input_mask.to(x.device).float().view(bsz, 1, n_channels, 1)
        signal = x.permute(0, 2, 1, 3).reshape(bsz, n_channels, n_windows * window_samples).unsqueeze(1)
        temporal = self.temporal(signal)

        temporal_by_window = temporal.reshape(
            bsz,
            self.temporal_filters,
            n_channels,
            n_windows,
            window_samples,
        )
        label_features = temporal_by_window[:, :, : self.n_label_channels].mean(dim=-1)
        label_features = label_features.permute(0, 3, 2, 1)
        window_channel_logits = self.channel_head(label_features).squeeze(-1)
        seizure_features = temporal_by_window.mean(dim=(2, 4)).permute(0, 2, 1)
        seizure_logits = self.seizure_head(seizure_features).squeeze(-1)

        spatial = self.spatial(temporal)
        encoded = self.separable(spatial)
        global_context = self.global_pool(encoded).flatten(1)

        temperature = max(float(self.attention_temperature), 1e-6)
        attention = F.softmax(seizure_logits / temperature, dim=1)
        channel_logits = (window_channel_logits * attention.unsqueeze(-1)).sum(dim=1)

        return {
            "channel_logits": channel_logits,
            "window_channel_logits": window_channel_logits,
            "seizure_logits": seizure_logits,
            "attention": attention,
            "region_logits": self.region_head(global_context),
            "propagation_logits": self.propagation_head(global_context),
            "hemisphere_logits": self.hemisphere_head(global_context),
            "global_context": global_context,
        }
