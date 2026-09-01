"""Strict common-17 input path for the audited official LaBraM-Base encoder.

Unlike slicing a standard-19 contextual representation, this adapter receives
only the 17 retained physical electrodes.  The official patch embedding and
transformer therefore see 68 patch tokens plus CLS (69 tokens total); FZ and
PZ waveforms, positions, and tokens are absent from every attention block.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Sequence

import torch
import torch.nn as nn

from ..geometry import STANDARD_19, normalize_electrode_name
from .labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_LEGACY_POSITION_NAMES,
    LABRAM_POSITION_ID_BY_NAME,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    LaBraMFeatureReceipt,
    OfficialLaBraMEncoder,
    _raw_electrode_position_name,
)
from .labram_peft import _validate_official_suffix_backbone


COMMON17_CHANNELS: Final[tuple[str, ...]] = tuple(
    channel for channel in STANDARD_19 if channel not in {"FZ", "PZ"}
)
COMMON17_LABRAM_LEGACY_POSITION_NAMES: Final[tuple[str, ...]] = tuple(
    position
    for semantic, position in zip(STANDARD_19, LABRAM_LEGACY_POSITION_NAMES)
    if semantic in COMMON17_CHANNELS
)
COMMON17_COUNT: Final[int] = 17
COMMON17_SECONDS_PER_CALL: Final[int] = 4
COMMON17_SAMPLES_PER_TOKEN: Final[int] = 200
COMMON17_TOKEN_DIM: Final[int] = 200
COMMON17_PATCH_TOKENS: Final[int] = COMMON17_COUNT * COMMON17_SECONDS_PER_CALL
COMMON17_PREFIX_TOKENS: Final[int] = 1 + COMMON17_PATCH_TOKENS
COMMON17_EVENT_SECONDS: Final[int] = 60
COMMON17_EVENT_SAMPLES: Final[int] = 12_000
COMMON17_EVENT_CALLS: Final[int] = 15
COMMON17_PHASES: Final[int] = 5
COMMON17_PHASE_NAMES: Final[tuple[str, ...]] = (
    "pre_-12_0_mean",
    "early_0_12_mean",
    "late_12_48_mean",
    "early_minus_pre",
    "late_minus_early",
)


@dataclass(frozen=True)
class Common17LaBraMPositionBinding:
    """Record-specific raw-header position IDs for the retained 17 electrodes."""

    policy: str
    semantic_channels: tuple[str, ...]
    raw_channel_names: tuple[str, ...]
    position_names: tuple[str, ...]
    position_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.policy != LABRAM_RAW_HEADER_POSITION_BINDING_POLICY:
            raise ValueError("unsupported LaBraM position-binding policy")
        if self.semantic_channels != COMMON17_CHANNELS:
            raise ValueError("common17 binding semantic order changed")
        if any(
            len(values) != COMMON17_COUNT
            for values in (self.raw_channel_names, self.position_names, self.position_ids)
        ):
            raise ValueError("common17 position binding must contain 17 values")
        expected_names = tuple(
            _raw_electrode_position_name(value) for value in self.raw_channel_names
        )
        if self.position_names != expected_names:
            raise ValueError("position names drifted from raw EEG headers")
        if any(
            normalize_electrode_name(position) != semantic
            for semantic, position in zip(self.semantic_channels, self.position_names)
        ):
            raise ValueError("raw electrode aliases do not align with common17 semantics")
        expected_ids = tuple(LABRAM_POSITION_ID_BY_NAME[name] for name in self.position_names)
        if self.position_ids != expected_ids or len(set(self.position_ids)) != COMMON17_COUNT:
            raise ValueError("common17 official LaBraM position IDs are invalid")
        if any(channel in {"FZ", "PZ"} for channel in self.semantic_channels):
            raise RuntimeError("FZ/PZ leaked into the common17 binding")


def bind_common17_labram_record_positions(
    raw_channel_names: Sequence[object],
    *,
    semantic_channels: Sequence[str] = COMMON17_CHANNELS,
) -> Common17LaBraMPositionBinding:
    raw = tuple(str(value).strip() for value in raw_channel_names)
    semantic = tuple(str(value).strip().upper() for value in semantic_channels)
    if len(raw) != COMMON17_COUNT or semantic != COMMON17_CHANNELS:
        raise ValueError("raw headers must align with the frozen common17 order")
    names = tuple(_raw_electrode_position_name(value) for value in raw)
    return Common17LaBraMPositionBinding(
        policy=LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
        semantic_channels=semantic,
        raw_channel_names=raw,
        position_names=names,
        position_ids=tuple(LABRAM_POSITION_ID_BY_NAME[name] for name in names),
    )


def _validate_position_names(position_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names = tuple(str(value).strip().upper() for value in position_names)
    if len(names) != COMMON17_COUNT:
        raise ValueError("common17 LaBraM position_names must contain 17 values")
    if any(name not in LABRAM_POSITION_ID_BY_NAME for name in names):
        raise ValueError("common17 LaBraM position_names contain an unknown electrode")
    if any(
        normalize_electrode_name(position) != semantic
        for semantic, position in zip(COMMON17_CHANNELS, names)
    ):
        raise ValueError("common17 LaBraM positions do not match physical semantics")
    ids = tuple(LABRAM_POSITION_ID_BY_NAME[name] for name in names)
    if len(set(ids)) != COMMON17_COUNT:
        raise ValueError("common17 LaBraM positions contain duplicate IDs")
    return names, ids


class OfficialLaBraMCommon17FrozenPrefixEncoder(nn.Module):
    """Official frozen input path through block 9 on 17 electrodes/69 tokens."""

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
        position_names: Sequence[str] = COMMON17_LABRAM_LEGACY_POSITION_NAMES,
    ) -> None:
        super().__init__()
        names, position_ids = _validate_position_names(position_names)
        official = OfficialLaBraMEncoder(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            expected_modeling_sha256=expected_modeling_sha256,
            tile_seconds=COMMON17_SECONDS_PER_CALL,
        )
        self.backbone = official.backbone
        _validate_official_suffix_backbone(self.backbone)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer(
            "input_chans",
            torch.tensor((0, *position_ids), dtype=torch.long),
            persistent=True,
        )
        self.receipt: LaBraMFeatureReceipt = replace(
            official.receipt,
            semantic_channels=COMMON17_CHANNELS,
            position_names=names,
            position_ids=position_ids,
            tile_seconds=COMMON17_SECONDS_PER_CALL,
        )

    def train(self, mode: bool = True) -> "OfficialLaBraMCommon17FrozenPrefixEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def _forward_with_input_chans(
        self,
        patches: torch.Tensor,
        input_chans: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            COMMON17_COUNT,
            COMMON17_SECONDS_PER_CALL,
            COMMON17_SAMPLES_PER_TOKEN,
        )
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                "common17 LaBraM input must have shape [B,17,4,200], got "
                f"{tuple(patches.shape)}"
            )
        if patches.shape[0] < 1 or not patches.is_floating_point() or not torch.isfinite(patches).all():
            raise ValueError("common17 LaBraM input must be non-empty finite floating point")
        if tuple(input_chans.shape) != (COMMON17_COUNT + 1,) or input_chans.dtype != torch.long:
            raise TypeError("common17 LaBraM input_chans must be long [18] including CLS")
        reference = next(self.backbone.parameters())
        if patches.device != reference.device or patches.dtype != reference.dtype:
            raise ValueError("common17 input must share official backbone dtype/device")

        input_chans = input_chans.to(device=patches.device)
        self.backbone.eval()
        with torch.no_grad():
            tokens = self.backbone.patch_embed(patches * self.receipt.input_scale_from_volts)
            expected_patch = (patches.shape[0], COMMON17_PATCH_TOKENS, COMMON17_TOKEN_DIM)
            if tuple(tokens.shape) != expected_patch:
                raise RuntimeError("official patch embedding did not return 68 common17 tokens")
            cls = self.backbone.cls_token.expand(patches.shape[0], -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)

            position_used = self.backbone.pos_embed[:, input_chans]
            position = (
                position_used[:, 1:, :]
                .unsqueeze(2)
                .expand(patches.shape[0], -1, COMMON17_SECONDS_PER_CALL, -1)
                .flatten(1, 2)
            )
            position = torch.cat(
                (position_used[:, :1].expand(patches.shape[0], -1, -1), position),
                dim=1,
            )
            tokens = tokens + position
            time = (
                self.backbone.time_embed[:, :COMMON17_SECONDS_PER_CALL]
                .unsqueeze(1)
                .expand(patches.shape[0], COMMON17_COUNT, -1, -1)
                .flatten(1, 2)
            )
            tokens[:, 1:] += time
            tokens = self.backbone.pos_drop(tokens)
            for block in self.backbone.blocks[:10]:
                tokens = block(tokens, rel_pos_bias=None)

        expected_prefix = (patches.shape[0], COMMON17_PREFIX_TOKENS, COMMON17_TOKEN_DIM)
        if tuple(tokens.shape) != expected_prefix or not torch.isfinite(tokens).all():
            raise RuntimeError("common17 block-9 prefix is invalid")
        return tokens.detach()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self._forward_with_input_chans(patches, self.input_chans)

    def forward_with_record_binding(
        self,
        patches: torch.Tensor,
        binding: Common17LaBraMPositionBinding,
    ) -> torch.Tensor:
        if not isinstance(binding, Common17LaBraMPositionBinding):
            raise TypeError("binding must be Common17LaBraMPositionBinding")
        input_chans = torch.tensor(
            (0, *binding.position_ids), dtype=torch.long, device=patches.device
        )
        return self._forward_with_input_chans(patches, input_chans)


def common17_event_calls(waveform: torch.Tensor) -> torch.Tensor:
    """Convert one [-12,+48) common17 event into fifteen four-second calls."""

    if tuple(waveform.shape) != (COMMON17_COUNT, COMMON17_EVENT_SAMPLES):
        raise ValueError("common17 event waveform must have shape [17,12000]")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError("common17 event waveform must be finite floating point")
    return (
        waveform.reshape(
            COMMON17_COUNT,
            COMMON17_EVENT_CALLS,
            COMMON17_SECONDS_PER_CALL,
            COMMON17_SAMPLES_PER_TOKEN,
        )
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def extract_common17_phase_features(prefix: torch.Tensor) -> torch.Tensor:
    """Map block-9 prefixes to pre/early/late and two change tensors.

    Accepts either one event ``[15,69,200]`` or a batch
    ``[E,15,69,200]``.  The result is respectively ``[17,5,200]`` or
    ``[E,17,5,200]``.
    """

    single = prefix.ndim == 3
    value = prefix.unsqueeze(0) if single else prefix
    if value.ndim != 4 or tuple(value.shape[1:]) != (
        COMMON17_EVENT_CALLS,
        COMMON17_PREFIX_TOKENS,
        COMMON17_TOKEN_DIM,
    ):
        raise ValueError("common17 prefix must end in [15,69,200]")
    if value.requires_grad or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("common17 prefix must be detached finite floating point")
    events = len(value)
    tiles = (
        value[:, :, 1:, :]
        .reshape(
            events,
            COMMON17_EVENT_CALLS,
            COMMON17_COUNT,
            COMMON17_SECONDS_PER_CALL,
            COMMON17_TOKEN_DIM,
        )
        .mean(dim=3)
    )
    pre = tiles[:, 0:3].mean(dim=1)
    early = tiles[:, 3:6].mean(dim=1)
    late = tiles[:, 6:15].mean(dim=1)
    result = torch.stack((pre, early, late, early - pre, late - early), dim=2).contiguous()
    expected = (events, COMMON17_COUNT, COMMON17_PHASES, COMMON17_TOKEN_DIM)
    if tuple(result.shape) != expected or not torch.isfinite(result).all():
        raise RuntimeError("common17 phase feature extraction failed")
    return result[0] if single else result


__all__ = [
    "COMMON17_CHANNELS",
    "COMMON17_EVENT_CALLS",
    "COMMON17_LABRAM_LEGACY_POSITION_NAMES",
    "COMMON17_PHASE_NAMES",
    "COMMON17_PREFIX_TOKENS",
    "Common17LaBraMPositionBinding",
    "OfficialLaBraMCommon17FrozenPrefixEncoder",
    "bind_common17_labram_record_positions",
    "common17_event_calls",
    "extract_common17_phase_features",
]

