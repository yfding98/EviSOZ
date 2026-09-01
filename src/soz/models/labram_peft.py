"""Minimal differentiable qkv-LoRA suffix for the audited LaBraM-Base.

This module is intentionally separate from :class:`OfficialLaBraMEncoder`.
The latter remains the fail-closed, ``no_grad``/detached feature extractor.
Here the input is instead a detached activation cached immediately before
official transformer block 10, including the CLS token.  Only rank-four
weight parametrizations on the qkv weights of blocks 10 and 11 are trainable.

The official attention implementation calls ``F.linear(x, self.qkv.weight)``
rather than ``self.qkv(x)``.  A PyTorch weight parametrization is therefore
used deliberately: reading ``self.qkv.weight`` materializes the effective
``W + (alpha / rank) B A`` weight even along that functional call path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from ..geometry import N_STANDARD_CHANNELS
from .labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LaBraMRecordPositionBinding,
    OfficialLaBraMEncoder,
)


LABRAM_PEFT_BLOCKS: tuple[int, int] = (10, 11)
LABRAM_PEFT_RANK = 4
LABRAM_PEFT_ALPHA = 8.0
LABRAM_PEFT_DROPOUT = 0.0
LABRAM_PEFT_TOKEN_DIM = 200
LABRAM_PEFT_SECONDS_PER_CALL = 4
LABRAM_PEFT_PATCH_TOKENS = N_STANDARD_CHANNELS * LABRAM_PEFT_SECONDS_PER_CALL
LABRAM_PEFT_PREFIX_TOKENS = 1 + LABRAM_PEFT_PATCH_TOKENS
LABRAM_PEFT_QKV_SHAPE = (3 * LABRAM_PEFT_TOKEN_DIM, LABRAM_PEFT_TOKEN_DIM)
LABRAM_PEFT_TRAINABLE_PARAMETERS = 6_400


@dataclass(frozen=True)
class LaBraMMinimalPEFTConfig:
    """Closed configuration for the single preregistered PEFT candidate."""

    block_indices: tuple[int, int] = LABRAM_PEFT_BLOCKS
    rank: int = LABRAM_PEFT_RANK
    alpha: float = LABRAM_PEFT_ALPHA
    dropout: float = LABRAM_PEFT_DROPOUT

    def __post_init__(self) -> None:
        if self.block_indices != LABRAM_PEFT_BLOCKS:
            raise ValueError("LaBraM PEFT is locked to transformer blocks 10 and 11")
        if type(self.rank) is not int or self.rank != LABRAM_PEFT_RANK:
            raise ValueError("LaBraM PEFT rank is locked to 4")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) != 8.0:
            raise ValueError("LaBraM PEFT alpha is locked to 8")
        if not math.isfinite(float(self.dropout)) or float(self.dropout) != 0.0:
            raise ValueError("LaBraM PEFT dropout is locked to zero")


class LaBraMQKVWeightLoRA(nn.Module):
    """Parametrize one frozen ``[600,200]`` qkv weight with rank-four LoRA."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(
            torch.empty(LABRAM_PEFT_RANK, LABRAM_PEFT_TOKEN_DIM)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(3 * LABRAM_PEFT_TOKEN_DIM, LABRAM_PEFT_RANK)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def scale(self) -> float:
        return LABRAM_PEFT_ALPHA / LABRAM_PEFT_RANK

    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        if tuple(original_weight.shape) != LABRAM_PEFT_QKV_SHAPE:
            raise ValueError(
                "LaBraM qkv original weight must have shape [600,200]"
            )
        if not original_weight.is_floating_point():
            raise TypeError("LaBraM qkv original weight must be floating point")
        if (
            self.lora_A.device != original_weight.device
            or self.lora_B.device != original_weight.device
            or self.lora_A.dtype != original_weight.dtype
            or self.lora_B.dtype != original_weight.dtype
        ):
            raise ValueError("LoRA factors and original qkv weight must share dtype/device")
        delta = self.lora_B @ self.lora_A
        return original_weight + self.scale * delta


def _validate_official_suffix_backbone(backbone: nn.Module) -> None:
    """Fail closed if the loaded official architecture is not LaBraM-Base."""

    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 12:
        raise ValueError("LaBraM PEFT requires the official 12-block Base encoder")
    if getattr(backbone, "embed_dim", None) != LABRAM_PEFT_TOKEN_DIM:
        raise ValueError("LaBraM PEFT requires embed_dim=200")
    if getattr(backbone, "patch_size", None) != 200:
        raise ValueError("LaBraM PEFT requires one-second 200-sample patches")
    if getattr(backbone, "fc_norm", object()) is not None:
        raise ValueError("LaBraM PEFT requires the official token-output norm path")
    if not isinstance(getattr(backbone, "norm", None), nn.LayerNorm):
        raise ValueError("LaBraM PEFT requires the official final LayerNorm")
    if not isinstance(getattr(backbone, "head", None), nn.Identity):
        raise ValueError("LaBraM PEFT requires the checkpoint encoder without a head")

    for block_index in LABRAM_PEFT_BLOCKS:
        block = blocks[block_index]
        attention = getattr(block, "attn", None)
        qkv = getattr(attention, "qkv", None)
        if not isinstance(qkv, nn.Linear):
            raise ValueError(f"LaBraM block {block_index} qkv must be nn.Linear")
        if tuple(qkv.weight.shape) != LABRAM_PEFT_QKV_SHAPE:
            raise ValueError(f"LaBraM block {block_index} qkv shape changed")
        if qkv.bias is not None:
            raise ValueError(f"LaBraM block {block_index} qkv bias must be absent")
        if getattr(attention, "q_bias", object()) is not None or getattr(
            attention, "v_bias", object()
        ) is not None:
            raise ValueError(
                f"LaBraM block {block_index} functional qkv bias contract changed"
            )
        if getattr(attention, "num_heads", None) != 10:
            raise ValueError(f"LaBraM block {block_index} must use 10 heads")
        if parametrize.is_parametrized(qkv, "weight"):
            raise ValueError(f"LaBraM block {block_index} qkv is already parametrized")


class OfficialLaBraMFrozenPrefixEncoder(nn.Module):
    """Frozen official input path through block 9, retaining the CLS token.

    The input is four seconds of standard-19 EEG in volts with shape
    ``[B,19,4,200]``.  The output is the detached activation immediately
    before block 10 with shape ``[B,77,200]``.  Record-specific legacy/modern
    electrode position IDs are supported exactly as in
    :class:`OfficialLaBraMEncoder`.
    """

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
    ) -> None:
        super().__init__()
        self.encoder = OfficialLaBraMEncoder(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            expected_modeling_sha256=expected_modeling_sha256,
            tile_seconds=LABRAM_PEFT_SECONDS_PER_CALL,
        )
        self.receipt = self.encoder.receipt
        _validate_official_suffix_backbone(self.encoder.backbone)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "OfficialLaBraMFrozenPrefixEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def _forward_with_input_chans(
        self,
        patches: torch.Tensor,
        input_chans: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            N_STANDARD_CHANNELS,
            LABRAM_PEFT_SECONDS_PER_CALL,
            self.encoder.samples_per_token,
        )
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                "LaBraM frozen prefix input must have shape [B,19,4,200], got "
                f"{tuple(patches.shape)}"
            )
        if patches.shape[0] < 1:
            raise ValueError("LaBraM frozen prefix batch must be non-empty")
        if not patches.is_floating_point():
            raise TypeError("LaBraM frozen prefix input must be floating point")
        if not torch.isfinite(patches).all():
            raise ValueError("LaBraM frozen prefix input must be finite")
        if tuple(input_chans.shape) != (N_STANDARD_CHANNELS + 1,):
            raise ValueError("LaBraM frozen prefix input_chans must have shape [20]")
        if input_chans.dtype != torch.long:
            raise TypeError("LaBraM frozen prefix input_chans must be torch.long")

        backbone = self.encoder.backbone
        reference = next(backbone.parameters())
        if patches.device != reference.device or patches.dtype != reference.dtype:
            raise ValueError("LaBraM frozen prefix input must share suffix dtype/device")
        input_chans = input_chans.to(device=patches.device)
        self.encoder.eval()
        with torch.no_grad():
            tokens = backbone.patch_embed(
                patches * self.receipt.input_scale_from_volts
            )
            cls = backbone.cls_token.expand(patches.shape[0], -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)

            position_used = backbone.pos_embed[:, input_chans]
            position = (
                position_used[:, 1:, :]
                .unsqueeze(2)
                .expand(
                    patches.shape[0],
                    -1,
                    LABRAM_PEFT_SECONDS_PER_CALL,
                    -1,
                )
                .flatten(1, 2)
            )
            position = torch.cat(
                (
                    position_used[:, 0:1, :].expand(patches.shape[0], -1, -1),
                    position,
                ),
                dim=1,
            )
            tokens = tokens + position
            time = (
                backbone.time_embed[:, :LABRAM_PEFT_SECONDS_PER_CALL, :]
                .unsqueeze(1)
                .expand(patches.shape[0], N_STANDARD_CHANNELS, -1, -1)
                .flatten(1, 2)
            )
            tokens[:, 1:, :] += time
            tokens = backbone.pos_drop(tokens)
            for block in backbone.blocks[: LABRAM_PEFT_BLOCKS[0]]:
                tokens = block(tokens, rel_pos_bias=None)

        expected_output = (
            patches.shape[0],
            LABRAM_PEFT_PREFIX_TOKENS,
            LABRAM_PEFT_TOKEN_DIM,
        )
        if tuple(tokens.shape) != expected_output:
            raise RuntimeError("LaBraM frozen prefix output shape drifted")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("LaBraM frozen prefix returned non-finite tokens")
        return tokens.detach()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self._forward_with_input_chans(patches, self.encoder.input_chans)

    def forward_with_record_binding(
        self,
        patches: torch.Tensor,
        binding: LaBraMRecordPositionBinding,
    ) -> torch.Tensor:
        receipt = self.encoder.feature_receipt_for_record_binding(binding)
        input_chans = torch.tensor(
            (0, *receipt.position_ids),
            dtype=torch.long,
            device=patches.device,
        )
        return self._forward_with_input_chans(patches, input_chans)


class OfficialLaBraMMinimalPEFTSuffix(nn.Module):
    """Differentiable blocks-10/11 suffix over a detached prefix cache.

    Input shape is ``[BT,77,200]`` where ``BT`` is any positive number of
    independent four-second LaBraM calls.  Token zero is the CLS token and the
    remaining 76 tokens retain official channel-major, second-minor order.
    Output shape is ``[BT,19,4,200]`` after blocks 10/11 and the official final
    LayerNorm.  The CLS output is not returned, matching
    ``OfficialLaBraMEncoder(..., tile_seconds=4)``.
    """

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
        config: LaBraMMinimalPEFTConfig = LaBraMMinimalPEFTConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(config, LaBraMMinimalPEFTConfig):
            raise TypeError("config must be LaBraMMinimalPEFTConfig")
        # Load and strictly validate the official checkpoint before changing
        # any parameter name with torch parametrizations.
        official = OfficialLaBraMEncoder(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            expected_modeling_sha256=expected_modeling_sha256,
            tile_seconds=LABRAM_PEFT_SECONDS_PER_CALL,
        )
        self.backbone = official.backbone
        self.receipt = official.receipt
        self.config = config
        _validate_official_suffix_backbone(self.backbone)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block_index in self.config.block_indices:
            qkv = self.backbone.blocks[block_index].attn.qkv
            parametrize.register_parametrization(
                qkv,
                "weight",
                LaBraMQKVWeightLoRA(),
                unsafe=False,
            )
            qkv.parametrizations.weight.original.requires_grad_(False)

        self.backbone.eval()
        self._assert_trainable_contract()

    def train(self, mode: bool = True) -> "OfficialLaBraMMinimalPEFTSuffix":
        # Evaluation mode is intentional: the locked candidate has no adapter
        # dropout, and the official backbone must never introduce stochastic
        # dropout/drop-path state.  eval() does not disable autograd.
        super().train(mode)
        self.backbone.eval()
        return self

    def _lora(self, block_index: int) -> LaBraMQKVWeightLoRA:
        qkv = self.backbone.blocks[block_index].attn.qkv
        if not parametrize.is_parametrized(qkv, "weight"):
            raise RuntimeError(f"LaBraM block {block_index} lost qkv parametrization")
        parametrizations = qkv.parametrizations.weight
        if len(parametrizations) != 1 or not isinstance(
            parametrizations[0], LaBraMQKVWeightLoRA
        ):
            raise RuntimeError(f"LaBraM block {block_index} qkv adapter changed")
        return parametrizations[0]

    @property
    def n_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def _assert_trainable_contract(self) -> None:
        expected = tuple(
            f"backbone.blocks.{block}.attn.qkv.parametrizations.weight.0.lora_{factor}"
            for block in LABRAM_PEFT_BLOCKS
            for factor in ("A", "B")
        )
        if self.trainable_parameter_names != expected:
            raise RuntimeError(
                "LaBraM PEFT trainable scope changed: "
                f"expected {expected}, got {self.trainable_parameter_names}"
            )
        if self.n_trainable_parameters != LABRAM_PEFT_TRAINABLE_PARAMETERS:
            raise RuntimeError("LaBraM PEFT must expose exactly 6,400 parameters")
        for block_index in LABRAM_PEFT_BLOCKS:
            original = self.backbone.blocks[
                block_index
            ].attn.qkv.parametrizations.weight.original
            if original.requires_grad:
                raise RuntimeError("Original LaBraM qkv weights must remain frozen")

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        expected_tail = (LABRAM_PEFT_PREFIX_TOKENS, LABRAM_PEFT_TOKEN_DIM)
        if prefix_tokens.ndim != 3 or tuple(prefix_tokens.shape[1:]) != expected_tail:
            raise ValueError(
                "LaBraM PEFT prefix must have shape [BT,77,200], got "
                f"{tuple(prefix_tokens.shape)}"
            )
        if prefix_tokens.shape[0] < 1:
            raise ValueError("LaBraM PEFT prefix batch must be non-empty")
        if not prefix_tokens.is_floating_point():
            raise TypeError("LaBraM PEFT prefix must be floating point")
        if prefix_tokens.requires_grad:
            raise ValueError("LaBraM PEFT prefix cache must be detached")
        if not torch.isfinite(prefix_tokens).all():
            raise ValueError("LaBraM PEFT prefix must be finite")

        reference = self.backbone.blocks[10].attn.qkv.parametrizations.weight.original
        if prefix_tokens.device != reference.device or prefix_tokens.dtype != reference.dtype:
            raise ValueError("LaBraM PEFT prefix and suffix must share dtype/device")
        self._assert_trainable_contract()

        tokens = prefix_tokens
        for block_index in self.config.block_indices:
            tokens = self.backbone.blocks[block_index](tokens, rel_pos_bias=None)
        tokens = self.backbone.norm(tokens)
        patch_tokens = tokens[:, 1:, :]
        output = patch_tokens.reshape(
            prefix_tokens.shape[0],
            N_STANDARD_CHANNELS,
            LABRAM_PEFT_SECONDS_PER_CALL,
            LABRAM_PEFT_TOKEN_DIM,
        )
        if tuple(output.shape) != (
            prefix_tokens.shape[0],
            N_STANDARD_CHANNELS,
            LABRAM_PEFT_SECONDS_PER_CALL,
            LABRAM_PEFT_TOKEN_DIM,
        ):
            raise RuntimeError("LaBraM PEFT suffix output shape drifted")
        if not torch.isfinite(output).all():
            raise RuntimeError("LaBraM PEFT suffix returned non-finite tokens")
        return output

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a portable adapter-only state, never checkpoint weights."""

        state: dict[str, torch.Tensor] = {}
        for block_index in LABRAM_PEFT_BLOCKS:
            adapter = self._lora(block_index)
            state[f"blocks.{block_index}.attn.qkv.lora_A"] = (
                adapter.lora_A.detach().cpu().clone()
            )
            state[f"blocks.{block_index}.attn.qkv.lora_B"] = (
                adapter.lora_B.detach().cpu().clone()
            )
        return state

    def load_lora_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Strictly and atomically restore the four LoRA factor tensors."""

        if not isinstance(state, Mapping):
            raise TypeError("LoRA state must be a tensor mapping")
        if any(not isinstance(key, str) for key in state):
            raise TypeError("LoRA state keys must be strings")
        expected = {
            f"blocks.{block}.attn.qkv.lora_{factor}"
            for block in LABRAM_PEFT_BLOCKS
            for factor in ("A", "B")
        }
        if set(state) != expected:
            raise ValueError(
                "LoRA state keys changed; "
                f"missing={sorted(expected-set(state))}, "
                f"extra={sorted(set(state)-expected)}"
            )

        validated: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block_index in LABRAM_PEFT_BLOCKS:
            adapter = self._lora(block_index)
            for factor in ("A", "B"):
                key = f"blocks.{block_index}.attn.qkv.lora_{factor}"
                value = state[key]
                target = getattr(adapter, f"lora_{factor}")
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"LoRA state {key} must be a tensor")
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(f"LoRA state {key} has the wrong shape")
                if not value.is_floating_point():
                    raise TypeError(f"LoRA state {key} must be floating point")
                if not torch.isfinite(value).all():
                    raise ValueError(f"LoRA state {key} must be finite")
                validated.append((target, value))

        with torch.no_grad():
            for target, value in validated:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
        self._assert_trainable_contract()


__all__ = [
    "LABRAM_PEFT_ALPHA",
    "LABRAM_PEFT_BLOCKS",
    "LABRAM_PEFT_DROPOUT",
    "LABRAM_PEFT_PREFIX_TOKENS",
    "LABRAM_PEFT_RANK",
    "LABRAM_PEFT_TRAINABLE_PARAMETERS",
    "LaBraMMinimalPEFTConfig",
    "LaBraMQKVWeightLoRA",
    "OfficialLaBraMFrozenPrefixEncoder",
    "OfficialLaBraMMinimalPEFTSuffix",
]
