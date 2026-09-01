"""Strictly frozen LaBraM blocks-10/11 suffix for locked comparisons.

The historical PEFT suffix deliberately requires its four LoRA factors to be
trainable.  A qualified source-only adapter must instead be read-only during
downstream SOZ fitting.  This subclass preserves every architectural and
adapter-state check while switching, only after construction/loading, to a
zero-trainable-parameter inference contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from .labram import AUDITED_LABRAM_BASE_SHA256, AUDITED_LABRAM_MODELING_SHA256
from .labram_peft import (
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_QKV_SHAPE,
    LABRAM_PEFT_RANK,
    LABRAM_PEFT_TOKEN_DIM,
    LaBraMMinimalPEFTConfig,
    OfficialLaBraMMinimalPEFTSuffix,
)


class OfficialLaBraMStaticAdapterSuffix(OfficialLaBraMMinimalPEFTSuffix):
    """Official final suffix with an exact-zero or fixed LoRA adapter.

    ``adapter_state=None`` creates the exact zero-LoRA arm (every LoRA-B is
    exactly zero).  Supplying a state strictly restores all four rank-four
    factors before every model parameter is frozen.  Calls to ``train`` never
    leave evaluation mode, and the inherited forward dynamically invokes the
    static contract below.
    """

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        adapter_state: Mapping[str, torch.Tensor] | None = None,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
        config: LaBraMMinimalPEFTConfig = LaBraMMinimalPEFTConfig(),
    ) -> None:
        # The parent constructor calls ``self._assert_trainable_contract``.
        # Dynamic dispatch must use the original PEFT contract until loading
        # has completed, then permanently switch to the static contract.
        object.__setattr__(self, "_static_contract_active", False)
        super().__init__(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            expected_modeling_sha256=expected_modeling_sha256,
            config=config,
        )
        if adapter_state is not None:
            self.load_lora_state_dict(adapter_state)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_static_contract_active", True)
        self.train(False)
        self._assert_trainable_contract()

    def _assert_trainable_contract(self) -> None:
        if not bool(getattr(self, "_static_contract_active", False)):
            super()._assert_trainable_contract()
            return
        if self.config.block_indices != LABRAM_PEFT_BLOCKS:
            raise RuntimeError("Static LaBraM suffix block scope changed")
        if self.config.rank != LABRAM_PEFT_RANK:
            raise RuntimeError("Static LaBraM suffix rank changed")
        for block in LABRAM_PEFT_BLOCKS:
            adapter = self._lora(block)
            if tuple(adapter.lora_A.shape) != (LABRAM_PEFT_RANK, LABRAM_PEFT_TOKEN_DIM):
                raise RuntimeError("Static LaBraM LoRA-A shape changed")
            if tuple(adapter.lora_B.shape) != (
                LABRAM_PEFT_QKV_SHAPE[0],
                LABRAM_PEFT_RANK,
            ):
                raise RuntimeError("Static LaBraM LoRA-B shape changed")
            original = self.backbone.blocks[
                block
            ].attn.qkv.parametrizations.weight.original
            if original.requires_grad:
                raise RuntimeError("Static LaBraM original qkv must remain frozen")
        if self.trainable_parameter_names or self.n_trainable_parameters != 0:
            raise RuntimeError("Static LaBraM suffix must expose zero trainable parameters")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("Static LaBraM suffix contains a trainable parameter")

    def train(self, mode: bool = True) -> "OfficialLaBraMStaticAdapterSuffix":
        # Fixed suffixes are feature producers only.  Ignore attempts to enter
        # train mode while preserving normal ``eval()`` compatibility.
        nn.Module.train(self, False)
        self.backbone.eval()
        if bool(getattr(self, "_static_contract_active", False)):
            self._assert_trainable_contract()
        return self


__all__ = ["OfficialLaBraMStaticAdapterSuffix"]
