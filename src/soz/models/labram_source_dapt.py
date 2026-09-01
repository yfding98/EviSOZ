"""Official masked-neural-code LaBraM DAPT with one locked qkv-LoRA scope.

This is not a downstream SOZ head.  It reproduces LaBraM's original two-pass
masked neural-code prediction objective: a frozen official VQ-NSP tokenizer
provides one 8192-way code per channel-second patch, while the pretrained
LaBraM code-prediction head scores masked patches and their complementary
mask.  Only qkv LoRA in transformer blocks 10 and 11 is trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from ..geometry import N_STANDARD_CHANNELS
from .labram import (
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    OfficialLaBraMEncoder,
    _load_checkpoint_snapshot,
    _stable_file_snapshot,
)
from .labram_peft import (
    LABRAM_PEFT_ALPHA,
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_RANK,
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
    LaBraMQKVWeightLoRA,
    _validate_official_suffix_backbone,
)


AUDITED_VQNSP_SHA256 = (
    "d8f14b4232f06a5c37b6386ab021c270dd4bfa12bfecdd60372dc8ac7b711101"
)
AUDITED_VQNSP_MODELING_SHA256 = (
    "f849a0bda827f767d0b75c7d28b4b14dfa2d87911d5fcbaa195dc8628e48c625"
)
AUDITED_VQNSP_QUANTIZER_SHA256 = (
    "9234ec1848ca602b6cfec1e24d1a1c5bd6a59b9c938259b26cf256d014409506"
)
AUDITED_LABRAM_PRETRAINING_MODELING_SHA256 = (
    "c5a013c9a220b14697556e12195a8bfb5ba29a79031fadffc545bea8bd463b6a"
)

LABRAM_DAPT_SECONDS = 8
LABRAM_DAPT_SAMPLES_PER_TOKEN = 200
LABRAM_DAPT_TOKEN_DIM = 200
LABRAM_DAPT_PATCH_TOKENS = N_STANDARD_CHANNELS * LABRAM_DAPT_SECONDS
LABRAM_DAPT_SEQUENCE_TOKENS = 1 + LABRAM_DAPT_PATCH_TOKENS
LABRAM_DAPT_VOCAB_SIZE = 8192
LABRAM_DAPT_CODE_DIM = 64
LABRAM_DAPT_MASK_RATIO = 0.5
LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS = 1e4
LABRAM_DAPT_OBJECTIVE = "official_two_pass_complementary_masked_neural_code_ce_v1"


def _require_sha256(path: Path, expected: str, label: str) -> None:
    snapshot = _stable_file_snapshot(path, label=label)
    if snapshot.sha256 != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {snapshot.sha256}"
        )


def _load_exact_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_path != path.resolve(strict=True):
            raise RuntimeError(
                f"Python module {name!r} is already loaded from {existing_path}, "
                f"not the audited official source {path}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audited official module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_official_vqnsp_module(official_root_value: str | Path) -> ModuleType:
    root = Path(official_root_value).resolve(strict=True)
    modeling_finetune = root / "modeling_finetune.py"
    quantizer = root / "norm_ema_quantizer.py"
    modeling_vqnsp = root / "modeling_vqnsp.py"
    _require_sha256(
        modeling_finetune,
        AUDITED_LABRAM_MODELING_SHA256,
        "Official LaBraM fine-tuning model source",
    )
    _require_sha256(
        quantizer,
        AUDITED_VQNSP_QUANTIZER_SHA256,
        "Official LaBraM VQ quantizer source",
    )
    _require_sha256(
        modeling_vqnsp,
        AUDITED_VQNSP_MODELING_SHA256,
        "Official LaBraM VQ-NSP model source",
    )
    _load_exact_module("modeling_finetune", modeling_finetune)
    _load_exact_module("norm_ema_quantizer", quantizer)
    return _load_exact_module("modeling_vqnsp", modeling_vqnsp)


class OfficialFrozenLaBraMVQTokenizer(nn.Module):
    """Pinned official 8192-code, 64-dimensional VQ-NSP tokenizer."""

    def __init__(
        self,
        *,
        official_root: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_VQNSP_SHA256,
    ) -> None:
        super().__init__()
        module = _load_official_vqnsp_module(official_root)
        payload, snapshot = _load_checkpoint_snapshot(
            checkpoint_path, expected_sha256=expected_sha256
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
            raise TypeError("Official VQ-NSP checkpoint must contain a model mapping")
        args = payload.get("args")
        if args is None:
            raise TypeError("Official VQ-NSP checkpoint lacks its training arguments")
        contract = {
            "model": getattr(args, "model", None),
            "codebook_n_emd": getattr(args, "codebook_n_emd", None),
            "codebook_emd_dim": getattr(args, "codebook_emd_dim", None),
            "input_size": getattr(args, "input_size", None),
        }
        if contract != {
            "model": "vqkd_encoder_base_decoder_3x200x12",
            "codebook_n_emd": LABRAM_DAPT_VOCAB_SIZE,
            "codebook_emd_dim": LABRAM_DAPT_CODE_DIM,
            "input_size": LABRAM_DAPT_SECONDS * LABRAM_DAPT_SAMPLES_PER_TOKEN,
        }:
            raise ValueError(f"Official VQ-NSP tokenizer contract changed: {contract}")
        self.tokenizer = module.vqnsp_encoder_base_decoder_3x200x12(
            pretrained=False,
            EEG_size=LABRAM_DAPT_SECONDS * LABRAM_DAPT_SAMPLES_PER_TOKEN,
            n_code=LABRAM_DAPT_VOCAB_SIZE,
            code_dim=LABRAM_DAPT_CODE_DIM,
            quantize_kmeans_init=True,
        )
        self.tokenizer.load_state_dict(dict(payload["model"]), strict=True)
        embedding = self.tokenizer.quantize.embedding.weight
        if tuple(embedding.shape) != (LABRAM_DAPT_VOCAB_SIZE, LABRAM_DAPT_CODE_DIM):
            raise ValueError("Official VQ-NSP embedding must have shape [8192,64]")
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad_(False)
        if not bool(self.tokenizer.quantize.embedding.initted.item()):
            raise ValueError("Official VQ-NSP checkpoint codebook is not initialized")
        # Upstream permits EMA codebook mutation in training mode.  DAPT uses
        # a frozen tokenizer, so disable that path in addition to eval().
        self.tokenizer.quantize.embedding.update = False
        self.tokenizer.eval()
        self.checkpoint_path = snapshot.path
        self.checkpoint_sha256 = snapshot.sha256
        self.codebook_size = LABRAM_DAPT_VOCAB_SIZE
        self.code_dim = LABRAM_DAPT_CODE_DIM
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.tokenizer.parameters()):
            raise RuntimeError("Official VQ-NSP tokenizer must remain fully frozen")
        embedding = self.tokenizer.quantize.embedding.weight
        if tuple(embedding.shape) != (LABRAM_DAPT_VOCAB_SIZE, LABRAM_DAPT_CODE_DIM):
            raise RuntimeError("Official VQ-NSP codebook shape changed")
        if self.tokenizer.quantize.embedding.update is not False:
            raise RuntimeError("Official VQ-NSP EMA codebook updates must be disabled")
        if not bool(self.tokenizer.quantize.embedding.initted.item()):
            raise RuntimeError("Official VQ-NSP codebook lost its initialized state")

    def train(self, mode: bool = True) -> "OfficialFrozenLaBraMVQTokenizer":
        super().train(mode)
        self.tokenizer.eval()
        self._assert_frozen()
        return self

    def forward(
        self, patches_volts: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        expected = (
            N_STANDARD_CHANNELS,
            LABRAM_DAPT_SECONDS,
            LABRAM_DAPT_SAMPLES_PER_TOKEN,
        )
        if patches_volts.ndim != 4 or tuple(patches_volts.shape[1:]) != expected:
            raise ValueError(
                "Official VQ-NSP input must have shape [B,19,8,200], got "
                f"{tuple(patches_volts.shape)}"
            )
        if not patches_volts.is_floating_point() or not torch.isfinite(patches_volts).all():
            raise ValueError("Official VQ-NSP input must be finite floating point")
        if tuple(position_ids.shape) != (N_STANDARD_CHANNELS,) or position_ids.dtype != torch.long:
            raise ValueError("VQ-NSP position_ids must be long [19]")
        reference = next(self.tokenizer.parameters())
        if patches_volts.device != reference.device or patches_volts.dtype != reference.dtype:
            raise ValueError("VQ-NSP input and tokenizer must share dtype/device")
        input_chans = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=patches_volts.device),
                position_ids.to(device=patches_volts.device),
            )
        )
        self.tokenizer.eval()
        self._assert_frozen()
        frozen_buffers = {
            name: value.detach().clone()
            for name, value in self.tokenizer.named_buffers()
        }
        with torch.no_grad():
            try:
                indices = self.tokenizer.get_codebook_indices(
                    patches_volts * LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS,
                    input_chans,
                )
            finally:
                current_buffers = dict(self.tokenizer.named_buffers())
                if set(current_buffers) != set(frozen_buffers):
                    raise RuntimeError("Official VQ-NSP buffer roster changed")
                for name, before in frozen_buffers.items():
                    current_buffers[name].copy_(before)
        expected_output = (patches_volts.shape[0], LABRAM_DAPT_PATCH_TOKENS)
        if tuple(indices.shape) != expected_output or indices.dtype != torch.long:
            raise RuntimeError(
                f"Official VQ-NSP returned {tuple(indices.shape)} {indices.dtype}, "
                f"expected {expected_output} torch.long"
            )
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= LABRAM_DAPT_VOCAB_SIZE
        ):
            raise RuntimeError("Official VQ-NSP returned an out-of-range neural code")
        return indices.detach()


class OfficialLaBraMSourceDAPT(nn.Module):
    """Mask-before-block-0 LaBraM code predictor with locked qkv LoRA."""

    def __init__(
        self,
        *,
        modeling_path: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str = AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256: str = AUDITED_LABRAM_MODELING_SHA256,
    ) -> None:
        super().__init__()
        official = OfficialLaBraMEncoder(
            modeling_path=modeling_path,
            checkpoint_path=checkpoint_path,
            expected_sha256=expected_sha256,
            expected_modeling_sha256=expected_modeling_sha256,
            tile_seconds=LABRAM_DAPT_SECONDS,
        )
        self.backbone = official.backbone
        self.receipt = official.receipt
        _validate_official_suffix_backbone(self.backbone)
        payload, checkpoint = _load_checkpoint_snapshot(
            checkpoint_path, expected_sha256=expected_sha256
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
            raise TypeError("Official LaBraM checkpoint must contain a model mapping")
        state = payload["model"]
        required = {
            "student.mask_token": (1, 1, LABRAM_DAPT_TOKEN_DIM),
            "lm_head.weight": (LABRAM_DAPT_VOCAB_SIZE, LABRAM_DAPT_TOKEN_DIM),
            "lm_head.bias": (LABRAM_DAPT_VOCAB_SIZE,),
        }
        for key, shape in required.items():
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Official LaBraM pretext tensor {key} changed")
            if not torch.isfinite(value).all():
                raise ValueError(f"Official LaBraM pretext tensor {key} is non-finite")
        self.register_buffer(
            "mask_token", state["student.mask_token"].detach().clone(), persistent=True
        )
        self.lm_head = nn.Linear(
            LABRAM_DAPT_TOKEN_DIM, LABRAM_DAPT_VOCAB_SIZE, bias=True
        )
        with torch.no_grad():
            self.lm_head.weight.copy_(state["lm_head.weight"])
            self.lm_head.bias.copy_(state["lm_head.bias"])
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lm_head.parameters():
            parameter.requires_grad_(False)
        for block_index in LABRAM_PEFT_BLOCKS:
            qkv = self.backbone.blocks[block_index].attn.qkv
            parametrize.register_parametrization(
                qkv, "weight", LaBraMQKVWeightLoRA(), unsafe=False
            )
            qkv.parametrizations.weight.original.requires_grad_(False)
        self.checkpoint_sha256 = checkpoint.sha256
        self.backbone.eval()
        self.lm_head.eval()
        self._assert_contract()

    def train(self, mode: bool = True) -> "OfficialLaBraMSourceDAPT":
        super().train(mode)
        self.backbone.eval()
        self.lm_head.eval()
        self._assert_contract()
        return self

    def _lora(self, block_index: int) -> LaBraMQKVWeightLoRA:
        qkv = self.backbone.blocks[block_index].attn.qkv
        if not parametrize.is_parametrized(qkv, "weight"):
            raise RuntimeError(f"LaBraM DAPT block {block_index} lost qkv LoRA")
        adapters = qkv.parametrizations.weight
        if len(adapters) != 1 or not isinstance(adapters[0], LaBraMQKVWeightLoRA):
            raise RuntimeError(f"LaBraM DAPT block {block_index} LoRA changed")
        return adapters[0]

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    @property
    def n_trainable_parameters(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _assert_contract(self) -> None:
        expected = tuple(
            f"backbone.blocks.{block}.attn.qkv.parametrizations.weight.0.lora_{factor}"
            for block in LABRAM_PEFT_BLOCKS
            for factor in ("A", "B")
        )
        if self.trainable_parameter_names != expected:
            raise RuntimeError(
                "LaBraM source DAPT trainable scope changed: "
                f"expected={expected}, got={self.trainable_parameter_names}"
            )
        if self.n_trainable_parameters != LABRAM_PEFT_TRAINABLE_PARAMETERS:
            raise RuntimeError("LaBraM source DAPT must expose exactly 6,400 parameters")
        if LABRAM_PEFT_RANK != 4 or LABRAM_PEFT_ALPHA != 8.0:
            raise RuntimeError("LaBraM source DAPT LoRA rank/alpha contract changed")
        if self.mask_token.requires_grad or any(
            parameter.requires_grad for parameter in self.lm_head.parameters()
        ):
            raise RuntimeError("LaBraM mask token and neural-code head must be frozen")

    def _masked_prefix(
        self,
        patches_volts: torch.Tensor,
        position_ids: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch = patches_volts.shape[0]
        input_chans = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=patches_volts.device),
                position_ids.to(device=patches_volts.device),
            )
        )
        backbone = self.backbone
        with torch.no_grad():
            tokens = backbone.patch_embed(
                patches_volts * LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS
            )
            mask = bool_masked_pos.unsqueeze(-1).to(dtype=tokens.dtype)
            mask_token = self.mask_token.to(dtype=tokens.dtype).expand(
                batch, LABRAM_DAPT_PATCH_TOKENS, -1
            )
            tokens = tokens * (1.0 - mask) + mask_token * mask
            cls = backbone.cls_token.expand(batch, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            position_used = backbone.pos_embed[:, input_chans]
            patch_position = (
                position_used[:, 1:, :]
                .unsqueeze(2)
                .expand(batch, -1, LABRAM_DAPT_SECONDS, -1)
                .flatten(1, 2)
            )
            # Exact upstream pretraining implementation: its CLS slot uses
            # the first expanded patch position, not position_used[:,0].
            position = torch.cat(
                (patch_position[:, 0:1, :].expand(batch, -1, -1), patch_position),
                dim=1,
            )
            tokens = tokens + position
            time = (
                backbone.time_embed[:, :LABRAM_DAPT_SECONDS, :]
                .unsqueeze(1)
                .expand(batch, N_STANDARD_CHANNELS, -1, -1)
                .flatten(1, 2)
            )
            tokens[:, 1:, :] += time
            tokens = backbone.pos_drop(tokens)
            for block in backbone.blocks[: LABRAM_PEFT_BLOCKS[0]]:
                tokens = block(tokens, rel_pos_bias=None)
        expected = (batch, LABRAM_DAPT_SEQUENCE_TOKENS, LABRAM_DAPT_TOKEN_DIM)
        if tuple(tokens.shape) != expected or tokens.requires_grad:
            raise RuntimeError("LaBraM DAPT frozen prefix contract changed")
        return tokens.detach()

    def forward_selected_logits(
        self,
        patches_volts: torch.Tensor,
        position_ids: torch.Tensor,
        bool_masked_pos: torch.Tensor,
        *,
        selected_positions: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            N_STANDARD_CHANNELS,
            LABRAM_DAPT_SECONDS,
            LABRAM_DAPT_SAMPLES_PER_TOKEN,
        )
        if patches_volts.ndim != 4 or tuple(patches_volts.shape[1:]) != expected:
            raise ValueError("LaBraM DAPT input must have shape [B,19,8,200]")
        if not patches_volts.is_floating_point() or not torch.isfinite(patches_volts).all():
            raise ValueError("LaBraM DAPT input must be finite floating point")
        expected_mask = (patches_volts.shape[0], LABRAM_DAPT_PATCH_TOKENS)
        for name, value in (
            ("bool_masked_pos", bool_masked_pos),
            ("selected_positions", selected_positions),
        ):
            if tuple(value.shape) != expected_mask or value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool {expected_mask}")
        if tuple(position_ids.shape) != (N_STANDARD_CHANNELS,) or position_ids.dtype != torch.long:
            raise ValueError("LaBraM DAPT position_ids must be long [19]")
        reference = self.backbone.blocks[10].attn.qkv.parametrizations.weight.original
        if patches_volts.device != reference.device or patches_volts.dtype != reference.dtype:
            raise ValueError("LaBraM DAPT input and model must share dtype/device")
        self._assert_contract()
        tokens = self._masked_prefix(patches_volts, position_ids, bool_masked_pos)
        for block_index in LABRAM_PEFT_BLOCKS:
            tokens = self.backbone.blocks[block_index](tokens, rel_pos_bias=None)
        tokens = self.backbone.norm(tokens)[:, 1:, :]
        selected = tokens[selected_positions]
        logits = self.lm_head(selected)
        if tuple(logits.shape) != (int(selected_positions.sum()), LABRAM_DAPT_VOCAB_SIZE):
            raise RuntimeError("LaBraM DAPT neural-code logits have the wrong shape")
        if not torch.isfinite(logits).all():
            raise RuntimeError("LaBraM DAPT neural-code logits are non-finite")
        return logits

    def complementary_logits(
        self,
        patches_volts: torch.Tensor,
        position_ids: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = self.forward_selected_logits(
            patches_volts,
            position_ids,
            bool_masked_pos,
            selected_positions=bool_masked_pos,
        )
        visible = self.forward_selected_logits(
            patches_volts,
            position_ids,
            ~bool_masked_pos,
            selected_positions=~bool_masked_pos,
        )
        return masked, visible

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            f"blocks.{block}.attn.qkv.lora_{factor}": getattr(
                self._lora(block), f"lora_{factor}"
            ).detach().cpu().clone()
            for block in LABRAM_PEFT_BLOCKS
            for factor in ("A", "B")
        }

    def load_lora_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {
            f"blocks.{block}.attn.qkv.lora_{factor}"
            for block in LABRAM_PEFT_BLOCKS
            for factor in ("A", "B")
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("LaBraM source-DAPT adapter state keys changed")
        validated: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in LABRAM_PEFT_BLOCKS:
            adapter = self._lora(block)
            for factor in ("A", "B"):
                key = f"blocks.{block}.attn.qkv.lora_{factor}"
                value = state[key]
                target = getattr(adapter, f"lora_{factor}")
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != tuple(target.shape)
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid LaBraM source-DAPT state tensor: {key}")
                validated.append((target, value))
        with torch.no_grad():
            for target, value in validated:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
        self._assert_contract()


@dataclass(frozen=True)
class MaskedNeuralCodeObjectiveOutput:
    loss: torch.Tensor
    masked_loss: torch.Tensor
    complementary_loss: torch.Tensor
    masked_accuracy: torch.Tensor
    complementary_accuracy: torch.Tensor
    neural_codes: torch.Tensor


def exact_random_mask(
    batch_size: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.rand(
        batch_size,
        LABRAM_DAPT_PATCH_TOKENS,
        generator=generator,
        device=device,
    )
    order = torch.argsort(noise, dim=1)
    keep = int(LABRAM_DAPT_PATCH_TOKENS * (1.0 - LABRAM_DAPT_MASK_RATIO))
    mask = torch.ones(
        batch_size, LABRAM_DAPT_PATCH_TOKENS, dtype=torch.bool, device=device
    )
    mask.scatter_(1, order[:, :keep], False)
    if not torch.all(mask.sum(dim=1) == LABRAM_DAPT_PATCH_TOKENS - keep):
        raise RuntimeError("Exact LaBraM DAPT mask cardinality changed")
    return mask


def masked_neural_code_objective(
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    patches_volts: torch.Tensor,
    position_ids_by_sample: torch.Tensor,
    bool_masked_pos: torch.Tensor,
) -> MaskedNeuralCodeObjectiveOutput:
    """Evaluate the official objective, grouping real EDF position bindings."""

    batch = patches_volts.shape[0]
    if tuple(position_ids_by_sample.shape) != (batch, N_STANDARD_CHANNELS):
        raise ValueError("Per-sample LaBraM position IDs must have shape [B,19]")
    if position_ids_by_sample.dtype != torch.long:
        raise TypeError("Per-sample LaBraM position IDs must be torch.long")
    if tuple(bool_masked_pos.shape) != (batch, LABRAM_DAPT_PATCH_TOKENS):
        raise ValueError("LaBraM DAPT mask must have shape [B,152]")
    unique, inverse = torch.unique(
        position_ids_by_sample, dim=0, sorted=True, return_inverse=True
    )
    masked_loss_sum = patches_volts.new_zeros(())
    complementary_loss_sum = patches_volts.new_zeros(())
    masked_correct = patches_volts.new_zeros(())
    complementary_correct = patches_volts.new_zeros(())
    masked_count = 0
    complementary_count = 0
    all_codes = torch.empty(
        batch,
        LABRAM_DAPT_PATCH_TOKENS,
        dtype=torch.long,
        device=patches_volts.device,
    )
    for binding_index in range(unique.shape[0]):
        rows = torch.nonzero(inverse == binding_index, as_tuple=False).flatten()
        group_eeg = patches_volts.index_select(0, rows)
        group_mask = bool_masked_pos.index_select(0, rows)
        position_ids = unique[binding_index]
        codes = tokenizer(group_eeg, position_ids)
        all_codes.index_copy_(0, rows, codes)
        masked_logits, complementary_logits = model.complementary_logits(
            group_eeg, position_ids, group_mask
        )
        masked_target = codes[group_mask]
        complementary_target = codes[~group_mask]
        masked_loss_sum = masked_loss_sum + F.cross_entropy(
            masked_logits, masked_target, reduction="sum"
        )
        complementary_loss_sum = complementary_loss_sum + F.cross_entropy(
            complementary_logits, complementary_target, reduction="sum"
        )
        masked_correct = masked_correct + (masked_logits.argmax(-1) == masked_target).sum()
        complementary_correct = complementary_correct + (
            complementary_logits.argmax(-1) == complementary_target
        ).sum()
        masked_count += masked_target.numel()
        complementary_count += complementary_target.numel()
    if masked_count < 1 or complementary_count < 1:
        raise RuntimeError("LaBraM DAPT objective has an empty mask partition")
    masked_loss = masked_loss_sum / masked_count
    complementary_loss = complementary_loss_sum / complementary_count
    loss = masked_loss + complementary_loss
    if not torch.isfinite(loss):
        raise RuntimeError("LaBraM DAPT objective is non-finite")
    return MaskedNeuralCodeObjectiveOutput(
        loss=loss,
        masked_loss=masked_loss,
        complementary_loss=complementary_loss,
        masked_accuracy=masked_correct / masked_count,
        complementary_accuracy=complementary_correct / complementary_count,
        neural_codes=all_codes.detach(),
    )


def verify_zero_lora_official_pretraining_parity(
    model: OfficialLaBraMSourceDAPT,
    *,
    official_root: str | Path,
    checkpoint_path: str | Path,
    patches_volts: torch.Tensor,
    position_ids: torch.Tensor,
    bool_masked_pos: torch.Tensor,
) -> dict[str, float]:
    """Compare zero-LoRA logits with the pinned upstream pretraining class."""

    for block in LABRAM_PEFT_BLOCKS:
        if torch.count_nonzero(model._lora(block).lora_B).item() != 0:
            raise ValueError("Official parity requires the initial zero-LoRA state")
    root = Path(official_root).resolve(strict=True)
    _require_sha256(
        root / "modeling_pretrain.py",
        AUDITED_LABRAM_PRETRAINING_MODELING_SHA256,
        "Official LaBraM pretraining model source",
    )
    _load_exact_module("modeling_finetune", root / "modeling_finetune.py")
    module = _load_exact_module("modeling_pretrain", root / "modeling_pretrain.py")
    payload, _ = _load_checkpoint_snapshot(
        checkpoint_path, expected_sha256=AUDITED_LABRAM_BASE_SHA256
    )
    state = dict(payload["model"])
    extra = state.pop("logit_scale", None)
    if not isinstance(extra, torch.Tensor) or tuple(extra.shape) != ():
        raise ValueError("Official LaBraM checkpoint logit_scale contract changed")
    reference = module.labram_base_patch200_1600_8k_vocab(
        drop_path_rate=0.0,
        use_shared_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        vocab_size=LABRAM_DAPT_VOCAB_SIZE,
    )
    reference.load_state_dict(state, strict=True)
    reference.to(device=patches_volts.device, dtype=patches_volts.dtype)
    reference.eval()
    input_chans = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=patches_volts.device),
            position_ids.to(device=patches_volts.device),
        )
    )
    with torch.no_grad():
        expected_masked, expected_complementary = reference(
            patches_volts * LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS,
            input_chans,
            bool_masked_pos,
        )
        actual_masked, actual_complementary = model.complementary_logits(
            patches_volts, position_ids, bool_masked_pos
        )
    errors = {
        "masked_max_abs_error": float(
            (actual_masked - expected_masked).abs().max().detach().cpu()
        ),
        "complementary_max_abs_error": float(
            (actual_complementary - expected_complementary).abs().max().detach().cpu()
        ),
    }
    if errors != {
        "masked_max_abs_error": 0.0,
        "complementary_max_abs_error": 0.0,
    }:
        raise RuntimeError(f"Zero-LoRA official pretraining parity failed: {errors}")
    del reference
    return errors


__all__ = [
    "AUDITED_VQNSP_SHA256",
    "LABRAM_DAPT_CODE_DIM",
    "LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS",
    "LABRAM_DAPT_MASK_RATIO",
    "LABRAM_DAPT_OBJECTIVE",
    "LABRAM_DAPT_PATCH_TOKENS",
    "LABRAM_DAPT_VOCAB_SIZE",
    "MaskedNeuralCodeObjectiveOutput",
    "OfficialFrozenLaBraMVQTokenizer",
    "OfficialLaBraMSourceDAPT",
    "exact_random_mask",
    "masked_neural_code_objective",
    "verify_zero_lora_official_pretraining_parity",
]
