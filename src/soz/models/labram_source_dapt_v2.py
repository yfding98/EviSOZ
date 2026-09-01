"""Diversity-preserving continuation objective for the locked LaBraM DAPT model.

This module defines only an objective.  It does not choose a dataset split,
sampler, optimizer, checkpoint, epoch, or downstream SOZ endpoint.  The
official LaBraM/VQ implementations remain owned by ``labram_source_dapt``;
this v2 layer reuses them without changing the completed v1 implementation.

The zero-LoRA teacher is the same official model evaluated with both LoRA-B
factors temporarily set to exact zero.  Teacher logits are computed before
the student graph, under ``torch.no_grad()``, and every LoRA-B tensor is then
restored bitwise before the student forward.  This ordering is important:
mutating a parameter after a student forward would invalidate autograd's
saved tensor versions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
import torch.nn.functional as F

from ..geometry import N_STANDARD_CHANNELS
from .labram_peft import LABRAM_PEFT_BLOCKS
from .labram_source_dapt import (
    LABRAM_DAPT_PATCH_TOKENS,
    LABRAM_DAPT_VOCAB_SIZE,
    OfficialFrozenLaBraMVQTokenizer,
    OfficialLaBraMSourceDAPT,
)


LABRAM_DAPT_V2_OBJECTIVE = (
    "official_two_pass_ce_plus_zero_lora_teacher_T2_kl_plus_"
    "teacher_relative_batch_marginal_entropy_floor_v1"
)
LABRAM_DAPT_V2_TEACHER_TEMPERATURE = 2.0
LABRAM_DAPT_V2_TEACHER_KL_WEIGHT = 1.0
LABRAM_DAPT_V2_ENTROPY_FLOOR_WEIGHT = 1.0
LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO = 0.98
LABRAM_DAPT_V2_LOG_PERPLEXITY_MARGIN = math.log(
    LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO
)


@dataclass(frozen=True)
class MarginalEntropyFloorOutput:
    """Differentiable student penalty and detached teacher-relative floor."""

    loss: torch.Tensor
    student_entropy: torch.Tensor
    teacher_entropy: torch.Tensor
    entropy_floor: torch.Tensor
    student_effective_perplexity: torch.Tensor
    teacher_effective_perplexity: torch.Tensor
    student_top_probability: torch.Tensor
    teacher_top_probability: torch.Tensor


@dataclass(frozen=True)
class DiversityPreservingObjectiveOutput:
    """All optimized terms and monitoring metrics for one v2 batch."""

    loss: torch.Tensor
    official_ce_loss: torch.Tensor
    masked_ce_loss: torch.Tensor
    complementary_ce_loss: torch.Tensor
    teacher_kl_loss: torch.Tensor
    masked_teacher_kl_loss: torch.Tensor
    complementary_teacher_kl_loss: torch.Tensor
    entropy_floor_loss: torch.Tensor
    student_batch_marginal_entropy: torch.Tensor
    teacher_batch_marginal_entropy: torch.Tensor
    batch_marginal_entropy_floor: torch.Tensor
    student_batch_marginal_effective_perplexity: torch.Tensor
    teacher_batch_marginal_effective_perplexity: torch.Tensor
    student_batch_marginal_top_probability: torch.Tensor
    teacher_batch_marginal_top_probability: torch.Tensor
    masked_accuracy: torch.Tensor
    complementary_accuracy: torch.Tensor
    neural_codes: torch.Tensor


@contextmanager
def _temporarily_disable_lora_b(
    model: OfficialLaBraMSourceDAPT,
) -> Iterator[None]:
    """Set every allowed LoRA-B to zero and restore it bitwise on all exits."""

    model._assert_contract()
    guard_name = "_labram_dapt_v2_zero_teacher_active"
    if bool(getattr(model, guard_name, False)):
        raise RuntimeError("Nested/concurrent zero-LoRA teacher evaluation is forbidden")
    setattr(model, guard_name, True)
    originals = {
        block: model._lora(block).lora_B.detach().clone()
        for block in LABRAM_PEFT_BLOCKS
    }
    try:
        with torch.no_grad():
            for block in LABRAM_PEFT_BLOCKS:
                model._lora(block).lora_B.zero_()
        if any(
            torch.count_nonzero(model._lora(block).lora_B).item() != 0
            for block in LABRAM_PEFT_BLOCKS
        ):
            raise RuntimeError("Zero-LoRA teacher could not disable every LoRA-B")
        yield
    finally:
        with torch.no_grad():
            for block in LABRAM_PEFT_BLOCKS:
                model._lora(block).lora_B.copy_(originals[block])
        delattr(model, guard_name)
        for block in LABRAM_PEFT_BLOCKS:
            if not torch.equal(model._lora(block).lora_B.detach(), originals[block]):
                raise RuntimeError("Zero-LoRA teacher failed to restore LoRA-B bitwise")
        model._assert_contract()


def temperature_scaled_teacher_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = LABRAM_DAPT_V2_TEACHER_TEMPERATURE,
) -> torch.Tensor:
    """Mean token KL(teacher || student), including the standard T^2 factor."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Teacher KL logits must share a two-dimensional shape")
    if student_logits.shape[1] != LABRAM_DAPT_VOCAB_SIZE:
        raise ValueError("Teacher KL requires the official 8192-code logits")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Teacher KL temperature must be finite and positive")
    if teacher_logits.requires_grad:
        raise ValueError("Zero-LoRA teacher logits must not require gradients")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(
        teacher_logits
    ).all():
        raise ValueError("Teacher KL logits must be finite")
    scaled_student_log = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        scaled_teacher_log = F.log_softmax(teacher_logits / temperature, dim=-1)
        scaled_teacher_probability = torch.exp(scaled_teacher_log)
    per_token = torch.sum(
        scaled_teacher_probability
        * (scaled_teacher_log - scaled_student_log),
        dim=-1,
    )
    value = per_token.mean() * (float(temperature) ** 2)
    if not torch.isfinite(value):
        raise RuntimeError("Temperature-scaled teacher KL is non-finite")
    return value


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(probability.dtype).tiny
    return -torch.sum(probability * torch.log(torch.clamp_min(probability, tiny)))


def differentiable_batch_marginal_entropy_floor(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    minimum_perplexity_ratio: float = LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO,
) -> MarginalEntropyFloorOutput:
    """Squared hinge preventing >2% soft marginal perplexity loss vs teacher.

    The batch marginal is the mean full-softmax distribution over the 152
    complementary predictions contributed by every sample.  The teacher floor
    is detached.  Only the student entropy and hinge retain gradients.
    """

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Marginal entropy logits must share [tokens,codes] geometry")
    if student_logits.shape[0] < 1 or student_logits.shape[1] != LABRAM_DAPT_VOCAB_SIZE:
        raise ValueError("Marginal entropy requires non-empty official 8192-code logits")
    if teacher_logits.requires_grad:
        raise ValueError("Marginal entropy teacher logits must be detached")
    ratio = float(minimum_perplexity_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("Minimum perplexity ratio must lie in (0,1]")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(
        teacher_logits
    ).all():
        raise ValueError("Marginal entropy logits must be finite")

    student_marginal = F.softmax(student_logits, dim=-1).mean(dim=0)
    with torch.no_grad():
        teacher_marginal = F.softmax(teacher_logits, dim=-1).mean(dim=0)
        teacher_entropy = _entropy(teacher_marginal)
        entropy_floor = torch.clamp_min(
            teacher_entropy + math.log(ratio), 0.0
        )
        teacher_effective_perplexity = torch.exp(teacher_entropy)
        teacher_top_probability = torch.max(teacher_marginal)
    student_entropy = _entropy(student_marginal)
    shortfall = F.relu(entropy_floor - student_entropy)
    loss = shortfall.square()
    output = MarginalEntropyFloorOutput(
        loss=loss,
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy.detach(),
        entropy_floor=entropy_floor.detach(),
        student_effective_perplexity=torch.exp(student_entropy),
        teacher_effective_perplexity=teacher_effective_perplexity.detach(),
        student_top_probability=torch.max(student_marginal),
        teacher_top_probability=teacher_top_probability.detach(),
    )
    if any(
        not torch.isfinite(value)
        for value in (
            output.loss,
            output.student_entropy,
            output.teacher_entropy,
            output.entropy_floor,
            output.student_effective_perplexity,
            output.teacher_effective_perplexity,
            output.student_top_probability,
            output.teacher_top_probability,
        )
    ):
        raise RuntimeError("Marginal entropy floor produced a non-finite metric")
    return output


def diversity_preserving_masked_neural_code_objective(
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    patches_volts: torch.Tensor,
    position_ids_by_sample: torch.Tensor,
    bool_masked_pos: torch.Tensor,
) -> DiversityPreservingObjectiveOutput:
    """Official two-pass CE + zero-LoRA T=2 KL + marginal entropy floor."""

    batch = patches_volts.shape[0]
    if tuple(position_ids_by_sample.shape) != (batch, N_STANDARD_CHANNELS):
        raise ValueError("Per-sample LaBraM position IDs must have shape [B,19]")
    if position_ids_by_sample.dtype != torch.long:
        raise TypeError("Per-sample LaBraM position IDs must be torch.long")
    if tuple(bool_masked_pos.shape) != (batch, LABRAM_DAPT_PATCH_TOKENS):
        raise ValueError("LaBraM DAPT-v2 mask must have shape [B,152]")
    if bool_masked_pos.dtype != torch.bool:
        raise TypeError("LaBraM DAPT-v2 mask must be boolean")
    if not torch.all(bool_masked_pos.sum(dim=1) == LABRAM_DAPT_PATCH_TOKENS // 2):
        raise ValueError("Every DAPT-v2 sample must mask exactly 76 of 152 tokens")
    if patches_volts.device != position_ids_by_sample.device or (
        patches_volts.device != bool_masked_pos.device
    ):
        raise ValueError("DAPT-v2 EEG, positions, and masks must share a device")
    model._assert_contract()
    tokenizer._assert_frozen()

    unique_positions, inverse = torch.unique(
        position_ids_by_sample, dim=0, sorted=True, return_inverse=True
    )
    all_codes = torch.empty(
        batch,
        LABRAM_DAPT_PATCH_TOKENS,
        dtype=torch.long,
        device=patches_volts.device,
    )
    cached_groups: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []

    # All teacher forwards happen before any student autograd graph.  This is
    # the only interval in which LoRA-B is mutated.
    with _temporarily_disable_lora_b(model):
        with torch.no_grad():
            for binding_index in range(unique_positions.shape[0]):
                rows = torch.nonzero(
                    inverse == binding_index, as_tuple=False
                ).flatten()
                group_eeg = patches_volts.index_select(0, rows)
                group_mask = bool_masked_pos.index_select(0, rows)
                position_ids = unique_positions[binding_index]
                codes = tokenizer(group_eeg, position_ids)
                all_codes.index_copy_(0, rows, codes)
                teacher_masked, teacher_complementary = model.complementary_logits(
                    group_eeg, position_ids, group_mask
                )
                if teacher_masked.requires_grad or teacher_complementary.requires_grad:
                    raise RuntimeError("Zero-LoRA teacher logits unexpectedly require gradients")
                cached_groups.append(
                    (
                        group_eeg,
                        group_mask,
                        position_ids,
                        codes.detach(),
                        teacher_masked.detach(),
                        teacher_complementary.detach(),
                    )
                )

    masked_student_parts: list[torch.Tensor] = []
    complementary_student_parts: list[torch.Tensor] = []
    masked_teacher_parts: list[torch.Tensor] = []
    complementary_teacher_parts: list[torch.Tensor] = []
    masked_target_parts: list[torch.Tensor] = []
    complementary_target_parts: list[torch.Tensor] = []
    for (
        group_eeg,
        group_mask,
        position_ids,
        codes,
        teacher_masked,
        teacher_complementary,
    ) in cached_groups:
        student_masked, student_complementary = model.complementary_logits(
            group_eeg, position_ids, group_mask
        )
        masked_student_parts.append(student_masked)
        complementary_student_parts.append(student_complementary)
        masked_teacher_parts.append(teacher_masked)
        complementary_teacher_parts.append(teacher_complementary)
        masked_target_parts.append(codes[group_mask])
        complementary_target_parts.append(codes[~group_mask])

    masked_student = torch.cat(masked_student_parts, dim=0)
    complementary_student = torch.cat(complementary_student_parts, dim=0)
    masked_teacher = torch.cat(masked_teacher_parts, dim=0)
    complementary_teacher = torch.cat(complementary_teacher_parts, dim=0)
    masked_target = torch.cat(masked_target_parts, dim=0)
    complementary_target = torch.cat(complementary_target_parts, dim=0)
    expected_partition = batch * LABRAM_DAPT_PATCH_TOKENS // 2
    expected_logits = (expected_partition, LABRAM_DAPT_VOCAB_SIZE)
    if (
        tuple(masked_student.shape) != expected_logits
        or tuple(complementary_student.shape) != expected_logits
        or tuple(masked_teacher.shape) != expected_logits
        or tuple(complementary_teacher.shape) != expected_logits
        or tuple(masked_target.shape) != (expected_partition,)
        or tuple(complementary_target.shape) != (expected_partition,)
    ):
        raise RuntimeError("DAPT-v2 complementary prediction geometry changed")

    masked_ce = F.cross_entropy(masked_student, masked_target, reduction="mean")
    complementary_ce = F.cross_entropy(
        complementary_student, complementary_target, reduction="mean"
    )
    official_ce = masked_ce + complementary_ce
    masked_kl = temperature_scaled_teacher_kl(masked_student, masked_teacher)
    complementary_kl = temperature_scaled_teacher_kl(
        complementary_student, complementary_teacher
    )
    teacher_kl = masked_kl + complementary_kl
    entropy = differentiable_batch_marginal_entropy_floor(
        torch.cat((masked_student, complementary_student), dim=0),
        torch.cat((masked_teacher, complementary_teacher), dim=0),
    )
    total = (
        official_ce
        + LABRAM_DAPT_V2_TEACHER_KL_WEIGHT * teacher_kl
        + LABRAM_DAPT_V2_ENTROPY_FLOOR_WEIGHT * entropy.loss
    )
    if not torch.isfinite(total) or not total.requires_grad:
        raise RuntimeError("DAPT-v2 total objective is non-finite or detached")
    return DiversityPreservingObjectiveOutput(
        loss=total,
        official_ce_loss=official_ce,
        masked_ce_loss=masked_ce,
        complementary_ce_loss=complementary_ce,
        teacher_kl_loss=teacher_kl,
        masked_teacher_kl_loss=masked_kl,
        complementary_teacher_kl_loss=complementary_kl,
        entropy_floor_loss=entropy.loss,
        student_batch_marginal_entropy=entropy.student_entropy,
        teacher_batch_marginal_entropy=entropy.teacher_entropy,
        batch_marginal_entropy_floor=entropy.entropy_floor,
        student_batch_marginal_effective_perplexity=(
            entropy.student_effective_perplexity
        ),
        teacher_batch_marginal_effective_perplexity=(
            entropy.teacher_effective_perplexity
        ),
        student_batch_marginal_top_probability=entropy.student_top_probability,
        teacher_batch_marginal_top_probability=entropy.teacher_top_probability,
        masked_accuracy=(masked_student.argmax(-1) == masked_target).float().mean(),
        complementary_accuracy=(
            complementary_student.argmax(-1) == complementary_target
        )
        .float()
        .mean(),
        neural_codes=all_codes.detach(),
    )


__all__ = [
    "DiversityPreservingObjectiveOutput",
    "LABRAM_DAPT_V2_ENTROPY_FLOOR_WEIGHT",
    "LABRAM_DAPT_V2_LOG_PERPLEXITY_MARGIN",
    "LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO",
    "LABRAM_DAPT_V2_OBJECTIVE",
    "LABRAM_DAPT_V2_TEACHER_KL_WEIGHT",
    "LABRAM_DAPT_V2_TEACHER_TEMPERATURE",
    "MarginalEntropyFloorOutput",
    "differentiable_batch_marginal_entropy_floor",
    "diversity_preserving_masked_neural_code_objective",
    "temperature_scaled_teacher_kl",
]
