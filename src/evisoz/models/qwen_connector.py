"""Numerical EEG-to-Qwen connector and text-alignment utilities.

This module contains no Qwen dependency and never performs text generation.
It only turns already validated evidence tokens into the Qwen hidden size and
provides masked, multi-positive alignment primitives.  A trainer must still
pass the aggregate Stage-0 guard before these parameters can be optimized.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


QWEN3_8_27B_HIDDEN_SIZE = 5120
DEFAULT_EVIDENCE_TOKEN_COUNT = 32


def _check_tokens(tokens: Tensor, *, name: str, rank: int = 3) -> None:
    if not isinstance(tokens, Tensor) or tokens.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} tensor")
    if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError(f"{name} must be finite floating point")


def _check_mask(mask: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")


def _check_embedding_sequence(
    embeddings: Tensor,
    *,
    name: str,
    hidden_size: int,
) -> tuple[int, int, int]:
    _check_tokens(embeddings, name=name)
    batch, length, width = embeddings.shape
    if width != hidden_size:
        raise ValueError(f"{name} hidden size does not match Qwen")
    return int(batch), int(length), int(width)


class EvidenceTokenResampler(nn.Module):
    """Resample variable-length evidence into fixed Qwen-compatible tokens."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int = QWEN3_8_27B_HIDDEN_SIZE,
        token_count: int = DEFAULT_EVIDENCE_TOKEN_COUNT,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if input_dim < 4 or output_dim < 4 or token_count < 1:
            raise ValueError("input_dim, output_dim and token_count must be positive")
        if input_dim % heads:
            raise ValueError("input_dim must be divisible by attention heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_count = token_count
        self.query = nn.Parameter(torch.empty(token_count, input_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(input_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.output_projection = nn.Linear(input_dim, output_dim)

    def forward(self, evidence_tokens: Tensor, token_mask: Tensor) -> Tensor:
        _check_tokens(evidence_tokens, name="evidence_tokens")
        b, n, d = evidence_tokens.shape
        if d != self.input_dim:
            raise ValueError("evidence token dimension does not match resampler")
        _check_mask(token_mask, name="token_mask", shape=(b, n))
        source = evidence_tokens
        valid = token_mask
        empty = ~valid.any(dim=1)
        if empty.any():
            # A zero sentinel avoids all-key-padding NaNs without representing
            # a clinical fact.  The caller still receives the same fixed shape.
            source = source.clone()
            valid = valid.clone()
            source[empty, 0] = 0
            valid[empty, 0] = True
        queries = self.query.unsqueeze(0).expand(b, -1, -1)
        attended, _ = self.attention(
            queries,
            source,
            source,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return self.output_projection(self.norm(queries + attended))


def assemble_qwen_embedding_inputs(
    text_embeddings: Tensor,
    eeg_embeddings: Tensor,
    text_attention_mask: Tensor,
    *,
    eeg_attention_mask: Optional[Tensor] = None,
    insertion_index: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Insert fixed EEG evidence embeddings into a text embedding sequence.

    The returned tuple is ``(inputs_embeds, attention_mask, eeg_modality_mask)``.
    ``eeg_modality_mask`` identifies the inserted slots independently of the
    attention mask, which lets an adapter audit exactly where evidence tokens
    entered the Qwen sequence.  This is a tensor-only operation: it does not
    tokenize text, invoke Qwen, or authorize generation.
    """

    batch, text_length, hidden = _check_embedding_sequence(
        text_embeddings,
        name="text_embeddings",
        hidden_size=QWEN3_8_27B_HIDDEN_SIZE,
    )
    eeg_batch, eeg_length, _ = _check_embedding_sequence(
        eeg_embeddings,
        name="eeg_embeddings",
        hidden_size=QWEN3_8_27B_HIDDEN_SIZE,
    )
    if eeg_batch != batch or eeg_length < 1:
        raise ValueError("EEG embeddings must match text batch and be non-empty")
    _check_mask(
        text_attention_mask,
        name="text_attention_mask",
        shape=(batch, text_length),
    )
    if eeg_attention_mask is None:
        eeg_attention_mask = torch.ones(
            batch,
            eeg_length,
            dtype=torch.bool,
            device=eeg_embeddings.device,
        )
    else:
        _check_mask(
            eeg_attention_mask,
            name="eeg_attention_mask",
            shape=(batch, eeg_length),
        )
    if not isinstance(insertion_index, int) or not 0 <= insertion_index <= text_length:
        raise ValueError("insertion_index must be within the text sequence")
    prefix = text_embeddings[:, :insertion_index]
    suffix = text_embeddings[:, insertion_index:]
    inputs_embeds = torch.cat((prefix, eeg_embeddings, suffix), dim=1)
    attention_mask = torch.cat(
        (
            text_attention_mask[:, :insertion_index],
            eeg_attention_mask,
            text_attention_mask[:, insertion_index:],
        ),
        dim=1,
    )
    eeg_modality_mask = torch.cat(
        (
            torch.zeros(
                batch,
                insertion_index,
                dtype=torch.bool,
                device=text_embeddings.device,
            ),
            torch.ones(
                batch,
                eeg_length,
                dtype=torch.bool,
                device=text_embeddings.device,
            ),
            torch.zeros(
                batch,
                text_length - insertion_index,
                dtype=torch.bool,
                device=text_embeddings.device,
            ),
        ),
        dim=1,
    )
    if inputs_embeds.shape != (batch, text_length + eeg_length, hidden):
        raise RuntimeError("Qwen embedding concatenation shape drifted")
    return inputs_embeds, attention_mask, eeg_modality_mask


def clause_mil_alignment_loss(
    local_tokens: Tensor,
    clause_embeddings: Tensor,
    positive_mask: Tensor,
    *,
    token_mask: Optional[Tensor] = None,
    clause_weights: Optional[Tensor] = None,
    temperature: float = 0.07,
) -> Tensor:
    """Compute a multi-positive MIL loss for coarse clause-to-EEG alignment.

    ``positive_mask[b, clause, token]`` identifies an allowed *set* of local
    instances.  Clauses with no valid positive instance are excluded rather
    than turned into false negatives.  No text embedding is used to create a
    node/SOZ label.
    """

    _check_tokens(local_tokens, name="local_tokens")
    _check_tokens(clause_embeddings, name="clause_embeddings")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    b, n, d = local_tokens.shape
    if tuple(clause_embeddings.shape[:1]) != (b,) or clause_embeddings.shape[-1] != d:
        raise ValueError("clause embedding shape does not match local tokens")
    clauses = clause_embeddings.shape[1]
    _check_mask(positive_mask, name="positive_mask", shape=(b, clauses, n))
    if token_mask is None:
        token_mask = torch.ones(b, n, dtype=torch.bool, device=local_tokens.device)
    _check_mask(token_mask, name="token_mask", shape=(b, n))
    allowed = positive_mask & token_mask.unsqueeze(1)
    valid_clauses = allowed.any(dim=-1)
    if not valid_clauses.any():
        return local_tokens.sum() * 0.0
    local = F.normalize(local_tokens, dim=-1)
    clauses_norm = F.normalize(clause_embeddings, dim=-1)
    logits = torch.einsum("bnd,bcd->bcn", local, clauses_norm) / temperature
    logits = logits.masked_fill(~token_mask.unsqueeze(1), float("-inf"))
    positive_logits = logits.masked_fill(~allowed, float("-inf"))
    per_clause = torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(logits, dim=-1)
    per_clause = per_clause[valid_clauses]
    if clause_weights is not None:
        if tuple(clause_weights.shape) != (b, clauses) or not torch.isfinite(clause_weights).all():
            raise ValueError("clause_weights must have shape [B,C] and be finite")
        weights = clause_weights[valid_clauses].clamp_min(0)
        if weights.sum() <= 0:
            return local_tokens.sum() * 0.0
        return -(per_clause * weights).sum() / weights.sum()
    return -per_clause.mean()


def evidence_guided_mask(
    tokens: Tensor,
    evidence_mask: Tensor,
    *,
    mask_probability: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Mask only evidence-designated token cells for MedIM-style pretraining."""

    _check_tokens(tokens, name="tokens")
    b, n, _ = tokens.shape
    _check_mask(evidence_mask, name="evidence_mask", shape=(b, n))
    if not 0 <= mask_probability <= 1:
        raise ValueError("mask_probability must be in [0,1]")
    if mask_probability == 0:
        selected = torch.zeros_like(evidence_mask)
    elif mask_probability == 1:
        selected = evidence_mask.clone()
    else:
        random = torch.rand(
            evidence_mask.shape,
            device=tokens.device,
            generator=generator,
        )
        selected = evidence_mask & (random < mask_probability)
    masked = tokens.masked_fill(selected.unsqueeze(-1), 0)
    return masked, selected


__all__ = [
    "assemble_qwen_embedding_inputs",
    "DEFAULT_EVIDENCE_TOKEN_COUNT",
    "EvidenceTokenResampler",
    "QWEN3_8_27B_HIDDEN_SIZE",
    "clause_mil_alignment_loss",
    "evidence_guided_mask",
]
