"""Fail-closed inference bridge from bound evidence to candidate report plans.

The bridge is deliberately a *shadow* path.  It consumes only a validated
``BoundEvidenceRecord`` and caller-supplied frozen model functions; it never
reads physician DOCX, knowledge files, labels, or teacher runtimes.  The
returned packet and plan remain candidate-only until a separate authorized
release changes that policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.evisoz.baseline.frozen_v29 import V29_CANDIDATE_MASK
from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord
from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline
from src.evisoz.models.predicted_evidence import build_predicted_evidence_packet
from src.evisoz.reporting.bound_shadow_report import build_bound_shadow_report_plan
from src.evisoz.reporting.qwen_structured_input import build_qwen_structured_input


TokenEncoder = Callable[[Tensor], Any]
BaselineLogits = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ShadowInferenceResult:
    """Candidate-only outputs tied to one loader-replayed event."""

    event_id: str
    predicted_evidence: Mapping[str, Any]
    report_plan: Mapping[str, Any]
    qwen_structured_input: Mapping[str, Any] | None
    baseline_logits: Tensor
    residual_delta: Tensor
    residual_gate: Tensor


def _coerce_tokens(
    encoded: Any,
    *,
    raw: Tensor,
    unit_mask: Sequence[bool],
    units: int,
    token_dim: int,
    name: str,
) -> tuple[Tensor, Tensor]:
    """Normalize an injected LaBraM encoder result to ``[1,U,T,D]``."""

    returned_mask: Tensor | None = None
    if isinstance(encoded, (tuple, list)):
        if len(encoded) != 2:
            raise ValueError(f"{name} encoder tuple must contain tokens and mask")
        encoded, returned_mask = encoded
    if not isinstance(encoded, Tensor):
        raise ValueError(f"{name} encoder must return a tensor")
    if encoded.ndim == 3:
        encoded = encoded.unsqueeze(0)
    if encoded.ndim != 4 or tuple(encoded.shape[:2]) != (1, units):
        raise ValueError(f"{name} tokens must have shape [1,{units},T,D]")
    if encoded.shape[-1] != token_dim or not encoded.is_floating_point():
        raise ValueError(f"{name} token dimension/dtype drifted")
    if not torch.isfinite(encoded).all():
        raise ValueError(f"{name} tokens must be finite")
    token_count = int(encoded.shape[2])
    if token_count < 1:
        raise ValueError(f"{name} token sequence is empty")
    if returned_mask is None:
        base = torch.tensor(list(unit_mask), dtype=torch.bool, device=encoded.device)
        returned_mask = base.view(1, units, 1).expand(1, units, token_count).clone()
    else:
        if returned_mask.ndim == 2:
            returned_mask = returned_mask.unsqueeze(0)
        if returned_mask.ndim != 3 or tuple(returned_mask.shape) != (1, units, token_count):
            raise ValueError(f"{name} token mask shape drifted")
        if returned_mask.dtype is not torch.bool:
            raise ValueError(f"{name} token mask must be boolean")
        returned_mask = returned_mask.to(device=encoded.device)
    # The loader mask is authoritative.  An encoder may further invalidate a
    # token, but it may never make an unobserved channel appear observed.
    observed = torch.tensor(list(unit_mask), dtype=torch.bool, device=encoded.device)
    returned_mask = returned_mask & observed.view(1, units, 1)
    return encoded, returned_mask


def _baseline_batch(
    baseline_inference: BaselineLogits,
    waveform: Tensor,
    observed_mask: Tensor,
) -> Tensor:
    logits = baseline_inference(waveform.clone(), observed_mask.clone())
    if not isinstance(logits, Tensor):
        raise ValueError("baseline inference must return a tensor")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if tuple(logits.shape) != (1, 19) or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("baseline logits must have shape [1,19] and be finite")
    return logits


def run_bound_evidence_shadow_inference(
    record: BoundEvidenceRecord,
    *,
    node_encoder: TokenEncoder,
    edge_encoder: TokenEncoder,
    baseline_inference: BaselineLogits,
    evidence_pipeline: EviSOZEvidencePipeline,
    node_units: Sequence[str],
    edge_units: Sequence[str],
    knowledge_card_ids: Sequence[str] | None = None,
    stage0_status: str = "NO_GO",
) -> ShadowInferenceResult:
    """Run candidate inference on one loader-replayed event.

    ``stage0_status`` is intentionally defaulted to ``NO_GO``.  This function
    always disables residual correction; a future training/inference route
    must explicitly implement and audit a separate authorized path before it
    can alter frozen v29 logits.
    """

    if not isinstance(record, BoundEvidenceRecord):
        raise TypeError("record must come from bound_evidence_loader")
    if stage0_status not in {"GO", "NO_GO"}:
        raise ValueError("stage0_status must be GO or NO_GO")
    if len(node_units) != 19 or len(edge_units) != 22:
        raise ValueError("node/edge unit rosters must be Standard19/TCP22")
    if not isinstance(evidence_pipeline, nn.Module):
        raise TypeError("evidence_pipeline must be a torch module")

    inputs = record.checkout_inputs()
    v29 = inputs["v29_reference"]
    if v29 is None:
        raise ValueError("bound event has no complete v29 reference view")
    tcp22 = inputs["tcp22_context"]
    node_observed = torch.tensor(inputs["standard19_observed_mask"], dtype=torch.bool)
    edge_observed = torch.tensor(inputs["tcp22_observed_mask"], dtype=torch.bool)
    token_dim = int(evidence_pipeline.motif_adapter.token_dim)
    node_tokens, node_mask = _coerce_tokens(
        node_encoder(v29.clone()),
        raw=v29,
        unit_mask=inputs["standard19_observed_mask"],
        units=19,
        token_dim=token_dim,
        name="v29",
    )
    edge_tokens, edge_mask = _coerce_tokens(
        edge_encoder(tcp22.clone()),
        raw=tcp22,
        unit_mask=inputs["tcp22_observed_mask"],
        units=22,
        token_dim=token_dim,
        name="tcp22",
    )
    if node_tokens.shape[2] != edge_tokens.shape[2]:
        raise ValueError("v29 and TCP22 encoders must expose the same token count")
    z0 = _baseline_batch(baseline_inference, v29, node_observed)
    candidate = torch.tensor(V29_CANDIDATE_MASK, dtype=torch.bool).view(1, 19)
    candidate = candidate & node_observed.view(1, 19)
    # Shadow inference is evidence/report generation only.  The residual head
    # is called in its exact identity mode, so NO_GO can never be bypassed.
    output, z1, delta, gate = evidence_pipeline(
        node_tokens,
        node_mask,
        edge_tokens=edge_tokens,
        edge_mask=edge_mask,
        z0_node=z0,
        candidate_mask=candidate,
        residual_mode_eligible=torch.zeros(1, dtype=torch.bool),
        residual_enabled=False,
        alpha=0.0,
        stage0_status=stage0_status,
    )
    if z1 is None or delta is None or gate is None or not torch.equal(z1, z0):
        raise RuntimeError("shadow inference residual path was not an identity")
    packet = build_predicted_evidence_packet(
        event_id=record.event_id,
        output=output,
        node_mask=node_mask,
        edge_mask=edge_mask,
        candidate_node_mask=candidate[0],
        node_units=node_units,
        edge_units=edge_units,
        stage0_status=stage0_status,
    )
    plan = build_bound_shadow_report_plan(
        record,
        packet,
        knowledge_card_ids=knowledge_card_ids,
    )
    qwen_input = None
    if record.knowledge_selection is not None:
        # This is still a no-generation packet.  It proves that the loader's
        # verified knowledge/eeg selection is the only route into a future
        # Qwen adapter; it does not open card text or invoke a language model.
        qwen_input = build_qwen_structured_input(
            report_plan=plan,
            knowledge_selection=record.knowledge_selection,
        )
    return ShadowInferenceResult(
        event_id=record.event_id,
        predicted_evidence=packet,
        report_plan=plan,
        qwen_structured_input=qwen_input,
        baseline_logits=z0.detach().clone(),
        residual_delta=delta.detach().clone(),
        residual_gate=gate.detach().clone(),
    )


__all__ = [
    "ShadowInferenceResult",
    "run_bound_evidence_shadow_inference",
]
