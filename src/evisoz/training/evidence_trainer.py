"""Fail-closed Stage-1 evidence trainer wiring.

This module is the first real-data training consumer for the EviSOZ route,
but it is intentionally gated before any model or record is opened.  Once an
authorised Stage-0 gate and a non-blocked pipeline config are supplied, it
uses the real Standard19/CAR + signed TCP22 adapter and the typed field
target converter.  It does not instantiate Qwen or enable the residual.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.evisoz.data.bound_evidence_loader import BoundEvidenceRecord
from src.evisoz.models.clinical_evidence import EviSOZEvidencePipeline
from src.evisoz.models.real_signal_adapter import RealDualMontageTokenAdapter

from .stage0_guard import require_stage0_training_authorized
from .targets import build_typed_loss_targets
from .typed_loss import compute_typed_evidence_losses


def forward_real_evidence_record(
    record: BoundEvidenceRecord,
    *,
    adapter: RealDualMontageTokenAdapter,
    pipeline: EviSOZEvidencePipeline,
) -> tuple[Any, Tensor, Tensor, Tensor, Tensor]:
    """Encode one loader-bound real event and run the evidence decoder only."""

    if not isinstance(record, BoundEvidenceRecord):
        raise TypeError("record must come from bound_evidence_loader")
    if not isinstance(adapter, RealDualMontageTokenAdapter):
        raise TypeError("adapter must be RealDualMontageTokenAdapter")
    if not isinstance(pipeline, EviSOZEvidencePipeline):
        raise TypeError("pipeline must be EviSOZEvidencePipeline")
    inputs = record.checkout_inputs()
    v29 = inputs["v29_reference"]
    tcp22 = inputs["tcp22_context"]
    if v29 is None:
        raise ValueError("Stage-1 evidence training requires a complete v29 reference view")
    parameter = next(adapter.parameters(), None)
    target_device = parameter.device if parameter is not None else torch.device("cpu")
    v29 = v29.to(target_device)
    tcp22 = tcp22.to(target_device)
    node_tokens, node_mask = adapter.encode_node_view(v29.unsqueeze(0))
    edge_tokens, edge_mask = adapter.encode_edge_view(tcp22.unsqueeze(0))
    output, z1, delta, gate = pipeline(
        node_tokens,
        node_mask,
        edge_tokens=edge_tokens,
        edge_mask=edge_mask,
        residual_enabled=False,
        alpha=0.0,
        stage0_status="GO",
    )
    if z1 is not None or delta is not None or gate is not None:
        raise RuntimeError("Stage-1 evidence trainer must not construct residual outputs")
    return output, node_mask, edge_mask, node_tokens, edge_tokens


def run_stage1_evidence_epoch(
    *,
    gate: Mapping[str, object],
    pipeline_config: Mapping[str, object],
    bound_evidence_root: str,
    private_examples_root: str,
    findings_claim_report_root: str,
    private_cohort_root: str,
    split_roster_path: str,
    modeling_path: str,
    checkpoint_path: str,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> dict[str, object]:
    """Run one authorised typed-evidence epoch after the Stage-0 guard.

    Records with no enabled typed loss are skipped.  This is deliberate: a
    future self-supervised/motif objective must be supplied as a separate,
    explicitly audited port instead of silently training on zero gradients.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    authorization = require_stage0_training_authorized(
        gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
    )
    # Imports/record construction are intentionally below the guard.
    from .loader_entrypoint import open_authorized_training_records

    _, records = open_authorized_training_records(
        gate=gate,
        pipeline_config=pipeline_config,
        requested_actions=("query_decoder_or_residual_formal_training",),
        bound_evidence_root=bound_evidence_root,
        private_examples_root=private_examples_root,
        findings_claim_report_root=findings_claim_report_root,
        private_cohort_root=private_cohort_root,
        split_roster_path=split_roster_path,
        evisoz_role="development_cv",
    )
    target_device = torch.device(device)
    adapter = RealDualMontageTokenAdapter(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        projection_mode="learnable",
    ).to(target_device)
    # Stage-1 starts with the official LaBraM representation frozen.  Only
    # the explicit 200→128 projections and evidence decoder are trainable;
    # later unfreezing of the final LaBraM blocks must be a separate audited
    # configuration rather than an optimizer default.
    for parameter in adapter.labram.parameters():
        parameter.requires_grad_(False)
    pipeline = EviSOZEvidencePipeline(128, query_heads=4).to(target_device)
    model = nn.ModuleDict({"adapter": adapter, "pipeline": pipeline})
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("Stage-1 trainer has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    losses: list[float] = []
    skipped = 0
    for record in records:
        enabled = tuple(record.training_example["enabled_loss_ports"])
        if not enabled:
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        output, node_mask, _edge_mask, _node_tokens, _edge_tokens = forward_real_evidence_record(
            record, adapter=adapter, pipeline=pipeline
        )
        targets = build_typed_loss_targets(
            record.field_release,
            observed_node_mask=node_mask[0].any(dim=-1).detach().cpu().tolist(),
        )
        result = compute_typed_evidence_losses(
            output,
            training_example=record.training_example,
            targets={
                key: {
                    name: value.unsqueeze(0).to(output.evidence_tokens.device)
                    for name, value in item.items()
                }
                for key, item in targets.items()
            },
            stage0_gate=gate,
            pipeline_config=pipeline_config,
        )
        loss = result["total"]
        if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("typed evidence loss must be a finite scalar")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError(
            "no enabled typed evidence targets were available; use an audited self-supervised port"
        )
    return {
        "status": "stage1_evidence_epoch_complete",
        "authorization": authorization,
        "event_count": len(losses),
        "skipped_no_loss_port_count": skipped,
        "mean_loss": sum(losses) / len(losses),
        "residual_enabled": False,
        "qwen_generation": False,
    }


__all__ = ["forward_real_evidence_record", "run_stage1_evidence_epoch"]
