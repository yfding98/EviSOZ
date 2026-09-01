"""Fail-closed typed loss ports for the EviSOZ evidence decoder.

This module is intentionally a small training boundary, not a trainer.  It
validates the event envelope's enabled loss ports and asks the aggregate
Stage-0 guard for authorization *before* evaluating any enabled objective.
With the current real gate all formal ports therefore fail closed.  The
report-text port is named here only so an EEG trainer cannot accidentally
consume it; text loss belongs to a separate Qwen-side trainer.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from src.evisoz.data.dataset_policy import LOSS_PORTS
from src.evisoz.models.clinical_evidence import ClinicalEvidenceOutput

from .stage0_guard import require_stage0_training_authorized


class TypedLossContractError(PermissionError):
    """Raised when an objective is not authorized by the typed contract."""


_SLOT_LOGITS = {
    "quality": "quality_logits",
    "morphology": "morphology_logits",
    "onset": "onset_logits",
    "spread": "spread_logits",
    "evolution": "evolution_logits",
    "localizability": "localizability_logits",
}
_SLOT_TARGET_KEYS = frozenset(_SLOT_LOGITS)


def _masked_mean(loss: Tensor, mask: Tensor, *, name: str) -> Tensor:
    if mask.dtype is not torch.bool or tuple(mask.shape) != tuple(loss.shape):
        raise ValueError(f"{name} mask must match its loss shape and be boolean")
    count = mask.sum()
    if int(count.item()) == 0:
        raise ValueError(f"{name} has no evaluable targets")
    return loss.masked_select(mask).mean()


def _validate_ports(training_example: Mapping[str, object]) -> tuple[str, ...]:
    ports = training_example.get("enabled_loss_ports")
    if not isinstance(ports, list) or ports != sorted(set(ports)):
        raise ValueError("training example enabled_loss_ports must be sorted/unique")
    if any(port not in LOSS_PORTS for port in ports):
        raise ValueError("training example contains an unknown loss port")
    role = training_example.get("split_assignment", {}).get("evisoz_role")
    if role != "development_cv" and ports:
        raise ValueError("locked or external examples cannot enable loss ports")
    return tuple(str(port) for port in ports)


def _validate_targets(targets: Mapping[str, object]) -> None:
    allowed = _SLOT_TARGET_KEYS | {"node_localization"}
    unknown = set(targets).difference(allowed)
    if unknown:
        raise ValueError(f"typed loss targets contain unknown keys: {sorted(unknown)}")


def _slot_loss(output: ClinicalEvidenceOutput, targets: Mapping[str, object]) -> Tensor:
    values: list[Tensor] = []
    for name, logits_name in _SLOT_LOGITS.items():
        if name not in targets:
            continue
        item = targets[name]
        if type(item) is not dict or set(item) != {"target", "mask"}:
            raise ValueError(f"typed slot target {name} must contain target and mask")
        target = item["target"]
        mask = item["mask"]
        logits = getattr(output, logits_name)
        if not isinstance(target, Tensor) or target.dtype != torch.long:
            raise ValueError(f"typed slot target {name} must be int64")
        if tuple(target.shape) != tuple(logits.shape[:1]):
            raise ValueError(f"typed slot target {name} must have shape [B]")
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(target.shape):
            raise ValueError(f"typed slot mask {name} must have shape [B] and be boolean")
        if mask.any() and (
            int(target[mask].min().item()) < 0
            or int(target[mask].max().item()) >= int(logits.shape[-1])
        ):
            raise ValueError(f"typed slot target {name} has an out-of-range class")
        losses = F.cross_entropy(logits, target, reduction="none")
        values.append(_masked_mean(losses, mask, name=f"typed slot {name}"))
    if not values:
        raise ValueError("typed_slot_loss has no supplied slot targets")
    return torch.stack(values).mean()


def _node_localization_loss(output: ClinicalEvidenceOutput, item: object) -> Tensor:
    if type(item) is not dict or set(item) != {"target", "mask"}:
        raise ValueError("node_localization target must contain target and mask")
    target = item["target"]
    mask = item["mask"]
    if not isinstance(target, Tensor) or not target.is_floating_point() or tuple(target.shape) != tuple(output.onset_logits.shape):
        raise ValueError("node_localization target must have shape [B,19]")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(target.shape):
        raise ValueError("node_localization mask must have shape [B,19] and be boolean")
    if not torch.isfinite(target).all() or bool((target < 0).any()) or bool((target > 1).any()):
        raise ValueError("node_localization target must be finite in [0,1]")
    losses = F.binary_cross_entropy_with_logits(
        output.onset_logits, target, reduction="none"
    )
    return _masked_mean(losses, mask, name="node_localization")


def compute_typed_evidence_losses(
    output: ClinicalEvidenceOutput,
    *,
    training_example: Mapping[str, object],
    targets: Mapping[str, object],
    stage0_gate: Mapping[str, object],
    pipeline_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Compute only explicitly enabled, Stage-0-authorized EEG loss ports.

    ``training_example`` is expected to be the already fully validated
    envelope returned by the bound loader.  The function repeats the small
    permission checks needed at this boundary and never treats generated text,
    knowledge rules, or edge endpoints as node labels.
    """

    if not isinstance(output, ClinicalEvidenceOutput):
        raise TypeError("output must be ClinicalEvidenceOutput")
    if not isinstance(training_example, Mapping):
        raise TypeError("training_example must be a mapping")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping")
    ports = _validate_ports(training_example)
    _validate_targets(targets)
    if not ports:
        zero = output.evidence_tokens.sum() * 0.0
        return {"total": zero, "by_port": {}, "authorization": None}

    # The guard is intentionally before objective construction.  Under the
    # current r29 NO_GO gate this raises and no enabled loss is evaluated.
    try:
        authorization = require_stage0_training_authorized(
            stage0_gate,
            pipeline_config=pipeline_config,
            requested_actions=ports,
        )
    except PermissionError as exc:
        raise TypedLossContractError(str(exc)) from exc

    by_port: dict[str, Tensor] = {}
    if "typed_slot_loss" in ports:
        by_port["typed_slot_loss"] = _slot_loss(output, targets)
    if "node_localization_loss" in ports:
        if "node_localization" not in targets:
            raise ValueError("node_localization_loss requires node_localization targets")
        by_port["node_localization_loss"] = _node_localization_loss(
            output, targets["node_localization"]
        )
    if "report_text_loss" in ports:
        raise TypedLossContractError(
            "report_text_loss is Qwen-side and cannot be consumed by the EEG loss contract"
        )
    total = torch.stack(list(by_port.values())).mean()
    if not torch.isfinite(total):
        raise ValueError("typed evidence loss must be finite")
    return {"total": total, "by_port": by_port, "authorization": authorization}


__all__ = [
    "TypedLossContractError",
    "compute_typed_evidence_losses",
]
