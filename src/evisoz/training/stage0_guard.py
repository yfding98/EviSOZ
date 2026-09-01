"""Aggregate Stage-0 authorization guard for all future training consumers.

The guard is intentionally independent of a trainer implementation.  Any
formal Query Decoder, residual, teacher, Qwen, or private-label consumer must
call it before constructing a trainable data loader.  A checked-in gate with
``NO_GO`` (including ``PARTIAL`` or evaluator-only checks) fails closed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.evisoz.data.stage0_gate import validate_stage0_gate
from src.evisoz.models.clinical_evidence import (
    validate_structured_evidence_pipeline_config,
)


class Stage0TrainingBlocked(PermissionError):
    """Raised when the aggregate Stage-0 gate does not authorize training."""


def require_stage0_training_authorized(
    gate: Mapping[str, object],
    *,
    pipeline_config: Mapping[str, object] | None = None,
    requested_actions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a small authorization receipt or fail closed.

    ``requested_actions`` is optional so callers can state exactly which
    training operation they intend to launch.  When Stage 0 is not GO, the
    exception includes both the blocking check IDs and prohibited operations;
    no model or data loader is opened by this function.
    """

    validated_gate = validate_stage0_gate(dict(gate))
    if pipeline_config is not None:
        validated_config = validate_structured_evidence_pipeline_config(
            dict(pipeline_config)
        )
    else:
        validated_config = None
    blockers = list(validated_gate["blocking_check_ids"])
    prohibited = set(validated_gate["prohibited_actions"])
    conflicts = sorted(prohibited.intersection(requested_actions))
    if validated_gate["status"] != "GO" or blockers or conflicts:
        detail = {
            "gate_id": validated_gate["gate_id"],
            "status": validated_gate["status"],
            "blocking_check_ids": blockers,
            "prohibited_actions": sorted(prohibited),
            "requested_action_conflicts": conflicts,
        }
        raise Stage0TrainingBlocked(
            "EviSOZ formal training is blocked by Stage-0: " + repr(detail)
        )
    if validated_config is not None and validated_config["status"] == (
        "implementation_only_stage0_training_blocked"
    ):
        raise Stage0TrainingBlocked(
            "pipeline config remains implementation_only_stage0_training_blocked"
        )
    return {
        "stage0_gate_id": validated_gate["gate_id"],
        "stage0_gate_receipt_sha256": validated_gate["receipt_sha256"],
        "authorized_actions": list(requested_actions),
        "pipeline_config": deepcopy(validated_config)
        if validated_config is not None
        else None,
    }


__all__ = ["Stage0TrainingBlocked", "require_stage0_training_authorized"]
