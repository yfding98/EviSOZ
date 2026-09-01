"""Content-addressed receipts for guarded training entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.evisoz.data.artifact_ref import canonical_json_sha256


STAGE1_TRAINING_BLOCK_RECEIPT_SCHEMA_VERSION = "evisoz_stage1_training_block_receipt_v1"
_HASH_PLACEHOLDER = "0" * 64


def validate_stage1_training_block_receipt(value: object) -> dict[str, Any]:
    required = {
        "schema_version", "status", "stage0_gate_id", "stage0_status",
        "blocking_check_ids", "error", "runtime", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("Stage-1 block receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != STAGE1_TRAINING_BLOCK_RECEIPT_SCHEMA_VERSION:
        raise ValueError("Stage-1 block receipt schema drifted")
    if data["status"] != "blocked_before_model_or_loader_construction":
        raise ValueError("Stage-1 block receipt status drifted")
    if not isinstance(data["stage0_gate_id"], str) or not data["stage0_gate_id"]:
        raise ValueError("Stage-1 block receipt gate ID is invalid")
    if data["stage0_status"] != "NO_GO":
        raise ValueError("Stage-1 block receipt must carry Stage-0 NO_GO")
    if not isinstance(data["blocking_check_ids"], list) or data["blocking_check_ids"] != sorted(data["blocking_check_ids"]):
        raise ValueError("Stage-1 block receipt blockers are invalid")
    if not isinstance(data["error"], str) or not data["error"]:
        raise ValueError("Stage-1 block receipt error is empty")
    runtime = data["runtime"]
    expected_runtime = {
        "model_constructed": False,
        "optimizer_constructed": False,
        "training_loader_opened": False,
        "teacher_runtime_opened": False,
        "qwen_generation": False,
        "residual_enabled": False,
    }
    if runtime != expected_runtime:
        raise ValueError("Stage-1 block receipt runtime policy drifted")
    if not isinstance(data["receipt_sha256"], str) or len(data["receipt_sha256"]) != 64:
        raise ValueError("Stage-1 block receipt hash is invalid")
    body = deepcopy(data)
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    if data["receipt_sha256"] != canonical_json_sha256(body):
        raise ValueError("Stage-1 block receipt hash drifted")
    return data


__all__ = [
    "STAGE1_TRAINING_BLOCK_RECEIPT_SCHEMA_VERSION",
    "validate_stage1_training_block_receipt",
]
