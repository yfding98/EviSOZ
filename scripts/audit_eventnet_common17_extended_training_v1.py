#!/usr/bin/env python3
"""Audit an extended EN17 checkpoint against its deterministic initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_cleanroom_registry_v1 import (  # noqa: E402
    EN17_VARIANT_ID,
    build_randomly_initialized_model,
)


_PENDING = "__PENDING__"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_address(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["receipt_sha256"] = _PENDING
    payload["receipt_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def audit_training(checkpoint_path: Path, training_receipt_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    initial_model, initialization = build_randomly_initialized_model(
        variant_id=EN17_VARIANT_ID, outer_fold=0, stage="final_refit"
    )
    initial_state = initial_model.state_dict()
    trained_state = checkpoint["model_state"]
    if tuple(initial_state) != tuple(trained_state):
        raise ValueError("checkpoint state roster differs from deterministic initialization")
    changed_state_names = [
        name
        for name in initial_state
        if not torch.equal(initial_state[name], trained_state[name])
    ]
    parameter_names = tuple(name for name, _ in initial_model.named_parameters())
    changed_parameter_names = [
        name
        for name in parameter_names
        if not torch.equal(initial_state[name], trained_state[name])
    ]
    optimizer_steps = []
    for state in checkpoint["optimizer_state"]["state"].values():
        step = state.get("step")
        if step is not None:
            optimizer_steps.append(int(step.item() if torch.is_tensor(step) else step))
    hyperparameters = checkpoint["hyperparameters"]
    expected_steps = int(hyperparameters["epochs"]) * 584
    status = (
        checkpoint.get("training_complete") is True
        and int(checkpoint["global_step"]) == expected_steps
        and len(changed_parameter_names) == len(parameter_names)
        and optimizer_steps
        and min(optimizer_steps) == max(optimizer_steps) == expected_steps
    )
    return _content_address(
        {
            "schema_version": "eventnet_common17_extended_training_audit_v1",
            "status": "pass" if status else "fail",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_file_sha256": _file_sha256(checkpoint_path),
            "training_receipt_path": str(training_receipt_path.resolve()),
            "training_receipt_file_sha256": _file_sha256(training_receipt_path),
            "training_receipt_internal_sha256": training_receipt["receipt_sha256"],
            "manifest_receipt_sha256": checkpoint["manifest_receipt_sha256"],
            "common17_channel_order": checkpoint["common17_channel_order"],
            "FZ_or_PZ_model_axis_present": checkpoint["FZ_or_PZ_model_axis_present"],
            "hyperparameters": hyperparameters,
            "global_step": int(checkpoint["global_step"]),
            "expected_global_step": expected_steps,
            "training_complete": checkpoint["training_complete"],
            "initialization_receipt": initialization,
            "state_tensor_count": len(initial_state),
            "changed_state_tensor_count": len(changed_state_names),
            "parameter_tensor_count": len(parameter_names),
            "changed_parameter_tensor_count": len(changed_parameter_names),
            "optimizer_parameter_step_minimum": min(optimizer_steps),
            "optimizer_parameter_step_maximum": max(optimizer_steps),
            "final_epoch_mean_loss": float(training_receipt["history"][-1]["mean_loss"]),
            "scientific_scope": {
                "exploratory_training_not_frozen_formal_protocol": True,
                "detector_performance_established_here": False,
                "source_dev_or_source_eval_accessed_here": False,
            },
            "receipt_sha256": _PENDING,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_training(
        args.checkpoint.resolve(strict=True),
        args.training_receipt.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
