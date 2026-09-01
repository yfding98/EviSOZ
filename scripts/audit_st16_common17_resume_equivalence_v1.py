#!/usr/bin/env python3
"""Audit tensor-exact ST16 uninterrupted versus batch-boundary resume state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _tree_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor\0")
            digest.update(str(item.dtype).encode())
            digest.update(json.dumps(list(item.shape)).encode())
            digest.update(_tensor_bytes(item))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                update(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode())
        else:
            raise TypeError(f"unsupported checkpoint tree value: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("checkpoint must be a dictionary")
    return value


def audit(resumed_path: Path, uninterrupted_path: Path) -> dict[str, Any]:
    resumed = _load(resumed_path)
    uninterrupted = _load(uninterrupted_path)
    control_fields = (
        "schema_version",
        "provider_id",
        "variant_id",
        "next_epoch",
        "next_batch",
        "global_step",
        "completed_epoch_count",
        "training_complete",
        "inference_eligible",
    )
    control_equal = all(
        resumed.get(field) == uninterrupted.get(field) for field in control_fields
    )
    # Checkpoint cadence controls persistence only; it does not alter model,
    # optimiser, sampling, precision, or random-number evolution.  The resume
    # probe deliberately persisted every batch while the uninterrupted control
    # persisted after its second batch, so compare the computational training
    # configuration after removing this expected operational difference.
    resumed_training_config = dict(resumed.get("training_config") or {})
    uninterrupted_training_config = dict(uninterrupted.get("training_config") or {})
    resumed_checkpoint_cadence = resumed_training_config.pop(
        "checkpoint_every_batches", None
    )
    uninterrupted_checkpoint_cadence = uninterrupted_training_config.pop(
        "checkpoint_every_batches", None
    )
    training_config_equal = resumed_training_config == uninterrupted_training_config
    control_equal = control_equal and training_config_equal
    state_digests = {
        key: {
            "resumed": _tree_digest(resumed[key]),
            "uninterrupted": _tree_digest(uninterrupted[key]),
        }
        for key in (
            "model_state",
            "optimizer_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state_all",
        )
    }
    state_equal = {
        key: row["resumed"] == row["uninterrupted"]
        for key, row in state_digests.items()
    }
    passed = control_equal and all(state_equal.values())
    return {
        "schema_version": "st16_common17_resume_equivalence_audit_v1",
        "status": "pass_tensor_exact" if passed else "fail_state_drift",
        "resumed_checkpoint": str(resumed_path.resolve()),
        "uninterrupted_checkpoint": str(uninterrupted_path.resolve()),
        "control_fields_equal": control_equal,
        "computational_training_config_equal": training_config_equal,
        "checkpoint_cadence": {
            "resumed": resumed_checkpoint_cadence,
            "uninterrupted": uninterrupted_checkpoint_cadence,
            "treated_as_operational_metadata": True,
        },
        "state_equal": state_equal,
        "state_digests": state_digests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.resumed, args.uninterrupted)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass_tensor_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
