#!/usr/bin/env python3
"""Audit a third-party conversion of the SeizureTransformer Docker checkpoint.

The artifact is useful only as a reproducibility and warm-start sensitivity
reference.  Its distributor is not the original paper author, its exact TUSZ
training exposure is unknown, and it expects the full 19-channel referential
axis.  This audit therefore proves tensor/architecture compatibility and the
mechanical feasibility of a *signal-side* common17 first-layer projection; it
does not admit the checkpoint as a clean benchmark provider and performs no
EEG inference or metric evaluation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/external_seizuretransformer_checkpoint_audit_v1_20260825/source"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/external_seizuretransformer_checkpoint_audit_v1_20260825/receipt.json"
)
EXPECTED_MODEL_SHA256 = (
    "2cdc841001a0fbcdf1dfcbb02b3a26fa7af14002e01ebf9815fa09c82be06f61"
)
EXPECTED_HF_COMMIT = "92c2bffa632d967868a820ba3153f2828d72b496"
STANDARD19 = (
    "FP1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "FZ", "CZ",
    "PZ", "FP2", "F4", "C4", "P4", "O2", "F8", "T8", "P8",
)
COMMON17 = tuple(channel for channel in STANDARD19 if channel not in {"FZ", "PZ"})
COMMON17_INDICES = tuple(
    index for index, channel in enumerate(STANDARD19) if channel in COMMON17
)
PENDING = "CONTENT-ADDRESS-PENDING"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _atomic_json(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_official_model_class() -> type[torch.nn.Module]:
    from third_party.SeizureTransformer.time_step_level.model import (  # noqa: PLC0415
        SeizureTransformer,
    )

    return SeizureTransformer


def _project_state_to_common17(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tuple(state["encoder.convs.0.weight"].shape) != (32, 19, 11):
        raise ValueError("external checkpoint first convolution is not 19-channel")
    projected = {key: tensor for key, tensor in state.items()}
    projected["encoder.convs.0.weight"] = state[
        "encoder.convs.0.weight"
    ][:, COMMON17_INDICES, :].contiguous()
    if tuple(projected["encoder.convs.0.weight"].shape) != (32, 17, 11):
        raise RuntimeError("common17 projection emitted the wrong first-layer shape")
    return projected


def audit(source: Path) -> dict[str, Any]:
    source_root = source.resolve(strict=True)
    paths = {
        name: (source_root / name).resolve(strict=True)
        for name in ("model.safetensors", "config.json", "README.md", "LICENSE")
    }
    model_sha = _file_sha256(paths["model.safetensors"])
    if model_sha != EXPECTED_MODEL_SHA256:
        raise ValueError("external checkpoint differs from the frozen HF artifact")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "SeizureTransformer",
        "task": "eeg-seizure-detection",
        "in_channels": 19,
        "in_samples": 15360,
        "dim_feedforward": 2048,
        "num_layers": 8,
        "num_heads": 4,
        "drop_rate": 0.1,
        "max_pos_len": 6000,
        "output": "time_step_probability",
        "output_shape": ["batch", "time"],
    }
    if config != expected_config:
        raise ValueError("external checkpoint config drifted")
    readme = paths["README.md"].read_text(encoding="utf-8")
    license_text = paths["LICENSE"].read_text(encoding="utf-8")
    required_provenance = (
        "Source project: `keruiwu/SeizureTransformer`",
        "Public container source used to obtain checkpoint: `yujjio/seizure_transformer`",
        "Extracted checkpoint path in container: `wu_2025/model.pth`",
    )
    if not all(value in readme for value in required_provenance):
        raise ValueError("third-party provenance statement is incomplete")
    if "RESEARCH-ONLY" not in license_text or "No commercial use" not in license_text:
        raise ValueError("third-party distribution restrictions drifted")

    state = load_file(str(paths["model.safetensors"]), device="cpu")
    model_class = _load_official_model_class()
    model19 = model_class(
        in_channels=19,
        in_samples=15360,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    model19.load_state_dict(state, strict=True)
    projected = _project_state_to_common17(state)
    model17 = model_class(
        in_channels=17,
        in_samples=15360,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    model17.load_state_dict(projected, strict=True)
    different_shapes = [
        key for key in state if tuple(state[key].shape) != tuple(projected[key].shape)
    ]
    if different_shapes != ["encoder.convs.0.weight"]:
        raise RuntimeError("common17 projection changed an unexpected state tensor")
    removed_values = state["encoder.convs.0.weight"].numel() - projected[
        "encoder.convs.0.weight"
    ].numel()
    if removed_values != 32 * 2 * 11:
        raise RuntimeError("common17 projection removed the wrong number of weights")

    return _content_address(
        {
            "schema_version": "external_seizuretransformer_checkpoint_audit_v1",
            "status": "pass_structure_and_common17_projection_feasibility_not_admitted_provider",
            "artifact": {
                "hf_repository": "eugenehp/seizuretransformer",
                "hf_commit": EXPECTED_HF_COMMIT,
                "model_sha256": model_sha,
                "model_size_bytes": paths["model.safetensors"].stat().st_size,
                "tensor_count": len(state),
                "state_value_count_19": sum(tensor.numel() for tensor in state.values()),
                "trainable_parameter_count_19": sum(
                    parameter.numel() for parameter in model19.parameters()
                ),
                "strict_official_source_model_load": True,
            },
            "claimed_provenance": {
                "statement_owner": "third_party_Hugging_Face_distributor_not_original_paper_authors",
                "source_project": "keruiwu/SeizureTransformer",
                "container": "yujjio/seizure_transformer:mutable_latest_tag",
                "container_checkpoint_path": "wu_2025/model.pth",
                "original_container_digest_verified": False,
                "conversion_equivalence_to_original_pth_verified": False,
            },
            "common17_projection": {
                "source_axis": list(STANDARD19),
                "projected_axis": list(COMMON17),
                "deleted_signal_axes": ["FZ", "PZ"],
                "prediction_or_label_mapping_to_CZ_used": False,
                "changed_state_tensors": different_shapes,
                "removed_first_layer_weight_values": removed_values,
                "state_value_count_17": sum(
                    tensor.numel() for tensor in projected.values()
                ),
                "trainable_parameter_count_17": sum(
                    parameter.numel() for parameter in model17.parameters()
                ),
                "strict_projected_model_load": True,
                "projection_is_training_or_domain_adaptation": False,
            },
            "access_receipt": {
                "raw_EEG_loaded": False,
                "TUSZ_or_other_labels_loaded": False,
                "source_dev_or_source_eval_opened": False,
                "inference_performed": False,
                "training_or_finetuning_performed": False,
            },
            "license_and_exposure_boundary": {
                "distribution_terms": "research_only_noncommercial_not_for_medical_use",
                "upstream_rights_precedence_stated": True,
                "exact_training_patient_and_record_exposure_known": False,
                "TUSZ_source_dev_model_selection_exposure_possible": True,
                "TUSZ_source_eval_exposure_excluded": False,
                "eligible_as_clean_common17_primary": False,
                "allowed_role": "external_exposure_unknown_sensitivity_or_warm_start_diagnostic_only",
            },
            "claim_boundary": {
                "checkpoint_performance_metric_available": False,
                "common17_prediction_inventory_available": False,
                "warm_start_improves_common17_detection": False,
                "clinical_or_production_use_authorized": False,
            },
            "source_files": {
                name: {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
                for name, path in paths.items()
            },
            "receipt_sha256": PENDING,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.source)
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
