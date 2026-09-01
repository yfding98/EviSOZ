"""Closed v1.2 artifacts for target-free LaBraM k31 OOF recovery.

Version 1.2 is intentionally a distinct capability from the superseded v1.1
bundle.  It records that TUSZ ictal-involvement supervision was loaded while
the DeepSOZ SOZ target source and values were structurally unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file

from .concept_run import ictal_head_state_sha256
from .ictal_recovery_oof import (
    LABRAM_K31_CANDIDATE,
    LABRAM_K31_CONTEXT_SECONDS,
    LABRAM_K31_OOF_CHECKPOINT_FILENAME,
    LABRAM_K31_OOF_MANIFEST_FILENAME,
    LABRAM_K31_TARGET_SEMANTICS,
    _MANIFEST_FIELDS as _V11_FIELDS,
    _canonical_json_bytes,
    _file_sha256,
    _patient_roster,
    _require_sha256,
    _safe_new_output,
    _selection,
    _validated_training_metadata,
    patient_roster_sha256,
    save_labram_k31_oof_recovery_run,
)
from .models.concept_heads import LongContextTemporalResidualIctalInvolvementHead


LABRAM_K31_OOF_RUN_SCHEMA_V1_2 = "soz_labram_k31_ictal_oof_recovery_run_v1_2"
LABRAM_K31_EXECUTION_RECEIPT_SCHEMA = "soz_labram_k31_execution_receipt_v1"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_V12_FIELDS = frozenset(
    {
        *_V11_FIELDS,
        "deepsoz_target_source_loaded",
        "deepsoz_target_values_reachable",
        "tusz_ictal_involvement_targets_loaded",
        "v5_split_sha256",
        "execution_receipt",
        "execution_receipt_sha256",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "device_type",
        "device_name",
        "compute_capability",
        "optimizer_class",
        "optimizer_effective_hyperparameters",
        "training_config_sha256",
    }
)
_OPTIMIZER_FIELDS = frozenset(
    {
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _execution_receipt(value: object, *, training_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_FIELDS:
        raise ValueError("execution_receipt violates its closed schema")
    receipt = dict(value)
    if receipt["schema_version"] != LABRAM_K31_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("Unsupported execution receipt schema")
    for field in ("torch_version", "device_type", "device_name", "optimizer_class"):
        if not isinstance(receipt[field], str) or not receipt[field]:
            raise ValueError(f"execution_receipt.{field} must be non-empty")
    if receipt["device_type"] not in {"cpu", "cuda"}:
        raise ValueError("execution receipt device_type must be cpu or cuda")
    for field in ("cuda_runtime_version",):
        if receipt[field] is not None and not isinstance(receipt[field], str):
            raise TypeError(f"execution_receipt.{field} must be string or null")
    if receipt["cudnn_version"] is not None and (
        isinstance(receipt["cudnn_version"], bool)
        or not isinstance(receipt["cudnn_version"], int)
        or receipt["cudnn_version"] < 1
    ):
        raise ValueError("execution_receipt.cudnn_version must be positive or null")
    capability = receipt["compute_capability"]
    if capability is not None and (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in capability)
    ):
        raise ValueError("execution_receipt.compute_capability is invalid")
    if receipt["optimizer_class"] != "torch.optim.AdamW":
        raise ValueError("Recovery optimizer must be torch.optim.AdamW")
    optimizer = receipt["optimizer_effective_hyperparameters"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != _OPTIMIZER_FIELDS:
        raise ValueError("optimizer effective parameters violate their closed schema")
    optimizer = dict(optimizer)
    if (
        float(optimizer["lr"]) != float(training_config["learning_rate"])
        or float(optimizer["weight_decay"]) != float(training_config["weight_decay"])
    ):
        raise ValueError("Optimizer receipt disagrees with the training config")
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2 or any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0
        for value in betas
    ):
        raise ValueError("AdamW beta receipt is invalid")
    if not math.isfinite(float(optimizer["eps"])) or float(optimizer["eps"]) <= 0:
        raise ValueError("AdamW eps receipt is invalid")
    for field in ("amsgrad", "maximize", "capturable", "differentiable"):
        if not isinstance(optimizer[field], bool):
            raise TypeError(f"AdamW {field} receipt must be boolean")
    for field in ("foreach", "fused"):
        if optimizer[field] is not None and not isinstance(optimizer[field], bool):
            raise TypeError(f"AdamW {field} receipt must be bool or null")
    expected_config_sha = _canonical_sha256(dict(training_config))
    if receipt["training_config_sha256"] != expected_config_sha:
        raise ValueError("Execution receipt does not bind the training config")
    receipt["optimizer_effective_hyperparameters"] = optimizer
    return receipt


@dataclass(frozen=True)
class LoadedLaBraMK31OOFRecoveryRunV12:
    path: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    head: LongContextTemporalResidualIctalInvolvementHead


def save_labram_k31_oof_recovery_run_v1_2(
    output_directory: str | Path,
    *,
    v5_split_sha256: str,
    execution_receipt: Mapping[str, object],
    **legacy_arguments: object,
) -> LoadedLaBraMK31OOFRecoveryRunV12:
    """Atomically save a target-free v1.2 run without accepting target flags."""

    target = _safe_new_output(output_directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.v12-", dir=target.parent))
    moved = False
    try:
        intermediate = staging / "bundle"
        saved = save_labram_k31_oof_recovery_run(intermediate, **legacy_arguments)
        manifest_path = saved.path / LABRAM_K31_OOF_MANIFEST_FILENAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = LABRAM_K31_OOF_RUN_SCHEMA_V1_2
        payload["deepsoz_target_source_loaded"] = False
        payload["deepsoz_target_values_reachable"] = False
        payload["tusz_ictal_involvement_targets_loaded"] = True
        payload["v5_split_sha256"] = _require_sha256(
            v5_split_sha256, field="v5_split_sha256"
        )
        normalized_execution = _execution_receipt(
            execution_receipt, training_config=payload["training_config"]
        )
        payload["execution_receipt"] = normalized_execution
        payload["execution_receipt_sha256"] = _canonical_sha256(normalized_execution)
        manifest_path.write_bytes(_canonical_json_bytes(payload))
        os.rename(intermediate, target)
        moved = True
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if not moved:  # pragma: no cover - retained for explicit atomicity reasoning
        raise RuntimeError("v1.2 recovery bundle was not published")
    return load_labram_k31_oof_recovery_run_v1_2(
        target,
        expected_manifest_sha256=_file_sha256(
            target / LABRAM_K31_OOF_MANIFEST_FILENAME
        ),
    )


def load_labram_k31_oof_recovery_run_v1_2(
    path: str | Path, *, expected_manifest_sha256: str | None = None
) -> LoadedLaBraMK31OOFRecoveryRunV12:
    """Load only v1.2; v1/v1.1 bundles fail the closed field/schema checks."""

    source = Path(os.path.abspath(path))
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("v1.2 recovery bundle must be a regular absolute directory")
    if {item.name for item in source.iterdir()} != {
        LABRAM_K31_OOF_MANIFEST_FILENAME,
        LABRAM_K31_OOF_CHECKPOINT_FILENAME,
    }:
        raise ValueError("v1.2 recovery bundle has missing or unknown files")
    raw = (source / LABRAM_K31_OOF_MANIFEST_FILENAME).read_bytes()
    if not 1 <= len(raw) <= _MAX_MANIFEST_BYTES:
        raise ValueError("v1.2 recovery manifest has invalid size")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v1.2 recovery manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _V12_FIELDS
        or _canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("v1.2 recovery manifest violates its closed schema")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("v1.2 recovery manifest SHA mismatch")
    fixed = {
        "schema_version": LABRAM_K31_OOF_RUN_SCHEMA_V1_2,
        "candidate": LABRAM_K31_CANDIDATE,
        "context_seconds": LABRAM_K31_CONTEXT_SECONDS,
        "context_direction": "symmetric_retrospective_not_causal_onset",
        "target_semantics": LABRAM_K31_TARGET_SEMANTICS,
        "development_only": True,
        "architecture_selected_after_opened_i_dev": True,
        "formal_promotion": False,
        "checkpoint_authorized_for_formal_evidence_or_reasoner": False,
        "deepsoz_soz_labels_used": False,
        "private_labels_used": False,
        "missing_tusz_cells_imputed_as_negative": False,
        "i_gate_outcomes_opened": False,
        "deepsoz_target_source_loaded": False,
        "deepsoz_target_values_reachable": False,
        "tusz_ictal_involvement_targets_loaded": True,
        "checkpoint_filename": LABRAM_K31_OOF_CHECKPOINT_FILENAME,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("v1.2 recovery changed a target-access/scientific boundary")
    for field in (
        "split_manifest_sha256",
        "oof_protocol_artifact_sha256",
        "oof_protocol_receipt_sha256",
        "oof_plan_receipt_sha256",
        "training_manifest_sha256",
        "training_corpus_index_sha256",
        "target_snapshot_manifest_sha256",
        "target_snapshot_receipt_sha256",
        "native_evaluation_manifest_sha256",
        "native_evaluation_corpus_index_sha256",
        "training_public_roster_sha256",
        "held_out_exclusion_public_roster_sha256",
        "native_evaluation_public_roster_sha256",
        "i_gate_patient_roster_sha256",
        "head_state_sha256",
        "checkpoint_sha256",
        "v5_split_sha256",
        "execution_receipt_sha256",
    ):
        _require_sha256(manifest[field], field=field)
    selection, fold = _selection(manifest["selection"])
    if manifest["oof_fold"] != fold:
        raise ValueError("v1.2 selection/fold mismatch")
    rosters = {
        field: _patient_roster(manifest[field], field=field, allow_empty=False)
        for field in (
            "training_public_patient_ids",
            "held_out_exclusion_public_patient_ids",
            "native_evaluation_public_patient_ids",
            "i_gate_patient_ids_excluded_unopened",
        )
    }
    roster_receipts = {
        "training_public_patient_ids": "training_public_roster_sha256",
        "held_out_exclusion_public_patient_ids": "held_out_exclusion_public_roster_sha256",
        "native_evaluation_public_patient_ids": "native_evaluation_public_roster_sha256",
        "i_gate_patient_ids_excluded_unopened": "i_gate_patient_roster_sha256",
    }
    for roster, receipt in roster_receipts.items():
        if patient_roster_sha256(rosters[roster]) != manifest[receipt]:
            raise ValueError(f"v1.2 {roster} receipt mismatch")
    training = set(rosters["training_public_patient_ids"])
    held = set(rosters["held_out_exclusion_public_patient_ids"])
    native = set(rosters["native_evaluation_public_patient_ids"])
    gate = set(rosters["i_gate_patient_ids_excluded_unopened"])
    if len(gate) != 12 or training & (held | native | gate) or native & gate:
        raise ValueError("v1.2 recovery patient firewall failed")
    _validated_training_metadata(
        manifest["training_config"],
        manifest["training_run"],
        head_state_sha256=str(manifest["head_state_sha256"]),
        training_patient_count=len(training),
        native_patient_count=len(native),
    )
    execution = _execution_receipt(
        manifest["execution_receipt"], training_config=manifest["training_config"]
    )
    if _canonical_sha256(execution) != manifest["execution_receipt_sha256"]:
        raise ValueError("v1.2 execution receipt SHA mismatch")
    if manifest["head_config"] != {"token_dim": 200, "hidden_dim": 128}:
        raise ValueError("v1.2 recovery head configuration changed")
    checkpoint = source / LABRAM_K31_OOF_CHECKPOINT_FILENAME
    if _file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("v1.2 checkpoint SHA mismatch")
    state = load_file(str(checkpoint), device="cpu")
    head = LongContextTemporalResidualIctalInvolvementHead(
        token_dim=200, hidden_dim=128
    )
    expected = head.state_dict()
    if set(state) != set(expected):
        raise ValueError("v1.2 checkpoint tensor names changed")
    for name, reference in expected.items():
        tensor = state[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"v1.2 checkpoint tensor changed: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"v1.2 checkpoint tensor is non-finite: {name}")
    head.load_state_dict(state, strict=True)
    if ictal_head_state_sha256(head) != manifest["head_state_sha256"]:
        raise ValueError("v1.2 head-state receipt mismatch")
    head.eval()
    return LoadedLaBraMK31OOFRecoveryRunV12(
        path=source,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        head=head,
    )


__all__ = (
    "LABRAM_K31_EXECUTION_RECEIPT_SCHEMA",
    "LABRAM_K31_OOF_RUN_SCHEMA_V1_2",
    "LoadedLaBraMK31OOFRecoveryRunV12",
    "load_labram_k31_oof_recovery_run_v1_2",
    "save_labram_k31_oof_recovery_run_v1_2",
)
