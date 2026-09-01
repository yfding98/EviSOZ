"""Closed development-only artifacts for hierarchical morphology OOF runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from .morphology_recovery import (
    AUXILIARY_MORPHOLOGY_ROLES,
    MORPHOLOGY_RECOVERY_PROTOCOL_SHA256,
    HierarchicalMorphologyEvidenceHead,
)


MORPHOLOGY_RECOVERY_OOF_SCHEMA = (
    "soz_labram_morphology_hierarchical_oof_recovery_run_v1"
)
MORPHOLOGY_RECOVERY_OOF_CANDIDATE = (
    "labram_frozen_shared_adapter_ce6_plus_three_roles"
)
MORPHOLOGY_RECOVERY_OOF_MANIFEST = "recovery_run.json"
MORPHOLOGY_RECOVERY_OOF_CHECKPOINT = "model.safetensors"
MORPHOLOGY_RECOVERY_OOF_PREDICTIONS = "held_predictions.safetensors"

_SELECTION_RE = re.compile(r"fold([0-4])")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_FILE_NAMES = frozenset(("run_plan", "tokens", "labels", "mask", "weights"))
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "selection",
        "fold",
        "development_only",
        "formal_promotion",
        "dense_deployment_authorized",
        "soz_reasoner_authorized",
        "official_tuev_eval_used",
        "tusz_labels_used",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "unknown_cells_imputed_as_negative",
        "threshold_selection_performed",
        "architecture_selected_after_opened_m0_development",
        "target_semantics",
        "input_shape",
        "native_classes",
        "auxiliary_roles",
        "protocol_sha256",
        "preflight_receipt_sha256",
        "source_plan_sha256",
        "source_files_sha256",
        "fit_group_ids",
        "fit_group_roster_sha256",
        "held_group_ids",
        "held_group_roster_sha256",
        "training_config",
        "ce6_class_weights",
        "auxiliary_pos_weights",
        "epoch_group_mean_losses",
        "metrics",
        "head_config",
        "initial_head_state_sha256",
        "head_state_sha256",
        "checkpoint_filename",
        "checkpoint_sha256",
        "predictions_filename",
        "predictions_sha256",
        "held_item_count",
        "held_item_indices_sha256",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha_payload(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, field: str) -> str:
    text = str(value)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _selection(value: object) -> tuple[str, int]:
    text = str(value)
    match = _SELECTION_RE.fullmatch(text)
    if match is None:
        raise ValueError("Morphology recovery selection must be fold0..fold4")
    return text, int(match.group(1))


def _roster(values: Sequence[object], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(str(value) for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be non-empty, sorted, and unique")
    return result


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def morphology_recovery_head_state_sha256(
    head: HierarchicalMorphologyEvidenceHead,
) -> str:
    if not isinstance(head, HierarchicalMorphologyEvidenceHead):
        raise TypeError("head must be HierarchicalMorphologyEvidenceHead")
    return _state_sha256(head.state_dict())


def morphology_recovery_training_config(fold: int) -> dict[str, object]:
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("Morphology recovery fold must be 0--4")
    return {
        "fixed_epochs": 20,
        "seed": 20260808 + fold,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "crop_microbatch_size": 32,
        "hidden_dim": 128,
        "ce6_class_weight_cap": 10.0,
        "auxiliary_pos_weight_cap": 10.0,
        "loss": "group_equal_component_weighted_CE6_plus_mean_three_role_BCE",
        "checkpoint_selection": "fixed_final_epoch",
        "early_stopping": False,
        "hyperparameter_sweep": False,
        "foundation_frozen": True,
    }


def _validate_probabilities(
    held_item_indices: torch.Tensor,
    ce6_probabilities: torch.Tensor,
    auxiliary_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = held_item_indices.detach().cpu().contiguous()
    ce6 = ce6_probabilities.detach().cpu().float().contiguous()
    auxiliary = auxiliary_probabilities.detach().cpu().float().contiguous()
    if indices.dtype != torch.long or indices.ndim != 1 or indices.numel() < 1:
        raise ValueError("Held item indices must be non-empty int64 [N]")
    if not torch.equal(indices, torch.unique(indices, sorted=True)):
        raise ValueError("Held item indices must be sorted and unique")
    expected_n = int(indices.numel())
    if tuple(ce6.shape) != (expected_n, 20, 6):
        raise ValueError("Held CE6 probabilities must have shape [N,20,6]")
    if tuple(auxiliary.shape) != (expected_n, 20, 3):
        raise ValueError("Held auxiliary probabilities must have shape [N,20,3]")
    for name, values in (("CE6", ce6), ("auxiliary", auxiliary)):
        if not torch.isfinite(values).all() or torch.any((values < 0) | (values > 1)):
            raise ValueError(f"Held {name} probabilities must be finite in [0,1]")
    if not torch.allclose(
        ce6.sum(dim=-1), torch.ones_like(ce6[..., 0]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Held CE6 probabilities must sum to one")
    return indices, ce6, auxiliary


def _validate_metrics(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("Morphology recovery metrics must be a non-empty mapping")

    def walk(item: object, field: str) -> object:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{field} must be finite")
            return item
        if isinstance(item, list):
            return [walk(child, f"{field}[]") for child in item]
        if isinstance(item, Mapping):
            return {str(key): walk(child, f"{field}.{key}") for key, child in item.items()}
        raise TypeError(f"Unsupported metric value at {field}")

    return walk(dict(value), "metrics")  # type: ignore[return-value]


def _validate_manifest(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Morphology recovery manifest must be a mapping")
    manifest = dict(payload)
    if set(manifest) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST_FIELDS)
        raise ValueError(f"Closed recovery manifest mismatch; missing={missing}, extra={extra}")
    if manifest["schema_version"] != MORPHOLOGY_RECOVERY_OOF_SCHEMA:
        raise ValueError("Unexpected morphology recovery OOF schema")
    if manifest["candidate"] != MORPHOLOGY_RECOVERY_OOF_CANDIDATE:
        raise ValueError("Unexpected morphology recovery candidate")
    selection, fold = _selection(manifest["selection"])
    if manifest["fold"] != fold:
        raise ValueError("Recovery selection and fold disagree")
    required_true = {
        "development_only",
        "architecture_selected_after_opened_m0_development",
    }
    required_false = {
        "formal_promotion",
        "dense_deployment_authorized",
        "soz_reasoner_authorized",
        "official_tuev_eval_used",
        "tusz_labels_used",
        "deepsoz_soz_labels_used",
        "private_labels_used",
        "unknown_cells_imputed_as_negative",
        "threshold_selection_performed",
    }
    if any(manifest[field] is not True for field in required_true) or any(
        manifest[field] is not False for field in required_false
    ):
        raise ValueError("Morphology recovery development/safety flags changed")
    if manifest["target_semantics"] != "tuev_native_ce6_bipolar_edge_not_soz":
        raise ValueError("Morphology recovery target semantics changed")
    if manifest["input_shape"] != ["N", 19, 1, 200]:
        raise ValueError("Morphology recovery input shape changed")
    if manifest["native_classes"] != ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]:
        raise ValueError("Morphology recovery CE6 ontology changed")
    if manifest["auxiliary_roles"] != list(AUXILIARY_MORPHOLOGY_ROLES):
        raise ValueError("Morphology recovery auxiliary ontology changed")
    if _require_sha(manifest["protocol_sha256"], "protocol_sha256") != MORPHOLOGY_RECOVERY_PROTOCOL_SHA256:
        raise ValueError("Morphology recovery protocol SHA changed")
    for field in (
        "preflight_receipt_sha256",
        "source_plan_sha256",
        "fit_group_roster_sha256",
        "held_group_roster_sha256",
        "head_state_sha256",
        "initial_head_state_sha256",
        "checkpoint_sha256",
        "predictions_sha256",
        "held_item_indices_sha256",
    ):
        _require_sha(manifest[field], field)
    sources = manifest["source_files_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != _SOURCE_FILE_NAMES:
        raise ValueError("Morphology recovery source-file SHA roster changed")
    for field, value in sources.items():
        _require_sha(value, f"source_files_sha256.{field}")
    fit = _roster(manifest["fit_group_ids"], "fit_group_ids")
    held = _roster(manifest["held_group_ids"], "held_group_ids")
    if set(fit) & set(held):
        raise ValueError("Morphology recovery fit and held groups overlap")
    if manifest["fit_group_roster_sha256"] != _sha_payload(fit) or manifest[
        "held_group_roster_sha256"
    ] != _sha_payload(held):
        raise ValueError("Morphology recovery group-roster SHA mismatch")
    if manifest["training_config"] != morphology_recovery_training_config(fold):
        raise ValueError("Morphology recovery training config changed")
    for field, length in (("ce6_class_weights", 6), ("auxiliary_pos_weights", 3)):
        values = manifest[field]
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0 for value in values)
        ):
            raise ValueError(f"Morphology recovery {field} is invalid")
    losses = manifest["epoch_group_mean_losses"]
    if (
        not isinstance(losses, list)
        or len(losses) != 20
        or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 for value in losses)
    ):
        raise ValueError("Morphology recovery epoch history is incomplete")
    _validate_metrics(manifest["metrics"])
    if manifest["head_config"] != {"token_dim": 200, "hidden_dim": 128}:
        raise ValueError("Morphology recovery head config changed")
    if manifest["initial_head_state_sha256"] == manifest["head_state_sha256"]:
        raise ValueError("Morphology recovery optimizer did not change the head state")
    if manifest["checkpoint_filename"] != MORPHOLOGY_RECOVERY_OOF_CHECKPOINT or manifest[
        "predictions_filename"
    ] != MORPHOLOGY_RECOVERY_OOF_PREDICTIONS:
        raise ValueError("Morphology recovery artifact filenames changed")
    if isinstance(manifest["held_item_count"], bool) or not isinstance(
        manifest["held_item_count"], int
    ) or manifest["held_item_count"] < 1:
        raise ValueError("Morphology recovery held item count must be positive")
    return manifest


@dataclass(frozen=True)
class LoadedMorphologyRecoveryOOFRun:
    path: Path
    manifest: dict[str, object]
    manifest_file_sha256: str
    head: HierarchicalMorphologyEvidenceHead
    held_item_indices: torch.Tensor
    ce6_probabilities: torch.Tensor
    auxiliary_probabilities: torch.Tensor


def save_morphology_recovery_oof_run(
    output_directory: str | Path,
    *,
    selection: str,
    head: HierarchicalMorphologyEvidenceHead,
    held_item_indices: torch.Tensor,
    ce6_probabilities: torch.Tensor,
    auxiliary_probabilities: torch.Tensor,
    fit_group_ids: Sequence[str],
    held_group_ids: Sequence[str],
    preflight_receipt_sha256: str,
    source_plan_sha256: str,
    source_files_sha256: Mapping[str, str],
    ce6_class_weights: Sequence[float],
    auxiliary_pos_weights: Sequence[float],
    epoch_group_mean_losses: Sequence[float],
    metrics: Mapping[str, object],
    initial_head_state_sha256: str,
) -> LoadedMorphologyRecoveryOOFRun:
    """Atomically save one fold; existing outputs are never overwritten."""

    selection, fold = _selection(selection)
    if not isinstance(head, HierarchicalMorphologyEvidenceHead):
        raise TypeError("head must be HierarchicalMorphologyEvidenceHead")
    fit = _roster(fit_group_ids, "fit_group_ids")
    held = _roster(held_group_ids, "held_group_ids")
    if set(fit) & set(held):
        raise ValueError("Morphology recovery fit and held groups overlap")
    indices, ce6, auxiliary = _validate_probabilities(
        held_item_indices, ce6_probabilities, auxiliary_probabilities
    )
    source_hashes = {str(key): str(value) for key, value in source_files_sha256.items()}
    if set(source_hashes) != _SOURCE_FILE_NAMES:
        raise ValueError("Morphology recovery source-file SHA roster is incomplete")
    for field, value in source_hashes.items():
        _require_sha(value, f"source_files_sha256.{field}")
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in head.state_dict().items()
    }
    head_state_sha = _state_sha256(state)
    target = Path(os.path.abspath(output_directory))
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise ValueError("Morphology recovery output requires a concrete existing parent")
    if os.path.lexists(target):
        raise FileExistsError(f"Morphology recovery output already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        checkpoint = temporary / MORPHOLOGY_RECOVERY_OOF_CHECKPOINT
        predictions = temporary / MORPHOLOGY_RECOVERY_OOF_PREDICTIONS
        save_file(state, str(checkpoint))
        prediction_state = {
            "held_item_indices": indices,
            "ce6_probabilities": ce6,
            "auxiliary_probabilities": auxiliary,
        }
        save_file(prediction_state, str(predictions))
        manifest = {
            "schema_version": MORPHOLOGY_RECOVERY_OOF_SCHEMA,
            "candidate": MORPHOLOGY_RECOVERY_OOF_CANDIDATE,
            "selection": selection,
            "fold": fold,
            "development_only": True,
            "formal_promotion": False,
            "dense_deployment_authorized": False,
            "soz_reasoner_authorized": False,
            "official_tuev_eval_used": False,
            "tusz_labels_used": False,
            "deepsoz_soz_labels_used": False,
            "private_labels_used": False,
            "unknown_cells_imputed_as_negative": False,
            "threshold_selection_performed": False,
            "architecture_selected_after_opened_m0_development": True,
            "target_semantics": "tuev_native_ce6_bipolar_edge_not_soz",
            "input_shape": ["N", 19, 1, 200],
            "native_classes": ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"],
            "auxiliary_roles": list(AUXILIARY_MORPHOLOGY_ROLES),
            "protocol_sha256": MORPHOLOGY_RECOVERY_PROTOCOL_SHA256,
            "preflight_receipt_sha256": _require_sha(
                preflight_receipt_sha256, "preflight_receipt_sha256"
            ),
            "source_plan_sha256": _require_sha(source_plan_sha256, "source_plan_sha256"),
            "source_files_sha256": source_hashes,
            "fit_group_ids": list(fit),
            "fit_group_roster_sha256": _sha_payload(fit),
            "held_group_ids": list(held),
            "held_group_roster_sha256": _sha_payload(held),
            "training_config": morphology_recovery_training_config(fold),
            "ce6_class_weights": [float(value) for value in ce6_class_weights],
            "auxiliary_pos_weights": [float(value) for value in auxiliary_pos_weights],
            "epoch_group_mean_losses": [float(value) for value in epoch_group_mean_losses],
            "metrics": _validate_metrics(metrics),
            "head_config": {"token_dim": 200, "hidden_dim": 128},
            "initial_head_state_sha256": _require_sha(
                initial_head_state_sha256, "initial_head_state_sha256"
            ),
            "head_state_sha256": head_state_sha,
            "checkpoint_filename": MORPHOLOGY_RECOVERY_OOF_CHECKPOINT,
            "checkpoint_sha256": _file_sha256(checkpoint),
            "predictions_filename": MORPHOLOGY_RECOVERY_OOF_PREDICTIONS,
            "predictions_sha256": _file_sha256(predictions),
            "held_item_count": int(indices.numel()),
            "held_item_indices_sha256": _sha_payload(indices.tolist()),
        }
        manifest = _validate_manifest(manifest)
        manifest_path = temporary / MORPHOLOGY_RECOVERY_OOF_MANIFEST
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
        for path in (checkpoint, predictions, manifest_path):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_morphology_recovery_oof_run(target)


def load_morphology_recovery_oof_run(
    value: str | Path,
    *,
    expected_manifest_file_sha256: str | None = None,
) -> LoadedMorphologyRecoveryOOFRun:
    """Strict loader that preserves the development-only safety boundary."""

    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Morphology recovery OOF artifact must be a directory")
    manifest_path = path / MORPHOLOGY_RECOVERY_OOF_MANIFEST
    if not manifest_path.is_file() or manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Morphology recovery OOF manifest is missing or oversized")
    manifest_file_sha = _file_sha256(manifest_path)
    if expected_manifest_file_sha256 is not None and manifest_file_sha != _require_sha(
        expected_manifest_file_sha256, "expected_manifest_file_sha256"
    ):
        raise ValueError("Morphology recovery OOF manifest is not the expected artifact")
    manifest = _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    checkpoint = path / str(manifest["checkpoint_filename"])
    predictions = path / str(manifest["predictions_filename"])
    if _file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Morphology recovery checkpoint SHA mismatch")
    if _file_sha256(predictions) != manifest["predictions_sha256"]:
        raise ValueError("Morphology recovery prediction SHA mismatch")
    state = load_file(str(checkpoint), device="cpu")
    if _state_sha256(state) != manifest["head_state_sha256"]:
        raise ValueError("Morphology recovery head-state SHA mismatch")
    head = HierarchicalMorphologyEvidenceHead(token_dim=200, hidden_dim=128)
    head.load_state_dict(state, strict=True)
    prediction_state = load_file(str(predictions), device="cpu")
    if set(prediction_state) != {
        "held_item_indices",
        "ce6_probabilities",
        "auxiliary_probabilities",
    }:
        raise ValueError("Morphology recovery prediction tensor roster changed")
    indices, ce6, auxiliary = _validate_probabilities(
        prediction_state["held_item_indices"],
        prediction_state["ce6_probabilities"],
        prediction_state["auxiliary_probabilities"],
    )
    if int(indices.numel()) != manifest["held_item_count"] or _sha_payload(
        indices.tolist()
    ) != manifest["held_item_indices_sha256"]:
        raise ValueError("Morphology recovery held-item roster changed")
    return LoadedMorphologyRecoveryOOFRun(
        path=path,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        head=head,
        held_item_indices=indices,
        ce6_probabilities=ce6,
        auxiliary_probabilities=auxiliary,
    )


__all__ = [
    "MORPHOLOGY_RECOVERY_OOF_CANDIDATE",
    "MORPHOLOGY_RECOVERY_OOF_CHECKPOINT",
    "MORPHOLOGY_RECOVERY_OOF_MANIFEST",
    "MORPHOLOGY_RECOVERY_OOF_PREDICTIONS",
    "MORPHOLOGY_RECOVERY_OOF_SCHEMA",
    "LoadedMorphologyRecoveryOOFRun",
    "load_morphology_recovery_oof_run",
    "morphology_recovery_head_state_sha256",
    "morphology_recovery_training_config",
    "save_morphology_recovery_oof_run",
]
