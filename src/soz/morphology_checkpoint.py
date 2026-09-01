"""Safe, provenance-complete checkpoints for the native TUEV morphology head."""

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
from typing import Mapping

import numpy as np
import torch

from .data.tuev_morphology import FOLD_COUNT_SEMANTICS, TUEVMorphologyManifest
from .geometry import MORPHOLOGY_CLASSES
from .models.concept_heads import MorphologyEvidenceHead
from .models.labram import LaBraMFeatureReceipt
from .morphology_token_io import morphology_foundation_receipt_sha256
from .morphology_training import (
    MORPHOLOGY_EVALUATION_SCHEMA,
    MORPHOLOGY_TRAINING_CONFIG_SCHEMA,
    MORPHOLOGY_TRAINING_RUN_SCHEMA,
    MORPHOLOGY_TYPED_ROUTING_POLICY,
    MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256,
    MorphologyEvaluationReceipt,
    MorphologyTrainingRunReceipt,
    morphology_target_bearing_group_ids,
)

try:
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except ImportError:  # pragma: no cover
    _load_safetensors = None
    _save_safetensors = None


MORPHOLOGY_CHECKPOINT_SCHEMA = "soz_morphology_checkpoint_bundle_v2"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_STATE_BYTES = 32 * 1024 * 1024
_STATE_FILE_BY_FORMAT = {
    "safetensors": "morphology_head.safetensors",
    "npz": "morphology_head.npz",
}
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "serialization",
        "model_class",
        "token_dim",
        "hidden_dim",
        "state_format",
        "state_file",
        "state_names",
        "state_specs",
        "state_payload_sha256",
        "state_file_sha256",
        "state_file_size_bytes",
        "training_run_file",
        "training_run_sha256",
        "training_run_size_bytes",
        "evaluation_file",
        "evaluation_sha256",
        "evaluation_size_bytes",
        "routing_policy_file",
        "routing_policy_sha256",
        "routing_policy_size_bytes",
        "fold_manifest_sha256",
        "master_manifest_sha256",
        "master_token_corpus_index_sha256",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "preprocessing_receipt_roster_sha256",
        "fit_group_ids",
        "held_group_ids",
        "fit_group_roster_sha256",
        "held_group_roster_sha256",
        "class_weights",
    }
)
_FOUNDATION_FIELDS = frozenset(
    {
        "checkpoint_path",
        "checkpoint_sha256",
        "modeling_path",
        "modeling_sha256",
        "encoder_tensor_count",
        "semantic_channels",
        "position_names",
        "position_ids",
        "tile_seconds",
        "pretraining_window_seconds",
        "samples_per_token",
        "token_dim",
        "input_scale_from_volts",
    }
)
_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "fold_manifest_sha256",
        "master_manifest_sha256",
        "master_token_corpus_index_sha256",
        "foundation_feature_receipt_sha256",
        "routing_policy_sha256",
        "fit_group_ids",
        "held_group_ids",
        "fit_group_roster_sha256",
        "held_group_roster_sha256",
        "class_weights",
        "epoch_group_mean_losses",
        "fit_crop_count",
        "fit_target_count",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "fixed_epochs",
        "learning_rate",
        "weight_decay",
        "crop_microbatch_size",
        "gradient_clip_norm",
        "class_weight_cap",
        "seed",
        "deterministic_algorithms",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_role",
        "fold_manifest_sha256",
        "master_token_corpus_index_sha256",
        "checkpoint_or_run_sha256",
        "group_ids",
        "group_roster_sha256",
        "group_kind_counts",
        "target_count",
        "weighted_nll",
        "weighted_brier",
        "weighted_ece",
        "group_macro_balanced_accuracy",
        "class_metrics",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_json(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def _exact(value: Mapping[str, object], fields: frozenset[str], *, label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _state_arrays(head: MorphologyEvidenceHead) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in head.state_dict().items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"Morphology state entry {name!r} is not a dense tensor")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Morphology state entry {name!r} is non-finite")
        arrays[name] = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return arrays


def _specs(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in sorted(arrays.items())
    }


def _state_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(array)
        header = _canonical_json(
            {"name": name, "shape": list(contiguous.shape), "dtype": str(contiguous.dtype)}
        )
        digest.update(len(header).to_bytes(4, "little"))
        digest.update(header)
        raw = contiguous.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _write_state(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if _save_safetensors is not None:
        _save_safetensors(dict(arrays), str(path))
        return "safetensors"
    with path.open("wb") as handle:  # pragma: no cover
        np.savez_compressed(handle, **arrays)
    return "npz"


def _read_state(path: Path, state_format: str) -> dict[str, np.ndarray]:
    if state_format == "safetensors":
        if _load_safetensors is None:
            raise RuntimeError("safetensors is required to load this morphology checkpoint")
        return dict(_load_safetensors(str(path)))
    if state_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError("Unsupported morphology checkpoint state format")


def _foundation_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    payload = receipt.to_dict()
    payload["semantic_channels"] = list(receipt.semantic_channels)
    payload["position_names"] = list(receipt.position_names)
    payload["position_ids"] = list(receipt.position_ids)
    _exact(payload, _FOUNDATION_FIELDS, label="foundation_feature_receipt")
    if morphology_foundation_receipt_sha256(receipt) != hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest():
        raise ValueError("Foundation feature receipt is not canonically stable")
    return payload


def _foundation_from_payload(payload: Mapping[str, object]) -> LaBraMFeatureReceipt:
    _exact(payload, _FOUNDATION_FIELDS, label="foundation_feature_receipt")
    for field in ("semantic_channels", "position_names", "position_ids"):
        if not isinstance(payload[field], list):
            raise TypeError(f"foundation_feature_receipt.{field} must be an array")
    receipt = LaBraMFeatureReceipt(
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_sha256=payload["checkpoint_sha256"],
        modeling_path=payload["modeling_path"],
        modeling_sha256=payload["modeling_sha256"],
        encoder_tensor_count=payload["encoder_tensor_count"],
        semantic_channels=tuple(payload["semantic_channels"]),
        position_names=tuple(payload["position_names"]),
        position_ids=tuple(payload["position_ids"]),
        tile_seconds=payload["tile_seconds"],
        pretraining_window_seconds=payload["pretraining_window_seconds"],
        samples_per_token=payload["samples_per_token"],
        token_dim=payload["token_dim"],
        input_scale_from_volts=payload["input_scale_from_volts"],
    )
    # The public morphology cache validator performs all semantic geometry and
    # dimensional checks; hashing it here replays that strict path.
    morphology_foundation_receipt_sha256(receipt)
    return receipt


def _preprocessing_roster_sha256(manifest: TUEVMorphologyManifest) -> str:
    return hashlib.sha256(
        _canonical_json(
            [
                [record.record_id, record.metadata.preprocessing_receipt_sha256]
                for record in manifest.records
            ]
        )
    ).hexdigest()


def _infer_hidden_dim(head: MorphologyEvidenceHead) -> int:
    adapter = head.adapter[0]
    if not isinstance(adapter, torch.nn.Linear) or adapter.in_features != 600:
        raise ValueError("Morphology head edge adapter architecture drifted")
    if head.classifier.in_features != adapter.out_features or head.classifier.out_features != 6:
        raise ValueError("Morphology head classifier architecture drifted")
    return int(adapter.out_features)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class MorphologyCheckpointArtifact:
    path: Path
    manifest_sha256: str
    state_sha256: str
    training_run_sha256: str
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Checkpoint artifact path must be absolute")
        for field in (
            "manifest_sha256",
            "state_sha256",
            "training_run_sha256",
            "evaluation_sha256",
        ):
            _sha(getattr(self, field), field=field)


def save_morphology_checkpoint(
    path: str | Path,
    head: MorphologyEvidenceHead,
    *,
    training_run: MorphologyTrainingRunReceipt,
    evaluation: MorphologyEvaluationReceipt,
    fold_manifest: TUEVMorphologyManifest,
    foundation_feature_receipt: LaBraMFeatureReceipt,
) -> MorphologyCheckpointArtifact:
    """Atomically publish weights plus exact run/evaluation/provenance receipts."""

    if not isinstance(head, MorphologyEvidenceHead):
        raise TypeError("head must be MorphologyEvidenceHead")
    if not isinstance(training_run, MorphologyTrainingRunReceipt):
        raise TypeError("training_run must be MorphologyTrainingRunReceipt")
    if not isinstance(evaluation, MorphologyEvaluationReceipt):
        raise TypeError("evaluation must be MorphologyEvaluationReceipt")
    if not isinstance(fold_manifest, TUEVMorphologyManifest) or fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Morphology checkpoints require a fold-specific manifest")
    if training_run.fold_manifest_sha256 != fold_manifest.manifest_sha256:
        raise ValueError("Training run belongs to another morphology fold manifest")
    expected_fit_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="fit"
    )
    expected_held_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="held"
    )
    if (
        training_run.fit_group_ids != expected_fit_groups
        or training_run.held_group_ids != expected_held_groups
    ):
        raise ValueError("Training run fit/held rosters differ from the fold manifest")
    if evaluation.group_ids != expected_held_groups:
        raise ValueError("Evaluation does not cover the held target-bearing roster")
    if evaluation.fold_manifest_sha256 != fold_manifest.manifest_sha256:
        raise ValueError("Evaluation belongs to another morphology fold manifest")
    if evaluation.master_token_corpus_index_sha256 != training_run.master_token_corpus_index_sha256:
        raise ValueError("Training/evaluation use different morphology token corpora")
    if evaluation.checkpoint_or_run_sha256 != training_run.receipt_sha256:
        raise ValueError("Pre-checkpoint evaluation must bind the exact training run")
    foundation_payload = _foundation_payload(foundation_feature_receipt)
    foundation_sha = morphology_foundation_receipt_sha256(foundation_feature_receipt)
    if foundation_sha != training_run.foundation_feature_receipt_sha256:
        raise ValueError("Training run uses another foundation feature receipt")
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Morphology checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        arrays = _state_arrays(head)
        state_format = "safetensors" if _save_safetensors is not None else "npz"
        state_file = _STATE_FILE_BY_FORMAT[state_format]
        state_path = staging / state_file
        if _write_state(state_path, arrays) != state_format:
            raise RuntimeError("Morphology checkpoint serializer changed format")
        _fsync_file(state_path)
        state_size = state_path.stat().st_size
        if not 1 <= state_size <= _MAX_STATE_BYTES:
            raise ValueError("Morphology checkpoint state file size is invalid")

        run_bytes = _canonical_json(training_run.canonical_payload)
        evaluation_bytes = _canonical_json(evaluation.canonical_payload)
        routing_bytes = _canonical_json(MORPHOLOGY_TYPED_ROUTING_POLICY)
        receipt_items = (
            ("training_run.json", run_bytes),
            ("evaluation.json", evaluation_bytes),
            ("routing_policy.json", routing_bytes),
        )
        if any(not 1 <= len(raw) <= _MAX_RECEIPT_BYTES for _, raw in receipt_items):
            raise ValueError("Morphology checkpoint receipt size is invalid")
        for filename, raw in receipt_items:
            receipt_path = staging / filename
            receipt_path.write_bytes(raw)
            _fsync_file(receipt_path)

        hidden_dim = _infer_hidden_dim(head)
        manifest = {
            "schema_version": MORPHOLOGY_CHECKPOINT_SCHEMA,
            "serialization": "canonical_json_and_safe_tensors_no_pickle",
            "model_class": "MorphologyEvidenceHead",
            "token_dim": 200,
            "hidden_dim": hidden_dim,
            "state_format": state_format,
            "state_file": state_file,
            "state_names": sorted(arrays),
            "state_specs": _specs(arrays),
            "state_payload_sha256": _state_sha256(arrays),
            "state_file_sha256": _file_sha256(state_path),
            "state_file_size_bytes": state_size,
            "training_run_file": "training_run.json",
            "training_run_sha256": hashlib.sha256(run_bytes).hexdigest(),
            "training_run_size_bytes": len(run_bytes),
            "evaluation_file": "evaluation.json",
            "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
            "evaluation_size_bytes": len(evaluation_bytes),
            "routing_policy_file": "routing_policy.json",
            "routing_policy_sha256": hashlib.sha256(routing_bytes).hexdigest(),
            "routing_policy_size_bytes": len(routing_bytes),
            "fold_manifest_sha256": fold_manifest.manifest_sha256,
            "master_manifest_sha256": training_run.master_manifest_sha256,
            "master_token_corpus_index_sha256": training_run.master_token_corpus_index_sha256,
            "foundation_feature_receipt": foundation_payload,
            "foundation_feature_receipt_sha256": foundation_sha,
            "foundation_checkpoint_sha256": foundation_feature_receipt.checkpoint_sha256,
            "preprocessing_receipt_roster_sha256": _preprocessing_roster_sha256(fold_manifest),
            "fit_group_ids": list(training_run.fit_group_ids),
            "held_group_ids": list(training_run.held_group_ids),
            "fit_group_roster_sha256": training_run.fit_group_roster_sha256,
            "held_group_roster_sha256": training_run.held_group_roster_sha256,
            "class_weights": list(training_run.class_weights),
        }
        _exact(manifest, _MANIFEST_FIELDS, label="manifest.json")
        manifest_bytes = _canonical_json(manifest)
        if not 1 <= len(manifest_bytes) <= _MAX_MANIFEST_BYTES:
            raise ValueError("Morphology checkpoint manifest size is invalid")
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)
        _fsync_dir(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Morphology checkpoint already exists: {target}")
        os.replace(staging, target)
        _fsync_dir(target.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return MorphologyCheckpointArtifact(
        path=target,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        state_sha256=manifest["state_payload_sha256"],
        training_run_sha256=manifest["training_run_sha256"],
        evaluation_sha256=manifest["evaluation_sha256"],
    )


def _read_bound_receipt(
    source: Path,
    manifest: Mapping[str, object],
    *,
    file_field: str,
    sha_field: str,
    size_field: str,
    expected_filename: str,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if manifest[file_field] != expected_filename:
        raise ValueError(f"{label} filename is invalid")
    path = source / expected_filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    size = _integer(manifest[size_field], field=size_field, minimum=1)
    if size > _MAX_RECEIPT_BYTES or path.stat().st_size != size:
        raise ValueError(f"{label} file size mismatch")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(manifest[sha_field], field=sha_field):
        raise ValueError(f"{label} SHA-256 mismatch")
    payload = _parse(raw, label=label)
    _exact(payload, expected_fields, label=label)
    return payload


def _validate_run_payload(
    run: Mapping[str, object], manifest: Mapping[str, object]
) -> None:
    if run["schema_version"] != MORPHOLOGY_TRAINING_RUN_SCHEMA:
        raise ValueError("Checkpoint embeds an unsupported morphology training run")
    config = run["config"]
    if not isinstance(config, dict):
        raise TypeError("Morphology training config must be an object")
    _exact(config, _CONFIG_FIELDS, label="training_run.config")
    if config["schema_version"] != MORPHOLOGY_TRAINING_CONFIG_SCHEMA:
        raise ValueError("Morphology training config schema drifted")
    bindings = (
        ("fold_manifest_sha256", "fold_manifest_sha256"),
        ("master_manifest_sha256", "master_manifest_sha256"),
        ("master_token_corpus_index_sha256", "master_token_corpus_index_sha256"),
        ("foundation_feature_receipt_sha256", "foundation_feature_receipt_sha256"),
        ("routing_policy_sha256", "routing_policy_sha256"),
        ("fit_group_ids", "fit_group_ids"),
        ("held_group_ids", "held_group_ids"),
        ("fit_group_roster_sha256", "fit_group_roster_sha256"),
        ("held_group_roster_sha256", "held_group_roster_sha256"),
        ("class_weights", "class_weights"),
    )
    for run_field, manifest_field in bindings:
        if run[run_field] != manifest[manifest_field]:
            raise ValueError(f"Training run binding drifted: {run_field}")


def _validate_evaluation_payload(evaluation: Mapping[str, object]) -> None:
    if (
        evaluation["schema_version"] != MORPHOLOGY_EVALUATION_SCHEMA
        or evaluation["dataset_role"] != "held"
    ):
        raise ValueError("Checkpoint embeds an invalid morphology evaluation")
    _integer(evaluation["target_count"], field="evaluation.target_count", minimum=1)
    for field in (
        "weighted_nll",
        "weighted_brier",
        "weighted_ece",
        "group_macro_balanced_accuracy",
    ):
        value = evaluation[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Morphology evaluation {field} must be finite")
    rows = evaluation["class_metrics"]
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("Morphology evaluation must contain six native class rows")
    for class_name, row in zip(MORPHOLOGY_CLASSES, rows):
        if not isinstance(row, list) or len(row) != 6 or row[0] != class_name:
            raise ValueError(
                "Morphology class metrics must be "
                "[class,support,precision,recall,f1,average_precision] in CE6 order"
            )
        values = row[1:5]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("Morphology class metrics contain a non-finite value")
        support, precision, recall, f1 = (float(value) for value in values)
        if support < 0 or any(not 0 <= value <= 1 for value in (precision, recall, f1)):
            raise ValueError("Morphology class support/rate range is invalid")
        average_precision = row[5]
        if support == 0:
            if average_precision is not None:
                raise ValueError("Unsupported morphology classes require AP=null")
        elif (
            isinstance(average_precision, bool)
            or not isinstance(average_precision, (int, float))
            or not math.isfinite(float(average_precision))
            or not 0 <= float(average_precision) <= 1
        ):
            raise ValueError("Supported morphology classes require finite AP in [0,1]")


@dataclass(frozen=True)
class LoadedMorphologyCheckpoint:
    path: Path
    manifest_sha256: str
    state_sha256: str
    training_run_sha256: str
    evaluation_sha256: str
    foundation_feature_receipt: LaBraMFeatureReceipt
    head: MorphologyEvidenceHead
    training_run_payload: dict[str, object]
    evaluation_payload: dict[str, object]


def load_morphology_checkpoint(
    path: str | Path,
    fold_manifest: TUEVMorphologyManifest,
    *,
    expected_manifest_sha256: str | None = None,
    expected_master_manifest_sha256: str | None = None,
    expected_master_token_corpus_index_sha256: str | None = None,
) -> LoadedMorphologyCheckpoint:
    """Strictly load weights and reject roster/source/corpus substitution."""

    if not isinstance(fold_manifest, TUEVMorphologyManifest) or fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Checkpoint loader requires its exact fold-specific manifest")
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology checkpoint must be a canonical regular directory")
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Morphology checkpoint lacks a regular manifest.json")
    raw_manifest = manifest_path.read_bytes()
    if not 1 <= len(raw_manifest) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Morphology checkpoint manifest size is invalid")
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != _sha(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Morphology checkpoint manifest SHA-256 mismatch")
    manifest = _parse(raw_manifest, label="checkpoint manifest")
    _exact(manifest, _MANIFEST_FIELDS, label="checkpoint manifest")
    if (
        manifest["schema_version"] != MORPHOLOGY_CHECKPOINT_SCHEMA
        or manifest["serialization"]
        != "canonical_json_and_safe_tensors_no_pickle"
        or manifest["model_class"] != "MorphologyEvidenceHead"
    ):
        raise ValueError("Morphology checkpoint schema/model boundary mismatch")
    if manifest["fold_manifest_sha256"] != fold_manifest.manifest_sha256:
        raise ValueError("Morphology checkpoint belongs to another fold manifest")
    expected_fit_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="fit"
    )
    expected_held_groups = morphology_target_bearing_group_ids(
        fold_manifest, role="held"
    )
    if manifest["fit_group_ids"] != list(expected_fit_groups) or manifest[
        "held_group_ids"
    ] != list(expected_held_groups):
        raise ValueError("Morphology checkpoint fit/held roster substitution")
    if manifest["preprocessing_receipt_roster_sha256"] != _preprocessing_roster_sha256(
        fold_manifest
    ):
        raise ValueError("Morphology preprocessing receipt roster changed")
    if expected_master_manifest_sha256 is not None and manifest[
        "master_manifest_sha256"
    ] != _sha(expected_master_manifest_sha256, field="expected_master_manifest_sha256"):
        raise ValueError("Morphology master manifest substitution")
    if expected_master_token_corpus_index_sha256 is not None and manifest[
        "master_token_corpus_index_sha256"
    ] != _sha(
        expected_master_token_corpus_index_sha256,
        field="expected_master_token_corpus_index_sha256",
    ):
        raise ValueError("Morphology master token-corpus substitution")
    if manifest["routing_policy_sha256"] != MORPHOLOGY_TYPED_ROUTING_POLICY_SHA256:
        raise ValueError("Morphology typed-routing policy hash changed")

    state_format = manifest["state_format"]
    if not isinstance(state_format, str):
        raise TypeError("state_format must be a string")
    expected_state_file = _STATE_FILE_BY_FORMAT.get(state_format)
    if expected_state_file is None or manifest["state_file"] != expected_state_file:
        raise ValueError("Morphology state format/file pair is invalid")
    expected_files = {
        "manifest.json",
        expected_state_file,
        "training_run.json",
        "evaluation.json",
        "routing_policy.json",
    }
    if {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("Morphology checkpoint contains missing or unknown files")
    state_path = source / expected_state_file
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("Morphology checkpoint state must be a regular file")
    state_size = _integer(manifest["state_file_size_bytes"], field="state_file_size_bytes", minimum=1)
    if state_size > _MAX_STATE_BYTES or state_path.stat().st_size != state_size:
        raise ValueError("Morphology checkpoint state size mismatch")
    if _file_sha256(state_path) != _sha(manifest["state_file_sha256"], field="state_file_sha256"):
        raise ValueError("Morphology checkpoint state file SHA mismatch")
    arrays = _read_state(state_path, state_format)
    if manifest["state_names"] != sorted(arrays) or manifest["state_specs"] != _specs(arrays):
        raise ValueError("Morphology checkpoint state roster/spec mismatch")
    state_sha = _state_sha256(arrays)
    if state_sha != _sha(manifest["state_payload_sha256"], field="state_payload_sha256"):
        raise ValueError("Morphology checkpoint state payload SHA mismatch")

    run = _read_bound_receipt(
        source,
        manifest,
        file_field="training_run_file",
        sha_field="training_run_sha256",
        size_field="training_run_size_bytes",
        expected_filename="training_run.json",
        expected_fields=_RUN_FIELDS,
        label="training_run.json",
    )
    evaluation = _read_bound_receipt(
        source,
        manifest,
        file_field="evaluation_file",
        sha_field="evaluation_sha256",
        size_field="evaluation_size_bytes",
        expected_filename="evaluation.json",
        expected_fields=_EVALUATION_FIELDS,
        label="evaluation.json",
    )
    routing = _read_bound_receipt(
        source,
        manifest,
        file_field="routing_policy_file",
        sha_field="routing_policy_sha256",
        size_field="routing_policy_size_bytes",
        expected_filename="routing_policy.json",
        expected_fields=frozenset(MORPHOLOGY_TYPED_ROUTING_POLICY),
        label="routing_policy.json",
    )
    if routing != MORPHOLOGY_TYPED_ROUTING_POLICY:
        raise ValueError("Morphology typed-routing policy bytes changed")
    _validate_run_payload(run, manifest)
    _validate_evaluation_payload(evaluation)
    if evaluation["fold_manifest_sha256"] != manifest["fold_manifest_sha256"] or evaluation[
        "master_token_corpus_index_sha256"
    ] != manifest["master_token_corpus_index_sha256"]:
        raise ValueError("Morphology evaluation source binding changed")
    if evaluation["checkpoint_or_run_sha256"] != manifest["training_run_sha256"]:
        raise ValueError("Morphology evaluation is not bound to the training run")

    foundation_payload = manifest["foundation_feature_receipt"]
    if not isinstance(foundation_payload, dict):
        raise TypeError("foundation_feature_receipt must be an object")
    foundation = _foundation_from_payload(foundation_payload)
    foundation_sha = morphology_foundation_receipt_sha256(foundation)
    if foundation_sha != manifest["foundation_feature_receipt_sha256"] or foundation.checkpoint_sha256 != manifest[
        "foundation_checkpoint_sha256"
    ]:
        raise ValueError("Morphology checkpoint foundation binding changed")
    token_dim = _integer(manifest["token_dim"], field="token_dim", minimum=1)
    hidden_dim = _integer(manifest["hidden_dim"], field="hidden_dim", minimum=1)
    if token_dim != 200:
        raise ValueError("Morphology checkpoint token dimension is not audited LaBraM")
    head = MorphologyEvidenceHead(token_dim=token_dim, hidden_dim=hidden_dim)
    expected_arrays = _state_arrays(head)
    if set(arrays) != set(expected_arrays):
        raise ValueError("Morphology checkpoint state keys do not match the head")
    state: dict[str, torch.Tensor] = {}
    for name, expected in expected_arrays.items():
        array = np.asarray(arrays[name])
        if array.shape != expected.shape or array.dtype != expected.dtype:
            raise ValueError(f"Morphology checkpoint state spec changed for {name}")
        state[name] = torch.from_numpy(np.array(array, copy=True))
    head.load_state_dict(state, strict=True)
    return LoadedMorphologyCheckpoint(
        path=source,
        manifest_sha256=manifest_sha,
        state_sha256=state_sha,
        training_run_sha256=manifest["training_run_sha256"],
        evaluation_sha256=manifest["evaluation_sha256"],
        foundation_feature_receipt=foundation,
        head=head,
        training_run_payload=run,
        evaluation_payload=evaluation,
    )


__all__ = [
    "LoadedMorphologyCheckpoint",
    "MORPHOLOGY_CHECKPOINT_SCHEMA",
    "MorphologyCheckpointArtifact",
    "load_morphology_checkpoint",
    "save_morphology_checkpoint",
]
