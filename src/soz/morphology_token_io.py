"""Safe, purpose-separated LaBraM caches for morphology.

Two incompatible artifacts are defined:

* one master TUEV *interval-group* token, ``[19,4,200]``, computed once per
  unique annotation-aligned crop and shared by all fold manifests; and
* one deployment/OOF event token, ``[19,57,200]``, produced by 57 overlapping
  four-second calls at one-second stride while reading slot zero only.

Neither artifact accepts the ictal cache shape ``[19,60,200]`` or its purpose.
Labels, SOZ targets, private metadata, and optimizer state are never stored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from .data.tuev_morphology import (
    FOLD_COUNT_SEMANTICS,
    HOLDING_COUNT_SEMANTICS,
    TUEVMorphologyIntervalGroup,
    TUEVMorphologyManifest,
    TUEVMorphologyRecordReceipt,
)
from .geometry import N_STANDARD_CHANNELS, N_TCP_EDGES, STANDARD_19
from .models.labram import (
    AUDITED_ENCODER_TENSOR_COUNT,
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_LEGACY_POSITION_NAMES,
    LABRAM_POSITION_ID_BY_NAME,
    LaBraMFeatureReceipt,
)
from .morphology_features import (
    MORPHOLOGY_ANCHOR_COUNT,
    MORPHOLOGY_CONTEXT_SECONDS,
    MORPHOLOGY_READ_SLOT,
    MORPHOLOGY_SAMPLES_PER_SECOND,
    MORPHOLOGY_STRIDE_SECONDS,
    MORPHOLOGY_TILE_COUNT,
    MORPHOLOGY_TOKEN_DIM,
    MORPHOLOGY_WINDOW_SECONDS,
    MorphologyDeploymentMasks,
    morphology_deployment_masks,
)

try:
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except ImportError:  # pragma: no cover - portability fallback
    _load_safetensors = None
    _save_safetensors = None


MORPHOLOGY_TRAINING_GROUP_BUNDLE_SCHEMA = (
    "soz_morphology_training_group_token_bundle_v1"
)
MORPHOLOGY_DEPLOYMENT_BUNDLE_SCHEMA = "soz_morphology_deployment_token_bundle_v2"
MORPHOLOGY_TRAINING_CORPUS_SCHEMA = "soz_morphology_master_token_corpus_v1"

MORPHOLOGY_TRAINING_GROUP_PURPOSE = "tuev_morphology_interval_group_training_only"
MORPHOLOGY_DEPLOYMENT_PURPOSE = "morphology_oof_evidence_slot0_stride1_only"
MORPHOLOGY_TRAINING_CORPUS_PURPOSE = "master_tuev_morphology_group_tokens_once"

MORPHOLOGY_TRAINING_TOKEN_NAME = "labram_morphology_group_tokens"
MORPHOLOGY_DEPLOYMENT_TOKEN_NAME = "labram_morphology_deployment_tokens"
MORPHOLOGY_SECOND_MASK_NAME = "second_available_mask"
MORPHOLOGY_PHASE_MASK_NAME = "phase_tile_mask"

MORPHOLOGY_TRAINING_TOKEN_SHAPE = (
    N_STANDARD_CHANNELS,
    MORPHOLOGY_CONTEXT_SECONDS,
    MORPHOLOGY_TOKEN_DIM,
)
MORPHOLOGY_DEPLOYMENT_TOKEN_SHAPE = (
    N_STANDARD_CHANNELS,
    MORPHOLOGY_ANCHOR_COUNT,
    MORPHOLOGY_TOKEN_DIM,
)
MORPHOLOGY_SECOND_MASK_SHAPE = (N_TCP_EDGES, MORPHOLOGY_WINDOW_SECONDS)
MORPHOLOGY_PHASE_MASK_SHAPE = (N_TCP_EDGES, MORPHOLOGY_TILE_COUNT)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_TENSOR_FILE_BYTES = 8 * 1024 * 1024
_MAX_CORPUS_INDEX_BYTES = 128 * 1024 * 1024
_TENSOR_FILE_BY_FORMAT = {
    "safetensors": "morphology_tokens.safetensors",
    "npz": "morphology_tokens.npz",
}
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
_TRAINING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "crop_id",
        "record_id",
        "parent_group_id",
        "parent_group_receipt_sha256",
        "source_morphology_manifest_sha256",
        "event_record_sha256",
        "preprocess_receipt_sha256",
        "target_start_sample",
        "source_target_mask_sha256",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "context_seconds",
        "call_count",
        "read_slot",
        "tensor_format",
        "tensor_file",
        "tensor_names",
        "tensor_specs",
        "tensor_payload_sha256",
        "tensor_file_sha256",
        "tensor_file_size_bytes",
    }
)
_DEPLOYMENT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "event_id",
        "record_id",
        "target_patient_id",
        "public_patient_id",
        "model_split",
        "relative_edf_path",
        "global_event_index",
        "global_t0_sec",
        "global_stop_sec",
        "timeline_context_receipt_sha256",
        "signal_preflight_artifact_sha256",
        "signal_preflight_receipt_sha256",
        "event_record_sha256",
        "edf_sha256",
        "edf_receipt_sha256",
        "signal_receipt_sha256",
        "processed_window_sha256",
        "preprocess_config_sha256",
        "window_start_sec",
        "window_stop_sec",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "soz_labels_used",
        "private_labels_used",
        "source_annotation_coverage_used",
        "window_seconds",
        "context_seconds",
        "stride_seconds",
        "call_count",
        "read_slot",
        "tensor_format",
        "tensor_file",
        "tensor_names",
        "tensor_specs",
        "tensor_payload_sha256",
        "tensor_file_sha256",
        "tensor_file_size_bytes",
    }
)
_TENSOR_SPEC_FIELDS = frozenset({"shape", "dtype"})
_CORPUS_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "serialization",
        "source_morphology_manifest_sha256",
        "foundation_feature_receipt_sha256",
        "crop_count",
        "crop_roster_sha256",
        "tensor_roster_sha256",
        "entries",
    }
)
_CORPUS_ENTRY_FIELDS = frozenset(
    {"crop_id", "relative_bundle_path", "bundle_manifest_sha256", "tensor_sha256"}
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_json(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} contains invalid characters")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _relative(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a canonical relative path")
    if path.as_posix() != text:
        raise ValueError(f"{field} must use POSIX separators")
    return text


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


def _foundation_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("foundation_feature_receipt must be LaBraMFeatureReceipt")
    raw = receipt.to_dict()
    raw["semantic_channels"] = list(receipt.semantic_channels)
    raw["position_names"] = list(receipt.position_names)
    raw["position_ids"] = list(receipt.position_ids)
    return _validate_foundation_payload(raw)


def _validate_foundation_payload(value: Mapping[str, object]) -> dict[str, object]:
    _require_exact_fields(value, _FOUNDATION_FIELDS, label="foundation_feature_receipt")
    payload = dict(value)
    for field in ("checkpoint_path", "modeling_path"):
        _text(payload[field], field=f"foundation.{field}")
    for field in ("checkpoint_sha256", "modeling_sha256"):
        _sha(payload[field], field=f"foundation.{field}")
    _integer(payload["encoder_tensor_count"], field="foundation.encoder_tensor_count", minimum=1)
    if payload["checkpoint_sha256"] != AUDITED_LABRAM_BASE_SHA256:
        raise ValueError("Morphology requires the audited official LaBraM-Base checkpoint")
    if payload["modeling_sha256"] != AUDITED_LABRAM_MODELING_SHA256:
        raise ValueError("Morphology requires the audited LaBraM modeling source")
    if payload["encoder_tensor_count"] != AUDITED_ENCODER_TENSOR_COUNT:
        raise ValueError("Morphology LaBraM encoder tensor count drifted")
    if payload["semantic_channels"] != list(STANDARD_19):
        raise ValueError("Morphology foundation must use frozen standard-19 order")
    if not isinstance(payload["position_names"], list) or len(payload["position_names"]) != N_STANDARD_CHANNELS:
        raise ValueError("Foundation position_names must contain 19 values")
    if not isinstance(payload["position_ids"], list) or len(payload["position_ids"]) != N_STANDARD_CHANNELS:
        raise ValueError("Foundation position_ids must contain 19 values")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in payload["position_ids"]):
        raise TypeError("Foundation position IDs must be positive integers")
    if len(set(payload["position_ids"])) != N_STANDARD_CHANNELS:
        raise ValueError("Foundation position IDs must be unique")
    expected_position_names = list(LABRAM_LEGACY_POSITION_NAMES)
    expected_position_ids = [
        LABRAM_POSITION_ID_BY_NAME[name] for name in LABRAM_LEGACY_POSITION_NAMES
    ]
    if payload["position_names"] != expected_position_names:
        raise ValueError(
            "Morphology foundation position_names must preserve the audited "
            "legacy T3/T4/T5/T6 LaBraM contract"
        )
    if payload["position_ids"] != expected_position_ids:
        raise ValueError("Morphology foundation position IDs disagree with position names")
    for field in ("tile_seconds", "pretraining_window_seconds", "samples_per_token", "token_dim"):
        _integer(payload[field], field=f"foundation.{field}", minimum=1)
    if payload["tile_seconds"] != MORPHOLOGY_CONTEXT_SECONDS:
        raise ValueError("Morphology caches require four-second LaBraM calls")
    if payload["pretraining_window_seconds"] != 8:
        raise ValueError("Morphology requires the audited eight-second pretraining window")
    if payload["samples_per_token"] != MORPHOLOGY_SAMPLES_PER_SECOND:
        raise ValueError("Morphology caches require 200 samples per token")
    if payload["token_dim"] != MORPHOLOGY_TOKEN_DIM:
        raise ValueError("Morphology caches require 200-dimensional tokens")
    scale = payload["input_scale_from_volts"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("Foundation input scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) != 1e4:
        raise ValueError("Morphology LaBraM input scale must be volts x 1e4")
    payload["input_scale_from_volts"] = float(scale)
    return payload


def _foundation_from_payload(value: Mapping[str, object]) -> LaBraMFeatureReceipt:
    payload = _validate_foundation_payload(value)
    return LaBraMFeatureReceipt(
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


def morphology_foundation_receipt_sha256(receipt: LaBraMFeatureReceipt) -> str:
    return hashlib.sha256(_canonical_json(_foundation_payload(receipt))).hexdigest()


def _float32_array(tokens: torch.Tensor, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(tokens, torch.Tensor) or tokens.layout != torch.strided:
        raise TypeError(f"{label} must be a dense torch tensor")
    if tokens.requires_grad or tokens.grad_fn is not None:
        raise ValueError(f"{label} must be detached before serialization")
    if tuple(tokens.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}, got {tuple(tokens.shape)}")
    if tokens.dtype != torch.float32:
        raise TypeError(f"{label} must use torch.float32")
    if not torch.isfinite(tokens).all().item():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(tokens.detach().cpu().numpy(), dtype=np.float32)


def _array_specs(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in sorted(arrays.items())
    }


def _payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
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


def _write_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if _save_safetensors is not None:
        _save_safetensors(dict(arrays), str(path))
        return "safetensors"
    with path.open("wb") as handle:  # pragma: no cover
        np.savez_compressed(handle, **arrays)
    return "npz"


def _read_arrays(path: Path, tensor_format: str) -> dict[str, np.ndarray]:
    if tensor_format == "safetensors":
        if _load_safetensors is None:
            raise RuntimeError("safetensors is required to read this morphology cache")
        return dict(_load_safetensors(str(path)))
    if tensor_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError("Unsupported morphology tensor format")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_bundle(
    path: str | Path,
    *,
    base_manifest: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, str, str]:
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Morphology token bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        tensor_format = "safetensors" if _save_safetensors is not None else "npz"
        tensor_name = _TENSOR_FILE_BY_FORMAT[tensor_format]
        tensor_path = temporary / tensor_name
        actual_format = _write_arrays(tensor_path, arrays)
        if actual_format != tensor_format:
            raise RuntimeError("Morphology tensor serializer changed format")
        _fsync_file(tensor_path)
        size = tensor_path.stat().st_size
        if not 1 <= size <= _MAX_TENSOR_FILE_BYTES:
            raise ValueError("Morphology tensor file size is invalid")
        payload_sha = _payload_sha256(arrays)
        manifest = dict(base_manifest)
        manifest.update(
            {
                "tensor_format": tensor_format,
                "tensor_file": tensor_name,
                "tensor_names": sorted(arrays),
                "tensor_specs": _array_specs(arrays),
                "tensor_payload_sha256": payload_sha,
                "tensor_file_sha256": _file_sha256(tensor_path),
                "tensor_file_size_bytes": size,
            }
        )
        manifest_bytes = _canonical_json(manifest)
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("Morphology token manifest is unexpectedly large")
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Morphology token bundle already exists: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return target, hashlib.sha256(manifest_bytes).hexdigest(), payload_sha
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_bundle(
    path: str | Path,
    *,
    expected_schema: str,
    expected_purpose: str,
    expected_fields: frozenset[str],
    expected_manifest_sha256: str | None,
) -> tuple[Path, dict[str, object], dict[str, np.ndarray], str]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology token bundle must be a canonical regular directory")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Morphology token bundle lacks a regular manifest.json")
    raw = manifest_path.read_bytes()
    if not 1 <= len(raw) <= _MAX_MANIFEST_BYTES:
        raise ValueError("Morphology token manifest size is invalid")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != _sha(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise ValueError("Morphology token manifest SHA-256 mismatch")
    manifest = _parse_canonical_json(raw, label="morphology token manifest")
    _require_exact_fields(manifest, expected_fields, label="morphology token manifest")
    if manifest["schema_version"] != expected_schema or manifest["purpose"] != expected_purpose:
        raise ValueError("Morphology token schema/purpose boundary mismatch")
    tensor_format = manifest["tensor_format"]
    if not isinstance(tensor_format, str):
        raise TypeError("tensor_format must be a string")
    expected_file = _TENSOR_FILE_BY_FORMAT.get(tensor_format)
    if expected_file is None or manifest["tensor_file"] != expected_file:
        raise ValueError("Morphology token format/file pair is invalid")
    actual_files = {item.name for item in source.iterdir()}
    if actual_files != {"manifest.json", expected_file}:
        raise ValueError("Morphology token bundle contains missing or unknown files")
    tensor_path = source / expected_file
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("Morphology tensor payload must be a regular file")
    size = _integer(manifest["tensor_file_size_bytes"], field="tensor_file_size_bytes", minimum=1)
    if size > _MAX_TENSOR_FILE_BYTES or tensor_path.stat().st_size != size:
        raise ValueError("Morphology tensor file size mismatch")
    if _file_sha256(tensor_path) != _sha(manifest["tensor_file_sha256"], field="tensor_file_sha256"):
        raise ValueError("Morphology tensor file SHA-256 mismatch")
    arrays = _read_arrays(tensor_path, tensor_format)
    names = manifest["tensor_names"]
    if not isinstance(names, list) or names != sorted(arrays):
        raise ValueError("Morphology tensor-name roster mismatch")
    specs = manifest["tensor_specs"]
    if not isinstance(specs, dict) or specs != _array_specs(arrays):
        raise ValueError("Morphology tensor specs mismatch")
    payload_sha = _payload_sha256(arrays)
    if payload_sha != _sha(manifest["tensor_payload_sha256"], field="tensor_payload_sha256"):
        raise ValueError("Morphology tensor payload SHA-256 mismatch")
    return source, manifest, arrays, manifest_sha


@dataclass(frozen=True)
class MorphologyTrainingGroupTokenArtifact:
    path: Path
    manifest_sha256: str
    tensor_sha256: str
    crop_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Token artifact path must be absolute")
        _sha(self.manifest_sha256, field="manifest_sha256")
        _sha(self.tensor_sha256, field="tensor_sha256")
        _text(self.crop_id, field="crop_id")


@dataclass(frozen=True)
class LoadedMorphologyTrainingGroupTokens:
    crop_id: str
    record_id: str
    parent_group_id: str
    parent_group_receipt_sha256: str
    source_morphology_manifest_sha256: str
    event_record_sha256: str
    preprocess_receipt_sha256: str
    target_start_sample: int
    source_target_mask_sha256: str
    foundation_feature_receipt: LaBraMFeatureReceipt
    foundation_feature_receipt_sha256: str
    tensor_sha256: str
    manifest_sha256: str
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        _text(self.crop_id, field="crop_id")
        for field in ("record_id", "parent_group_id"):
            _text(getattr(self, field), field=field)
        for field in (
            "parent_group_receipt_sha256",
            "source_morphology_manifest_sha256",
            "event_record_sha256",
            "preprocess_receipt_sha256",
            "source_target_mask_sha256",
            "foundation_feature_receipt_sha256",
            "tensor_sha256",
            "manifest_sha256",
        ):
            _sha(getattr(self, field), field=field)
        _integer(self.target_start_sample, field="target_start_sample")
        if morphology_foundation_receipt_sha256(self.foundation_feature_receipt) != self.foundation_feature_receipt_sha256:
            raise ValueError("Foundation receipt SHA mismatch")
        array = _float32_array(
            self.tokens, shape=MORPHOLOGY_TRAINING_TOKEN_SHAPE, label="training-group tokens"
        )
        if _payload_sha256({MORPHOLOGY_TRAINING_TOKEN_NAME: array}) != self.tensor_sha256:
            raise ValueError("Training-group tensor SHA mismatch")


def _parent_group_receipt_sha256(record: TUEVMorphologyRecordReceipt) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "parent_group_id": record.parent_group_id,
                "group_kind": record.group_kind,
                "official_split": record.official_split,
                "source_subject_id": record.source_subject_id,
                "group_file_roster_sha256": record.group_file_roster_sha256,
            }
        )
    ).hexdigest()


def save_morphology_training_group_tokens(
    path: str | Path,
    tokens: torch.Tensor,
    *,
    interval_group: TUEVMorphologyIntervalGroup,
    record: TUEVMorphologyRecordReceipt,
    source_morphology_manifest_sha256: str,
    foundation_feature_receipt: LaBraMFeatureReceipt,
) -> MorphologyTrainingGroupTokenArtifact:
    """Publish one target-free master token for one unique TUEV signal crop."""

    if not isinstance(interval_group, TUEVMorphologyIntervalGroup) or not isinstance(
        record, TUEVMorphologyRecordReceipt
    ):
        raise TypeError("interval_group/record must come from the formal TUEV manifest")
    if (
        interval_group.record_id != record.record_id
        or interval_group.parent_group_id != record.parent_group_id
        or interval_group.edf_sha256 != record.edf_sha256
    ):
        raise ValueError("Interval group was swapped across its record/parent receipt")
    source_sha = _sha(
        source_morphology_manifest_sha256,
        field="source_morphology_manifest_sha256",
    )
    foundation_payload = _foundation_payload(foundation_feature_receipt)
    array = _float32_array(
        tokens, shape=MORPHOLOGY_TRAINING_TOKEN_SHAPE, label="training-group tokens"
    )
    arrays = {MORPHOLOGY_TRAINING_TOKEN_NAME: array}
    base = {
        "schema_version": MORPHOLOGY_TRAINING_GROUP_BUNDLE_SCHEMA,
        "purpose": MORPHOLOGY_TRAINING_GROUP_PURPOSE,
        "crop_id": interval_group.crop_id,
        "record_id": record.record_id,
        "parent_group_id": record.parent_group_id,
        "parent_group_receipt_sha256": _parent_group_receipt_sha256(record),
        "source_morphology_manifest_sha256": source_sha,
        "event_record_sha256": record.edf_sha256,
        "preprocess_receipt_sha256": record.metadata.preprocessing_receipt_sha256,
        "target_start_sample": interval_group.start_sample,
        "source_target_mask_sha256": interval_group.source_target_mask_sha256,
        "foundation_feature_receipt": foundation_payload,
        "foundation_feature_receipt_sha256": hashlib.sha256(_canonical_json(foundation_payload)).hexdigest(),
        "foundation_checkpoint_sha256": foundation_feature_receipt.checkpoint_sha256,
        "context_seconds": MORPHOLOGY_CONTEXT_SECONDS,
        "call_count": 1,
        "read_slot": MORPHOLOGY_READ_SLOT,
    }
    target, manifest_sha, tensor_sha = _save_bundle(path, base_manifest=base, arrays=arrays)
    return MorphologyTrainingGroupTokenArtifact(
        path=target,
        manifest_sha256=manifest_sha,
        tensor_sha256=tensor_sha,
        crop_id=interval_group.crop_id,
    )


def load_morphology_training_group_tokens(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedMorphologyTrainingGroupTokens:
    _, manifest, arrays, manifest_sha = _load_bundle(
        path,
        expected_schema=MORPHOLOGY_TRAINING_GROUP_BUNDLE_SCHEMA,
        expected_purpose=MORPHOLOGY_TRAINING_GROUP_PURPOSE,
        expected_fields=_TRAINING_MANIFEST_FIELDS,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if manifest["context_seconds"] != MORPHOLOGY_CONTEXT_SECONDS or manifest["call_count"] != 1 or manifest["read_slot"] != MORPHOLOGY_READ_SLOT:
        raise ValueError("Training-group morphology call/read-slot contract drifted")
    if set(arrays) != {MORPHOLOGY_TRAINING_TOKEN_NAME}:
        raise ValueError("Training-group bundle contains an invalid tensor roster")
    array = np.asarray(arrays[MORPHOLOGY_TRAINING_TOKEN_NAME])
    if tuple(array.shape) != MORPHOLOGY_TRAINING_TOKEN_SHAPE or array.dtype != np.dtype("float32") or not np.isfinite(array).all():
        raise ValueError("Training-group token tensor violates [19,4,200] float32")
    foundation_raw = manifest["foundation_feature_receipt"]
    if not isinstance(foundation_raw, dict):
        raise TypeError("foundation_feature_receipt must be an object")
    foundation = _foundation_from_payload(foundation_raw)
    foundation_sha = hashlib.sha256(_canonical_json(_validate_foundation_payload(foundation_raw))).hexdigest()
    if foundation_sha != _sha(manifest["foundation_feature_receipt_sha256"], field="foundation_feature_receipt_sha256"):
        raise ValueError("Foundation receipt SHA mismatch")
    if foundation.checkpoint_sha256 != manifest["foundation_checkpoint_sha256"]:
        raise ValueError("Foundation checkpoint SHA mismatch")
    tokens = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).detach()
    return LoadedMorphologyTrainingGroupTokens(
        crop_id=_text(manifest["crop_id"], field="crop_id"),
        record_id=_text(manifest["record_id"], field="record_id"),
        parent_group_id=_text(manifest["parent_group_id"], field="parent_group_id"),
        parent_group_receipt_sha256=_sha(manifest["parent_group_receipt_sha256"], field="parent_group_receipt_sha256"),
        source_morphology_manifest_sha256=_sha(manifest["source_morphology_manifest_sha256"], field="source_morphology_manifest_sha256"),
        event_record_sha256=_sha(manifest["event_record_sha256"], field="event_record_sha256"),
        preprocess_receipt_sha256=_sha(manifest["preprocess_receipt_sha256"], field="preprocess_receipt_sha256"),
        target_start_sample=_integer(manifest["target_start_sample"], field="target_start_sample"),
        source_target_mask_sha256=_sha(manifest["source_target_mask_sha256"], field="source_target_mask_sha256"),
        foundation_feature_receipt=foundation,
        foundation_feature_receipt_sha256=foundation_sha,
        tensor_sha256=_sha(manifest["tensor_payload_sha256"], field="tensor_payload_sha256"),
        manifest_sha256=manifest_sha,
        tokens=tokens,
    )


@dataclass(frozen=True)
class MorphologyDeploymentTokenArtifact:
    path: Path
    manifest_sha256: str
    tensor_sha256: str
    event_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Deployment token artifact path must be absolute")
        _sha(self.manifest_sha256, field="manifest_sha256")
        _sha(self.tensor_sha256, field="tensor_sha256")
        _text(self.event_id, field="event_id")


@dataclass(frozen=True)
class LoadedMorphologyDeploymentTokens:
    event_id: str
    record_id: str
    target_patient_id: str
    public_patient_id: str
    model_split: str
    relative_edf_path: str
    global_event_index: int
    global_t0_sec: float
    global_stop_sec: float
    timeline_context_receipt_sha256: str
    signal_preflight_artifact_sha256: str
    signal_preflight_receipt_sha256: str
    event_record_sha256: str
    edf_sha256: str
    edf_receipt_sha256: str
    signal_receipt_sha256: str
    processed_window_sha256: str
    preprocess_config_sha256: str
    window_start_sec: float
    window_stop_sec: float
    foundation_feature_receipt: LaBraMFeatureReceipt
    foundation_feature_receipt_sha256: str
    tensor_sha256: str
    manifest_sha256: str
    tokens: torch.Tensor
    second_available_mask: torch.Tensor
    phase_tile_mask: torch.Tensor

    def __post_init__(self) -> None:
        for field in (
            "event_id",
            "record_id",
            "target_patient_id",
            "public_patient_id",
        ):
            _text(getattr(self, field), field=field)
        if self.model_split not in {"source_train", "source_dev"}:
            raise ValueError("Morphology deployment permits source_train/source_dev only")
        relative = PurePosixPath(
            _relative(self.relative_edf_path, field="relative_edf_path")
        )
        expected_official_split = (
            "train" if self.model_split == "source_train" else "dev"
        )
        if (
            len(relative.parts) != 5
            or relative.parts[0] != expected_official_split
            or relative.parts[1] != self.public_patient_id
            or relative.suffix != ".edf"
            or relative.stem != self.record_id
        ):
            raise ValueError(
                "Morphology deployment EDF path disagrees with split/patient/record"
            )
        _integer(self.global_event_index, field="global_event_index")
        if self.event_id != f"{self.record_id}__ev{self.global_event_index:04d}":
            raise ValueError(
                "Morphology deployment event ID disagrees with record/global index"
            )
        start = _finite_float(self.global_t0_sec, field="global_t0_sec")
        stop = _finite_float(self.global_stop_sec, field="global_stop_sec")
        window_start = _finite_float(self.window_start_sec, field="window_start_sec")
        window_stop = _finite_float(self.window_stop_sec, field="window_stop_sec")
        if stop <= start:
            raise ValueError("Morphology deployment global event interval is invalid")
        if (
            abs(window_start - (start - 12.0)) > 1e-6
            or abs(window_stop - (start + 48.0)) > 1e-6
        ):
            raise ValueError("Morphology deployment window must be fixed [-12,+48)")
        for field in (
            "timeline_context_receipt_sha256",
            "signal_preflight_artifact_sha256",
            "signal_preflight_receipt_sha256",
            "event_record_sha256",
            "edf_sha256",
            "edf_receipt_sha256",
            "signal_receipt_sha256",
            "processed_window_sha256",
            "preprocess_config_sha256",
            "foundation_feature_receipt_sha256",
            "tensor_sha256",
            "manifest_sha256",
        ):
            _sha(getattr(self, field), field=field)
        token_array = _float32_array(
            self.tokens,
            shape=MORPHOLOGY_DEPLOYMENT_TOKEN_SHAPE,
            label="deployment morphology tokens",
        )
        if tuple(self.second_available_mask.shape) != MORPHOLOGY_SECOND_MASK_SHAPE or self.second_available_mask.dtype != torch.bool:
            raise ValueError("second_available_mask must be bool [20,60]")
        if tuple(self.phase_tile_mask.shape) != MORPHOLOGY_PHASE_MASK_SHAPE or self.phase_tile_mask.dtype != torch.bool:
            raise ValueError("phase_tile_mask must be bool [20,15]")
        edge_available = self.second_available_mask[:, 0].unsqueeze(0)
        MorphologyDeploymentMasks(
            edge_available_mask=edge_available,
            second_available_mask=self.second_available_mask.unsqueeze(0),
            phase_tile_mask=self.phase_tile_mask.unsqueeze(0),
        )
        arrays = {
            MORPHOLOGY_DEPLOYMENT_TOKEN_NAME: token_array,
            MORPHOLOGY_SECOND_MASK_NAME: np.ascontiguousarray(self.second_available_mask.cpu().numpy(), dtype=np.bool_),
            MORPHOLOGY_PHASE_MASK_NAME: np.ascontiguousarray(self.phase_tile_mask.cpu().numpy(), dtype=np.bool_),
        }
        if _payload_sha256(arrays) != self.tensor_sha256:
            raise ValueError("Deployment morphology tensor SHA mismatch")


def save_morphology_deployment_tokens(
    path: str | Path,
    tokens: torch.Tensor,
    *,
    event_id: str,
    record_id: str,
    target_patient_id: str,
    public_patient_id: str,
    model_split: str,
    relative_edf_path: str,
    global_event_index: int,
    global_t0_sec: float,
    global_stop_sec: float,
    timeline_context_receipt_sha256: str,
    signal_preflight_artifact_sha256: str,
    signal_preflight_receipt_sha256: str,
    event_record_sha256: str,
    edf_sha256: str,
    edf_receipt_sha256: str,
    signal_receipt_sha256: str,
    processed_window_sha256: str,
    preprocess_config_sha256: str,
    window_start_sec: float,
    window_stop_sec: float,
    foundation_feature_receipt: LaBraMFeatureReceipt,
    edge_available_mask: torch.Tensor | None = None,
) -> MorphologyDeploymentTokenArtifact:
    """Publish one 60-second event's 57-anchor, slot-zero morphology latent."""

    token_array = _float32_array(
        tokens,
        shape=MORPHOLOGY_DEPLOYMENT_TOKEN_SHAPE,
        label="deployment morphology tokens",
    )
    if edge_available_mask is not None:
        if tuple(edge_available_mask.shape) != (N_TCP_EDGES,) or edge_available_mask.dtype != torch.bool:
            raise ValueError("edge_available_mask must be bool [20]")
        edge_batch = edge_available_mask.detach().cpu().unsqueeze(0)
    else:
        edge_batch = None
    masks = morphology_deployment_masks(1, edge_available_mask=edge_batch)
    foundation_payload = _foundation_payload(foundation_feature_receipt)
    event_identity = _text(event_id, field="event_id")
    source_record = _text(record_id, field="record_id")
    target_patient = _text(target_patient_id, field="target_patient_id")
    public_patient = _text(public_patient_id, field="public_patient_id")
    split = _text(model_split, field="model_split")
    if split not in {"source_train", "source_dev"}:
        raise ValueError("Morphology deployment permits source_train/source_dev only")
    relative_text = _relative(relative_edf_path, field="relative_edf_path")
    relative = PurePosixPath(relative_text)
    official_split = "train" if split == "source_train" else "dev"
    event_index = _integer(global_event_index, field="global_event_index")
    if (
        len(relative.parts) != 5
        or relative.parts[0] != official_split
        or relative.parts[1] != public_patient
        or relative.suffix != ".edf"
        or relative.stem != source_record
    ):
        raise ValueError(
            "Morphology deployment EDF path disagrees with split/patient/record"
        )
    if event_identity != f"{source_record}__ev{event_index:04d}":
        raise ValueError(
            "Morphology deployment event ID disagrees with record/global index"
        )
    start = _finite_float(global_t0_sec, field="global_t0_sec")
    stop = _finite_float(global_stop_sec, field="global_stop_sec")
    crop_start = _finite_float(window_start_sec, field="window_start_sec")
    crop_stop = _finite_float(window_stop_sec, field="window_stop_sec")
    if stop <= start:
        raise ValueError("Morphology deployment global event interval is invalid")
    if (
        abs(crop_start - (start - 12.0)) > 1e-6
        or abs(crop_stop - (start + 48.0)) > 1e-6
    ):
        raise ValueError("Morphology deployment window must be fixed [-12,+48)")
    arrays = {
        MORPHOLOGY_DEPLOYMENT_TOKEN_NAME: token_array,
        MORPHOLOGY_SECOND_MASK_NAME: np.ascontiguousarray(masks.second_available_mask[0].cpu().numpy(), dtype=np.bool_),
        MORPHOLOGY_PHASE_MASK_NAME: np.ascontiguousarray(masks.phase_tile_mask[0].cpu().numpy(), dtype=np.bool_),
    }
    base = {
        "schema_version": MORPHOLOGY_DEPLOYMENT_BUNDLE_SCHEMA,
        "purpose": MORPHOLOGY_DEPLOYMENT_PURPOSE,
        "event_id": event_identity,
        "record_id": source_record,
        "target_patient_id": target_patient,
        "public_patient_id": public_patient,
        "model_split": split,
        "relative_edf_path": relative_text,
        "global_event_index": event_index,
        "global_t0_sec": start,
        "global_stop_sec": stop,
        "timeline_context_receipt_sha256": _sha(timeline_context_receipt_sha256, field="timeline_context_receipt_sha256"),
        "signal_preflight_artifact_sha256": _sha(signal_preflight_artifact_sha256, field="signal_preflight_artifact_sha256"),
        "signal_preflight_receipt_sha256": _sha(signal_preflight_receipt_sha256, field="signal_preflight_receipt_sha256"),
        "event_record_sha256": _sha(event_record_sha256, field="event_record_sha256"),
        "edf_sha256": _sha(edf_sha256, field="edf_sha256"),
        "edf_receipt_sha256": _sha(edf_receipt_sha256, field="edf_receipt_sha256"),
        "signal_receipt_sha256": _sha(signal_receipt_sha256, field="signal_receipt_sha256"),
        "processed_window_sha256": _sha(processed_window_sha256, field="processed_window_sha256"),
        "preprocess_config_sha256": _sha(preprocess_config_sha256, field="preprocess_config_sha256"),
        "window_start_sec": crop_start,
        "window_stop_sec": crop_stop,
        "foundation_feature_receipt": foundation_payload,
        "foundation_feature_receipt_sha256": hashlib.sha256(_canonical_json(foundation_payload)).hexdigest(),
        "foundation_checkpoint_sha256": foundation_feature_receipt.checkpoint_sha256,
        "soz_labels_used": False,
        "private_labels_used": False,
        "source_annotation_coverage_used": False,
        "window_seconds": MORPHOLOGY_WINDOW_SECONDS,
        "context_seconds": MORPHOLOGY_CONTEXT_SECONDS,
        "stride_seconds": MORPHOLOGY_STRIDE_SECONDS,
        "call_count": MORPHOLOGY_ANCHOR_COUNT,
        "read_slot": MORPHOLOGY_READ_SLOT,
    }
    target, manifest_sha, tensor_sha = _save_bundle(path, base_manifest=base, arrays=arrays)
    return MorphologyDeploymentTokenArtifact(
        path=target,
        manifest_sha256=manifest_sha,
        tensor_sha256=tensor_sha,
        event_id=event_id,
    )


def load_morphology_deployment_tokens(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedMorphologyDeploymentTokens:
    _, manifest, arrays, manifest_sha = _load_bundle(
        path,
        expected_schema=MORPHOLOGY_DEPLOYMENT_BUNDLE_SCHEMA,
        expected_purpose=MORPHOLOGY_DEPLOYMENT_PURPOSE,
        expected_fields=_DEPLOYMENT_MANIFEST_FIELDS,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    expected_contract = (
        MORPHOLOGY_WINDOW_SECONDS,
        MORPHOLOGY_CONTEXT_SECONDS,
        MORPHOLOGY_STRIDE_SECONDS,
        MORPHOLOGY_ANCHOR_COUNT,
        MORPHOLOGY_READ_SLOT,
    )
    observed_contract = tuple(
        manifest[field]
        for field in ("window_seconds", "context_seconds", "stride_seconds", "call_count", "read_slot")
    )
    if observed_contract != expected_contract:
        raise ValueError("Deployment morphology stride/context/read-slot contract drifted")
    if set(arrays) != {
        MORPHOLOGY_DEPLOYMENT_TOKEN_NAME,
        MORPHOLOGY_SECOND_MASK_NAME,
        MORPHOLOGY_PHASE_MASK_NAME,
    }:
        raise ValueError("Deployment morphology bundle has an invalid tensor roster")
    tokens_array = np.asarray(arrays[MORPHOLOGY_DEPLOYMENT_TOKEN_NAME])
    second_array = np.asarray(arrays[MORPHOLOGY_SECOND_MASK_NAME])
    phase_array = np.asarray(arrays[MORPHOLOGY_PHASE_MASK_NAME])
    if tuple(tokens_array.shape) != MORPHOLOGY_DEPLOYMENT_TOKEN_SHAPE or tokens_array.dtype != np.dtype("float32") or not np.isfinite(tokens_array).all():
        raise ValueError("Deployment tokens must be float32 [19,57,200]")
    if tuple(second_array.shape) != MORPHOLOGY_SECOND_MASK_SHAPE or second_array.dtype != np.dtype("bool"):
        raise ValueError("Deployment second mask must be bool [20,60]")
    if tuple(phase_array.shape) != MORPHOLOGY_PHASE_MASK_SHAPE or phase_array.dtype != np.dtype("bool"):
        raise ValueError("Deployment phase mask must be bool [20,15]")
    if any(
        manifest[field] is not False
        for field in (
            "soz_labels_used",
            "private_labels_used",
            "source_annotation_coverage_used",
        )
    ):
        raise ValueError(
            "Morphology deployment tokens must remain target/coverage free"
        )
    foundation_raw = manifest["foundation_feature_receipt"]
    if not isinstance(foundation_raw, dict):
        raise TypeError("foundation_feature_receipt must be an object")
    foundation = _foundation_from_payload(foundation_raw)
    foundation_sha = hashlib.sha256(_canonical_json(_validate_foundation_payload(foundation_raw))).hexdigest()
    if foundation_sha != manifest["foundation_feature_receipt_sha256"] or foundation.checkpoint_sha256 != manifest["foundation_checkpoint_sha256"]:
        raise ValueError("Deployment foundation receipt/checkpoint mismatch")
    return LoadedMorphologyDeploymentTokens(
        event_id=_text(manifest["event_id"], field="event_id"),
        record_id=_text(manifest["record_id"], field="record_id"),
        target_patient_id=_text(manifest["target_patient_id"], field="target_patient_id"),
        public_patient_id=_text(manifest["public_patient_id"], field="public_patient_id"),
        model_split=_text(manifest["model_split"], field="model_split"),
        relative_edf_path=_relative(manifest["relative_edf_path"], field="relative_edf_path"),
        global_event_index=_integer(manifest["global_event_index"], field="global_event_index"),
        global_t0_sec=_finite_float(manifest["global_t0_sec"], field="global_t0_sec"),
        global_stop_sec=_finite_float(manifest["global_stop_sec"], field="global_stop_sec"),
        timeline_context_receipt_sha256=_sha(manifest["timeline_context_receipt_sha256"], field="timeline_context_receipt_sha256"),
        signal_preflight_artifact_sha256=_sha(manifest["signal_preflight_artifact_sha256"], field="signal_preflight_artifact_sha256"),
        signal_preflight_receipt_sha256=_sha(manifest["signal_preflight_receipt_sha256"], field="signal_preflight_receipt_sha256"),
        event_record_sha256=_sha(manifest["event_record_sha256"], field="event_record_sha256"),
        edf_sha256=_sha(manifest["edf_sha256"], field="edf_sha256"),
        edf_receipt_sha256=_sha(manifest["edf_receipt_sha256"], field="edf_receipt_sha256"),
        signal_receipt_sha256=_sha(manifest["signal_receipt_sha256"], field="signal_receipt_sha256"),
        processed_window_sha256=_sha(manifest["processed_window_sha256"], field="processed_window_sha256"),
        preprocess_config_sha256=_sha(manifest["preprocess_config_sha256"], field="preprocess_config_sha256"),
        window_start_sec=_finite_float(manifest["window_start_sec"], field="window_start_sec"),
        window_stop_sec=_finite_float(manifest["window_stop_sec"], field="window_stop_sec"),
        foundation_feature_receipt=foundation,
        foundation_feature_receipt_sha256=foundation_sha,
        tensor_sha256=_sha(manifest["tensor_payload_sha256"], field="tensor_payload_sha256"),
        manifest_sha256=manifest_sha,
        tokens=torch.from_numpy(np.array(tokens_array, dtype=np.float32, copy=True)).detach(),
        second_available_mask=torch.from_numpy(np.array(second_array, dtype=np.bool_, copy=True)),
        phase_tile_mask=torch.from_numpy(np.array(phase_array, dtype=np.bool_, copy=True)),
    )


@dataclass(frozen=True)
class MorphologyTrainingTokenBinding:
    crop_id: str
    bundle_path: Path
    bundle_manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        _text(self.crop_id, field="crop_id")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise ValueError("bundle_path must be an absolute pathlib.Path")
        _sha(self.bundle_manifest_sha256, field="bundle_manifest_sha256")
        _sha(self.tensor_sha256, field="tensor_sha256")


_VERIFIED_MASTER_CORPUS_MARKER = object()


@dataclass(frozen=True, init=False)
class VerifiedMorphologyTrainingTokenCorpus:
    """Opaque attestation issued only by the strict master-corpus loader."""

    path: Path
    index_sha256: str
    source_morphology_manifest_sha256: str
    foundation_feature_receipt_sha256: str
    crop_roster_sha256: str
    tensor_roster_sha256: str
    bindings: tuple[MorphologyTrainingTokenBinding, ...]

    def __init__(
        self,
        *,
        _marker: object,
        path: Path,
        index_sha256: str,
        source_morphology_manifest_sha256: str,
        foundation_feature_receipt_sha256: str,
        crop_roster_sha256: str,
        tensor_roster_sha256: str,
        bindings: Sequence[MorphologyTrainingTokenBinding],
    ) -> None:
        if _marker is not _VERIFIED_MASTER_CORPUS_MARKER:
            raise TypeError(
                "VerifiedMorphologyTrainingTokenCorpus can only be issued by the strict loader"
            )
        values = {
            "path": path,
            "index_sha256": index_sha256,
            "source_morphology_manifest_sha256": source_morphology_manifest_sha256,
            "foundation_feature_receipt_sha256": foundation_feature_receipt_sha256,
            "crop_roster_sha256": crop_roster_sha256,
            "tensor_roster_sha256": tensor_roster_sha256,
            "bindings": tuple(bindings),
        }
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Corpus path must be absolute")
        for field in (
            "index_sha256",
            "source_morphology_manifest_sha256",
            "foundation_feature_receipt_sha256",
            "crop_roster_sha256",
            "tensor_roster_sha256",
        ):
            _sha(values[field], field=field)
        if not values["bindings"] or any(
            not isinstance(item, MorphologyTrainingTokenBinding)
            for item in values["bindings"]
        ):
            raise ValueError("Verified corpus bindings cannot be empty")
        crop_ids = tuple(item.crop_id for item in values["bindings"])
        if crop_ids != tuple(sorted(set(crop_ids))):
            raise ValueError("Verified corpus crop bindings must be unique and sorted")
        for field, value in values.items():
            object.__setattr__(self, field, value)

    @property
    def crop_count(self) -> int:
        return len(self.bindings)

    def binding_for_crop(self, crop_id: str) -> MorphologyTrainingTokenBinding:
        key = _text(crop_id, field="crop_id")
        for binding in self.bindings:
            if binding.crop_id == key:
                return binding
        raise KeyError(key)


@dataclass(frozen=True)
class MorphologyTrainingTokenCorpusArtifact:
    path: Path
    index_sha256: str
    source_morphology_manifest_sha256: str
    crop_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Corpus artifact path must be absolute")
        _sha(self.index_sha256, field="index_sha256")
        _sha(
            self.source_morphology_manifest_sha256,
            field="source_morphology_manifest_sha256",
        )
        _integer(self.crop_count, field="crop_count", minimum=1)


def _crop_directory_name(crop_id: str) -> str:
    return "crop-" + hashlib.sha256(crop_id.encode("utf-8")).hexdigest()


class MorphologyFirstPartyProducerRequiredError(RuntimeError):
    """Formal morphology tokens cannot be supplied as caller-owned tensors."""


def save_morphology_training_token_corpus(
    path: str | Path,
    manifest: TUEVMorphologyManifest,
    tokens_by_crop_id: Mapping[str, torch.Tensor],
    *,
    foundation_feature_receipt: LaBraMFeatureReceipt,
) -> MorphologyTrainingTokenCorpusArtifact:
    """Fail closed until a first-party EDF-to-LaBraM producer is available."""

    del path, manifest, tokens_by_crop_id, foundation_feature_receipt
    raise MorphologyFirstPartyProducerRequiredError(
        "Formal TUEV morphology token publication requires a first-party "
        "EDF->preprocess->crop->LaBraM producer receipt; caller-supplied token "
        "tensors are candidate/test fixtures and cannot be promoted"
    )


def _save_morphology_training_token_corpus_for_testing(
    path: str | Path,
    manifest: TUEVMorphologyManifest,
    tokens_by_crop_id: Mapping[str, torch.Tensor],
    *,
    foundation_feature_receipt: LaBraMFeatureReceipt,
) -> MorphologyTrainingTokenCorpusArtifact:
    """TEST ONLY: build a synthetic corpus from caller-supplied tensors.

    Production callers must use :func:`save_morphology_training_token_corpus`,
    which remains fail-closed until the first-party signal producer exists.
    """

    if not isinstance(manifest, TUEVMorphologyManifest):
        raise TypeError("manifest must be TUEVMorphologyManifest")
    if manifest.count_semantics != HOLDING_COUNT_SEMANTICS:
        raise ValueError("Only a holding/master morphology manifest may own token bytes")
    expected_crop_ids = tuple(group.crop_id for group in manifest.interval_groups)
    if not expected_crop_ids:
        raise ValueError("Cannot publish an empty morphology master token corpus")
    if set(tokens_by_crop_id) != set(expected_crop_ids):
        missing = sorted(set(expected_crop_ids) - set(tokens_by_crop_id))
        unknown = sorted(set(tokens_by_crop_id) - set(expected_crop_ids))
        raise ValueError(
            f"Master token mapping must cover each unique crop exactly once; "
            f"missing={missing}, unknown={unknown}"
        )
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Morphology master token corpus already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        groups_dir = staging / "groups"
        groups_dir.mkdir()
        records = {record.record_id: record for record in manifest.records}
        entries: list[dict[str, object]] = []
        for interval_group in manifest.interval_groups:
            child_name = _crop_directory_name(interval_group.crop_id)
            relative_path = f"groups/{child_name}"
            artifact = save_morphology_training_group_tokens(
                groups_dir / child_name,
                tokens_by_crop_id[interval_group.crop_id],
                interval_group=interval_group,
                record=records[interval_group.record_id],
                source_morphology_manifest_sha256=manifest.manifest_sha256,
                foundation_feature_receipt=foundation_feature_receipt,
            )
            entries.append(
                {
                    "crop_id": interval_group.crop_id,
                    "relative_bundle_path": relative_path,
                    "bundle_manifest_sha256": artifact.manifest_sha256,
                    "tensor_sha256": artifact.tensor_sha256,
                }
            )
        entries.sort(key=lambda item: str(item["crop_id"]))
        crop_roster_sha = hashlib.sha256(
            _canonical_json([entry["crop_id"] for entry in entries])
        ).hexdigest()
        tensor_roster_sha = hashlib.sha256(
            _canonical_json(
                [
                    [entry["crop_id"], entry["tensor_sha256"]]
                    for entry in entries
                ]
            )
        ).hexdigest()
        index = {
            "schema_version": MORPHOLOGY_TRAINING_CORPUS_SCHEMA,
            "purpose": MORPHOLOGY_TRAINING_CORPUS_PURPOSE,
            "serialization": "canonical_json_and_safe_tensors_no_pickle",
            "source_morphology_manifest_sha256": manifest.manifest_sha256,
            "foundation_feature_receipt_sha256": morphology_foundation_receipt_sha256(
                foundation_feature_receipt
            ),
            "crop_count": len(entries),
            "crop_roster_sha256": crop_roster_sha,
            "tensor_roster_sha256": tensor_roster_sha,
            "entries": entries,
        }
        _require_exact_fields(index, _CORPUS_INDEX_FIELDS, label="index.json")
        index_bytes = _canonical_json(index)
        if not 1 <= len(index_bytes) <= _MAX_CORPUS_INDEX_BYTES:
            raise ValueError("Morphology corpus index size is invalid")
        index_path = staging / "index.json"
        index_path.write_bytes(index_bytes)
        _fsync_file(index_path)
        _fsync_directory(groups_dir)
        _fsync_directory(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Morphology master token corpus already exists: {target}"
            )
        os.replace(staging, target)
        _fsync_directory(target.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return MorphologyTrainingTokenCorpusArtifact(
        path=target,
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        source_morphology_manifest_sha256=manifest.manifest_sha256,
        crop_count=len(entries),
    )


def _validate_corpus_entry(value: object, *, index: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError(f"entries[{index}] must be an object")
    _require_exact_fields(value, _CORPUS_ENTRY_FIELDS, label=f"entries[{index}]")
    return {
        "crop_id": _text(value["crop_id"], field=f"entries[{index}].crop_id"),
        "relative_bundle_path": _relative(
            value["relative_bundle_path"],
            field=f"entries[{index}].relative_bundle_path",
        ),
        "bundle_manifest_sha256": _sha(
            value["bundle_manifest_sha256"],
            field=f"entries[{index}].bundle_manifest_sha256",
        ),
        "tensor_sha256": _sha(
            value["tensor_sha256"], field=f"entries[{index}].tensor_sha256"
        ),
    }


def load_morphology_training_token_corpus(
    path: str | Path,
    master_manifest: TUEVMorphologyManifest,
    *,
    expected_index_sha256: str | None = None,
) -> VerifiedMorphologyTrainingTokenCorpus:
    """Refuse legacy/caller-produced corpora at the formal authorization edge."""

    del path, master_manifest, expected_index_sha256
    raise MorphologyFirstPartyProducerRequiredError(
        "No formal TUEV morphology token corpus can be verified before the "
        "first-party EDF->preprocess->crop->LaBraM producer is implemented"
    )


def _load_morphology_training_token_corpus_structural(
    path: str | Path,
    master_manifest: TUEVMorphologyManifest,
    *,
    expected_index_sha256: str | None = None,
) -> VerifiedMorphologyTrainingTokenCorpus:
    """Strictly replay the label-free token-corpus serialization.

    This internal primitive verifies only the safe tensor/index structure and
    its binding to a holding manifest.  It deliberately does *not* authorize
    how the tensors were produced.  The explicit test-only loader below uses
    it for synthetic fixtures; the first-party TUEV producer wraps it only
    after replaying the EDF, preprocessing-selection, crop, and checkpoint
    receipts.
    """

    if not isinstance(master_manifest, TUEVMorphologyManifest):
        raise TypeError("master_manifest must be TUEVMorphologyManifest")
    if master_manifest.count_semantics != HOLDING_COUNT_SEMANTICS:
        raise ValueError("Master token corpus requires a holding/master manifest")
    # This canonical payload contains every source record and interval group.
    # Computing it inside the per-crop loop turns a linear corpus load into
    # thousands of repeated full-manifest serializations.
    master_manifest_sha256 = master_manifest.manifest_sha256
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("Morphology master corpus must be a canonical regular directory")
    if {item.name for item in source.iterdir()} != {"index.json", "groups"}:
        raise ValueError("Morphology master corpus contains missing or unknown root entries")
    groups_dir = source / "groups"
    index_path = source / "index.json"
    if groups_dir.is_symlink() or not groups_dir.is_dir():
        raise ValueError("Morphology master groups path must be a regular directory")
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Morphology master index must be a regular file")
    raw = index_path.read_bytes()
    if not 1 <= len(raw) <= _MAX_CORPUS_INDEX_BYTES:
        raise ValueError("Morphology master index size is invalid")
    index_sha = hashlib.sha256(raw).hexdigest()
    if expected_index_sha256 is not None and index_sha != _sha(
        expected_index_sha256, field="expected_index_sha256"
    ):
        raise ValueError("Morphology master corpus index SHA-256 mismatch")
    index = _parse_canonical_json(raw, label="morphology master index")
    _require_exact_fields(index, _CORPUS_INDEX_FIELDS, label="index.json")
    if (
        index["schema_version"] != MORPHOLOGY_TRAINING_CORPUS_SCHEMA
        or index["purpose"] != MORPHOLOGY_TRAINING_CORPUS_PURPOSE
        or index["serialization"]
        != "canonical_json_and_safe_tensors_no_pickle"
    ):
        raise ValueError("Morphology master corpus schema/purpose boundary mismatch")
    if index["source_morphology_manifest_sha256"] != master_manifest_sha256:
        raise ValueError("Morphology master corpus is bound to another source manifest")
    raw_entries = index["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("Morphology master corpus entries must be an array")
    entries = tuple(
        _validate_corpus_entry(value, index=position)
        for position, value in enumerate(raw_entries)
    )
    declared_count = _integer(index["crop_count"], field="crop_count", minimum=1)
    if len(entries) != declared_count:
        raise ValueError("Morphology master corpus crop count mismatch")
    crop_ids = tuple(entry["crop_id"] for entry in entries)
    expected_crop_ids = tuple(group.crop_id for group in master_manifest.interval_groups)
    if crop_ids != tuple(sorted(set(crop_ids))) or crop_ids != expected_crop_ids:
        raise ValueError("Morphology master crop roster differs from source manifest")
    expected_crop_roster_sha = hashlib.sha256(_canonical_json(list(crop_ids))).hexdigest()
    if expected_crop_roster_sha != _sha(index["crop_roster_sha256"], field="crop_roster_sha256"):
        raise ValueError("Morphology master crop-roster SHA mismatch")
    expected_children = {_crop_directory_name(crop_id) for crop_id in crop_ids}
    actual_children = {item.name for item in groups_dir.iterdir()}
    if actual_children != expected_children or any(item.is_symlink() for item in groups_dir.iterdir()):
        raise ValueError("Morphology master groups roster contains substitutions")
    groups = {group.crop_id: group for group in master_manifest.interval_groups}
    records = {record.record_id: record for record in master_manifest.records}
    record_bindings = {
        record_id: (
            _parent_group_receipt_sha256(record),
            record.edf_sha256,
            record.metadata.preprocessing_receipt_sha256,
        )
        for record_id, record in records.items()
    }
    bindings: list[MorphologyTrainingTokenBinding] = []
    foundation_sha: str | None = None
    for entry in entries:
        expected_relative = f"groups/{_crop_directory_name(entry['crop_id'])}"
        if entry["relative_bundle_path"] != expected_relative:
            raise ValueError("Morphology crop path is not the canonical hash-derived path")
        bundle_path = source / PurePosixPath(entry["relative_bundle_path"])
        if bundle_path.resolve(strict=True).parent != groups_dir:
            raise ValueError("Morphology crop path escapes or traverses its groups directory")
        loaded = load_morphology_training_group_tokens(
            bundle_path,
            expected_manifest_sha256=entry["bundle_manifest_sha256"],
        )
        interval_group = groups[entry["crop_id"]]
        parent_group_receipt_sha256, edf_sha256, preprocessing_receipt_sha256 = (
            record_bindings[interval_group.record_id]
        )
        expected_binding = (
            interval_group.crop_id,
            interval_group.record_id,
            interval_group.parent_group_id,
            parent_group_receipt_sha256,
            master_manifest_sha256,
            edf_sha256,
            preprocessing_receipt_sha256,
            interval_group.start_sample,
            interval_group.source_target_mask_sha256,
        )
        observed_binding = (
            loaded.crop_id,
            loaded.record_id,
            loaded.parent_group_id,
            loaded.parent_group_receipt_sha256,
            loaded.source_morphology_manifest_sha256,
            loaded.event_record_sha256,
            loaded.preprocess_receipt_sha256,
            loaded.target_start_sample,
            loaded.source_target_mask_sha256,
        )
        if observed_binding != expected_binding:
            raise ValueError("Morphology token was swapped across crop/record/group receipts")
        if loaded.tensor_sha256 != entry["tensor_sha256"]:
            raise ValueError("Morphology corpus entry tensor SHA mismatch")
        if foundation_sha is None:
            foundation_sha = loaded.foundation_feature_receipt_sha256
        elif foundation_sha != loaded.foundation_feature_receipt_sha256:
            raise ValueError("Morphology master corpus mixes foundation receipts")
        bindings.append(
            MorphologyTrainingTokenBinding(
                crop_id=loaded.crop_id,
                bundle_path=bundle_path,
                bundle_manifest_sha256=loaded.manifest_sha256,
                tensor_sha256=loaded.tensor_sha256,
            )
        )
    assert foundation_sha is not None
    if foundation_sha != _sha(
        index["foundation_feature_receipt_sha256"],
        field="foundation_feature_receipt_sha256",
    ):
        raise ValueError("Morphology master corpus foundation SHA mismatch")
    expected_tensor_roster_sha = hashlib.sha256(
        _canonical_json([[entry["crop_id"], entry["tensor_sha256"]] for entry in entries])
    ).hexdigest()
    if expected_tensor_roster_sha != _sha(index["tensor_roster_sha256"], field="tensor_roster_sha256"):
        raise ValueError("Morphology master tensor-roster SHA mismatch")
    return VerifiedMorphologyTrainingTokenCorpus(
        _marker=_VERIFIED_MASTER_CORPUS_MARKER,
        path=source,
        index_sha256=index_sha,
        source_morphology_manifest_sha256=master_manifest_sha256,
        foundation_feature_receipt_sha256=foundation_sha,
        crop_roster_sha256=expected_crop_roster_sha,
        tensor_roster_sha256=expected_tensor_roster_sha,
        bindings=tuple(bindings),
    )


def _load_morphology_training_token_corpus_for_testing(
    path: str | Path,
    master_manifest: TUEVMorphologyManifest,
    *,
    expected_index_sha256: str | None = None,
) -> VerifiedMorphologyTrainingTokenCorpus:
    """TEST ONLY: strictly replay a synthetic caller-supplied corpus fixture."""

    return _load_morphology_training_token_corpus_structural(
        path,
        master_manifest,
        expected_index_sha256=expected_index_sha256,
    )


def select_morphology_fold_bindings(
    corpus: VerifiedMorphologyTrainingTokenCorpus,
    fold_manifest: TUEVMorphologyManifest,
    *,
    role: str,
) -> tuple[MorphologyTrainingTokenBinding, ...]:
    """Reference master tokens for a fold without copying or rematerialising them."""

    if not isinstance(corpus, VerifiedMorphologyTrainingTokenCorpus):
        raise TypeError("corpus must be issued by the strict morphology corpus loader")
    if not isinstance(fold_manifest, TUEVMorphologyManifest):
        raise TypeError("fold_manifest must be TUEVMorphologyManifest")
    if fold_manifest.count_semantics != FOLD_COUNT_SEMANTICS:
        raise ValueError("Token selection requires a fold-specific final manifest")
    if fold_manifest.derived_from_manifest_sha256 != (
        corpus.source_morphology_manifest_sha256
    ):
        raise ValueError("Fold manifest derives from another morphology master manifest")
    if role == "fit":
        selected_groups = set(fold_manifest.fit_group_ids)
    elif role == "held":
        selected_groups = set(fold_manifest.held_group_ids)
    else:
        raise ValueError("role must be fit or held")
    crop_ids = tuple(
        group.crop_id
        for group in fold_manifest.interval_groups
        if group.parent_group_id in selected_groups
    )
    if not crop_ids:
        raise ValueError(f"Fold manifest has no {role} morphology interval groups")
    by_crop = {binding.crop_id: binding for binding in corpus.bindings}
    missing = sorted(set(crop_ids) - set(by_crop))
    if missing:
        raise ValueError(f"Fold references crops absent from the master corpus: {missing}")
    selected = tuple(by_crop[crop_id] for crop_id in crop_ids)
    if len({binding.bundle_path for binding in selected}) != len(selected):
        raise ValueError("A fold would reuse one master bundle for multiple crop identities")
    return selected


__all__ = [
    "LoadedMorphologyDeploymentTokens",
    "LoadedMorphologyTrainingGroupTokens",
    "MORPHOLOGY_DEPLOYMENT_BUNDLE_SCHEMA",
    "MORPHOLOGY_DEPLOYMENT_PURPOSE",
    "MORPHOLOGY_DEPLOYMENT_TOKEN_SHAPE",
    "MORPHOLOGY_TRAINING_CORPUS_PURPOSE",
    "MORPHOLOGY_TRAINING_CORPUS_SCHEMA",
    "MORPHOLOGY_TRAINING_GROUP_BUNDLE_SCHEMA",
    "MORPHOLOGY_TRAINING_GROUP_PURPOSE",
    "MORPHOLOGY_TRAINING_TOKEN_SHAPE",
    "MorphologyDeploymentTokenArtifact",
    "MorphologyFirstPartyProducerRequiredError",
    "MorphologyTrainingGroupTokenArtifact",
    "MorphologyTrainingTokenBinding",
    "MorphologyTrainingTokenCorpusArtifact",
    "VerifiedMorphologyTrainingTokenCorpus",
    "load_morphology_deployment_tokens",
    "load_morphology_training_group_tokens",
    "load_morphology_training_token_corpus",
    "morphology_foundation_receipt_sha256",
    "save_morphology_deployment_tokens",
    "save_morphology_training_group_tokens",
    "save_morphology_training_token_corpus",
    "select_morphology_fold_bindings",
]
