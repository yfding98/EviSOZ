"""Safe, lineage-bound LaBraM tokens for *concept training only*.

This module deliberately defines a cache type that is separate from the SOZ
evidence bottleneck.  A bundle contains exactly one detached LaBraM token
tensor for one EEG event.  It contains no target, SOZ label, raw EEG,
optimizer state, private-dataset metadata, or reasoner-ready evidence.

The cache is an optimization for frozen-foundation concept training.  The
loader therefore returns :class:`LoadedLaBraMConceptTokens`, never an
``EvidenceEvent`` or ``EvidenceBatch``.
"""

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

from .geometry import N_STANDARD_CHANNELS, STANDARD_19, normalize_electrode_name
from .models.labram import LaBraMFeatureReceipt

try:  # safetensors is preferred; NPZ is a non-pickle portability fallback.
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except ImportError:  # pragma: no cover - only exercised in minimal environments
    _load_safetensors = None
    _save_safetensors = None


CONCEPT_TOKEN_BUNDLE_SCHEMA = "soz_labram_concept_token_bundle_v2"
CONCEPT_TOKEN_PURPOSE = "ictal_concept_training_only"
CONCEPT_TOKEN_NAME = "labram_concept_tokens"
CONCEPT_TOKEN_SHAPE = (N_STANDARD_CHANNELS, 60, 200)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_TENSOR_FILE_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_TENSOR_SPEC = {"shape": list(CONCEPT_TOKEN_SHAPE), "dtype": "float32"}
_TENSOR_FILE_BY_FORMAT = {
    "safetensors": "concept_tokens.safetensors",
    "npz": "concept_tokens.npz",
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
_TENSOR_SPEC_FIELDS = frozenset({"shape", "dtype"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "event_id",
        "source_concept_manifest_sha256",
        "event_record_sha256",
        "preprocess_receipt_sha256",
        "foundation_feature_receipt",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "tensor_format",
        "tensor_file",
        "tensor_name",
        "tensor_spec",
        "tensor_sha256",
        "tensor_file_sha256",
        "tensor_file_size_bytes",
    }
)


@dataclass(frozen=True)
class LaBraMConceptTokenArtifact:
    """Hashes returned after atomically publishing a concept-token bundle."""

    path: Path
    manifest_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        _require_sha256(self.tensor_sha256, field="tensor_sha256")


@dataclass(frozen=True)
class LoadedLaBraMConceptTokens:
    """One validated, detached latent for frozen-foundation concept training.

    This type is intentionally not compatible with the SOZ reasoner evidence
    types.  Targets must come from the independently validated concept-data
    adapter at training time; they are never serialized in this cache.
    """

    event_id: str
    source_concept_manifest_sha256: str
    event_record_sha256: str
    preprocess_receipt_sha256: str
    foundation_feature_receipt: LaBraMFeatureReceipt
    foundation_feature_receipt_sha256: str
    foundation_checkpoint_sha256: str
    tensor_sha256: str
    manifest_sha256: str
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        _validate_event_id(self.event_id)
        for field in (
            "source_concept_manifest_sha256",
            "event_record_sha256",
            "preprocess_receipt_sha256",
            "foundation_feature_receipt_sha256",
            "foundation_checkpoint_sha256",
            "tensor_sha256",
            "manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        receipt_payload = _foundation_payload(self.foundation_feature_receipt)
        actual_receipt_sha256 = _sha256_bytes(_canonical_json(receipt_payload))
        if self.foundation_feature_receipt_sha256 != actual_receipt_sha256:
            raise ValueError("Foundation feature receipt SHA-256 mismatch")
        if (
            self.foundation_checkpoint_sha256
            != self.foundation_feature_receipt.checkpoint_sha256
        ):
            raise ValueError("Foundation checkpoint SHA does not match feature receipt")
        array = _token_array(self.tokens)
        if _token_sha256(array) != self.tensor_sha256:
            raise ValueError("Loaded concept-token tensor SHA-256 mismatch")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
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


def _parse_canonical_manifest(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Concept-token manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Concept-token manifest must be a JSON object")
    if raw != _canonical_json(payload):
        raise ValueError("Concept-token manifest must use canonical JSON encoding")
    return payload


def _require_exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the closed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _validate_event_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("event_id must be a string")
    if not value or value != value.strip() or len(value) > 512:
        raise ValueError("event_id must be non-empty, trimmed, and at most 512 chars")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("event_id cannot contain control characters")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _foundation_payload(receipt: LaBraMFeatureReceipt) -> dict[str, object]:
    if not isinstance(receipt, LaBraMFeatureReceipt):
        raise TypeError("foundation_feature_receipt must be LaBraMFeatureReceipt")
    payload = receipt.to_dict()
    payload["semantic_channels"] = list(payload["semantic_channels"])
    payload["position_names"] = list(payload["position_names"])
    payload["position_ids"] = list(payload["position_ids"])
    normalized = _validate_foundation_payload(payload)
    if _foundation_receipt_from_payload(normalized) != receipt:
        raise ValueError("LaBraMFeatureReceipt cannot be losslessly serialized")
    return normalized


def _validate_foundation_payload(payload: Mapping[str, object]) -> dict[str, object]:
    _require_exact_fields(
        payload, _FOUNDATION_FIELDS, label="foundation_feature_receipt"
    )
    normalized = dict(payload)
    for path_field in ("checkpoint_path", "modeling_path"):
        value = normalized[path_field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"foundation_feature_receipt.{path_field} must be a trimmed string"
            )
    normalized["checkpoint_sha256"] = _require_sha256(
        normalized["checkpoint_sha256"],
        field="foundation_feature_receipt.checkpoint_sha256",
    )
    normalized["modeling_sha256"] = _require_sha256(
        normalized["modeling_sha256"],
        field="foundation_feature_receipt.modeling_sha256",
    )
    normalized["encoder_tensor_count"] = _positive_int(
        normalized["encoder_tensor_count"],
        field="foundation_feature_receipt.encoder_tensor_count",
    )

    semantic = normalized["semantic_channels"]
    positions = normalized["position_names"]
    position_ids = normalized["position_ids"]
    if semantic != list(STANDARD_19):
        raise ValueError("Foundation semantic channels must use frozen standard-19")
    if not isinstance(positions, list) or len(positions) != N_STANDARD_CHANNELS:
        raise ValueError("Foundation position_names must align with standard-19")
    for position, semantic_name in zip(positions, STANDARD_19):
        if (
            not isinstance(position, str)
            or position != position.strip().upper()
            or normalize_electrode_name(position) != semantic_name
        ):
            raise ValueError("Foundation position names are semantically misaligned")
    if (
        not isinstance(position_ids, list)
        or len(position_ids) != N_STANDARD_CHANNELS
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in position_ids
        )
        or len(set(position_ids)) != N_STANDARD_CHANNELS
    ):
        raise ValueError("Foundation position IDs must be 19 unique positive integers")

    for field in (
        "tile_seconds",
        "pretraining_window_seconds",
        "samples_per_token",
        "token_dim",
    ):
        normalized[field] = _positive_int(
            normalized[field], field=f"foundation_feature_receipt.{field}"
        )
    if normalized["tile_seconds"] != 4:
        raise ValueError("Concept-token caches require four-second LaBraM calls")
    if normalized["samples_per_token"] != 200:
        raise ValueError("Concept-token caches require 200 samples per token")
    if normalized["token_dim"] != CONCEPT_TOKEN_SHAPE[-1]:
        raise ValueError("Foundation token_dim does not match the cache schema")
    if normalized["pretraining_window_seconds"] < normalized["tile_seconds"]:
        raise ValueError("Foundation tile exceeds the documented pretraining window")
    scale = normalized["input_scale_from_volts"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("Foundation input scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Foundation input scale must be finite and positive")
    normalized["input_scale_from_volts"] = float(scale)
    return normalized


def _foundation_receipt_from_payload(
    payload: Mapping[str, object],
) -> LaBraMFeatureReceipt:
    normalized = _validate_foundation_payload(payload)
    return LaBraMFeatureReceipt(
        checkpoint_path=normalized["checkpoint_path"],
        checkpoint_sha256=normalized["checkpoint_sha256"],
        modeling_path=normalized["modeling_path"],
        modeling_sha256=normalized["modeling_sha256"],
        encoder_tensor_count=normalized["encoder_tensor_count"],
        semantic_channels=tuple(normalized["semantic_channels"]),
        position_names=tuple(normalized["position_names"]),
        position_ids=tuple(normalized["position_ids"]),
        tile_seconds=normalized["tile_seconds"],
        pretraining_window_seconds=normalized["pretraining_window_seconds"],
        samples_per_token=normalized["samples_per_token"],
        token_dim=normalized["token_dim"],
        input_scale_from_volts=normalized["input_scale_from_volts"],
    )


def labram_feature_receipt_sha256(receipt: LaBraMFeatureReceipt) -> str:
    """Return the canonical SHA-256 bound into every concept-token cache."""

    return _sha256_bytes(_canonical_json(_foundation_payload(receipt)))


def _token_array(tokens: torch.Tensor) -> np.ndarray:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.layout != torch.strided:
        raise TypeError("tokens must be a dense strided tensor")
    if tokens.requires_grad or tokens.grad_fn is not None:
        raise ValueError("Concept tokens must be detached before serialization")
    if tuple(tokens.shape) != CONCEPT_TOKEN_SHAPE:
        raise ValueError(
            f"Concept tokens must have shape {CONCEPT_TOKEN_SHAPE}, "
            f"got {tuple(tokens.shape)}"
        )
    if tokens.dtype != torch.float32:
        raise TypeError(f"Concept tokens must use torch.float32, got {tokens.dtype}")
    if not torch.isfinite(tokens).all().item():
        raise ValueError("Concept tokens contain non-finite values")
    return np.ascontiguousarray(tokens.detach().cpu().numpy(), dtype=np.float32)


def _validate_token_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if set(arrays) != {CONCEPT_TOKEN_NAME}:
        raise ValueError(
            "Tensor payload must contain only labram_concept_tokens; "
            f"actual={sorted(arrays)}"
        )
    array = np.asarray(arrays[CONCEPT_TOKEN_NAME])
    if tuple(array.shape) != CONCEPT_TOKEN_SHAPE:
        raise ValueError(
            f"Concept-token shape mismatch: expected {CONCEPT_TOKEN_SHAPE}, "
            f"got {array.shape}"
        )
    if array.dtype != np.dtype("float32"):
        raise TypeError(
            f"Concept-token dtype mismatch: expected float32, got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Concept-token payload contains non-finite values")
    return np.ascontiguousarray(array)


def _token_sha256(array: np.ndarray) -> str:
    array = _validate_token_arrays({CONCEPT_TOKEN_NAME: array})
    header = _canonical_json(
        {
            "tensor_name": CONCEPT_TOKEN_NAME,
            "shape": list(CONCEPT_TOKEN_SHAPE),
            "dtype": "float32",
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def concept_token_sha256(tokens: torch.Tensor) -> str:
    """Hash one detached concept-only token tensor independent of file format."""

    return _token_sha256(_token_array(tokens))


def _write_tensor_file(path: Path, array: np.ndarray) -> str:
    arrays = {CONCEPT_TOKEN_NAME: array}
    if _save_safetensors is not None:
        _save_safetensors(arrays, str(path))
        return "safetensors"
    with path.open("wb") as handle:  # pragma: no cover - fallback only
        np.savez_compressed(handle, **arrays)
    return "npz"


def _read_tensor_file(path: Path, tensor_format: str) -> dict[str, np.ndarray]:
    if tensor_format == "safetensors":
        if _load_safetensors is None:
            raise RuntimeError("safetensors is required to read this token cache")
        return dict(_load_safetensors(str(path)))
    if tensor_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError(f"Unsupported concept-token format: {tensor_format!r}")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_labram_concept_tokens(
    path: str | Path,
    tokens: torch.Tensor,
    *,
    event_id: str,
    source_concept_manifest_sha256: str,
    event_record_sha256: str,
    preprocess_receipt_sha256: str,
    foundation_feature_receipt: LaBraMFeatureReceipt,
) -> LaBraMConceptTokenArtifact:
    """Atomically publish one detached, target-free concept-training latent."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Concept-token cache already exists: {target}")
    event_id = _validate_event_id(event_id)
    source_concept_manifest_sha256 = _require_sha256(
        source_concept_manifest_sha256, field="source_concept_manifest_sha256"
    )
    event_record_sha256 = _require_sha256(
        event_record_sha256, field="event_record_sha256"
    )
    preprocess_receipt_sha256 = _require_sha256(
        preprocess_receipt_sha256, field="preprocess_receipt_sha256"
    )
    foundation_payload = _foundation_payload(foundation_feature_receipt)
    foundation_receipt_sha256 = _sha256_bytes(_canonical_json(foundation_payload))
    foundation_checkpoint_sha256 = foundation_feature_receipt.checkpoint_sha256
    array = _token_array(tokens)
    tensor_sha256 = _token_sha256(array)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(parent)))
    try:
        tensor_format = "safetensors" if _save_safetensors is not None else "npz"
        tensor_name = _TENSOR_FILE_BY_FORMAT[tensor_format]
        tensor_path = temporary / tensor_name
        actual_format = _write_tensor_file(tensor_path, array)
        if actual_format != tensor_format:
            raise RuntimeError("Concept-token serializer changed format unexpectedly")
        _fsync_file(tensor_path)
        tensor_size = tensor_path.stat().st_size
        if tensor_size < 1 or tensor_size > _MAX_TENSOR_FILE_BYTES:
            raise ValueError("Concept-token file size is outside the accepted range")

        manifest = {
            "schema_version": CONCEPT_TOKEN_BUNDLE_SCHEMA,
            "purpose": CONCEPT_TOKEN_PURPOSE,
            "event_id": event_id,
            "source_concept_manifest_sha256": source_concept_manifest_sha256,
            "event_record_sha256": event_record_sha256,
            "preprocess_receipt_sha256": preprocess_receipt_sha256,
            "foundation_feature_receipt": foundation_payload,
            "foundation_feature_receipt_sha256": foundation_receipt_sha256,
            "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
            "tensor_format": tensor_format,
            "tensor_file": tensor_name,
            "tensor_name": CONCEPT_TOKEN_NAME,
            "tensor_spec": _TENSOR_SPEC,
            "tensor_sha256": tensor_sha256,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "tensor_file_size_bytes": tensor_size,
        }
        manifest_bytes = _canonical_json(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        manifest_sha256 = _sha256_bytes(manifest_bytes)

        # The first check gives a clear error; the second narrows the race
        # window while retaining atomic directory publication.
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Concept-token cache already exists: {target}")
        os.rename(temporary, target)
        _fsync_directory(parent)
        return LaBraMConceptTokenArtifact(
            path=target,
            manifest_sha256=manifest_sha256,
            tensor_sha256=tensor_sha256,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_manifest(
    source: Path, *, expected_manifest_sha256: str | None
) -> tuple[dict[str, object], LaBraMFeatureReceipt, str]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Concept-token cache lacks a regular manifest.json")
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Concept-token manifest is unreasonably large")
    raw = manifest_path.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(raw)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
        if actual_manifest_sha256 != expected:
            raise ValueError("Concept-token manifest SHA-256 mismatch")
    manifest = _parse_canonical_manifest(raw)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, label="Concept-token manifest")
    if manifest["schema_version"] != CONCEPT_TOKEN_BUNDLE_SCHEMA:
        raise ValueError(
            f"Unsupported concept-token schema: {manifest['schema_version']!r}"
        )
    if manifest["purpose"] != CONCEPT_TOKEN_PURPOSE:
        raise ValueError("Concept-token manifest has an invalid purpose boundary")
    _validate_event_id(manifest["event_id"])
    for field in (
        "source_concept_manifest_sha256",
        "event_record_sha256",
        "preprocess_receipt_sha256",
        "foundation_feature_receipt_sha256",
        "foundation_checkpoint_sha256",
        "tensor_sha256",
        "tensor_file_sha256",
    ):
        _require_sha256(manifest[field], field=field)

    raw_foundation = manifest["foundation_feature_receipt"]
    if not isinstance(raw_foundation, dict):
        raise TypeError("foundation_feature_receipt must be a JSON object")
    foundation_payload = _validate_foundation_payload(raw_foundation)
    foundation_receipt = _foundation_receipt_from_payload(foundation_payload)
    actual_receipt_sha256 = _sha256_bytes(_canonical_json(foundation_payload))
    if manifest["foundation_feature_receipt_sha256"] != actual_receipt_sha256:
        raise ValueError("Foundation feature receipt SHA-256 mismatch")
    if (
        manifest["foundation_checkpoint_sha256"]
        != foundation_receipt.checkpoint_sha256
    ):
        raise ValueError("Foundation checkpoint SHA does not match feature receipt")

    tensor_format = manifest["tensor_format"]
    if not isinstance(tensor_format, str):
        raise TypeError("tensor_format must be a string")
    expected_tensor_file = _TENSOR_FILE_BY_FORMAT.get(tensor_format)
    if expected_tensor_file is None or manifest["tensor_file"] != expected_tensor_file:
        raise ValueError("Concept-token manifest has an invalid format/file pair")
    if manifest["tensor_name"] != CONCEPT_TOKEN_NAME:
        raise ValueError("Concept-token manifest has an invalid tensor name")
    tensor_spec = manifest["tensor_spec"]
    if not isinstance(tensor_spec, dict):
        raise TypeError("tensor_spec must be a JSON object")
    _require_exact_fields(tensor_spec, _TENSOR_SPEC_FIELDS, label="tensor_spec")
    if tensor_spec != _TENSOR_SPEC:
        raise ValueError("Concept-token tensor spec does not match the fixed schema")

    declared_size = manifest["tensor_file_size_bytes"]
    if isinstance(declared_size, bool) or not isinstance(declared_size, int):
        raise TypeError("tensor_file_size_bytes must be an integer")
    if declared_size < 1 or declared_size > _MAX_TENSOR_FILE_BYTES:
        raise ValueError("Declared concept-token file size is invalid")
    expected_files = {"manifest.json", expected_tensor_file}
    actual_files = {entry.name for entry in source.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            "Concept-token cache contains missing or unknown files; "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    tensor_path = source / expected_tensor_file
    if not tensor_path.is_file() or tensor_path.is_symlink():
        raise ValueError("Concept-token payload must be a regular file")
    if tensor_path.stat().st_size != declared_size:
        raise ValueError("Concept-token file size mismatch")
    if _file_sha256(tensor_path) != manifest["tensor_file_sha256"]:
        raise ValueError("Concept-token file SHA-256 mismatch")
    return manifest, foundation_receipt, actual_manifest_sha256


def load_labram_concept_tokens(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedLaBraMConceptTokens:
    """Load one validated concept-only latent without constructing SOZ evidence."""

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Concept-token cache must be a regular directory: {source}")
    manifest, foundation_receipt, manifest_sha256 = _validated_manifest(
        source, expected_manifest_sha256=expected_manifest_sha256
    )
    tensor_path = source / str(manifest["tensor_file"])
    array = _validate_token_arrays(
        _read_tensor_file(tensor_path, str(manifest["tensor_format"]))
    )
    actual_tensor_sha256 = _token_sha256(array)
    if actual_tensor_sha256 != manifest["tensor_sha256"]:
        raise ValueError("Concept-token tensor SHA-256 mismatch")
    tokens = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).detach()
    return LoadedLaBraMConceptTokens(
        event_id=manifest["event_id"],
        source_concept_manifest_sha256=manifest[
            "source_concept_manifest_sha256"
        ],
        event_record_sha256=manifest["event_record_sha256"],
        preprocess_receipt_sha256=manifest["preprocess_receipt_sha256"],
        foundation_feature_receipt=foundation_receipt,
        foundation_feature_receipt_sha256=manifest[
            "foundation_feature_receipt_sha256"
        ],
        foundation_checkpoint_sha256=manifest["foundation_checkpoint_sha256"],
        tensor_sha256=manifest["tensor_sha256"],
        manifest_sha256=manifest_sha256,
        tokens=tokens,
    )


__all__ = [
    "CONCEPT_TOKEN_BUNDLE_SCHEMA",
    "CONCEPT_TOKEN_NAME",
    "CONCEPT_TOKEN_PURPOSE",
    "CONCEPT_TOKEN_SHAPE",
    "LaBraMConceptTokenArtifact",
    "LoadedLaBraMConceptTokens",
    "concept_token_sha256",
    "labram_feature_receipt_sha256",
    "load_labram_concept_tokens",
    "save_labram_concept_tokens",
]
