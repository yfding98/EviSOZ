"""Versioned, non-pickle serialization for one-event evidence caches.

Each cache is an atomically published directory containing a numeric tensor
file, the complete :class:`EvidenceCacheReceipt` JSON, and a manifest that
binds both files by SHA-256.  The tensor payload has a closed schema: only the
finite typed evidence bottleneck and its explicit masks are accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

import numpy as np
import torch

from .data.batching import EvidenceEvent
from .data.provenance import (
    ConceptExtractorReceipt,
    EVIDENCE_CACHE_SCHEMA,
    EventTemporalProvenanceReceipt,
    EvidenceCacheReceipt,
    evidence_batch_sha256,
)
from .evidence import EvidenceBatch
from .evidence_schema import (
    EVIDENCE_TENSOR_SEMANTICS_SHA256,
    evidence_tensor_semantics_payload,
    require_current_evidence_semantics,
    validate_typed_edge_cache,
)
from .geometry import (
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    N_STANDARD_CHANNELS,
    N_TCP_EDGES,
    N_TIME_TILES,
)

try:  # safetensors is preferred; NPZ remains a portable safe fallback.
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _load_safetensors = None
    _save_safetensors = None


EVIDENCE_BUNDLE_SCHEMA = "soz_evidence_bundle_v4"
_TENSOR_NAMES = (
    "node",
    "edge",
    "node_mask",
    "edge_mask",
    "physical_signal_mask",
    "ictal_phase_mask",
    "morphology_mask",
    "morphology_context_mask",
    "ictal_mask",
)
_FLOAT_TENSORS = frozenset({"node", "edge"})
_MASK_TENSORS = frozenset(set(_TENSOR_NAMES) - _FLOAT_TENSORS)
_RECEIPT_FIELDS = frozenset(
    {
        "event_id",
        "event_registry_sha256",
        "event_record_sha256",
        "evidence_sha256",
        "evidence_semantics_sha256",
        "extractors",
        "authorization_sha256",
        "temporal_provenance",
        "schema_version",
    }
)
_TEMPORAL_PROVENANCE_FIELDS = frozenset(
    {
        "event_id",
        "global_timeline_receipt_sha256",
        "temporal_phase_policy_sha256",
        "ictal_phase_mask_sha256",
        "offset_trustworthy",
        "seizure_duration_sec",
        "previous_timeline_trustworthy",
        "has_previous_seizure",
        "previous_seizure_overlap",
        "previous_seizure_gap_sec",
        "current_offset_semantics",
        "pre_anchor_semantics",
        "gap_usage",
        "schema_version",
    }
)
_EXTRACTOR_FIELDS = frozenset(
    {
        "concept_family",
        "checkpoint_sha256",
        "scaler_sha256",
        "split_manifest_sha256",
        "oof_fold",
        "training_target_patient_ids",
        "held_out_target_patient_ids",
        "training_target_roster_sha256",
        "held_out_target_roster_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "tensor_format",
        "tensor_file",
        "receipt_file",
        "tensor_content_sha256",
        "tensor_specs",
        "tensor_semantics",
        "tensor_semantics_sha256",
        "files",
    }
)
_FILE_RECORD_FIELDS = frozenset({"sha256", "size_bytes"})
_TENSOR_SPEC_FIELDS = frozenset({"shape", "dtype"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_BUNDLE_FILE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class EvidenceCacheArtifact:
    """External hash receipt returned after atomic cache publication."""

    path: Path
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")


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


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if raw != _canonical_json(payload):
        raise ValueError(f"{label} must use canonical JSON encoding")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


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


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "node": (1, N_STANDARD_CHANNELS, N_TIME_TILES, N_NODE_FEATURES),
        "edge": (1, N_TCP_EDGES, N_TIME_TILES, N_EDGE_FEATURES),
        "node_mask": (1, N_STANDARD_CHANNELS, N_TIME_TILES),
        "edge_mask": (1, N_TCP_EDGES, N_TIME_TILES),
        "physical_signal_mask": (1, N_STANDARD_CHANNELS, N_TIME_TILES),
        "ictal_phase_mask": (1, N_TIME_TILES),
        "morphology_mask": (1, N_TCP_EDGES, N_TIME_TILES),
        "morphology_context_mask": (1, N_TCP_EDGES, N_TIME_TILES),
        "ictal_mask": (1, N_TCP_EDGES, N_TIME_TILES),
    }


def _expected_dtypes() -> dict[str, str]:
    return {
        "node": "float32",
        "edge": "float32",
        "node_mask": "bool",
        "edge_mask": "bool",
        "physical_signal_mask": "bool",
        "ictal_phase_mask": "bool",
        "morphology_mask": "bool",
        "morphology_context_mask": "bool",
        "ictal_mask": "bool",
    }


def _tensor_specs_payload() -> dict[str, dict[str, object]]:
    shapes = _expected_shapes()
    dtypes = _expected_dtypes()
    return {
        name: {"shape": list(shapes[name]), "dtype": dtypes[name]}
        for name in _TENSOR_NAMES
    }


def _evidence_arrays(evidence: EvidenceBatch) -> dict[str, np.ndarray]:
    if not isinstance(evidence, EvidenceBatch):
        raise TypeError("evidence must be an EvidenceBatch")
    evidence.validate()
    if evidence.batch_size != 1 or evidence.n_tiles != N_TIME_TILES:
        raise ValueError(
            f"Evidence cache requires one event with {N_TIME_TILES} time tiles"
        )
    if evidence.node.requires_grad or evidence.edge.requires_grad:
        raise ValueError("Evidence must be detached before cache serialization")
    validate_typed_edge_cache(
        evidence.edge,
        evidence.morphology_mask,
        evidence.morphology_context_mask,
        evidence.ictal_mask,
        require_zero_masked=True,
    )
    tensors = {
        "node": evidence.node,
        "edge": evidence.edge,
        "node_mask": evidence.node_mask,
        "edge_mask": evidence.edge_mask,
        "physical_signal_mask": evidence.physical_signal_mask,
        "ictal_phase_mask": evidence.ictal_phase_mask,
        "morphology_mask": evidence.morphology_mask,
        "morphology_context_mask": evidence.morphology_context_mask,
        "ictal_mask": evidence.ictal_mask,
    }
    arrays: dict[str, np.ndarray] = {}
    for name in _TENSOR_NAMES:
        tensor = tensors[name]
        if tensor is None:
            raise RuntimeError(f"Evidence mask {name} was not initialized")
        expected_dtype = torch.float32 if name in _FLOAT_TENSORS else torch.bool
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"{name} must use {expected_dtype} for the versioned cache, "
                f"got {tensor.dtype}"
            )
        arrays[name] = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return _validate_arrays(arrays)


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    actual = set(arrays)
    expected = set(_TENSOR_NAMES)
    if actual != expected:
        raise ValueError(
            "Tensor payload fields do not match the closed evidence schema; "
            f"missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}"
        )
    shapes = _expected_shapes()
    dtypes = _expected_dtypes()
    validated: dict[str, np.ndarray] = {}
    for name in _TENSOR_NAMES:
        array = np.asarray(arrays[name])
        if tuple(array.shape) != shapes[name]:
            raise ValueError(
                f"{name} shape mismatch: expected {shapes[name]}, got {array.shape}"
            )
        if str(array.dtype) != dtypes[name]:
            raise TypeError(
                f"{name} dtype mismatch: expected {dtypes[name]}, got {array.dtype}"
            )
        if name in _FLOAT_TENSORS and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite evidence")
        validated[name] = np.ascontiguousarray(array)
    validate_typed_edge_cache(
        torch.from_numpy(validated["edge"]),
        torch.from_numpy(validated["morphology_mask"]),
        torch.from_numpy(validated["morphology_context_mask"]),
        torch.from_numpy(validated["ictal_mask"]),
        require_zero_masked=True,
    )
    return validated


def _receipt_payload(receipt: EvidenceCacheReceipt) -> dict[str, object]:
    return {
        "event_id": receipt.event_id,
        "event_registry_sha256": receipt.event_registry_sha256,
        "event_record_sha256": receipt.event_record_sha256,
        "evidence_sha256": receipt.evidence_sha256,
        "evidence_semantics_sha256": receipt.evidence_semantics_sha256,
        "authorization_sha256": receipt.authorization_sha256,
        "temporal_provenance": (
            None
            if receipt.temporal_provenance is None
            else asdict(receipt.temporal_provenance)
        ),
        "extractors": [
            {
                "concept_family": extractor.concept_family,
                "checkpoint_sha256": extractor.checkpoint_sha256,
                "scaler_sha256": extractor.scaler_sha256,
                "split_manifest_sha256": extractor.split_manifest_sha256,
                "oof_fold": extractor.oof_fold,
                "training_target_patient_ids": list(
                    extractor.training_target_patient_ids
                ),
                "held_out_target_patient_ids": list(
                    extractor.held_out_target_patient_ids
                ),
                "training_target_roster_sha256": (
                    extractor.training_target_roster_sha256
                ),
                "held_out_target_roster_sha256": (
                    extractor.held_out_target_roster_sha256
                ),
            }
            for extractor in receipt.extractors
        ],
        "schema_version": receipt.schema_version,
    }


def _receipt_from_payload(payload: Mapping[str, object]) -> EvidenceCacheReceipt:
    _require_exact_fields(payload, _RECEIPT_FIELDS, label="EvidenceCacheReceipt")
    if payload["schema_version"] != EVIDENCE_CACHE_SCHEMA:
        raise ValueError(
            f"Unsupported evidence receipt schema: {payload['schema_version']!r}"
        )
    evidence_semantics_sha256 = require_current_evidence_semantics(
        payload["evidence_semantics_sha256"]
    )
    raw_temporal = payload["temporal_provenance"]
    temporal_provenance: EventTemporalProvenanceReceipt | None
    if raw_temporal is None:
        temporal_provenance = None
    else:
        if not isinstance(raw_temporal, dict):
            raise TypeError("temporal_provenance must be a JSON object or null")
        _require_exact_fields(
            raw_temporal,
            _TEMPORAL_PROVENANCE_FIELDS,
            label="EventTemporalProvenanceReceipt",
        )
        temporal_provenance = EventTemporalProvenanceReceipt(**raw_temporal)
    raw_extractors = payload["extractors"]
    if not isinstance(raw_extractors, list):
        raise TypeError("EvidenceCacheReceipt extractors must be a JSON list")
    extractors: list[ConceptExtractorReceipt] = []
    for index, raw_extractor in enumerate(raw_extractors):
        if not isinstance(raw_extractor, dict):
            raise TypeError(f"Extractor receipt {index} must be a JSON object")
        _require_exact_fields(
            raw_extractor,
            _EXTRACTOR_FIELDS,
            label=f"ConceptExtractorReceipt[{index}]",
        )
        for roster_field in (
            "training_target_patient_ids",
            "held_out_target_patient_ids",
        ):
            if not isinstance(raw_extractor[roster_field], list) or any(
                not isinstance(value, str) for value in raw_extractor[roster_field]
            ):
                raise TypeError(f"{roster_field} must be a JSON string list")
        extractors.append(
            ConceptExtractorReceipt(
                concept_family=raw_extractor["concept_family"],
                checkpoint_sha256=raw_extractor["checkpoint_sha256"],
                scaler_sha256=raw_extractor["scaler_sha256"],
                split_manifest_sha256=raw_extractor["split_manifest_sha256"],
                oof_fold=raw_extractor["oof_fold"],
                training_target_patient_ids=tuple(
                    raw_extractor["training_target_patient_ids"]
                ),
                held_out_target_patient_ids=tuple(
                    raw_extractor["held_out_target_patient_ids"]
                ),
                training_target_roster_sha256=raw_extractor[
                    "training_target_roster_sha256"
                ],
                held_out_target_roster_sha256=raw_extractor[
                    "held_out_target_roster_sha256"
                ],
            )
        )
    return EvidenceCacheReceipt(
        event_id=payload["event_id"],
        event_registry_sha256=payload["event_registry_sha256"],
        event_record_sha256=payload["event_record_sha256"],
        evidence_sha256=payload["evidence_sha256"],
        extractors=tuple(extractors),
        evidence_semantics_sha256=evidence_semantics_sha256,
        authorization_sha256=payload["authorization_sha256"],
        temporal_provenance=temporal_provenance,
        schema_version=payload["schema_version"],
    )


def _write_tensor_file(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if _save_safetensors is not None:
        _save_safetensors(dict(arrays), str(path))
        return "safetensors"
    with path.open("wb") as handle:  # pragma: no cover - fallback only
        np.savez_compressed(handle, **arrays)
    return "npz"


def _read_tensor_file(path: Path, tensor_format: str) -> dict[str, np.ndarray]:
    if tensor_format == "safetensors":
        if _load_safetensors is None:
            raise RuntimeError("safetensors is required to read this evidence cache")
        return dict(_load_safetensors(str(path)))
    if tensor_format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError(f"Unsupported tensor format: {tensor_format!r}")


def _file_record(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    if size < 1 or size > _MAX_BUNDLE_FILE_BYTES:
        raise ValueError(f"Cache file size is outside the accepted range: {path.name}")
    return {"sha256": _file_sha256(path), "size_bytes": size}


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_evidence_cache(
    path: str | Path,
    evidence: EvidenceBatch,
    receipt: EvidenceCacheReceipt,
) -> EvidenceCacheArtifact:
    """Atomically publish one detached event cache and return its manifest SHA."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Evidence cache target already exists: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    arrays = _evidence_arrays(evidence)
    content_sha256 = evidence_batch_sha256(evidence)
    if receipt.evidence_sha256 != content_sha256:
        raise ValueError("EvidenceCacheReceipt content SHA does not match EvidenceBatch")
    # Round-trip through the strict JSON schema before writing anything.
    receipt_payload = _receipt_payload(receipt)
    receipt_bytes = _canonical_json(receipt_payload)
    if _receipt_from_payload(receipt_payload) != receipt:
        raise ValueError("EvidenceCacheReceipt cannot be losslessly serialized")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(parent))
    )
    try:
        tensor_format = "safetensors" if _save_safetensors is not None else "npz"
        tensor_name = (
            "evidence.safetensors" if tensor_format == "safetensors" else "evidence.npz"
        )
        tensor_path = temporary / tensor_name
        actual_format = _write_tensor_file(tensor_path, arrays)
        if actual_format != tensor_format:
            raise RuntimeError("Tensor serializer changed format unexpectedly")
        receipt_path = temporary / "receipt.json"
        receipt_path.write_bytes(receipt_bytes)
        _fsync_file(tensor_path)
        _fsync_file(receipt_path)

        manifest = {
            "schema_version": EVIDENCE_BUNDLE_SCHEMA,
            "tensor_format": tensor_format,
            "tensor_file": tensor_name,
            "receipt_file": "receipt.json",
            "tensor_content_sha256": content_sha256,
            "tensor_specs": _tensor_specs_payload(),
            "tensor_semantics": evidence_tensor_semantics_payload(),
            "tensor_semantics_sha256": EVIDENCE_TENSOR_SEMANTICS_SHA256,
            "files": {
                tensor_name: _file_record(tensor_path),
                "receipt.json": _file_record(receipt_path),
            },
        }
        manifest_bytes = _canonical_json(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        os.replace(temporary, target)
        _fsync_directory(parent)
        return EvidenceCacheArtifact(
            path=target,
            manifest_sha256=manifest_sha256,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_manifest(
    path: Path, *, expected_manifest_sha256: str | None
) -> tuple[dict[str, object], str]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Evidence cache lacks a regular manifest.json")
    raw = manifest_path.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(raw)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
        if actual_manifest_sha256 != expected:
            raise ValueError("Evidence cache manifest SHA-256 mismatch")
    manifest = _parse_canonical_json(raw, label="manifest.json")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, label="Evidence manifest")
    if manifest["schema_version"] != EVIDENCE_BUNDLE_SCHEMA:
        raise ValueError(
            f"Unsupported evidence bundle schema: {manifest['schema_version']!r}"
        )
    tensor_format = manifest["tensor_format"]
    expected_tensor_name = {
        "safetensors": "evidence.safetensors",
        "npz": "evidence.npz",
    }.get(tensor_format)
    if expected_tensor_name is None or manifest["tensor_file"] != expected_tensor_name:
        raise ValueError("Evidence manifest has an invalid tensor format/file pair")
    if manifest["receipt_file"] != "receipt.json":
        raise ValueError("Evidence manifest has an invalid receipt filename")
    _require_sha256(
        manifest["tensor_content_sha256"], field="tensor_content_sha256"
    )
    semantic_sha256 = require_current_evidence_semantics(
        manifest["tensor_semantics_sha256"]
    )
    if manifest["tensor_semantics"] != evidence_tensor_semantics_payload():
        raise ValueError("Evidence tensor semantics payload changed")
    if semantic_sha256 != EVIDENCE_TENSOR_SEMANTICS_SHA256:
        raise ValueError("Evidence tensor semantics digest changed")

    raw_specs = manifest["tensor_specs"]
    if not isinstance(raw_specs, dict):
        raise TypeError("tensor_specs must be a JSON object")
    if set(raw_specs) != set(_TENSOR_NAMES):
        raise ValueError("tensor_specs do not match the closed evidence schema")
    expected_specs = _tensor_specs_payload()
    for name in _TENSOR_NAMES:
        spec = raw_specs[name]
        if not isinstance(spec, dict):
            raise TypeError(f"Tensor spec {name} must be a JSON object")
        _require_exact_fields(spec, _TENSOR_SPEC_FIELDS, label=f"Tensor spec {name}")
        if spec != expected_specs[name]:
            raise ValueError(f"Tensor spec mismatch for {name}")

    files = manifest["files"]
    if not isinstance(files, dict):
        raise TypeError("Evidence manifest files must be a JSON object")
    expected_files = {expected_tensor_name, "receipt.json"}
    if set(files) != expected_files:
        raise ValueError("Evidence manifest file list is not the closed bundle schema")
    actual_files = {item.name for item in path.iterdir()}
    if actual_files != expected_files | {"manifest.json"}:
        raise ValueError(
            "Evidence cache contains missing or unknown files; "
            f"expected={sorted(expected_files | {'manifest.json'})}, "
            f"actual={sorted(actual_files)}"
        )
    for filename in sorted(expected_files):
        record = files[filename]
        if not isinstance(record, dict):
            raise TypeError(f"File receipt {filename} must be a JSON object")
        _require_exact_fields(
            record, _FILE_RECORD_FIELDS, label=f"File receipt {filename}"
        )
        file_path = path / filename
        if not file_path.is_file() or file_path.is_symlink():
            raise ValueError(f"Cache member is not a regular file: {filename}")
        declared_size = record["size_bytes"]
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise TypeError(f"File size for {filename} must be an integer")
        if declared_size < 1 or declared_size > _MAX_BUNDLE_FILE_BYTES:
            raise ValueError(f"Declared file size is invalid for {filename}")
        if file_path.stat().st_size != declared_size:
            raise ValueError(f"Evidence cache file size mismatch: {filename}")
        expected_sha = _require_sha256(
            record["sha256"], field=f"files.{filename}.sha256"
        )
        if _file_sha256(file_path) != expected_sha:
            raise ValueError(f"Evidence cache file SHA-256 mismatch: {filename}")
    return manifest, actual_manifest_sha256


def load_evidence_cache(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> EvidenceEvent:
    """Validate a cache bundle and return a detached, registry-ready event."""

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Evidence cache must be a regular directory: {source}")
    manifest, _ = _validated_manifest(
        source, expected_manifest_sha256=expected_manifest_sha256
    )
    receipt_path = source / str(manifest["receipt_file"])
    receipt_payload = _parse_canonical_json(
        receipt_path.read_bytes(), label="receipt.json"
    )
    receipt = _receipt_from_payload(receipt_payload)
    if receipt.evidence_sha256 != manifest["tensor_content_sha256"]:
        raise ValueError("Receipt and manifest evidence content SHA mismatch")
    if receipt.evidence_semantics_sha256 != manifest["tensor_semantics_sha256"]:
        raise ValueError("Receipt and manifest evidence semantics mismatch")

    tensor_path = source / str(manifest["tensor_file"])
    arrays = _validate_arrays(
        _read_tensor_file(tensor_path, str(manifest["tensor_format"]))
    )
    evidence = EvidenceBatch(
        node=torch.from_numpy(np.array(arrays["node"], copy=True)),
        edge=torch.from_numpy(np.array(arrays["edge"], copy=True)),
        node_mask=torch.from_numpy(np.array(arrays["node_mask"], copy=True)),
        edge_mask=torch.from_numpy(np.array(arrays["edge_mask"], copy=True)),
        physical_signal_mask=torch.from_numpy(
            np.array(arrays["physical_signal_mask"], copy=True)
        ),
        ictal_phase_mask=torch.from_numpy(
            np.array(arrays["ictal_phase_mask"], copy=True)
        ),
        morphology_mask=torch.from_numpy(
            np.array(arrays["morphology_mask"], copy=True)
        ),
        morphology_context_mask=torch.from_numpy(
            np.array(arrays["morphology_context_mask"], copy=True)
        ),
        ictal_mask=torch.from_numpy(np.array(arrays["ictal_mask"], copy=True)),
    ).detach()
    actual_content_sha256 = evidence_batch_sha256(evidence)
    if actual_content_sha256 != receipt.evidence_sha256:
        raise ValueError("Loaded evidence tensor content SHA mismatch")
    return EvidenceEvent(
        event_id=receipt.event_id,
        evidence=evidence,
        cache_receipt=receipt,
    )


__all__ = [
    "EVIDENCE_BUNDLE_SCHEMA",
    "EvidenceCacheArtifact",
    "load_evidence_cache",
    "save_evidence_cache",
]
