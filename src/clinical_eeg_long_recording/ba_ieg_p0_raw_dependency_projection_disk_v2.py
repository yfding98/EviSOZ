"""Append-only disk transport for projection-v2 P0 raw dependencies.

This module is deliberately independent of the segmental disk-v1 tensor and
target formats.  It persists exactly one canonical-JSON provenance artifact:
the projection-v2 raw-sample dependency sidecar plus content bindings to the
host event input, projection, P0 materialization, canonical signal, and the
complete ordered trusted-view registry.  It never contributes an NPZ array,
batch-packing field, model-forward input, target, or raw signal value.

The detached disk reference binds the artifact bytes (size and SHA-256) and
the same host roots.  Loading is fail closed: callers must provide independent
expected event/projection/P0 hashes, the host canonical receipt, and the exact
trusted-view registry; every dependency is then replayed from those roots.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Final, Mapping

from .ba_ieg_event_model_input_projection_v2 import (
    BAIEGEventModelInputProjectionV2,
    validate_ba_ieg_event_model_input_projection_v2,
)
from .canonical_signal_views import validate_canonical_signal_receipt


BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2: Final[
    str
] = "ba_ieg_p0_raw_dependency_projection_disk_artifact_v2"
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V2: Final[
    str
] = "ba_ieg_p0_raw_dependency_projection_disk_reference_v2"
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2: Final[
    str
] = "ba_ieg_projection_v2_raw_dependency_append_only_json_disk_v2"
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2: Final[int] = (
    512 * 1024 * 1024
)

_ARTIFACT_ID_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-disk-artifact-id-v2"
_ARTIFACT_SHA_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-disk-artifact-sha-v2"
_REFERENCE_ID_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-disk-reference-id-v2"
_REFERENCE_SHA_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-disk-reference-sha-v2"
_VIEW_REGISTRY_BINDING_SCHEMA: Final[
    str
] = "ba_ieg_complete_ordered_trusted_view_registry_binding_v2"
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")

_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "canonical_json_provenance_only": True,
    "npz_array_payload_present": False,
    "tensor_payload_present": False,
    "raw_signal_sample_values_present": False,
    "supervision_target_payload_present": False,
    "available_to_model_forward": False,
    "available_to_batch_packing": False,
    "host_canonical_receipt_required_for_load": True,
    "complete_trusted_view_registry_required_for_load": True,
    "host_dependency_replay_required_for_load": True,
    "append_only_no_clobber_publication": True,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
}

_SOURCE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "event_model_input_receipt_sha256",
        "projection_v2_receipt_sha256",
        "source_p0_materialization_receipt_sha256",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "trusted_view_registry_binding_sha256",
        "trusted_view_receipt_sha256s",
        "raw_sample_dependency_sidecar_id",
        "raw_sample_dependency_sidecar_sha256",
        "raw_dependency_roster_sha256",
        "token_count",
        "dependency_count",
    }
)
_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "artifact_id",
        "artifact_sha256",
        "source_binding",
        "raw_sample_dependency_sidecar",
        "scope_receipt",
    }
)
_REFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "reference_id",
        "reference_sha256",
        "relative_path",
        "file_size_bytes",
        "file_sha256",
        "artifact_id",
        "artifact_sha256",
        *_SOURCE_BINDING_KEYS,
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a valid non-empty trimmed identifier")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _relative_json_path(value: object) -> str:
    text = _identifier(value, "relative_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
        or path.suffix != ".json"
        or "\\" in text
    ):
        raise ValueError("relative_path must be a canonical relative .json path")
    return text


def _finalize_artifact(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["artifact_id"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_id"] = (
        "P0-RAWDEP-DISK-"
        + _canonical_sha256({"domain": _ARTIFACT_ID_DOMAIN, "artifact": result})[:24]
    )
    result["artifact_sha256"] = _canonical_sha256(
        {"domain": _ARTIFACT_SHA_DOMAIN, "artifact": result}
    )
    return result


def _finalize_reference(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["reference_id"] = "CONTENT-ADDRESS-PENDING"
    result["reference_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["reference_id"] = (
        "P0-RAWDEP-DISKREF-"
        + _canonical_sha256({"domain": _REFERENCE_ID_DOMAIN, "reference": result})[:24]
    )
    result["reference_sha256"] = _canonical_sha256(
        {"domain": _REFERENCE_SHA_DOMAIN, "reference": result}
    )
    return result


def _validate_embedded_artifact(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _ARTIFACT_KEYS:
        raise ValueError("raw-dependency disk artifact has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2
        or data["method_id"] != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2
        or data["scope_receipt"] != _SCOPE_RECEIPT
    ):
        raise ValueError("raw-dependency disk artifact contract drifted")
    binding = data["source_binding"]
    if type(binding) is not dict or set(binding) != _SOURCE_BINDING_KEYS:
        raise ValueError("raw-dependency disk source binding drifted")
    for name in (
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "canonical_signal_id",
        "raw_sample_dependency_sidecar_id",
    ):
        _identifier(binding[name], name)
    for name in (
        "event_model_input_receipt_sha256",
        "projection_v2_receipt_sha256",
        "source_p0_materialization_receipt_sha256",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "trusted_view_registry_binding_sha256",
        "raw_sample_dependency_sidecar_sha256",
        "raw_dependency_roster_sha256",
    ):
        _sha256(binding[name], name)
    view_hashes = binding["trusted_view_receipt_sha256s"]
    if not isinstance(view_hashes, list) or not view_hashes:
        raise ValueError("trusted view receipt binding must be a non-empty array")
    for index, digest in enumerate(view_hashes):
        _sha256(digest, f"trusted_view_receipt_sha256s[{index}]")
    token_count = _positive_integer(binding["token_count"], "token_count")
    dependency_count = _positive_integer(
        binding["dependency_count"], "dependency_count"
    )
    if token_count != dependency_count:
        raise ValueError("disk artifact lost a token dependency")
    if type(data["raw_sample_dependency_sidecar"]) is not dict:
        raise TypeError("raw dependency sidecar must be a JSON object")
    body = deepcopy(data)
    supplied_id = _identifier(body.pop("artifact_id"), "artifact_id")
    supplied_sha = _sha256(body.pop("artifact_sha256"), "artifact_sha256")
    expected = _finalize_artifact(body)
    if (
        supplied_id != expected["artifact_id"]
        or supplied_sha != expected["artifact_sha256"]
    ):
        raise ValueError("raw-dependency disk artifact content hash drifted")
    return data


def _validate_embedded_reference(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _REFERENCE_KEYS:
        raise ValueError("raw-dependency disk reference has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V2
        or data["method_id"] != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2
    ):
        raise ValueError("raw-dependency disk reference contract drifted")
    _relative_json_path(data["relative_path"])
    _positive_integer(data["file_size_bytes"], "file_size_bytes")
    for name in (
        "file_sha256",
        "artifact_sha256",
        "event_model_input_receipt_sha256",
        "projection_v2_receipt_sha256",
        "source_p0_materialization_receipt_sha256",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "trusted_view_registry_binding_sha256",
        "raw_sample_dependency_sidecar_sha256",
        "raw_dependency_roster_sha256",
    ):
        _sha256(data[name], name)
    for name in (
        "artifact_id",
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "canonical_signal_id",
        "raw_sample_dependency_sidecar_id",
    ):
        _identifier(data[name], name)
    view_hashes = data["trusted_view_receipt_sha256s"]
    if not isinstance(view_hashes, list) or not view_hashes:
        raise ValueError("reference trusted-view binding must be non-empty")
    for index, digest in enumerate(view_hashes):
        _sha256(digest, f"trusted_view_receipt_sha256s[{index}]")
    token_count = _positive_integer(data["token_count"], "token_count")
    dependency_count = _positive_integer(data["dependency_count"], "dependency_count")
    if token_count != dependency_count:
        raise ValueError("disk reference lost a token dependency")
    body = deepcopy(data)
    supplied_id = _identifier(body.pop("reference_id"), "reference_id")
    supplied_sha = _sha256(body.pop("reference_sha256"), "reference_sha256")
    expected = _finalize_reference(body)
    if (
        supplied_id != expected["reference_id"]
        or supplied_sha != expected["reference_sha256"]
    ):
        raise ValueError("raw-dependency disk reference content hash drifted")
    return data


def _trusted_view_registry_binding_sha256(
    projection: BAIEGEventModelInputProjectionV2,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> str:
    event = projection.model_input_event
    if not isinstance(trusted_view_receipts, Mapping):
        raise TypeError("trusted_view_receipts must be a host-supplied mapping")
    if set(trusted_view_receipts) != set(event.view_ids):
        raise ValueError("trusted view registry must exactly cover the event")
    rows: list[dict[str, Any]] = []
    for index, view_id in enumerate(event.view_ids):
        receipt = trusted_view_receipts[view_id]
        if not isinstance(receipt, Mapping) or receipt.get("view_id") != view_id:
            raise ValueError("trusted view registry key/view_id mismatch")
        transform = receipt.get("transform_spec")
        if not isinstance(transform, Mapping):
            raise TypeError("trusted view transform receipt is invalid")
        rows.append(
            {
                "view_index": index,
                "view_id": view_id,
                "view_receipt_sha256": _sha256(
                    receipt.get("receipt_sha256"), "view receipt_sha256"
                ),
                "view_transform_spec_sha256": _sha256(
                    transform.get("transform_spec_sha256"),
                    "view transform_spec_sha256",
                ),
                "processed_view_sha256": _sha256(
                    receipt.get("processed_view_sha256"),
                    "processed_view_sha256",
                ),
            }
        )
    return _canonical_sha256(
        {"schema_version": _VIEW_REGISTRY_BINDING_SCHEMA, "ordered_views": rows}
    )


def _host_binding(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    validate_ba_ieg_event_model_input_projection_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    canonical = validate_canonical_signal_receipt(canonical_signal_receipt)
    event = projection.model_input_event
    raw = projection.raw_sample_dependency_sidecar
    raw_binding = raw["source_binding"]
    token_count = int(event.token_values.shape[0])
    dependency_count = len(raw["dependencies"])
    if token_count < 1 or dependency_count != token_count:
        raise ValueError("projection raw dependencies do not cover every token")
    binding = {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "event_model_input_receipt_sha256": event.input_receipt_sha256,
        "projection_v2_receipt_sha256": projection.receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            projection.source_p0_materialization_receipt_sha256
        ),
        "canonical_signal_id": str(canonical["canonical_signal_id"]),
        "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
        "source_signal_sha256": str(canonical["source_signal_sha256"]),
        "trusted_view_registry_binding_sha256": (
            _trusted_view_registry_binding_sha256(projection, trusted_view_receipts)
        ),
        "trusted_view_receipt_sha256s": [
            str(trusted_view_receipts[view_id]["receipt_sha256"])
            for view_id in event.view_ids
        ],
        "raw_sample_dependency_sidecar_id": str(raw["sidecar_id"]),
        "raw_sample_dependency_sidecar_sha256": str(raw["sidecar_sha256"]),
        "raw_dependency_roster_sha256": str(raw["dependency_roster_sha256"]),
        "token_count": token_count,
        "dependency_count": dependency_count,
    }
    cross_checks = {
        "event_id": raw_binding["event_id"],
        "recording_id": raw_binding["recording_id"],
        "patient_uid": raw_binding["patient_uid"],
        "model_split": raw_binding["model_split"],
        "event_model_input_receipt_sha256": raw_binding[
            "event_model_input_receipt_sha256"
        ],
        "source_p0_materialization_receipt_sha256": raw_binding[
            "source_p0_materialization_receipt_sha256"
        ],
        "canonical_signal_id": raw_binding["canonical_signal_id"],
        "canonical_receipt_sha256": raw_binding["canonical_receipt_sha256"],
        "source_signal_sha256": raw_binding["source_signal_sha256"],
        "trusted_view_receipt_sha256s": raw_binding["trusted_view_receipt_sha256s"],
        "token_count": raw_binding["token_count"],
    }
    for name, value in cross_checks.items():
        if binding[name] != value:
            raise ValueError(f"projection/raw disk binding {name} drifted")
    return binding


def materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Build the target-free canonical-JSON provenance artifact in memory."""

    binding = _host_binding(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    artifact = _finalize_artifact(
        {
            "schema_version": (
                BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2
            ),
            "method_id": BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2,
            "source_binding": binding,
            "raw_sample_dependency_sidecar": deepcopy(
                dict(projection.raw_sample_dependency_sidecar)
            ),
            "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        }
    )
    return _validate_embedded_artifact(artifact)


def validate_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Validate content hashes and replay the sidecar from host authority."""

    data = _validate_embedded_artifact(payload)
    expected = materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    if data != expected:
        raise ValueError(
            "raw-dependency disk artifact did not replay from host-supplied roots"
        )
    return data


def _require_expected_host_hashes(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
) -> None:
    expected = {
        "event_model_input_receipt_sha256": _sha256(
            expected_event_model_input_receipt_sha256,
            "expected_event_model_input_receipt_sha256",
        ),
        "projection_v2_receipt_sha256": _sha256(
            expected_projection_v2_receipt_sha256,
            "expected_projection_v2_receipt_sha256",
        ),
        "source_p0_materialization_receipt_sha256": _sha256(
            expected_source_p0_materialization_receipt_sha256,
            "expected_source_p0_materialization_receipt_sha256",
        ),
    }
    actual = {
        "event_model_input_receipt_sha256": (
            projection.model_input_event.input_receipt_sha256
        ),
        "projection_v2_receipt_sha256": projection.receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            projection.source_p0_materialization_receipt_sha256
        ),
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(f"host expected {name} does not match projection")


def _reference_from_artifact(
    artifact: Mapping[str, Any], relative_path: str, payload: bytes
) -> dict[str, Any]:
    binding = artifact["source_binding"]
    body = {
        "schema_version": (
            BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V2
        ),
        "method_id": BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2,
        "relative_path": _relative_json_path(relative_path),
        "file_size_bytes": len(payload),
        "file_sha256": _file_sha256(payload),
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": artifact["artifact_sha256"],
    }
    body.update(deepcopy(binding))
    return _finalize_reference(body)


def validate_ba_ieg_p0_raw_dependency_projection_disk_reference_v2(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Validate a detached file reference against independent host roots."""

    data = _validate_embedded_reference(payload)
    _require_expected_host_hashes(
        projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=(expected_projection_v2_receipt_sha256),
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
    )
    artifact = materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    expected_binding = artifact["source_binding"]
    expected = {
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        **expected_binding,
    }
    for name, value in expected.items():
        if data[name] != value:
            raise ValueError(f"disk reference {name} drifted from host authority")
    return data


def _resolve_root(root: os.PathLike[str] | str) -> Path:
    path = Path(root).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("disk sidecar root must be an existing directory")
    return path


def _resolve_destination(
    root: os.PathLike[str] | str, relative_path: str
) -> tuple[Path, Path]:
    root_path = _resolve_root(root)
    relative = _relative_json_path(relative_path)
    logical = root_path / relative
    parent = logical.parent.resolve(strict=True)
    if parent != root_path and root_path not in parent.parents:
        raise ValueError("disk sidecar path escapes its root")
    if not parent.is_dir():
        raise ValueError("disk sidecar parent must be an existing directory")
    destination = parent / logical.name
    return parent, destination


def _fsync_directory(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(payload: bytes, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"disk sidecar already exists: {destination.name}")
    descriptor = -1
    temporary_name: str | None = None
    committed = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard-link publication is atomic and fails if a
        # concurrent writer has already installed the destination name.
        os.link(temporary_name, destination)
        committed = True
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None and os.path.lexists(temporary_name):
            os.unlink(temporary_name)
            if committed:
                _fsync_directory(destination.parent)


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"strict JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"strict JSON contains a non-finite constant: {value}")


def _read_strict_canonical_json_file(path: Path) -> tuple[dict[str, Any], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("disk sidecar must be a single-link regular file")
        size = _positive_integer(before.st_size, "disk sidecar file size")
        if size > BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2:
            raise ValueError("disk sidecar exceeds the bounded JSON size limit")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("disk sidecar was truncated during loading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("disk sidecar grew during loading")
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ValueError("disk sidecar changed during loading")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("disk sidecar is not strict UTF-8 JSON") from exc
    try:
        parsed = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("disk sidecar is not strict JSON") from exc
    if type(parsed) is not dict:
        raise ValueError("disk sidecar top level must be a JSON object")
    try:
        canonical = _canonical_json_bytes(parsed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("disk sidecar contains non-canonical JSON values") from exc
    if raw != canonical:
        raise ValueError("disk sidecar bytes are not canonical JSON")
    return parsed, raw


def write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v2(
    root: os.PathLike[str] | str,
    relative_path: str,
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Atomically publish one append-only JSON sidecar and return its reference."""

    parent, destination = _resolve_destination(root, relative_path)
    del parent
    artifact = materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    payload = _canonical_json_bytes(artifact)
    if len(payload) > BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2:
        raise ValueError("disk sidecar exceeds the bounded JSON size limit")
    reference = _reference_from_artifact(artifact, relative_path, payload)
    _validate_embedded_reference(reference)
    _publish_no_clobber(payload, destination)
    written, written_bytes = _read_strict_canonical_json_file(destination)
    if written_bytes != payload or written != artifact:
        raise ValueError("published disk sidecar bytes changed unexpectedly")
    return reference


def load_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v2(
    root: os.PathLike[str] | str,
    detached_reference: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Load canonical JSON, verify file binding, and replay every dependency."""

    reference = validate_ba_ieg_p0_raw_dependency_projection_disk_reference_v2(
        detached_reference,
        projection=projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=(expected_projection_v2_receipt_sha256),
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    _, destination = _resolve_destination(root, reference["relative_path"])
    artifact, raw = _read_strict_canonical_json_file(destination)
    if len(raw) != reference["file_size_bytes"]:
        raise ValueError("disk sidecar file size drifted from its reference")
    if _file_sha256(raw) != reference["file_sha256"]:
        raise ValueError("disk sidecar file SHA-256 drifted from its reference")
    validated = validate_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        artifact,
        projection=projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    if (
        validated["artifact_id"] != reference["artifact_id"]
        or validated["artifact_sha256"] != reference["artifact_sha256"]
    ):
        raise ValueError("disk sidecar artifact drifted from its detached reference")
    return validated


__all__ = [
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V2",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V2",
    "load_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v2",
    "materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2",
    "validate_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2",
    "validate_ba_ieg_p0_raw_dependency_projection_disk_reference_v2",
    "write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v2",
]
