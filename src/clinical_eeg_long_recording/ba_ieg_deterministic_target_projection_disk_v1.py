"""Deterministic JSON+NPZ transport for detached BA-IEG measurement targets.

The model-forward event remains target-free.  This module persists only the
content-bound deterministic measurement sidecar produced by projection v2.
The NPZ contains five fixed, non-pickle arrays and is written with stable ZIP
metadata, so identical targets produce identical bytes.  A strict canonical
JSON artifact binds those bytes to the event input, P0 receipt and projection
v2 receipt; a detached reference binds both files.

Publication is append-only and no-clobber.  Loading requires the host
projection and independent expected hashes, rejects symlinks/path traversal,
checks bounded ZIP contents before NumPy opens them, reconstructs the target
and sidecar classes, and compares the replayed receipts to host authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Final, Mapping
import zipfile

import numpy as np
import torch

from .ba_ieg_event_model_input_projection_v1 import (
    BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION,
    BAIEGContentBoundDeterministicTargetSidecarV1,
)
from .ba_ieg_event_model_input_projection_v2 import (
    BAIEGEventModelInputProjectionV2,
)
from .ba_ieg_training_contract import (
    BA_IEG_DETERMINISTIC_TARGETS,
    BAIEGDeterministicTargets,
)


BA_IEG_DETERMINISTIC_TARGET_DISK_ARTIFACT_SCHEMA_V1: Final[str] = (
    "ba_ieg_deterministic_target_json_npz_artifact_v1"
)
BA_IEG_DETERMINISTIC_TARGET_DISK_REFERENCE_SCHEMA_V1: Final[str] = (
    "ba_ieg_deterministic_target_json_npz_reference_v1"
)
BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1: Final[str] = (
    "append_only_deterministic_nonpickle_target_json_npz_v1"
)
BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_JSON_BYTES_V1: Final[int] = 4 * 1024 * 1024
BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1: Final[int] = 2 * 1024 * 1024 * 1024
BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_UNCOMPRESSED_BYTES_V1: Final[int] = (
    4 * 1024 * 1024 * 1024
)

_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "values",
    "value_mask",
    "row_time_bounds_seconds",
    "row_unit_index",
    "row_view_index",
)
_ARRAY_DTYPES: Final[Mapping[str, str]] = {
    "values": "<f4",
    "value_mask": "|b1",
    "row_time_bounds_seconds": "<f8",
    "row_unit_index": "<i8",
    "row_view_index": "<i8",
}
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_SCOPE_RECEIPT: Final[dict[str, bool]] = {
    "eeg_signal_derived_measurement_supervision_only": True,
    "model_input_target_free": True,
    "available_to_model_forward": False,
    "available_to_batch_packing": False,
    "deterministic_npz_bytes": True,
    "numpy_pickle_allowed": False,
    "append_only_no_clobber_publication": True,
    "public_interval_target_present": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "private_doctor_label_used": False,
    "clinical_text_used": False,
}


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


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a valid non-empty trimmed identifier")
    return value


def _relative_path(value: object, *, suffix: str, name: str) -> str:
    text = _identifier(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != suffix
        or "\\" in text
    ):
        raise ValueError(f"{name} must be a canonical relative {suffix} path")
    return text


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_array(name: str, value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().contiguous().numpy()
    expected = np.dtype(_ARRAY_DTYPES[name])
    if array.dtype.kind != expected.kind or array.dtype.itemsize != expected.itemsize:
        raise TypeError(
            f"deterministic target {name} must have dtype {expected.str}"
        )
    return np.ascontiguousarray(array.astype(expected, copy=False))


def _target_arrays(
    sidecar: BAIEGContentBoundDeterministicTargetSidecarV1,
) -> dict[str, np.ndarray]:
    sidecar.verify_integrity()
    targets = sidecar.targets
    return {
        name: _canonical_array(name, getattr(targets, name))
        for name in _ARRAY_NAMES
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != _ARRAY_NAMES:
        raise ValueError("deterministic target NPZ array order drifted")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        for name in _ARRAY_NAMES:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.flag_bits = 0
            archive.writestr(info, _npy_bytes(arrays[name]))
    payload = stream.getvalue()
    if not payload or len(payload) > BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1:
        raise ValueError("deterministic target NPZ exceeds its bounded size")
    return payload


def _array_descriptors(
    arrays: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "contiguous_c_order": bool(arrays[name].flags.c_contiguous),
            "array_data_sha256": _bytes_sha256(
                arrays[name].tobytes(order="C")
            ),
        }
        for name in _ARRAY_NAMES
    ]


def _source_binding(
    projection: BAIEGEventModelInputProjectionV2,
) -> dict[str, Any]:
    if not isinstance(projection, BAIEGEventModelInputProjectionV2):
        raise TypeError("target disk transport requires projection v2")
    projection.verify_integrity()
    event = projection.model_input_event
    sidecar = projection.deterministic_target_sidecar
    return {
        "event_id": event.event_id,
        "recording_id": event.recording_id,
        "patient_uid": event.patient_uid,
        "model_split": event.model_split,
        "event_model_input_receipt_sha256": event.input_receipt_sha256,
        "source_p0_materialization_receipt_sha256": (
            projection.source_p0_materialization_receipt_sha256
        ),
        "projection_v2_receipt_sha256": projection.receipt_sha256,
        "canonical_receipt_sha256": event.canonical_receipt_sha256,
        "deterministic_target_sidecar_schema_version": sidecar.schema_version,
        "deterministic_target_sidecar_receipt_sha256": sidecar.receipt_sha256,
        "deterministic_target_receipt_sha256": sidecar.target_receipt_sha256,
        "source_event_input_receipt_sha256": (
            sidecar.source_event_input_receipt_sha256
        ),
        "dense_measurement_sidecar_receipt_sha256": (
            sidecar.dense_measurement_sidecar_receipt_sha256
        ),
        "dense_measurement_source_binding_sha256": (
            sidecar.dense_measurement_source_binding_sha256
        ),
        "feature_scope_sha256": sidecar.feature_scope_sha256,
        "target_policy_sha256": sidecar.targets.policy_sha256,
        "target_source_binding_sha256": sidecar.targets.source_binding_sha256,
    }


def _finalize_artifact(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["artifact_id"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["artifact_id"] = "BAIEG-DTGT-" + _canonical_sha256(result)[:24]
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def _finalize_reference(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["reference_id"] = "CONTENT-ADDRESS-PENDING"
    result["reference_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["reference_id"] = "BAIEG-DTGT-REF-" + _canonical_sha256(result)[:24]
    result["reference_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    data: Mapping[str, Any], *, id_field: str, hash_field: str, prefix: str
) -> None:
    _identifier(data[id_field], id_field)
    _sha256(data[hash_field], hash_field)
    digest_source = deepcopy(dict(data))
    digest_source[hash_field] = "CONTENT-ADDRESS-PENDING"
    if data[hash_field] != _canonical_sha256(digest_source):
        raise ValueError(f"{hash_field} does not bind content")
    id_source = deepcopy(dict(data))
    id_source[id_field] = "CONTENT-ADDRESS-PENDING"
    id_source[hash_field] = "CONTENT-ADDRESS-PENDING"
    if data[id_field] != prefix + _canonical_sha256(id_source)[:24]:
        raise ValueError(f"{id_field} does not bind content")


_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "recording_id",
        "patient_uid",
        "model_split",
        "event_model_input_receipt_sha256",
        "source_p0_materialization_receipt_sha256",
        "projection_v2_receipt_sha256",
        "canonical_receipt_sha256",
        "deterministic_target_sidecar_schema_version",
        "deterministic_target_sidecar_receipt_sha256",
        "deterministic_target_receipt_sha256",
        "source_event_input_receipt_sha256",
        "dense_measurement_sidecar_receipt_sha256",
        "dense_measurement_source_binding_sha256",
        "feature_scope_sha256",
        "target_policy_sha256",
        "target_source_binding_sha256",
    }
)


def _validate_binding(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _BINDING_KEYS:
        raise ValueError("deterministic target source binding drifted")
    result = deepcopy(value)
    for name in ("event_id", "recording_id", "patient_uid", "model_split"):
        _identifier(result[name], name)
    if (
        result["deterministic_target_sidecar_schema_version"]
        != BA_IEG_DETERMINISTIC_TARGET_SIDECAR_SCHEMA_VERSION
    ):
        raise ValueError("deterministic target sidecar schema drifted")
    for name in _BINDING_KEYS.difference(
        {
            "event_id",
            "recording_id",
            "patient_uid",
            "model_split",
            "deterministic_target_sidecar_schema_version",
        }
    ):
        _sha256(result[name], name)
    return result


def _validate_descriptors(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(_ARRAY_NAMES):
        raise ValueError("deterministic target array descriptor roster drifted")
    result: list[dict[str, Any]] = []
    for expected_name, item in zip(_ARRAY_NAMES, value):
        if type(item) is not dict or set(item) != {
            "name",
            "dtype",
            "shape",
            "contiguous_c_order",
            "array_data_sha256",
        }:
            raise ValueError("deterministic target array descriptor fields drifted")
        if (
            item["name"] != expected_name
            or item["dtype"] != _ARRAY_DTYPES[expected_name]
            or item["contiguous_c_order"] is not True
            or not isinstance(item["shape"], list)
            or not item["shape"]
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in item["shape"]
            )
        ):
            raise ValueError("deterministic target array descriptor is invalid")
        _sha256(item["array_data_sha256"], "array_data_sha256")
        result.append(deepcopy(item))
    rows = result[0]["shape"][0]
    expected_shapes = {
        "values": [rows, len(BA_IEG_DETERMINISTIC_TARGETS)],
        "value_mask": [rows, len(BA_IEG_DETERMINISTIC_TARGETS)],
        "row_time_bounds_seconds": [rows, 2],
        "row_unit_index": [rows],
        "row_view_index": [rows],
    }
    for descriptor in result:
        if descriptor["shape"] != expected_shapes[descriptor["name"]]:
            raise ValueError("deterministic target array shapes do not align")
    return result


def validate_ba_ieg_deterministic_target_disk_artifact_v1(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "artifact_id",
        "artifact_sha256",
        "source_binding",
        "target_names",
        "arrays",
        "npz_relative_path",
        "npz_file_size_bytes",
        "npz_file_sha256",
        "scope_receipt",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("deterministic target artifact has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != BA_IEG_DETERMINISTIC_TARGET_DISK_ARTIFACT_SCHEMA_V1
        or data["method_id"] != BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1
        or data["scope_receipt"] != _SCOPE_RECEIPT
        or data["target_names"] != list(BA_IEG_DETERMINISTIC_TARGETS)
    ):
        raise ValueError("deterministic target artifact contract drifted")
    data["source_binding"] = _validate_binding(data["source_binding"])
    data["arrays"] = _validate_descriptors(data["arrays"])
    data["npz_relative_path"] = _relative_path(
        data["npz_relative_path"], suffix=".npz", name="npz_relative_path"
    )
    _positive_integer(data["npz_file_size_bytes"], "npz_file_size_bytes")
    if data["npz_file_size_bytes"] > BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1:
        raise ValueError("deterministic target NPZ is too large")
    _sha256(data["npz_file_sha256"], "npz_file_sha256")
    _validate_content_address(
        data,
        id_field="artifact_id",
        hash_field="artifact_sha256",
        prefix="BAIEG-DTGT-",
    )
    return data


def materialize_ba_ieg_deterministic_target_disk_artifact_v1(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    npz_relative_path: str,
) -> tuple[dict[str, Any], bytes]:
    """Create deterministic target metadata and exact NPZ bytes in memory."""

    relative_npz = _relative_path(
        npz_relative_path, suffix=".npz", name="npz_relative_path"
    )
    arrays = _target_arrays(projection.deterministic_target_sidecar)
    npz_payload = _deterministic_npz_bytes(arrays)
    artifact = _finalize_artifact(
        {
            "schema_version": BA_IEG_DETERMINISTIC_TARGET_DISK_ARTIFACT_SCHEMA_V1,
            "method_id": BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1,
            "source_binding": _source_binding(projection),
            "target_names": list(BA_IEG_DETERMINISTIC_TARGETS),
            "arrays": _array_descriptors(arrays),
            "npz_relative_path": relative_npz,
            "npz_file_size_bytes": len(npz_payload),
            "npz_file_sha256": _bytes_sha256(npz_payload),
            "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        }
    )
    return validate_ba_ieg_deterministic_target_disk_artifact_v1(
        artifact
    ), npz_payload


def _reference_from_artifact(
    artifact: Mapping[str, Any], *, json_relative_path: str, json_payload: bytes
) -> dict[str, Any]:
    binding = artifact["source_binding"]
    body = {
        "schema_version": BA_IEG_DETERMINISTIC_TARGET_DISK_REFERENCE_SCHEMA_V1,
        "method_id": BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1,
        "reference_id": "CONTENT-ADDRESS-PENDING",
        "reference_sha256": "CONTENT-ADDRESS-PENDING",
        "json_relative_path": _relative_path(
            json_relative_path, suffix=".json", name="json_relative_path"
        ),
        "json_file_size_bytes": len(json_payload),
        "json_file_sha256": _bytes_sha256(json_payload),
        "npz_relative_path": artifact["npz_relative_path"],
        "npz_file_size_bytes": artifact["npz_file_size_bytes"],
        "npz_file_sha256": artifact["npz_file_sha256"],
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "source_binding": deepcopy(binding),
    }
    return _finalize_reference(body)


def validate_ba_ieg_deterministic_target_disk_reference_v1(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2 | None = None,
    expected_event_model_input_receipt_sha256: str | None = None,
    expected_projection_v2_receipt_sha256: str | None = None,
    expected_source_p0_materialization_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "reference_id",
        "reference_sha256",
        "json_relative_path",
        "json_file_size_bytes",
        "json_file_sha256",
        "npz_relative_path",
        "npz_file_size_bytes",
        "npz_file_sha256",
        "artifact_id",
        "artifact_sha256",
        "source_binding",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("deterministic target reference has missing/unknown fields")
    data = deepcopy(payload)
    if (
        data["schema_version"] != BA_IEG_DETERMINISTIC_TARGET_DISK_REFERENCE_SCHEMA_V1
        or data["method_id"] != BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1
    ):
        raise ValueError("deterministic target reference contract drifted")
    for name, suffix in (
        ("json_relative_path", ".json"),
        ("npz_relative_path", ".npz"),
    ):
        data[name] = _relative_path(data[name], suffix=suffix, name=name)
    for name, maximum in (
        ("json_file_size_bytes", BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_JSON_BYTES_V1),
        ("npz_file_size_bytes", BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1),
    ):
        if _positive_integer(data[name], name) > maximum:
            raise ValueError(f"{name} exceeds its bounded size")
    for name in (
        "json_file_sha256",
        "npz_file_sha256",
        "artifact_sha256",
    ):
        _sha256(data[name], name)
    _identifier(data["artifact_id"], "artifact_id")
    data["source_binding"] = _validate_binding(data["source_binding"])
    _validate_content_address(
        data,
        id_field="reference_id",
        hash_field="reference_sha256",
        prefix="BAIEG-DTGT-REF-",
    )
    supplied = (
        expected_event_model_input_receipt_sha256,
        expected_projection_v2_receipt_sha256,
        expected_source_p0_materialization_receipt_sha256,
    )
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError("all independent expected hashes must be supplied together")
    if all(value is not None for value in supplied):
        expected = {
            "event_model_input_receipt_sha256": _sha256(
                supplied[0], "expected event input receipt"
            ),
            "projection_v2_receipt_sha256": _sha256(
                supplied[1], "expected projection v2 receipt"
            ),
            "source_p0_materialization_receipt_sha256": _sha256(
                supplied[2], "expected P0 receipt"
            ),
        }
        for name, value in expected.items():
            if data["source_binding"][name] != value:
                raise ValueError(f"deterministic target reference {name} was rebound")
    if projection is not None:
        if data["source_binding"] != _source_binding(projection):
            raise ValueError("deterministic target reference disagrees with projection")
    return data


def _resolve_root(root: os.PathLike[str] | str) -> Path:
    result = Path(root).resolve(strict=True)
    if not result.is_dir():
        raise ValueError("deterministic target disk root must be a directory")
    return result


def _destination(
    root: os.PathLike[str] | str, relative_path: str, *, suffix: str
) -> Path:
    root_path = _resolve_root(root)
    relative = _relative_path(relative_path, suffix=suffix, name="relative_path")
    logical = root_path / relative
    parent = logical.parent.resolve(strict=True)
    if parent != root_path and root_path not in parent.parents:
        raise ValueError("deterministic target path escapes its root")
    if not parent.is_dir():
        raise ValueError("deterministic target parent must already exist")
    return parent / logical.name


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(payload: bytes, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"deterministic target file exists: {destination.name}")
    descriptor = -1
    temporary: str | None = None
    committed = False
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        committed = True
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and os.path.lexists(temporary):
            os.unlink(temporary)
            if committed:
                _fsync_directory(destination.parent)


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file(
        path, maximum=BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_JSON_BYTES_V1
    )
    try:
        decoded = raw.decode("utf-8", errors="strict")
        pairs_seen: list[str] = []

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate strict JSON key: {key}")
                result[key] = value
                pairs_seen.append(key)
            return result

        parsed = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("deterministic target metadata is not strict JSON") from exc
    del pairs_seen
    if type(parsed) is not dict or _canonical_json_bytes(parsed) != raw:
        raise ValueError("deterministic target metadata is not canonical JSON")
    return parsed, raw


def _read_regular_file(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("deterministic target artifact must be a single-link regular file")
        if before.st_size < 1 or before.st_size > maximum:
            raise ValueError("deterministic target artifact size is outside its bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("deterministic target artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("deterministic target artifact grew during load")
        after = os.fstat(descriptor)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, name) != getattr(after, name):
                raise ValueError("deterministic target artifact changed during load")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _arrays_from_npz(
    payload: bytes, descriptors: list[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [
                f"{name}.npy" for name in _ARRAY_NAMES
            ]:
                raise ValueError("deterministic target NPZ member roster drifted")
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits & 0x1
                or info.file_size < 1
                for info in infos
            ):
                raise ValueError("deterministic target NPZ member contract drifted")
            if sum(info.file_size for info in infos) > (
                BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_UNCOMPRESSED_BYTES_V1
            ):
                raise ValueError("deterministic target NPZ expands beyond its bound")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("deterministic target NPZ is invalid") from exc
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != list(_ARRAY_NAMES):
                raise ValueError("deterministic target NPZ array order drifted")
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in _ARRAY_NAMES
            }
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError("deterministic target NPZ cannot be loaded safely") from exc
    descriptor_by_name = {item["name"]: item for item in descriptors}
    for name, array in arrays.items():
        descriptor = descriptor_by_name[name]
        if (
            array.dtype.str != descriptor["dtype"]
            or list(array.shape) != descriptor["shape"]
            or not array.flags.c_contiguous
            or _bytes_sha256(array.tobytes(order="C"))
            != descriptor["array_data_sha256"]
        ):
            raise ValueError(f"deterministic target NPZ array {name} drifted")
    return arrays


def _sidecar_from_artifact_and_arrays(
    artifact: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> BAIEGContentBoundDeterministicTargetSidecarV1:
    binding = artifact["source_binding"]
    targets = BAIEGDeterministicTargets(
        values=torch.from_numpy(arrays["values"].copy()),
        value_mask=torch.from_numpy(arrays["value_mask"].copy()),
        row_time_bounds_seconds=torch.from_numpy(
            arrays["row_time_bounds_seconds"].copy()
        ),
        row_unit_index=torch.from_numpy(arrays["row_unit_index"].copy()),
        row_view_index=torch.from_numpy(arrays["row_view_index"].copy()),
        policy_sha256=binding["target_policy_sha256"],
        source_binding_sha256=binding["target_source_binding_sha256"],
        target_names=tuple(artifact["target_names"]),
    )
    if targets.receipt_sha256 != binding["deterministic_target_receipt_sha256"]:
        raise ValueError("reconstructed deterministic target receipt drifted")
    sidecar = BAIEGContentBoundDeterministicTargetSidecarV1(
        event_id=binding["event_id"],
        recording_id=binding["recording_id"],
        patient_uid=binding["patient_uid"],
        model_split=binding["model_split"],
        source_event_input_receipt_sha256=binding[
            "source_event_input_receipt_sha256"
        ],
        source_p0_materialization_receipt_sha256=binding[
            "source_p0_materialization_receipt_sha256"
        ],
        dense_measurement_sidecar_receipt_sha256=binding[
            "dense_measurement_sidecar_receipt_sha256"
        ],
        dense_measurement_source_binding_sha256=binding[
            "dense_measurement_source_binding_sha256"
        ],
        feature_scope_sha256=binding["feature_scope_sha256"],
        targets=targets,
    )
    if sidecar.receipt_sha256 != binding[
        "deterministic_target_sidecar_receipt_sha256"
    ]:
        raise ValueError("reconstructed deterministic target sidecar drifted")
    return sidecar


def write_ba_ieg_deterministic_target_projection_disk_v1(
    root: os.PathLike[str] | str,
    *,
    json_relative_path: str,
    npz_relative_path: str,
    projection: BAIEGEventModelInputProjectionV2,
) -> dict[str, Any]:
    """Publish one deterministic target pair and return its detached reference."""

    json_destination = _destination(root, json_relative_path, suffix=".json")
    npz_destination = _destination(root, npz_relative_path, suffix=".npz")
    if json_destination == npz_destination:
        raise ValueError("JSON and NPZ destinations must differ")
    if os.path.lexists(json_destination) or os.path.lexists(npz_destination):
        raise FileExistsError("deterministic target destination already exists")
    artifact, npz_payload = materialize_ba_ieg_deterministic_target_disk_artifact_v1(
        projection, npz_relative_path=npz_relative_path
    )
    json_payload = _canonical_json_bytes(artifact)
    if len(json_payload) > BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_JSON_BYTES_V1:
        raise ValueError("deterministic target JSON exceeds its bounded size")
    reference = _reference_from_artifact(
        artifact,
        json_relative_path=json_relative_path,
        json_payload=json_payload,
    )
    validate_ba_ieg_deterministic_target_disk_reference_v1(
        reference,
        projection=projection,
        expected_event_model_input_receipt_sha256=(
            projection.model_input_event.input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=projection.receipt_sha256,
        expected_source_p0_materialization_receipt_sha256=(
            projection.source_p0_materialization_receipt_sha256
        ),
    )
    npz_committed = False
    try:
        _publish_no_clobber(npz_payload, npz_destination)
        npz_committed = True
        _publish_no_clobber(json_payload, json_destination)
    except Exception:
        # JSON is the pair's commit marker.  If publication fails before that
        # marker exists, remove only the NPZ installed by this invocation.
        if npz_committed and not os.path.lexists(json_destination):
            try:
                if _read_regular_file(
                    npz_destination,
                    maximum=BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1,
                ) == npz_payload:
                    os.unlink(npz_destination)
                    _fsync_directory(npz_destination.parent)
            except (FileNotFoundError, ValueError, OSError):
                pass
        raise
    return reference


def load_ba_ieg_deterministic_target_projection_disk_v1(
    root: os.PathLike[str] | str,
    detached_reference: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
) -> BAIEGContentBoundDeterministicTargetSidecarV1:
    """Load, reconstruct and replay one deterministic target disk pair."""

    reference = validate_ba_ieg_deterministic_target_disk_reference_v1(
        detached_reference,
        projection=projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=expected_projection_v2_receipt_sha256,
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
    )
    json_path = _destination(
        root, reference["json_relative_path"], suffix=".json"
    )
    npz_path = _destination(root, reference["npz_relative_path"], suffix=".npz")
    artifact, json_payload = _strict_json(json_path)
    npz_payload = _read_regular_file(
        npz_path, maximum=BA_IEG_DETERMINISTIC_TARGET_MAXIMUM_NPZ_BYTES_V1
    )
    for prefix, payload in (("json", json_payload), ("npz", npz_payload)):
        if (
            len(payload) != reference[f"{prefix}_file_size_bytes"]
            or _bytes_sha256(payload) != reference[f"{prefix}_file_sha256"]
        ):
            raise ValueError(f"deterministic target {prefix} file binding drifted")
    artifact = validate_ba_ieg_deterministic_target_disk_artifact_v1(artifact)
    if (
        artifact["artifact_id"] != reference["artifact_id"]
        or artifact["artifact_sha256"] != reference["artifact_sha256"]
        or artifact["source_binding"] != reference["source_binding"]
        or artifact["npz_relative_path"] != reference["npz_relative_path"]
        or artifact["npz_file_size_bytes"] != len(npz_payload)
        or artifact["npz_file_sha256"] != _bytes_sha256(npz_payload)
    ):
        raise ValueError("deterministic target artifact/reference binding drifted")
    expected_artifact, expected_npz = (
        materialize_ba_ieg_deterministic_target_disk_artifact_v1(
            projection, npz_relative_path=reference["npz_relative_path"]
        )
    )
    if artifact != expected_artifact or npz_payload != expected_npz:
        raise ValueError("deterministic target disk pair did not replay from projection")
    arrays = _arrays_from_npz(npz_payload, artifact["arrays"])
    sidecar = _sidecar_from_artifact_and_arrays(artifact, arrays)
    if sidecar.receipt_sha256 != projection.deterministic_target_sidecar.receipt_sha256:
        raise ValueError("loaded deterministic target belongs to another projection")
    return sidecar


__all__ = [
    "BA_IEG_DETERMINISTIC_TARGET_DISK_ARTIFACT_SCHEMA_V1",
    "BA_IEG_DETERMINISTIC_TARGET_DISK_METHOD_ID_V1",
    "BA_IEG_DETERMINISTIC_TARGET_DISK_REFERENCE_SCHEMA_V1",
    "load_ba_ieg_deterministic_target_projection_disk_v1",
    "materialize_ba_ieg_deterministic_target_disk_artifact_v1",
    "validate_ba_ieg_deterministic_target_disk_artifact_v1",
    "validate_ba_ieg_deterministic_target_disk_reference_v1",
    "write_ba_ieg_deterministic_target_projection_disk_v1",
]
