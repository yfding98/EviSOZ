"""Atomic public-only materialization for a regenerable v29 research cache.

This module is deliberately narrower than :mod:`v29_cache_adapter`.  It only
publishes the historical 102-patient held-fold route and never accepts event
features, event identities, montage payloads, targets, or target masks.

Publication is fail-closed:

* patient identities are derived internally from the authoritative public
  index in its exact canonical order;
* the in-memory adapter bundle is fully reopened before any file is written;
* every tensor is stored as ``canonical_tensor_bytes`` rather than pickle or
  an unbound tensor container;
* the complete staging tree is captured once through anchored descriptors,
  then the same immutable byte buffers drive hashes, parsing and replay; and
* a no-replace directory rename publishes that precommit-verified tree
  atomically, with no fallible post-commit validation step.

The cache is a regenerable trusted-workspace research artifact.  Atomic
no-replace publication prevents partial content from becoming visible, but
this module does not promise that the final directory entry survives sudden
power loss because no fallible parent-directory fsync runs after commit.

The historical metric JSON is a SHA-frozen receipt replay.  It is explicitly
not an independent target-based recomputation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import struct
import sys
from types import MappingProxyType
from typing import Any

import torch

from src.evisoz.baseline import frozen_v29
from src.evisoz.baseline.frozen_v29 import (
    FROZEN_PUBLIC_IDENTITY_NAMESPACE,
    PUBLIC_MANIFEST_RESOURCE,
    PUBLIC_PATIENT_COUNT,
    FrozenV29ResourceRegistry,
    PublicV29RosterIndex,
    V29PatientIdentity,
    build_public_v29_roster_index,
    load_frozen_v29_resource_registry,
    replay_public_v29_manifest_metric_receipt,
    validate_frozen_v29_resource_registry,
    validate_public_v29_roster_index,
)
from src.evisoz.baseline.v29_cache import (
    CORE_TENSOR_NAMES,
    PUBLIC_OOF_ROUTE,
    TENSOR_ENCODING,
    canonical_tensor_bytes,
)
from src.evisoz.baseline.v29_cache_adapter import (
    MATERIALIZER_VERSION as ADAPTER_MATERIALIZER_VERSION,
    OpenedV29Cache,
    materialize_public_v29_cache,
    open_v29_cache_for_use,
)
from src.evisoz.data.artifact_ref import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    verify_artifact_content,
)


DISK_MATERIALIZER_VERSION = "evisoz_v29_public_cache_disk_materializer_v2"
MATERIALIZATION_RECEIPT_SCHEMA_VERSION = (
    "evisoz_v29_public_cache_materialization_receipt_v2"
)
DEFAULT_PUBLIC_CACHE_DIRECTORY = Path(
    "outputs/evisoz_v29_public_held_fold_cache_v2_20260831"
)
EXPECTED_P0_C18_SHA256 = (
    "6aae67212390f4037be896d6d020468a33d7210289a0d22a590819f740a470ea"
)

_TENSOR_DIRECTORY = "tensors"
_SIDECAR_DIRECTORY = "sidecars"
_ROUTE_RECEIPT_DIRECTORY = "sidecars/route_receipts"
_AUDIT_DIRECTORY = "audit"
_CACHE_FILE = "cache.json"
_IDENTITY_FILE = "sidecars/patient_identity_roster.json"
_ROUTE_ROSTER_FILE = "sidecars/route_receipt_roster.json"
_RESOURCE_CONFIG_FILE = "sidecars/evisoz_v29_frozen_resources_v1.json"
_FROZEN_MANIFEST_FILE = "audit/frozen_public_manifest.json"
_PUBLIC_METRIC_RECEIPT_FILE = "audit/public_metric_receipt.json"
_MATERIALIZATION_RECEIPT_FILE = "audit/materialization_receipt.json"
_RECEIPT_HASH_PLACEHOLDER = "0" * 64
_MAX_FILE_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CACHE_ID_RE = re.compile(r"^EVISOZ-V29-[0-9a-f]{24}$")
_EXPECTED_DIRECTORY_SET = {
    _TENSOR_DIRECTORY,
    _SIDECAR_DIRECTORY,
    _ROUTE_RECEIPT_DIRECTORY,
    _AUDIT_DIRECTORY,
}
_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bool": torch.bool,
    "int64": torch.int64,
}
_DTYPE_ITEMSIZE = {
    "float32": 4,
    "bool": 1,
    "int64": 8,
}
_MATERIALIZATION_RECEIPT_KEYS = {
    "schema_version",
    "materializer_version",
    "adapter_materializer_version",
    "status",
    "route",
    "unit_kind",
    "cache_id",
    "patient_count",
    "unit_count",
    "canonical_patient_order_source",
    "identity_namespace",
    "p0_c18_tensor_sha256",
    "resource_config_sha256",
    "resource_registry_projection_sha256",
    "public_roster_authority_sha256",
    "public_metric_receipt_sha256",
    "source",
    "independently_recomputed_from_targets",
    "target_tensor_values_deserialized",
    "targets_or_target_mask_get_tensor_calls",
    "event_payloads_present",
    "montage_payloads_present",
    "alpha_zero_hard_bypass_replayed",
    "files",
    "receipt_sha256",
}


@dataclass(frozen=True)
class PublicV29DiskMaterialization:
    """Result of one successfully published and replayed public cache."""

    path: Path
    cache_id: str
    opened: OpenedV29Cache
    materialization_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _DiskSnapshot:
    """One immutable fd-relative snapshot of an entire cache directory."""

    root: Path
    directories: frozenset[str]
    raw_files: Mapping[str, bytes]


def _strict_json_bytes(raw: bytes, context: str) -> Any:
    if not isinstance(raw, bytes):
        raise TypeError(f"{context} bytes must be bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains non-finite JSON value {value}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from exc
    return value


def _decode_canonical_json(raw: bytes, *, context: str) -> Any:
    value = _strict_json_bytes(raw, context)
    if raw != canonical_json_bytes(value):
        raise ValueError(f"{context} is not in the canonical JSON byte domain")
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON projection contains a non-finite float")
        return value
    raise TypeError(f"unsupported JSON projection type: {type(value).__name__}")


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _lexical_absolute(value: str | Path, *, context: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if path.name in {"", ".", ".."}:
        raise ValueError(f"{context} must be a concrete directory")
    return path


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_absolute_directory(path: Path, *, context: str) -> int:
    """Open every absolute path component relative to the previous dirfd."""

    absolute = _lexical_absolute(path, context=context)
    descriptor = os.open("/", _directory_open_flags())
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):  # pragma: no cover - O_DIRECTORY
            raise NotADirectoryError(absolute)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_at(directory_fd: int, name: str, *, context: str) -> bytes:
    """Read one leaf exactly once through an anchored directory descriptor."""

    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
        raise ValueError(f"{context} requires one safe leaf name")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{context} must be a regular file")
        if metadata.st_size < 0 or metadata.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"{context} has an unsafe byte length")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{context} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{context} grew while reading")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise ValueError(f"{context} changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _write_exclusive_bytes_at(directory_fd: int, name: str, raw: bytes) -> None:
    if not isinstance(raw, bytes):
        raise TypeError("exclusive file payload must be bytes")
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
        raise ValueError("exclusive output requires one safe leaf name")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("exclusive output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    else:
        os.close(descriptor)


def _fsync_directory_fd(directory_fd: int) -> None:
    os.fsync(directory_fd)


def _rename_directory_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a directory without replacing a raced destination."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            "public v29 cache destination appeared during publication",
            destination_name,
        )
    raise OSError(error, os.strerror(error), destination_name)


def _expected_files() -> set[str]:
    return {
        _CACHE_FILE,
        *{f"{_TENSOR_DIRECTORY}/{name}.tensor" for name in CORE_TENSOR_NAMES},
        _IDENTITY_FILE,
        _ROUTE_ROSTER_FILE,
        _RESOURCE_CONFIG_FILE,
        *{
            f"{_ROUTE_RECEIPT_DIRECTORY}/{index:03d}.json"
            for index in range(PUBLIC_PATIENT_COUNT)
        },
        _FROZEN_MANIFEST_FILE,
        _PUBLIC_METRIC_RECEIPT_FILE,
        _MATERIALIZATION_RECEIPT_FILE,
    }


def _snapshot_from_open_root(root_fd: int, *, root: Path) -> _DiskSnapshot:
    directories: set[str] = set()
    raw_files: dict[str, bytes] = {}

    def visit(directory_fd: int, relative_directory: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                raise ValueError("cache tree contains an unsafe entry name")
            relative = f"{relative_directory}/{name}" if relative_directory else name
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"cache tree contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
                try:
                    if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                        raise ValueError(f"cache tree directory changed: {relative}")
                    directories.add(relative)
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                raw_files[relative] = _read_regular_file_at(
                    directory_fd,
                    name,
                    context=relative,
                )
            else:
                raise ValueError(f"cache tree contains non-regular entry: {relative}")

    visit(root_fd, "")
    return _DiskSnapshot(
        root=root,
        directories=frozenset(directories),
        raw_files=MappingProxyType(raw_files),
    )


def _capture_disk_snapshot(root: str | Path) -> _DiskSnapshot:
    """Capture every cache file once through one fd-relative directory walk."""

    absolute = _lexical_absolute(root, context="public v29 cache directory")
    root_fd = _open_absolute_directory(absolute, context="public v29 cache directory")
    try:
        return _snapshot_from_open_root(root_fd, root=absolute)
    finally:
        os.close(root_fd)


def _capture_disk_snapshot_at(
    parent_fd: int,
    directory_name: str,
    *,
    root: Path,
) -> _DiskSnapshot:
    root_fd = os.open(directory_name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        return _snapshot_from_open_root(root_fd, root=root)
    finally:
        os.close(root_fd)


def _create_staging_directory_at(parent_fd: int, target_name: str) -> str:
    for _attempt in range(128):
        name = f".{target_name}.staging-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return name
    raise FileExistsError("could not allocate a unique staging directory")


def _decode_canonical_tensor(raw: bytes, *, context: str) -> torch.Tensor:
    if not isinstance(raw, bytes):
        raise TypeError(f"{context} must be bytes")
    if len(raw) < 16:
        raise ValueError(f"{context} is shorter than the tensor envelope")
    header_length = struct.unpack(">Q", raw[:8])[0]
    header_end = 8 + header_length
    if header_length < 2 or header_end + 8 > len(raw):
        raise ValueError(f"{context} has an invalid header length")
    header_raw = raw[8:header_end]
    header = _strict_json_bytes(header_raw, f"{context} header")
    if header_raw != canonical_json_bytes(header):
        raise ValueError(f"{context} header is not canonical JSON")
    if type(header) is not dict or set(header) != {
        "encoding",
        "dtype",
        "shape",
        "byte_order",
    }:
        raise ValueError(f"{context} tensor header fields drifted")
    if header["encoding"] != TENSOR_ENCODING or header["byte_order"] != "little":
        raise ValueError(f"{context} tensor encoding or byte order drifted")
    dtype_name = header["dtype"]
    if dtype_name not in _DTYPE_BY_NAME:
        raise ValueError(f"{context} uses an unsupported tensor dtype")
    shape_raw = header["shape"]
    if not isinstance(shape_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape_raw
    ):
        raise ValueError(f"{context} tensor shape is invalid")
    shape = tuple(shape_raw)
    raw_length = struct.unpack(">Q", raw[header_end : header_end + 8])[0]
    payload_start = header_end + 8
    payload_end = payload_start + raw_length
    if payload_end != len(raw):
        raise ValueError(f"{context} raw length or trailing bytes drifted")
    element_count = math.prod(shape)
    expected_length = element_count * _DTYPE_ITEMSIZE[dtype_name]
    if raw_length != expected_length:
        raise ValueError(f"{context} raw tensor byte length disagrees with shape")
    if sys.byteorder != "little":
        raise RuntimeError("canonical v29 tensor decoding requires a little-endian host")
    payload = bytearray(raw[payload_start:payload_end])
    tensor = torch.frombuffer(payload, dtype=_DTYPE_BY_NAME[dtype_name]).clone()
    tensor = tensor.reshape(shape).contiguous()
    if canonical_tensor_bytes(tensor) != raw:
        raise ValueError(f"{context} does not round-trip through canonical encoding")
    return tensor


def _authorities(
    registry: FrozenV29ResourceRegistry | None,
    public_index: PublicV29RosterIndex | None,
) -> tuple[FrozenV29ResourceRegistry, PublicV29RosterIndex]:
    trusted_registry = (
        load_frozen_v29_resource_registry()
        if registry is None
        else validate_frozen_v29_resource_registry(registry)
    )
    trusted_index = (
        build_public_v29_roster_index(trusted_registry)
        if public_index is None
        else validate_public_v29_roster_index(public_index, trusted_registry)
    )
    if len(trusted_index.patient_ids) != PUBLIC_PATIENT_COUNT:
        raise ValueError("frozen public v29 index must contain exactly 102 patients")
    return trusted_registry, trusted_index


def _canonical_identities(
    public_index: PublicV29RosterIndex,
) -> tuple[V29PatientIdentity, ...]:
    if len(public_index.patient_ids) != PUBLIC_PATIENT_COUNT:
        raise ValueError("canonical public identity roster must contain 102 patients")
    return tuple(
        V29PatientIdentity(
            namespace=FROZEN_PUBLIC_IDENTITY_NAMESPACE,
            patient_id=patient_id,
        )
        for patient_id in public_index.patient_ids
    )


def _open_bundle(
    *,
    cache: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    route_roster: Mapping[str, Any],
    route_receipts: tuple[Mapping[str, Any], ...],
    identity_payload: Mapping[str, Any],
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
    alpha: float,
    residual_supplier: Callable[[], Any] | None,
) -> OpenedV29Cache:
    opened = open_v29_cache_for_use(
        cache,
        tensor_payloads=tensors,
        route_receipt_roster=route_roster,
        route_receipts=route_receipts,
        identity_payload=identity_payload,
        event_identity_payload=None,
        montage_derivation_payload=None,
        registry=registry,
        public_index=public_index,
        alpha=alpha,
        residual_supplier=residual_supplier,
    )
    if len(opened.route_decisions) != PUBLIC_PATIENT_COUNT:
        raise ValueError("opened public v29 cache must contain 102 route decisions")
    if tuple(decision.patient_id for decision in opened.route_decisions) != tuple(
        public_index.patient_ids
    ):
        raise ValueError("opened public v29 patient order drifted")
    if "p0_c18" not in opened.tensor_names:
        raise ValueError("opened public v29 cache lacks p0_c18")
    if opened.tensor_sha256("p0_c18") != EXPECTED_P0_C18_SHA256:
        raise ValueError("canonical public p0_c18 tensor digest drifted")
    if float(alpha) == 0.0 and not torch.equal(
        opened.checkout_selected(),
        opened.checkout_p0(),
    ):
        raise ValueError("alpha=0 selected checkout differs from canonical p0")
    return opened


def _materialization_receipt(
    *,
    raw_files: Mapping[str, bytes],
    cache: Mapping[str, Any],
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
    metric_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    expected_bound_files = _expected_files() - {_MATERIALIZATION_RECEIPT_FILE}
    if set(raw_files) != expected_bound_files:
        raise ValueError("receipt source raw file set drifted")
    files: dict[str, dict[str, object]] = {}
    for relative in sorted(expected_bound_files):
        raw = raw_files[relative]
        files[relative] = {
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        }
    body: dict[str, Any] = {
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "materializer_version": DISK_MATERIALIZER_VERSION,
        "adapter_materializer_version": ADAPTER_MATERIALIZER_VERSION,
        "status": "PASS",
        "route": PUBLIC_OOF_ROUTE,
        "unit_kind": "patient",
        "cache_id": cache["cache_id"],
        "patient_count": PUBLIC_PATIENT_COUNT,
        "unit_count": PUBLIC_PATIENT_COUNT,
        "canonical_patient_order_source": "public_index.patient_ids",
        "identity_namespace": FROZEN_PUBLIC_IDENTITY_NAMESPACE,
        "p0_c18_tensor_sha256": EXPECTED_P0_C18_SHA256,
        "resource_config_sha256": registry.config_sha256,
        "resource_registry_projection_sha256": (
            public_index.resource_registry_projection_sha256
        ),
        "public_roster_authority_sha256": public_index.authority_sha256,
        "public_metric_receipt_sha256": canonical_json_sha256(metric_receipt),
        "source": "sha_frozen_public_manifest",
        "independently_recomputed_from_targets": False,
        "target_tensor_values_deserialized": False,
        "targets_or_target_mask_get_tensor_calls": 0,
        "event_payloads_present": False,
        "montage_payloads_present": False,
        "alpha_zero_hard_bypass_replayed": True,
        "files": files,
        "receipt_sha256": _RECEIPT_HASH_PLACEHOLDER,
    }
    body["receipt_sha256"] = canonical_json_sha256(body)
    return body


def _validate_materialization_receipt_structure(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("materialization receipt must be an object")
    data = _plain_json(value)
    if type(data) is not dict or set(data) != _MATERIALIZATION_RECEIPT_KEYS:
        raise ValueError("materialization receipt fields drifted")
    fixed_values = {
        "schema_version": MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "materializer_version": DISK_MATERIALIZER_VERSION,
        "adapter_materializer_version": ADAPTER_MATERIALIZER_VERSION,
        "status": "PASS",
        "route": PUBLIC_OOF_ROUTE,
        "unit_kind": "patient",
        "patient_count": PUBLIC_PATIENT_COUNT,
        "unit_count": PUBLIC_PATIENT_COUNT,
        "canonical_patient_order_source": "public_index.patient_ids",
        "identity_namespace": FROZEN_PUBLIC_IDENTITY_NAMESPACE,
        "p0_c18_tensor_sha256": EXPECTED_P0_C18_SHA256,
        "source": "sha_frozen_public_manifest",
        "independently_recomputed_from_targets": False,
        "target_tensor_values_deserialized": False,
        "targets_or_target_mask_get_tensor_calls": 0,
        "event_payloads_present": False,
        "montage_payloads_present": False,
        "alpha_zero_hard_bypass_replayed": True,
    }
    if any(data.get(key) != expected for key, expected in fixed_values.items()):
        raise ValueError("materialization receipt fixed contract drifted")
    if not isinstance(data["cache_id"], str) or _CACHE_ID_RE.fullmatch(
        data["cache_id"]
    ) is None:
        raise ValueError("materialization receipt cache_id is invalid")
    for key in (
        "resource_config_sha256",
        "resource_registry_projection_sha256",
        "public_roster_authority_sha256",
        "public_metric_receipt_sha256",
        "receipt_sha256",
    ):
        if not isinstance(data[key], str) or _SHA256_RE.fullmatch(data[key]) is None:
            raise ValueError(f"materialization receipt {key} is not a SHA-256")
    files = data["files"]
    expected_file_names = _expected_files() - {_MATERIALIZATION_RECEIPT_FILE}
    if type(files) is not dict or set(files) != expected_file_names:
        raise ValueError("materialization receipt file set drifted")
    for relative, file_receipt in files.items():
        if type(file_receipt) is not dict or set(file_receipt) != {
            "sha256",
            "size_bytes",
        }:
            raise ValueError(
                f"materialization receipt file fields drifted for {relative}"
            )
        if not isinstance(file_receipt["sha256"], str) or _SHA256_RE.fullmatch(
            file_receipt["sha256"]
        ) is None:
            raise ValueError(
                f"materialization receipt file SHA-256 is invalid for {relative}"
            )
        size = file_receipt["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(
                f"materialization receipt file size is invalid for {relative}"
            )
    if data["p0_c18_tensor_sha256"] != files[
        f"{_TENSOR_DIRECTORY}/p0_c18.tensor"
    ]["sha256"]:
        raise ValueError("p0_c18 digest is not cross-bound to its file receipt")
    observed_receipt_sha256 = data["receipt_sha256"]
    data["receipt_sha256"] = _RECEIPT_HASH_PLACEHOLDER
    if canonical_json_sha256(data) != observed_receipt_sha256:
        raise ValueError("materialization receipt self-hash drifted")
    data["receipt_sha256"] = observed_receipt_sha256
    return data


def _validate_materialization_receipt_bindings(
    value: Any,
    *,
    snapshot: _DiskSnapshot,
    cache: Mapping[str, Any],
    public_index: PublicV29RosterIndex,
    registry: FrozenV29ResourceRegistry,
    metric_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    data = _validate_materialization_receipt_structure(value)
    expected = _materialization_receipt(
        raw_files={
            relative: raw
            for relative, raw in snapshot.raw_files.items()
            if relative != _MATERIALIZATION_RECEIPT_FILE
        },
        cache=cache,
        public_index=public_index,
        registry=registry,
        metric_receipt=metric_receipt,
    )
    if data != expected:
        raise ValueError("materialization receipt or bound file bytes drifted")
    return data


def _open_disk_snapshot(
    snapshot: _DiskSnapshot,
    *,
    registry: FrozenV29ResourceRegistry,
    public_index: PublicV29RosterIndex,
    alpha: float,
    residual_supplier: Callable[[], Any] | None,
) -> tuple[OpenedV29Cache, dict[str, Any]]:
    if set(snapshot.directories) != _EXPECTED_DIRECTORY_SET:
        raise ValueError("public v29 cache directory set drifted")
    if set(snapshot.raw_files) != _expected_files():
        raise ValueError("public v29 cache file set drifted")
    raw_files = snapshot.raw_files

    config_raw = raw_files[_RESOURCE_CONFIG_FILE]
    if config_raw != registry.config_bytes:
        raise ValueError("frozen resource config sidecar bytes drifted")

    frozen_manifest_raw = raw_files[_FROZEN_MANIFEST_FILE]
    expected_manifest_raw = frozen_v29._read_verified_resource_bytes(
        registry.require(PUBLIC_MANIFEST_RESOURCE)
    )
    if frozen_manifest_raw != expected_manifest_raw:
        raise ValueError("frozen public manifest sidecar bytes drifted")

    _manifest, expected_metric_receipt = replay_public_v29_manifest_metric_receipt(
        registry
    )
    verify_artifact_content(
        expected_metric_receipt["public_manifest_ref"],
        frozen_manifest_raw,
    )
    metric_receipt = _decode_canonical_json(
        raw_files[_PUBLIC_METRIC_RECEIPT_FILE],
        context="public metric receipt",
    )
    if metric_receipt != expected_metric_receipt:
        raise ValueError("public metric receipt drifted from frozen manifest replay")
    if (
        metric_receipt.get("source") != "sha_frozen_public_manifest"
        or metric_receipt.get("independently_recomputed_from_targets") is not False
        or metric_receipt.get("targets_or_target_mask_read") is not False
    ):
        raise ValueError("public metric receipt overclaims target-based confirmation")

    cache = _decode_canonical_json(raw_files[_CACHE_FILE], context="v29 cache")
    identity_payload = _decode_canonical_json(
        raw_files[_IDENTITY_FILE],
        context="patient identity roster",
    )
    patient_rows = identity_payload.get("patients") if isinstance(identity_payload, dict) else None
    if not isinstance(patient_rows, list) or len(patient_rows) != PUBLIC_PATIENT_COUNT:
        raise ValueError("patient identity sidecar must contain exactly 102 rows")
    if tuple(row.get("patient_id") for row in patient_rows) != tuple(
        public_index.patient_ids
    ):
        raise ValueError("patient identity sidecar canonical order drifted")
    if any(
        row.get("identity_namespace") != FROZEN_PUBLIC_IDENTITY_NAMESPACE
        for row in patient_rows
    ):
        raise ValueError("patient identity sidecar namespace drifted")

    route_roster = _decode_canonical_json(
        raw_files[_ROUTE_ROSTER_FILE],
        context="route receipt roster",
    )
    route_receipts = tuple(
        _decode_canonical_json(
            raw_files[f"{_ROUTE_RECEIPT_DIRECTORY}/{index:03d}.json"],
            context=f"route receipt {index}",
        )
        for index in range(PUBLIC_PATIENT_COUNT)
    )
    tensors = {
        name: _decode_canonical_tensor(
            raw_files[f"{_TENSOR_DIRECTORY}/{name}.tensor"],
            context=f"tensor {name}",
        )
        for name in CORE_TENSOR_NAMES
    }
    opened = _open_bundle(
        cache=cache,
        tensors=tensors,
        route_roster=route_roster,
        route_receipts=route_receipts,
        identity_payload=identity_payload,
        registry=registry,
        public_index=public_index,
        alpha=alpha,
        residual_supplier=residual_supplier,
    )
    receipt = _decode_canonical_json(
        raw_files[_MATERIALIZATION_RECEIPT_FILE],
        context="materialization receipt",
    )
    receipt = _validate_materialization_receipt_bindings(
        receipt,
        snapshot=snapshot,
        cache=cache,
        public_index=public_index,
        registry=registry,
        metric_receipt=metric_receipt,
    )
    return opened, receipt


def validate_public_v29_cache_materialization_receipt(
    value: Any,
    *,
    root: str | Path | None = None,
    cache: Mapping[str, Any] | None = None,
    public_index: PublicV29RosterIndex | None = None,
    registry: FrozenV29ResourceRegistry | None = None,
    metric_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the receipt structure, self-hash, and optional disk replay.

    With no ``root`` this is a pure strict structure/self-hash validator.  If
    authorities, a cache, or a metric receipt are supplied, their explicit
    bindings are also checked.  Supplying ``root`` upgrades validation to a
    complete disk reconstruction and frozen-wrapper replay at ``alpha=0``.
    """

    data = _validate_materialization_receipt_structure(value)
    trusted_registry: FrozenV29ResourceRegistry | None = None
    trusted_index: PublicV29RosterIndex | None = None
    if registry is not None or public_index is not None or root is not None:
        trusted_registry, trusted_index = _authorities(registry, public_index)
        if data["resource_config_sha256"] != trusted_registry.config_sha256:
            raise ValueError("materialization receipt resource config binding drifted")
        if data["resource_registry_projection_sha256"] != (
            trusted_index.resource_registry_projection_sha256
        ):
            raise ValueError("materialization receipt registry projection drifted")
        if data["public_roster_authority_sha256"] != trusted_index.authority_sha256:
            raise ValueError("materialization receipt public roster authority drifted")

    if cache is not None:
        if not isinstance(cache, Mapping):
            raise TypeError("cache binding must be a mapping")
        if (
            cache.get("cache_id") != data["cache_id"]
            or cache.get("route") != PUBLIC_OOF_ROUTE
            or cache.get("unit_kind") != "patient"
            or cache.get("unit_count") != PUBLIC_PATIENT_COUNT
            or cache.get("patient_count") != PUBLIC_PATIENT_COUNT
        ):
            raise ValueError("materialization receipt cache binding drifted")

    if metric_receipt is not None:
        if not isinstance(metric_receipt, Mapping):
            raise TypeError("metric_receipt must be a mapping")
        metric_payload = _plain_json(metric_receipt)
        if (
            canonical_json_sha256(metric_payload)
            != data["public_metric_receipt_sha256"]
            or metric_payload.get("source") != "sha_frozen_public_manifest"
            or metric_payload.get("independently_recomputed_from_targets") is not False
            or metric_payload.get("targets_or_target_mask_read") is not False
        ):
            raise ValueError("materialization receipt public metric binding drifted")

    if root is not None:
        if trusted_registry is None or trusted_index is None:  # pragma: no cover
            raise RuntimeError("root replay authorities were not resolved")
        snapshot = _capture_disk_snapshot(root)
        opened, disk_receipt = _open_disk_snapshot(
            snapshot,
            registry=trusted_registry,
            public_index=trusted_index,
            alpha=0.0,
            residual_supplier=None,
        )
        if disk_receipt != data:
            raise ValueError("caller receipt does not equal the fully replayed disk receipt")
        if cache is not None and _plain_json(cache) != _plain_json(opened.cache):
            raise ValueError("caller cache does not equal the fully replayed disk cache")
        if metric_receipt is not None:
            disk_metric = _decode_canonical_json(
                snapshot.raw_files[_PUBLIC_METRIC_RECEIPT_FILE],
                context="public metric receipt",
            )
            if _plain_json(metric_receipt) != disk_metric:
                raise ValueError(
                    "caller metric receipt does not equal the replayed disk receipt"
                )
    return data


def open_public_v29_cache_from_disk(
    destination: str | Path,
    *,
    registry: FrozenV29ResourceRegistry | None = None,
    public_index: PublicV29RosterIndex | None = None,
    alpha: float = 0.0,
    residual_supplier: Callable[[], Any] | None = None,
) -> OpenedV29Cache:
    """Reconstruct and fully replay one published public-only v29 cache."""

    trusted_registry, trusted_index = _authorities(registry, public_index)
    snapshot = _capture_disk_snapshot(destination)
    opened, _receipt = _open_disk_snapshot(
        snapshot,
        registry=trusted_registry,
        public_index=trusted_index,
        alpha=alpha,
        residual_supplier=residual_supplier,
    )
    return opened


def materialize_public_v29_cache_to_disk(
    destination: str | Path,
    *,
    registry: FrozenV29ResourceRegistry | None = None,
    public_index: PublicV29RosterIndex | None = None,
    _write_hook: Callable[[Path], None] | None = None,
) -> PublicV29DiskMaterialization:
    """Build, precommit-replay, and atomically publish the canonical 102 rows.

    This is a trusted-workspace research artifact writer.  The target parent,
    staging directory, writes, snapshot, and rename are all anchored by
    directory descriptors.  The verified no-replace rename is the commit:
    no fallible validation or filesystem operation runs after it succeeds.
    """

    target = _lexical_absolute(destination, context="public v29 cache destination")
    parent_fd = _open_absolute_directory(
        target.parent,
        context="public v29 cache destination parent",
    )
    committed = False
    staging_name: str | None = None
    try:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"public v29 cache destination already exists: {target}"
            )

        trusted_registry, trusted_index = _authorities(registry, public_index)
        identities = _canonical_identities(trusted_index)
        bundle = materialize_public_v29_cache(
            identities,
            registry=trusted_registry,
            public_index=trusted_index,
        )
        initial_open = _open_bundle(
            cache=bundle.cache,
            tensors=bundle.tensor_payloads,
            route_roster=bundle.route_receipt_roster,
            route_receipts=bundle.route_receipts,
            identity_payload=bundle.identity_payload,
            registry=trusted_registry,
            public_index=trusted_index,
            alpha=0.0,
            residual_supplier=None,
        )
        if bundle.event_identity_payload is not None or (
            bundle.montage_derivation_payload is not None
        ):
            raise ValueError("public materializer unexpectedly produced event payloads")

        frozen_manifest, metric_receipt = (
            replay_public_v29_manifest_metric_receipt(trusted_registry)
        )
        del frozen_manifest
        frozen_manifest_raw = frozen_v29._read_verified_resource_bytes(
            trusted_registry.require(PUBLIC_MANIFEST_RESOURCE)
        )
        verify_artifact_content(
            metric_receipt["public_manifest_ref"],
            frozen_manifest_raw,
        )

        raw_payloads: dict[str, bytes] = {
            _CACHE_FILE: canonical_json_bytes(_plain_json(bundle.cache)),
            _IDENTITY_FILE: canonical_json_bytes(_plain_json(bundle.identity_payload)),
            _ROUTE_ROSTER_FILE: canonical_json_bytes(
                _plain_json(bundle.route_receipt_roster)
            ),
            _RESOURCE_CONFIG_FILE: trusted_registry.config_bytes,
            _FROZEN_MANIFEST_FILE: frozen_manifest_raw,
            _PUBLIC_METRIC_RECEIPT_FILE: canonical_json_bytes(
                _plain_json(metric_receipt)
            ),
        }
        raw_payloads.update(
            {
                f"{_TENSOR_DIRECTORY}/{name}.tensor": canonical_tensor_bytes(
                    bundle.tensor_payloads[name]
                )
                for name in CORE_TENSOR_NAMES
            }
        )
        raw_payloads.update(
            {
                f"{_ROUTE_RECEIPT_DIRECTORY}/{index:03d}.json": (
                    canonical_json_bytes(_plain_json(route_receipt))
                )
                for index, route_receipt in enumerate(bundle.route_receipts)
            }
        )
        if set(raw_payloads) != _expected_files() - {
            _MATERIALIZATION_RECEIPT_FILE
        }:
            raise ValueError("precommit raw payload set drifted")

        staging_name = _create_staging_directory_at(parent_fd, target.name)
        staging_path = target.parent / staging_name
        staging_fd = os.open(staging_name, _directory_open_flags(), dir_fd=parent_fd)
        directory_fds: dict[str, int] = {"": staging_fd}
        try:
            for name in (_TENSOR_DIRECTORY, _SIDECAR_DIRECTORY, _AUDIT_DIRECTORY):
                os.mkdir(name, mode=0o700, dir_fd=staging_fd)
                directory_fds[name] = os.open(
                    name,
                    _directory_open_flags(),
                    dir_fd=staging_fd,
                )
            os.mkdir(
                "route_receipts",
                mode=0o700,
                dir_fd=directory_fds[_SIDECAR_DIRECTORY],
            )
            directory_fds[_ROUTE_RECEIPT_DIRECTORY] = os.open(
                "route_receipts",
                _directory_open_flags(),
                dir_fd=directory_fds[_SIDECAR_DIRECTORY],
            )

            for relative, raw in sorted(raw_payloads.items()):
                parent_relative, leaf = relative.rsplit("/", 1) if "/" in relative else ("", relative)
                _write_exclusive_bytes_at(directory_fds[parent_relative], leaf, raw)

            if _write_hook is not None:
                _write_hook(staging_path)

            receipt = _materialization_receipt(
                raw_files=raw_payloads,
                cache=bundle.cache,
                public_index=trusted_index,
                registry=trusted_registry,
                metric_receipt=metric_receipt,
            )
            receipt_raw = canonical_json_bytes(receipt)
            _write_exclusive_bytes_at(
                directory_fds[_AUDIT_DIRECTORY],
                "materialization_receipt.json",
                receipt_raw,
            )
            for relative in (
                _ROUTE_RECEIPT_DIRECTORY,
                _TENSOR_DIRECTORY,
                _SIDECAR_DIRECTORY,
                _AUDIT_DIRECTORY,
                "",
            ):
                _fsync_directory_fd(directory_fds[relative])
        finally:
            for relative, descriptor in sorted(
                directory_fds.items(),
                key=lambda item: item[0].count("/"),
                reverse=True,
            ):
                del relative
                os.close(descriptor)

        staged_snapshot = _capture_disk_snapshot_at(
            parent_fd,
            staging_name,
            root=staging_path,
        )
        staged_open, staged_receipt = _open_disk_snapshot(
            staged_snapshot,
            registry=trusted_registry,
            public_index=trusted_index,
            alpha=0.0,
            residual_supplier=None,
        )
        if staged_receipt != receipt:
            raise ValueError("staged materialization receipt changed during replay")
        result = PublicV29DiskMaterialization(
            path=target,
            cache_id=str(staged_open.cache["cache_id"]),
            opened=staged_open,
            materialization_receipt=_deep_freeze_json(staged_receipt),
        )

        _fsync_directory_fd(parent_fd)
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"public v29 cache destination already exists: {target}"
            )
        _rename_directory_noreplace_at(parent_fd, staging_name, target.name)
        committed = True
        return result
    finally:
        if not committed and staging_name is not None:
            try:
                shutil.rmtree(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            if not committed:
                raise


__all__ = [
    "DISK_MATERIALIZER_VERSION",
    "MATERIALIZATION_RECEIPT_SCHEMA_VERSION",
    "DEFAULT_PUBLIC_CACHE_DIRECTORY",
    "EXPECTED_P0_C18_SHA256",
    "PublicV29DiskMaterialization",
    "validate_public_v29_cache_materialization_receipt",
    "materialize_public_v29_cache_to_disk",
    "open_public_v29_cache_from_disk",
]
