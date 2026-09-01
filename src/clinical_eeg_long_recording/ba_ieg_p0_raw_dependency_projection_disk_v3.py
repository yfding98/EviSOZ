"""Deterministic compressed transport for projection-v2 raw dependencies.

Disk-v2 deliberately persists a large canonical-JSON provenance artifact.
The object is highly repetitive (one fully self-contained dependency row per
P0 token), so storing the JSON verbatim is unnecessarily expensive for the A0
908-event materialization.  Disk-v3 changes only the transport:

* the semantic artifact is still the exact disk-v2 canonical JSON object;
* its disk-v2 ``artifact_id`` and ``artifact_sha256`` are unchanged;
* the exact canonical JSON byte length and SHA-256 are independently bound;
* those bytes are carried in one deterministic, filename-free gzip member;
* the detached v3 reference binds both compressed and decompressed bytes; and
* loading reconstructs the artifact from independent host authority and
  requires byte-for-byte equality after decompression.

This is not a new model input, a target, or a lossy columnar projection.  It
contains no pickle payload and is never available to batch packing or model
forward.  Existing disk-v2 ``.json`` artifacts remain valid and resumable.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import tempfile
from typing import Any, Final, Mapping
import zlib

from .ba_ieg_event_model_input_projection_v2 import (
    BAIEGEventModelInputProjectionV2,
)
from .ba_ieg_p0_raw_dependency_projection_disk_v2 import (
    BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2,
    BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2,
    materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2,
)


BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V3: Final[
    str
] = "ba_ieg_p0_raw_dependency_projection_disk_compressed_reference_v3"
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V3: Final[str] = (
    "ba_ieg_projection_v2_raw_dependency_exact_canonical_json_gzip_disk_v3"
)
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3: Final[
    int
] = BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_BYTES_V2
BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_COMPRESSED_BYTES_V3: Final[
    int
] = BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3 + (
    1024 * 1024
)

_REFERENCE_ID_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-compressed-reference-id-v3"
_REFERENCE_SHA_DOMAIN: Final[
    str
] = "ba-ieg-p0-raw-dependency-projection-compressed-reference-sha-v3"
_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")

# RFC 1952: ID1, ID2, CM=deflate, FLG=0, MTIME=0, XFL=maximum compression,
# OS=unknown.  Building the header ourselves avoids Python-version-specific OS
# header bytes while zlib provides the raw DEFLATE stream and CRC verification.
_GZIP_HEADER: Final[bytes] = bytes.fromhex("1f8b08000000000002ff")
_TRANSPORT_RECEIPT: Final[dict[str, Any]] = {
    "transport": "rfc1952_gzip_single_member",
    "compression_method": "raw_deflate",
    "compression_level": 9,
    "compression_memory_level": 9,
    "compression_strategy": "zlib_default_strategy",
    "gzip_header_hex": _GZIP_HEADER.hex(),
    "mtime_seconds": 0,
    "original_filename_present": False,
    "optional_gzip_header_fields_present": False,
    "trailing_or_concatenated_members_allowed": False,
    "decompressed_encoding": "utf-8",
    "decompressed_serialization": (
        "canonical_json_sort_keys_compact_ascii_allow_nan_false"
    ),
    "decompressed_artifact_schema_version": (
        BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2
    ),
    "disk_v2_artifact_identity_preserved": True,
    "lossy_projection_used": False,
    "pickle_used": False,
    "available_to_model_forward": False,
    "available_to_batch_packing": False,
}
_REFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "method_id",
        "reference_id",
        "reference_sha256",
        "relative_path",
        "file_size_bytes",
        "file_sha256",
        "canonical_json_size_bytes",
        "canonical_json_sha256",
        "artifact_schema_version",
        "artifact_id",
        "artifact_sha256",
        "source_binding",
        "transport_receipt",
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


def _identifier(value: object, name: str, *, prefix: str | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a valid non-empty trimmed identifier")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _relative_gzip_json_path(value: object) -> str:
    text = _identifier(value, "relative_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or not text.endswith(".json.gz")
        or "\\" in text
    ):
        raise ValueError(
            "relative_path must be a canonical relative .json.gz path"
        )
    return text


def _finalize_reference(body: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result["reference_id"] = "CONTENT-ADDRESS-PENDING"
    result["reference_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["reference_id"] = (
        "P0-RAWDEP-GZIPREF-"
        + _canonical_sha256(
            {"domain": _REFERENCE_ID_DOMAIN, "reference": result}
        )[:24]
    )
    result["reference_sha256"] = _canonical_sha256(
        {"domain": _REFERENCE_SHA_DOMAIN, "reference": result}
    )
    return result


def _validate_reference_shape(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _REFERENCE_KEYS:
        raise ValueError(
            "compressed raw-dependency reference has missing/unknown fields"
        )
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V3
        or data["method_id"]
        != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V3
        or data["artifact_schema_version"]
        != BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_ARTIFACT_SCHEMA_VERSION_V2
        or data["transport_receipt"] != _TRANSPORT_RECEIPT
    ):
        raise ValueError("compressed raw-dependency transport contract drifted")
    _relative_gzip_json_path(data["relative_path"])
    compressed_size = _positive_integer(
        data["file_size_bytes"], "file_size_bytes"
    )
    canonical_size = _positive_integer(
        data["canonical_json_size_bytes"], "canonical_json_size_bytes"
    )
    if (
        compressed_size
        > BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_COMPRESSED_BYTES_V3
        or canonical_size
        > BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3
    ):
        raise ValueError("compressed raw-dependency byte bound was exceeded")
    for name in (
        "file_sha256",
        "canonical_json_sha256",
        "artifact_sha256",
    ):
        _sha256(data[name], name)
    _identifier(data["artifact_id"], "artifact_id", prefix="P0-RAWDEP-DISK-")
    if type(data["source_binding"]) is not dict:
        raise ValueError("compressed raw-dependency source binding must be an object")
    body = deepcopy(data)
    supplied_id = _identifier(
        body.pop("reference_id"),
        "reference_id",
        prefix="P0-RAWDEP-GZIPREF-",
    )
    supplied_sha = _sha256(
        body.pop("reference_sha256"), "reference_sha256"
    )
    expected = _finalize_reference(body)
    if (
        supplied_id != expected["reference_id"]
        or supplied_sha != expected["reference_sha256"]
    ):
        raise ValueError(
            "compressed raw-dependency reference content hash drifted"
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


def _expected_artifact(
    projection: BAIEGEventModelInputProjectionV2,
    *,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    _require_expected_host_hashes(
        projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=(
            expected_projection_v2_receipt_sha256
        ),
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
    )
    return materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )


def _deterministic_gzip_bytes(canonical_json: bytes) -> bytes:
    if not canonical_json:
        raise ValueError("canonical raw-dependency JSON cannot be empty")
    compressor = zlib.compressobj(
        level=9,
        method=zlib.DEFLATED,
        wbits=-zlib.MAX_WBITS,
        memLevel=9,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    deflated = compressor.compress(canonical_json) + compressor.flush(
        zlib.Z_FINISH
    )
    trailer = struct.pack(
        "<II",
        zlib.crc32(canonical_json) & 0xFFFFFFFF,
        len(canonical_json) & 0xFFFFFFFF,
    )
    return _GZIP_HEADER + deflated + trailer


def _decompress_deterministic_gzip_bytes(payload: bytes) -> bytes:
    if len(payload) < len(_GZIP_HEADER) + 8 or not payload.startswith(
        _GZIP_HEADER
    ):
        raise ValueError("compressed raw dependency has a non-frozen gzip header")
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        canonical = decompressor.decompress(
            payload,
            BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3
            + 1,
        )
    except zlib.error as exc:
        raise ValueError("compressed raw dependency is not valid gzip") from exc
    if (
        len(canonical)
        > BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3
        or decompressor.unconsumed_tail
    ):
        raise ValueError("decompressed raw dependency exceeds its byte bound")
    try:
        tail = decompressor.flush()
    except zlib.error as exc:
        raise ValueError("compressed raw dependency gzip trailer is invalid") from exc
    canonical += tail
    if len(canonical) > (
        BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3
    ):
        raise ValueError("decompressed raw dependency exceeds its byte bound")
    if not decompressor.eof:
        raise ValueError("compressed raw dependency is truncated")
    if decompressor.unused_data:
        raise ValueError(
            "compressed raw dependency has a trailing or concatenated member"
        )
    return canonical


def _reference_from_artifact(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
    canonical_json: bytes,
    compressed_payload: bytes,
) -> dict[str, Any]:
    body = {
        "schema_version": (
            BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V3
        ),
        "method_id": BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V3,
        "relative_path": _relative_gzip_json_path(relative_path),
        "file_size_bytes": len(compressed_payload),
        "file_sha256": _bytes_sha256(compressed_payload),
        "canonical_json_size_bytes": len(canonical_json),
        "canonical_json_sha256": _bytes_sha256(canonical_json),
        "artifact_schema_version": artifact["schema_version"],
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "source_binding": deepcopy(artifact["source_binding"]),
        "transport_receipt": deepcopy(_TRANSPORT_RECEIPT),
    }
    return _finalize_reference(body)


def _validate_reference_against_artifact(
    payload: object,
    *,
    artifact: Mapping[str, Any],
    canonical_json: bytes,
) -> dict[str, Any]:
    data = _validate_reference_shape(payload)
    expected = {
        "artifact_schema_version": artifact["schema_version"],
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "source_binding": artifact["source_binding"],
        "canonical_json_size_bytes": len(canonical_json),
        "canonical_json_sha256": _bytes_sha256(canonical_json),
    }
    for name, value in expected.items():
        if data[name] != value:
            raise ValueError(
                f"compressed raw-dependency reference {name} drifted "
                "from host authority"
            )
    return data


def validate_ba_ieg_p0_raw_dependency_projection_disk_reference_v3(
    payload: object,
    *,
    projection: BAIEGEventModelInputProjectionV2,
    expected_event_model_input_receipt_sha256: str,
    expected_projection_v2_receipt_sha256: str,
    expected_source_p0_materialization_receipt_sha256: str,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Validate v3 metadata against an independently replayed v2 artifact."""

    artifact = _expected_artifact(
        projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=(
            expected_projection_v2_receipt_sha256
        ),
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    canonical = _canonical_json_bytes(artifact)
    return _validate_reference_against_artifact(
        payload, artifact=artifact, canonical_json=canonical
    )


def _resolve_root(root: os.PathLike[str] | str) -> Path:
    path = Path(root).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("compressed disk sidecar root must be a directory")
    return path


def _resolve_destination(
    root: os.PathLike[str] | str, relative_path: str
) -> tuple[Path, Path]:
    root_path = _resolve_root(root)
    relative = _relative_gzip_json_path(relative_path)
    logical = root_path / relative
    parent = logical.parent.resolve(strict=True)
    if parent != root_path and root_path not in parent.parents:
        raise ValueError("compressed disk sidecar path escapes its root")
    if not parent.is_dir():
        raise ValueError("compressed disk sidecar parent must be a directory")
    return parent, parent / logical.name


def _fsync_directory(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(payload: bytes, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(
            f"compressed disk sidecar already exists: {destination.name}"
        )
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


def _read_stable_single_link_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                "compressed disk sidecar must be a single-link regular file"
            )
        size = _positive_integer(before.st_size, "compressed sidecar file size")
        if size > (
            BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_COMPRESSED_BYTES_V3
        ):
            raise ValueError("compressed disk sidecar exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("compressed disk sidecar was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("compressed disk sidecar grew during loading")
        after = os.fstat(descriptor)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, name) != getattr(after, name):
                raise ValueError("compressed disk sidecar changed during loading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3(
    root: os.PathLike[str] | str,
    relative_path: str,
    projection: BAIEGEventModelInputProjectionV2,
    *,
    canonical_signal_receipt: object,
    trusted_view_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Publish the exact disk-v2 canonical artifact in deterministic gzip."""

    _parent, destination = _resolve_destination(root, relative_path)
    artifact = materialize_ba_ieg_p0_raw_dependency_projection_disk_artifact_v2(
        projection,
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    canonical = _canonical_json_bytes(artifact)
    if len(canonical) > (
        BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3
    ):
        raise ValueError("canonical raw-dependency artifact exceeds its byte bound")
    compressed = _deterministic_gzip_bytes(canonical)
    if len(compressed) > (
        BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_COMPRESSED_BYTES_V3
    ):
        raise ValueError("compressed raw-dependency artifact exceeds its byte bound")
    if _decompress_deterministic_gzip_bytes(compressed) != canonical:
        raise RuntimeError(
            "deterministic gzip did not replay the canonical artifact exactly"
        )
    reference = _reference_from_artifact(
        artifact,
        relative_path=relative_path,
        canonical_json=canonical,
        compressed_payload=compressed,
    )
    _validate_reference_against_artifact(
        reference, artifact=artifact, canonical_json=canonical
    )
    _publish_no_clobber(compressed, destination)
    if _read_stable_single_link_file(destination) != compressed:
        raise ValueError("published compressed sidecar bytes changed unexpectedly")
    return reference


def load_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3(
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
    """Verify compressed bytes and exact canonical replay from host roots."""

    artifact = _expected_artifact(
        projection,
        expected_event_model_input_receipt_sha256=(
            expected_event_model_input_receipt_sha256
        ),
        expected_projection_v2_receipt_sha256=(
            expected_projection_v2_receipt_sha256
        ),
        expected_source_p0_materialization_receipt_sha256=(
            expected_source_p0_materialization_receipt_sha256
        ),
        canonical_signal_receipt=canonical_signal_receipt,
        trusted_view_receipts=trusted_view_receipts,
    )
    expected_canonical = _canonical_json_bytes(artifact)
    reference = _validate_reference_against_artifact(
        detached_reference,
        artifact=artifact,
        canonical_json=expected_canonical,
    )
    _parent, destination = _resolve_destination(
        root, reference["relative_path"]
    )
    compressed = _read_stable_single_link_file(destination)
    if len(compressed) != reference["file_size_bytes"]:
        raise ValueError("compressed disk sidecar size drifted from its reference")
    if _bytes_sha256(compressed) != reference["file_sha256"]:
        raise ValueError(
            "compressed disk sidecar SHA-256 drifted from its reference"
        )
    canonical = _decompress_deterministic_gzip_bytes(compressed)
    if len(canonical) != reference["canonical_json_size_bytes"]:
        raise ValueError(
            "decompressed canonical JSON size drifted from its reference"
        )
    if _bytes_sha256(canonical) != reference["canonical_json_sha256"]:
        raise ValueError(
            "decompressed canonical JSON SHA-256 drifted from its reference"
        )
    if canonical != expected_canonical:
        raise ValueError(
            "decompressed raw dependency did not replay byte-exactly from "
            "host authority"
        )
    return artifact


__all__ = [
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_CANONICAL_BYTES_V3",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_MAXIMUM_COMPRESSED_BYTES_V3",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_METHOD_ID_V3",
    "BA_IEG_P0_RAW_DEPENDENCY_PROJECTION_DISK_REFERENCE_SCHEMA_VERSION_V3",
    "load_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3",
    "validate_ba_ieg_p0_raw_dependency_projection_disk_reference_v3",
    "write_ba_ieg_p0_raw_dependency_projection_disk_sidecar_v3",
]
