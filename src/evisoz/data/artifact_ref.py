"""Content-addressed artifact references used by EviSOZ Stage 0.

Two hash domains are deliberately distinct:

``raw_bytes_v1``
    Binds the exact bytes of files such as schemas, source code, tensors and
    checkpoints.

``canonical_json_v1``
    Binds a JSON value after strict canonical serialization.  This domain is
    suitable for receipts and ledgers, but must never be substituted for a
    raw-file hash.

Artifact references contain no filesystem path.  A host manifest may attach
a repository-relative location, but identity is determined solely by the
closed reference and its full content digest.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping


ARTIFACT_REF_SCHEMA_VERSION = "evisoz_artifact_ref_v1"
RAW_BYTES_HASH_DOMAIN = "raw_bytes_v1"
CANONICAL_JSON_HASH_DOMAIN = "canonical_json_v1"
HASH_DOMAINS = (RAW_BYTES_HASH_DOMAIN, CANONICAL_JSON_HASH_DOMAIN)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SCHEMA_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ID_PREFIX = "EVISOZ-ART-"
_REF_HASH_PLACEHOLDER = "0" * 64
_REQUIRED_KEYS = {
    "schema_version",
    "artifact_id",
    "artifact_kind",
    "media_type",
    "content_hash",
    "payload_schema_version",
    "ref_sha256",
}


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value in the frozen EviSOZ canonical domain."""

    _reject_nonfinite(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("value is not representable as strict JSON") from exc


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _trimmed(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _artifact_kind(value: object) -> str:
    result = _trimmed(value, "artifact_kind")
    if _KIND_RE.fullmatch(result) is None:
        raise ValueError("artifact_kind must be a lowercase stable identifier")
    return result


def _payload_schema_version(value: object) -> str | None:
    if value is None:
        return None
    result = _trimmed(value, "payload_schema_version")
    if _SCHEMA_VERSION_RE.fullmatch(result) is None:
        raise ValueError("payload_schema_version is not a stable identifier")
    return result


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _content_hash(*, domain: str, payload: bytes) -> dict[str, object]:
    if domain not in HASH_DOMAINS:
        raise ValueError(f"unsupported artifact hash domain: {domain!r}")
    return {
        "domain": domain,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _artifact_id_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["artifact_id"] = "CONTENT-ADDRESS-PENDING"
    source["ref_sha256"] = _REF_HASH_PLACEHOLDER
    return source


def _finalize_reference(body: dict[str, Any]) -> dict[str, Any]:
    body["artifact_id"] = _ID_PREFIX + canonical_json_sha256(
        _artifact_id_source(body)
    )[:24]
    body["ref_sha256"] = _REF_HASH_PLACEHOLDER
    body["ref_sha256"] = canonical_json_sha256(body)
    return validate_artifact_ref(body)


def build_raw_artifact_ref(
    payload: bytes,
    *,
    artifact_kind: str,
    media_type: str,
    payload_schema_version: str | None = None,
) -> dict[str, Any]:
    """Build a reference for exact raw bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("raw artifact payload must be bytes")
    body: dict[str, Any] = {
        "schema_version": ARTIFACT_REF_SCHEMA_VERSION,
        "artifact_id": "CONTENT-ADDRESS-PENDING",
        "artifact_kind": _artifact_kind(artifact_kind),
        "media_type": _trimmed(media_type, "media_type"),
        "content_hash": _content_hash(domain=RAW_BYTES_HASH_DOMAIN, payload=payload),
        "payload_schema_version": _payload_schema_version(payload_schema_version),
        "ref_sha256": _REF_HASH_PLACEHOLDER,
    }
    return _finalize_reference(body)


def build_json_artifact_ref(
    payload: object,
    *,
    artifact_kind: str,
    payload_schema_version: str,
    media_type: str = "application/json",
) -> dict[str, Any]:
    """Build a reference for a strict canonical-JSON payload."""

    encoded = canonical_json_bytes(payload)
    body: dict[str, Any] = {
        "schema_version": ARTIFACT_REF_SCHEMA_VERSION,
        "artifact_id": "CONTENT-ADDRESS-PENDING",
        "artifact_kind": _artifact_kind(artifact_kind),
        "media_type": _trimmed(media_type, "media_type"),
        "content_hash": _content_hash(
            domain=CANONICAL_JSON_HASH_DOMAIN,
            payload=encoded,
        ),
        "payload_schema_version": _payload_schema_version(payload_schema_version),
        "ref_sha256": _REF_HASH_PLACEHOLDER,
    }
    return _finalize_reference(body)


def validate_artifact_ref(value: object) -> dict[str, Any]:
    """Validate a closed reference and return a defensive copy."""

    if type(value) is not dict:
        raise TypeError("artifact reference must be an object")
    if set(value) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS.difference(value))
        unknown = sorted(set(value).difference(_REQUIRED_KEYS))
        raise ValueError(
            "artifact reference fields drifted; "
            f"missing={missing}, unknown={unknown}"
        )
    data = deepcopy(value)
    if data["schema_version"] != ARTIFACT_REF_SCHEMA_VERSION:
        raise ValueError("artifact reference schema_version drifted")
    kind = _artifact_kind(data["artifact_kind"])
    _trimmed(data["media_type"], "media_type")
    _payload_schema_version(data["payload_schema_version"])

    content_hash = data["content_hash"]
    if type(content_hash) is not dict or set(content_hash) != {
        "domain",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("artifact content_hash fields drifted")
    if content_hash["domain"] not in HASH_DOMAINS:
        raise ValueError("artifact content hash domain is unsupported")
    content_sha256 = _sha256(content_hash["sha256"], "content_hash.sha256")
    size_bytes = content_hash["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise TypeError("content_hash.size_bytes must be an integer >= 0")

    expected_id = _ID_PREFIX + canonical_json_sha256(_artifact_id_source(data))[:24]
    if data["artifact_id"] != expected_id:
        raise ValueError("artifact_id does not bind the full content digest")
    _sha256(data["ref_sha256"], "ref_sha256")
    digest_source = deepcopy(data)
    digest_source["ref_sha256"] = _REF_HASH_PLACEHOLDER
    if data["ref_sha256"] != canonical_json_sha256(digest_source):
        raise ValueError("ref_sha256 does not bind the closed reference")
    data["artifact_kind"] = kind
    return data


def verify_artifact_content(reference: object, payload: object) -> dict[str, Any]:
    """Verify content in the exact hash domain declared by ``reference``."""

    data = validate_artifact_ref(reference)
    domain = data["content_hash"]["domain"]
    if domain == RAW_BYTES_HASH_DOMAIN:
        if not isinstance(payload, bytes):
            raise TypeError("raw_bytes_v1 verification requires bytes")
        encoded = payload
    elif domain == CANONICAL_JSON_HASH_DOMAIN:
        encoded = canonical_json_bytes(payload)
    else:  # pragma: no cover - validate_artifact_ref already closes this
        raise ValueError("unsupported artifact hash domain")
    if len(encoded) != data["content_hash"]["size_bytes"]:
        raise ValueError("artifact byte size does not match the reference")
    if sha256_bytes(encoded) != data["content_hash"]["sha256"]:
        raise ValueError("artifact content digest does not match the reference")
    return data


__all__ = [
    "ARTIFACT_REF_SCHEMA_VERSION",
    "RAW_BYTES_HASH_DOMAIN",
    "CANONICAL_JSON_HASH_DOMAIN",
    "HASH_DOMAINS",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "sha256_bytes",
    "build_raw_artifact_ref",
    "build_json_artifact_ref",
    "validate_artifact_ref",
    "verify_artifact_content",
]
