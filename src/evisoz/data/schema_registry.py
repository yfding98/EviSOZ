"""Content-addressed schema/validator registry for EviSOZ artifacts."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .artifact_ref import (
    build_raw_artifact_ref,
    canonical_json_sha256,
    validate_artifact_ref,
    verify_artifact_content,
)


SCHEMA_REGISTRY_SCHEMA_VERSION = "evisoz_schema_registry_v1"

_ID_PREFIX = "EVISOZ-SCHEMAS-"
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_HASH_PLACEHOLDER = "0" * 64
_TOP_KEYS = {"schema_version", "registry_id", "entries", "registry_sha256"}
_ENTRY_KEYS = {
    "bound_schema_version",
    "schema_id",
    "schema_path",
    "schema_artifact_ref",
    "validator",
}
_VALIDATOR_KEYS = {
    "validator_id",
    "implementation_version",
    "module",
    "callable",
    "implementation_path",
    "implementation_artifact_ref",
}


def _strict_json_loads(payload: bytes, context: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _trimmed(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _relative_path(value: object, context: str) -> str:
    text = _trimmed(value, context).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} must be a normalized repository-relative path")
    return path.as_posix()


def _read_repository_file(root: Path, relative_path: str, context: str) -> bytes:
    repository = root.resolve(strict=True)
    candidate = root / relative_path
    if candidate.is_symlink():
        raise ValueError(f"{context} must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{context} escapes the repository root") from exc
    if not resolved.is_file():
        raise ValueError(f"{context} must be a regular file")
    return resolved.read_bytes()


def _schema_const(schema: Mapping[str, object]) -> str | None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    version = properties.get("schema_version")
    if not isinstance(version, Mapping):
        return None
    constant = version.get("const")
    return constant if isinstance(constant, str) else None


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["registry_id"] = _PENDING_ID
    source["registry_sha256"] = _HASH_PLACEHOLDER
    return source


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    source = deepcopy(dict(value))
    source["registry_sha256"] = _HASH_PLACEHOLDER
    return source


def build_schema_registry(
    *,
    repository_root: str | Path,
    bindings: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build a registry from exact repository files.

    Each binding must contain the entry/validator string fields but not the
    two artifact references, which are constructed from the current raw file
    bytes.
    """

    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise TypeError("bindings must be an array")
    root = Path(repository_root)
    entries: list[dict[str, Any]] = []
    required = {
        "bound_schema_version",
        "schema_path",
        "validator_id",
        "implementation_version",
        "module",
        "callable",
        "implementation_path",
    }
    for index, raw in enumerate(bindings):
        if type(raw) is not dict or set(raw) != required:
            raise ValueError(f"bindings[{index}] fields drifted")
        bound_version = _trimmed(raw["bound_schema_version"], f"bindings[{index}].bound_schema_version")
        schema_path = _relative_path(raw["schema_path"], f"bindings[{index}].schema_path")
        implementation_path = _relative_path(
            raw["implementation_path"],
            f"bindings[{index}].implementation_path",
        )
        schema_bytes = _read_repository_file(root, schema_path, f"bindings[{index}].schema_path")
        schema = _strict_json_loads(schema_bytes, f"bindings[{index}].schema")
        Draft202012Validator.check_schema(schema)
        schema_id = _trimmed(schema.get("$id"), f"bindings[{index}].schema.$id")
        if _schema_const(schema) != bound_version:
            raise ValueError(
                f"bindings[{index}] bound version does not match schema_version const"
            )
        implementation_bytes = _read_repository_file(
            root,
            implementation_path,
            f"bindings[{index}].implementation_path",
        )
        entries.append(
            {
                "bound_schema_version": bound_version,
                "schema_id": schema_id,
                "schema_path": schema_path,
                "schema_artifact_ref": build_raw_artifact_ref(
                    schema_bytes,
                    artifact_kind="json_schema",
                    media_type="application/schema+json",
                    payload_schema_version=bound_version,
                ),
                "validator": {
                    "validator_id": _trimmed(raw["validator_id"], f"bindings[{index}].validator_id"),
                    "implementation_version": _trimmed(
                        raw["implementation_version"],
                        f"bindings[{index}].implementation_version",
                    ),
                    "module": _trimmed(raw["module"], f"bindings[{index}].module"),
                    "callable": _trimmed(raw["callable"], f"bindings[{index}].callable"),
                    "implementation_path": implementation_path,
                    "implementation_artifact_ref": build_raw_artifact_ref(
                        implementation_bytes,
                        artifact_kind="validator_implementation",
                        media_type="text/x-python",
                    ),
                },
            }
        )
    entries.sort(key=lambda row: row["bound_schema_version"])
    body: dict[str, Any] = {
        "schema_version": SCHEMA_REGISTRY_SCHEMA_VERSION,
        "registry_id": _PENDING_ID,
        "entries": entries,
        "registry_sha256": _HASH_PLACEHOLDER,
    }
    body["registry_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["registry_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_schema_registry(body, repository_root=root)


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def validate_schema_registry(
    value: object,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a registry and optionally replay every repository file."""

    if type(value) is not dict or set(value) != _TOP_KEYS:
        raise ValueError("schema registry fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != SCHEMA_REGISTRY_SCHEMA_VERSION:
        raise ValueError("schema registry schema_version drifted")
    entries = data["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("schema registry entries must be non-empty")
    versions: list[str] = []
    schema_ids: list[str] = []
    validator_ids: list[str] = []
    root = Path(repository_root) if repository_root is not None else None
    for index, row in enumerate(entries):
        if type(row) is not dict or set(row) != _ENTRY_KEYS:
            raise ValueError(f"entries[{index}] fields drifted")
        version = _trimmed(row["bound_schema_version"], f"entries[{index}].bound_schema_version")
        schema_id = _trimmed(row["schema_id"], f"entries[{index}].schema_id")
        schema_path = _relative_path(row["schema_path"], f"entries[{index}].schema_path")
        schema_ref = validate_artifact_ref(row["schema_artifact_ref"])
        if schema_ref["artifact_kind"] != "json_schema" or schema_ref["content_hash"]["domain"] != "raw_bytes_v1":
            raise ValueError(f"entries[{index}] schema must bind raw JSON-schema bytes")
        if schema_ref["payload_schema_version"] != version:
            raise ValueError(f"entries[{index}] schema reference version drifted")
        validator = row["validator"]
        if type(validator) is not dict or set(validator) != _VALIDATOR_KEYS:
            raise ValueError(f"entries[{index}].validator fields drifted")
        validator_id = _trimmed(validator["validator_id"], f"entries[{index}].validator_id")
        for key in ("implementation_version", "module", "callable"):
            _trimmed(validator[key], f"entries[{index}].validator.{key}")
        implementation_path = _relative_path(
            validator["implementation_path"],
            f"entries[{index}].validator.implementation_path",
        )
        implementation_ref = validate_artifact_ref(validator["implementation_artifact_ref"])
        if implementation_ref["artifact_kind"] != "validator_implementation" or implementation_ref["content_hash"]["domain"] != "raw_bytes_v1":
            raise ValueError(f"entries[{index}] validator must bind raw implementation bytes")
        if root is not None:
            schema_bytes = _read_repository_file(root, schema_path, f"entries[{index}].schema_path")
            verify_artifact_content(schema_ref, schema_bytes)
            schema = _strict_json_loads(schema_bytes, f"entries[{index}].schema")
            Draft202012Validator.check_schema(schema)
            if schema.get("$id") != schema_id or _schema_const(schema) != version:
                raise ValueError(f"entries[{index}] schema identity/version drifted")
            implementation_bytes = _read_repository_file(
                root,
                implementation_path,
                f"entries[{index}].validator.implementation_path",
            )
            verify_artifact_content(implementation_ref, implementation_bytes)
        versions.append(version)
        schema_ids.append(schema_id)
        validator_ids.append(validator_id)
    if versions != sorted(versions) or len(versions) != len(set(versions)):
        raise ValueError("schema registry versions must be unique and sorted")
    if len(schema_ids) != len(set(schema_ids)):
        raise ValueError("schema registry schema IDs must be unique")
    if len(validator_ids) != len(set(validator_ids)):
        raise ValueError("schema registry validator IDs must be unique")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["registry_id"] != expected_id:
        raise ValueError("schema registry_id does not bind its content")
    if data["registry_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("schema registry_sha256 does not bind its content")
    return data


def validate_registered_payload(
    payload: object,
    *,
    schema_registry: Mapping[str, object],
    repository_root: str | Path,
    validator_dispatch: Mapping[str, Callable[..., object]],
    validator_kwargs_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> object:
    """Run both the bound JSON Schema and its registered semantic validator."""

    _reject_nonfinite(payload)
    registry = validate_schema_registry(
        schema_registry,
        repository_root=repository_root,
    )
    if type(payload) is not dict:
        raise TypeError("registered payload must be an object")
    version = payload.get("schema_version")
    entry = next(
        (row for row in registry["entries"] if row["bound_schema_version"] == version),
        None,
    )
    if entry is None:
        raise ValueError(f"unregistered schema_version: {version!r}")
    root = Path(repository_root)
    resources: list[tuple[str, Resource[Any]]] = []
    schemas: dict[str, dict[str, Any]] = {}
    for row in registry["entries"]:
        schema_bytes = _read_repository_file(root, row["schema_path"], "registered schema")
        schema = _strict_json_loads(schema_bytes, "registered schema")
        schemas[row["bound_schema_version"]] = schema
        resources.append((str(schema["$id"]), Resource.from_contents(schema)))
    resource_registry = Registry().with_resources(resources)
    schema = schemas[str(version)]
    errors = sorted(
        Draft202012Validator(schema, registry=resource_registry).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "$"
        raise ValueError(f"registered JSON-schema validation failed at {path}: {error.message}")
    validator_id = entry["validator"]["validator_id"]
    validator = validator_dispatch.get(validator_id)
    if validator is None:
        raise ValueError(f"registered semantic validator is unavailable: {validator_id}")
    validator_binding = entry["validator"]
    registered_module = validator_binding["module"]
    registered_callable = validator_binding["callable"]
    if (
        getattr(validator, "__module__", None) != registered_module
        or getattr(validator, "__name__", None) != registered_callable
    ):
        raise ValueError(
            "registered semantic validator dispatch identity drifted from "
            f"{registered_module}.{registered_callable}"
        )
    source_file = inspect.getsourcefile(validator)
    if source_file is None:
        raise ValueError("registered semantic validator has no inspectable source file")
    try:
        runtime_source = Path(source_file).resolve(strict=True)
        expected_source = (
            root / validator_binding["implementation_path"]
        ).resolve(strict=True)
    except OSError as exc:
        raise ValueError("registered semantic validator source path is unavailable") from exc
    if runtime_source != expected_source:
        raise ValueError(
            "registered semantic validator runtime source path drifted from its binding"
        )
    implementation_bytes = _read_repository_file(
        root,
        validator_binding["implementation_path"],
        "registered semantic validator implementation",
    )
    verify_artifact_content(
        validator_binding["implementation_artifact_ref"],
        implementation_bytes,
    )
    kwargs = (
        dict(validator_kwargs_by_id.get(validator_id, {}))
        if validator_kwargs_by_id is not None
        else {}
    )
    return validator(payload, **kwargs)


__all__ = [
    "SCHEMA_REGISTRY_SCHEMA_VERSION",
    "build_schema_registry",
    "validate_schema_registry",
    "validate_registered_payload",
]
