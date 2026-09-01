#!/usr/bin/env python3
"""Materialize or verify the content-addressed EviSOZ schema registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.artifact_ref import canonical_json_bytes  # noqa: E402
from src.evisoz.data.schema_registry import build_schema_registry  # noqa: E402


DEFAULT_BINDINGS = ROOT / "configs/evisoz_schema_bindings_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/evisoz_schema_registry_v1.json"


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if type(value) is not dict or set(value) != {"schema_version", "bindings"}:
        raise ValueError("schema binding spec fields drifted")
    if value["schema_version"] != "evisoz_schema_binding_spec_v1":
        raise ValueError("schema binding spec version drifted")
    if not isinstance(value["bindings"], list) or not value["bindings"]:
        raise ValueError("schema binding spec must be non-empty")
    return value


def _render(registry: dict[str, Any]) -> bytes:
    # Human-readable checked-in bytes are not the registry identity domain;
    # registry_sha256 already binds the canonical JSON value.
    return (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("registry output must not be a symbolic link")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    spec = _strict_json(args.bindings)
    registry = build_schema_registry(
        repository_root=ROOT,
        bindings=spec["bindings"],
    )
    rendered = _render(registry)
    if args.stdout:
        print(rendered.decode("utf-8"), end="")
        return 0
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            raise SystemExit("EviSOZ schema registry is missing or stale")
        return 0
    _atomic_write(args.output, rendered)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "registry_id": registry["registry_id"],
                "registry_sha256": registry["registry_sha256"],
                "canonical_registry_size_bytes": len(canonical_json_bytes(registry)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
