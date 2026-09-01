#!/usr/bin/env python3
"""Export EventNet identities from a frozen complete TUSZ roster receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_tusz_complete_identity_projection_v1 import (  # noqa: E402
    build_eventnet_tusz_complete_identity_projection_v1,
)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("complete roster must be a regular non-symlink JSON file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("complete roster JSON is empty")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    if type(value) is not dict:
        raise ValueError("complete roster JSON must contain an object")
    return value


def _write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise ValueError("output already exists; identity projections are append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ValueError("output appeared during identity projection")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a validated complete TUSZ roster into a reference-free "
            "EventNet identity projection"
        )
    )
    parser.add_argument("--complete-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    roster = _read_json_object(arguments.complete_roster)
    projection = build_eventnet_tusz_complete_identity_projection_v1(roster)
    _write_new_json(arguments.output, projection)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "projection_id": projection["projection_id"],
                "receipt_sha256": projection["receipt_sha256"],
                "recording_count": projection["source_roster_binding"][
                    "source_recording_count"
                ],
                "reference_files_opened": projection[
                    "reference_access_receipt"
                ]["reference_files_opened"],
                "source_eval_model_execution_authorized": projection[
                    "split_permissions"
                ]["source_eval"]["eventnet_model_execution_authorized"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
