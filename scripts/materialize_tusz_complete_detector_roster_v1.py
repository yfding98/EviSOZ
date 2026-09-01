#!/usr/bin/env python3
"""Materialize the complete reference-free TUSZ detector denominator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v1 import (  # noqa: E402
    TUSZ_V203_EXPECTED_INVENTORY,
    build_tusz_complete_detector_roster_v1,
)


DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")


def _load_expected_inventory(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return TUSZ_V203_EXPECTED_INVENTORY
    raw = path.read_bytes()
    if not raw:
        raise ValueError("expected inventory JSON is empty")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("expected inventory JSON must be an object")
    return parsed


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError("output already exists; detector rosters are append-only")
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
            raise ValueError("output appeared during materialization")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash and freeze every TUSZ EDF while retaining no reference labels"
        )
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--expected-inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.preflight_only == (arguments.output is not None):
        raise SystemExit("choose exactly one of --preflight-only or --output")
    roster = build_tusz_complete_detector_roster_v1(
        tusz_root=arguments.tusz_root,
        expected_inventory=_load_expected_inventory(arguments.expected_inventory),
    )
    if arguments.preflight_only:
        summary = {
            "roster_id": roster["roster_id"],
            "receipt_sha256": roster["receipt_sha256"],
            "observed_inventory": roster["observed_inventory"],
            "exact_container_duplicate_audit": roster[
                "exact_container_duplicate_audit"
            ],
            "scope_receipt": roster["scope_receipt"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        assert arguments.output is not None
        _write_new_json(arguments.output, roster)
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "roster_id": roster["roster_id"],
                    "receipt_sha256": roster["receipt_sha256"],
                    "recording_count": roster["observed_inventory"][
                        "total_recording_count"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
