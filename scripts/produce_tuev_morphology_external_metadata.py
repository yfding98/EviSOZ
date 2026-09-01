#!/usr/bin/env python3
"""Produce canonical first-party TUEV morphology signal metadata.

The command always performs the complete EDF/REC/LAB/HTK audit.  ``--dry-audit``
suppresses artifact publication only; it is not a header-only or sampled scan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.tuev_morphology_signal_preflight import (  # noqa: E402
    produce_tuev_morphology_external_metadata,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read every real TUEV EDF/REC/LAB/HTK parent group and produce "
            "replayable standard-19 header/signal metadata"
        )
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-json", type=Path)
    destination.add_argument(
        "--dry-audit",
        action="store_true",
        help="perform the full signal audit without writing metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    production = produce_tuev_morphology_external_metadata(
        args.edf_root,
        output_path=None if args.dry_audit else args.output_json,
    )
    print(
        json.dumps(
            production.summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
