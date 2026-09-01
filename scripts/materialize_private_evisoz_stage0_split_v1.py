#!/usr/bin/env python3
"""Freeze the target-free patient-level private EviSOZ split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_stage0_split import (  # noqa: E402
    build_private_stage0_split,
)


DEFAULT_ROSTER = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--signal-roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    roster_path = args.signal_roster.resolve(strict=True)
    with roster_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    split, summary = build_private_stage0_split(
        rows,
        signal_roster_sha256=_sha256(roster_path),
    )
    args.output.mkdir(parents=True)
    (args.output / "split_roster.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "patient_count": summary["patient_count"],
                "event_count": summary["event_count"],
                "role_patient_counts": summary["role_patient_counts"],
                "development_outer_fold_patient_counts": summary[
                    "development_outer_fold_patient_counts"
                ],
                "receipt_sha256": summary["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
