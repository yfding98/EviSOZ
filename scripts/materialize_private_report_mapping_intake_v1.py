#!/usr/bin/env python3
"""Materialize a privacy-safe intake for unresolved physician-report links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_report_mapping_intake import (  # noqa: E402
    build_private_report_mapping_intake,
)


DEFAULT_INVENTORY = ROOT / "outputs/evisoz_stage0_private_physician_report_inventory_v1_20260831/inventory.json"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_private_report_mapping_intake_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--split-roster", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    split = json.loads(args.split_roster.read_text(encoding="utf-8"))
    intake = build_private_report_mapping_intake(
        report_inventory=inventory,
        split_roster=split,
    )
    args.output.mkdir(parents=True)
    (args.output / "intake.json").write_text(
        json.dumps(intake, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": intake["status"],
        "intake_id": intake["intake_id"],
        "counts": intake["counts"],
        "training_authorized": intake["permissions"]["training_authorized"],
        "receipt_sha256": intake["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
