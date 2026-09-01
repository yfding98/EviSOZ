#!/usr/bin/env python3
"""Materialize the privacy-safe private physician-report Stage-0 inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.data.private_physician_reports import (  # noqa: E402
    build_private_physician_report_inventory,
)


DEFAULT_REPORT_ROOT = Path("/mnt/hd1/dyf/dataset/EEG_Reports/Reports")
DEFAULT_SOURCE_MANIFEST = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_SIGNAL_ROSTER = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/signal_roster.csv"
)
DEFAULT_SPLIT_ROSTER = (
    ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_stage0_private_physician_report_inventory_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--signal-roster", type=Path, default=DEFAULT_SIGNAL_ROSTER)
    parser.add_argument("--split-roster", type=Path, default=DEFAULT_SPLIT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    report_root = args.report_root.resolve(strict=True)
    if report_root.is_symlink() or not report_root.is_dir():
        raise ValueError("private physician report root must be a regular directory")
    report_paths = [path for path in report_root.iterdir() if path.is_file()]
    split_roster = json.loads(
        args.split_roster.resolve(strict=True).read_text(encoding="utf-8")
    )
    inventory = build_private_physician_report_inventory(
        report_paths=report_paths,
        source_manifest_path=args.source_manifest.resolve(strict=True),
        signal_roster_path=args.signal_roster.resolve(strict=True),
        split_roster=split_roster,
    )
    args.output.mkdir(parents=True)
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "privacy_safe_physician_report_inventory_materialized",
                "report_count": inventory["counts"]["report_count"],
                "valid_docx_count": inventory["counts"]["valid_docx_count"],
                "association_status_counts": inventory["counts"][
                    "association_status_counts"
                ],
                "association_basis_counts": inventory["counts"][
                    "association_basis_counts"
                ],
                "linked_split_role_counts": inventory["counts"][
                    "linked_split_role_counts"
                ],
                "deidentified_text_release_count": inventory["counts"][
                    "deidentified_text_release_count"
                ],
                "receipt_sha256": inventory["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
