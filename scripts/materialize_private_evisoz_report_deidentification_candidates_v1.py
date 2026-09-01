#!/usr/bin/env python3
"""Materialize non-trainable de-identified physician-report candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.forge.private_report_deidentification import (  # noqa: E402
    build_private_report_deidentification_candidates,
)


DEFAULT_REPORT_ROOT = Path("/mnt/hd1/dyf/dataset/EEG_Reports/Reports")
DEFAULT_INVENTORY = (
    ROOT
    / "outputs/evisoz_stage0_private_physician_report_inventory_v1_20260831/inventory.json"
)
DEFAULT_SOURCE = ROOT / "outputs/soz_pre/private_edf_soz_manifest.csv"
DEFAULT_OUTPUT = (
    ROOT / "outputs/evisoz_stage0_private_report_deid_candidates_v1_20260831"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_root = args.report_root.resolve(strict=True)
    inventory = json.loads(
        args.inventory.resolve(strict=True).read_text(encoding="utf-8")
    )
    result = build_private_report_deidentification_candidates(
        report_paths=[path for path in report_root.iterdir() if path.is_file()],
        report_inventory=inventory,
        source_manifest_path=args.source_manifest.resolve(strict=True),
        output=args.output,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "private_physician_report_deidentification_candidates_materialized",
                "candidate_count": result["counts"]["candidate_count"],
                "automated_phi_scan_pass_count": result["counts"][
                    "automated_phi_scan_pass_count"
                ],
                "split_role_candidate_counts": result["counts"][
                    "split_role_candidate_counts"
                ],
                "extraction_route_counts": result["counts"][
                    "extraction_route_counts"
                ],
                "manual_review_pass_count": result["counts"][
                    "manual_review_pass_count"
                ],
                "development_qwen_training_release_count": result["counts"][
                    "development_qwen_training_release_count"
                ],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
