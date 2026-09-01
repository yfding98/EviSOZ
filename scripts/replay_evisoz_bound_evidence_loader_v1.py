#!/usr/bin/env python3
"""Replay real Stage-0 bound evidence without opening physician DOCX files.

This command is intentionally a read-only smoke/replay entry point.  It
produces a content-addressed receipt whose runtime policy remains
non-authorizing while the aggregate Stage-0 gate is closed.
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

from src.evisoz.data.bound_evidence_loader import (  # noqa: E402
    build_bound_evidence_loader_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--bound-evidence",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260901_r27",
    )
    parser.add_argument(
        "--private-examples",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831",
    )
    parser.add_argument(
        "--findings-claim-reports",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3",
    )
    parser.add_argument(
        "--private-cohort",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831",
    )
    parser.add_argument(
        "--split-roster",
        type=Path,
        default=ROOT / "outputs/evisoz_stage0_private_split_v1_20260831/split_roster.json",
    )
    parser.add_argument("--role", choices=("development_cv", "locked_test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r28/receipt.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = build_bound_evidence_loader_receipt(
        bound_evidence_root=args.bound_evidence,
        private_examples_root=args.private_examples,
        findings_claim_report_root=args.findings_claim_reports,
        private_cohort_root=args.private_cohort,
        split_roster_path=args.split_roster,
        evisoz_role=args.role,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "loader_id": receipt["loader_id"],
                "selection": receipt["selection"],
                "counts": receipt["counts"],
                "receipt_sha256": receipt["receipt_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
