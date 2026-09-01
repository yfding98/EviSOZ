#!/usr/bin/env python3
"""Materialize EviSOZ event Findings, claim graphs, and shadow reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evisoz.forge.findings_claims_reports import (  # noqa: E402
    build_findings_claim_report_materialization,
)


DEFAULT_EXAMPLES = ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
DEFAULT_CANDIDATES = ROOT / "outputs/evisoz_stage0_deterministic_signal_candidates_v1_20260831"
DEFAULT_KNOWLEDGE = ROOT / "knowledge/eeg"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--private-examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--deterministic-candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--knowledge-root", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument(
        "--teacher-candidates",
        type=Path,
        default=None,
        help="optional validated development-only teacher candidate materialization",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_findings_claim_report_materialization(
        private_examples_root=args.private_examples,
        deterministic_candidates_root=args.deterministic_candidates,
        knowledge_root=args.knowledge_root,
        output=args.output,
        teacher_candidates_root=args.teacher_candidates,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "materialization_id": result["materialization_id"],
                "counts": result["counts"],
                "permissions": result["permissions"],
                "receipt_sha256": result["receipt_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
