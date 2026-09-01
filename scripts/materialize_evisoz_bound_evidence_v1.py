#!/usr/bin/env python3
"""Materialize patient-isolated Stage-0 bound evidence envelopes.

The result is a shadow/evaluator-only join.  An optional, independently
authorized physician-report release may be referenced as a report-only lane;
the report text is never copied and no training loss is authorized here.
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

from src.evisoz.forge.evidence_binding import (  # noqa: E402
    materialize_bound_evidence_examples,
)

DEFAULT_EXAMPLES = ROOT / "outputs/evisoz_stage0_private_real_examples_v1_20260831"
DEFAULT_FINDINGS = ROOT / "outputs/evisoz_stage0_findings_claim_reports_v1_20260831"
DEFAULT_COHORT = ROOT / "outputs/evisoz_stage0_private_real_dual_montage_v1_20260831"
DEFAULT_SPLIT = ROOT / "outputs/evisoz_stage0_private_split_v1_20260831" / "split_roster.json"
DEFAULT_PHYSICIAN_CANDIDATES = ROOT / "outputs/evisoz_stage0_private_report_deid_candidates_v1_20260831"
DEFAULT_OUTPUT = ROOT / "outputs/evisoz_stage0_bound_evidence_v1_20260831"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--private-examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--findings-claims-reports", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--private-cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--split-roster", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--physician-report-release",
        type=Path,
        help="optional independently authorized physician report release root",
    )
    parser.add_argument(
        "--physician-report-candidates",
        type=Path,
        default=DEFAULT_PHYSICIAN_CANDIDATES,
        help="de-identified candidate root used to replay an optional release",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_bound_evidence_examples(
        private_examples_root=args.private_examples,
        findings_claim_report_root=args.findings_claims_reports,
        private_cohort_root=args.private_cohort,
        split_roster_path=args.split_roster,
        physician_report_release_root=args.physician_report_release,
        physician_report_candidate_root=args.physician_report_candidates,
        output=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "materialization_id": result["materialization_id"],
        "counts": result["counts"],
        "permissions": result["permissions"],
        "receipt_sha256": result["receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
