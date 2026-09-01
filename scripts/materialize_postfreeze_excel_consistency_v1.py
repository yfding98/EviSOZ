#!/usr/bin/env python3
"""Compare a frozen EEG-only report with an optional PHI-free Excel review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.excel_consistency import (  # noqa: E402
    materialize_postfreeze_excel_consistency,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--report-bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--excel-review",
        type=Path,
        default=None,
        help=(
            "Optional clinical_eeg_excel_onset_review_v1 JSON. When omitted, "
            "all Excel-side comparisons are emitted as not_available."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = materialize_postfreeze_excel_consistency(
        report_bundle_dir=args.report_bundle_dir,
        excel_review_path=args.excel_review,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "recording_id": artifact["recording_id"],
                "event_count": artifact["event_count"],
                "excel_review_status": artifact["excel_review_receipt"]["status"],
                "report_bundle_modified": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
