#!/usr/bin/env python3
"""Generate one source-attributed private clinical EEG draft per physical EDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.private_contextual_reports import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_BASE_REPORT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DOCTOR_BUNDLE,
    DEFAULT_INVENTORY,
    DEFAULT_OUTPUT,
    DEFAULT_WORKBOOKS,
    materialize_private_contextual_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--base-report-root", type=Path, default=DEFAULT_BASE_REPORT_ROOT)
    parser.add_argument("--doctor-bundle", type=Path, default=DEFAULT_DOCTOR_BUNDLE)
    parser.add_argument("--edf-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--workbook",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeat to override the canonical, release-receipted workbooks. "
            "Versioned and nested duplicate copies are intentionally not auto-selected."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docx", action="store_true")
    parser.add_argument("--expected-physical-edf", type=int, default=144)
    parser.add_argument("--expected-unique-signals", type=int, default=141)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize_private_contextual_reports(
        dataset_root=args.dataset_root,
        inventory_path=args.inventory,
        base_report_root=args.base_report_root,
        doctor_bundle_path=args.doctor_bundle,
        annotation_csv_path=args.edf_annotations,
        workbook_paths=tuple(args.workbook or DEFAULT_WORKBOOKS),
        output_root=args.output,
        include_docx=args.docx,
        expected_physical_count=args.expected_physical_edf,
        expected_unique_signal_count=args.expected_unique_signals,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": manifest["status"],
                "physical_edf_count": manifest["physical_edf_count"],
                "unique_signal_count": manifest["unique_signal_count"],
                "report_count": manifest["report_count"],
                "format_counts": manifest["format_counts"],
                "duplicate_signal_alias_count": manifest[
                    "duplicate_signal_alias_count"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
