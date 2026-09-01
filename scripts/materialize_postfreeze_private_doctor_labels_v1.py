#!/usr/bin/env python3
"""Publish structured private doctor labels beside a frozen report cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.postfreeze_doctor_label_bundle import (  # noqa: E402
    materialize_postfreeze_doctor_label_bundle,
)


DEFAULT_DATASET_ROOT = Path("/mnt/hd1/dyf/dataset/EEG")
DEFAULT_WORKBOOKS = (
    DEFAULT_DATASET_ROOT / "EEG-fMRI颞叶癫痫(1).xls",
    DEFAULT_DATASET_ROOT / "头皮扩散.xlsx",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument(
        "--workbook",
        action="append",
        type=Path,
        default=None,
        help=(
            "Repeat to override the two canonical private doctor workbooks. "
            "Workbooks are opened only after release and report hashes pass."
        ),
    )
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--primary-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--remediation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = materialize_postfreeze_doctor_label_bundle(
        inventory_path=args.inventory,
        coverage_path=args.coverage,
        release_audit_path=args.release_audit,
        workbook_paths=tuple(args.workbook or DEFAULT_WORKBOOKS),
        output_path=args.output,
        full_root=args.full_root,
        primary_root=args.primary_root,
        recovery_root=args.recovery_root,
        remediation_root=args.remediation_root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "label_release_id": artifact["label_release_id"],
                "status": artifact["status"],
                "record_count": artifact["record_count"],
                "subject_count": artifact["subject_count"],
                **artifact["association_summary"],
                "report_artifacts_modified": False,
                "raw_private_content_released": False,
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
