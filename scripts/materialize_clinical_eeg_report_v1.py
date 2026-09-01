#!/usr/bin/env python3
"""Create an unsigned scalp-EEG-only draft from a validated SOZ fact ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_report.pipeline import materialize_clinical_eeg_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="EEG-only clinical_eeg_report_v1 JSON")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/clinical_eeg_report_v1.json",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=ROOT / "configs/clinical_eeg_report_style_zh_v1.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--waveform-manifest",
        type=Path,
        help=(
            "optional report-specific clinical_eeg_waveform_manifest_v1 JSON; "
            "figures are selected only through report evidence IDs"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip Qwen and exercise the deterministic fail-closed path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_clinical_eeg_report(
        input_path=args.input,
        output_dir=args.output,
        policy_path=args.policy,
        style_path=args.style,
        base_url=args.base_url,
        dry_run=args.dry_run,
        waveform_manifest_path=args.waveform_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
