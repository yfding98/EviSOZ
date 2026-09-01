#!/usr/bin/env python3
"""Materialize one multi-event long-recording trustworthy EEG AI draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.pipeline import (  # noqa: E402
    materialize_long_term_clinical_eeg_report,
)


DEFAULT_POLICY = ROOT / "configs/clinical_eeg_report_v1.json"
DEFAULT_STYLE = ROOT / "configs/clinical_eeg_report_style_zh_v1.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument(
        "--segment-receipt",
        action="append",
        type=Path,
        default=[],
        help=(
            "Repeat once for every selected candidate event; omit when the "
            "validated detection manifest has no selected candidates."
        ),
    )
    parser.add_argument(
        "--waveform-root",
        type=Path,
        required=True,
        help="Root under which each receipt's safe relative figure_file exists.",
    )
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument(
        "--analysis-selection",
        type=Path,
        help=(
            "Signal-only exact partition of detector-selected candidates; enables "
            "complete reports when some or all candidates are signal-ineligible."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help=(
            "Run local Qwen for separately validated event-language audit records; "
            "the deterministic recording report remains authoritative."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_long_term_clinical_eeg_report(
        detection_manifest_path=args.detection_manifest,
        segment_receipt_paths=tuple(args.segment_receipt or ()),
        waveform_root=args.waveform_root,
        output_dir=args.output,
        bundle_id=args.bundle_id,
        analysis_selection_path=args.analysis_selection,
        policy_path=args.policy,
        style_path=args.style,
        base_url=args.base_url,
        use_qwen=args.use_qwen,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": manifest["status"],
                "recording_id": manifest["recording_id"],
                "event_count": manifest["event_count"],
                "three_timebase_closure_verified": manifest["scope_receipt"][
                    "three_timebase_closure_verified"
                ],
                "physician_signed": manifest["scope_receipt"]["physician_signed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
