#!/usr/bin/env python3
"""Publish the opt-in EEG-only adaptive per-event analysis profile v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.adaptive_event_analysis_profile import (  # noqa: E402
    materialize_adaptive_event_analysis_profile,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_adaptive_event_analysis_profile(
        detection_manifest_path=args.detection_manifest,
        edf_path=args.edf,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "recording_id": manifest["recording_id"],
                "event_count": manifest["event_count"],
                "primary_findings_window": "adaptive_variable_per_event",
                "fixed_window_role": "compatibility_core_only",
                "edf_annotations_used": False,
                "excel_used": False,
                "labels_or_ground_truth_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

