#!/usr/bin/env python3
"""Build or preflight the locked target-free source-eval signal roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.locked_source_eval_roster import (  # noqa: E402
    build_locked_source_eval_roster,
    preflight_locked_source_eval_roster,
)


def _sha256(value: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-signal-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "signal_preflight_bundle": args.signal_preflight_bundle,
        "expected_signal_artifact_sha256": args.expected_signal_artifact_sha256,
        "expected_signal_receipt_sha256": args.expected_signal_receipt_sha256,
    }
    if args.preflight_only:
        if args.output_directory is not None:
            raise ValueError("--preflight-only does not accept --output-directory")
        result = preflight_locked_source_eval_roster(**common)
    else:
        if args.output_directory is None:
            raise ValueError("Roster publication requires --output-directory")
        artifact = build_locked_source_eval_roster(
            **common, output_directory=args.output_directory
        )
        result = {
            "status": "published_target_free_source_eval_roster",
            "path": str(artifact.path),
            "artifact_sha256": artifact.artifact_sha256,
            "receipt_sha256": artifact.receipt_sha256,
            "event_count": len(artifact.events),
            "patient_count": len(artifact.patient_ids),
            "contains_soz_labels": False,
            "contains_tusz_channel_targets_or_masks": False,
            "target_values_loaded": False,
            "target_paths_accepted": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
