#!/usr/bin/env python3
"""Preflight or materialize locked target-free source-eval V+A/Q evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.locked_source_eval_vaq import (  # noqa: E402
    materialize_locked_source_eval_vaq,
    preflight_locked_source_eval_vaq_inputs,
)


def _sha256(value: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--roster-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-roster-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-signal-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--evolution-scaler-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-evolution-scaler-artifact-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "roster_bundle": args.roster_bundle,
        "expected_roster_artifact_sha256": args.expected_roster_artifact_sha256,
        "signal_preflight_bundle": args.signal_preflight_bundle,
        "expected_signal_artifact_sha256": args.expected_signal_artifact_sha256,
        "expected_signal_receipt_sha256": args.expected_signal_receipt_sha256,
        "evolution_scaler_bundle": args.evolution_scaler_bundle,
        "expected_evolution_scaler_artifact_sha256": (
            args.expected_evolution_scaler_artifact_sha256
        ),
        "tusz_root": args.tusz_root,
    }
    if args.preflight_only:
        if args.output_directory is not None:
            raise ValueError("--preflight-only does not accept --output-directory")
        result = preflight_locked_source_eval_vaq_inputs(**common)
    else:
        if args.output_directory is None:
            raise ValueError("Materialization requires --output-directory")
        result = materialize_locked_source_eval_vaq(
            **common,
            output_directory=args.output_directory,
            progress_every=args.progress_every,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
