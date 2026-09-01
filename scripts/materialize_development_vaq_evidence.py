#!/usr/bin/env python3
"""Materialize target-free source-dev V and A/Q evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.soz.development_vaq import (  # noqa: E402
    materialize_development_vaq_evidence,
    preflight_development_vaq_inputs,
)


def _sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the externally pinned DeepSOZ signal-only source-dev timeline "
            "and publish deterministic V plus target-free A/Q evidence."
        )
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the complete source-dev input boundary without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "signal_preflight_bundle": args.signal_preflight_bundle,
        "expected_signal_artifact_sha256": (
            args.expected_signal_artifact_sha256
        ),
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
        result = preflight_development_vaq_inputs(**common)
    else:
        if args.output_directory is None:
            raise ValueError("Materialization requires --output-directory")
        result = materialize_development_vaq_evidence(
            **common,
            output_directory=args.output_directory,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
