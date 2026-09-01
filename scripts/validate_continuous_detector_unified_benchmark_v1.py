#!/usr/bin/env python3
"""Validate the unified long-EEG detector plan without loading any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.unified_continuous_detector_benchmark_v1 import (  # noqa: E402
    DEFAULT_UNIFIED_DETECTOR_PLAN_PATH,
    DEFAULT_UNIFIED_DETECTOR_REGISTRY_PATH,
    build_unified_detector_benchmark_readiness_v1,
    load_unified_continuous_detector_benchmark_plan_v1,
    load_unified_detector_provider_registry_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate content bindings, provider maturity and dual-OP benchmark "
            "semantics without downloading or executing detector weights."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_UNIFIED_DETECTOR_PLAN_PATH,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_UNIFIED_DETECTOR_REGISTRY_PATH,
    )
    parser.add_argument(
        "--no-verify-file-bindings",
        action="store_true",
        help="Validate schema/semantics only; byte binding verification is default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON receipt path; existing paths are never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verify = not args.no_verify_file_bindings
    plan = load_unified_continuous_detector_benchmark_plan_v1(
        args.plan, verify_file_bindings=verify
    )
    registry = load_unified_detector_provider_registry_v1(
        args.registry, verify_file_bindings=verify
    )
    readiness = build_unified_detector_benchmark_readiness_v1(
        plan=plan,
        registry=registry,
        verify_file_bindings=verify,
    )
    payload = json.dumps(
        readiness,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("refusing to overwrite readiness output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
