#!/usr/bin/env python3
"""Build the closed source-dev native-ictal evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.ictal_native_eval import (  # noqa: E402
    build_ictal_native_eval_manifest,
    load_bound_deepsoz_signal_preflight_artifact,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an evaluation-only native TUSZ manifest from the exact "
            "verified DeepSOZ signal-preflight source-dev roster"
        )
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-preflight-artifact-sha256", type=_sha, required=True
    )
    parser.add_argument(
        "--expected-signal-preflight-receipt-sha256", type=_sha, required=True
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    artifact = build_ictal_native_eval_manifest(
        signal,
        args.tusz_root,
        args.output_directory,
        expected_signal_artifact_sha256=(
            args.expected_signal_preflight_artifact_sha256
        ),
        expected_signal_receipt_sha256=(
            args.expected_signal_preflight_receipt_sha256
        ),
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "artifact_sha256": artifact.artifact_sha256,
                "receipt_sha256": artifact.receipt_sha256,
                "purpose": artifact.manifest.purpose,
                "model_split": artifact.manifest.model_split,
                "event_count": len(artifact.manifest),
                "patient_count": len(artifact.manifest.patient_ids),
                "training_authorized": artifact.manifest.training_authorized,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
