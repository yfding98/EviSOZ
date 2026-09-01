#!/usr/bin/env python3
"""Materialize the evaluation-only source-dev LaBraM token corpus."""

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
    load_bound_deepsoz_signal_preflight_artifact,
    load_ictal_native_eval_manifest,
    materialize_ictal_native_eval_token_corpus,
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
            "Materialize frozen LaBraM tokens for the evaluation-only "
            "source-dev native TUSZ manifest"
        )
    )
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-signal-preflight-artifact-sha256", type=_sha, required=True
    )
    parser.add_argument(
        "--expected-signal-preflight-receipt-sha256", type=_sha, required=True
    )
    parser.add_argument("--evaluation-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-evaluation-manifest-artifact-sha256", type=_sha, required=True
    )
    parser.add_argument(
        "--expected-evaluation-manifest-receipt-sha256", type=_sha, required=True
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--labram-modeling-path", type=Path, required=True)
    parser.add_argument("--labram-checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--expected-labram-modeling-sha256", type=_sha, required=True
    )
    parser.add_argument(
        "--expected-foundation-feature-receipt-sha256", type=_sha, required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    signal = load_bound_deepsoz_signal_preflight_artifact(
        args.signal_preflight_bundle,
        expected_artifact_sha256=args.expected_signal_preflight_artifact_sha256,
        expected_receipt_sha256=args.expected_signal_preflight_receipt_sha256,
    )
    manifest = load_ictal_native_eval_manifest(
        args.evaluation_manifest_bundle,
        signal,
        args.tusz_root,
        expected_artifact_sha256=(
            args.expected_evaluation_manifest_artifact_sha256
        ),
        expected_receipt_sha256=(
            args.expected_evaluation_manifest_receipt_sha256
        ),
        expected_signal_artifact_sha256=(
            args.expected_signal_preflight_artifact_sha256
        ),
        expected_signal_receipt_sha256=(
            args.expected_signal_preflight_receipt_sha256
        ),
    )
    corpus = materialize_ictal_native_eval_token_corpus(
        manifest_artifact=manifest,
        expected_manifest_artifact_sha256=(
            args.expected_evaluation_manifest_artifact_sha256
        ),
        expected_manifest_receipt_sha256=(
            args.expected_evaluation_manifest_receipt_sha256
        ),
        expected_signal_artifact_sha256=(
            args.expected_signal_preflight_artifact_sha256
        ),
        expected_signal_receipt_sha256=(
            args.expected_signal_preflight_receipt_sha256
        ),
        tusz_root=args.tusz_root,
        labram_modeling_path=args.labram_modeling_path,
        labram_checkpoint_path=args.labram_checkpoint_path,
        expected_labram_modeling_sha256=args.expected_labram_modeling_sha256,
        expected_foundation_feature_receipt_sha256=(
            args.expected_foundation_feature_receipt_sha256
        ),
        output_directory=args.output_directory,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "path": str(corpus.path),
                "index_sha256": corpus.index_sha256,
                "purpose": corpus.purpose,
                "event_count": corpus.event_count,
                "patient_count": corpus.patient_count,
                "training_authorized": corpus.training_authorized,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
