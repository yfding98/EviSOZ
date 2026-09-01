#!/usr/bin/env python3
"""Materialize target-free patient-OOF V+A/Q for source-train only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.development_vaq_oof import (  # noqa: E402
    materialize_source_train_oof_vaq_evidence,
    preflight_source_train_oof_vaq_inputs,
)


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def _fold_scaler(value: str) -> tuple[int, Path, str]:
    fields = value.split("=", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("use FOLD=PATH=ARTIFACT_SHA256")
    try:
        fold = int(fields[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be 0..4") from exc
    if fold not in range(5) or not fields[1]:
        raise argparse.ArgumentTypeError("fold must be 0..4; final is forbidden")
    return fold, Path(fields[1]), _sha256(fields[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-preflight-bundle", type=Path, required=True)
    parser.add_argument("--expected-signal-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-signal-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--oof-protocol-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-protocol-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-oof-protocol-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--expected-target-v2-artifact-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-receipt-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-target-v2-policy-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--fold-scaler",
        action="append",
        type=_fold_scaler,
        required=True,
        help="repeat exactly once for folds 0..4; final is forbidden",
    )
    parser.add_argument("--tusz-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.fold_scaler) != 5:
        raise ValueError("Exactly five --fold-scaler arguments are required")
    specs = {fold: (path, digest) for fold, path, digest in args.fold_scaler}
    if len(specs) != 5 or set(specs) != set(range(5)):
        raise ValueError("Fold scalers must contain 0..4 exactly once")
    common = {
        "signal_preflight_bundle": args.signal_preflight_bundle,
        "expected_signal_artifact_sha256": args.expected_signal_artifact_sha256,
        "expected_signal_receipt_sha256": args.expected_signal_receipt_sha256,
        "oof_protocol_bundle": args.oof_protocol_bundle,
        "expected_oof_protocol_artifact_sha256": (
            args.expected_oof_protocol_artifact_sha256
        ),
        "expected_oof_protocol_receipt_sha256": (
            args.expected_oof_protocol_receipt_sha256
        ),
        "expected_target_v2_artifact_sha256": args.expected_target_v2_artifact_sha256,
        "expected_target_v2_receipt_sha256": args.expected_target_v2_receipt_sha256,
        "expected_target_v2_policy_sha256": args.expected_target_v2_policy_sha256,
        "fold_scaler_specs": specs,
        "tusz_root": args.tusz_root,
    }
    if args.preflight_only:
        if args.output_directory is not None:
            raise ValueError("--preflight-only does not accept --output-directory")
        result = preflight_source_train_oof_vaq_inputs(**common)
    else:
        if args.output_directory is None:
            raise ValueError("Materialization requires --output-directory")
        result = materialize_source_train_oof_vaq_evidence(
            **common, output_directory=args.output_directory
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
