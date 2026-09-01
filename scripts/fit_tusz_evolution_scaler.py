#!/usr/bin/env python3
"""Fit one formal direct-evolution scaler for OOF fold 0--4 or final."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_tusz_ictal_oof_fold_manifests import (  # noqa: E402
    load_bound_deepsoz_registry,
)
from src.soz.concept_oof import load_ictal_concept_oof_protocol  # noqa: E402
from src.soz.data.public_ledger_builder import (  # noqa: E402
    load_tusz_deepsoz_public_ledger_build,
)
from src.soz.evolution_fit import (  # noqa: E402
    fit_and_publish_tusz_evolution_scaler,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "Expected a lowercase 64-character SHA256 digest"
        )
    return normalized


def _selection_arg(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized == "final":
        return None
    try:
        fold = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Selection must be one of 0,1,2,3,4,final"
        ) from exc
    if fold not in range(5):
        raise argparse.ArgumentTypeError(
            "Selection must be one of 0,1,2,3,4,final"
        )
    return fold


def _stable_file_sha256(path: Path, *, field: str) -> str:
    source = path.absolute()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{field} must be a regular file: {source}")
    if source.resolve(strict=True) != source:
        raise ValueError(f"{field} cannot traverse a symlink")
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{field} changed while it was read")
    return hashlib.sha256(payload).hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay every frozen TUSZ event and fit one patient-balanced "
            "six-feature direct-evolution scaler"
        )
    )
    parser.add_argument("--selection", type=_selection_arg, required=True)
    parser.add_argument("--master-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-master-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-master-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--training-manifest-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-training-bundle-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-training-source-manifest-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--oof-protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-oof-artifact-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-oof-protocol-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--public-ledger", type=Path, required=True)
    parser.add_argument(
        "--expected-public-ledger-bundle-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-public-ledger-build-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--deepsoz-source-csv", type=Path, required=True)
    parser.add_argument("--deepsoz-split-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-deepsoz-split-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry, registry_hashes = load_bound_deepsoz_registry(
        args.deepsoz_source_csv,
        args.deepsoz_split_csv,
        expected_source_sha256=args.expected_deepsoz_source_sha256,
        expected_split_sha256=args.expected_deepsoz_split_sha256,
    )

    public_artifact_file = args.public_ledger / "public_ledger_build.json"
    public_artifact_sha = _stable_file_sha256(
        public_artifact_file, field="public-ledger artifact"
    )
    if public_artifact_sha != args.expected_public_ledger_bundle_sha256:
        raise ValueError("Public-ledger artifact SHA256 mismatch")
    public_artifact = load_tusz_deepsoz_public_ledger_build(
        args.public_ledger,
        expected_bundle_sha256=public_artifact_sha,
        expected_build_sha256=args.expected_public_ledger_build_sha256,
    )

    protocol_file = args.oof_protocol / "ictal_concept_oof_protocol.json"
    protocol_artifact_sha = _stable_file_sha256(
        protocol_file, field="OOF protocol artifact"
    )
    if protocol_artifact_sha != args.expected_oof_artifact_sha256:
        raise ValueError("OOF protocol artifact SHA256 mismatch")
    protocol_artifact = load_ictal_concept_oof_protocol(
        args.oof_protocol,
        registry,
        public_artifact,
        expected_artifact_sha256=protocol_artifact_sha,
        expected_protocol_sha256=args.expected_oof_protocol_sha256,
    )
    if registry_hashes["split_csv_sha256"] != (
        protocol_artifact.protocol.receipt.split_manifest_sha256
    ):
        raise ValueError(
            "Current split CSV byte SHA does not match OOF protocol receipt"
        )

    saved = fit_and_publish_tusz_evolution_scaler(
        master_manifest_bundle=args.master_manifest_bundle,
        expected_master_bundle_manifest_sha256=(
            args.expected_master_bundle_manifest_sha256
        ),
        expected_master_source_manifest_sha256=(
            args.expected_master_source_manifest_sha256
        ),
        training_manifest_bundle=args.training_manifest_bundle,
        expected_training_bundle_manifest_sha256=(
            args.expected_training_bundle_manifest_sha256
        ),
        expected_training_source_manifest_sha256=(
            args.expected_training_source_manifest_sha256
        ),
        oof_fold=args.selection,
        oof_protocol=protocol_artifact.protocol,
        edf_root=args.edf_root,
        output_directory=args.output_directory,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "selection": "final" if args.selection is None else args.selection,
                "path": str(saved.path),
                "artifact_sha256": saved.artifact_sha256,
                "artifact_receipt_sha256": saved.artifact_receipt_sha256,
                "scaler_receipt_sha256": saved.scaler_receipt_sha256,
                "verification_receipt_sha256": (
                    saved.verification_receipt_sha256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
