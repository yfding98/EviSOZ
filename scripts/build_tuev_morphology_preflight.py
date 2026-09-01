#!/usr/bin/env python3
"""Publish and replay the formal first-party TUEV morphology preflight gate.

The canonical metadata must come from the repository producer.  This command
requires its current source/policy/mapping hashes and independently replays
the real EDF/REC/LAB/HTK inputs before both publication and strict reload.
"""

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

from src.soz.data.tuev_morphology import (  # noqa: E402
    load_tuev_morphology_preflight,
    materialize_tuev_morphology_preflight,
)
from src.soz.data.tuev_morphology_signal_preflight import (  # noqa: E402
    require_first_party_tuev_morphology_bindings,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    normalized = str(value).strip()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind externally produced TUEV header/QC metadata to the exact "
            "EDF/REC/parent-group roster and replay the published bundle"
        )
    )
    parser.add_argument("--edf-root", type=Path, required=True)
    parser.add_argument("--external-metadata-json", type=Path, required=True)
    parser.add_argument(
        "--expected-external-metadata-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-producer-source-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-preprocessing-policy-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-standard19-mapping-policy-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    require_first_party_tuev_morphology_bindings(
        producer_source_sha256=args.expected_producer_source_sha256,
        preprocessing_policy_sha256=args.expected_preprocessing_policy_sha256,
        standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
    )
    artifact = materialize_tuev_morphology_preflight(
        args.output_directory,
        edf_root=args.edf_root,
        external_metadata_path=args.external_metadata_json,
        expected_external_metadata_sha256=(
            args.expected_external_metadata_sha256
        ),
        expected_producer_source_sha256=args.expected_producer_source_sha256,
        expected_preprocessing_policy_sha256=(
            args.expected_preprocessing_policy_sha256
        ),
        expected_standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
    )
    verified = load_tuev_morphology_preflight(
        artifact.path,
        edf_root=args.edf_root,
        external_metadata_path=args.external_metadata_json,
        expected_bundle_manifest_sha256=artifact.bundle_manifest_sha256,
        expected_preflight_receipt_sha256=artifact.preflight_receipt_sha256,
        expected_external_metadata_sha256=(
            args.expected_external_metadata_sha256
        ),
        expected_producer_source_sha256=args.expected_producer_source_sha256,
        expected_preprocessing_policy_sha256=(
            args.expected_preprocessing_policy_sha256
        ),
        expected_standard19_mapping_policy_sha256=(
            args.expected_standard19_mapping_policy_sha256
        ),
    )
    print(
        json.dumps(
            {
                "bundle_manifest_sha256": artifact.bundle_manifest_sha256,
                "external_metadata_sha256": artifact.external_metadata_sha256,
                "duplicate_ledger_sha256": artifact.duplicate_ledger_sha256,
                "path": str(artifact.path),
                "preflight_receipt_sha256": (
                    artifact.preflight_receipt_sha256
                ),
                "producer_source_sha256": verified.producer_source_sha256,
                "record_count": len(verified.records),
                "exact_duplicate_class_count": len(
                    verified.duplicate_ledger.duplicate_classes
                ),
                "quarantined_record_count": sum(
                    decision.quarantined
                    for decision in verified.duplicate_ledger.record_decisions
                ),
                "source_roster_sha256": verified.source_roster_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
