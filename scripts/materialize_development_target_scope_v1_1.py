#!/usr/bin/env python3
"""Export the fixed DeepSOZ train/dev target scope in a separate process.

This command is the only stage that accepts the full DeepSOZ source and split
CSV inputs.  Downstream fit/diagnostic processes must use the scoped loader,
which has no full-source input parameters.
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

from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.development_target_scope_v1_1 import (  # noqa: E402
    load_frozen_signal_eligibility_for_target_export_v1_1,
    materialize_development_target_scopes_v1_1,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--target-v2-artifact-directory", type=Path, required=True)
    parser.add_argument("--full-deepsoz-source-csv", type=Path, required=True)
    parser.add_argument("--full-split-manifest-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-target-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-summary-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-readme-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument("--expected-source-input-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-split-input-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--expected-verified-target-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--expected-target-policy-sha256", type=_sha256, required=True)
    parser.add_argument("--eligibility-amendment-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-amendment-artifact-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--expected-amendment-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_target_policy_sha256 != TARGET_V2_POLICY_SHA256:
        raise ValueError("Exporter target-v2 policy SHA is not the frozen policy")
    verified = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_artifact_directory,
        args.full_deepsoz_source_csv,
        args.full_split_manifest_csv,
        expected_target_artifact_sha256=args.expected_target_artifact_sha256,
        expected_summary_artifact_sha256=args.expected_summary_artifact_sha256,
        expected_readme_artifact_sha256=args.expected_readme_artifact_sha256,
        expected_source_input_sha256=args.expected_source_input_sha256,
        expected_split_input_sha256=args.expected_split_input_sha256,
    )
    if verified.receipt.receipt_sha256 != (
        args.expected_verified_target_receipt_sha256
    ):
        raise ValueError("Exporter verified target-v2 receipt SHA mismatch")
    amendment = load_frozen_signal_eligibility_for_target_export_v1_1(
        args.eligibility_amendment_directory,
        expected_artifact_sha256=args.expected_amendment_artifact_sha256,
        expected_receipt_sha256=args.expected_amendment_receipt_sha256,
    )
    scoped = materialize_development_target_scopes_v1_1(
        verified,
        amendment,
        args.output_root,
        expected_original_target_artifact_sha256=(
            args.expected_target_artifact_sha256
        ),
        expected_original_verified_receipt_sha256=(
            args.expected_verified_target_receipt_sha256
        ),
    )
    print(
        json.dumps(
            {
                "status": "STRICT_SCOPED_EXPORT_PASS",
                "output": str(scoped.path),
                "source_train_directory": str(scoped.source_train.path),
                "source_dev_directory": str(scoped.source_dev.path),
                "source_train_receipt_file_sha256": (
                    scoped.source_train.receipt_file_sha256
                ),
                "source_dev_receipt_file_sha256": (
                    scoped.source_dev.receipt_file_sha256
                ),
                "source_train_patient_count": (
                    scoped.source_train.receipt.patient_count
                ),
                "source_dev_patient_count": scoped.source_dev.receipt.patient_count,
                "source_eval_patient_ids_or_targets_included": False,
                "private_payload_included": False,
                "exporter_read_full_verified_target": True,
                "consumer_reads_full_target_or_split": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
