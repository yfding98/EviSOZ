#!/usr/bin/env python3
"""Post-freeze source-dev reference join for continuous-detector calibration.

The command first validates the complete DeepSOZ posterior batch without any
reference-path input.  Only after that sealed Stage-1 validation succeeds does
it derive and open exact source-development ``.csv_bi`` references.  Outputs
are written once to a new directory and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_source_dev_join import (  # noqa: E402
    join_source_dev_references,
    write_source_dev_reference_join_append_only,
)
from src.clinical_eeg_long_recording.deepsoz_posterior_batch_validation import (  # noqa: E402
    DEEPSOZ_MATERIALIZER_CODE_SHA256,
    validate_deepsoz_posterior_batch_without_references,
)


def _read_roster(path: Path, context: str) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} roster must be a regular non-symlink file")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"{context} roster is empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{context} roster contains duplicates")
    if values != sorted(values):
        raise ValueError(f"{context} roster must be canonically sorted")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete frozen source-dev posterior batch, then join "
            "exact global TERM,seiz intervals into calibration rows"
        )
    )
    parser.add_argument("--posterior-batch-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-recording-roster",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-patient-roster",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-dev-reference-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--provider-registry", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Trust-boundary ordering is intentional.  Stage 1 has no reference-root
    # argument and must close the full prediction inventory first.
    validation = validate_deepsoz_posterior_batch_without_references(
        args.posterior_batch_root,
        expected_split="source_dev",
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_recording_ids=_read_roster(
            args.expected_recording_roster,
            "source-dev recording",
        ),
        expected_patient_ids=_read_roster(
            args.expected_patient_roster,
            "source-dev patient",
        ),
        require_complete_inventory=True,
        expected_materializer_code_sha256=(
            DEEPSOZ_MATERIALIZER_CODE_SHA256
        ),
        provider_registry_path=args.provider_registry,
    )
    joined = join_source_dev_references(
        validation,
        args.source_dev_reference_root,
    )
    write_receipt = write_source_dev_reference_join_append_only(
        joined,
        args.output_directory,
    )
    print(
        json.dumps(
            {
                "join_receipt": joined.join_receipt(),
                "write_receipt": write_receipt,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
