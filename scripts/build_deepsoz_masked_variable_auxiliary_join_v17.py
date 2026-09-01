#!/usr/bin/env python3
"""Build the v17 masked-variable auxiliary target/admission bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_masked_variable_auxiliary_join import (  # noqa: E402
    build_masked_variable_auxiliary_join,
    load_masked_variable_auxiliary_admission,
    load_masked_variable_auxiliary_join,
)


DEFAULT_SIGNAL_UNIVERSE = (
    ROOT / "outputs/deepsoz_target_independent_signal_universe_v1_20260812"
)
DEFAULT_TARGET_V2 = ROOT / "outputs/deepsoz_target_v2_identity_recovery_20260812"
DEFAULT_TARGET_SOURCE = (
    ROOT
    / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/TUH_manifest_final.csv"
)
DEFAULT_TARGET_SPLIT = (
    ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/split_manifest.csv"
)
DEFAULT_PROTOCOL = (
    ROOT
    / "research/02_method/"
    "labram_masked_variable_auxiliary_recovery_protocol_v17_20260812_zh.md"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected an exact lowercase SHA256")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--signal-universe-directory", type=Path, default=DEFAULT_SIGNAL_UNIVERSE
    )
    parser.add_argument(
        "--expected-signal-universe-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--target-v2-directory", type=Path, default=DEFAULT_TARGET_V2)
    parser.add_argument("--target-source-csv", type=Path, default=DEFAULT_TARGET_SOURCE)
    parser.add_argument("--target-split-csv", type=Path, default=DEFAULT_TARGET_SPLIT)
    parser.add_argument(
        "--expected-target-artifact-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-target-summary-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-target-readme-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-target-source-input-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument(
        "--expected-target-split-input-sha256",
        type=_sha256_arg,
        required=True,
    )
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--expected-protocol-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    built = build_masked_variable_auxiliary_join(
        args.signal_universe_directory,
        args.target_v2_directory,
        args.target_source_csv,
        args.target_split_csv,
        args.protocol_path,
        args.output_directory,
        expected_signal_universe_artifact_sha256=(
            args.expected_signal_universe_artifact_sha256
        ),
        expected_target_artifact_sha256=args.expected_target_artifact_sha256,
        expected_target_summary_artifact_sha256=(
            args.expected_target_summary_artifact_sha256
        ),
        expected_target_readme_artifact_sha256=(
            args.expected_target_readme_artifact_sha256
        ),
        expected_target_source_input_sha256=(
            args.expected_target_source_input_sha256
        ),
        expected_target_split_input_sha256=(
            args.expected_target_split_input_sha256
        ),
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    verified = load_masked_variable_auxiliary_join(
        args.output_directory,
        expected_artifact_sha256=built.artifact_sha256,
        expected_admission_artifact_sha256=built.admission_artifact_sha256,
    )
    admission = load_masked_variable_auxiliary_admission(
        args.output_directory,
        expected_artifact_sha256=built.admission_artifact_sha256,
    )
    if admission.receipt_sha256 != verified.admission_receipt_sha256:
        raise RuntimeError("Admission-only and full-bundle replay disagree")
    receipt = verified.receipt
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "policy": receipt["policy"],
                "artifact_sha256": verified.artifact_sha256,
                "receipt_sha256": verified.receipt_sha256,
                "admission_artifact_sha256": (
                    verified.admission_artifact_sha256
                ),
                "admission_receipt_sha256": (
                    verified.admission_receipt_sha256
                ),
                "candidate_patient_count": receipt["candidate_patient_count"],
                "admitted_patient_count": receipt["admitted_patient_count"],
                "excluded_patient_count": receipt["excluded_patient_count"],
                "admitted_event_count": receipt["admitted_event_count"],
                "startup_auxiliary_patient_count_gate_pass": receipt[
                    "startup_auxiliary_patient_count_gate_pass"
                ],
                "admitted_patient_ids": receipt["admitted_patient_ids"],
                "excluded_patient_ids": receipt["excluded_patient_ids"],
                "aux_outer_fold_patient_counts": receipt[
                    "aux_outer_fold_patient_counts"
                ],
                "aux_outer_fold_event_counts": receipt[
                    "aux_outer_fold_event_counts"
                ],
                "exclusion_reason_counts": receipt["exclusion_reason_counts"],
                "lineage_axes": receipt["lineage_axes"],
                "admission_lineage_axes": admission.receipt["lineage_axes"],
                "private_data_accessed": receipt["private_data_accessed"],
                "model_or_training_executed": receipt[
                    "model_or_training_executed"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
