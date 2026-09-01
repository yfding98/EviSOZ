#!/usr/bin/env python3
"""Build the closed core-plus-identity-recovered DeepSOZ signal receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_signal_identity_recovery import (  # noqa: E402
    build_deepsoz_signal_identity_recovery_bundle,
    load_deepsoz_signal_identity_recovery_bundle,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.data.edf import CausalEDFConfig  # noqa: E402


DEFAULT_BASE = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
DEFAULT_IDENTITY = ROOT / "outputs/deepsoz_tusz_identity_recovery_v2_20260812"
DEFAULT_SPLITS = ROOT / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812"
DEFAULT_TARGET = ROOT / "outputs/deepsoz_target_v2_identity_recovery_20260812"
DEFAULT_SOURCE = (
    ROOT
    / "outputs/deepsoz_tusz_adapted_manifest_20260803/source/TUH_manifest_final.csv"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected an exact lowercase SHA256")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--base-bundle", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--expected-base-artifact-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--identity-audit-csv",
        type=Path,
        default=DEFAULT_IDENTITY / "identity_recovery_audit.csv",
    )
    parser.add_argument(
        "--expected-identity-audit-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--identity-mapping-csv",
        type=Path,
        default=DEFAULT_IDENTITY / "mapping_identity_v2.csv",
    )
    parser.add_argument(
        "--expected-identity-mapping-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--event-inputs-csv",
        type=Path,
        default=DEFAULT_SPLITS / "event_inputs.csv",
    )
    parser.add_argument(
        "--expected-event-inputs-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--record-crosswalk-csv",
        type=Path,
        default=DEFAULT_SPLITS / "record_crosswalk.csv",
    )
    parser.add_argument(
        "--expected-record-crosswalk-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--split-manifest-csv",
        type=Path,
        default=DEFAULT_SPLITS / "split_manifest.csv",
    )
    parser.add_argument(
        "--expected-split-manifest-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--deepsoz-source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--target-v2-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--expected-target-artifact-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-target-summary-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument(
        "--expected-target-readme-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if os.path.lexists(args.output_directory):
        raise FileExistsError("Output exists; overwrite is forbidden")
    verified_target = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_directory,
        args.deepsoz_source_csv,
        args.split_manifest_csv,
        expected_target_artifact_sha256=args.expected_target_artifact_sha256,
        expected_summary_artifact_sha256=args.expected_target_summary_sha256,
        expected_readme_artifact_sha256=args.expected_target_readme_sha256,
        expected_source_input_sha256=args.expected_deepsoz_source_sha256,
        expected_split_input_sha256=args.expected_split_manifest_sha256,
    )
    built = build_deepsoz_signal_identity_recovery_bundle(
        args.base_bundle,
        args.identity_audit_csv,
        args.identity_mapping_csv,
        args.event_inputs_csv,
        args.record_crosswalk_csv,
        args.split_manifest_csv,
        args.deepsoz_source_csv,
        verified_target,
        args.tusz_root,
        args.output_directory,
        expected_base_artifact_sha256=args.expected_base_artifact_sha256,
        expected_identity_audit_sha256=args.expected_identity_audit_sha256,
        expected_identity_mapping_sha256=args.expected_identity_mapping_sha256,
        expected_event_inputs_sha256=args.expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=args.expected_record_crosswalk_sha256,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=args.expected_deepsoz_source_sha256,
        config=CausalEDFConfig(),
        expected_recovered_candidate_count=217,
        expected_recovered_eligible_count=161,
        expected_recovered_excluded_count=56,
        expected_combined_patient_count=103,
        expected_combined_event_count=1149,
        expected_fixed18_patient_count=102,
        expected_fixed18_event_count=1145,
    )
    bundle = load_deepsoz_signal_identity_recovery_bundle(
        args.output_directory,
        expected_artifact_sha256=built.artifact_sha256,
    )
    receipt = bundle.receipt
    summary = {
        "schema_version": receipt["schema_version"],
        "policy": receipt["policy"],
        "artifact_sha256": bundle.artifact_sha256,
        "receipt_sha256": bundle.receipt_sha256,
        "identity_recovered_row_count": len(receipt["identity_recovered_row_ids"]),
        "identity_recovered_patient_count": len(
            receipt["identity_recovered_patient_ids"]
        ),
        "variable_label_patient_count": len(receipt["variable_label_patient_ids"]),
        "base_candidate_event_count": receipt["base_candidate_event_count"],
        "base_eligible_event_count": receipt["base_eligible_event_count"],
        "recovered_candidate_event_count": receipt[
            "recovered_candidate_event_count"
        ],
        "recovered_eligible_event_count": receipt["recovered_eligible_event_count"],
        "recovered_excluded_event_count": receipt["recovered_excluded_event_count"],
        "recovered_exclusion_code_counts": receipt[
            "recovered_exclusion_code_counts"
        ],
        "combined_eligible_patient_count": receipt[
            "combined_eligible_patient_count"
        ],
        "combined_eligible_event_count": receipt["combined_eligible_event_count"],
        "combined_split_patient_counts": {
            split: len(patient_ids)
            for split, patient_ids in receipt["combined_eligible_split_patient_ids"]
        },
        "partial_reference_signal_patient_count": len(
            receipt["partial_reference_signal_patient_ids"]
        ),
        "fixed18_primary_patient_count": receipt["fixed18_primary_patient_count"],
        "fixed18_primary_event_count": receipt["fixed18_primary_event_count"],
        "fixed18_split_patient_counts": {
            split: len(patient_ids)
            for split, patient_ids in receipt["fixed18_primary_split_patient_ids"]
        },
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
