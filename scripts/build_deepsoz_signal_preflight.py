#!/usr/bin/env python3
"""Build the formal, replay-verified DeepSOZ signal-preflight bundle.

This entry point intentionally has no discovery, hash-computation, overwrite,
or preprocessing-override mode.  Every tabular or target-v2 artifact accepted
by the CLI is selected by path and an independently supplied SHA256; the EDF
and annotation bytes selected by those closed tables are hashed into the
verified receipt during replay.  The published bundle itself is label-free;
stdout is restricted further to aggregate counts and provenance hashes and
never includes patient identifiers, event identifiers, or SOZ targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_signal_preflight import (  # noqa: E402
    VerifiedDeepSOZSignalPreflightBundle,
    build_deepsoz_signal_preflight_bundle,
    load_deepsoz_signal_preflight_bundle,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.data.edf import CausalEDFConfig  # noqa: E402


SUMMARY_SCHEMA = "soz_deepsoz_signal_preflight_cli_summary_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    """Require a caller-asserted lowercase SHA256; never derive one here."""

    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "expected SHA256 must contain exactly 64 lowercase hexadecimal characters"
        )
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically build the closed DeepSOZ/TUSZ causal signal-preflight "
            "bundle from explicitly hash-pinned inputs. Existing destinations "
            "are always rejected."
        ),
        allow_abbrev=False,
    )
    inputs = parser.add_argument_group("hash-pinned signal inputs")
    inputs.add_argument("--event-inputs-csv", type=Path, required=True)
    inputs.add_argument(
        "--expected-event-inputs-sha256", type=_sha256_arg, required=True
    )
    inputs.add_argument("--record-crosswalk-csv", type=Path, required=True)
    inputs.add_argument(
        "--expected-record-crosswalk-sha256", type=_sha256_arg, required=True
    )
    inputs.add_argument("--split-manifest-csv", type=Path, required=True)
    inputs.add_argument(
        "--expected-split-manifest-sha256", type=_sha256_arg, required=True
    )
    inputs.add_argument("--deepsoz-source-csv", type=Path, required=True)
    inputs.add_argument(
        "--expected-deepsoz-source-sha256", type=_sha256_arg, required=True
    )
    inputs.add_argument("--conservative-mapping-csv", type=Path, required=True)
    inputs.add_argument(
        "--expected-conservative-mapping-sha256",
        type=_sha256_arg,
        required=True,
    )
    inputs.add_argument(
        "--tusz-root",
        type=Path,
        required=True,
        help="Frozen local TUSZ root containing the crosswalk-selected records",
    )

    target = parser.add_argument_group("verified target-v2 artifact")
    target.add_argument("--target-v2-directory", type=Path, required=True)
    target.add_argument(
        "--expected-target-v2-target-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    target.add_argument(
        "--expected-target-v2-summary-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )
    target.add_argument(
        "--expected-target-v2-readme-artifact-sha256",
        type=_sha256_arg,
        required=True,
    )

    output = parser.add_argument_group("atomic output")
    output.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="New bundle directory; its parent must already exist",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def label_free_summary(
    bundle: VerifiedDeepSOZSignalPreflightBundle,
) -> Mapping[str, object]:
    """Return an explicit aggregate-only projection of the verified receipt."""

    if not isinstance(bundle, VerifiedDeepSOZSignalPreflightBundle):
        raise TypeError("bundle must be a verified DeepSOZ signal-preflight bundle")
    receipt = bundle.receipt
    split_counts = {
        str(split): len(patient_ids)
        for split, patient_ids in receipt["eligible_split_patient_ids"]
    }
    return {
        "summary_schema": SUMMARY_SCHEMA,
        "signal_preflight_schema": receipt["schema_version"],
        "policy": receipt["policy"],
        "artifact_sha256": bundle.artifact_sha256,
        "receipt_sha256": bundle.receipt_sha256,
        "event_inputs_sha256": receipt["event_inputs_sha256"],
        "record_crosswalk_sha256": receipt["record_crosswalk_sha256"],
        "split_manifest_sha256": receipt["split_manifest_sha256"],
        "deepsoz_source_sha256": receipt["deepsoz_source_sha256"],
        "conservative_mapping_sha256": receipt[
            "conservative_mapping_sha256"
        ],
        "verified_target_v2_receipt_sha256": receipt[
            "verified_target_v2_receipt_sha256"
        ],
        "verified_target_v2_artifact_sha256": receipt[
            "verified_target_v2_artifact_sha256"
        ],
        "verified_target_v2_policy_sha256": receipt[
            "verified_target_v2_policy_sha256"
        ],
        "preprocess_schema": receipt["preprocess_schema"],
        "preprocess_config_sha256": receipt["preprocess_config_sha256"],
        "candidate_event_roster_sha256": receipt[
            "candidate_event_roster_sha256"
        ],
        "eligible_event_roster_sha256": receipt[
            "eligible_event_roster_sha256"
        ],
        "excluded_event_roster_sha256": receipt[
            "excluded_event_roster_sha256"
        ],
        "eligible_patient_roster_sha256": receipt[
            "eligible_patient_roster_sha256"
        ],
        "candidate_event_count": receipt["candidate_event_count"],
        "eligible_event_count": receipt["eligible_event_count"],
        "excluded_event_count": receipt["excluded_event_count"],
        "eligible_patient_count": receipt["eligible_patient_count"],
        "eligible_split_patient_counts": split_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Reject an existing destination before target parsing or expensive EDF IO;
    # the builder repeats this check immediately before atomic publication.
    if os.path.lexists(args.output_directory):
        raise FileExistsError(
            "Signal-preflight bundle destination already exists; overwrite is forbidden"
        )

    verified_target = load_verified_deepsoz_target_v2_artifact(
        args.target_v2_directory,
        args.deepsoz_source_csv,
        args.split_manifest_csv,
        expected_target_artifact_sha256=(
            args.expected_target_v2_target_artifact_sha256
        ),
        expected_summary_artifact_sha256=(
            args.expected_target_v2_summary_artifact_sha256
        ),
        expected_readme_artifact_sha256=(
            args.expected_target_v2_readme_artifact_sha256
        ),
        expected_source_input_sha256=args.expected_deepsoz_source_sha256,
        expected_split_input_sha256=args.expected_split_manifest_sha256,
    )
    built = build_deepsoz_signal_preflight_bundle(
        args.event_inputs_csv,
        args.record_crosswalk_csv,
        args.split_manifest_csv,
        args.deepsoz_source_csv,
        args.conservative_mapping_csv,
        verified_target,
        args.tusz_root,
        args.output_directory,
        expected_event_inputs_sha256=args.expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=(
            args.expected_record_crosswalk_sha256
        ),
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=args.expected_deepsoz_source_sha256,
        expected_conservative_mapping_sha256=(
            args.expected_conservative_mapping_sha256
        ),
        config=CausalEDFConfig(),
    )
    # Do not authorize a formal gate from the in-memory build result alone.
    # Pin the bytes just published and force the strict loader to parse the
    # closed artifact and replay every source table, annotation, EDF window,
    # preprocessing receipt, and processed-signal digest a second time.
    bundle = load_deepsoz_signal_preflight_bundle(
        args.output_directory,
        args.event_inputs_csv,
        args.record_crosswalk_csv,
        args.split_manifest_csv,
        args.deepsoz_source_csv,
        args.conservative_mapping_csv,
        verified_target,
        args.tusz_root,
        expected_artifact_sha256=built.artifact_sha256,
        expected_event_inputs_sha256=args.expected_event_inputs_sha256,
        expected_record_crosswalk_sha256=(
            args.expected_record_crosswalk_sha256
        ),
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        expected_deepsoz_source_sha256=args.expected_deepsoz_source_sha256,
        expected_conservative_mapping_sha256=(
            args.expected_conservative_mapping_sha256
        ),
        config=CausalEDFConfig(),
    )
    print(
        json.dumps(
            label_free_summary(bundle),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
