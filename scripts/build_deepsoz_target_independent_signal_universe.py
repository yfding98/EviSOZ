#!/usr/bin/env python3
"""Build the complete target-independent DeepSOZ/TUSZ signal universe."""

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

from src.soz.data.deepsoz_target_independent_signal_universe import (  # noqa: E402
    build_target_independent_signal_universe,
    load_target_independent_signal_universe,
)
from src.soz.data.edf import CausalEDFConfig  # noqa: E402


DEFAULT_IDENTITY_AUDIT = (
    ROOT
    / "outputs/deepsoz_tusz_identity_recovery_v2_20260812"
    / "identity_recovery_audit.csv"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_arg(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected an exact lowercase SHA256")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--identity-audit-csv", type=Path, default=DEFAULT_IDENTITY_AUDIT
    )
    parser.add_argument(
        "--expected-identity-audit-sha256", type=_sha256_arg, required=True
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if os.path.lexists(args.output_directory):
        raise FileExistsError("Output exists; overwrite is forbidden")
    built = build_target_independent_signal_universe(
        args.identity_audit_csv,
        args.tusz_root,
        args.output_directory,
        expected_identity_audit_sha256=args.expected_identity_audit_sha256,
        config=CausalEDFConfig(),
    )
    verified = load_target_independent_signal_universe(
        args.output_directory,
        expected_artifact_sha256=built.artifact_sha256,
    )
    receipt = verified.receipt
    print(
        json.dumps(
            {
                "schema_version": receipt["schema_version"],
                "policy": receipt["policy"],
                "artifact_sha256": verified.artifact_sha256,
                "receipt_sha256": verified.receipt_sha256,
                "lineage_axes": receipt["lineage_axes"],
                "benchmark_identity_overlay_conditioned": receipt[
                    "benchmark_identity_overlay_conditioned"
                ],
                "roster_scope": receipt["roster_scope"],
                "identity_record_count": receipt["identity_record_count"],
                "identity_patient_count": receipt["identity_patient_count"],
                "candidate_event_count": receipt["candidate_event_count"],
                "eligible_event_count": receipt["eligible_event_count"],
                "excluded_event_count": receipt["excluded_event_count"],
                "eligible_patient_count": receipt["eligible_patient_count"],
                "candidate_official_split_event_counts": receipt[
                    "candidate_official_split_event_counts"
                ],
                "eligible_official_split_event_counts": receipt[
                    "eligible_official_split_event_counts"
                ],
                "exclusion_code_counts": receipt["exclusion_code_counts"],
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
