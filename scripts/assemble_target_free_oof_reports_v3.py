#!/usr/bin/env python3
"""Assemble frozen target-free MRSC OOF reports with fail-closed receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.target_free_oof_report_assembler_v3 import (  # noqa: E402
    assemble_target_free_oof_reports_v3,
    write_target_free_oof_report_batch_v3,
)


DEFAULT_MRSC = ROOT / "outputs/labram_mrsc_target_free_oof_20260812"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--mrsc-directory", type=Path, default=DEFAULT_MRSC)
    parser.add_argument("--mrsc-manifest-sha256", required=True)
    parser.add_argument("--mrsc-tensor-sha256", required=True)
    parser.add_argument("--phenotype-artifact", type=Path, required=True)
    parser.add_argument("--phenotype-sha256", required=True)
    parser.add_argument(
        "--outer-state-container-sha256",
        required=True,
        help=(
            "Frozen v11.1 five-fold OOF outer-state container digest; this is "
            "not a single full-refit checkpoint and is supplied as a trust anchor."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch = assemble_target_free_oof_reports_v3(
        mrsc_directory=args.mrsc_directory,
        expected_mrsc_manifest_sha256=args.mrsc_manifest_sha256,
        expected_mrsc_tensor_sha256=args.mrsc_tensor_sha256,
        phenotype_artifact=args.phenotype_artifact,
        expected_phenotype_sha256=args.phenotype_sha256,
        outer_state_container_sha256=args.outer_state_container_sha256,
    )
    output_sha = write_target_free_oof_report_batch_v3(batch, args.output)
    print(
        json.dumps(
            {
                "schema_version": batch.schema_version,
                "status": batch.status,
                "assembled": batch.assembled_count,
                "blocked": batch.blocked_count,
                "phenotype_events": batch.phenotype_event_count,
                "mrsc_events": batch.mrsc_event_count,
                "output": str(args.output.absolute()),
                "output_sha256": output_sha,
                "car_patient_scores_elementwise_unchanged": True,
                "car_event_scores_elementwise_unchanged": True,
                "selective_threshold_defined": False,
                "all_rankings_abstain": True,
                "deepsoz_target_values_loaded": False,
                "private_data_loaded": False,
                "training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
