#!/usr/bin/env python3
"""Run the prediction-first EventNet v1.2 dual-OP source-dev diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_dual_op_source_dev_v1 import (  # noqa: E402
    calibrate_eventnet_decoder_grid_dual_op_source_dev_v1,
    freeze_eventnet_decoder_grid_for_dual_op_v1,
    write_eventnet_dual_op_calibration_append_only,
    write_eventnet_dual_op_prediction_freeze_append_only,
)
from src.clinical_eeg_long_recording.eventnet_native_decoder_grid import (  # noqa: E402
    validate_eventnet_native_decoder_grid_bundle_without_references,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze the complete reference-free EventNet decoder "
            "grid, persist its dual-OP adapter receipt, then open exact public "
            "source-dev TERM,seiz references for research-only diagnostics"
        )
    )
    parser.add_argument("--decoder-grid-root", type=Path, required=True)
    parser.add_argument("--source-dev-reference-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    # Stage 1 has no reference argument.  Its receipt is durably written before
    # Stage 2 is allowed to derive or open one .csv_bi path.
    grid = validate_eventnet_native_decoder_grid_bundle_without_references(
        args.decoder_grid_root
    )
    frozen = freeze_eventnet_decoder_grid_for_dual_op_v1(grid)
    freeze_write = write_eventnet_dual_op_prediction_freeze_append_only(
        frozen,
        args.output_directory,
    )

    calibration = calibrate_eventnet_decoder_grid_dual_op_source_dev_v1(
        frozen,
        source_dev_reference_root=args.source_dev_reference_root,
    )
    calibration_write = write_eventnet_dual_op_calibration_append_only(
        calibration,
        args.output_directory,
    )
    receipt = calibration.calibration_receipt()
    diagnostic = receipt["dual_op_diagnostic"]
    print(
        json.dumps(
            {
                "prediction_freeze": freeze_write,
                "calibration_receipt_sha256": receipt["receipt_sha256"],
                "dual_op_diagnostic_receipt_sha256": diagnostic["receipt_sha256"],
                "recording_count": diagnostic["coverage_accounting"]["recording_count"],
                "patient_count": diagnostic["coverage_accounting"]["patient_count"],
                "policy_count": diagnostic["coverage_accounting"]["policy_count"],
                "technical_coverage_qualification_eligible": receipt[
                    "technical_coverage_qualification_eligible"
                ],
                "navigation_query_cost_qualification_eligible": False,
                "qualification_granted": False,
                "provider_promotion_authorized": False,
                "write_receipt": calibration_write,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
