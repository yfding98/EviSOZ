#!/usr/bin/env python3
"""Calibrate a frozen EventNet native grid on public source-dev references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_native_decoder_grid import (  # noqa: E402
    validate_eventnet_native_decoder_grid_bundle_without_references,
)
from src.clinical_eeg_long_recording.eventnet_source_dev_native_calibration import (  # noqa: E402
    calibrate_eventnet_native_decoder_grid_source_dev,
    write_eventnet_source_dev_native_calibration_append_only,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "First validate a complete frozen EventNet prediction grid, then "
            "join exact public source-dev TERM,seiz intervals for calibration"
        )
    )
    parser.add_argument("--decoder-grid-root", type=Path, required=True)
    parser.add_argument("--source-dev-reference-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    # The Stage-1 API has no reference argument.  This complete validation must
    # return before the calibration function constructs or opens a CSV path.
    frozen = validate_eventnet_native_decoder_grid_bundle_without_references(
        args.decoder_grid_root
    )
    calibration = calibrate_eventnet_native_decoder_grid_source_dev(
        frozen,
        source_dev_reference_root=args.source_dev_reference_root,
    )
    write_receipt = write_eventnet_source_dev_native_calibration_append_only(
        calibration,
        args.output_directory,
    )
    best = calibration["best_effort_diagnostic_candidate"]
    print(
        json.dumps(
            {
                "calibration_id": calibration["calibration_id"],
                "receipt_sha256": calibration["receipt_sha256"],
                "constraint_status": calibration["constraint_status"],
                "research_navigation_admission_status": calibration[
                    "research_navigation_admission_status"
                ],
                "selected_operating_point": calibration["selected_operating_point"],
                "best_effort_diagnostic": {
                    "policy_id": best["policy_id"],
                    "high_recall_constraints_met": best["high_recall_constraints_met"],
                    "event_sensitivity": best["metrics"]["event_sensitivity"],
                    "patient_macro_event_sensitivity": best["metrics"]["patient_macro"][
                        "event_sensitivity_macro"
                    ],
                    "false_alarms_per_24h": best["metrics"][
                        "alarm_false_alarms_per_24h"
                    ],
                },
                "source_eval_use_authorized": calibration["source_eval_use_authorized"],
                "write_receipt": write_receipt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
