#!/usr/bin/env python3
"""Freeze an EventNet native decoder grid without opening any references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_roster import (  # noqa: E402
    validate_continuous_detector_split_roster,
)
from src.clinical_eeg_long_recording.eventnet_native_decoder_grid import (  # noqa: E402
    materialize_eventnet_native_decoder_grid,
    validate_eventnet_native_decoder_grid,
)
from src.clinical_eeg_long_recording.eventnet_raw_prediction_bundle import (  # noqa: E402
    validate_eventnet_raw_prediction_bundle_without_references,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, context: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{context} must contain one JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete frozen EventNet raw source-dev bundle and "
            "materialize a prediction-only provider-native decoder grid"
        )
    )
    parser.add_argument("--raw-bundle-root", type=Path, required=True)
    parser.add_argument("--split-roster-receipt", type=Path, required=True)
    parser.add_argument("--decoder-grid", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    roster = validate_continuous_detector_split_roster(
        _read_json(args.split_roster_receipt, "split-roster receipt")
    )
    source_dev = roster["split_rosters"].get("source_dev")
    if not isinstance(source_dev, dict):
        raise ValueError("split-roster receipt has no source_dev inventory")
    raw_bundle = validate_eventnet_raw_prediction_bundle_without_references(
        args.raw_bundle_root,
        expected_recording_ids=source_dev["recording_ids"],
    )
    grid = validate_eventnet_native_decoder_grid(
        _read_json(args.decoder_grid, "EventNet decoder grid")
    )
    binding = {
        "roster_id": roster["roster_id"],
        "roster_receipt_sha256": _canonical_sha256(roster),
        "source_manifest_file_sha256": roster["source_manifest_file_sha256"],
        "inventory_scope": roster["inventory_scope"],
        "complete_split_inventory_verified": roster["scope_receipt"][
            "complete_split_inventory_verified"
        ],
        "source_dev_recording_roster_sha256": source_dev["recording_roster_sha256"],
    }
    receipt = materialize_eventnet_native_decoder_grid(
        raw_bundle,
        grid_definition=grid,
        output_directory=args.output_directory,
        source_dev_roster_binding=binding,
    )
    print(
        json.dumps(
            {
                "bundle_id": receipt["bundle_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "raw_bundle_validation_id": receipt["raw_bundle_validation_id"],
                "record_count": receipt["record_count"],
                "policy_count": receipt["policy_count"],
                "prediction_row_count": receipt["prediction_row_count"],
                "raw_proposal_count": receipt["raw_proposal_count"],
                "merged_alarm_count": receipt["merged_alarm_count"],
                "reference_files_opened": receipt["reference_access"][
                    "reference_files_opened"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
