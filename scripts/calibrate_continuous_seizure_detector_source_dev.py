#!/usr/bin/env python3
"""Freeze one hysteresis operating point from precomputed source-dev posteriors.

This command never runs a detector and never accepts source-evaluation labels.
Its JSONL input must contain already materialized dense posterior timelines and
source-development reference intervals.  The output is a content-bound,
research-only operating-point receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_calibration import (  # noqa: E402
    select_continuous_detection_operating_point,
)


GRID_SCHEMA_VERSION = "continuous_detector_policy_grid_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid calibration JSONL at line {line_number}"
                ) from error
            if type(value) is not dict:
                raise TypeError(f"calibration JSONL line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("calibration JSONL is empty")
    return rows


def _read_grid(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "candidate_policies",
        "minimum_event_sensitivity",
        "minimum_patient_macro_event_sensitivity",
        "onset_tie_tolerance_seconds",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("policy grid has missing or unknown fields")
    if value["schema_version"] != GRID_SCHEMA_VERSION:
        raise ValueError("policy-grid schema version drifted")
    return value


def _read_roster(path: Path | None, context: str) -> list[str] | None:
    if path is None:
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"{context} roster is empty")
    return values


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a provider-neutral continuous seizure-decoder operating "
            "point on frozen source-development posteriors"
        )
    )
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--policy-grid-json", type=Path, required=True)
    parser.add_argument("--expected-source-dev-recording-roster", type=Path)
    parser.add_argument("--evaluation-patient-roster", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = _read_grid(args.policy_grid_json)
    receipt = select_continuous_detection_operating_point(
        provider_id=args.provider_id,
        rows=_read_jsonl(args.input_jsonl),
        candidate_policies=grid["candidate_policies"],
        minimum_event_sensitivity=grid["minimum_event_sensitivity"],
        minimum_patient_macro_event_sensitivity=grid[
            "minimum_patient_macro_event_sensitivity"
        ],
        onset_tie_tolerance_seconds=grid["onset_tie_tolerance_seconds"],
        expected_source_dev_recording_ids=_read_roster(
            args.expected_source_dev_recording_roster, "source_dev recording"
        ),
        evaluation_patient_ids=_read_roster(
            args.evaluation_patient_roster, "source_eval patient"
        ),
    )
    _atomic_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

