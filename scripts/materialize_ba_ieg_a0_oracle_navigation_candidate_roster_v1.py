#!/usr/bin/env python3
"""Materialize the frozen BA-IEG A0 oracle-navigation candidate roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clinical_eeg_long_recording.ba_ieg_a0_oracle_navigation_candidate_roster_v1 import (
    build_ba_ieg_a0_oracle_navigation_candidate_roster_from_paths_v1,
    materialize_ba_ieg_a0_oracle_navigation_candidate_roster_v1,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IDENTITY = (
    _PROJECT_ROOT
    / "outputs/deepsoz_tusz_source_train_identity_binding_v1_20260823/identity_binding.json"
)
_DEFAULT_EVENTS = (
    _PROJECT_ROOT
    / "outputs/deepsoz_tusz_patient_splits_identity_v2_20260812/event_inputs.csv"
)
_DEFAULT_OUTPUT = (
    _PROJECT_ROOT
    / "outputs/ba_ieg_a0_oracle_navigation_candidate_roster_v1_20260824r1/candidate_roster.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identity-binding", type=Path, default=_DEFAULT_IDENTITY
    )
    parser.add_argument("--event-inputs-csv", type=Path, default=_DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    payload = build_ba_ieg_a0_oracle_navigation_candidate_roster_from_paths_v1(
        identity_binding_path=args.identity_binding,
        event_inputs_csv_path=args.event_inputs_csv,
    )
    destination = materialize_ba_ieg_a0_oracle_navigation_candidate_roster_v1(
        payload, args.output
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "receipt_sha256": payload["receipt_sha256"],
                "oracle_navigation_receipt_sha256": payload[
                    "oracle_navigation_receipt_sha256"
                ],
                "counts": payload["counts"],
                "scope_receipt": payload["scope_receipt"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
