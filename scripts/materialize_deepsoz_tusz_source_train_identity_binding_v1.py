#!/usr/bin/env python3
"""Materialize the target-free DeepSOZ/TUSZ source-train identity binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clinical_eeg_long_recording.deepsoz_tusz_identity_binding_v1 import (
    build_deepsoz_tusz_source_train_identity_binding_v1,
    materialize_deepsoz_tusz_source_train_identity_binding_v1,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--record-crosswalk", type=Path, required=True)
    parser.add_argument("--complete-tusz-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=70)
    parser.add_argument("--expected-records", type=int, default=318)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    payload = build_deepsoz_tusz_source_train_identity_binding_v1(
        split_manifest_path=args.split_manifest,
        record_crosswalk_path=args.record_crosswalk,
        complete_tusz_roster_path=args.complete_tusz_roster,
        expected_patient_count=args.expected_patients,
        expected_record_count=args.expected_records,
    )
    destination = materialize_deepsoz_tusz_source_train_identity_binding_v1(
        payload, args.output
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "receipt_sha256": payload["receipt_sha256"],
                "counts": payload["counts"],
                "scope_receipt": payload["scope_receipt"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
