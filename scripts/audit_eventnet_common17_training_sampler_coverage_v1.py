#!/usr/bin/env python3
"""Audit deterministic EN17 training-sampler exposure on a frozen manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.eventnet_common17_streaming_v1 import (  # noqa: E402
    TARGET_TILE_SAMPLES,
    build_epoch_draws,
    load_common17_manifest,
)


_PENDING = "__PENDING__"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _content_address(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["receipt_sha256"] = _PENDING
    payload["receipt_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def audit_sampler_coverage(
    manifest_path: Path, *, epoch_counts: tuple[int, ...], seed: int
) -> dict[str, Any]:
    manifest = load_common17_manifest(manifest_path, require_complete=True)
    records = [
        row for row in manifest["records"] if row["model_split"] == "source_train"
    ]
    all_patients = {str(row["patient_id"]) for row in records}
    rows: list[dict[str, Any]] = []
    for epoch_count in epoch_counts:
        draws = [
            draw
            for epoch_index in range(epoch_count)
            for draw in build_epoch_draws(records, epoch_index=epoch_index, seed=seed)
        ]
        record_indices = {int(draw.record_index) for draw in draws}
        patients = {str(draw.patient_id) for draw in draws}
        pools = Counter(str(draw.pool) for draw in draws)
        rows.append(
            {
                "epoch_count": epoch_count,
                "draw_count": len(draws),
                "positive_draw_count": pools.get("positive", 0),
                "background_draw_count": pools.get("background", 0),
                "unique_record_count_with_gradient_exposure": len(record_indices),
                "unique_record_fraction_with_gradient_exposure": (
                    len(record_indices) / len(records)
                ),
                "unique_patient_count_with_gradient_exposure": len(patients),
                "patients_without_gradient_exposure": sorted(all_patients - patients),
                "records_without_gradient_exposure": len(records) - len(record_indices),
            }
        )
    never_eligible_patients = sorted(
        {
            str(row["patient_id"])
            for row in records
            if all(
                int(candidate["target_sample_count_256hz"]) < TARGET_TILE_SAMPLES
                for candidate in records
                if candidate["patient_id"] == row["patient_id"]
            )
        }
    )
    return _content_address(
        {
            "schema_version": "eventnet_common17_training_sampler_coverage_audit_v1",
            "manifest_path": str(manifest_path.resolve()),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "seed": seed,
            "target_tile_samples": TARGET_TILE_SAMPLES,
            "target_tile_seconds_at_256hz": TARGET_TILE_SAMPLES / 256.0,
            "source_train_manifest_record_count": len(records),
            "source_train_manifest_patient_count": len(all_patients),
            "patients_with_no_record_long_enough_for_one_training_tile": (
                never_eligible_patients
            ),
            "epoch_coverage": rows,
            "interpretation": {
                "manifest_membership_is_not_gradient_exposure": True,
                "evaluation_denominator_changed": False,
                "clinical_or_model_performance_claim": False,
            },
            "receipt_sha256": _PENDING,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epoch-counts", default="3,20")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epoch_counts = tuple(int(value) for value in args.epoch_counts.split(","))
    if not epoch_counts or any(value < 1 for value in epoch_counts):
        raise ValueError("epoch counts must be positive integers")
    result = audit_sampler_coverage(
        args.manifest.resolve(strict=True), epoch_counts=epoch_counts, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
