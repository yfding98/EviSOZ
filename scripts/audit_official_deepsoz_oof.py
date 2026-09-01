#!/usr/bin/env python3
"""Independent structural and endpoint audit of the local DeepSOZ transfer."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np


CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
PZ_INDEX = CHANNELS.index("PZ")
NEIGHBORS = {
    0: (1, 2, 3, 4), 1: (0, 4, 5, 6), 2: (0, 3, 4, 7, 8),
    3: (0, 2, 4, 8, 9), 4: (0, 1, 3, 5, 9),
    5: (1, 4, 6, 9, 10), 6: (1, 4, 5, 10, 11),
    7: (2, 8, 12, 13, 17), 8: (2, 3, 4, 7, 9, 12, 13, 14),
    9: (3, 4, 5, 8, 10, 13, 14, 15),
    10: (4, 5, 6, 9, 11, 14, 15, 16),
    11: (6, 10, 15, 16, 18), 12: (7, 8, 13, 17),
    13: (7, 8, 9, 12, 14, 17),
    14: (8, 9, 10, 13, 15, 17, 18),
    15: (9, 10, 11, 14, 16, 18), 16: (10, 11, 15, 18),
    17: (7, 12, 13, 14, 18), 18: (11, 14, 15, 16, 17),
}
EXPECTED_INCOMPLETE_REPEAT_PATIENTS = {
    "11604", "12742", "12858", "12870", "7032", "7584", "8608", "9578",
}
CLAIM_BOUNDARY = (
    "published_weight_signal_version_transfer; official held-out folds; "
    "not an exact original-data reproduction"
)


def _patient(value: object) -> str:
    return str(value).strip().lstrip("0") or "0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_targets(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient = _patient(row["deepsoz_patient_id"])
            target = np.asarray(
                [float(row[f"benchmark_value_{channel}"]) for channel in CHANNELS]
            )
            mask = np.asarray(
                [bool(int(float(row[f"benchmark_mask_{channel}"]))) for channel in CHANNELS]
            )
            result[patient] = target, mask
    return result


def _fold_membership(directory: Path) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = {}
    for fold in range(15):
        path = directory / f"deepsoz_official_pts_test_fold{fold}.npy"
        _require(path.is_file(), f"missing official test fold {fold}")
        values = np.load(path).astype(int).tolist()
        _require(len(values) == 24, f"official test fold {fold} does not contain 24 patients")
        _require(len(set(values)) == 24, f"official test fold {fold} contains duplicates")
        for value in values:
            result.setdefault(str(int(value)), []).append(fold)
    return {patient: tuple(folds) for patient, folds in result.items()}


def _signal_boundary_counts(path: Path, roster: set[str]) -> dict[str, int]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    receipt = artifact["receipt"]
    seen: set[str] = set()
    record_counts: Counter[tuple[str, str]] = Counter()
    for row in [*receipt.get("events", []), *receipt.get("exclusions", [])]:
        patient = _patient(row.get("patient_id", ""))
        event_id = str(row.get("event_id", "")).strip()
        relative = str(row.get("relative_edf_path", "")).strip()
        if patient not in roster or not event_id or not relative or event_id in seen:
            continue
        if row.get("global_t0_sec") is None or row.get("global_stop_sec") is None:
            continue
        seen.add(event_id)
        record_counts[(patient, relative)] += 1
    return {
        "raw_annotation_boundary_count": len(seen),
        "record_count": len(record_counts),
        "records_over_official_cap": sum(value > 10 for value in record_counts.values()),
        "boundaries_discarded_by_official_cap": sum(
            max(0, value - 10) for value in record_counts.values()
        ),
        "expected_crop_count_after_official_cap": sum(
            min(10, value) for value in record_counts.values()
        ),
    }


def _recompute_flags(
    row: dict[str, object], target: np.ndarray, mask: np.ndarray
) -> tuple[bool, bool, bool, int, list[str]]:
    score = np.asarray(row["score"], dtype=np.float64)
    _require(score.shape == (19,), "prediction score must have length 19")
    _require(bool(np.isfinite(score).all()), "prediction score must be finite")
    eligible = score.copy()
    eligible[~mask] = -np.inf
    top1 = int(np.argmax(eligible))
    positives = np.flatnonzero((target > 0.5) & mask)
    _require(positives.size > 0, "target must have an evaluable positive")
    exact = bool(top1 in set(positives.tolist()))
    neighbor = any(top1 in NEIGHBORS[int(index)] for index in positives)
    return (
        exact,
        bool(exact or (positives.size <= 2 and neighbor)),
        bool(exact or (positives.size <= 4 and neighbor)),
        top1,
        [CHANNELS[int(index)] for index in positives],
    )


def _audit_row(
    row: dict[str, object], targets: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    patient = _patient(row["patient_id"])
    _require(patient in targets, f"prediction has unknown target patient {patient}")
    flags = _recompute_flags(row, *targets[patient])
    exact, neighborhood2, neighborhood4, top1, positives = flags
    _require(int(row["top1_index"]) == top1, f"top1 index drift for {patient}")
    _require(row["top1_channel"] == CHANNELS[top1], f"top1 channel drift for {patient}")
    _require(top1 != PZ_INDEX, f"PZ selected for {patient}")
    _require(list(row["positive_channels"]) == positives, f"positive set drift for {patient}")
    _require(int(row["positive_count"]) == len(positives), f"positive count drift for {patient}")
    _require(bool(row["exact"]) == exact, f"strict endpoint drift for {patient}")
    _require(bool(row["neighborhood2"]) == neighborhood2, f"neighborhood2 drift for {patient}")
    _require(bool(row["neighborhood4"]) == neighborhood4, f"neighborhood4 drift for {patient}")


def _metric(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n": len(rows),
        "exact_n": sum(bool(row["exact"]) for row in rows),
        "exact": float(np.mean([bool(row["exact"]) for row in rows])),
        "neighborhood2_n": sum(bool(row["neighborhood2"]) for row in rows),
        "neighborhood2": float(np.mean([bool(row["neighborhood2"]) for row in rows])),
        "neighborhood4_n": sum(bool(row["neighborhood4"]) for row in rows),
        "neighborhood4": float(np.mean([bool(row["neighborhood4"]) for row in rows])),
    }


def _compare_metric(observed: dict[str, object], rows: list[dict[str, object]], name: str) -> None:
    expected = _metric(rows)
    _require(set(observed) == set(expected), f"{name} metric fields drifted")
    for key, value in expected.items():
        if isinstance(value, float):
            _require(math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1e-15), f"{name}.{key} drifted")
        else:
            _require(int(observed[key]) == value, f"{name}.{key} drifted")


def audit(args: argparse.Namespace) -> dict[str, object]:
    artifact = json.loads(Path(args.input).read_text(encoding="utf-8"))
    _require(artifact["schema_version"] == "official_deepsoz_local_oof_evaluation_v1", "schema drifted")
    _require(artifact["claim_boundary"] == CLAIM_BOUNDARY, "claim boundary drifted")
    _require(artifact["private_data_used"] is False, "private data was used")
    _require(artifact["unlabeled_36_used_as_soz_target"] is False, "36-person target firewall failed")
    _require(artifact["endpoint"]["primary"] == "C18 exact positive-set membership Top-1", "primary endpoint drifted")
    _require(artifact["endpoint"]["masked_channel"] == "PZ", "PZ policy drifted")

    local = json.loads(Path(args.local_manifest).read_text(encoding="utf-8"))
    full_roster = {_patient(value) for value in local["patient_ids"]}
    targets = _load_targets(Path(args.target_csv))
    folds = _fold_membership(Path(args.fold_directory))
    receipts = list(artifact["patient_receipts"])
    receipt_ids = [_patient(row["patient_id"]) for row in receipts]
    _require(len(receipt_ids) == len(set(receipt_ids)), "duplicate patient receipt")
    roster = set(receipt_ids)
    if args.allow_subset:
        _require(bool(roster) and roster <= full_roster, "smoke roster is not a nonempty local subset")
    else:
        _require(roster == full_roster, "full patient roster mismatch")
        _require(len(roster) == 102, "full evaluation must contain 102 patients")
        _require(artifact["status"] == "full", "full evaluation status is not full")
    _require(int(artifact["patient_count"]) == len(roster), "patient count drifted")

    receipt_by_patient = {_patient(row["patient_id"]): row for row in receipts}
    for patient, receipt in receipt_by_patient.items():
        expected_folds = list(folds[patient])
        _require(receipt["held_out_folds"] == expected_folds, f"held-out fold drift for {patient}")
        _require(int(receipt["held_out_repeat_count"]) == len(expected_folds), f"repeat count drift for {patient}")
        _require(int(receipt["record_count"]) >= 1, f"no record for {patient}")
        _require(int(receipt["event_count"]) >= 1, f"no event for {patient}")
        _require(len({fold // 5 for fold in expected_folds}) == len(expected_folds), f"duplicate repeat for {patient}")

    signal_counts = _signal_boundary_counts(Path(args.signal_artifact), roster)
    observed_crop_count = sum(int(row["event_count"]) for row in receipts)
    observed_record_count = sum(int(row["record_count"]) for row in receipts)
    _require(
        signal_counts["expected_crop_count_after_official_cap"] == observed_crop_count,
        "official max-10 crop count drifted",
    )
    _require(
        signal_counts["record_count"] == observed_record_count,
        "official record count drifted",
    )

    fold_rows = list(artifact["fold_predictions"])
    fold_keys = [(_patient(row["patient_id"]), int(row["fold"])) for row in fold_rows]
    _require(len(fold_keys) == len(set(fold_keys)), "duplicate patient-fold prediction")
    expected_keys = {(patient, fold) for patient in roster for fold in folds[patient]}
    _require(set(fold_keys) == expected_keys, "prediction was not exactly test-fold held out")
    _require(int(artifact["per_fold_prediction_count"]) == len(fold_rows), "per-fold count drifted")
    for row in fold_rows:
        patient = _patient(row["patient_id"])
        _require(int(row["repeat"]) == int(row["fold"]) // 5, f"repeat mapping drift for {patient}")
        _require(int(row["event_count"]) == int(receipt_by_patient[patient]["event_count"]), f"event count drift for {patient}")
        _audit_row(row, targets)

    ensemble_rows = list(artifact["held_out_ensemble_predictions"])
    ensemble_by_patient = {_patient(row["patient_id"]): row for row in ensemble_rows}
    _require(len(ensemble_rows) == len(ensemble_by_patient), "duplicate ensemble patient")
    _require(set(ensemble_by_patient) == roster, "ensemble roster mismatch")
    fold_by_patient: dict[str, list[dict[str, object]]] = {patient: [] for patient in roster}
    for row in fold_rows:
        fold_by_patient[_patient(row["patient_id"])].append(row)
    for patient, row in ensemble_by_patient.items():
        _audit_row(row, targets)
        sources = fold_by_patient[patient]
        _require(int(row["held_out_repeat_count"]) == len(sources), f"ensemble repeat count drift for {patient}")
        mean_score = np.mean([np.asarray(source["score"], dtype=np.float64) for source in sources], axis=0)
        mean_score /= max(float(mean_score.max()), 1e-12)
        _require(np.allclose(np.asarray(row["score"], dtype=np.float64), mean_score, rtol=1e-12, atol=1e-12), f"ensemble score drift for {patient}")

    for repeat in range(3):
        rows = [row for row in fold_rows if int(row["repeat"]) == repeat]
        _compare_metric(artifact["repeat_metrics"][str(repeat)], rows, f"repeat{repeat}")
    _compare_metric(artifact["pooled_repeat_metrics"], fold_rows, "pooled_repeat")
    _compare_metric(artifact["held_out_ensemble_metrics"], ensemble_rows, "held_out_ensemble")

    incomplete = sorted(patient for patient in roster if len(folds[patient]) != 3)
    if not args.allow_subset:
        _require(set(incomplete) == EXPECTED_INCOMPLETE_REPEAT_PATIENTS, "incomplete-repeat roster drifted")
    zero_filled = {
        patient: list(receipt_by_patient[patient]["zero_filled_channels"])
        for patient in sorted(roster)
        if receipt_by_patient[patient]["zero_filled_channels"]
    }
    return {
        "schema_version": "official_deepsoz_local_oof_audit_v1",
        "status": "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "patient_count": len(roster),
        "prediction_count": len(fold_rows),
        "incomplete_repeat_patients": incomplete,
        "zero_filled_patient_count": len(zero_filled),
        "zero_filled_channels_by_patient": zero_filled,
        "signal_boundary_contract": signal_counts,
        "pooled_repeat_metrics": artifact["pooled_repeat_metrics"],
        "held_out_ensemble_metrics": artifact["held_out_ensemble_metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--fold-directory", default="/tmp")
    parser.add_argument("--target-csv", default="outputs/deepsoz_target_v2_identity_recovery_20260812/patient_targets_v2.csv")
    parser.add_argument("--local-manifest", default="outputs/labram_identity_recovery_closed_replay_v16_20260812/manifest.json")
    parser.add_argument(
        "--signal-artifact",
        default="outputs/deepsoz_signal_preflight_identity_v3_20260812/deepsoz_signal_preflight_identity_v3.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = audit(parsed)
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
