#!/usr/bin/env python3
"""Evaluate frozen Siena predictions against patient-level weak phenotypes.

The weak ledger is opened only after target-blind predictions, event reports,
and patient summaries have been closed on disk.  The only evaluation unit is
the patient.  This script never computes a channel-level SOZ metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from safetensors.numpy import load_file as load_safetensors


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SCHEMA = "siena_frozen_external_bundle_v1"
PREDICTION_SCHEMA = "siena_frozen_external_target_blind_predictions_v1"
SCHEMA = "siena_frozen_external_weak_region_evaluation_v1"
DEFAULT_BUNDLE = ROOT / "outputs/siena_frozen_external_bundle_v1_20260815"
DEFAULT_PREDICTIONS = ROOT / "outputs/siena_frozen_external_predictions_v1_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/siena_external_weak_region_evaluation_v1_20260815"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260815
EXPECTED_SOURCE_PATIENTS = 14
EXPECTED_SOURCE_EVENTS = 47
EXPECTED_TIME_READY_EVENTS = 44
EXPECTED_SIGNAL_EVENTS = 42
EXPECTED_SIGNAL_PATIENTS = 13
EXPECTED_FULL_CHANNELS = 19
EXPECTED_FOLDS = 5
PZ_CARRIER_INDEX = 14
PROBABILITY_ATOL = 1e-6

LOBE_MAP = {"T": "temporal", "F": "frontal"}
LATERALITY_MAP = {"L": "left", "R": "right", "Bilateral": "bilateral"}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row is not an object: {path}")
            rows.append(value)
    return rows


def _validate_probability_tensor(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    if value.shape != expected_shape or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"Siena {name} tensor shape/dtype drifted")
    if (
        not np.isfinite(value).all()
        or (value < 0).any()
        or (value > 1).any()
        or not np.allclose(
            value.sum(axis=-1), 1.0, atol=PROBABILITY_ATOL, rtol=0
        )
        or not np.array_equal(value[..., PZ_CARRIER_INDEX], np.zeros(value.shape[:-1]))
    ):
        raise ValueError(f"Siena {name} probability/PZ contract failed")


def _validate_prediction_tensors(
    path: Path,
    *,
    event_patient_ids: Sequence[str],
    patient_ids: Sequence[str],
) -> None:
    tensors = load_safetensors(path)
    required = {
        "event_fold_probability",
        "event_patient_index",
        "event_probability",
        "patient_fold_probability",
        "patient_probability",
    }
    if set(tensors) != required:
        raise ValueError("Siena prediction tensor roster drifted")
    event_count = len(event_patient_ids)
    patient_count = len(patient_ids)
    _validate_probability_tensor(
        tensors["event_fold_probability"],
        expected_shape=(event_count, EXPECTED_FOLDS, EXPECTED_FULL_CHANNELS),
        name="event-fold",
    )
    _validate_probability_tensor(
        tensors["event_probability"],
        expected_shape=(event_count, EXPECTED_FULL_CHANNELS),
        name="event",
    )
    _validate_probability_tensor(
        tensors["patient_fold_probability"],
        expected_shape=(patient_count, EXPECTED_FOLDS, EXPECTED_FULL_CHANNELS),
        name="patient-fold",
    )
    _validate_probability_tensor(
        tensors["patient_probability"],
        expected_shape=(patient_count, EXPECTED_FULL_CHANNELS),
        name="patient",
    )
    index = tensors["event_patient_index"]
    expected_index = np.asarray(
        [patient_ids.index(patient_id) for patient_id in event_patient_ids],
        dtype=np.int64,
    )
    if index.dtype != np.int64 or not np.array_equal(index, expected_index):
        raise ValueError("Siena event-patient tensor identity drifted")
    if not np.allclose(
        tensors["event_probability"],
        tensors["event_fold_probability"].mean(axis=1),
        atol=PROBABILITY_ATOL,
        rtol=0,
    ):
        raise ValueError("Siena event fold aggregation drifted")
    expected_patient_fold = np.stack(
        [
            tensors["event_fold_probability"][index == patient_index].mean(axis=0)
            for patient_index in range(patient_count)
        ]
    )
    if not np.allclose(
        tensors["patient_fold_probability"],
        expected_patient_fold,
        atol=PROBABILITY_ATOL,
        rtol=0,
    ) or not np.allclose(
        tensors["patient_probability"],
        tensors["patient_fold_probability"].mean(axis=1),
        atol=PROBABILITY_ATOL,
        rtol=0,
    ):
        raise ValueError("Siena equal-event patient aggregation drifted")


def _target_blind_outputs_ready(
    predictions: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    required = (
        predictions / "predictions.safetensors",
        predictions / "prediction_manifest.json",
        predictions / "structured_reports.jsonl",
        predictions / "patient_summaries.jsonl",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("Siena weak ledger cannot open before target-blind outputs")
    manifest = _read_json(predictions / "prediction_manifest.json")
    if manifest.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError("Siena target-blind prediction schema mismatch")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping) or (
        access.get("weak_patient_target_ledger_opened") is not False
        or access.get("siena_weak_target_values_loaded") is not False
        or access.get("c18_soz_target_values_loaded") is not False
        or access.get("private_data_loaded") is not False
        or access.get("training_performed") is not False
        or access.get("calibration_performed") is not False
        or access.get("model_or_threshold_selection_performed") is not False
        or access.get("all_automatic_decisions_abstained") is not True
    ):
        raise ValueError("Siena prediction target/private/fit firewall failed")
    event_ids = manifest.get("event_ids")
    event_patient_ids = manifest.get("event_patient_ids")
    patient_ids = manifest.get("patient_ids")
    event_counts = manifest.get("event_count_distribution")
    model = manifest.get("model")
    if (
        not isinstance(event_ids, list)
        or not all(isinstance(value, str) and value for value in event_ids)
        or len(event_ids) != len(set(event_ids))
        or not isinstance(event_patient_ids, list)
        or not all(isinstance(value, str) and value for value in event_patient_ids)
        or len(event_patient_ids) != len(event_ids)
        or not isinstance(patient_ids, list)
        or not all(isinstance(value, str) and value for value in patient_ids)
        or len(patient_ids) != len(set(patient_ids))
        or not set(event_patient_ids) == set(patient_ids)
        or not isinstance(event_counts, Mapping)
        or not isinstance(model, Mapping)
        or manifest.get("source_time_supported_events") != EXPECTED_TIME_READY_EVENTS
        or manifest.get("signal_eligible_event_count") != EXPECTED_SIGNAL_EVENTS
        or manifest.get("signal_eligible_patient_count") != EXPECTED_SIGNAL_PATIENTS
        or len(event_ids) != EXPECTED_SIGNAL_EVENTS
        or len(patient_ids) != EXPECTED_SIGNAL_PATIENTS
        or len(model.get("candidate_space", [])) != EXPECTED_FULL_CHANNELS - 1
        or "PZ" in model.get("candidate_space", [])
    ):
        raise ValueError("Siena target-blind prediction identity contract failed")
    observed_counts = Counter(event_patient_ids)
    if dict(observed_counts) != {str(key): int(value) for key, value in event_counts.items()}:
        raise ValueError("Siena prediction patient event counts drifted")

    # Parse and validate every target-blind artifact before the caller is
    # allowed to open the physically separate weak-target ledger.
    _validate_prediction_tensors(
        predictions / "predictions.safetensors",
        event_patient_ids=event_patient_ids,
        patient_ids=patient_ids,
    )
    reports = _read_jsonl(predictions / "structured_reports.jsonl")
    summaries = _read_jsonl(predictions / "patient_summaries.jsonl")
    if (
        [row.get("event_id") for row in reports] != event_ids
        or [row.get("patient_id") for row in reports] != event_patient_ids
        or any(
            row.get("automatic_decision_status")
            != "abstained_uncalibrated_observability"
            or row.get("external_role")
            != "frozen_descriptive_external_signal_audit_not_c18_soz_validation"
            for row in reports
        )
        or [row.get("patient_id") for row in summaries] != patient_ids
        or any(
            row.get("automatic_decision_status")
            != "abstained_uncalibrated_observability"
            for row in summaries
        )
        or [int(row.get("eligible_event_count", -1)) for row in summaries]
        != [observed_counts[patient_id] for patient_id in patient_ids]
    ):
        raise ValueError("Siena sealed report/summary identities drifted")
    return manifest, summaries


def _validate_bundle_manifest(manifest: Mapping[str, object]) -> None:
    summary = manifest.get("summary")
    policy = manifest.get("frozen_policy")
    access = manifest.get("access_receipt")
    if (
        not isinstance(summary, Mapping)
        or summary.get("patient_count") != EXPECTED_SOURCE_PATIENTS
        or summary.get("event_count") != EXPECTED_SOURCE_EVENTS
        or summary.get("time_support_preeligible") != EXPECTED_TIME_READY_EVENTS
        or not isinstance(policy, Mapping)
        or policy.get("evaluation_unit") != "patient"
        or policy.get("event_aggregation") != "equal_event_mean"
        or policy.get("weak_labels_are_soz") is not False
        or not isinstance(access, Mapping)
        or access.get("private_data_loaded") is not False
        or access.get("training_performed") is not False
        or access.get("calibration_performed") is not False
        or access.get("model_or_threshold_selection_performed") is not False
    ):
        raise ValueError("Siena frozen external bundle contract failed")


def _bootstrap_ci(values: Sequence[float], *, seed_offset: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    samples = array[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def patient_weak_rows(
    targets: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    target_by_patient = {str(row.get("patient_id", "")): row for row in targets}
    summary_by_patient = {str(row.get("patient_id", "")): row for row in summaries}
    if (
        "" in target_by_patient
        or len(target_by_patient) != len(targets)
        or "" in summary_by_patient
        or len(summary_by_patient) != len(summaries)
        or not set(summary_by_patient) <= set(target_by_patient)
    ):
        raise ValueError("Siena target/prediction patient identities are invalid")
    rows: list[dict[str, object]] = []
    for patient_id in sorted(target_by_patient):
        target = target_by_patient[patient_id]
        weak_lobe = str(target.get("weak_localization", ""))
        weak_laterality = str(target.get("weak_lateralization", ""))
        if (
            weak_lobe not in LOBE_MAP
            or weak_laterality not in LATERALITY_MAP
            or target.get("label_granularity") != "patient"
            or target.get("label_role") != "weak_external_phenotype_not_soz"
        ):
            raise ValueError("Siena weak phenotype value drifted")
        summary = summary_by_patient.get(patient_id)
        if summary is None:
            rows.append(
                {
                    "patient_id": patient_id,
                    "weak_lobe": LOBE_MAP[weak_lobe],
                    "weak_laterality": LATERALITY_MAP[weak_laterality],
                    "evaluable": False,
                    "not_evaluable_reason": "no_signal_qc_eligible_event",
                    "eligible_event_count": 0,
                    "predicted_top_channel": None,
                    "predicted_lobe": None,
                    "predicted_laterality": None,
                    "lobe_match": 0,
                    "laterality_match": 0,
                    "joint_match": 0,
                }
            )
            continue
        if summary.get("automatic_decision_status") != (
            "abstained_uncalibrated_observability"
        ):
            raise ValueError("Siena automatic abstention status drifted")
        top5 = summary.get("top5_channels")
        spatial = summary.get("spatial_view")
        if not isinstance(top5, list) or not top5 or not isinstance(spatial, Mapping):
            raise TypeError("Siena patient target-blind summary is incomplete")
        top_regions_raw = spatial.get("top_scalp_regions")
        if not isinstance(top_regions_raw, list):
            raise TypeError("Siena top scalp regions are invalid")
        top_regions = tuple(str(value) for value in top_regions_raw)
        predicted_lobe = top_regions[0] if len(top_regions) == 1 else "ambiguous"
        predicted_laterality = str(spatial.get("laterality", "indeterminate"))
        lobe_match = int(predicted_lobe == LOBE_MAP[weak_lobe])
        laterality_match = int(
            predicted_laterality == LATERALITY_MAP[weak_laterality]
        )
        rows.append(
            {
                "patient_id": patient_id,
                "weak_lobe": LOBE_MAP[weak_lobe],
                "weak_laterality": LATERALITY_MAP[weak_laterality],
                "evaluable": True,
                "not_evaluable_reason": None,
                "eligible_event_count": int(summary["eligible_event_count"]),
                "predicted_top_channel": str(top5[0]),
                "predicted_lobe": predicted_lobe,
                "predicted_laterality": predicted_laterality,
                "lobe_match": lobe_match,
                "laterality_match": laterality_match,
                "joint_match": int(lobe_match and laterality_match),
            }
        )
    return rows


def _metric(
    rows: Sequence[Mapping[str, object]], field: str, *, seed_offset: int
) -> dict[str, object]:
    full = [float(row[field]) for row in rows]
    evaluable = [float(row[field]) for row in rows if bool(row["evaluable"])]
    if not evaluable:
        raise RuntimeError("Siena weak evaluation has no evaluable patient")
    return {
        "full_cohort_numerator": int(sum(full)),
        "full_cohort_denominator": len(full),
        "full_cohort_rate": float(np.mean(full)),
        "full_cohort_patient_bootstrap_ci95": _bootstrap_ci(
            full, seed_offset=seed_offset
        ),
        "evaluable_numerator": int(sum(evaluable)),
        "evaluable_denominator": len(evaluable),
        "evaluable_rate_descriptive_only": float(np.mean(evaluable)),
        "evaluable_patient_bootstrap_ci95": _bootstrap_ci(
            evaluable, seed_offset=seed_offset + 100
        ),
    }


def evaluate(bundle: Path, predictions: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    bundle_manifest = _read_json(bundle / "manifest.json")
    if bundle_manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("Siena frozen external bundle schema mismatch")
    _validate_bundle_manifest(bundle_manifest)
    prediction_manifest, summaries = _target_blind_outputs_ready(predictions)

    # This is intentionally the first weak-target-ledger read in the frozen
    # prediction/evaluation sequence.  All four prediction artifacts above
    # have already been parsed and cross-checked, not merely found on disk.
    targets = _read_csv(bundle / "weak_patient_target_ledger.csv")
    rows = patient_weak_rows(targets, summaries)
    if (
        len(rows) != EXPECTED_SOURCE_PATIENTS
        or sum(bool(row["evaluable"]) for row in rows)
        != EXPECTED_SIGNAL_PATIENTS
    ):
        raise RuntimeError("official Siena patient coverage drifted")

    metrics = {
        "weak_lobe_agreement": _metric(rows, "lobe_match", seed_offset=1),
        "weak_laterality_agreement": _metric(
            rows, "laterality_match", seed_offset=2
        ),
        "joint_weak_lobe_and_laterality_agreement": _metric(
            rows, "joint_match", seed_offset=3
        ),
    }
    predicted_lobes = Counter(
        str(row["predicted_lobe"]) for row in rows if bool(row["evaluable"])
    )
    predicted_lateralities = Counter(
        str(row["predicted_laterality"])
        for row in rows
        if bool(row["evaluable"])
    )
    cross_table = Counter(
        (
            str(row["weak_lobe"]),
            str(row["weak_laterality"]),
            str(row["predicted_lobe"]),
            str(row["predicted_laterality"]),
        )
        for row in rows
        if bool(row["evaluable"])
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "scientific_role": (
            "frozen_descriptive_patient_level_weak_region_laterality_audit"
        ),
        "evaluation_unit": "patient",
        "target_semantics": "weak_patient_lobe_and_laterality_phenotype_not_soz",
        "denominators": {
            "source_patients": EXPECTED_SOURCE_PATIENTS,
            "source_events": EXPECTED_SOURCE_EVENTS,
            "timing_and_standard19_ready_events": EXPECTED_TIME_READY_EVENTS,
            "signal_qc_eligible_events": int(
                prediction_manifest["signal_eligible_event_count"]
            ),
            "signal_qc_eligible_patients": sum(
                bool(row["evaluable"]) for row in rows
            ),
            "not_evaluable_patients": sum(
                not bool(row["evaluable"]) for row in rows
            ),
        },
        "metrics": metrics,
        "predicted_lobe_distribution_evaluable": dict(sorted(predicted_lobes.items())),
        "predicted_laterality_distribution_evaluable": dict(
            sorted(predicted_lateralities.items())
        ),
        "weak_to_predicted_cross_table": [
            {
                "weak_lobe": key[0],
                "weak_laterality": key[1],
                "predicted_lobe": key[2],
                "predicted_laterality": key[3],
                "patient_count": count,
            }
            for key, count in sorted(cross_table.items())
        ],
        "prohibited_interpretations": [
            "c18_soz_accuracy",
            "strict_or_neighborhood_channel_hit",
            "cortical_soz_or_ez_validation",
            "private_80_or_85_percent_goal_evidence",
            "model_or_threshold_selection",
        ],
        "access_receipt": {
            "target_blind_prediction_files_verified_before_target_read": True,
            "prediction_tensors_semantically_validated_before_target_read": True,
            "event_reports_parsed_before_target_read": True,
            "patient_summaries_parsed_before_target_read": True,
            "weak_patient_target_ledger_opened_after_predictions": True,
            "c18_soz_target_values_loaded": False,
            "private_data_loaded": False,
            "training_performed": False,
            "calibration_performed": False,
            "model_or_threshold_selection_performed": False,
            "event_level_accuracy_reported": False,
            "all_automatic_decisions_abstained": True,
        },
    }
    output.mkdir(parents=True)
    with (output / "patient_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.bundle, args.predictions, args.output)
    print(json.dumps({"denominators": result["denominators"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
