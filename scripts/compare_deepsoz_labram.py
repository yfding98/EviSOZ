#!/usr/bin/env python3
"""Endpoint-aligned descriptive comparison of official DeepSOZ and LaBraM."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
PZ = CHANNELS.index("PZ")
NEIGHBORS = {
    0: (1, 2, 3, 4), 1: (0, 4, 5, 6), 2: (0, 3, 4, 7, 8),
    3: (0, 2, 4, 8, 9), 4: (0, 1, 3, 5, 9), 5: (1, 4, 6, 9, 10),
    6: (1, 4, 5, 10, 11), 7: (2, 8, 12, 13, 17),
    8: (2, 3, 4, 7, 9, 12, 13, 14),
    9: (3, 4, 5, 8, 10, 13, 14, 15),
    10: (4, 5, 6, 9, 11, 14, 15, 16), 11: (6, 10, 15, 16, 18),
    12: (7, 8, 13, 17), 13: (7, 8, 9, 12, 14, 17),
    14: (8, 9, 10, 13, 15, 17, 18), 15: (9, 10, 11, 14, 16, 18),
    16: (10, 11, 15, 18), 17: (7, 12, 13, 14, 18),
    18: (11, 14, 15, 16, 17),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _patient(value: object) -> str:
    return str(value).strip().lstrip("0") or "0"


def _evaluate(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    eligible = scores.copy()
    eligible[~mask] = -np.inf
    top1 = int(np.argmax(eligible))
    positives = np.flatnonzero((target > 0.5) & mask)
    exact = bool(top1 in set(positives.tolist()))
    neighbor = any(top1 in NEIGHBORS[int(index)] for index in positives)
    return {
        "top1_index": top1,
        "top1_channel": CHANNELS[top1],
        "positive_count": int(positives.size),
        "exact": exact,
        "neighborhood2": bool(exact or (positives.size <= 2 and neighbor)),
        "neighborhood4": bool(exact or (positives.size <= 4 and neighbor)),
    }


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    result: dict[str, object] = {"n": n}
    for key in ("exact", "neighborhood2", "neighborhood4"):
        count = sum(bool(row[key]) for row in rows)
        result[f"{key}_n"] = count
        result[key] = count / n if n else None
    return result


def _bootstrap_delta(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    key: str,
) -> dict[str, object]:
    left = np.asarray([bool(row[key]) for row in candidate], dtype=np.float64)
    right = np.asarray([bool(row[key]) for row in comparator], dtype=np.float64)
    delta = left - right
    generator = np.random.default_rng(20260814)
    indices = generator.integers(0, len(delta), size=(20000, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "candidate_only_correct_n": int(np.sum((left == 1) & (right == 0))),
        "comparator_only_correct_n": int(np.sum((left == 0) & (right == 1))),
        "both_correct_n": int(np.sum((left == 1) & (right == 1))),
        "both_incorrect_n": int(np.sum((left == 0) & (right == 0))),
    }


def _paired_error_sets(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    key: str,
) -> dict[str, list[str]]:
    _require(len(candidate) == len(comparator), "paired row length mismatch")
    candidate_only: list[str] = []
    comparator_only: list[str] = []
    both_incorrect: list[str] = []
    for left, right in zip(candidate, comparator, strict=True):
        left_patient = _patient(left["patient_id"])
        right_patient = _patient(right["patient_id"])
        _require(left_patient == right_patient, "paired patient order mismatch")
        left_correct = bool(left[key])
        right_correct = bool(right[key])
        if left_correct and not right_correct:
            candidate_only.append(left_patient)
        elif right_correct and not left_correct:
            comparator_only.append(left_patient)
        elif not left_correct and not right_correct:
            both_incorrect.append(left_patient)
    return {
        "candidate_only_correct_patients": candidate_only,
        "comparator_only_correct_patients": comparator_only,
        "both_incorrect_patients": both_incorrect,
    }


def _stratum(value: int, kind: str) -> str:
    if kind == "positive":
        return str(value) if value <= 4 else "ge_5"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3_to_5"
    return "ge_6"


def _stratified(
    rows_by_model: dict[str, list[dict[str, object]]],
    *,
    field: str,
    kind: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    strata = sorted({_stratum(int(row[field]), kind) for rows in rows_by_model.values() for row in rows})
    for stratum in strata:
        result[stratum] = {
            model: _metrics([row for row in rows if _stratum(int(row[field]), kind) == stratum])
            for model, rows in rows_by_model.items()
        }
    return result


def compare(args: argparse.Namespace) -> dict[str, object]:
    official = json.loads(Path(args.official).read_text(encoding="utf-8"))
    _require(official["status"] == "full", "official transfer evaluation is not full")
    manifest = json.loads(Path(args.labram_manifest).read_text(encoding="utf-8"))
    patient_ids = [_patient(value) for value in manifest["stable_evaluation"]["patient_ids"]]
    event_counts = [int(value) for value in manifest["stable_evaluation"]["event_counts"]]
    _require(len(patient_ids) == 102, "LaBraM roster must contain 102 patients")

    with safe_open(args.labram_predictions, framework="pt", device="cpu") as handle:
        v16 = handle.get_tensor("oof.identity_v16_full").numpy().astype(np.float64)
        v17 = handle.get_tensor("oof.masked_variable_auxiliary_full").numpy().astype(np.float64)
        targets = handle.get_tensor("targets").numpy().astype(np.float64)
        masks = handle.get_tensor("target_mask").numpy().astype(bool)
    _require(v16.shape == v17.shape == targets.shape == masks.shape == (102, 19), "LaBraM tensor shape drifted")
    _require(bool((~masks[:, PZ]).all()), "LaBraM PZ mask drifted")

    official_rows = {
        _patient(row["patient_id"]): row
        for row in official["held_out_ensemble_predictions"]
    }
    _require(set(official_rows) == set(patient_ids), "official/LaBraM roster mismatch")
    receipt_by_patient = {
        _patient(row["patient_id"]): row for row in official["patient_receipts"]
    }
    rows_by_model: dict[str, list[dict[str, object]]] = {
        "labram_identity_v16": [],
        "labram_auxiliary_v17": [],
        "deepsoz_held_out_ensemble": [],
    }
    for index, patient in enumerate(patient_ids):
        official_event_count = int(receipt_by_patient[patient]["event_count"])
        common = {
            "patient_id": patient,
            "labram_event_count": event_counts[index],
            "official_event_count": official_event_count,
        }
        target = targets[index]
        mask = masks[index]
        official_source = official_rows[patient]
        expected_positive_channels = [
            CHANNELS[position]
            for position in np.flatnonzero((target > 0.5) & mask)
        ]
        _require(
            list(official_source["positive_channels"]) == expected_positive_channels,
            f"DeepSOZ/LaBraM target mismatch for {patient}",
        )
        for model, scores in (
            ("labram_identity_v16", v16[index]),
            ("labram_auxiliary_v17", v17[index]),
            ("deepsoz_held_out_ensemble", np.asarray(official_rows[patient]["score"], dtype=np.float64)),
        ):
            row = {**common, **_evaluate(scores, target, mask)}
            rows_by_model[model].append(row)
            if model == "deepsoz_held_out_ensemble":
                for endpoint in ("exact", "neighborhood2", "neighborhood4"):
                    _require(
                        bool(row[endpoint]) == bool(official_source[endpoint]),
                        f"DeepSOZ endpoint replay mismatch for {patient}/{endpoint}",
                    )

    per_repeat: dict[str, object] = {}
    fold_rows = list(official["fold_predictions"])
    labram_by_model = {
        model: {_patient(row["patient_id"]): row for row in rows}
        for model, rows in rows_by_model.items()
        if model.startswith("labram_")
    }
    for repeat in range(3):
        repeat_rows = [row for row in fold_rows if int(row["repeat"]) == repeat]
        repeat_result: dict[str, object] = {"deepsoz": _metrics(repeat_rows)}
        for labram_model, patient_rows in labram_by_model.items():
            matching = [patient_rows[_patient(row["patient_id"])] for row in repeat_rows]
            suffix = labram_model.removeprefix("labram_")
            repeat_result[f"labram_{suffix}_same_patients"] = _metrics(matching)
            repeat_result[f"paired_deepsoz_minus_labram_{suffix}"] = {
                key: {
                    **_bootstrap_delta(repeat_rows, matching, key),
                    **_paired_error_sets(repeat_rows, matching, key),
                }
                for key in ("exact", "neighborhood2", "neighborhood4")
            }
        per_repeat[str(repeat)] = repeat_result
    repeat_macro: dict[str, object] = {}
    for endpoint in ("exact", "neighborhood2", "neighborhood4"):
        values = np.asarray(
            [per_repeat[str(repeat)]["deepsoz"][endpoint] for repeat in range(3)],
            dtype=np.float64,
        )
        repeat_macro[endpoint] = {
            "mean": float(values.mean()),
            "sd_across_three_repeats": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    distributions = {
        model: dict(sorted(Counter(str(row["top1_channel"]) for row in rows).items()))
        for model, rows in rows_by_model.items()
    }
    zero_filled = {
        patient for patient, receipt in receipt_by_patient.items()
        if receipt["zero_filled_channels"]
    }
    zero_fill_strata = {
        "complete_standard19": {
            model: _metrics([row for row in rows if row["patient_id"] not in zero_filled])
            for model, rows in rows_by_model.items()
        },
        "official_zero_fill_compatibility": {
            model: _metrics([row for row in rows if row["patient_id"] in zero_filled])
            for model, rows in rows_by_model.items()
        },
    }

    official_ensemble = rows_by_model["deepsoz_held_out_ensemble"]
    v16_rows = rows_by_model["labram_identity_v16"]
    v17_rows = rows_by_model["labram_auxiliary_v17"]
    return {
        "schema_version": "deepsoz_labram_endpoint_aligned_comparison_v1",
        "status": "descriptive_public_development_comparison_only",
        "patient_count": 102,
        "private_data_used": False,
        "unlabeled_36_used_as_soz_target": False,
        "fairness_note": (
            "DeepSOZ per-repeat rows use one held-out model per patient; its held-out "
            "ensemble averages up to three held-out repeats, whereas LaBraM uses one "
            "five-fold OOF model per patient. Ensemble and single-repeat results are "
            "therefore reported separately."
        ),
        "claim_boundary": (
            "published-weight signal-version transfer versus repeatedly-used public "
            "development LaBraM; neither is an independent external confirmation"
        ),
        "metrics": {model: _metrics(rows) for model, rows in rows_by_model.items()},
        "deepsoz_single_repeat_metrics": per_repeat,
        "deepsoz_single_repeat_macro_summary": repeat_macro,
        "paired_deepsoz_ensemble_minus_labram_v16": {
            key: {
                **_bootstrap_delta(official_ensemble, v16_rows, key),
                **_paired_error_sets(official_ensemble, v16_rows, key),
            }
            for key in ("exact", "neighborhood2", "neighborhood4")
        },
        "paired_deepsoz_ensemble_minus_labram_v17": {
            key: {
                **_bootstrap_delta(official_ensemble, v17_rows, key),
                **_paired_error_sets(official_ensemble, v17_rows, key),
            }
            for key in ("exact", "neighborhood2", "neighborhood4")
        },
        "positive_count_strata": _stratified(
            rows_by_model, field="positive_count", kind="positive"
        ),
        "labram_available_event_count_strata": _stratified(
            rows_by_model, field="labram_event_count", kind="event"
        ),
        "official_consumed_event_count_strata": _stratified(
            rows_by_model, field="official_event_count", kind="event"
        ),
        "event_count_contract_difference": {
            "labram_total": int(sum(event_counts)),
            "official_total": int(
                sum(int(row["event_count"]) for row in receipt_by_patient.values())
            ),
            "equal_patient_count": int(
                sum(
                    event_counts[index]
                    == int(receipt_by_patient[patient]["event_count"])
                    for index, patient in enumerate(patient_ids)
                )
            ),
            "different_patient_count": int(
                sum(
                    event_counts[index]
                    != int(receipt_by_patient[patient]["event_count"])
                    for index, patient in enumerate(patient_ids)
                )
            ),
            "interpretation": (
                "This comparison is pipeline-level, not an encoder-only ablation: "
                "DeepSOZ applies record-level maxSeiz=10 and 600-second contexts, "
                "whereas LaBraM uses its frozen signal-eligible event contract."
            ),
        },
        "official_zero_fill_strata": zero_fill_strata,
        "top1_channel_distributions": distributions,
        "patient_rows": {
            model: rows for model, rows in rows_by_model.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labram-manifest", default="outputs/labram_masked_variable_auxiliary_oof_v17_20260812/manifest.json")
    parser.add_argument("--labram-predictions", default="outputs/labram_masked_variable_auxiliary_oof_v17_20260812/oof_predictions.safetensors")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = compare(parsed)
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": result["metrics"]}, sort_keys=True))
