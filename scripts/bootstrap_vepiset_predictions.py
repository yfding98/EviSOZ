#!/usr/bin/env python3
"""Patient-level bootstrap audit for VEPiSet prediction CSVs.

The usual window-level test metrics can look deceptively precise because each
test patient contributes hundreds of windows.  This script resamples patients
instead of windows, giving a more honest uncertainty estimate for patient-wise
splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


CLASS_NAMES: Tuple[str, ...] = (
    "Non-IED",
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)


def load_prediction_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    targets = np.asarray([CLASS_NAMES.index(row["target"]) for row in rows], dtype=np.int64)
    if "prediction" in rows[0] and rows[0]["prediction"]:
        pred = np.asarray([CLASS_NAMES.index(row["prediction"]) for row in rows], dtype=np.int64)
    else:
        probs = np.asarray(
            [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows],
            dtype=np.float64,
        )
        pred = probs.argmax(axis=1).astype(np.int64)
    patients = [str(row.get("patient_id", "")) for row in rows]
    return targets, pred, patients


def metrics_from_predictions(targets: np.ndarray, pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    matrix = np.bincount(
        targets.astype(np.int64) * int(n_classes) + pred.astype(np.int64),
        minlength=int(n_classes) * int(n_classes),
    ).reshape(int(n_classes), int(n_classes))
    support = matrix.sum(axis=1).astype(np.float64)
    pred_support = matrix.sum(axis=0).astype(np.float64)
    tp = np.diag(matrix).astype(np.float64)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = max(float(support.sum()), 1.0)
    out = {
        "accuracy": float(tp.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
    }
    for idx, name in enumerate(CLASS_NAMES):
        out[f"f1_{name}"] = float(f1[idx])
        out[f"recall_{name}"] = float(recall[idx])
        out[f"support_{name}"] = int(support[idx])
    return out


def group_by_patient(patients: Sequence[str]) -> Dict[str, np.ndarray]:
    by_patient: Dict[str, List[int]] = defaultdict(list)
    for idx, patient_id in enumerate(patients):
        by_patient[patient_id].append(idx)
    return {patient_id: np.asarray(indices, dtype=np.int64) for patient_id, indices in by_patient.items()}


def percentile_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p2_5": float(np.percentile(arr, 2.5)),
        "p50": float(np.percentile(arr, 50.0)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    targets, pred, patients = load_prediction_csv(Path(args.predictions))
    by_patient = group_by_patient(patients)
    patient_ids = sorted(by_patient)
    rng = random.Random(int(args.seed))

    full_metrics = metrics_from_predictions(targets, pred, len(CLASS_NAMES))

    bootstrap_values: Dict[str, List[float]] = defaultdict(list)
    for _ in range(int(args.n_bootstrap)):
        sampled_patients = [rng.choice(patient_ids) for _ in patient_ids]
        indices = np.concatenate([by_patient[patient_id] for patient_id in sampled_patients])
        metrics = metrics_from_predictions(targets[indices], pred[indices], len(CLASS_NAMES))
        for key, value in metrics.items():
            if key.startswith("support_"):
                continue
            bootstrap_values[key].append(float(value))

    loo_rows = []
    for held_out in patient_ids:
        kept = np.concatenate([indices for patient_id, indices in by_patient.items() if patient_id != held_out])
        metrics = metrics_from_predictions(targets[kept], pred[kept], len(CLASS_NAMES))
        held_indices = by_patient[held_out]
        held_metrics = metrics_from_predictions(targets[held_indices], pred[held_indices], len(CLASS_NAMES))
        loo_rows.append({
            "held_out_patient": held_out,
            "remaining_metrics": metrics,
            "held_out_metrics": held_metrics,
            "held_out_n": int(len(held_indices)),
        })

    payload = {
        "predictions": str(Path(args.predictions)),
        "n_patients": len(patient_ids),
        "n_windows": int(len(targets)),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "note": (
            "Bootstrap metrics use the fixed six-class label set. If a resampled "
            "patient cohort lacks a class, that class contributes zero recall/F1; "
            "this intentionally reflects uncertainty from rare-class patient sparsity."
        ),
        "full_metrics": full_metrics,
        "bootstrap_ci": {key: percentile_ci(values) for key, values in sorted(bootstrap_values.items())},
        "leave_one_patient_out": loo_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "predictions": str(Path(args.predictions)),
        "output": str(output),
        "n_patients": len(patient_ids),
        "full_accuracy": full_metrics["accuracy"],
        "full_balanced_accuracy": full_metrics["balanced_accuracy"],
        "full_macro_f1": full_metrics["macro_f1"],
        "full_weighted_f1": full_metrics["weighted_f1"],
        "macro_f1_ci": payload["bootstrap_ci"]["macro_f1"],
        "balanced_accuracy_ci": payload["bootstrap_ci"]["balanced_accuracy"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
