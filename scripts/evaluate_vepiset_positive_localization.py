#!/usr/bin/env python3
"""Evaluate VEPiSet IED-positive spatial localization from prediction CSVs.

The 6-class VEPiSet task combines Non-IED detection and IED spatial
classification.  This audit filters to true IED windows and asks whether the
model's top positive class is correct when Non-IED is removed from the decision
set.  It is a localization proxy, not clinical SOZ evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
except ImportError:  # pragma: no cover
    accuracy_score = balanced_accuracy_score = classification_report = None
    confusion_matrix = f1_score = precision_recall_fscore_support = None


CLASS_NAMES: Tuple[str, ...] = (
    "Non-IED",
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)
POSITIVE_CLASS_NAMES: Tuple[str, ...] = CLASS_NAMES[1:]


def load_prediction_csv(path: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"patient_id", "target", "prediction"} | {f"prob_{name}" for name in CLASS_NAMES}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return rows


def positive_arrays(rows: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[Mapping[str, str]]]:
    positive_rows = [row for row in rows if str(row["target"]) != "Non-IED"]
    if not positive_rows:
        raise ValueError("No positive IED rows found")
    targets = np.asarray(
        [POSITIVE_CLASS_NAMES.index(str(row["target"])) for row in positive_rows],
        dtype=np.int64,
    )
    probs = np.asarray(
        [[float(row[f"prob_{name}"]) for name in POSITIVE_CLASS_NAMES] for row in positive_rows],
        dtype=np.float64,
    )
    return targets, probs, positive_rows


def compute_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if accuracy_score is None:
        out["accuracy"] = float((targets == preds).mean())
        return out
    out["accuracy"] = float(accuracy_score(targets, preds))
    out["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    out["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(targets, preds, average="weighted", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(len(POSITIVE_CLASS_NAMES))),
        zero_division=0,
    )
    out["per_class"] = {
        POSITIVE_CLASS_NAMES[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(POSITIVE_CLASS_NAMES))
    }
    out["confusion_matrix"] = confusion_matrix(
        targets,
        preds,
        labels=list(range(len(POSITIVE_CLASS_NAMES))),
    ).astype(int).tolist()
    out["classification_report"] = classification_report(
        targets,
        preds,
        labels=list(range(len(POSITIVE_CLASS_NAMES))),
        target_names=list(POSITIVE_CLASS_NAMES),
        zero_division=0,
        output_dict=True,
    )
    return out


def patient_metrics(positive_rows: Sequence[Mapping[str, str]], preds: np.ndarray) -> Dict[str, object]:
    grouped: Dict[str, List[int]] = {}
    for idx, row in enumerate(positive_rows):
        grouped.setdefault(str(row["patient_id"]), []).append(idx)
    patient_records: List[Dict[str, object]] = []
    hits: List[bool] = []
    for patient_id, indices in sorted(grouped.items()):
        true_classes = sorted({str(positive_rows[idx]["target"]) for idx in indices})
        pred_labels = [POSITIVE_CLASS_NAMES[int(preds[idx])] for idx in indices]
        counts = {name: pred_labels.count(name) for name in POSITIVE_CLASS_NAMES}
        top_pred = max(POSITIVE_CLASS_NAMES, key=lambda name: (counts[name], name))
        hit = top_pred in set(true_classes)
        hits.append(hit)
        patient_records.append({
            "patient_id": patient_id,
            "n_positive_windows": len(indices),
            "true_positive_classes": true_classes,
            "top_predicted_class": top_pred,
            "hit": bool(hit),
            "predicted_class_counts": counts,
        })
    return {
        "n_positive_patients": len(patient_records),
        "hit_count": int(sum(hits)),
        "hit_accuracy": float(sum(hits) / max(len(hits), 1)),
        "patients": patient_records,
    }


def evaluate_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    targets, raw_positive_probs, positive_rows = positive_arrays(rows)
    raw_preds = raw_positive_probs.argmax(axis=1)
    conditional_probs = raw_positive_probs / np.clip(raw_positive_probs.sum(axis=1, keepdims=True), 1e-12, None)
    conditional_preds = conditional_probs.argmax(axis=1)
    return {
        "n_positive_windows": int(len(positive_rows)),
        "class_support": {
            name: int((targets == idx).sum())
            for idx, name in enumerate(POSITIVE_CLASS_NAMES)
        },
        "raw_positive_argmax": {
            "window_metrics": compute_metrics(targets, raw_preds),
            "patient_metrics": patient_metrics(positive_rows, raw_preds),
        },
        "conditional_positive": {
            "window_metrics": compute_metrics(targets, conditional_preds),
            "patient_metrics": patient_metrics(positive_rows, conditional_preds),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    val_payload = evaluate_rows(load_prediction_csv(run_dir / "val_predictions.csv"))
    test_payload = evaluate_rows(load_prediction_csv(run_dir / "test_predictions.csv"))
    payload = {
        "run_dir": str(run_dir),
        "positive_classes": list(POSITIVE_CLASS_NAMES),
        "val": val_payload,
        "test": test_payload,
    }
    output_json = Path(args.output_json) if args.output_json else run_dir / "positive_localization_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    test_cond = test_payload["conditional_positive"]
    test_raw = test_payload["raw_positive_argmax"]
    print(json.dumps({
        "run_dir": str(run_dir),
        "output_json": str(output_json),
        "val_conditional_positive_accuracy": val_payload["conditional_positive"]["window_metrics"]["accuracy"],
        "val_conditional_positive_macro_f1": val_payload["conditional_positive"]["window_metrics"]["macro_f1"],
        "test_conditional_positive_accuracy": test_cond["window_metrics"]["accuracy"],
        "test_conditional_positive_balanced_accuracy": test_cond["window_metrics"]["balanced_accuracy"],
        "test_conditional_positive_macro_f1": test_cond["window_metrics"]["macro_f1"],
        "test_conditional_positive_patient_hit_accuracy": test_cond["patient_metrics"]["hit_accuracy"],
        "test_raw_positive_accuracy": test_raw["window_metrics"]["accuracy"],
        "test_raw_positive_macro_f1": test_raw["window_metrics"]["macro_f1"],
        "test_raw_positive_patient_hit_accuracy": test_raw["patient_metrics"]["hit_accuracy"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
