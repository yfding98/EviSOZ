#!/usr/bin/env python3
"""Audit a strict patient-disjoint VEPiSet IED result.

The script recomputes metrics from prediction CSVs, checks patient overlap
against a split summary, and writes a compact JSON artifact.  It is meant to be
used as a guardrail for SOZ-like VEPiSet experiments: test metrics are reported
only after the split and predictions already exist.
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


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"patient_id", "target", "prediction"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return rows


def labels_from_rows(rows: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([CLASS_NAMES.index(row["target"]) for row in rows], dtype=np.int64)
    preds = np.asarray([CLASS_NAMES.index(row["prediction"]) for row in rows], dtype=np.int64)
    return targets, preds


def compute_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    targets, preds = labels_from_rows(rows)
    out: Dict[str, object] = {}
    if accuracy_score is None:
        out["accuracy"] = float((targets == preds).mean())
        return out
    out["accuracy"] = float(accuracy_score(targets, preds))
    out["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    out["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(targets, preds, average="weighted", zero_division=0))
    majority = int(np.bincount(targets, minlength=len(CLASS_NAMES)).argmax())
    majority_preds = np.full_like(targets, majority)
    out["majority_class"] = CLASS_NAMES[majority]
    out["majority_accuracy"] = float(accuracy_score(targets, majority_preds))
    out["majority_macro_f1"] = float(f1_score(targets, majority_preds, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    out["per_class"] = {
        CLASS_NAMES[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(CLASS_NAMES))
    }
    out["confusion_matrix"] = confusion_matrix(
        targets,
        preds,
        labels=list(range(len(CLASS_NAMES))),
    ).astype(int).tolist()
    out["classification_report"] = classification_report(
        targets,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        zero_division=0,
        output_dict=True,
    )
    return out


def patient_ids(rows: Sequence[Mapping[str, str]]) -> List[str]:
    return sorted({str(row["patient_id"]) for row in rows})


def load_split_patients(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train_split_meta", "val_split_meta", "test_split_meta"):
        if isinstance(data, dict) and key in data:
            patients = data[key].get("patients", {})
            return {
                "train": sorted(str(v) for v in patients.get("train", [])),
                "val": sorted(str(v) for v in patients.get("val", [])),
                "test": sorted(str(v) for v in patients.get("test", [])),
            }
    if isinstance(data, dict) and "patients" in data:
        patients = data["patients"]
        return {
            "train": sorted(str(v) for v in patients.get("train", [])),
            "val": sorted(str(v) for v in patients.get("val", [])),
            "test": sorted(str(v) for v in patients.get("test", [])),
        }
    raise ValueError(f"Could not find patient split metadata in {path}")


def overlap_report(split_patients: Mapping[str, Sequence[str]]) -> Dict[str, object]:
    train = set(split_patients.get("train", []))
    val = set(split_patients.get("val", []))
    test = set(split_patients.get("test", []))
    return {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
        "has_overlap": bool((train & val) or (train & test) or (val & test)),
    }


def class_supports(rows: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for row in rows:
        counts[row["target"]] += 1
    return counts


def patient_class_supports(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        patient = str(row["patient_id"])
        out.setdefault(patient, {name: 0 for name in CLASS_NAMES})
        out[patient][row["target"]] += 1
    return out


def compare_metric_dict(recomputed: Mapping[str, object], recorded: Mapping[str, object] | None) -> Dict[str, float]:
    if not recorded:
        return {}
    out: Dict[str, float] = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        if key in recorded and key in recomputed:
            out[key] = abs(float(recomputed[key]) - float(recorded[key]))
    return out


def load_recorded_test_metrics(run_dir: Path) -> Mapping[str, object] | None:
    candidates = (
        "patient_prior_metrics.json",
        "calibrated_metrics.json",
        "logprob_calibrator_metrics.json",
        "test_metrics.json",
    )
    for name in candidates:
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "test_metrics" in data:
            return data["test_metrics"]
        if {"accuracy", "macro_f1"}.issubset(data):
            return data
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-summary", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-test-accuracy", type=float, default=0.0)
    parser.add_argument("--min-test-macro-f1", type=float, default=0.0)
    parser.add_argument("--require-no-patient-overlap", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    split_summary = Path(args.split_summary)
    val_rows = load_rows(run_dir / "val_predictions.csv")
    test_rows = load_rows(run_dir / "test_predictions.csv")
    split_patients = load_split_patients(split_summary)
    observed_patients = {
        "val": patient_ids(val_rows),
        "test": patient_ids(test_rows),
    }
    overlap = overlap_report(split_patients)
    val_metrics = compute_metrics(val_rows)
    test_metrics = compute_metrics(test_rows)
    recorded_test = load_recorded_test_metrics(run_dir)
    audit = {
        "run_dir": str(run_dir),
        "split_summary": str(split_summary),
        "split_patients": split_patients,
        "observed_patients": observed_patients,
        "patient_overlap": overlap,
        "val_supports": class_supports(val_rows),
        "test_supports": class_supports(test_rows),
        "test_patient_class_supports": patient_class_supports(test_rows),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "recorded_test_metric_abs_diff": compare_metric_dict(test_metrics, recorded_test),
    }

    output_json = Path(args.output_json) if args.output_json else run_dir / "strict_result_audit.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    failures: List[str] = []
    if args.require_no_patient_overlap and bool(overlap["has_overlap"]):
        failures.append("patient overlap detected")
    if float(test_metrics.get("accuracy", 0.0)) < float(args.min_test_accuracy):
        failures.append(
            f"test accuracy {float(test_metrics.get('accuracy', 0.0)):.4f} < {args.min_test_accuracy:.4f}"
        )
    if float(test_metrics.get("macro_f1", 0.0)) < float(args.min_test_macro_f1):
        failures.append(
            f"test macro-F1 {float(test_metrics.get('macro_f1', 0.0)):.4f} < {args.min_test_macro_f1:.4f}"
        )

    print(json.dumps({
        "run_dir": str(run_dir),
        "output_json": str(output_json),
        "patient_overlap": overlap,
        "val_accuracy": val_metrics.get("accuracy", 0.0),
        "val_macro_f1": val_metrics.get("macro_f1", 0.0),
        "test_accuracy": test_metrics.get("accuracy", 0.0),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
        "test_macro_f1": test_metrics.get("macro_f1", 0.0),
        "test_weighted_f1": test_metrics.get("weighted_f1", 0.0),
        "failures": failures,
    }, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
