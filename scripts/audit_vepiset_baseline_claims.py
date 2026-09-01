#!/usr/bin/env python3
"""Audit class-imbalance baselines and claim safety for VEPiSet results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
except ImportError:  # pragma: no cover
    accuracy_score = balanced_accuracy_score = f1_score = None


CLASS_NAMES: Tuple[str, ...] = (
    "Non-IED",
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)

DEFAULT_MAIN_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87")
DEFAULT_LOGPROB_RUN = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logprob_calibrator_macro_cv_valacc87")
DEFAULT_MAIN_LOGPROB_ENSEMBLE = Path("outputs/vepiset_ied_v2_full6_patientclasssplit_main_logprob_ensemble_macro_valacc87")
DEFAULT_HIGH_ACCURACY_AUDIT = Path("outputs/vepiset_ied_v2_full6_patientclasssplit_main_singlebias_accuracy_valmacro43")


def load_prediction_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    targets = np.asarray([CLASS_NAMES.index(row["target"]) for row in rows], dtype=np.int64)
    if all(f"prob_{name}" in rows[0] for name in CLASS_NAMES):
        probs = np.asarray(
            [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows],
            dtype=np.float64,
        )
        preds = probs.argmax(axis=1)
    else:
        preds = np.asarray([CLASS_NAMES.index(row["prediction"]) for row in rows], dtype=np.int64)
    return targets, preds, rows


def compute_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if accuracy_score is None:
        return {"accuracy": float((targets == preds).mean())}
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
    }


def class_support(targets: np.ndarray) -> Dict[str, int]:
    counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    return {CLASS_NAMES[idx]: int(counts[idx]) for idx in range(len(CLASS_NAMES))}


def majority_baseline(targets: np.ndarray) -> Dict[str, object]:
    counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    majority_idx = int(counts.argmax())
    preds = np.full_like(targets, majority_idx)
    return {
        "class": CLASS_NAMES[majority_idx],
        "support": int(counts[majority_idx]),
        "fraction": float(counts[majority_idx] / max(len(targets), 1)),
        "metrics": compute_metrics(targets, preds),
    }


def metric_delta(left: Mapping[str, float], right: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }


def load_run_metrics(run_dir: Path) -> Dict[str, object]:
    targets, preds, _ = load_prediction_csv(run_dir / "test_predictions.csv")
    return {
        "run_dir": str(run_dir),
        "metrics": compute_metrics(targets, preds),
        "supports": class_support(targets),
    }


def maybe_load_run_metrics(run_dir: Path) -> Dict[str, object] | None:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        return None
    return load_run_metrics(run_dir)


def best_accuracy_run(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not runs:
        return {}
    return dict(max(runs, key=lambda item: float(item.get("metrics", {}).get("accuracy", 0.0))))


def recommended_high_accuracy_run(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    eligible = [
        item
        for item in runs
        if float(item.get("delta_vs_majority", {}).get("accuracy", 0.0)) > 0.0
    ]
    if not eligible:
        return best_accuracy_run(runs)
    return dict(max(
        eligible,
        key=lambda item: (
            float(item.get("metrics", {}).get("macro_f1", 0.0)),
            float(item.get("metrics", {}).get("balanced_accuracy", 0.0)),
            float(item.get("metrics", {}).get("accuracy", 0.0)),
        ),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", default=str(DEFAULT_MAIN_RUN))
    parser.add_argument("--logprob-run", default=str(DEFAULT_LOGPROB_RUN))
    parser.add_argument("--main-logprob-ensemble-run", default=str(DEFAULT_MAIN_LOGPROB_ENSEMBLE))
    parser.add_argument("--high-accuracy-audit-run", default=str(DEFAULT_HIGH_ACCURACY_AUDIT))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-macro-f1", type=float, default=0.40)
    args = parser.parse_args()

    main_run = Path(args.main_run)
    output_json = Path(args.output_json) if args.output_json else main_run / "baseline_claim_audit.json"

    targets, preds, _ = load_prediction_csv(main_run / "test_predictions.csv")
    main_metrics = compute_metrics(targets, preds)
    majority = majority_baseline(targets)
    majority_metrics = majority["metrics"]
    deltas = metric_delta(main_metrics, majority_metrics)

    comparison_runs: List[Dict[str, object]] = []
    for label, run_dir in (
        ("single_bias_high_accuracy_audit", Path(args.high_accuracy_audit_run)),
        ("logprob_high_accuracy_audit", Path(args.logprob_run)),
        ("main_logprob_validation_ensemble_audit", Path(args.main_logprob_ensemble_run)),
    ):
        loaded = maybe_load_run_metrics(run_dir)
        if loaded is None:
            continue
        loaded["label"] = label
        loaded["delta_vs_main"] = metric_delta(loaded["metrics"], main_metrics)
        loaded["delta_vs_majority"] = metric_delta(loaded["metrics"], majority_metrics)
        comparison_runs.append(loaded)
    max_accuracy_audit = best_accuracy_run(comparison_runs)
    high_accuracy_operating_point = recommended_high_accuracy_run(comparison_runs)

    checks = {
        "nominal_accuracy_ge_threshold": bool(main_metrics["accuracy"] >= float(args.min_accuracy)),
        "nominal_macro_f1_ge_threshold": bool(main_metrics["macro_f1"] >= float(args.min_macro_f1)),
        "accuracy_beats_majority": bool(deltas["accuracy"] > 0.0),
        "balanced_accuracy_beats_majority": bool(deltas["balanced_accuracy"] > 0.0),
        "macro_f1_beats_majority": bool(deltas["macro_f1"] > 0.0),
        "weighted_f1_beats_majority": bool(deltas["weighted_f1"] > 0.0),
        "accuracy_claim_needs_class_imbalance_caveat": bool(deltas["accuracy"] <= 0.0),
    }
    audit = {
        "main_run": str(main_run),
        "supports": class_support(targets),
        "main_metrics": main_metrics,
        "majority_baseline": majority,
        "main_delta_vs_majority": deltas,
        "comparison_runs": comparison_runs,
        "max_accuracy_audit": max_accuracy_audit,
        "high_accuracy_operating_point": high_accuracy_operating_point,
        "claim_checks": checks,
        "recommended_accuracy_wording": (
            "Report the >80% accuracy threshold as satisfied, but state that window-level "
            "accuracy is below the all-Non-IED majority baseline because VEPiSet is highly "
            "imbalanced. Emphasize macro-F1, balanced accuracy, weighted-F1, patient-level "
            "proxy, and IED-positive localization instead."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "main_accuracy": main_metrics["accuracy"],
        "majority_accuracy": majority_metrics["accuracy"],
        "main_macro_f1": main_metrics["macro_f1"],
        "majority_macro_f1": majority_metrics["macro_f1"],
        "accuracy_beats_majority": checks["accuracy_beats_majority"],
        "macro_f1_beats_majority": checks["macro_f1_beats_majority"],
        "accuracy_claim_needs_class_imbalance_caveat": checks["accuracy_claim_needs_class_imbalance_caveat"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
