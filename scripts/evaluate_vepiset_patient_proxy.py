#!/usr/bin/env python3
"""Patient-level VEPiSet IED spatial-distribution proxy evaluation.

This script aggregates window-level predictions to patients.  It reports two
separate proxy tasks:

1. Positive-known localization: evaluate only patients with at least one IED
   window, and ask whether the top aggregated positive class is in that
   patient's true positive class set.
2. Thresholded all-patient proxy: tune a patient-level positive-evidence
   threshold on validation patients only.  Patients below threshold are
   predicted as Non-IED; otherwise the top positive class is predicted.

These metrics are SOZ-like patient-level proxies, not clinical SOZ metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


CLASS_NAMES: Tuple[str, ...] = (
    "Non-IED",
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)


def load_prediction_csv(path: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"patient_id", "target", "prediction"} | {f"prob_{name}" for name in CLASS_NAMES}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return rows


def aggregate_scores(rows: Sequence[Mapping[str, str]], mode: str, top_frac: float) -> np.ndarray:
    probs = np.asarray(
        [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows],
        dtype=np.float64,
    )
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    mode = str(mode).lower()
    if mode == "mean":
        return probs.mean(axis=0)
    if mode == "top_frac":
        frac = float(top_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("--top-frac must be in (0, 1]")
        k = max(1, int(np.ceil(frac * probs.shape[0])))
        return np.asarray([np.sort(probs[:, idx])[-k:].mean() for idx in range(probs.shape[1])])
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def group_by_patient(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Mapping[str, str]]]:
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]), []).append(row)
    return grouped


def summarize_patients(
    rows: Sequence[Mapping[str, str]],
    *,
    mode: str,
    top_frac: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for patient_id, group in sorted(group_by_patient(rows).items()):
        scores = aggregate_scores(group, mode, top_frac)
        true_positive = sorted({str(row["target"]) for row in group if str(row["target"]) != "Non-IED"})
        top_positive_idx = 1 + int(np.argmax(scores[1:]))
        all_top_idx = int(np.argmax(scores))
        out.append({
            "patient_id": patient_id,
            "n_windows": len(group),
            "true_positive_classes": true_positive,
            "is_ied_positive": bool(true_positive),
            "all_top_class": CLASS_NAMES[all_top_idx],
            "top_positive_class": CLASS_NAMES[top_positive_idx],
            "non_ied_score": float(scores[0]),
            "top_positive_score": float(scores[top_positive_idx]),
            "scores": {CLASS_NAMES[idx]: float(scores[idx]) for idx in range(len(CLASS_NAMES))},
        })
    return out


def known_positive_metrics(patients: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    positives = [p for p in patients if bool(p["is_ied_positive"])]
    hits = [
        str(p["top_positive_class"]) in set(p["true_positive_classes"])  # type: ignore[arg-type]
        for p in positives
    ]
    return {
        "n_positive_patients": len(positives),
        "hit_count": int(sum(hits)),
        "hit_accuracy": float(sum(hits) / max(len(hits), 1)),
        "patients": [
            {
                "patient_id": p["patient_id"],
                "true_positive_classes": p["true_positive_classes"],
                "top_positive_class": p["top_positive_class"],
                "hit": bool(hit),
                "top_positive_score": p["top_positive_score"],
                "non_ied_score": p["non_ied_score"],
            }
            for p, hit in zip(positives, hits)
        ],
    }


def thresholded_hits(
    patients: Sequence[Mapping[str, object]],
    threshold: float,
) -> Tuple[List[bool], List[str]]:
    hits: List[bool] = []
    preds: List[str] = []
    for patient in patients:
        pred = "Non-IED"
        if float(patient["top_positive_score"]) >= float(threshold):
            pred = str(patient["top_positive_class"])
        preds.append(pred)
        true_positive = set(patient["true_positive_classes"])  # type: ignore[arg-type]
        if true_positive:
            hits.append(pred in true_positive)
        else:
            hits.append(pred == "Non-IED")
    return hits, preds


def thresholded_metrics(
    patients: Sequence[Mapping[str, object]],
    threshold: float,
) -> Dict[str, object]:
    hits, preds = thresholded_hits(patients, threshold)
    positives = [idx for idx, p in enumerate(patients) if bool(p["is_ied_positive"])]
    non_ied = [idx for idx, p in enumerate(patients) if not bool(p["is_ied_positive"])]
    pos_hits = [hits[idx] for idx in positives]
    non_hits = [hits[idx] for idx in non_ied]
    return {
        "threshold": float(threshold),
        "n_patients": len(patients),
        "hit_count": int(sum(hits)),
        "hit_accuracy": float(sum(hits) / max(len(hits), 1)),
        "positive_hit_count": int(sum(pos_hits)),
        "n_positive_patients": len(pos_hits),
        "positive_hit_accuracy": float(sum(pos_hits) / max(len(pos_hits), 1)),
        "non_ied_hit_count": int(sum(non_hits)),
        "n_non_ied_patients": len(non_hits),
        "non_ied_hit_accuracy": float(sum(non_hits) / max(len(non_hits), 1)),
        "patients": [
            {
                "patient_id": patient["patient_id"],
                "true_positive_classes": patient["true_positive_classes"],
                "prediction": pred,
                "hit": bool(hit),
                "top_positive_class": patient["top_positive_class"],
                "top_positive_score": patient["top_positive_score"],
                "non_ied_score": patient["non_ied_score"],
            }
            for patient, pred, hit in zip(patients, preds, hits)
        ],
    }


def score_for_selector(metrics: Mapping[str, object], selector: str) -> Tuple[float, float, float]:
    selector = str(selector).lower()
    hit = float(metrics["hit_accuracy"])
    pos = float(metrics["positive_hit_accuracy"])
    non = float(metrics["non_ied_hit_accuracy"])
    threshold = float(metrics["threshold"])
    if selector == "hit_accuracy":
        return hit, pos, non
    if selector == "positive_hit_accuracy":
        return pos, hit, non
    if selector == "balanced_patient_accuracy":
        return (pos + non) / 2.0, hit, pos
    if selector == "conservative":
        return hit, (pos + non) / 2.0, -threshold
    raise ValueError(f"Unsupported selector: {selector}")


def tune_threshold(
    val_patients: Sequence[Mapping[str, object]],
    *,
    selector: str,
    steps: int,
) -> Tuple[float, Dict[str, object]]:
    scores = sorted({float(p["top_positive_score"]) for p in val_patients})
    candidates = [0.0, 1.0]
    candidates.extend(scores)
    for left, right in zip(scores, scores[1:]):
        candidates.append((left + right) / 2.0)
    if steps > 0:
        candidates.extend(np.linspace(0.0, 1.0, int(steps)).tolist())
    best_threshold = 0.0
    best_metrics: Dict[str, object] = {}
    best_score: Tuple[float, float, float] | None = None
    for threshold in sorted(set(float(v) for v in candidates)):
        metrics = thresholded_metrics(val_patients, threshold)
        score = score_for_selector(metrics, selector)
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--aggregation", choices=["mean", "top_frac"], default="mean")
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument(
        "--selector",
        choices=["hit_accuracy", "positive_hit_accuracy", "balanced_patient_accuracy", "conservative"],
        default="balanced_patient_accuracy",
    )
    parser.add_argument("--threshold-steps", type=int, default=501)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    val_patients = summarize_patients(
        load_prediction_csv(run_dir / "val_predictions.csv"),
        mode=args.aggregation,
        top_frac=args.top_frac,
    )
    test_patients = summarize_patients(
        load_prediction_csv(run_dir / "test_predictions.csv"),
        mode=args.aggregation,
        top_frac=args.top_frac,
    )
    threshold, val_threshold_metrics = tune_threshold(
        val_patients,
        selector=args.selector,
        steps=args.threshold_steps,
    )
    test_threshold_metrics = thresholded_metrics(test_patients, threshold)
    payload = {
        "run_dir": str(run_dir),
        "aggregation": args.aggregation,
        "top_frac": float(args.top_frac),
        "selector": args.selector,
        "selected_threshold": float(threshold),
        "val_known_positive": known_positive_metrics(val_patients),
        "test_known_positive": known_positive_metrics(test_patients),
        "val_thresholded": val_threshold_metrics,
        "test_thresholded": test_threshold_metrics,
    }
    output_json = Path(args.output_json) if args.output_json else run_dir / "patient_proxy_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run_dir),
        "output_json": str(output_json),
        "aggregation": args.aggregation,
        "top_frac": float(args.top_frac),
        "selector": args.selector,
        "selected_threshold": float(threshold),
        "val_known_positive_accuracy": payload["val_known_positive"]["hit_accuracy"],
        "test_known_positive_accuracy": payload["test_known_positive"]["hit_accuracy"],
        "val_thresholded_hit_accuracy": payload["val_thresholded"]["hit_accuracy"],
        "test_thresholded_hit_accuracy": payload["test_thresholded"]["hit_accuracy"],
        "test_thresholded_positive_hit_accuracy": payload["test_thresholded"]["positive_hit_accuracy"],
        "test_thresholded_non_ied_hit_accuracy": payload["test_thresholded"]["non_ied_hit_accuracy"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
