"""Label-aware EviSOZ localization evaluation primitives.

These functions are evaluator-side only.  They consume a released field
payload, never teacher/derived candidates or physician text, and preserve
incomplete-positive semantics when computing candidate-set metrics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.soz.geometry import STANDARD_19

from .metrics import (
    brier_score_multiclass,
    correction_corruption_rates,
    expected_calibration_error,
    mean_reciprocal_rank,
    risk_coverage_curve,
    top_k_candidate_hit,
)


NODE_LATERALITY: dict[str, str] = {
    **{name: "left" for name in ("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1")},
    **{name: "right" for name in ("FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2")},
    "FZ": "midline", "CZ": "midline", "PZ": "midline",
}
NODE_REGION: dict[str, str] = {
    **{name: "frontal" for name in ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8")},
    **{name: "temporal" for name in ("T7", "T8")},
    **{name: "central" for name in ("C3", "CZ", "C4")},
    **{name: "parietal" for name in ("P7", "P3", "PZ", "P4", "P8")},
    **{name: "occipital" for name in ("O1", "O2")},
}


def extract_released_node_target(
    field_release: Mapping[str, object],
    *,
    observed_node_mask: Sequence[bool] | None = None,
) -> dict[str, Any] | None:
    """Extract one direct node-label target, or ``None`` if unavailable."""

    fields = field_release.get("fields")
    if not isinstance(fields, list):
        raise ValueError("field_release.fields must be a list")
    matches = [
        field for field in fields
        if isinstance(field, Mapping)
        and field.get("state") == "provided"
        and field.get("semantic_role") == "node_label"
        and field.get("authority") in {"physician", "dataset_direct"}
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple direct node-label fields are ambiguous")
    field = matches[0]
    payload = field.get("value_payload")
    if not isinstance(payload, Mapping) or set(payload) != {"values", "semantics"}:
        raise ValueError("released node-label payload is malformed")
    values = payload["values"]
    semantics = payload["semantics"]
    if not isinstance(values, list) or not values or any(value not in STANDARD_19 for value in values):
        raise ValueError("released node-label values are outside Standard19")
    if len(values) != len(set(values)) or semantics not in {"exhaustive", "incomplete_positive", "unknown"}:
        raise ValueError("released node-label semantics are invalid")
    indices = {STANDARD_19.index(value) for value in values}
    if observed_node_mask is not None:
        if len(observed_node_mask) != len(STANDARD_19):
            raise ValueError("observed_node_mask must contain 19 values")
        indices &= {index for index, observed in enumerate(observed_node_mask) if observed}
    if not indices:
        return None
    return {
        "positive_indices": tuple(sorted(indices)),
        "semantics": semantics,
        "source_field_id": str(field.get("field_id", "")),
        "completeness": semantics == "exhaustive",
    }


def _predicted_group(values: Sequence[float], mapping: Mapping[str, str], labels: Sequence[str]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for index, score in enumerate(values):
        scores[mapping[STANDARD_19[index]]] += float(score)
    return max(labels, key=lambda label: scores.get(label, 0.0))


def evaluate_localization_predictions(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Mapping[str, Any] | None],
    *,
    event_ids: Sequence[str] | None = None,
    laterality_targets: Sequence[str | None] | None = None,
    region_targets: Sequence[Sequence[str] | None] | None = None,
) -> dict[str, Any]:
    """Evaluate event-level candidate localization with incomplete-safe masks."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STANDARD_19) or values.shape[0] != len(targets):
        raise ValueError("probabilities/targets must align as [N,19]")
    if values.shape[0] < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if event_ids is None:
        event_ids = [f"event-{index}" for index in range(values.shape[0])]
    if len(event_ids) != values.shape[0] or len(set(event_ids)) != len(event_ids):
        raise ValueError("event_ids must be unique and aligned")
    rows = [(index, target) for index, target in enumerate(targets) if target is not None]
    if not rows:
        raise ValueError("no evaluable direct localization targets")
    positive_sets = [row[1]["positive_indices"] for row in rows]
    selected = values[[row[0] for row in rows]]
    top1 = top_k_candidate_hit(selected, positive_sets, k=1)
    top3 = top_k_candidate_hit(selected, positive_sets, k=3)
    mrr = mean_reciprocal_rank(selected, positive_sets)
    confidence = selected.max(axis=1)
    correctness = np.asarray([
        bool(set(np.argsort(-row, kind="stable")[:1]) & set(positive))
        for row, positive in zip(selected, positive_sets)
    ], dtype=bool)
    output: dict[str, Any] = {
        "status": "clinical_localization_evaluation",
        "event_count": int(values.shape[0]),
        "evaluable_event_count": len(rows),
        "non_evaluable_event_count": int(values.shape[0] - len(rows)),
        "metrics": {
            "top1_candidate_hit": top1,
            "top3_candidate_hit": top3,
            "mrr": mrr,
            "ece": expected_calibration_error(confidence, correctness),
            "risk_coverage": risk_coverage_curve(confidence, correctness),
        },
        "event_ids": list(event_ids),
        "evaluator_policy": {
            "direct_labels_only": True,
            "teacher_candidates_used": False,
            "derived_candidates_used": False,
            "physician_report_text_used": False,
            "tcp22_edges_expanded_to_nodes": False,
        },
    }
    if laterality_targets is not None:
        if len(laterality_targets) != values.shape[0]:
            raise ValueError("laterality_targets must align with probabilities")
        valid = [
            (index, target) for index, target in rows
            if laterality_targets[index] in {"left", "right", "midline"}
        ]
        if valid:
            predicted = [
                _predicted_group(values[index], NODE_LATERALITY, ("left", "right", "midline"))
                for index, _ in valid
            ]
            output["metrics"]["laterality_accuracy"] = float(
                np.mean([pred == laterality_targets[index] for pred, (index, _) in zip(predicted, valid)])
            )
    if region_targets is not None:
        if len(region_targets) != values.shape[0]:
            raise ValueError("region_targets must align with probabilities")
        valid = [
            (index, target) for index, target in rows
            if isinstance(region_targets[index], Sequence) and region_targets[index]
        ]
        if valid:
            predicted = [
                _predicted_group(values[index], NODE_REGION, ("frontal", "temporal", "central", "parietal", "occipital"))
                for index, _ in valid
            ]
            output["metrics"]["region_hit"] = float(
                np.mean([pred in {str(item).split("_")[-1] for item in region_targets[index]} for pred, (index, _) in zip(predicted, valid)])
            )
    return output


def aggregate_event_probabilities_by_patient(
    patient_ids: Sequence[str],
    event_probabilities: Sequence[Sequence[float]],
    localization_quality: Sequence[float],
    signal_quality: Sequence[float],
    uncertainty: Sequence[float],
    *,
    localizing_event_mask: Sequence[bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate event distributions using the frozen patient-level policy.

    The operation is evaluator-side and accepts model probabilities only.  It
    never consumes direct labels, teacher candidates, or report text.  Events
    marked non-localizing contribute no node probability but are counted so a
    report/evaluation layer can preserve the abstention rate.
    """

    values = np.asarray(event_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STANDARD_19) or values.shape[0] < 1:
        raise ValueError("event_probabilities must have shape [N,19]")
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("event probabilities must be finite non-negative rows summing to one")
    n = values.shape[0]
    if len(patient_ids) != n or not all(isinstance(item, str) and item for item in patient_ids):
        raise ValueError("patient_ids must align with event probabilities")
    if len(set(patient_ids)) < 1:
        raise ValueError("patient_ids must not be empty")
    quality = {}
    for name, raw in (
        ("localization_quality", localization_quality),
        ("signal_quality", signal_quality),
        ("uncertainty", uncertainty),
    ):
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != n or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite vector aligned with events")
        quality[name] = np.clip(array, 0.0, 1.0)
    if localizing_event_mask is None:
        localizing = np.ones(n, dtype=bool)
    else:
        localizing = np.asarray(localizing_event_mask, dtype=bool)
        if localizing.ndim != 1 or localizing.shape[0] != n:
            raise ValueError("localizing_event_mask must align with events")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        groups[patient_id].append(index)
    output: dict[str, dict[str, Any]] = {}
    for patient_id in sorted(groups):
        indices = np.asarray(groups[patient_id], dtype=np.int64)
        weights = (
            quality["localization_quality"][indices]
            * quality["signal_quality"][indices]
            * (1.0 - quality["uncertainty"][indices])
            * localizing[indices].astype(np.float64)
        )
        denominator = float(weights.sum())
        if denominator <= 0:
            patient_probability = None
            abstain = True
        else:
            patient_probability = (weights[:, None] * values[indices]).sum(axis=0) / denominator
            patient_probability = patient_probability.tolist()
            abstain = False
        output[patient_id] = {
            "patient_probability": patient_probability,
            "event_ids": [int(index) for index in indices],
            "event_count": int(indices.size),
            "nonlocalizing_event_count": int((~localizing[indices]).sum()),
            "weights": weights.tolist(),
            "abstain": abstain,
        }
    return output


def evaluate_patient_localization_predictions(
    patient_probabilities: Mapping[str, Sequence[float]],
    targets: Mapping[str, Mapping[str, Any] | None],
    *,
    laterality_targets: Mapping[str, str | None] | None = None,
    region_targets: Mapping[str, Sequence[str] | None] | None = None,
    baseline_patient_probabilities: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Evaluate patient-level probabilities and optional frozen-baseline deltas.

    Only released direct node targets are accepted in ``targets``.  The
    optional baseline comparison reports correction/corruption rates against
    exactly the same patients and target sets; it does not alter predictions.
    """

    if not isinstance(patient_probabilities, Mapping) or not patient_probabilities:
        raise ValueError("patient_probabilities must be a non-empty mapping")
    if not all(isinstance(key, str) and key for key in patient_probabilities):
        raise ValueError("patient probability keys must be non-empty strings")
    if not all(isinstance(key, str) and key for key in targets):
        raise ValueError("target keys must be non-empty strings")
    keys = sorted(str(key) for key in patient_probabilities)
    if len(keys) != len(set(keys)) or set(keys) != set(targets):
        raise ValueError("patient probabilities and targets must have the same patient roster")
    rows = [patient_probabilities[key] for key in keys]
    target_rows = [targets[key] for key in keys]
    laterality = [laterality_targets.get(key) if laterality_targets is not None else None for key in keys]
    regions = [region_targets.get(key) if region_targets is not None else None for key in keys]
    result = evaluate_localization_predictions(
        rows,
        target_rows,
        event_ids=keys,
        laterality_targets=laterality,
        region_targets=regions,
    )
    result["status"] = "clinical_patient_localization_evaluation"
    result["patient_count"] = len(keys)
    # The event evaluator intentionally computes ECE/risk coverage on max
    # class confidence.  Add a multiclass Brier score only where each target
    # is an exhaustive singleton; set-valued/incomplete labels remain safe.
    singleton = [
        (index, row)
        for index, row in enumerate(target_rows)
        if row is not None and row.get("semantics") == "exhaustive" and len(row.get("positive_indices", ())) == 1
    ]
    if singleton:
        result["metrics"]["brier_singleton_exhaustive"] = brier_score_multiclass(
            [rows[index] for index, _ in singleton],
            [int(row["positive_indices"][0]) for _, row in singleton],
        )
    result["evaluator_policy"] = {
        "patient_level": True,
        "direct_labels_only": True,
        "teacher_candidates_used": False,
        "derived_candidates_used": False,
        "physician_report_text_used": False,
        "tcp22_edges_expanded_to_nodes": False,
    }
    if baseline_patient_probabilities is not None:
        if set(baseline_patient_probabilities) != set(patient_probabilities):
            raise ValueError("baseline and new patient rosters must match")
        baseline_rows = [baseline_patient_probabilities[key] for key in keys]
        baseline_eval = evaluate_localization_predictions(
            baseline_rows,
            target_rows,
            event_ids=keys,
        )
        new_correct = []
        base_correct = []
        for row, base, target in zip(rows, baseline_rows, target_rows):
            if target is None:
                continue
            positive = set(target["positive_indices"])
            new_correct.append(bool(set(np.argsort(-np.asarray(row), kind="stable")[:1]) & positive))
            base_correct.append(bool(set(np.argsort(-np.asarray(base), kind="stable")[:1]) & positive))
        result["baseline_comparison"] = {
            "baseline_metrics": baseline_eval["metrics"],
            "new_metrics": result["metrics"],
            "correction_corruption": correction_corruption_rates(base_correct, new_correct),
            "same_patient_roster": True,
        }
    return result


__all__ = [
    "aggregate_event_probabilities_by_patient",
    "NODE_LATERALITY",
    "NODE_REGION",
    "evaluate_localization_predictions",
    "evaluate_patient_localization_predictions",
    "extract_released_node_target",
]
