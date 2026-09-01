#!/usr/bin/env python3
"""Audit event-level private reference stability and prediction correspondence.

This post-open, reference-aware audit uses two already-frozen private models.
It does not train, tune, select, calibrate, aggregate, route, or fuse a model.
Within each patient, it permutes the mapping between frozen event predictions
and documented event-level significant-electrode sets.  The resulting null
asks whether formal event pairing adds agreement beyond patient identity, the
patient's prediction multiset, and the patient's reference-set multiset.

The significant-electrode sets remain evaluation references.  Spread
electrodes are loaded only to verify disjointness and are never positives.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_dual_model_private_construct_repeatability_v71 import (  # noqa: E402
    DEFAULT_PRIVATE_AUDIT,
    DEFAULT_RAW200,
    DEFAULT_V29,
    _load_predictions,
)
from scripts.audit_dual_model_private_hierarchical_repeatability_v72 import (  # noqa: E402
    _candidate_probability,
    _pairwise_metrics,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_private_event_reference_alignment_v74"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_private_event_reference_alignment_v74_20260816"
PERMUTATIONS = 10_000
SEED = 20260874
BOOTSTRAP_REPLICATES = 10_000
ALIGNMENT_METRICS = ("strict", "positive_mass", "reciprocal_first_positive_rank")
REFERENCE_PAIR_METRICS = (
    "reference_exact",
    "reference_any_overlap",
    "reference_jaccard",
    "reference_laterality_equal",
    "reference_cardinality_equal",
)
MODEL_PAIR_METRICS = (
    "top1_agreement",
    "top3_jaccard",
    "rank_spearman",
    "jensen_shannon_distance",
)


def _load_references(
    *,
    events: Sequence[Mapping[str, object]],
    private_audit_directory: Path,
) -> list[dict[str, object]]:
    event_index = {str(row["event_id"]): index for index, row in enumerate(events)}
    candidate_globals = torch.nonzero(V11_CANDIDATE_MASK, as_tuple=False).flatten().tolist()
    candidate_local = {global_index: local for local, global_index in enumerate(candidate_globals)}
    rows: list[dict[str, object]] = []
    source = (private_audit_directory / "private_event_error_audit.csv").resolve(strict=True)
    with source.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            event_id = str(raw["unit_id"])
            if event_id not in event_index:
                raise ValueError(f"reference event not found in frozen prediction roster: {event_id}")
            positive_names = tuple(str(value) for value in ast.literal_eval(raw["positive_channels"]))
            spread_names = tuple(str(value) for value in ast.literal_eval(raw["known_spread_channels"]))
            positive_global = tuple(sorted({CHANNEL_INDEX[value] for value in positive_names}))
            spread_global = tuple(sorted({CHANNEL_INDEX[value] for value in spread_names}))
            if not positive_global or not set(positive_global) <= set(candidate_globals):
                raise ValueError("private significant set must be a nonempty C18 subset")
            if set(positive_global) & set(spread_global):
                raise ValueError("private significant and spread sets must remain disjoint")
            manifest_patient = str(events[event_index[event_id]]["patient_id"])
            if manifest_patient != str(raw["patient_id"]):
                raise ValueError(f"patient identity mismatch for {event_id}")
            rows.append(
                {
                    "event_id": event_id,
                    "patient_id": manifest_patient,
                    "event_index": event_index[event_id],
                    "positive_names": positive_names,
                    "positive_local": tuple(candidate_local[value] for value in positive_global),
                    "spread_names": spread_names,
                    "reference_laterality": str(raw["reference_laterality_stratum"]),
                }
            )
    rows.sort(key=lambda row: int(row["event_index"]))
    if len(rows) != 51 or len({str(row["patient_id"]) for row in rows}) != 23:
        raise ValueError("private reference-evaluable roster changed")
    return rows


def _rank_positions(candidate_probability: np.ndarray) -> np.ndarray:
    order = np.argsort(-candidate_probability, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for event in range(len(order)):
        ranks[event, order[event]] = np.arange(1, order.shape[1] + 1)
    return ranks


def _event_metric(
    *,
    candidate_probability: np.ndarray,
    ranks: np.ndarray,
    prediction_index: int,
    positive_local: Sequence[int],
) -> dict[str, float]:
    positives = np.asarray(positive_local, dtype=np.int64)
    top1 = int(np.argmax(candidate_probability[prediction_index]))
    return {
        "strict": float(top1 in set(positives.tolist())),
        "positive_mass": float(candidate_probability[prediction_index, positives].sum()),
        "reciprocal_first_positive_rank": float(1.0 / ranks[prediction_index, positives].min()),
    }


def _alignment_matrices(
    *,
    candidate_probability: np.ndarray,
    references: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    ranks = _rank_positions(candidate_probability)
    matrices = {
        metric: np.empty((len(references), len(references)), dtype=np.float64)
        for metric in ALIGNMENT_METRICS
    }
    for target_index, reference in enumerate(references):
        for prediction_index in range(len(references)):
            metrics = _event_metric(
                candidate_probability=candidate_probability,
                ranks=ranks,
                prediction_index=prediction_index,
                positive_local=reference["positive_local"],
            )
            for metric, value in metrics.items():
                matrices[metric][target_index, prediction_index] = value
    return matrices


def _assignment_summary(
    *,
    alignment_matrices: Mapping[str, np.ndarray],
    patient_groups: Mapping[str, Sequence[int]],
    prediction_assignment: Sequence[int],
) -> dict[str, dict[str, float]]:
    target_indices = np.arange(len(prediction_assignment), dtype=np.int64)
    event_values = {
        metric: matrix[target_indices, np.asarray(prediction_assignment, dtype=np.int64)]
        for metric, matrix in alignment_matrices.items()
    }
    patient_values: dict[str, dict[str, list[float]]] = {
        patient: {metric: [] for metric in ALIGNMENT_METRICS}
        for patient in patient_groups
    }
    for patient, indices in patient_groups.items():
        for metric in ALIGNMENT_METRICS:
            patient_values[patient][metric] = event_values[metric][list(indices)].tolist()

    multi_patients = [patient for patient in sorted(patient_groups) if len(patient_groups[patient]) >= 2]
    multi_indices = [index for patient in multi_patients for index in patient_groups[patient]]
    output: dict[str, dict[str, float]] = {}
    for scope, patients, indices in (
        ("all", sorted(patient_groups), list(range(len(prediction_assignment)))),
        ("multi_event_only", multi_patients, multi_indices),
    ):
        output[f"{scope}_event_micro"] = {
            metric: float(event_values[metric][indices].mean())
            for metric in ALIGNMENT_METRICS
        }
        output[f"{scope}_patient_equal"] = {
            metric: float(
                np.mean([np.mean(patient_values[patient][metric]) for patient in patients])
            )
            for metric in ALIGNMENT_METRICS
        }
    return output


def _permutation_audit(
    *,
    candidate_probability: np.ndarray,
    references: Sequence[Mapping[str, object]],
    patient_groups: Mapping[str, Sequence[int]],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if repetitions < 100:
        raise ValueError("event-reference alignment audit requires at least 100 permutations")
    alignment_matrices = _alignment_matrices(
        candidate_probability=candidate_probability,
        references=references,
    )
    identity = np.arange(len(references), dtype=np.int64)
    formal = _assignment_summary(
        alignment_matrices=alignment_matrices,
        patient_groups=patient_groups,
        prediction_assignment=identity,
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for repetition in range(repetitions):
        assignment = identity.copy()
        for patient in sorted(patient_groups):
            indices = np.asarray(patient_groups[patient], dtype=np.int64)
            assignment[indices] = rng.permutation(indices)
        summary = _assignment_summary(
            alignment_matrices=alignment_matrices,
            patient_groups=patient_groups,
            prediction_assignment=assignment,
        )
        row: dict[str, object] = {"repetition": repetition}
        for scope, values in summary.items():
            for metric, value in values.items():
                row[f"{scope}_{metric}"] = value
        rows.append(row)

    null: dict[str, object] = {}
    effects: dict[str, float] = {}
    tails: dict[str, float] = {}
    for scope, formal_values in formal.items():
        null[scope] = {}
        for metric, observed in formal_values.items():
            key = f"{scope}_{metric}"
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            null[scope][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "quantile_025_50_975": [
                    float(value) for value in np.quantile(values, (0.025, 0.5, 0.975))
                ],
                "range": [float(values.min()), float(values.max())],
            }
            effects[key] = float(observed - values.mean())
            tails[key] = float((1 + np.sum(values >= observed)) / (repetitions + 1))
    return {
        "formal_event_pairing": formal,
        "within_patient_event_permutation_null": null,
        "formal_minus_null_mean": effects,
        "descriptive_tail_probability_null_ge_formal": tails,
    }, rows


def _set_pair_metrics(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, float]:
    left_set = set(int(value) for value in left["positive_local"])
    right_set = set(int(value) for value in right["positive_local"])
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "reference_exact": float(left_set == right_set),
        "reference_any_overlap": float(bool(intersection)),
        "reference_jaccard": float(len(intersection) / len(union)),
        "reference_laterality_equal": float(
            str(left["reference_laterality"]) == str(right["reference_laterality"])
        ),
        "reference_cardinality_equal": float(len(left_set) == len(right_set)),
    }


def _bootstrap_patient_equal(
    patient_rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    seed: int,
) -> list[float]:
    values = np.asarray([float(row[metric]) for row in patient_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return [float(value) for value in np.quantile(values[sampled].mean(axis=1), (0.025, 0.975))]


def _pair_stability(
    *,
    references: Sequence[Mapping[str, object]],
    patient_groups: Mapping[str, Sequence[int]],
    probabilities: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    selected_probabilities = {
        model: probability[[int(row["event_index"]) for row in references]]
        for model, probability in probabilities.items()
    }
    matrices = {
        model: _pairwise_metrics(probability)
        for model, probability in selected_probabilities.items()
    }
    pair_rows: list[dict[str, object]] = []
    for patient in sorted(patient_groups):
        indices = list(patient_groups[patient])
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                row: dict[str, object] = {
                    "patient_id": patient,
                    "left_event_id": str(references[left]["event_id"]),
                    "right_event_id": str(references[right]["event_id"]),
                    **_set_pair_metrics(references[left], references[right]),
                }
                for model in ("v29", "raw200"):
                    for metric in MODEL_PAIR_METRICS:
                        row[f"{model}_{metric}"] = float(matrices[model][metric][left, right])
                pair_rows.append(row)
    if not pair_rows:
        raise ValueError("no within-patient evaluable event pairs")

    patient_rows: list[dict[str, object]] = []
    by_patient: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_patient[str(row["patient_id"])].append(row)
    all_metrics = list(REFERENCE_PAIR_METRICS) + [
        f"{model}_{metric}"
        for model in ("v29", "raw200")
        for metric in MODEL_PAIR_METRICS
    ]
    for patient in sorted(by_patient):
        rows = by_patient[patient]
        patient_rows.append(
            {
                "patient_id": patient,
                "event_count": len(patient_groups[patient]),
                "event_pair_count": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in all_metrics
                },
            }
        )

    summary: dict[str, object] = {
        "multi_event_patients": len(patient_rows),
        "event_pairs": len(pair_rows),
        "reference": {},
        "models": {},
    }
    for offset, metric in enumerate(REFERENCE_PAIR_METRICS):
        summary["reference"][metric] = {
            "event_pair_micro": float(np.mean([float(row[metric]) for row in pair_rows])),
            "patient_equal": float(np.mean([float(row[metric]) for row in patient_rows])),
            "patient_bootstrap_ci95": _bootstrap_patient_equal(
                patient_rows, metric, seed=SEED + 20_000 + offset
            ),
        }
    for model_index, model in enumerate(("v29", "raw200")):
        summary["models"][model] = {}
        for metric_index, metric in enumerate(MODEL_PAIR_METRICS):
            key = f"{model}_{metric}"
            summary["models"][model][metric] = {
                "event_pair_micro": float(np.mean([float(row[key]) for row in pair_rows])),
                "patient_equal": float(np.mean([float(row[key]) for row in patient_rows])),
                "patient_bootstrap_ci95": _bootstrap_patient_equal(
                    patient_rows,
                    key,
                    seed=SEED + 30_000 + model_index * 100 + metric_index,
                ),
            }
    return summary, patient_rows, pair_rows


def _formal_event_rows(
    *,
    references: Sequence[Mapping[str, object]],
    probabilities: Mapping[str, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, probability in probabilities.items():
        candidate = _candidate_probability(probability)[
            [int(row["event_index"]) for row in references]
        ]
        ranks = _rank_positions(candidate)
        for index, reference in enumerate(references):
            metrics = _event_metric(
                candidate_probability=candidate,
                ranks=ranks,
                prediction_index=index,
                positive_local=reference["positive_local"],
            )
            top1_local = int(np.argmax(candidate[index]))
            candidate_names = tuple(
                STANDARD_19[global_index]
                for global_index in torch.nonzero(V11_CANDIDATE_MASK, as_tuple=False).flatten().tolist()
            )
            rows.append(
                {
                    "model": model,
                    "event_id": str(reference["event_id"]),
                    "patient_id": str(reference["patient_id"]),
                    "top1": candidate_names[top1_local],
                    "positive_channels": list(reference["positive_names"]),
                    "known_spread_channels": list(reference["spread_names"]),
                    **metrics,
                }
            )
    return rows


def run(
    *,
    v29_directory: Path,
    raw200_directory: Path,
    private_audit_directory: Path,
    repetitions: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    events, probabilities = _load_predictions(v29_directory, raw200_directory)
    references = _load_references(events=events, private_audit_directory=private_audit_directory)
    patient_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(references):
        patient_groups[str(row["patient_id"])].append(index)
    multi_patients = [patient for patient in patient_groups if len(patient_groups[patient]) >= 2]

    pair_stability, patient_rows, pair_rows = _pair_stability(
        references=references,
        patient_groups=patient_groups,
        probabilities=probabilities,
    )
    alignment: dict[str, object] = {}
    permutation_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(("v29", "raw200")):
        candidate = _candidate_probability(probabilities[model])[
            [int(row["event_index"]) for row in references]
        ]
        model_result, rows = _permutation_audit(
            candidate_probability=candidate,
            references=references,
            patient_groups=patient_groups,
            repetitions=repetitions,
            seed=SEED + model_index * 100_000,
        )
        for row in rows:
            row["model"] = model
        alignment[model] = model_result
        permutation_rows.extend(rows)

    formal_v29 = alignment["v29"]["formal_event_pairing"]
    formal_raw = alignment["raw200"]["formal_event_pairing"]
    if formal_v29["all_event_micro"]["strict"] != 25 / 51:
        raise RuntimeError("v74 did not reproduce formal private v29 strict agreement")
    if formal_raw["all_event_micro"]["strict"] != 21 / 51:
        raise RuntimeError("v74 did not reproduce formal private Raw200 strict agreement")

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_post_open_private_event_reference_alignment_audit",
        "analysis_role": "reference_aware_longitudinal_private_audit_without_model_change",
        "cohort": {
            "reference_evaluable_events": len(references),
            "patient_clusters": len(patient_groups),
            "multi_event_patient_clusters": len(multi_patients),
            "multi_event_events": sum(len(patient_groups[patient]) for patient in multi_patients),
            "within_patient_event_pairs": len(pair_rows),
        },
        "reference_stability": pair_stability,
        "event_specific_prediction_reference_alignment": alignment,
        "metric_definitions": {
            "reference_jaccard": "intersection over union of two documented significant-electrode sets",
            "strict": "frozen Top-1 membership in the documented event-level significant-electrode set",
            "positive_mass": "renormalized C18 probability mass assigned to the documented event-level set; not calibrated confidence",
            "reciprocal_first_positive_rank": "reciprocal rank of the highest-ranked documented significant electrode",
            "within_patient_event_permutation": (
                "permutes frozen event predictions against the observed event reference sets within each patient; "
                "preserves patient identity, event count, prediction multiset, and reference-set multiset"
            ),
        },
        "audit_contract": {
            "private_significant_reference_loaded_for_evaluation": True,
            "private_spread_loaded_only_for_disjointness_audit": True,
            "private_spread_added_to_positive": False,
            "model_trained_tuned_selected_calibrated_aggregated_routed_or_fused": False,
            "formal_v29_or_raw200_prediction_changed": False,
            "patient_consensus_target_inferred": False,
            "within_patient_permutation_preserves_patient_and_both_event_multisets": True,
            "analysis_designed_after_private_opening": True,
        },
        "interpretation_boundary": {
            "permutation_tail_is_confirmatory_p_value": False,
            "reference_stability_is_cortical_SOZ_stability": False,
            "formal_minus_null_is_new_accuracy": False,
            "audit_can_select_or_fuse_models": False,
            "allowed_claim": (
                "documented event-reference stability and incremental event-specific alignment of two frozen "
                "private rankers are quantified beyond within-patient prediction/reference multisets"
            ),
        },
        "access_receipt": {
            "frozen_private_probability_tensors_loaded": True,
            "private_significant_and_spread_reference_loaded": True,
            "raw_EEG_loaded": False,
            "foundation_forward_performed": False,
            "model_training_or_selection_performed": False,
        },
        "permutation": {
            "repetitions_per_model": repetitions,
            "seed": SEED,
            "unit": "prediction-to-reference assignment within patient",
        },
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPLICATES,
            "unit": "multi-event patient cluster",
        },
        "files": {
            "event_rows": "event_rows.csv",
            "patient_rows": "patient_rows.csv",
            "within_patient_pair_rows": "within_patient_pair_rows.csv",
            "permutation_rows": "permutation_rows.csv",
        },
    }
    return (
        result,
        _formal_event_rows(references=references, probabilities=probabilities),
        patient_rows,
        pair_rows,
        permutation_rows,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    event_rows: Sequence[Mapping[str, object]],
    patient_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    permutation_rows: Sequence[Mapping[str, object]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(staging / "event_rows.csv", event_rows)
        _write_csv(staging / "patient_rows.csv", patient_rows)
        _write_csv(staging / "within_patient_pair_rows.csv", pair_rows)
        _write_csv(staging / "permutation_rows.csv", permutation_rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v29-directory", type=Path, default=DEFAULT_V29)
    parser.add_argument("--raw200-directory", type=Path, default=DEFAULT_RAW200)
    parser.add_argument("--private-audit-directory", type=Path, default=DEFAULT_PRIVATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, event_rows, patient_rows, pair_rows, permutation_rows = run(
        v29_directory=args.v29_directory,
        raw200_directory=args.raw200_directory,
        private_audit_directory=args.private_audit_directory,
        repetitions=args.permutations,
    )
    published = publish(
        output=args.output,
        result=result,
        event_rows=event_rows,
        patient_rows=patient_rows,
        pair_rows=pair_rows,
        permutation_rows=permutation_rows,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
