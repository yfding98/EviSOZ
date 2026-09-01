#!/usr/bin/env python3
"""Target-blind hierarchical repeatability audit for two frozen private rankers.

The audit asks whether event predictions from the same patient are more similar
than predictions from different patients when the complete C18 distribution
and ranking are considered.  It uses no SOZ/significant/spread reference and
does not infer a patient-consensus target.  Patient-label permutations preserve
the observed event-count profile and are descriptive because this audit was
designed after the private cohort had been opened.
"""

from __future__ import annotations

import argparse
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
    DEFAULT_RAW200,
    DEFAULT_V29,
    _load_predictions,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_dual_model_private_hierarchical_repeatability_v72"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_dual_model_private_hierarchical_repeatability_v72_20260816"
PERMUTATIONS = 10_000
SEED = 20260872
BOOTSTRAP_REPLICATES = 10_000
METRICS = (
    "top1_agreement",
    "top3_jaccard",
    "rank_spearman",
    "jensen_shannon_distance",
)
SIMILARITY_METRICS = frozenset(METRICS[:3])


def _candidate_probability(probability: torch.Tensor) -> np.ndarray:
    selected = probability[:, V11_CANDIDATE_MASK].double().numpy()
    selected = np.clip(selected, 1e-12, None)
    selected /= selected.sum(axis=1, keepdims=True)
    return selected


def _pairwise_metrics(probability: torch.Tensor) -> dict[str, np.ndarray]:
    values = _candidate_probability(probability)
    n_events, n_channels = values.shape
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for event in range(n_events):
        ranks[event, order[event]] = np.arange(1, n_channels + 1)
    top1 = order[:, 0]
    top3 = [set(row[:3].tolist()) for row in order]
    matrices = {
        metric: np.eye(n_events, dtype=np.float64)
        for metric in METRICS
    }
    matrices["jensen_shannon_distance"].fill(0.0)
    spearman_denominator = n_channels * (n_channels**2 - 1)
    for left in range(n_events):
        for right in range(left + 1, n_events):
            top1_agreement = float(top1[left] == top1[right])
            top3_jaccard = len(top3[left] & top3[right]) / len(top3[left] | top3[right])
            squared_rank_difference = float(
                np.square(ranks[left] - ranks[right]).sum()
            )
            rank_spearman = 1.0 - 6.0 * squared_rank_difference / spearman_denominator
            midpoint = 0.5 * (values[left] + values[right])
            js_divergence = 0.5 * (
                np.sum(values[left] * np.log(values[left] / midpoint))
                + np.sum(values[right] * np.log(values[right] / midpoint))
            )
            js_distance = float(np.sqrt(max(js_divergence, 0.0) / np.log(2.0)))
            pair = {
                "top1_agreement": top1_agreement,
                "top3_jaccard": top3_jaccard,
                "rank_spearman": rank_spearman,
                "jensen_shannon_distance": js_distance,
            }
            for metric, value in pair.items():
                matrices[metric][left, right] = value
                matrices[metric][right, left] = value
    return matrices


def _mean_pairs(matrix: np.ndarray, indices: Sequence[int]) -> float:
    selected = np.asarray(indices, dtype=np.int64)
    upper = np.triu_indices(len(selected), k=1)
    return float(matrix[np.ix_(selected, selected)][upper].mean())


def _within_patient(
    matrices: Mapping[str, np.ndarray],
    patient_groups: Mapping[str, Sequence[int]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for patient_id in sorted(patient_groups):
        indices = patient_groups[patient_id]
        if len(indices) < 2:
            continue
        row: dict[str, object] = {
            "patient_id": patient_id,
            "event_count": len(indices),
            "event_pair_count": len(indices) * (len(indices) - 1) // 2,
        }
        for metric in METRICS:
            row[metric] = _mean_pairs(matrices[metric], indices)
        rows.append(row)
    return {
        "patients": len(rows),
        "events": sum(int(row["event_count"]) for row in rows),
        "event_pairs": sum(int(row["event_pair_count"]) for row in rows),
        "patient_equal": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
    }, rows


def _between_patient(
    matrices: Mapping[str, np.ndarray],
    patient_groups: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    patients = sorted(patient_groups)
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    event_pairs = 0
    for left_index, left_patient in enumerate(patients):
        left = np.asarray(patient_groups[left_patient], dtype=np.int64)
        for right_patient in patients[left_index + 1 :]:
            right = np.asarray(patient_groups[right_patient], dtype=np.int64)
            event_pairs += len(left) * len(right)
            for metric in METRICS:
                values[metric].append(
                    float(matrices[metric][np.ix_(left, right)].mean())
                )
    return {
        "patient_pairs": len(patients) * (len(patients) - 1) // 2,
        "event_pairs": event_pairs,
        "patient_pair_equal": {
            metric: float(np.mean(values[metric])) for metric in METRICS
        },
    }


def _permutation_null(
    *,
    matrices: Mapping[str, np.ndarray],
    group_sizes: Sequence[int],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    if repetitions < 100:
        raise ValueError("hierarchical repeatability audit requires at least 100 permutations")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n_events = next(iter(matrices.values())).shape[0]
    for repetition in range(repetitions):
        permutation = rng.permutation(n_events)
        offset = 0
        patient_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for size in group_sizes:
            indices = permutation[offset : offset + size]
            offset += size
            if size < 2:
                continue
            for metric in METRICS:
                patient_values[metric].append(_mean_pairs(matrices[metric], indices))
        if offset != n_events:
            raise RuntimeError("permutation group sizes do not cover event roster")
        rows.append(
            {
                "repetition": float(repetition),
                **{
                    metric: float(np.mean(patient_values[metric]))
                    for metric in METRICS
                },
            }
        )
    summary: dict[str, object] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "range": [float(values.min()), float(values.max())],
            "quantile_05_50_95": [
                float(value) for value in np.quantile(values, (0.05, 0.5, 0.95))
            ],
        }
    return summary, rows


def _paired_model_differences(
    v29_rows: Sequence[Mapping[str, object]],
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    raw_by_patient = {str(row["patient_id"]): row for row in raw_rows}
    output: dict[str, object] = {}
    for offset, metric in enumerate(METRICS):
        values = np.asarray(
            [
                float(row[metric]) - float(raw_by_patient[str(row["patient_id"])][metric])
                for row in v29_rows
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(SEED + 50_000 + offset)
        sampled = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
        )
        bootstrap = values[sampled].mean(axis=1)
        output[metric] = {
            "patients": len(values),
            "patient_equal_delta_v29_minus_raw200": float(values.mean()),
            "patient_bootstrap_ci95": [
                float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
            ],
        }
    return output


def _pair_rows(
    *,
    events: Sequence[Mapping[str, object]],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            row: dict[str, object] = {
                "left_event_id": str(events[left]["event_id"]),
                "right_event_id": str(events[right]["event_id"]),
                "left_patient_id": str(events[left]["patient_id"]),
                "right_patient_id": str(events[right]["patient_id"]),
                "same_patient": str(events[left]["patient_id"]) == str(events[right]["patient_id"]),
            }
            for model in ("v29", "raw200"):
                for metric in METRICS:
                    row[f"{model}_{metric}"] = float(matrices[model][metric][left, right])
            rows.append(row)
    return rows


def run(
    *,
    v29_directory: Path,
    raw200_directory: Path,
    repetitions: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    events, probabilities = _load_predictions(v29_directory, raw200_directory)
    patient_groups: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        patient_groups[str(event["patient_id"])].append(index)
    group_sizes = [len(patient_groups[patient]) for patient in sorted(patient_groups)]
    if len(events) != 88 or len(patient_groups) != 31 or sum(size >= 2 for size in group_sizes) != 28:
        raise ValueError("private patient/event roster changed")

    matrices = {
        model: _pairwise_metrics(probability)
        for model, probability in probabilities.items()
    }
    model_results: dict[str, object] = {}
    patient_rows: dict[str, list[dict[str, object]]] = {}
    permutation_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(("v29", "raw200")):
        within, rows = _within_patient(matrices[model], patient_groups)
        between = _between_patient(matrices[model], patient_groups)
        null, null_rows = _permutation_null(
            matrices=matrices[model],
            group_sizes=group_sizes,
            repetitions=repetitions,
            seed=SEED + model_index * 100_000,
        )
        tails: dict[str, float] = {}
        effects: dict[str, float] = {}
        for metric in METRICS:
            observed = float(within["patient_equal"][metric])
            null_values = np.asarray([float(row[metric]) for row in null_rows])
            if metric in SIMILARITY_METRICS:
                tails[metric] = float((1 + np.sum(null_values >= observed)) / (repetitions + 1))
                effects[metric] = observed - float(null[metric]["mean"])
            else:
                tails[metric] = float((1 + np.sum(null_values <= observed)) / (repetitions + 1))
                effects[metric] = float(null[metric]["mean"]) - observed
        for row in rows:
            row["model"] = model
        for row in null_rows:
            row["model"] = model
        patient_rows[model] = rows
        permutation_rows.extend(null_rows)
        model_results[model] = {
            "within_patient": within,
            "between_patient": between,
            "event_count_preserving_permutation_null": null,
            "same_patient_similarity_advantage_over_null": effects,
            "descriptive_tail_probability": tails,
        }

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_target_blind_dual_model_private_hierarchical_repeatability_audit",
        "analysis_role": "post_open_target_blind_patient_structure_audit",
        "cohort": {
            "events": len(events),
            "patients": len(patient_groups),
            "multi_event_patients": sum(size >= 2 for size in group_sizes),
            "single_event_patients": sum(size == 1 for size in group_sizes),
            "event_count_range": [min(group_sizes), max(group_sizes)],
            "group_sizes_preserved_in_null": True,
        },
        "metric_definitions": {
            "top1_agreement": "binary equality of C18 Top-1 candidates",
            "top3_jaccard": "intersection over union of C18 Top-3 sets",
            "rank_spearman": "Spearman correlation of complete C18 rank positions",
            "jensen_shannon_distance": "square-root Jensen-Shannon distance in bits on renormalized C18 probability, range 0 to 1",
        },
        "models": model_results,
        "paired_v29_minus_raw200_within_patient": _paired_model_differences(
            patient_rows["v29"], patient_rows["raw200"]
        ),
        "audit_contract": {
            "SOZ_significant_or_spread_reference_loaded": False,
            "patient_identity_used_only_for_grouping": True,
            "patient_consensus_target_inferred": False,
            "model_trained_tuned_selected_calibrated_aggregated_or_fused": False,
            "formal_v29_or_raw200_prediction_changed": False,
            "permutation_preserves_event_count_profile": True,
        },
        "interpretation_boundary": {
            "same_patient_similarity_is_SOZ_stability": False,
            "same_patient_similarity_excludes_acquisition_or_session_shortcuts": False,
            "descriptive_tail_is_confirmatory_p_value": False,
            "between_patient_difference_is_clinical_discrimination": False,
            "allowed_claim": (
                "complete C18 probability and rank similarities within patients are "
                "quantified relative to between-patient and event-count-preserving null structure"
            ),
        },
        "access_receipt": {
            "frozen_private_probability_tensors_loaded": True,
            "private_SOZ_reference_loaded": False,
            "raw_EEG_loaded": False,
            "foundation_forward_performed": False,
            "model_training_or_selection_performed": False,
        },
        "permutation": {
            "repetitions": repetitions,
            "seed": SEED,
            "unit": "event assignment to fixed patient-sized groups",
        },
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPLICATES,
            "unit": "multi-event patient",
        },
        "files": {
            "patient_rows": "patient_rows.csv",
            "event_pair_rows": "event_pair_rows.csv",
            "permutation_rows": "permutation_rows.csv",
        },
    }
    flat_patient_rows = patient_rows["v29"] + patient_rows["raw200"]
    return result, flat_patient_rows, _pair_rows(events=events, matrices=matrices), permutation_rows


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
    patient_rows: Sequence[Mapping[str, object]],
    event_pair_rows: Sequence[Mapping[str, object]],
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
        _write_csv(staging / "patient_rows.csv", patient_rows)
        _write_csv(staging / "event_pair_rows.csv", event_pair_rows)
        _write_csv(staging / "permutation_rows.csv", permutation_rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v29-directory", type=Path, default=DEFAULT_V29)
    parser.add_argument("--raw200-directory", type=Path, default=DEFAULT_RAW200)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, patient_rows, event_pair_rows, permutation_rows = run(
        v29_directory=args.v29_directory,
        raw200_directory=args.raw200_directory,
        repetitions=args.permutations,
    )
    output = publish(
        output=args.output,
        result=result,
        patient_rows=patient_rows,
        event_pair_rows=event_pair_rows,
        permutation_rows=permutation_rows,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "v29_within_rank_spearman": result["models"]["v29"]["within_patient"]["patient_equal"]["rank_spearman"],
                "raw200_within_rank_spearman": result["models"]["raw200"]["within_patient"]["patient_equal"]["rank_spearman"],
                "v29_rank_tail": result["models"]["v29"]["descriptive_tail_probability"]["rank_spearman"],
                "raw200_rank_tail": result["models"]["raw200"]["descriptive_tail_probability"]["rank_spearman"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
