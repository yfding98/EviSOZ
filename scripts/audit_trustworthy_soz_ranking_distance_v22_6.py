#!/usr/bin/env python3
"""Read-only full-coverage ranking and clinical-distance error audit.

The audit evaluates the already frozen portable H-only predictions.  It does
not fit, select, calibrate, threshold, or route a model and does not read raw
EEG.  Public targets are patient-level DeepSOZ references; private targets are
the already opened event-level descriptive reference and are never mixed into
one statistical denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402


DEFAULT_PUBLIC = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
)
DEFAULT_PRIVATE = ROOT / "outputs/trustworthy_soz_candidate_v21_20260815"
DEFAULT_PRIVATE_SIGNAL = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814/manifest.json"
)
DEFAULT_PRIVATE_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_ranking_distance_v22_6_20260815/result.json"
)

SCHEMA = "trustworthy_soz_ranking_distance_audit_v22_6"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260815

LEFT = frozenset(("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"))
RIGHT = frozenset(("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"))
MIDLINE = frozenset(("FZ", "CZ", "PZ"))

NUMERIC_METRICS = (
    "strict",
    "neighbor_only",
    "relaxed",
    "far",
    "contralateral_far",
    "ipsilateral_far",
    "midline_far_against_unilateral",
    "nonunilateral_reference_far",
    "known_spread_top1",
    "hit_at_3",
    "hit_at_5",
    "positive_recall_at_3",
    "positive_recall_at_5",
    "mrr",
    "average_precision",
    "laterality_agreement",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _json_list(value: str, *, name: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise TypeError(f"{name} must be a JSON string list")
    return parsed


def _side(channel: str) -> str:
    if channel in LEFT:
        return "left"
    if channel in RIGHT:
        return "right"
    if channel in MIDLINE:
        return "midline"
    raise ValueError(f"unknown standard-19 channel: {channel!r}")


def _reference_lateral_sides(positive_indices: set[int]) -> set[str]:
    return {
        _side(STANDARD_19[index])
        for index in positive_indices
        if _side(STANDARD_19[index]) in {"left", "right"}
    }


def _accepted_indices(
    positive_indices: set[int], spread_indices: set[int], evaluable: set[int]
) -> set[int]:
    accepted = set(positive_indices)
    if len(positive_indices) <= 4:
        for index in positive_indices:
            accepted.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
    accepted.intersection_update(evaluable)
    accepted.difference_update(spread_indices - positive_indices)
    return accepted


def rank_row(
    *,
    unit_id: str,
    patient_id: str,
    scores: torch.Tensor,
    positive_indices: set[int],
    evaluable_indices: set[int],
    spread_indices: set[int] | None = None,
) -> dict[str, object]:
    if scores.ndim != 1 or scores.numel() != len(STANDARD_19):
        raise ValueError("scores must have shape [19]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    spread = set() if spread_indices is None else set(spread_indices)
    if not positive_indices or not positive_indices <= evaluable_indices:
        raise ValueError("each row requires an evaluable positive")
    if not evaluable_indices or not evaluable_indices <= set(range(len(STANDARD_19))):
        raise ValueError("invalid evaluable index set")
    if positive_indices & spread:
        raise ValueError("positive and known-spread sets overlap")

    ranking = sorted(
        evaluable_indices,
        key=lambda index: (-float(scores[index]), index),
    )
    exact_score_ties = sum(
        float(scores[ranking[index]]) == float(scores[ranking[index - 1]])
        for index in range(1, len(ranking))
    )
    top = ranking[0]
    accepted = _accepted_indices(positive_indices, spread, evaluable_indices)
    exact = float(top in positive_indices)
    relaxed = float(top in accepted)
    neighbor_only = float(relaxed == 1.0 and exact == 0.0)
    far = 1.0 - relaxed

    positive_ranks = [ranking.index(index) + 1 for index in positive_indices]
    first_positive_rank = min(positive_ranks)
    precision_terms = []
    positives_seen = 0
    for rank, index in enumerate(ranking, start=1):
        if index in positive_indices:
            positives_seen += 1
            precision_terms.append(positives_seen / rank)
    average_precision = float(sum(precision_terms) / len(positive_indices))

    target_lr = _reference_lateral_sides(positive_indices)
    top_side = _side(STANDARD_19[top])
    contralateral = 0.0
    ipsilateral = 0.0
    midline_far = 0.0
    nonunilateral_far = 0.0
    far_subtype: str | None = None
    if far:
        if target_lr == {"left"} and top_side == "right":
            contralateral = 1.0
            far_subtype = "contralateral_far"
        elif target_lr == {"right"} and top_side == "left":
            contralateral = 1.0
            far_subtype = "contralateral_far"
        elif len(target_lr) == 1 and top_side == "midline":
            midline_far = 1.0
            far_subtype = "midline_far_against_unilateral"
        elif len(target_lr) == 1 and top_side in target_lr:
            ipsilateral = 1.0
            far_subtype = "ipsilateral_far"
        else:
            nonunilateral_far = 1.0
            far_subtype = "nonunilateral_reference_far"

    top3 = set(ranking[:3])
    top5 = set(ranking[:5])
    positive_sides = {_side(STANDARD_19[index]) for index in positive_indices}
    return {
        "unit_id": unit_id,
        "patient_id": patient_id,
        "top1": STANDARD_19[top],
        "top1_side": top_side,
        "positive_channels": [STANDARD_19[index] for index in sorted(positive_indices)],
        "known_spread_channels": [STANDARD_19[index] for index in sorted(spread)],
        "reference_lateral_sides": sorted(target_lr),
        "strict": exact,
        "neighbor_only": neighbor_only,
        "relaxed": relaxed,
        "far": far,
        "far_subtype": far_subtype,
        "contralateral_far": contralateral,
        "ipsilateral_far": ipsilateral,
        "midline_far_against_unilateral": midline_far,
        "nonunilateral_reference_far": nonunilateral_far,
        "known_spread_top1": float(top in spread),
        "first_positive_rank": first_positive_rank,
        "hit_at_3": float(bool(top3 & positive_indices)),
        "hit_at_5": float(bool(top5 & positive_indices)),
        "positive_recall_at_3": len(top3 & positive_indices) / len(positive_indices),
        "positive_recall_at_5": len(top5 & positive_indices) / len(positive_indices),
        "mrr": 1.0 / first_positive_rank,
        "average_precision": average_precision,
        "laterality_agreement": float(top_side in positive_sides),
        "exact_ranking_tie_boundaries": exact_score_ties,
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _rank_burden(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ranks = np.asarray([int(row["first_positive_rank"]) for row in rows])
    counts = Counter(
        str(rank) if rank <= 5 else ">5"
        for rank in ranks.tolist()
    )
    return {
        "mean_first_positive_rank": float(ranks.mean()),
        "median_first_positive_rank": float(np.median(ranks)),
        "first_positive_rank_counts": {
            key: counts.get(key, 0) for key in ("1", "2", "3", "4", "5", ">5")
        },
        "candidate_list_size_for_at_least_80pct_hit": next(
            (
                k
                for k in range(1, int(ranks.max()) + 1)
                if float(np.mean(ranks <= k)) >= 0.80
            ),
            None,
        ),
        "candidate_list_size_for_at_least_90pct_hit": next(
            (
                k
                for k in range(1, int(ranks.max()) + 1)
                if float(np.mean(ranks <= k)) >= 0.90
            ),
            None,
        ),
    }


def _bootstrap(
    rows: Sequence[Mapping[str, object]], *, groups: Sequence[str], seed: int
) -> dict[str, list[float]]:
    if len(rows) != len(groups):
        raise ValueError("rows/groups length differs")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    unique = sorted(by_group)
    if len(unique) < 2:
        raise ValueError("bootstrap requires at least two groups")
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in NUMERIC_METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        indices = [index for group in selected_groups for index in by_group[str(group)]]
        for metric in NUMERIC_METRICS:
            values[metric].append(float(np.mean([float(rows[index][metric]) for index in indices])))
    return {
        metric: [float(value) for value in np.quantile(metric_values, [0.025, 0.975])]
        for metric, metric_values in values.items()
    }


def summarize(
    rows: Sequence[Mapping[str, object]], *, groups: Sequence[str], seed: int
) -> dict[str, object]:
    category_counts = Counter(
        "exact" if row["strict"] else "neighbor_only" if row["neighbor_only"] else "far"
        for row in rows
    )
    far_counts = Counter(
        str(row["far_subtype"]) for row in rows if row["far_subtype"] is not None
    )
    metrics = {metric: _mean(rows, metric) for metric in NUMERIC_METRICS}
    metrics["n"] = len(rows)
    metrics["patient_count"] = len(set(groups))
    return {
        "metrics": metrics,
        "category_counts": dict(sorted(category_counts.items())),
        "far_subtype_counts": dict(sorted(far_counts.items())),
        "rank_burden": _rank_burden(rows),
        "bootstrap_ci95": _bootstrap(rows, groups=groups, seed=seed),
        "exact_score_tie_boundaries": sum(
            int(row["exact_ranking_tie_boundaries"]) for row in rows
        ),
        "far_cases": [
            {
                key: row[key]
                for key in (
                    "unit_id",
                    "patient_id",
                    "top1",
                    "positive_channels",
                    "known_spread_channels",
                    "far_subtype",
                    "first_positive_rank",
                )
            }
            for row in rows
            if row["far"]
        ],
    }


def _public_rows(directory: Path) -> list[dict[str, object]]:
    manifest = _read_json(directory / "manifest.json")
    patient_ids = manifest.get("patient_ids")
    if not isinstance(patient_ids, list) or len(patient_ids) != 102:
        raise ValueError("public identity-v16 patient roster drifted")
    tensors = load_file(str((directory / "oof_predictions.safetensors").resolve(strict=True)))
    logits = tensors["oof.frozen_labram_only"].float()
    targets = tensors["targets"].float()
    masks = tensors["target_mask"].bool()
    candidate_mask = tensors["config.candidate_mask"].bool()
    if tuple(logits.shape) != (102, 19) or not torch.equal(candidate_mask, masks[0]):
        raise ValueError("public prediction/mask contract drifted")
    rows = []
    for index, patient_id in enumerate(patient_ids):
        evaluable = set(torch.nonzero(masks[index], as_tuple=False).flatten().tolist())
        positives = set(
            torch.nonzero((targets[index] == 1) & masks[index], as_tuple=False)
            .flatten()
            .tolist()
        )
        rows.append(
            rank_row(
                unit_id=str(patient_id),
                patient_id=str(patient_id),
                scores=logits[index],
                positive_indices=positives,
                evaluable_indices=evaluable,
            )
        )
    return rows


def _private_rows(
    prediction_directory: Path, signal_manifest_path: Path, target_path: Path
) -> list[dict[str, object]]:
    tensors = load_file(str((prediction_directory / "predictions.safetensors").resolve(strict=True)))
    probability = tensors["private_h_only_probability"].float()
    signal_manifest = _read_json(signal_manifest_path)
    events = signal_manifest.get("events")
    if not isinstance(events, list) or len(events) != probability.shape[0]:
        raise ValueError("private signal/prediction roster drifted")
    index_by_event = {str(row["event_id"]): index for index, row in enumerate(events)}
    targets = [
        row
        for row in _read_csv(target_path)
        if row.get("primary_reference_preeligible") == "1"
        and row.get("event_id") in index_by_event
    ]
    if len(targets) != 51 or len({row["patient_id"] for row in targets}) != 23:
        raise ValueError("private primary target denominator drifted")
    evaluable = {index for index, channel in enumerate(STANDARD_19) if channel != "PZ"}
    rows = []
    for target in targets:
        event_id = target["event_id"]
        positives = {
            CHANNEL_INDEX[channel]
            for channel in _json_list(
                target["candidate_positive_electrodes"], name="private positives"
            )
            if channel in STANDARD_19 and channel != "PZ"
        }
        spread = {
            CHANNEL_INDEX[channel]
            for channel in _json_list(
                target["known_spread_electrodes"], name="private spread"
            )
            if channel in STANDARD_19 and channel != "PZ"
        }
        rows.append(
            rank_row(
                unit_id=event_id,
                patient_id=target["patient_id"],
                scores=probability[index_by_event[event_id]],
                positive_indices=positives,
                evaluable_indices=evaluable,
                spread_indices=spread,
            )
        )
    return rows


def audit(
    public_directory: Path,
    private_directory: Path,
    private_signal_manifest: Path,
    private_target: Path,
) -> dict[str, object]:
    public_rows = _public_rows(public_directory)
    private_rows = _private_rows(
        private_directory, private_signal_manifest, private_target
    )
    public = summarize(
        public_rows,
        groups=[str(row["patient_id"]) for row in public_rows],
        seed=BOOTSTRAP_SEED,
    )
    private = summarize(
        private_rows,
        groups=[str(row["patient_id"]) for row in private_rows],
        seed=BOOTSTRAP_SEED + 1,
    )
    return {
        "schema_version": SCHEMA,
        "status": "completed_frozen_full_coverage_ranking_distance_audit",
        "arm": "portable_frozen_labram_h_only_v16",
        "endpoint_policy": {
            "strict": "top1_in_reference_positive_set",
            "neighbor_only": "official_neighborhood4_hit_but_not_strict",
            "far": "outside_positive_and_official_neighborhood4_acceptance",
            "private_known_spread_removed_from_neighbor_acceptance": True,
            "contralateral_far": (
                "far_top1_on_opposite_hemisphere_when_reference_has_one_"
                "nonmidline_side"
            ),
            "ranking_candidate_space": "C18_standard19_without_PZ",
        },
        "public_patient_level": public,
        "private_event_level_patient_clustered": private,
        "comparison_boundary": {
            "public_and_private_units_match": False,
            "public_is_repeatedly_used_development": True,
            "private_is_post_open_descriptive": True,
            "direct_superiority_test": False,
            "neighbor_only_is_strict_localization": False,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "model_weights_loaded": False,
            "existing_frozen_predictions_loaded": True,
            "public_existing_targets_loaded_for_evaluation": True,
            "private_existing_opened_targets_loaded_for_descriptive_evaluation": True,
            "training_performed": False,
            "model_or_threshold_selection_performed": False,
            "candidate_or_report_changed": False,
            "llm_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument(
        "--private-signal-manifest", type=Path, default=DEFAULT_PRIVATE_SIGNAL
    )
    parser.add_argument("--private-target", type=Path, default=DEFAULT_PRIVATE_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        args.public, args.private, args.private_signal_manifest, args.private_target
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "public": result["public_patient_level"]["metrics"],
                "private": result["private_event_level_patient_clustered"]["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
