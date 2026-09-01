#!/usr/bin/env python3
"""Audit frozen-v29 sensitivity to ambiguity inside documented positive sets.

The audit never adds an unlabeled electrode and never promotes private spread
to a positive.  It evaluates the unchanged ranking against (a) the complete
documented set, (b) each documented positive as a counterfactual singleton,
and (c) every leave-one-documented-positive-out set.  These are sensitivity
functionals, not alternative gold standards or new performance estimates.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
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
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_v29_reference_set_perturbation_v67"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PUBLIC_V16 = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_PRIVATE_PREDICTIONS = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_PRIVATE_AUDIT = ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_reference_set_perturbation_v67_20260816"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260867
CANDIDATES = tuple(channel for channel in STANDARD_19 if channel != "PZ")


def _rank_positions(probability: torch.Tensor, evaluable: torch.Tensor) -> tuple[list[int], dict[int, int]]:
    order = evaluable[torch.argsort(probability[evaluable], descending=True, stable=True)].tolist()
    return order, {int(channel): rank + 1 for rank, channel in enumerate(order)}


def _unit_row(
    *,
    dataset: str,
    unit_id: str,
    patient_id: str,
    probability: torch.Tensor,
    evaluable_indices: Sequence[int],
    positive_indices: Sequence[int],
    spread_indices: Sequence[int],
) -> dict[str, object]:
    evaluable = torch.tensor(sorted(set(int(value) for value in evaluable_indices)), dtype=torch.long)
    positives = tuple(sorted(set(int(value) for value in positive_indices)))
    spread = tuple(sorted(set(int(value) for value in spread_indices)))
    if not positives or not set(positives) <= set(evaluable.tolist()):
        raise ValueError("positive set must be a nonempty evaluable subset")
    if set(positives) & set(spread):
        raise ValueError("positive and spread sets must remain disjoint")
    order, ranks = _rank_positions(probability, evaluable)
    top1 = order[0]
    top3 = set(order[:3])
    top5 = set(order[:5])
    positive_set = set(positives)
    size = len(positives)
    strict = float(top1 in positive_set)
    hit3 = float(bool(top3 & positive_set))
    hit5 = float(bool(top5 & positive_set))
    singleton_top1 = [float(top1 == value) for value in positives]
    singleton_hit3 = [float(value in top3) for value in positives]
    singleton_hit5 = [float(value in top5) for value in positives]
    singleton_ranks = [ranks[value] for value in positives]
    probability_values = [float(probability[value]) for value in positives]

    if size > 1:
        leave_sets = [positive_set - {removed} for removed in positives]
        leave_top1 = [float(top1 in values) for values in leave_sets]
        leave_hit3 = [float(bool(top3 & values)) for values in leave_sets]
        leave_hit5 = [float(bool(top5 & values)) for values in leave_sets]
    else:
        leave_top1 = []
        leave_hit3 = []
        leave_hit5 = []

    return {
        "dataset": dataset,
        "unit_id": unit_id,
        "patient_id": patient_id,
        "top1": STANDARD_19[top1],
        "positive_channels": [STANDARD_19[value] for value in positives],
        "known_spread_channels": [STANDARD_19[value] for value in spread],
        "positive_set_size": size,
        "original_set_strict": strict,
        "original_set_hit_at_3": hit3,
        "original_set_hit_at_5": hit5,
        "original_first_positive_rank": min(singleton_ranks),
        "documented_singleton_uniform_top1": float(np.mean(singleton_top1)),
        "documented_singleton_uniform_hit_at_3": float(np.mean(singleton_hit3)),
        "documented_singleton_uniform_hit_at_5": float(np.mean(singleton_hit5)),
        "documented_singleton_mean_rank": float(np.mean(singleton_ranks)),
        "documented_singleton_worst_rank": int(max(singleton_ranks)),
        "documented_singleton_top1_lower": float(min(singleton_top1)),
        "documented_singleton_top1_upper": float(max(singleton_top1)),
        "set_positive_probability_mass": float(sum(probability_values)),
        "documented_singleton_mean_probability": float(np.mean(probability_values)),
        "set_cardinality_gain_top1": strict - float(np.mean(singleton_top1)),
        "leave_one_eligible": size > 1,
        "leave_one_mean_top1": None if not leave_top1 else float(np.mean(leave_top1)),
        "leave_one_worst_top1": None if not leave_top1 else float(min(leave_top1)),
        "leave_one_mean_hit_at_3": None if not leave_hit3 else float(np.mean(leave_hit3)),
        "leave_one_mean_hit_at_5": None if not leave_hit5 else float(np.mean(leave_hit5)),
        "single_positive_deletion_can_remove_top1_hit": bool(leave_top1 and strict == 1.0 and min(leave_top1) == 0.0),
        "known_spread_top1": float(top1 in set(spread)),
    }


def _public_rows(public_directory: Path, v16_directory: Path) -> list[dict[str, object]]:
    payload = load_file(str((public_directory / "oof_predictions.safetensors").resolve(strict=True)), device="cpu")
    probability = payload["oof.portable_equal_ensemble_probability"].float()
    targets = payload["targets"].float()
    mask = payload["target_mask"].bool()
    manifest = json.loads((v16_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8"))
    patient_ids = [str(value) for value in manifest["patient_ids"]]
    if tuple(probability.shape) != (102, 19) or len(patient_ids) != 102:
        raise ValueError("public v29 roster changed")
    rows = []
    for index, patient_id in enumerate(patient_ids):
        rows.append(_unit_row(
            dataset="public_consumed_development",
            unit_id=f"PUBLIC-{index:03d}",
            patient_id=patient_id,
            probability=probability[index],
            evaluable_indices=torch.nonzero(mask[index], as_tuple=False).flatten().tolist(),
            positive_indices=torch.nonzero((targets[index] == 1) & mask[index], as_tuple=False).flatten().tolist(),
            spread_indices=(),
        ))
    return rows


def _private_rows(prediction_directory: Path, audit_directory: Path) -> list[dict[str, object]]:
    manifest = json.loads((prediction_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8"))
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private v29 prediction roster changed")
    event_index = {str(row["event_id"]): index for index, row in enumerate(events)}
    probability = load_file(str((prediction_directory / "predictions.safetensors").resolve(strict=True)), device="cpu")["private_portable_equal_probability"].float()
    rows = []
    with (audit_directory / "private_event_error_audit.csv").resolve(strict=True).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            positives = [CHANNEL_INDEX[str(value)] for value in ast.literal_eval(raw["positive_channels"])]
            spread = [CHANNEL_INDEX[str(value)] for value in ast.literal_eval(raw["known_spread_channels"])]
            rows.append(_unit_row(
                dataset="private_post_open_transport",
                unit_id=str(raw["unit_id"]),
                patient_id=str(raw["patient_id"]),
                probability=probability[event_index[str(raw["unit_id"])]],
                evaluable_indices=[CHANNEL_INDEX[value] for value in CANDIDATES],
                positive_indices=positives,
                spread_indices=spread,
            ))
    if len(rows) != 51 or len({row["patient_id"] for row in rows}) != 23:
        raise ValueError("private evaluable roster changed")
    return rows


METRICS = (
    "original_set_strict",
    "original_set_hit_at_3",
    "original_set_hit_at_5",
    "documented_singleton_uniform_top1",
    "documented_singleton_uniform_hit_at_3",
    "documented_singleton_uniform_hit_at_5",
    "documented_singleton_mean_rank",
    "documented_singleton_worst_rank",
    "documented_singleton_top1_lower",
    "documented_singleton_top1_upper",
    "set_cardinality_gain_top1",
    "known_spread_top1",
)


def _patient_values(rows: Sequence[Mapping[str, object]], metric: str) -> np.ndarray:
    bags: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bags[str(row["patient_id"])].append(float(row[metric]))
    return np.asarray([np.mean(bags[key]) for key in sorted(bags)], dtype=np.float64)


def _metric_summary(rows: Sequence[Mapping[str, object]], metric: str, *, seed: int) -> dict[str, object]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    patients = _patient_values(rows, metric)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(patients), size=(BOOTSTRAP_REPLICATES, len(patients)))
    bootstrap = patients[sampled].mean(axis=1)
    return {
        "unit_micro": float(values.mean()),
        "patient_equal": float(patients.mean()),
        "patient_cluster_bootstrap_ci95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
    }


def _summary(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
    result: dict[str, object] = {
        "units": len(rows),
        "patients": len({str(row["patient_id"]) for row in rows}),
        "positive_set_size_counts": dict(sorted(Counter(str(row["positive_set_size"]) for row in rows).items())),
    }
    for offset, metric in enumerate(METRICS):
        result[metric] = _metric_summary(rows, metric, seed=seed + offset)
    eligible = [row for row in rows if bool(row["leave_one_eligible"])]
    result["leave_one_documented_positive_out"] = {
        "eligible_units": len(eligible),
        "original_top1_hits": int(sum(float(row["original_set_strict"]) for row in eligible)),
        "fragile_top1_hits": int(sum(bool(row["single_positive_deletion_can_remove_top1_hit"]) for row in eligible)),
        "mean_top1_over_all_deletions": None if not eligible else float(np.mean([float(row["leave_one_mean_top1"]) for row in eligible])),
        "worst_case_top1_over_units": None if not eligible else float(np.mean([float(row["leave_one_worst_top1"]) for row in eligible])),
        "mean_hit_at_3_over_all_deletions": None if not eligible else float(np.mean([float(row["leave_one_mean_hit_at_3"]) for row in eligible])),
        "mean_hit_at_5_over_all_deletions": None if not eligible else float(np.mean([float(row["leave_one_mean_hit_at_5"]) for row in eligible])),
    }
    return result


def _size_strata(rows: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        size = int(row["positive_set_size"])
        key = "1" if size == 1 else "2" if size == 2 else "3-4" if size <= 4 else ">=5"
        groups[key].append(row)
    return {key: _summary(values, seed=seed + 100 * index) for index, (key, values) in enumerate(sorted(groups.items()))}


def run(*, public_directory: Path, public_v16_directory: Path, private_prediction_directory: Path, private_audit_directory: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    public = _public_rows(public_directory, public_v16_directory)
    private = _private_rows(private_prediction_directory, private_audit_directory)
    if sum(float(row["original_set_strict"]) for row in public) != 54 or sum(float(row["original_set_strict"]) for row in private) != 25:
        raise RuntimeError("v67 did not reproduce formal v29 strict endpoints")
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_v29_documented_reference_set_perturbation",
        "primary_reference": "complete_documented_positive_set_remains_formal",
        "public": {
            "role": "posthoc_consumed_development_reference_sensitivity",
            "summary": _summary(public, seed=BOOTSTRAP_SEED),
            "positive_set_size_strata": _size_strata(public, seed=BOOTSTRAP_SEED + 10_000),
        },
        "private": {
            "role": "post_open_transport_reference_sensitivity",
            "summary": _summary(private, seed=BOOTSTRAP_SEED + 20_000),
            "positive_set_size_strata": _size_strata(private, seed=BOOTSTRAP_SEED + 30_000),
        },
        "perturbation_contract": {
            "unlabeled_electrodes_added": False,
            "known_spread_added_to_positive": False,
            "documented_positive_removed_or_isolated_only": True,
            "uniform_documented_singleton_is_assumed_medical_truth": False,
            "leave_one_out_is_assumed_corrected_gold": False,
            "formal_v29_prediction_or_endpoint_changed": False,
        },
        "access_receipt": {
            "frozen_public_and_private_predictions_loaded": True,
            "opened_reference_loaded_for_read_only_sensitivity": True,
            "raw_EEG_foundation_forward_training_or_model_selection_performed": False,
            "private_used_for_adaptation_calibration_or_selection": False,
        },
        "interpretation_boundary": {
            "singleton_uniform_values_are_accuracy_estimates": False,
            "sensitivity_bounds_identify_the_true_positive_electrode": False,
            "spread_is_a_valid_SOZ_positive": False,
            "allowed_claim": "agreement with the unchanged v29 ranking depends on whether the documented multi-positive reference is treated as a complete set or as an unresolved collection of possible singleton references",
        },
        "bootstrap": {"unit": "patient_cluster", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "files": {"unit_table": "reference_set_perturbation_rows.csv"},
    }
    return result, public + private


def publish(output: Path, result: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (staging / "reference_set_perturbation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--public-directory", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-v16-directory", type=Path, default=DEFAULT_PUBLIC_V16)
    parser.add_argument("--private-prediction-directory", type=Path, default=DEFAULT_PRIVATE_PREDICTIONS)
    parser.add_argument("--private-audit-directory", type=Path, default=DEFAULT_PRIVATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, rows = run(public_directory=args.public_directory, public_v16_directory=args.public_v16_directory, private_prediction_directory=args.private_prediction_directory, private_audit_directory=args.private_audit_directory)
    output = publish(args.output, result, rows)
    print(json.dumps({
        "output": str(output),
        "public_set_strict": result["public"]["summary"]["original_set_strict"]["unit_micro"],
        "private_set_strict": result["private"]["summary"]["original_set_strict"]["unit_micro"],
        "private_uniform_documented_singleton": result["private"]["summary"]["documented_singleton_uniform_top1"]["unit_micro"],
        "private_spread_added": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
