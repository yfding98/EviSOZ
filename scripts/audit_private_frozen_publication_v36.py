#!/usr/bin/env python3
"""Publication-grade, no-tuning audit of frozen v29 private transfer results.

The private labels were opened in earlier project iterations.  This script
therefore performs descriptive evaluation only: it does not fit, select,
calibrate, threshold, route, or rewrite reports.  Events remain clustered by
patient and are never presented as independent clinical validation units.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_trustworthy_soz_ranking_distance_v22_6 import (  # noqa: E402
    _json_list,
    rank_row,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_private_frozen_publication_audit_v36"
DEFAULT_PREDICTION = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_TARGET = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816

LEFT = frozenset(("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"))
RIGHT = frozenset(("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"))
MIDLINE = frozenset(("FZ", "CZ", "PZ"))
METRICS = (
    "strict",
    "neighbor_only",
    "relaxed",
    "far",
    "contralateral_far",
    "known_spread_top1",
    "hit_at_3",
    "hit_at_5",
    "positive_recall_at_3",
    "positive_recall_at_5",
    "mrr",
    "average_precision",
    "laterality_agreement",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _reference_laterality(channels: Sequence[str]) -> str:
    values = set(channels)
    if values and values.issubset(LEFT):
        return "left_only"
    if values and values.issubset(RIGHT):
        return "right_only"
    if values and values.issubset(MIDLINE):
        return "midline_only"
    return "bilateral_or_mixed"


def _event_rows(
    *,
    scores: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tuple(scores.shape) != (len(events), 19):
        raise ValueError("Private scores/event roster shape mismatch")
    event_index = {str(event["event_id"]): index for index, event in enumerate(events)}
    if len(event_index) != len(events):
        raise ValueError("Private prediction event IDs are not unique")
    target_preeligible = [row for row in target_rows if row["primary_reference_preeligible"] == "1"]
    selected = [row for row in target_preeligible if row["event_id"] in event_index]
    if len(target_preeligible) != 52 or len({row["patient_id"] for row in target_preeligible}) != 24:
        raise ValueError("Private target-preeligible flow changed")
    if len(selected) != 51 or len({row["patient_id"] for row in selected}) != 23:
        raise ValueError("Private primary prediction/reference intersection changed")
    evaluable = {index for index, channel in enumerate(STANDARD_19) if channel != "PZ"}
    rows: list[dict[str, Any]] = []
    for target in selected:
        event = events[event_index[target["event_id"]]]
        positives = {
            CHANNEL_INDEX[channel]
            for channel in _json_list(
                target["candidate_positive_electrodes"], name="private positives"
            )
            if channel in CHANNEL_INDEX and channel != "PZ"
        }
        spread = {
            CHANNEL_INDEX[channel]
            for channel in _json_list(
                target["known_spread_electrodes"], name="private known spread"
            )
            if channel in CHANNEL_INDEX and channel != "PZ"
        }
        ranked = rank_row(
            unit_id=target["event_id"],
            patient_id=target["patient_id"],
            scores=scores[event_index[target["event_id"]]],
            positive_indices=positives,
            evaluable_indices=evaluable,
            spread_indices=spread,
        )
        ranked.update(
            {
                "source_sfreq_hz": float(event["source_sfreq_hz"]),
                "positive_set_size": len(positives),
                "known_spread_set_size": len(spread),
                "reference_laterality_stratum": _reference_laterality(
                    ranked["positive_channels"]
                ),
                "positive_set_exhaustiveness": target["positive_set_exhaustiveness"],
            }
        )
        rows.append(ranked)
    missing = sorted(set(row["event_id"] for row in target_preeligible) - set(event_index))
    flow = {
        "target_ledger_rows": len(target_rows),
        "target_preeligible_events": len(target_preeligible),
        "target_preeligible_patients": len({row["patient_id"] for row in target_preeligible}),
        "target_blind_prediction_events": len(events),
        "target_blind_prediction_patients": len({str(row["patient_id"]) for row in events}),
        "primary_intersection_events": len(rows),
        "primary_intersection_patients": len({row["patient_id"] for row in rows}),
        "preeligible_without_prediction_count": len(missing),
        "preeligible_without_prediction_reason": "target_blind_signal_or_anchor_ineligible",
    }
    return rows, flow


def _patient_values(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            grouped[str(row["patient_id"])].append(value)
    return {patient: float(np.mean(values)) for patient, values in grouped.items() if values}


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]], *, metric: str, seed: int
) -> dict[str, list[float] | None]:
    by_patient: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_patient[str(row["patient_id"])].append(value)
    patients = sorted(by_patient)
    if len(patients) < 2:
        return {"patient_equal_ci95": None, "cluster_event_micro_ci95": None}
    rng = np.random.default_rng(seed)
    patient_means = np.asarray(
        [np.mean(by_patient[patient]) for patient in patients], dtype=np.float64
    )
    patient_sums = np.asarray(
        [np.sum(by_patient[patient]) for patient in patients], dtype=np.float64
    )
    patient_counts = np.asarray(
        [len(by_patient[patient]) for patient in patients], dtype=np.float64
    )
    sampled = rng.integers(
        0, len(patients), size=(BOOTSTRAP_REPLICATES, len(patients))
    )
    patient_equal = patient_means[sampled].mean(axis=1)
    event_micro = patient_sums[sampled].sum(axis=1) / patient_counts[sampled].sum(axis=1)
    return {
        "patient_equal_ci95": [float(value) for value in np.quantile(patient_equal, [0.025, 0.975])],
        "cluster_event_micro_ci95": [float(value) for value in np.quantile(event_micro, [0.025, 0.975])],
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    event_micro: dict[str, float | None] = {}
    patient_equal: dict[str, float | None] = {}
    intervals: dict[str, Any] = {}
    for offset, metric in enumerate(METRICS):
        values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
        patient = _patient_values(rows, metric)
        event_micro[metric] = None if not values else float(np.mean(values))
        patient_equal[metric] = None if not patient else float(np.mean(list(patient.values())))
        intervals[metric] = _bootstrap_intervals(rows, metric=metric, seed=seed + offset)
    categories = Counter(
        "strict_exact" if row["strict"] else "neighbor_only" if row["neighbor_only"] else "far"
        for row in rows
    )
    far_subtypes = Counter(str(row["far_subtype"]) for row in rows if row["far"])
    ranks = [int(row["first_positive_rank"]) for row in rows]
    spread_positive_rows = [row for row in rows if int(row["known_spread_set_size"]) > 0]
    return {
        "event_count": len(rows),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "event_micro": event_micro,
        "patient_equal_event_macro": patient_equal,
        "patient_cluster_bootstrap_ci95": intervals,
        "endpoint_counts": {
            "strict_exact": categories.get("strict_exact", 0),
            "neighbor_only": categories.get("neighbor_only", 0),
            "far": categories.get("far", 0),
            "contralateral_far": sum(int(row["contralateral_far"]) for row in rows),
            "known_spread_top1_all_enrolled": sum(int(row["known_spread_top1"]) for row in rows),
            "known_spread_positive_event_denominator": len(spread_positive_rows),
            "known_spread_top1_on_positive_spread_denominator": sum(
                int(row["known_spread_top1"]) for row in spread_positive_rows
            ),
        },
        "far_subtype_counts": dict(sorted(far_subtypes.items())),
        "candidate_burden": {
            "mean_first_positive_rank": float(np.mean(ranks)),
            "median_first_positive_rank": float(np.median(ranks)),
            "first_positive_rank_counts": dict(
                sorted(Counter(str(rank) if rank <= 5 else ">5" for rank in ranks).items())
            ),
        },
    }


def _paired(
    proposed: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    proposed_by_id = {str(row["unit_id"]): row for row in proposed}
    comparator_by_id = {str(row["unit_id"]): row for row in comparator}
    if set(proposed_by_id) != set(comparator_by_id):
        raise ValueError("Paired private arm rosters differ")
    result: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for metric in METRICS:
        patient_delta: dict[str, list[float]] = defaultdict(list)
        proposed_only = 0
        comparator_only = 0
        for unit_id in sorted(proposed_by_id):
            row = proposed_by_id[unit_id]
            other = comparator_by_id[unit_id]
            left = float(row[metric])
            right = float(other[metric])
            if not math.isfinite(left) or not math.isfinite(right):
                continue
            patient_delta[str(row["patient_id"])].append(left - right)
            if metric in {"strict", "relaxed", "hit_at_3", "hit_at_5"}:
                proposed_only += int(left == 1.0 and right == 0.0)
                comparator_only += int(left == 0.0 and right == 1.0)
        patient_means = np.asarray(
            [np.mean(patient_delta[patient]) for patient in sorted(patient_delta)],
            dtype=np.float64,
        )
        sampled = rng.integers(
            0,
            len(patient_means),
            size=(BOOTSTRAP_REPLICATES, len(patient_means)),
        )
        bootstrap = patient_means[sampled].mean(axis=1)
        result[metric] = {
            "patient_equal_difference": float(patient_means.mean()),
            "patient_cluster_bootstrap_ci95": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "event_discordance": (
                None
                if metric not in {"strict", "relaxed", "hit_at_3", "hit_at_5"}
                else {
                    "proposed_only_success": proposed_only,
                    "comparator_only_success": comparator_only,
                }
            ),
        }
    return result


def _stratified(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    definitions = {
        "reference_laterality": lambda row: str(row["reference_laterality_stratum"]),
        "positive_set_size": lambda row: (
            "1-2" if int(row["positive_set_size"]) <= 2 else "3-4" if int(row["positive_set_size"]) <= 4 else ">=5"
        ),
        "source_sampling_rate_hz": lambda row: str(int(float(row["source_sfreq_hz"]))),
        "known_spread_annotation": lambda row: (
            "nonempty_known_spread" if int(row["known_spread_set_size"]) > 0 else "empty_known_spread_field"
        ),
    }
    output: dict[str, Any] = {}
    for stratum_index, (name, key_fn) in enumerate(definitions.items()):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row)
        output[name] = {
            key: _summary(values, seed=seed + 1000 * (stratum_index + 1) + index * 100)
            for index, (key, values) in enumerate(sorted(grouped.items()))
        }
    return output


def _within_patient(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(row)
    summaries = []
    for patient, values in sorted(grouped.items()):
        top1 = [str(row["top1"]) for row in values]
        reference_sets = [set(row["positive_channels"]) for row in values]
        pairwise_jaccard = []
        for left in range(len(reference_sets)):
            for right in range(left + 1, len(reference_sets)):
                union = reference_sets[left] | reference_sets[right]
                pairwise_jaccard.append(len(reference_sets[left] & reference_sets[right]) / len(union))
        summaries.append(
            {
                "patient_id": patient,
                "event_count": len(values),
                "unique_top1_count": len(set(top1)),
                "top1_modal_share": max(Counter(top1).values()) / len(top1),
                "reference_sets_all_identical": len({tuple(sorted(value)) for value in reference_sets}) == 1,
                "mean_pairwise_reference_jaccard": (
                    None if not pairwise_jaccard else float(np.mean(pairwise_jaccard))
                ),
            }
        )
    return {
        "patient_count": len(summaries),
        "events_per_patient": {
            "minimum": min(row["event_count"] for row in summaries),
            "median": float(np.median([row["event_count"] for row in summaries])),
            "maximum": max(row["event_count"] for row in summaries),
        },
        "mean_top1_modal_share": float(np.mean([row["top1_modal_share"] for row in summaries])),
        "patients_with_identical_event_references": sum(
            bool(row["reference_sets_all_identical"]) for row in summaries
        ),
        "patient_rows": summaries,
    }


def audit(
    *, prediction_directory: Path, target_path: Path, public_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = _read_json(prediction_directory / "manifest.json")
    access = manifest.get("access_receipt", {})
    if manifest.get("status") != "completed_frozen_target_blind_private_inference":
        raise ValueError("Private v29 predictions are not frozen")
    if access.get("private_target_values_loaded") is not False or access.get(
        "training_calibration_or_model_selection_performed"
    ) is not False:
        raise ValueError("Private v29 prediction firewall failed")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("Private v29 signal roster changed")
    tensors = load_file(str((prediction_directory / str(manifest["tensor_file"])).resolve(strict=True)))
    candidate_mask = tensors["candidate_mask"].bool()
    expected_mask = torch.tensor([channel != "PZ" for channel in STANDARD_19])
    if not torch.equal(candidate_mask, expected_mask):
        raise ValueError("Private v29 candidate mask changed")
    target_rows = _read_csv(target_path)
    arm_scores = {
        "v29_equal_H_D": tensors["private_portable_equal_probability"].float(),
        "H_fold_mean": tensors["private_h_only_fold_probability"].float().mean(dim=1),
        "D_fold_mean": tensors["private_rank1_direct_fold_probability"].float().mean(dim=1),
    }
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    flow = None
    for arm, scores in arm_scores.items():
        rows, current_flow = _event_rows(scores=scores, events=events, target_rows=target_rows)
        rows_by_arm[arm] = rows
        if flow is None:
            flow = current_flow
        elif flow != current_flow:
            raise RuntimeError("Private cohort flow changed between frozen arms")
    assert flow is not None
    summaries = {
        arm: _summary(rows, seed=BOOTSTRAP_SEED + arm_index * 100)
        for arm_index, (arm, rows) in enumerate(rows_by_arm.items())
    }
    proposed = rows_by_arm["v29_equal_H_D"]
    public = _read_json(public_manifest_path)
    public_metrics = public["metrics"]
    result = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_post_open_private_publication_audit",
        "analysis_role": "descriptive_cross_domain_cross_granularity_transport",
        "primary_private_unit": "seizure_event_with_patient_clustered_uncertainty",
        "cohort_flow": flow,
        "frozen_arms": summaries,
        "paired_proposed_minus_H": _paired(
            proposed, rows_by_arm["H_fold_mean"], seed=BOOTSTRAP_SEED + 10_000
        ),
        "paired_proposed_minus_D": _paired(
            proposed, rows_by_arm["D_fold_mean"], seed=BOOTSTRAP_SEED + 20_000
        ),
        "private_proposed_strata": _stratified(proposed, seed=BOOTSTRAP_SEED + 30_000),
        "private_within_patient": _within_patient(proposed),
        "distribution": {
            "reference_positive_set_size": dict(
                sorted(Counter(str(row["positive_set_size"]) for row in proposed).items())
            ),
            "known_spread_set_size": dict(
                sorted(Counter(str(row["known_spread_set_size"]) for row in proposed).items())
            ),
            "reference_channel_occurrence": dict(
                sorted(Counter(channel for row in proposed for channel in row["positive_channels"]).items())
            ),
            "predicted_top1_channel": dict(sorted(Counter(row["top1"] for row in proposed).items())),
            "source_sampling_rate_hz": dict(
                sorted(Counter(str(int(row["source_sfreq_hz"])) for row in proposed).items())
            ),
        },
        "same_model_public_development_context": {
            "v29_equal_H_D": public_metrics["portable_equal_ensemble"],
            "H_only": public_metrics["h_only"],
            "public_unit": "patient_level_clinical_note_reference",
            "direct_public_private_superiority_test_authorized": False,
        },
        "headline_audit": {
            "private_event_micro_strict": summaries["v29_equal_H_D"]["event_micro"]["strict"],
            "private_event_micro_neighborhood4": summaries["v29_equal_H_D"]["event_micro"]["relaxed"],
            "private_patient_equal_event_macro_strict": summaries["v29_equal_H_D"]["patient_equal_event_macro"]["strict"],
            "private_patient_equal_event_macro_neighborhood4": summaries["v29_equal_H_D"]["patient_equal_event_macro"]["relaxed"],
            "private_patient_equal_laterality": summaries["v29_equal_H_D"]["patient_equal_event_macro"]["laterality_agreement"],
            "difference_from_deepsoz_paper_neighborhood_point_0_744": summaries["v29_equal_H_D"]["event_micro"]["relaxed"] - 0.744,
            "confidence_interval_supports_exceeding_0_744": summaries["v29_equal_H_D"]["patient_cluster_bootstrap_ci95"]["relaxed"]["patient_equal_ci95"][0] > 0.744,
        },
        "reference_semantics": {
            "positive": "event-level clinician significant-electrode set integrating full seizure EEG and semiology",
            "known_spread": "separate observed spread field; removed from neighborhood acceptance",
            "unmentioned_electrode": "unknown_not_reviewed_not_clinically_confirmed_negative",
            "positive_set_exhaustiveness": "positive_only_unknown_complement",
            "patient_consensus_reference_available": False,
        },
        "access_receipt": {
            "existing_frozen_prediction_loaded": True,
            "opened_private_reference_loaded_for_descriptive_evaluation": True,
            "raw_eeg_loaded": False,
            "training_performed": False,
            "model_or_fusion_weight_selected": False,
            "threshold_or_abstention_selected": False,
            "report_text_changed": False,
            "llm_used": False,
        },
        "claim_boundary": {
            "private_used_for_v29_training": False,
            "private_predictions_frozen_before_v29_target_read": True,
            "private_has_been_opened_in_prior_project_iterations": True,
            "fresh_external_confirmation": False,
            "event_rows_independent": False,
            "neighborhood4_is_strict_electrode_accuracy": False,
            "output_is_cortical_soz_or_surgical_target": False,
            "post_open_paired_differences_are_confirmatory": False,
        },
    }
    return result, rows_by_arm


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def publish(
    *, output: Path, result: Mapping[str, Any], rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    proposed = rows_by_arm["v29_equal_H_D"]
    _write_csv(
        output / "private_event_error_audit.csv",
        proposed,
        (
            "unit_id",
            "patient_id",
            "top1",
            "positive_channels",
            "known_spread_channels",
            "strict",
            "neighbor_only",
            "far",
            "far_subtype",
            "known_spread_top1",
            "first_positive_rank",
            "reference_laterality_stratum",
            "source_sfreq_hz",
        ),
    )
    summary_rows = []
    for arm, summary in result["frozen_arms"].items():
        summary_rows.append(
            {
                "arm": arm,
                "events": summary["event_count"],
                "patients": summary["patient_count"],
                "strict_event_micro": summary["event_micro"]["strict"],
                "strict_patient_equal": summary["patient_equal_event_macro"]["strict"],
                "neighborhood4_event_micro": summary["event_micro"]["relaxed"],
                "neighborhood4_patient_equal": summary["patient_equal_event_macro"]["relaxed"],
                "hit_at_3_event_micro": summary["event_micro"]["hit_at_3"],
                "hit_at_5_event_micro": summary["event_micro"]["hit_at_5"],
                "laterality_patient_equal": summary["patient_equal_event_macro"]["laterality_agreement"],
                "far_count": summary["endpoint_counts"]["far"],
                "contralateral_far_count": summary["endpoint_counts"]["contralateral_far"],
                "known_spread_top1_count": summary["endpoint_counts"]["known_spread_top1_all_enrolled"],
            }
        )
    _write_csv(output / "private_frozen_arm_table.csv", summary_rows, tuple(summary_rows[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--public-manifest", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result, rows = audit(
        prediction_directory=args.prediction,
        target_path=args.target,
        public_manifest_path=args.public_manifest,
    )
    publish(output=args.output, result=result, rows_by_arm=rows)
    print(json.dumps({"output": str(args.output), **result["headline_audit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
