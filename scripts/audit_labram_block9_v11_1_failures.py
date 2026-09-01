#!/usr/bin/env python3
"""Aggregate-only failure audit for the public v11.1 block-9 OOF.

This script never reads raw EEG, private data, or patient-level prediction
tables.  It does not fit or select a model.  It deterministically recomputes
predefined aggregate strata from the already-published public OOF tensors.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    patient_localization_metrics,
)


DEFAULT_MAIN = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
DEFAULT_DAPT = ROOT / "outputs/labram_dapt_v2_locked_downstream_oof_20260812"
DEFAULT_OUTPUT = ROOT / "outputs/block9_v11_1_failure_audit_20260812/audit.json"

ARM_KEYS = {
    "full_block9_plus_fine": "oof.full_frozen_labram_plus_fine",
    "H_block9_only": "oof.frozen_labram_only",
    "fine_only": "oof.fine_change_only",
    # Contextual comparator only: it uses a final-suffix carrier and is not a
    # capacity-matched ablation of the block-9 model.
    "DAPT_v2_final_suffix_context_only": "oof.qualified_static_dapt_v2_final_suffix",
}

CANDIDATE_CHANNELS = tuple(channel for channel in STANDARD_19 if channel != "PZ")
LEFT = frozenset(("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"))
RIGHT = frozenset(("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"))
MIDLINE = frozenset(("FZ", "CZ"))
HEMISPHERE = {
    channel: "L" if channel in LEFT else "R" if channel in RIGHT else "M"
    for channel in CANDIDATE_CHANNELS
}

# These are 10-20 scalp electrode-name families, not inferred cortical lobes.
# P7/P8 retain their modern names here; clinical temporal-chain strata below
# separately reflect their legacy T5/T6 use.
TOPOGRAPHIC_LOBES = {
    "frontal": frozenset(("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8")),
    "temporal": frozenset(("T7", "T8")),
    "central": frozenset(("C3", "CZ", "C4")),
    "parietal": frozenset(("P7", "P3", "P4", "P8")),
    "occipital": frozenset(("O1", "O2")),
}

# A disjoint scalp-chain partition used only for descriptive subgrouping.
SCALP_CHAINS = {
    "left_temporal_chain": frozenset(("F7", "T7", "P7")),
    "right_temporal_chain": frozenset(("F8", "T8", "P8")),
    "left_parasagittal": frozenset(("F3", "C3", "P3")),
    "right_parasagittal": frozenset(("F4", "C4", "P4")),
    "midline": frozenset(("FZ", "CZ")),
    "frontopolar": frozenset(("FP1", "FP2")),
    "occipital": frozenset(("O1", "O2")),
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _json_safe(value):
    """Replace undefined numeric cells by JSON null without hiding infinities."""

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            raise ValueError("Audit unexpectedly produced an infinite value")
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _top_ties(logits: torch.Tensor, mask: torch.Tensor, row: int) -> torch.Tensor:
    indices = torch.nonzero(mask[row], as_tuple=False).flatten()
    scores = logits[row, indices]
    return indices[scores == scores.max()]


def _accepted(target: torch.Tensor, mask: torch.Tensor, row: int) -> torch.Tensor:
    positive = target[row].bool() & mask[row]
    accepted = positive.clone()
    if int(positive.sum()) <= 4:
        for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
            accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
    return accepted & mask[row]


def _wrong_hemisphere_value(
    ties: torch.Tensor,
    positive_indices: torch.Tensor,
) -> float:
    target_sides = {
        HEMISPHERE[STANDARD_19[index]]
        for index in positive_indices.tolist()
        if HEMISPHERE[STANDARD_19[index]] in {"L", "R"}
    }
    if target_sides == {"L"}:
        wrong_side = "R"
    elif target_sides == {"R"}:
        wrong_side = "L"
    else:
        return 0.0
    return sum(
        HEMISPHERE[STANDARD_19[index]] == wrong_side for index in ties.tolist()
    ) / int(ties.numel())


def _wrong_hemisphere_far_value(
    ties: torch.Tensor,
    positive_indices: torch.Tensor,
    accepted: torch.Tensor,
) -> float:
    """Top-tie mass that is both contralateral and outside one-hop acceptance."""

    target_sides = {
        HEMISPHERE[STANDARD_19[index]]
        for index in positive_indices.tolist()
        if HEMISPHERE[STANDARD_19[index]] in {"L", "R"}
    }
    if target_sides == {"L"}:
        wrong_side = "R"
    elif target_sides == {"R"}:
        wrong_side = "L"
    else:
        return 0.0
    return sum(
        HEMISPHERE[STANDARD_19[index]] == wrong_side and not bool(accepted[index])
        for index in ties.tolist()
    ) / int(ties.numel())


def _patient_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    prevalence_tier: torch.Tensor,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for patient in range(logits.shape[0]):
        ties = _top_ties(logits, target_mask, patient)
        positive = targets[patient].bool() & target_mask[patient]
        positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
        accepted = _accepted(targets, target_mask, patient)
        strict = float(positive[ties].float().mean())
        relaxed = float(accepted[ties].float().mean())
        metrics = patient_localization_metrics(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
            k_values=(1, 3, 5),
        )
        tier_weights = defaultdict(float)
        for index in ties.tolist():
            tier_weights[str(prevalence_tier[patient, index])] += 1.0 / len(ties)
        rows.append(
            {
                "strict": strict,
                "relaxed": relaxed,
                "one_hop_only": relaxed - strict,
                "far_error": 1.0 - relaxed,
                "wrong_hemisphere": _wrong_hemisphere_value(ties, positive_indices),
                "wrong_hemisphere_far_error": _wrong_hemisphere_far_value(
                    ties, positive_indices, accepted
                ),
                "macro_ap": float(metrics.macro_average_precision),
                "mrr": float(metrics.mean_reciprocal_rank),
                "predicted_low_prevalence": tier_weights["low"],
                "predicted_mid_prevalence": tier_weights["mid"],
                "predicted_high_prevalence": tier_weights["high"],
            }
        )
    return rows


def _mean(rows: list[dict[str, float | str]], indices: list[int], key: str) -> float:
    if not indices:
        return math.nan
    return sum(float(rows[index][key]) for index in indices) / len(indices)


def _summarize_group(
    indices: Iterable[int],
    *,
    arm_rows: dict[str, list[dict[str, float | str]]],
) -> dict:
    selected = list(indices)
    result: dict[str, object] = {"n_patients": len(selected), "arms": {}}
    for arm, rows in arm_rows.items():
        strict = _mean(rows, selected, "strict")
        relaxed = _mean(rows, selected, "relaxed")
        far = _mean(rows, selected, "far_error")
        wrong = _mean(rows, selected, "wrong_hemisphere")
        wrong_far = _mean(rows, selected, "wrong_hemisphere_far_error")
        failure_count = len(selected) * (1.0 - strict) if selected else math.nan
        wrong_count = len(selected) * wrong if selected else math.nan
        result["arms"][arm] = {
            "strict_success": len(selected) * strict if selected else math.nan,
            "strict_accuracy": strict,
            "relaxed_success": len(selected) * relaxed if selected else math.nan,
            "relaxed_accuracy": relaxed,
            "one_hop_only_count": len(selected)
            * _mean(rows, selected, "one_hop_only")
            if selected
            else math.nan,
            "one_hop_only_rate": _mean(rows, selected, "one_hop_only"),
            "far_error_count": len(selected) * far if selected else math.nan,
            "far_error_rate": far,
            "wrong_hemisphere_count": wrong_count,
            "wrong_hemisphere_rate_all": wrong,
            "wrong_hemisphere_fraction_of_strict_failures": (
                wrong_count / failure_count if selected and failure_count > 0 else 0.0
            ),
            "wrong_hemisphere_far_error_count": len(selected) * wrong_far
            if selected
            else math.nan,
            "wrong_hemisphere_fraction_of_far_errors": (
                wrong_far / far if selected and far > 0 else 0.0
            ),
            "same_side_or_midline_far_error_count": len(selected) * (far - wrong_far)
            if selected
            else math.nan,
            "macro_ap": _mean(rows, selected, "macro_ap"),
            "mrr": _mean(rows, selected, "mrr"),
        }
    full = result["arms"]["full_block9_plus_fine"]
    result["strict_accuracy_deltas_vs_full"] = {
        arm: full["strict_accuracy"] - values["strict_accuracy"]
        for arm, values in result["arms"].items()
        if arm != "full_block9_plus_fine"
    }
    return result


def _index_groups(values: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return dict(sorted(groups.items()))


def _summarize_groups(
    groups: dict[str, list[int]],
    *,
    arm_rows: dict[str, list[dict[str, float | str]]],
) -> dict:
    return {
        name: _summarize_group(indices, arm_rows=arm_rows)
        for name, indices in groups.items()
    }


def _binary_failure_association(
    exposed: list[bool],
    full_rows: list[dict[str, float | str]],
) -> dict:
    exposed_indices = [index for index, value in enumerate(exposed) if value]
    other_indices = [index for index, value in enumerate(exposed) if not value]
    exposed_failure = sum(1.0 - float(full_rows[index]["strict"]) for index in exposed_indices)
    other_failure = sum(1.0 - float(full_rows[index]["strict"]) for index in other_indices)
    exposed_rate = exposed_failure / len(exposed_indices) if exposed_indices else math.nan
    other_rate = other_failure / len(other_indices) if other_indices else math.nan
    # Haldane-Anscombe correction is used only to keep a descriptive OR finite.
    a = exposed_failure + 0.5
    b = len(exposed_indices) - exposed_failure + 0.5
    c = other_failure + 0.5
    d = len(other_indices) - other_failure + 0.5
    return {
        "n_exposed": len(exposed_indices),
        "n_unexposed": len(other_indices),
        "failure_count_exposed": exposed_failure,
        "failure_count_unexposed": other_failure,
        "failure_rate_exposed": exposed_rate,
        "failure_rate_unexposed": other_rate,
        "risk_difference_exposed_minus_unexposed": exposed_rate - other_rate,
        "haldane_anscombe_failure_odds_ratio": (a * d) / (b * c),
        "interpretation_scope": "post_hoc_descriptive_not_confirmatory",
    }


def _channel_table(
    *,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    folds: torch.Tensor,
    arm_logits: dict[str, torch.Tensor],
) -> dict:
    table: dict[str, dict] = {}
    for channel_index, channel in enumerate(STANDARD_19):
        if not bool(target_mask[:, channel_index].all()):
            continue
        positive = targets[:, channel_index].bool()
        fold_local_support = []
        for patient in range(targets.shape[0]):
            train = folds != folds[patient]
            fold_local_support.append(int((targets[train, channel_index] == 1).sum()))
        row: dict[str, object] = {
            "gold_patient_support": int(positive.sum()),
            "gold_patient_prevalence": float(positive.float().mean()),
            "outer_train_support_min": min(fold_local_support),
            "outer_train_support_max": max(fold_local_support),
            "hemisphere": HEMISPHERE[channel],
            "topographic_lobe": next(
                name for name, members in TOPOGRAPHIC_LOBES.items() if channel in members
            ),
            "arms": {},
        }
        for arm, logits in arm_logits.items():
            top_mass = torch.zeros(targets.shape[0], dtype=torch.float64)
            for patient in range(targets.shape[0]):
                ties = _top_ties(logits, target_mask, patient)
                top_mass[patient] = float((ties == channel_index).float().mean())
            predicted = float(top_mass.sum())
            true_predicted = float((top_mass * positive.double()).sum())
            row["arms"][arm] = {
                "top1_prediction_count": predicted,
                "channel_recall_at_1": (
                    true_predicted / int(positive.sum()) if bool(positive.any()) else math.nan
                ),
                "reference_precision_at_1": (
                    true_predicted / predicted if predicted > 0 else math.nan
                ),
            }
        table[channel] = row
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--dapt", type=Path, default=DEFAULT_DAPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    main_manifest = _read_json(args.main / "manifest.json")
    dapt_manifest = _read_json(args.dapt / "manifest.json")
    main_tensors = load_file(args.main / "oof_predictions.safetensors", device="cpu")
    dapt_tensors = load_file(args.dapt / "oof_predictions.safetensors", device="cpu")

    patient_ids = [str(value) for value in main_manifest["patient_ids"]]
    if patient_ids != [str(value) for value in dapt_manifest["patient_ids"]]:
        raise ValueError("DAPT and v11.1 patient order differs")
    targets = main_tensors["targets"].float()
    target_mask = main_tensors["target_mask"].bool()
    folds = main_tensors["patient_folds"].long()
    event_counts = main_tensors["patient_event_counts"].long()
    if not torch.equal(targets, dapt_tensors["targets"].float()):
        raise ValueError("DAPT and v11.1 targets differ")
    if not torch.equal(target_mask, dapt_tensors["target_mask"].bool()):
        raise ValueError("DAPT and v11.1 target masks differ")
    if len(patient_ids) != 101 or targets.shape != (101, 19):
        raise ValueError("Audit expects the frozen 101-patient v11.1 cohort")
    if tuple(STANDARD_19) != (
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
        "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
    ):
        raise RuntimeError("Unexpected standard-19 order")

    arm_logits = {
        "full_block9_plus_fine": main_tensors[ARM_KEYS["full_block9_plus_fine"]],
        "H_block9_only": main_tensors[ARM_KEYS["H_block9_only"]],
        "fine_only": main_tensors[ARM_KEYS["fine_only"]],
        "DAPT_v2_final_suffix_context_only": dapt_tensors[
            ARM_KEYS["DAPT_v2_final_suffix_context_only"]
        ],
    }

    # Each held patient receives prevalence ranks computed from the other four
    # folds only.  The fixed 18 candidates are split into bottom/middle/top six.
    prevalence_labels: list[list[str]] = [["excluded"] * 19 for _ in patient_ids]
    tier_rank_by_fold: dict[str, dict[str, object]] = {}
    candidate_indices = [CHANNEL_INDEX[channel] for channel in CANDIDATE_CHANNELS]
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        supports = {
            index: int((targets[train, index] == 1).sum()) for index in candidate_indices
        }
        ordered = sorted(candidate_indices, key=lambda index: (supports[index], index))
        tiers = {
            index: "low" if rank < 6 else "mid" if rank < 12 else "high"
            for rank, index in enumerate(ordered)
        }
        for patient in torch.nonzero(folds == fold, as_tuple=False).flatten().tolist():
            for index in candidate_indices:
                prevalence_labels[patient][index] = tiers[index]
        tier_rank_by_fold[str(fold)] = {
            tier: [
                {"channel": STANDARD_19[index], "outer_train_positive_support": supports[index]}
                for index in ordered
                if tiers[index] == tier
            ]
            for tier in ("low", "mid", "high")
        }

    # String tensors are not portable in torch; rows accept a nested Python list.
    arm_rows = {
        arm: _patient_rows_with_string_tiers(logits, targets, target_mask, prevalence_labels)
        for arm, logits in arm_logits.items()
    }

    positive_counts = targets.bool().sum(dim=1).tolist()
    positive_channels = [
        {
            STANDARD_19[index]
            for index in torch.nonzero(targets[patient].bool(), as_tuple=False).flatten().tolist()
        }
        for patient in range(101)
    ]
    hemisphere_patterns = [
        "+".join(
            side
            for side in ("L", "R", "M")
            if any(HEMISPHERE[channel] == side for channel in channels)
        )
        for channels in positive_channels
    ]
    event_bins = [
        "1" if count == 1 else "2" if count == 2 else "3-5" if count <= 5 else ">=6"
        for count in event_counts.tolist()
    ]
    positive_size_bins = [
        "1" if count == 1 else "2" if count == 2 else "3-4" if count <= 4 else ">=5"
        for count in positive_counts
    ]
    rare_any = []
    rare_all = []
    prevalence_profiles = []
    for patient, channels in enumerate(positive_channels):
        tiers = {
            prevalence_labels[patient][CHANNEL_INDEX[channel]] for channel in channels
        }
        rare_any.append("low" in tiers)
        rare_all.append(tiers == {"low"})
        prevalence_profiles.append("+".join(tier for tier in ("low", "mid", "high") if tier in tiers))

    all_indices = list(range(101))
    strata = {
        "hemisphere_pattern": _summarize_groups(
            _index_groups(hemisphere_patterns), arm_rows=arm_rows
        ),
        "topographic_lobe_present_overlapping": {
            lobe: _summarize_group(
                [index for index, channels in enumerate(positive_channels) if channels & members],
                arm_rows=arm_rows,
            )
            for lobe, members in TOPOGRAPHIC_LOBES.items()
        },
        "scalp_chain_present_overlapping": {
            chain: _summarize_group(
                [index for index, channels in enumerate(positive_channels) if channels & members],
                arm_rows=arm_rows,
            )
            for chain, members in SCALP_CHAINS.items()
        },
        "positive_set_size_exact": _summarize_groups(
            _index_groups([str(value) for value in positive_counts]), arm_rows=arm_rows
        ),
        "positive_set_size_bin": _summarize_groups(
            _index_groups(positive_size_bins), arm_rows=arm_rows
        ),
        "event_count_bin": _summarize_groups(_index_groups(event_bins), arm_rows=arm_rows),
        "outer_fold": _summarize_groups(
            _index_groups([str(value) for value in folds.tolist()]), arm_rows=arm_rows
        ),
        "fold_local_positive_prevalence_profile": _summarize_groups(
            _index_groups(prevalence_profiles), arm_rows=arm_rows
        ),
        "contains_fold_local_bottom_six_positive": {
            "yes": _summarize_group(
                [index for index, value in enumerate(rare_any) if value], arm_rows=arm_rows
            ),
            "no": _summarize_group(
                [index for index, value in enumerate(rare_any) if not value], arm_rows=arm_rows
            ),
        },
    }

    full_rows = arm_rows["full_block9_plus_fine"]
    associations = {
        "single_positive_vs_multiple": _binary_failure_association(
            [value == 1 for value in positive_counts], full_rows
        ),
        "positive_set_at_most_2_vs_at_least_3": _binary_failure_association(
            [value <= 2 for value in positive_counts], full_rows
        ),
        "events_at_most_2_vs_more": _binary_failure_association(
            [value <= 2 for value in event_counts.tolist()], full_rows
        ),
        "events_at_least_6_vs_fewer": _binary_failure_association(
            [value >= 6 for value in event_counts.tolist()], full_rows
        ),
        "contains_fold_local_bottom_six_positive": _binary_failure_association(
            rare_any, full_rows
        ),
        "all_positives_fold_local_bottom_six": _binary_failure_association(
            rare_all, full_rows
        ),
        "single_lateral_hemisphere_target": _binary_failure_association(
            [value in {"L", "R", "L+M", "R+M"} for value in hemisphere_patterns],
            full_rows,
        ),
        "bilateral_lateral_target": _binary_failure_association(
            ["L" in value and "R" in value for value in hemisphere_patterns], full_rows
        ),
    }
    for lobe, members in TOPOGRAPHIC_LOBES.items():
        associations[f"contains_{lobe}_target"] = _binary_failure_association(
            [bool(channels & members) for channels in positive_channels], full_rows
        )

    cross_arm = {}
    full_success = torch.tensor([float(row["strict"]) for row in full_rows])
    for arm, rows in arm_rows.items():
        if arm == "full_block9_plus_fine":
            continue
        success = torch.tensor([float(row["strict"]) for row in rows])
        cross_arm[arm] = {
            "full_win_other_loss": float((full_success * (1.0 - success)).sum()),
            "full_loss_other_win": float(((1.0 - full_success) * success).sum()),
            "both_success": float((full_success * success).sum()),
            "both_failure": float(((1.0 - full_success) * (1.0 - success)).sum()),
            "comparison_scope": (
                "context_only_not_capacity_matched"
                if arm == "DAPT_v2_final_suffix_context_only"
                else "same_v11_1_nested_oof_ablation"
            ),
        }

    error_destination = {}
    for arm, rows in arm_rows.items():
        destination = {tier: {"all_top1_mass": 0.0, "far_error_top1_mass": 0.0} for tier in ("low", "mid", "high")}
        for row in rows:
            for tier in ("low", "mid", "high"):
                mass = float(row[f"predicted_{tier}_prevalence"])
                destination[tier]["all_top1_mass"] += mass
                destination[tier]["far_error_top1_mass"] += mass * float(row["far_error"])
        error_destination[arm] = destination

    consistency_rows = main_manifest["event_to_patient_consistency"]["patients"]
    consistency_by_id = {str(row["patient_id"]): row for row in consistency_rows}
    if set(consistency_by_id) != set(patient_ids):
        raise ValueError("Event-consistency rows do not match the frozen patient cohort")

    def consistency_summary(indices: list[int]) -> dict:
        keys = (
            "agreement_with_patient_top1",
            "modal_share",
            "normalized_vote_entropy",
        )
        return {
            "n_patients": len(indices),
            **{
                f"mean_{key}": sum(
                    float(consistency_by_id[patient_ids[index]][key]) for index in indices
                )
                / len(indices)
                if indices
                else math.nan
                for key in keys
            },
        }

    outcome_indices = {
        "strict_success": [
            index for index, row in enumerate(full_rows) if float(row["strict"]) == 1.0
        ],
        "one_hop_only": [
            index
            for index, row in enumerate(full_rows)
            if float(row["one_hop_only"]) == 1.0
        ],
        "far_error": [
            index for index, row in enumerate(full_rows) if float(row["far_error"]) == 1.0
        ],
        "wrong_hemisphere_far_error": [
            index
            for index, row in enumerate(full_rows)
            if float(row["wrong_hemisphere_far_error"]) == 1.0
        ],
    }

    audit = {
        "schema_version": "soz_block9_v11_1_aggregate_failure_audit_v1",
        "status": "completed_read_only_public_oof_audit",
        "scope": {
            "patient_count": 101,
            "event_count": int(event_counts.sum()),
            "private_data_read": False,
            "raw_eeg_read": False,
            "model_fit_or_training": False,
            "patient_level_error_table_published": False,
            "selection_warning": "All strata are post-hoc descriptive; do not tune per-patient rules from them.",
            "dapt_warning": "DAPT-v2 final-suffix is contextual only, not a capacity-matched block-9 ablation.",
        },
        "definitions": {
            "candidate_channels": list(CANDIDATE_CHANNELS),
            "excluded_carrier_only_channel": "PZ",
            "hemisphere": HEMISPHERE,
            "topographic_lobes_not_cortical_localization": {
                key: sorted(value, key=STANDARD_19.index) for key, value in TOPOGRAPHIC_LOBES.items()
            },
            "scalp_chains_not_cortical_localization": {
                key: sorted(value, key=STANDARD_19.index) for key, value in SCALP_CHAINS.items()
            },
            "one_hop_only": "DeepSOZ neighbor accepted, but no exact positive predicted; expansion only when positive-set size <=4.",
            "wrong_hemisphere": "Opposite lateral side predicted when all lateral positives are unilateral; midline predictions are not counted as contralateral.",
            "far_error": "Neither exact positive nor eligible DeepSOZ one-hop neighbor.",
            "fold_local_prevalence_tier": "Within each outer-train split, fixed 18 channels ranked by positive-patient support; bottom/middle/top six.",
        },
        "overall": _summarize_group(all_indices, arm_rows=arm_rows),
        "cross_arm_strict_discordance": cross_arm,
        "strata": strata,
        "full_block9_failure_associations": associations,
        "channel_prevalence_and_retrieval": _channel_table(
            targets=targets,
            target_mask=target_mask,
            folds=folds,
            arm_logits=arm_logits,
        ),
        "fold_local_prevalence_tiers": tier_rank_by_fold,
        "top1_and_far_error_destination_by_fold_local_prevalence_tier": error_destination,
        "event_prediction_consistency_by_full_block9_outcome": {
            name: consistency_summary(indices) for name, indices in outcome_indices.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(audit), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(args.output)
    print(args.output)
    return 0


def _patient_rows_with_string_tiers(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    prevalence_labels: list[list[str]],
) -> list[dict[str, float | str]]:
    """Variant of _patient_rows using string tiers without tensor coercion."""

    rows: list[dict[str, float | str]] = []
    for patient in range(logits.shape[0]):
        ties = _top_ties(logits, target_mask, patient)
        positive = targets[patient].bool() & target_mask[patient]
        positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
        accepted = _accepted(targets, target_mask, patient)
        strict = float(positive[ties].float().mean())
        relaxed = float(accepted[ties].float().mean())
        metrics = patient_localization_metrics(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
            k_values=(1, 3, 5),
        )
        tier_weights = defaultdict(float)
        for index in ties.tolist():
            tier_weights[prevalence_labels[patient][index]] += 1.0 / len(ties)
        rows.append(
            {
                "strict": strict,
                "relaxed": relaxed,
                "one_hop_only": relaxed - strict,
                "far_error": 1.0 - relaxed,
                "wrong_hemisphere": _wrong_hemisphere_value(ties, positive_indices),
                "wrong_hemisphere_far_error": _wrong_hemisphere_far_value(
                    ties, positive_indices, accepted
                ),
                "macro_ap": float(metrics.macro_average_precision),
                "mrr": float(metrics.mean_reciprocal_rank),
                "predicted_low_prevalence": tier_weights["low"],
                "predicted_mid_prevalence": tier_weights["mid"],
                "predicted_high_prevalence": tier_weights["high"],
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
