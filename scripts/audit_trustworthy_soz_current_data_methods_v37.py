#!/usr/bin/env python3
"""Audit frozen public SOZ predictions without fitting or selecting a model.

This script deliberately operates on already frozen patient-level predictions.
It does not load raw EEG, refit a head, choose an ensemble weight, or tune a
display threshold.  Its purpose is to put the v29 ranker, its two carriers,
the fold-local prevalence prior, and the locally replayed DeepSOZ model on one
patient roster and to quantify paired differences and carrier-replacement
stress tests for a method-audit paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


SCHEMA = "trustworthy_soz_current_data_method_audit_v37"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_V16 = ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
DEFAULT_DEEPSOZ = ROOT / "outputs/deepsoz_official_local_oof_full.json"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_current_data_method_audit_v37_20260816"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _masked_probability(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    if not torch.isfinite(probability).all():
        raise ValueError("masked probability is non-finite")
    return probability


def _probability_logits(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probability.shape != mask.shape or not torch.isfinite(probability).all():
        raise ValueError("probability/mask carrier is invalid")
    if bool((probability < 0).any()):
        raise ValueError("probability carrier contains a negative value")
    normalized = probability.masked_fill(~mask, 0.0)
    normalized = normalized / normalized.sum(dim=1, keepdim=True)
    return torch.log(normalized.clamp_min(1e-12))


def _metric_row(name: str, result: Mapping[str, object]) -> dict[str, object]:
    ranking = result["ranking"]
    top1 = result["top1"]
    assert isinstance(ranking, Mapping) and isinstance(top1, Mapping)
    hit_at_k = ranking["hit_at_k"]
    assert isinstance(hit_at_k, Mapping)
    return {
        "arm": name,
        "strict_top1": float(top1["strict_accuracy"]),
        "neighborhood4_top1": float(top1["relaxed_accuracy"]),
        "neighbor_only_gain": float(top1["neighbor_only_accuracy_gain"]),
        "macro_average_precision": float(ranking["macro_average_precision"]),
        "mean_reciprocal_rank": float(ranking["mean_reciprocal_rank"]),
        "hit_at_3": float(hit_at_k[3]),
        "hit_at_5": float(hit_at_k[5]),
        "far_error_count": float(result["far_error_count"]),
        "reference_membership_brier": float(ranking["reference_membership_brier"]),
        "reference_membership_nll": float(ranking["reference_membership_nll"]),
    }


def _topk(probability: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(probability.masked_fill(~mask, -torch.inf), k=k, dim=1).indices


def _topk_jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("top-k tensors must have equal shape [P,K]")
    rows = []
    for lhs, rhs in zip(left.tolist(), right.tolist()):
        lhs_set = set(lhs)
        rhs_set = set(rhs)
        rows.append(len(lhs_set & rhs_set) / len(lhs_set | rhs_set))
    return float(sum(rows) / len(rows))


def _replacement_diagnostics(
    original: torch.Tensor,
    replacement: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    original_top1 = _topk(original, mask, 1).squeeze(1)
    replacement_top1 = _topk(replacement, mask, 1).squeeze(1)
    original_top1_probability = original.gather(1, original_top1[:, None]).squeeze(1)
    replacement_at_original_top1 = replacement.gather(
        1, original_top1[:, None]
    ).squeeze(1)
    return {
        "top1_retention": float((original_top1 == replacement_top1).float().mean()),
        "top3_jaccard": _topk_jaccard(
            _topk(original, mask, 3), _topk(replacement, mask, 3)
        ),
        "mean_absolute_probability_shift": float(
            (original - replacement).abs()[mask].mean()
        ),
        "mean_original_top1_probability_drop": float(
            (original_top1_probability - replacement_at_original_top1).mean()
        ),
    }


def _prediction_summary(probability: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    top1 = _topk(probability, mask, 1).squeeze(1)
    ordered = torch.topk(probability.masked_fill(~mask, -torch.inf), k=2, dim=1).values
    entropy = -(probability.clamp_min(1e-12).log() * probability).sum(dim=1)
    distribution: dict[str, int] = {}
    for index in top1.tolist():
        channel = STANDARD_19[index]
        distribution[channel] = distribution.get(channel, 0) + 1
    return {
        "mean_entropy_nats": float(entropy.mean()),
        "mean_top1_margin": float((ordered[:, 0] - ordered[:, 1]).mean()),
        "top1_channel_counts": dict(sorted(distribution.items())),
    }


def _load_deepsoz(
    path: Path,
    patient_ids: tuple[str, ...],
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    channels = tuple(payload["preprocessing"]["channels"])
    if channels != tuple(STANDARD_19):
        raise ValueError("DeepSOZ channel order differs from canonical standard-19")
    by_patient = {
        str(row["patient_id"]): row for row in payload["held_out_ensemble_predictions"]
    }
    if set(by_patient) != set(patient_ids):
        raise ValueError("DeepSOZ and v29 patient rosters differ")
    scores = []
    for patient_index, patient_id in enumerate(patient_ids):
        row = by_patient[patient_id]
        score = torch.tensor(row["score"], dtype=torch.float32)
        if tuple(score.shape) != (19,) or not torch.isfinite(score).all():
            raise ValueError(f"invalid DeepSOZ score for patient {patient_id}")
        declared = set(row["positive_channels"])
        observed = {
            STANDARD_19[index]
            for index in torch.nonzero(
                (targets[patient_index] == 1) & mask[patient_index], as_tuple=False
            ).flatten().tolist()
        }
        if declared - {"PZ"} != observed:
            raise ValueError(f"DeepSOZ target mismatch for patient {patient_id}")
        scores.append(score.clamp_min(0.0))
    score_tensor = torch.stack(scores)
    score_tensor = score_tensor.masked_fill(~mask, 0.0)
    probability = score_tensor / score_tensor.sum(dim=1, keepdim=True)
    declared = payload["held_out_ensemble_metrics"]
    return probability, {
        "source_status": payload["status"],
        "declared_exact": float(declared["exact"]),
        "declared_neighborhood4": float(declared["neighborhood4"]),
        "patient_count": int(payload["patient_count"]),
        "score_conversion": "nonnegative_score_normalized_over_C18_for_ranking_replay",
        "calibration_scores_comparable_to_v29": False,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    v29_manifest_path = (args.v29 / "manifest.json").resolve(strict=True)
    v29_tensor_path = (args.v29 / "oof_predictions.safetensors").resolve(strict=True)
    v16_manifest_path = (args.v16 / "manifest.json").resolve(strict=True)
    v16_tensor_path = (args.v16 / "oof_predictions.safetensors").resolve(strict=True)
    deepsoz_path = args.deepsoz.resolve(strict=True)

    v29_manifest = json.loads(v29_manifest_path.read_text(encoding="utf-8"))
    v16_manifest = json.loads(v16_manifest_path.read_text(encoding="utf-8"))
    v29 = load_file(str(v29_tensor_path), device="cpu")
    v16 = load_file(str(v16_tensor_path), device="cpu")

    patient_ids = tuple(str(value) for value in v16_manifest["patient_ids"])
    if len(patient_ids) != 102 or len(set(patient_ids)) != 102:
        raise ValueError("v37 requires the frozen 102-patient public roster")
    for name in ("targets", "target_mask", "patient_folds"):
        if not torch.equal(v29[name], v16[name]):
            raise ValueError(f"v29/v16 carrier mismatch: {name}")
    targets = v29["targets"].float()
    mask = v29["target_mask"].bool()
    if tuple(targets.shape) != (102, 19) or not bool((mask == mask[0]).all()):
        raise ValueError("v37 requires a fixed C18 target carrier")

    h = v29["oof.h_only_probability"].float()
    d = v29["oof.rank1_direct_probability"].float()
    v17 = v29["oof.v17_probability"].float()
    proposed = v29["oof.portable_equal_ensemble_probability"].float()
    prevalence = _masked_probability(v16["oof.prevalence_only"].float(), mask)
    deepsoz, deepsoz_receipt = _load_deepsoz(
        deepsoz_path, patient_ids, targets, mask
    )
    expected = 0.5 * h + 0.5 * d
    if not torch.allclose(proposed, expected, atol=1e-7, rtol=0.0):
        raise ValueError("v29 is not the declared equal H/D probability ensemble")

    arms_probability = {
        "v29_equal_H_D": proposed,
        "H_only": h,
        "D_only": d,
        "v17_H_plus_fine": v17,
        "fold_local_prevalence_only": prevalence,
        "DeepSOZ_local_replay": deepsoz,
        "replace_D_with_prevalence": 0.5 * h + 0.5 * prevalence,
        "replace_H_with_prevalence": 0.5 * prevalence + 0.5 * d,
    }
    metrics: dict[str, dict[str, object]] = {}
    arm_rows: list[dict[str, object]] = []
    for name, probability in arms_probability.items():
        result = _evaluate(_probability_logits(probability, mask), targets, mask)
        metrics[name] = result
        arm_rows.append(_metric_row(name, result))

    declared = v29_manifest["metrics"]["portable_equal_ensemble"]
    replay = metrics["v29_equal_H_D"]
    for left, right in (
        (replay["top1"]["strict_accuracy"], declared["top1"]["strict_accuracy"]),
        (replay["top1"]["relaxed_accuracy"], declared["top1"]["relaxed_accuracy"]),
        (
            replay["ranking"]["macro_average_precision"],
            declared["ranking"]["macro_average_precision"],
        ),
    ):
        if not math.isclose(float(left), float(right), abs_tol=1e-7, rel_tol=0.0):
            raise ValueError("v29 metric replay differs from frozen manifest")
    if not math.isclose(
        float(metrics["DeepSOZ_local_replay"]["top1"]["strict_accuracy"]),
        deepsoz_receipt["declared_exact"],
        abs_tol=1e-7,
        rel_tol=0.0,
    ) or not math.isclose(
        float(metrics["DeepSOZ_local_replay"]["top1"]["relaxed_accuracy"]),
        deepsoz_receipt["declared_neighborhood4"],
        abs_tol=1e-7,
        rel_tol=0.0,
    ):
        raise ValueError("DeepSOZ ranking replay differs from frozen local audit")

    comparator_names = (
        "H_only",
        "D_only",
        "v17_H_plus_fine",
        "fold_local_prevalence_only",
        "DeepSOZ_local_replay",
        "replace_D_with_prevalence",
        "replace_H_with_prevalence",
    )
    paired: dict[str, object] = {}
    paired_rows: list[dict[str, object]] = []
    proposed_logits = _probability_logits(proposed, mask)
    for comparator in comparator_names:
        comparison = _paired_bootstrap(
            proposed_logits,
            _probability_logits(arms_probability[comparator], mask),
            targets,
            mask,
        )
        paired[comparator] = comparison
        for endpoint, values in comparison.items():
            paired_rows.append(
                {
                    "candidate": "v29_equal_H_D",
                    "comparator": comparator,
                    "endpoint": endpoint,
                    "difference": float(values["delta"]),
                    "ci95_low": float(values["ci95"][0]),
                    "ci95_high": float(values["ci95"][1]),
                }
            )

    h_top1 = _topk(h, mask, 1).squeeze(1)
    d_top1 = _topk(d, mask, 1).squeeze(1)
    proposed_top1 = _topk(proposed, mask, 1).squeeze(1)
    carrier_agreement = {
        "H_vs_D_top1": float((h_top1 == d_top1).float().mean()),
        "H_vs_v29_top1": float((h_top1 == proposed_top1).float().mean()),
        "D_vs_v29_top1": float((d_top1 == proposed_top1).float().mean()),
        "H_vs_D_top3_jaccard": _topk_jaccard(_topk(h, mask, 3), _topk(d, mask, 3)),
    }

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_posthoc_development_method_audit",
        "analysis_role": "no_fit_no_selection_current_data_method_audit",
        "public_patient_count": 102,
        "metrics": metrics,
        "paired_v29_minus_comparator_patient_bootstrap": paired,
        "carrier_agreement": carrier_agreement,
        "prediction_summaries": {
            name: _prediction_summary(probability, mask)
            for name, probability in arms_probability.items()
        },
        "carrier_replacement_stress": {
            "replace_D_with_prevalence": _replacement_diagnostics(
                proposed, arms_probability["replace_D_with_prevalence"], mask
            ),
            "replace_H_with_prevalence": _replacement_diagnostics(
                proposed, arms_probability["replace_H_with_prevalence"], mask
            ),
            "replace_both_with_prevalence": _replacement_diagnostics(
                proposed, prevalence, mask
            ),
        },
        "deepsoz_receipt": deepsoz_receipt,
        "source_files": {
            "v29_manifest": str(v29_manifest_path.relative_to(ROOT)),
            "v29_manifest_sha256": _sha256(v29_manifest_path),
            "v29_tensor": str(v29_tensor_path.relative_to(ROOT)),
            "v29_tensor_sha256": _sha256(v29_tensor_path),
            "v16_manifest": str(v16_manifest_path.relative_to(ROOT)),
            "v16_manifest_sha256": _sha256(v16_manifest_path),
            "v16_tensor": str(v16_tensor_path.relative_to(ROOT)),
            "v16_tensor_sha256": _sha256(v16_tensor_path),
            "deepsoz_local_replay": str(deepsoz_path.relative_to(ROOT)),
            "deepsoz_local_replay_sha256": _sha256(deepsoz_path),
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "model_training_performed": False,
            "foundation_forward_performed": False,
            "ensemble_weight_selected": False,
            "threshold_or_abstention_selected": False,
            "private_eeg_or_target_loaded": False,
            "existing_public_targets_loaded_for_evaluation": True,
        },
        "interpretation_boundary": {
            "all_102_public_patients_consumed_development": True,
            "paired_intervals_confirmatory": False,
            "prevalence_only_is_signal_free": True,
            "carrier_replacement_is_raw_signal_destruction": False,
            "carrier_replacement_interpretation": (
                "posthoc branch-level reliance audit using an existing fold-local "
                "signal-free prior; it is not channel-time causal attribution"
            ),
            "neighborhood4_is_strict_electrode_accuracy": False,
            "deepsoz_calibration_compared": False,
            "model_or_claim_selected_from_this_audit": False,
        },
        "system_identity_audit": {
            "development_ranker": "v29_equal_H_D",
            "existing_v34_report_candidate_profile": "v21_H_only",
            "profiles_end_to_end_identical": False,
            "required_action": (
                "generate a v29-bound deterministic research report or keep the "
                "v21 language experiment explicitly decoupled from v29 accuracy"
            ),
        },
    }
    return result, arm_rows, paired_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(
    output: Path,
    result: Mapping[str, object],
    arm_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
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
        _write_csv(staging / "public_arm_table.csv", arm_rows)
        _write_csv(staging / "paired_v29_comparisons.csv", paired_rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v29", type=Path, default=DEFAULT_V29)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--deepsoz", type=Path, default=DEFAULT_DEEPSOZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, arm_rows, paired_rows = run(args)
    output = publish(args.output, result, arm_rows, paired_rows)
    proposed = result["metrics"]["v29_equal_H_D"]
    prevalence = result["metrics"]["fold_local_prevalence_only"]
    print(
        json.dumps(
            {
                "output": str(output),
                "v29_strict": proposed["top1"]["strict_accuracy"],
                "v29_neighborhood4": proposed["top1"]["relaxed_accuracy"],
                "prevalence_strict": prevalence["top1"]["strict_accuracy"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
