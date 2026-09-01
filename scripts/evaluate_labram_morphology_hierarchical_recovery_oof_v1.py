#!/usr/bin/env python3
"""Paired complete-OOF comparison of hierarchical recovery versus old M0.

This CPU-only command reads the same source-train TUEV items for both models,
uses a parent-group paired bootstrap, performs no fitting or threshold search,
and has no official-evaluation input path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_preprocessing_parity_formal import _state_sha256  # noqa: E402
from scripts.select_tuev_morphology_oof_thresholds import (  # noqa: E402
    _load_oof_probabilities,
)
from src.soz.morphology_recovery import (  # noqa: E402
    audit_morphology_recovery_source,
    load_morphology_recovery_preflight,
)
from src.soz.morphology_recovery_oof import (  # noqa: E402
    MORPHOLOGY_RECOVERY_OOF_CANDIDATE,
    load_morphology_recovery_oof_run,
)
from src.soz.morphology_recovery_summary import (  # noqa: E402
    MORPHOLOGY_RECOVERY_SUMMARY_SCHEMA,
    load_morphology_recovery_summary,
    save_morphology_recovery_summary,
)


_SHA_RE = re.compile(r"[0-9a-f]{64}")
_BOOTSTRAP_REPLICATES = 2000
_BOOTSTRAP_SEED = 20260808


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--parity-directory",
        type=Path,
        default=Path("outputs/preprocessing_parity_formal_v1_20260809"),
    )
    parser.add_argument(
        "--preflight-bundle",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_preflight_v1_20260810"
        ),
    )
    parser.add_argument(
        "--expected-preflight-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_oof_v1_20260810"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_oof_v1_20260810/"
            "paired_development_summary_v1"
        ),
    )
    return parser


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _BinaryScorePlan:
    truth: np.ndarray
    base_weights: np.ndarray
    group_indices: np.ndarray
    order: np.ndarray
    tie_indices: np.ndarray
    tie_count: int

    @classmethod
    def build(
        cls,
        truth: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        group_indices: np.ndarray,
    ) -> "_BinaryScorePlan":
        truth = np.asarray(truth, dtype=np.bool_)
        scores = np.asarray(scores, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        groups = np.asarray(group_indices, dtype=np.int64)
        if not (
            truth.ndim == scores.ndim == weights.ndim == groups.ndim == 1
            and len(truth) == len(scores) == len(weights) == len(groups)
            and len(truth) > 0
        ):
            raise ValueError("Binary metric inputs must be aligned non-empty vectors")
        if (
            not np.isfinite(scores).all()
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or np.any(groups < 0)
        ):
            raise ValueError("Binary metric values/weights/groups are invalid")
        order = np.argsort(-scores, kind="stable")
        ordered_scores = scores[order]
        starts = np.ones(len(order), dtype=np.bool_)
        starts[1:] = ordered_scores[1:] != ordered_scores[:-1]
        tie_indices = np.cumsum(starts, dtype=np.int64) - 1
        return cls(
            truth=truth,
            base_weights=weights,
            group_indices=groups,
            order=order,
            tie_indices=tie_indices,
            tie_count=int(tie_indices[-1]) + 1,
        )

    def ap_auroc(self, multiplicities: np.ndarray) -> tuple[float, float]:
        multiplicities = np.asarray(multiplicities, dtype=np.float64)
        if multiplicities.ndim != 1 or np.any(multiplicities < 0):
            raise ValueError("Bootstrap multiplicities must be non-negative [G]")
        weights = self.base_weights * multiplicities[self.group_indices]
        ordered_weights = weights[self.order]
        ordered_truth = self.truth[self.order]
        positive_by_tie = np.bincount(
            self.tie_indices,
            weights=ordered_weights * ordered_truth,
            minlength=self.tie_count,
        )
        negative_by_tie = np.bincount(
            self.tie_indices,
            weights=ordered_weights * (~ordered_truth),
            minlength=self.tie_count,
        )
        positive = float(positive_by_tie.sum())
        negative = float(negative_by_tie.sum())
        if positive <= 0 or negative <= 0:
            raise ValueError("Resampled binary metric lacks positive/negative support")
        cumulative_positive = np.cumsum(positive_by_tie)
        cumulative_total = np.cumsum(positive_by_tie + negative_by_tie)
        precision = cumulative_positive / np.maximum(
            cumulative_total, np.finfo(np.float64).eps
        )
        average_precision = float(np.sum(precision * positive_by_tie) / positive)
        negative_below = negative - np.cumsum(negative_by_tie)
        concordant = np.sum(
            positive_by_tie * (negative_below + 0.5 * negative_by_tie),
            dtype=np.float64,
        )
        auroc = float(concordant / (positive * negative))
        return average_precision, auroc


def _group_macro_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    group_indices: np.ndarray,
    group_count: int,
) -> np.ndarray:
    values = np.empty(group_count, dtype=np.float64)
    for group in range(group_count):
        member = group_indices == group
        recalls = []
        for class_index in range(6):
            truth = member & (labels == class_index)
            if truth.any():
                recalls.append(
                    float(
                        weights[truth & (predictions == class_index)].sum()
                        / weights[truth].sum()
                    )
                )
        if not recalls:
            raise ValueError("OOF group has no explicit CE6 target")
        values[group] = sum(recalls) / len(recalls)
    return values


def _classwise_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    group_indices: np.ndarray,
    class_index: int,
) -> dict[str, float]:
    truth = labels == class_index
    predicted = probabilities.argmax(axis=-1) == class_index
    tp = weights[truth & predicted].sum()
    fp = weights[(~truth) & predicted].sum()
    fn = weights[truth & (~predicted)].sum()
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    plan = _BinaryScorePlan.build(
        truth, probabilities[:, class_index], weights, group_indices
    )
    ap, _ = plan.ap_auroc(np.ones(int(group_indices.max()) + 1))
    return {
        "support": float(weights[truth].sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
    }


def _ci(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _metric_row(
    baseline: float,
    candidate: float,
    bootstrap_delta: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    return {
        "baseline": float(baseline),
        "candidate": float(candidate),
        "delta_candidate_minus_baseline": float(candidate - baseline),
        "delta_ci95": _ci(bootstrap_delta),
        "higher_is_better": higher_is_better,
    }


def _strict_baseline_artifacts(
    parity: Path,
) -> list[dict[str, object]]:
    rows = []
    for fold in range(5):
        directory = parity / "nested-checkpoints" / "tuev" / "C-CAR19" / f"fold-{fold}"
        receipt_path = (directory / "receipt.json").resolve(strict=True)
        model_path = (directory / "model.safetensors").resolve(strict=True)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_fields = {
            "schema_version",
            "dataset",
            "arm_id",
            "fold",
            "config",
            "config_sha256",
            "fit_ids",
            "fit_roster_sha256",
            "held_ids",
            "held_roster_sha256",
            "model_file",
            "model_file_sha256",
            "state_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected_fields:
            raise ValueError("Old M0 checkpoint receipt violates its closed schema")
        if (
            receipt["dataset"] != "TUEV"
            or receipt["arm_id"] != "C-CAR19"
            or receipt["fold"] != fold
            or receipt["config"].get("loss")
            != "group_equal_overlap_component_weighted_CE6"
            or receipt["config"].get("checkpoint_selection") != "fixed_final_epoch"
        ):
            raise ValueError("Old M0 checkpoint identity/config changed")
        model_sha = _file_sha256(model_path)
        if model_sha != receipt["model_file_sha256"]:
            raise ValueError("Old M0 checkpoint file SHA changed")
        state = load_file(str(model_path), device="cpu")
        if _state_sha256(state) != receipt["state_sha256"]:
            raise ValueError("Old M0 checkpoint state SHA changed")
        rows.append(
            {
                "fold": fold,
                "receipt_file_sha256": _file_sha256(receipt_path),
                "checkpoint_file_sha256": model_sha,
                "state_sha256": str(receipt["state_sha256"]),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if torch.cuda.is_initialized():
        raise RuntimeError("Paired morphology summary must remain CPU-only")
    receipt, preflight_payload = load_morphology_recovery_preflight(
        args.preflight_bundle,
        expected_receipt_sha256=args.expected_preflight_receipt_sha256,
        verify_source_files=True,
    )
    parity = args.parity_directory.resolve(strict=True)
    paths = {
        "run_plan": parity / "run-plan.json",
        "tokens": parity / "arrays" / "tuev_tokens_C-CAR19.npy",
        "labels": parity / "arrays" / "tuev_labels.npy",
        "mask": parity / "arrays" / "tuev_mask.npy",
        "weights": parity / "arrays" / "tuev_weights.npy",
    }
    for name, path in paths.items():
        if path.resolve(strict=True) != Path(
            str(preflight_payload["source_files"][name]["path"])
        ).resolve(strict=True):
            raise ValueError("Paired summary parity source differs from preflight")
    run_plan = json.loads(paths["run_plan"].read_text(encoding="utf-8"))
    tokens = np.load(paths["tokens"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    masks = np.load(paths["mask"], mmap_mode="r")
    weights = np.load(paths["weights"], mmap_mode="r")
    current = audit_morphology_recovery_source(
        run_plan=run_plan,
        tokens=tokens,
        labels=labels,
        source_target_mask=masks,
        overlap_component_weights=weights,
    )
    if current.canonical_payload != receipt.canonical_payload:
        raise ValueError("Paired summary source changed after preflight")
    items = run_plan["tuev_items"]
    item_count = len(items)
    group_names = tuple(sorted({str(item["parent_group_id"]) for item in items}))
    group_to_index = {group: index for index, group in enumerate(group_names)}
    item_group = np.empty(item_count, dtype=np.int64)
    item_fold = np.empty(item_count, dtype=np.int64)
    group_fold: dict[str, int] = {}
    for item in items:
        index = int(item["index"])
        group = str(item["parent_group_id"])
        fold = int(item["fold"])
        item_group[index] = group_to_index[group]
        item_fold[index] = fold
        previous = group_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError("Paired summary group crosses folds")

    candidate_root = args.candidate_root.resolve(strict=True)
    candidate = np.empty((item_count, 20, 6), dtype=np.float32)
    seen = np.zeros(item_count, dtype=np.bool_)
    candidate_folds = []
    for fold in range(5):
        run = load_morphology_recovery_oof_run(candidate_root / f"fold{fold}")
        expected_groups = tuple(
            sorted(group for group, value in group_fold.items() if value == fold)
        )
        if tuple(run.manifest["held_group_ids"]) != expected_groups:
            raise ValueError("Candidate held group roster differs from source plan")
        if (
            run.manifest["preflight_receipt_sha256"] != receipt.receipt_sha256
            or run.manifest["source_plan_sha256"] != receipt.source_plan_sha256
        ):
            raise ValueError("Candidate fold lineage differs from paired source")
        indices = run.held_item_indices.numpy()
        expected_indices = np.flatnonzero(item_fold == fold)
        if not np.array_equal(indices, expected_indices) or seen[indices].any():
            raise ValueError("Candidate fold item roster is not exact OOF")
        candidate[indices] = run.ce6_probabilities.numpy()
        seen[indices] = True
        candidate_folds.append(
            {"fold": fold, "manifest_file_sha256": run.manifest_file_sha256}
        )
    if not seen.all():
        raise RuntimeError("Candidate complete-OOF prediction coverage is incomplete")

    baseline_folds = _strict_baseline_artifacts(parity)
    baseline = _load_oof_probabilities(
        parity,
        items,
        tokens,
        device=torch.device("cpu"),
        batch_size=256,
    )
    if baseline.shape != candidate.shape or not np.isfinite(baseline).all():
        raise ValueError("Old M0 item-level scores are unavailable or incompatible")

    observed_labels = np.asarray(labels)[np.asarray(masks)]
    observed_weights = np.asarray(weights, dtype=np.float64)[np.asarray(masks)]
    observed_group = np.broadcast_to(item_group[:, None], masks.shape)[np.asarray(masks)]
    candidate_observed = candidate[np.asarray(masks)]
    baseline_observed = baseline[np.asarray(masks)]
    if len(observed_labels) != receipt.observed_cell_count:
        raise ValueError("Paired observed-cell denominator changed")
    local_truth = np.isin(observed_labels, (0, 2))
    candidate_local = candidate_observed[:, 0] + candidate_observed[:, 2]
    baseline_local = baseline_observed[:, 0] + baseline_observed[:, 2]
    candidate_plan = _BinaryScorePlan.build(
        local_truth, candidate_local, observed_weights, observed_group
    )
    baseline_plan = _BinaryScorePlan.build(
        local_truth, baseline_local, observed_weights, observed_group
    )
    group_count = len(group_names)
    ones = np.ones(group_count, dtype=np.float64)
    candidate_ap, candidate_auc = candidate_plan.ap_auroc(ones)
    baseline_ap, baseline_auc = baseline_plan.ap_auroc(ones)

    group_mass = np.bincount(
        observed_group, weights=observed_weights, minlength=group_count
    )
    candidate_brier_group = np.bincount(
        observed_group,
        weights=observed_weights * np.square(candidate_local - local_truth),
        minlength=group_count,
    )
    baseline_brier_group = np.bincount(
        observed_group,
        weights=observed_weights * np.square(baseline_local - local_truth),
        minlength=group_count,
    )
    candidate_nll_values = -(
        local_truth * np.log(np.maximum(candidate_local, 1e-12))
        + (~local_truth) * np.log(np.maximum(1.0 - candidate_local, 1e-12))
    )
    baseline_nll_values = -(
        local_truth * np.log(np.maximum(baseline_local, 1e-12))
        + (~local_truth) * np.log(np.maximum(1.0 - baseline_local, 1e-12))
    )
    candidate_nll_group = np.bincount(
        observed_group,
        weights=observed_weights * candidate_nll_values,
        minlength=group_count,
    )
    baseline_nll_group = np.bincount(
        observed_group,
        weights=observed_weights * baseline_nll_values,
        minlength=group_count,
    )
    total_mass = group_mass.sum()
    candidate_brier = float(candidate_brier_group.sum() / total_mass)
    baseline_brier = float(baseline_brier_group.sum() / total_mass)
    candidate_nll = float(candidate_nll_group.sum() / total_mass)
    baseline_nll = float(baseline_nll_group.sum() / total_mass)

    candidate_group_ba = _group_macro_balanced_accuracy(
        observed_labels,
        candidate_observed.argmax(axis=-1),
        observed_weights,
        observed_group,
        group_count,
    )
    baseline_group_ba = _group_macro_balanced_accuracy(
        observed_labels,
        baseline_observed.argmax(axis=-1),
        observed_weights,
        observed_group,
        group_count,
    )
    candidate_ba = float(candidate_group_ba.mean())
    baseline_ba = float(baseline_group_ba.mean())

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    delta_ap = np.empty(_BOOTSTRAP_REPLICATES, dtype=np.float64)
    delta_auc = np.empty_like(delta_ap)
    delta_brier = np.empty_like(delta_ap)
    delta_nll = np.empty_like(delta_ap)
    delta_ba = np.empty_like(delta_ap)
    probability = np.full(group_count, 1.0 / group_count)
    for replicate in range(_BOOTSTRAP_REPLICATES):
        multiplicities = rng.multinomial(group_count, probability).astype(np.float64)
        cand_ap, cand_auc = candidate_plan.ap_auroc(multiplicities)
        base_ap, base_auc = baseline_plan.ap_auroc(multiplicities)
        denominator = float(np.dot(multiplicities, group_mass))
        if denominator <= 0:
            raise RuntimeError("Cluster bootstrap produced no observed target mass")
        delta_ap[replicate] = cand_ap - base_ap
        delta_auc[replicate] = cand_auc - base_auc
        delta_brier[replicate] = float(
            np.dot(multiplicities, candidate_brier_group - baseline_brier_group)
            / denominator
        )
        delta_nll[replicate] = float(
            np.dot(multiplicities, candidate_nll_group - baseline_nll_group)
            / denominator
        )
        delta_ba[replicate] = float(
            np.dot(multiplicities, candidate_group_ba - baseline_group_ba)
            / group_count
        )

    classwise = {}
    for name, class_index in (("SPSW", 0), ("PLED", 2)):
        candidate_class = _classwise_metrics(
            candidate_observed,
            observed_labels,
            observed_weights,
            observed_group,
            class_index,
        )
        baseline_class = _classwise_metrics(
            baseline_observed,
            observed_labels,
            observed_weights,
            observed_group,
            class_index,
        )
        classwise[name] = {
            "baseline": baseline_class,
            "candidate": candidate_class,
            "delta_candidate_minus_baseline": {
                metric: float(candidate_class[metric] - baseline_class[metric])
                for metric in ("precision", "recall", "f1", "average_precision")
            },
        }

    localizing = {
        "average_precision": _metric_row(
            baseline_ap, candidate_ap, delta_ap, higher_is_better=True
        ),
        "auroc": _metric_row(
            baseline_auc, candidate_auc, delta_auc, higher_is_better=True
        ),
        "brier": _metric_row(
            baseline_brier, candidate_brier, delta_brier, higher_is_better=False
        ),
        "nll": _metric_row(
            baseline_nll, candidate_nll, delta_nll, higher_is_better=False
        ),
    }
    ce6_ba = _metric_row(
        baseline_ba, candidate_ba, delta_ba, higher_is_better=True
    )
    checks = {
        "ap_delta_ci95_lower_gt_zero": localizing["average_precision"]["delta_ci95"][0] > 0,
        "auroc_point_delta_nonnegative": localizing["auroc"]["delta_candidate_minus_baseline"] >= 0,
        "brier_point_delta_nonpositive": localizing["brier"]["delta_candidate_minus_baseline"] <= 0,
        "nll_point_delta_nonpositive": localizing["nll"]["delta_candidate_minus_baseline"] <= 0,
        "ce6_group_macro_balanced_accuracy_delta_nonnegative": ce6_ba[
            "delta_candidate_minus_baseline"
        ]
        >= 0,
        "spsw_ap_delta_nonnegative": classwise["SPSW"][
            "delta_candidate_minus_baseline"
        ]["average_precision"]
        >= 0,
        "pled_ap_delta_nonnegative": classwise["PLED"][
            "delta_candidate_minus_baseline"
        ]["average_precision"]
        >= 0,
    }
    development_signal = all(checks.values())
    metrics = {
        "localizing": localizing,
        "ce6": {
            "group_macro_balanced_accuracy": ce6_ba,
            "classwise": classwise,
        },
        "recovery_decision": {
            "rule": (
                "all: AP delta CI95 lower>0; AUROC delta>=0; Brier/NLL delta<=0; "
                "CE6 group-macro BA delta>=0; SPSW/PLED AP deltas>=0"
            ),
            "checks": checks,
            "development_recovery_signal": development_signal,
            "decision": "GO_DEVELOPMENT_SIGNAL" if development_signal else "NO_GO",
            "formal_promotion": False,
        },
    }
    source_hashes = {
        name: str(row["sha256"])
        for name, row in preflight_payload["source_files"].items()
    }
    item_roster = tuple(
        (
            int(item["index"]),
            str(item["crop_id"]),
            str(item["parent_group_id"]),
            int(item["fold"]),
        )
        for item in items
    )
    payload = {
        "schema_version": MORPHOLOGY_RECOVERY_SUMMARY_SCHEMA,
        "development_only": True,
        "formal_promotion": False,
        "dense_deployment_authorized": False,
        "soz_reasoner_authorized": False,
        "official_tuev_eval_used": False,
        "threshold_selection_performed": False,
        "training_performed_by_summary": False,
        "comparison_scope": "same_source_train_items_complete_group_oof_paired",
        "target_semantics": "tuev_native_ce6_bipolar_edge_not_soz",
        "candidate_name": MORPHOLOGY_RECOVERY_OOF_CANDIDATE,
        "baseline_name": "labram_frozen_independent_ce6_m0",
        "protocol_sha256": str(candidate_folds and load_morphology_recovery_oof_run(candidate_root / "fold0").manifest["protocol_sha256"]),
        "preflight_receipt_sha256": receipt.receipt_sha256,
        "source_plan_sha256": receipt.source_plan_sha256,
        "source_files_sha256": source_hashes,
        "source_item_count": item_count,
        "source_group_count": group_count,
        "observed_cell_count": int(len(observed_labels)),
        "source_item_roster_sha256": _canonical_sha256(item_roster),
        "source_group_roster_sha256": _canonical_sha256(group_names),
        "oof_prediction_coverage_complete": True,
        "candidate_fold_manifests": candidate_folds,
        "baseline_fold_artifacts": baseline_folds,
        "bootstrap": {
            "unit": "tuev_parent_group",
            "paired": True,
            "replicates": _BOOTSTRAP_REPLICATES,
            "seed": _BOOTSTRAP_SEED,
            "interval": "percentile_2.5_97.5",
        },
        "metrics": metrics,
        "interpretation_boundary": (
            "retrospective_development_comparison_not_dense_M1_or_SOZ_evidence"
        ),
    }
    saved = save_morphology_recovery_summary(args.output_directory, payload)
    replayed = load_morphology_recovery_summary(
        saved.path, expected_file_sha256=saved.file_sha256
    )
    print(
        json.dumps(
            {
                "status": "STRICT_REPLAY_PASS",
                "output": str(replayed.path),
                "summary_file_sha256": replayed.file_sha256,
                "metrics": replayed.payload["metrics"],
                "official_tuev_eval_used": False,
                "formal_promotion": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

