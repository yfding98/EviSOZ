#!/usr/bin/env python3
"""Audit fixed v29 uncertainty proxies without selecting an abstention rule.

The proxy family is fixed in source: score margin, predictive entropy, H/D
Jensen-Shannon disagreement, H/D Top-1 disagreement, and (private only)
five-fold probability variance and fold Top-1 disagreement.  Higher values
always mean more uncertainty.  The audit estimates error-detection AUROC and
descriptive risk-coverage curves on consumed public and opened private data.

No proxy, threshold or coverage point is selected for deployment.  In the
absence of a label-fresh calibration cohort the final status is necessarily
NO_CLINICAL_UNCERTAINTY_QUALIFICATION, regardless of point estimates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
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

from scripts.audit_v29_public_private_construct_shift_v47 import (  # noqa: E402
    _private_rows,
    _public_rows,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_v29_uncertainty_proxy_audit_v48"
DEFAULT_PUBLIC = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_PREDICTION = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_PRIVATE_AUDIT = (
    ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_uncertainty_proxies_v48_20260816"
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260818
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def _candidate_probability(value: torch.Tensor) -> torch.Tensor:
    result = value[:, V11_CANDIDATE_MASK].double()
    result = result / result.sum(dim=1, keepdim=True)
    if not torch.isfinite(result).all():
        raise ValueError("candidate probability is non-finite")
    return result


def _proxy_values(
    *,
    full: torch.Tensor,
    h: torch.Tensor,
    d: torch.Tensor,
    folds: torch.Tensor | None,
) -> dict[str, np.ndarray]:
    full_c = _candidate_probability(full)
    h_c = _candidate_probability(h)
    d_c = _candidate_probability(d)
    top2 = torch.topk(full_c, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(full_c.clamp_min(1e-12) * full_c.clamp_min(1e-12).log()).sum(dim=1)
    entropy /= math.log(full_c.shape[1])
    midpoint = 0.5 * (h_c + d_c)
    js = 0.5 * (
        (h_c * (h_c.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(dim=1)
        + (d_c * (d_c.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum(dim=1)
    )
    output = {
        "negative_top1_top2_margin": (-margin).numpy(),
        "normalized_predictive_entropy": entropy.numpy(),
        "H_D_jensen_shannon": js.numpy(),
        "H_D_top1_disagreement": (
            h_c.argmax(dim=1) != d_c.argmax(dim=1)
        ).double().numpy(),
    }
    if folds is not None:
        if folds.ndim != 3 or folds.shape[0] != full.shape[0] or folds.shape[2] != 19:
            raise ValueError("fold probabilities must be [N,F,19]")
        fold_c = folds[:, :, V11_CANDIDATE_MASK].double()
        fold_c = fold_c / fold_c.sum(dim=2, keepdim=True)
        output["fold_candidate_probability_variance"] = (
            fold_c.var(dim=1, unbiased=False).mean(dim=1).numpy()
        )
        fold_top = fold_c.argmax(dim=2)
        disagreements = []
        for row in fold_top:
            counts = torch.bincount(row, minlength=fold_c.shape[2])
            disagreements.append(1.0 - float(counts.max()) / len(row))
        output["fold_top1_disagreement"] = np.asarray(
            disagreements, dtype=np.float64
        )
    return output


def _patient_equal_weights(cluster_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for cluster in cluster_ids:
        counts[str(cluster)] += 1
    patients = len(counts)
    return np.asarray(
        [1.0 / (patients * counts[str(cluster)]) for cluster in cluster_ids],
        dtype=np.float64,
    )


def _weighted_auc(
    uncertainty: np.ndarray, outcome: np.ndarray, weights: np.ndarray
) -> float:
    if uncertainty.ndim != 1 or outcome.shape != uncertainty.shape or weights.shape != uncertainty.shape:
        raise ValueError("weighted AUROC arrays differ")
    positive_weight = float(weights[outcome == 1].sum())
    negative_weight = float(weights[outcome == 0].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return math.nan
    order = np.argsort(uncertainty, kind="mergesort")
    score = uncertainty[order]
    label = outcome[order]
    weight = weights[order]
    numerator = 0.0
    negative_below = 0.0
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and score[stop] == score[start]:
            stop += 1
        tied_negative = float(weight[start:stop][label[start:stop] == 0].sum())
        tied_positive = float(weight[start:stop][label[start:stop] == 1].sum())
        numerator += tied_positive * (negative_below + 0.5 * tied_negative)
        negative_below += tied_negative
        start = stop
    return numerator / (positive_weight * negative_weight)


def _cluster_auc(
    *,
    uncertainty: np.ndarray,
    outcome: np.ndarray,
    cluster_ids: Sequence[str],
    seed: int,
) -> dict[str, object]:
    cluster_rows: dict[str, list[int]] = defaultdict(list)
    for row, cluster in enumerate(cluster_ids):
        cluster_rows[str(cluster)].append(row)
    clusters = sorted(cluster_rows)
    weights = _patient_equal_weights(cluster_ids)
    point = _weighted_auc(uncertainty, outcome, weights)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_scores: list[float] = []
        sampled_outcomes: list[int] = []
        sampled_weights: list[float] = []
        for cluster in selected_clusters:
            rows = cluster_rows[str(cluster)]
            weight = 1.0 / (len(clusters) * len(rows))
            for row in rows:
                sampled_scores.append(float(uncertainty[row]))
                sampled_outcomes.append(int(outcome[row]))
                sampled_weights.append(weight)
        value = _weighted_auc(
            np.asarray(sampled_scores),
            np.asarray(sampled_outcomes),
            np.asarray(sampled_weights),
        )
        if math.isfinite(value):
            samples.append(value)
    return {
        "patient_equal_weighted_AUROC": point,
        "patient_cluster_bootstrap_ci95": (
            None
            if len(samples) < BOOTSTRAP_REPLICATES * 0.9
            else [float(value) for value in np.quantile(samples, (0.025, 0.975))]
        ),
        "bootstrap_valid_replicates": len(samples),
        "failure_patient_equal_prevalence": float(
            np.sum(weights * outcome.astype(np.float64))
        ),
    }


def _selected_risk(
    outcome: np.ndarray,
    selected: np.ndarray,
    cluster_ids: Sequence[str],
) -> dict[str, float]:
    values = outcome[selected]
    selected_clusters = [str(cluster_ids[index]) for index in selected.tolist()]
    by_patient: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values.tolist(), selected_clusters):
        by_patient[cluster].append(float(value))
    return {
        "unit_risk": float(np.mean(values)),
        "patient_equal_risk": float(
            np.mean([np.mean(items) for items in by_patient.values()])
        ),
        "selected_units": len(selected),
        "represented_patients": len(by_patient),
    }


def _risk_coverage(
    *,
    uncertainty: np.ndarray,
    strict_failure: np.ndarray,
    neighborhood_failure: np.ndarray,
    cluster_ids: Sequence[str],
) -> list[dict[str, object]]:
    order = np.argsort(uncertainty, kind="mergesort")
    rows: list[dict[str, object]] = []
    for coverage in COVERAGES:
        count = max(1, int(math.ceil(coverage * len(order))))
        selected = order[:count]
        rows.append(
            {
                "nominal_coverage": coverage,
                "actual_unit_coverage": count / len(order),
                "strict_failure": _selected_risk(
                    strict_failure, selected, cluster_ids
                ),
                "neighborhood4_failure": _selected_risk(
                    neighborhood_failure, selected, cluster_ids
                ),
                "uncertainty_cutoff": float(uncertainty[selected[-1]]),
            }
        )
    return rows


def _audit_dataset(
    *,
    rows: Sequence[Mapping[str, object]],
    proxies: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    strict_failure = np.asarray(
        [1 - int(float(row["strict"])) for row in rows], dtype=np.int64
    )
    neighborhood_failure = np.asarray(
        [int(float(row["far"])) for row in rows], dtype=np.int64
    )
    contralateral = np.asarray(
        [int(float(row["contralateral_far"])) for row in rows], dtype=np.int64
    )
    cluster_ids = [str(row["patient_id"]) for row in rows]
    result: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for index, (name, uncertainty) in enumerate(proxies.items()):
        if len(uncertainty) != len(rows) or not np.isfinite(uncertainty).all():
            raise ValueError(f"proxy {name} does not match evaluated rows")
        strict = _cluster_auc(
            uncertainty=uncertainty,
            outcome=strict_failure,
            cluster_ids=cluster_ids,
            seed=seed + 10_000 * index,
        )
        neighborhood = _cluster_auc(
            uncertainty=uncertainty,
            outcome=neighborhood_failure,
            cluster_ids=cluster_ids,
            seed=seed + 10_000 * index + 1,
        )
        contra = _cluster_auc(
            uncertainty=uncertainty,
            outcome=contralateral,
            cluster_ids=cluster_ids,
            seed=seed + 10_000 * index + 2,
        )
        curve = _risk_coverage(
            uncertainty=uncertainty,
            strict_failure=strict_failure,
            neighborhood_failure=neighborhood_failure,
            cluster_ids=cluster_ids,
        )
        result[name] = {
            "direction": "higher_means_more_uncertain",
            "distribution": {
                "minimum": float(np.min(uncertainty)),
                "median": float(np.median(uncertainty)),
                "maximum": float(np.max(uncertainty)),
            },
            "strict_failure_detection": strict,
            "neighborhood4_failure_detection": neighborhood,
            "contralateral_far_detection": contra,
            "risk_coverage": curve,
            "clinical_threshold_selected": False,
        }
        summary_rows.append(
            {
                "proxy": name,
                "strict_failure_AUROC": strict["patient_equal_weighted_AUROC"],
                "strict_ci_low": (
                    "" if strict["patient_cluster_bootstrap_ci95"] is None else strict["patient_cluster_bootstrap_ci95"][0]
                ),
                "strict_ci_high": (
                    "" if strict["patient_cluster_bootstrap_ci95"] is None else strict["patient_cluster_bootstrap_ci95"][1]
                ),
                "neighborhood_failure_AUROC": neighborhood[
                    "patient_equal_weighted_AUROC"
                ],
                "neighborhood_ci_low": (
                    "" if neighborhood["patient_cluster_bootstrap_ci95"] is None else neighborhood["patient_cluster_bootstrap_ci95"][0]
                ),
                "neighborhood_ci_high": (
                    "" if neighborhood["patient_cluster_bootstrap_ci95"] is None else neighborhood["patient_cluster_bootstrap_ci95"][1]
                ),
                "contralateral_far_AUROC": contra["patient_equal_weighted_AUROC"],
            }
        )
        for value in curve:
            coverage_rows.append(
                {
                    "proxy": name,
                    "nominal_coverage": value["nominal_coverage"],
                    "actual_unit_coverage": value["actual_unit_coverage"],
                    "strict_unit_risk": value["strict_failure"]["unit_risk"],
                    "strict_patient_equal_risk": value["strict_failure"][
                        "patient_equal_risk"
                    ],
                    "neighborhood4_unit_risk": value["neighborhood4_failure"][
                        "unit_risk"
                    ],
                    "neighborhood4_patient_equal_risk": value[
                        "neighborhood4_failure"
                    ]["patient_equal_risk"],
                    "represented_patients": value["strict_failure"][
                        "represented_patients"
                    ],
                }
            )
    return result, summary_rows, coverage_rows


def run(
    *,
    public_directory: Path,
    private_prediction_directory: Path,
    private_audit_directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    public_rows, _, public_path = _public_rows(public_directory)
    public_payload = load_file(str(public_path), device="cpu")
    public_full = public_payload["oof.portable_equal_ensemble_probability"].float()
    public_h = public_payload["oof.h_only_probability"].float()
    public_d = public_payload["oof.rank1_direct_probability"].float()
    public_proxies = _proxy_values(
        full=public_full, h=public_h, d=public_d, folds=None
    )

    private_rows, private_audit_path = _private_rows(private_audit_directory)
    private_manifest_path = (
        private_prediction_directory / "manifest.json"
    ).resolve(strict=True)
    private_tensor_path = (
        private_prediction_directory / "predictions.safetensors"
    ).resolve(strict=True)
    manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 88:
        raise ValueError("private prediction roster changed")
    event_index = {str(row["event_id"]): index for index, row in enumerate(events)}
    selected = torch.tensor(
        [event_index[str(row["unit_id"])] for row in private_rows], dtype=torch.long
    )
    private_payload = load_file(str(private_tensor_path), device="cpu")
    full_all = private_payload["private_portable_equal_probability"].float()
    h_fold_all = private_payload["private_h_only_fold_probability"].float()
    d_fold_all = private_payload["private_rank1_direct_fold_probability"].float()
    h_all = h_fold_all.mean(dim=1)
    d_all = d_fold_all.mean(dim=1)
    combined_fold_all = 0.5 * h_fold_all + 0.5 * d_fold_all
    private_proxies_all = _proxy_values(
        full=full_all,
        h=h_all,
        d=d_all,
        folds=combined_fold_all,
    )
    private_proxies = {
        name: value[selected.numpy()] for name, value in private_proxies_all.items()
    }

    public_result, public_summary, public_coverage = _audit_dataset(
        rows=public_rows, proxies=public_proxies, seed=BOOTSTRAP_SEED
    )
    private_result, private_summary, private_coverage = _audit_dataset(
        rows=private_rows, proxies=private_proxies, seed=BOOTSTRAP_SEED + 1_000_000
    )
    for row in public_summary:
        row["dataset"] = "public_consumed_development"
    for row in private_summary:
        row["dataset"] = "private_post_open_transport"
    for row in public_coverage:
        row["dataset"] = "public_consumed_development"
    for row in private_coverage:
        row["dataset"] = "private_post_open_transport"

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "NO_CLINICAL_UNCERTAINTY_QUALIFICATION",
        "analysis_role": {
            "public": "posthoc_consumed_development_uncertainty_audit",
            "private": "post_open_descriptive_uncertainty_transport_audit",
        },
        "proxy_family_fixed_before_this_execution": list(public_proxies),
        "private_additional_fold_proxies": [
            name for name in private_proxies if name not in public_proxies
        ],
        "public": {
            "patient_count": len(public_rows),
            "proxies": public_result,
        },
        "private": {
            "event_count": len(private_rows),
            "patient_cluster_count": len({str(row["patient_id"]) for row in private_rows}),
            "proxies": private_result,
        },
        "qualification": {
            "clinical_uncertainty_qualified": False,
            "clinical_abstention_threshold_selected": False,
            "reasons": [
                "no_label_fresh_calibration_cohort",
                "public_is_consumed_adaptive_development",
                "private_is_historically_opened_and_cross_construct",
                "multiple_proxy_results_are_descriptive_not_a_selection_pool",
                "a_score_or_fold_disagreement_is_not_a_calibrated_error_probability",
            ],
            "existing_v42_margin_decision_preserved": "NO_CLINICAL_RISK_QUALIFICATION",
        },
        "source_files": {
            "public_prediction": str(public_path.relative_to(ROOT)),
            "private_prediction": str(private_tensor_path.relative_to(ROOT)),
            "private_event_audit": str(private_audit_path.relative_to(ROOT)),
        },
        "access_receipt": {
            "raw_EEG_loaded": False,
            "model_training_or_calibration_performed": False,
            "proxy_threshold_or_coverage_selected": False,
            "public_targets_loaded_for_descriptive_error_detection": True,
            "opened_private_targets_loaded_for_descriptive_error_detection": True,
            "report_candidate_or_wording_changed": False,
        },
        "interpretation_boundary": {
            "AUROC_is_calibrated_probability": False,
            "risk_coverage_is_future_risk_guarantee": False,
            "best_proxy_may_be_selected_for_current_reports": False,
            "private_result_is_fresh_validation": False,
            "allowed_claim": (
                "fixed score- and disagreement-based uncertainty proxies were "
                "audited descriptively and none was promoted to clinical confidence"
            ),
        },
    }
    return result, public_summary + private_summary, public_coverage + private_coverage


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
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
        for filename, rows in (
            ("proxy_summary.csv", summary_rows),
            ("risk_coverage.csv", coverage_rows),
        ):
            fields = sorted({key for row in rows for key in row})
            with (staging / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
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
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument(
        "--private-prediction", type=Path, default=DEFAULT_PRIVATE_PREDICTION
    )
    parser.add_argument("--private-audit", type=Path, default=DEFAULT_PRIVATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, summary, coverage = run(
        public_directory=args.public,
        private_prediction_directory=args.private_prediction,
        private_audit_directory=args.private_audit,
    )
    output = publish(
        output=args.output,
        result=result,
        summary_rows=summary,
        coverage_rows=coverage,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_proxies": len(result["public"]["proxies"]),
                "private_proxies": len(result["private"]["proxies"]),
                "threshold_selected": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
