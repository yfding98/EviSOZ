#!/usr/bin/env python3
"""Collect the strict VEPiSet main-result audits into one JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path) -> Dict[str, Any]:
    return read_json(path) if path.exists() else {}


def metric_block(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
    }


def ci_block(bootstrap: Mapping[str, Any], metric: str) -> Dict[str, float]:
    stats = bootstrap.get("bootstrap_ci", {}).get(metric, {})
    return {
        "mean": float(stats.get("mean", 0.0)),
        "std": float(stats.get("std", 0.0)),
        "p2_5": float(stats.get("p2_5", 0.0)),
        "p50": float(stats.get("p50", 0.0)),
        "p97_5": float(stats.get("p97_5", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--positive-spatial-bias-run", default="")
    parser.add_argument("--min-test-accuracy", type=float, default=0.80)
    parser.add_argument("--min-test-macro-f1", type=float, default=0.40)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_run_dir = Path(args.base_run_dir)
    positive_spatial_bias_run = (
        Path(args.positive_spatial_bias_run)
        if args.positive_spatial_bias_run
        else run_dir.parent / "vepiset_ied_v2_full6_patientclasssplit_main_positive_spatial_bias_weighted_tiny"
    )
    output_json = Path(args.output_json) if args.output_json else run_dir / "strict_main_summary.json"

    config = maybe_read_json(base_run_dir / "config.json")
    strict_audit = read_json(run_dir / "strict_result_audit.json")
    lineage_audit = maybe_read_json(run_dir / "strict_lineage_audit.json")
    patient_prior = maybe_read_json(run_dir / "patient_prior_metrics.json")
    bootstrap = maybe_read_json(run_dir / "patient_bootstrap_test.json")
    patient_proxy_mean = maybe_read_json(run_dir / "patient_proxy_metrics_mean.json")
    patient_proxy_positive = maybe_read_json(run_dir / "patient_proxy_metrics_mean_positive_selector.json")
    positive_localization = maybe_read_json(run_dir / "positive_localization_metrics.json")
    positive_spatial_bias = maybe_read_json(positive_spatial_bias_run / "positive_spatial_bias_metrics.json")
    positive_spatial_bias_full = maybe_read_json(positive_spatial_bias_run / "strict_result_audit.json")
    baseline_claim = maybe_read_json(run_dir / "baseline_claim_audit.json")

    test_metrics = metric_block(strict_audit["test_metrics"])
    overlap = strict_audit.get("patient_overlap", {})
    no_overlap = not bool(overlap.get("has_overlap", True))
    lineage_ok = bool(lineage_audit.get("lineage_requirements_met", False))
    accuracy_ok = test_metrics["accuracy"] >= float(args.min_test_accuracy)
    macro_ok = test_metrics["macro_f1"] >= float(args.min_test_macro_f1)

    pos_loc_test = positive_localization.get("test", {}).get("conditional_positive", {})
    pos_loc_window = metric_block(pos_loc_test.get("window_metrics", {}))
    pos_loc_patient = pos_loc_test.get("patient_metrics", {})
    patient_proxy_known = patient_proxy_mean.get("test_known_positive", {})
    patient_proxy_thresholded = patient_proxy_positive.get("test_thresholded", {})
    positive_spatial_bias_test = positive_spatial_bias.get("test", {})
    positive_spatial_bias_window = metric_block(positive_spatial_bias_test.get("window_metrics", {}))
    positive_spatial_bias_patient = positive_spatial_bias_test.get("patient_metrics", {})
    positive_spatial_bias_full_metrics = metric_block(positive_spatial_bias_full.get("test_metrics", {}))

    summary = {
        "run_dir": str(run_dir),
        "base_run_dir": str(base_run_dir),
        "dataset_root": config.get("vepiset_root", ""),
        "model_core": "code/models/integration_model_v2.py",
        "training_script": "code/models/train_vepiset_ied_v2.py",
        "split_strategy": config.get("split_strategy", ""),
        "seed": config.get("seed", None),
        "split_seed": config.get("split_seed", config.get("seed", None)),
        "test_window_metrics": test_metrics,
        "patient_overlap": overlap,
        "lineage_audit": {
            "lineage_requirements_met": lineage_ok,
            "audit_path": str(run_dir / "strict_lineage_audit.json"),
            "failures": lineage_audit.get("failures", []),
        },
        "baseline_claim_audit": {
            "audit_path": str(run_dir / "baseline_claim_audit.json"),
            "majority_baseline": baseline_claim.get("majority_baseline", {}),
            "main_delta_vs_majority": baseline_claim.get("main_delta_vs_majority", {}),
            "max_accuracy_audit": baseline_claim.get("max_accuracy_audit", {}),
            "high_accuracy_operating_point": baseline_claim.get("high_accuracy_operating_point", {}),
            "claim_checks": baseline_claim.get("claim_checks", {}),
        },
        "secondary_operating_points": {
            "recommended_high_accuracy": baseline_claim.get("high_accuracy_operating_point", {}),
            "max_accuracy_audit": baseline_claim.get("max_accuracy_audit", {}),
            "conditional_positive_spatial_bias": {
                "run_dir": str(positive_spatial_bias_run),
                "audit_path": str(positive_spatial_bias_run / "positive_spatial_bias_metrics.json"),
                "selector": positive_spatial_bias.get("selector", ""),
                "selection_aggregation": positive_spatial_bias.get("selection_aggregation", ""),
                "positive_class_bias": positive_spatial_bias.get("positive_class_bias", {}),
                "test_window_metrics": positive_spatial_bias_window,
                "test_patient_hit_accuracy": float(positive_spatial_bias_patient.get("hit_accuracy", 0.0)),
                "full_six_class_test_metrics": positive_spatial_bias_full_metrics,
                "adopted_as_main": False,
                "reason_not_main": (
                    "This audit conditions on true IED-positive windows and lowers full six-class "
                    "macro-F1 relative to the strict main result."
                ),
            },
        },
        "validation_selection": {
            "checkpoint_metric": "validation macro_f1",
            "calibration_scope": "validation only",
            "test_scope": "report only after choices fixed",
            "patient_prior_params": patient_prior.get("params", {}),
        },
        "patient_bootstrap": {
            "n_patients": int(bootstrap.get("n_patients", 0)),
            "n_bootstrap": int(bootstrap.get("n_bootstrap", 0)),
            "macro_f1_ci": ci_block(bootstrap, "macro_f1"),
            "balanced_accuracy_ci": ci_block(bootstrap, "balanced_accuracy"),
        },
        "patient_level_proxy": {
            "known_positive_test_hit_accuracy": float(patient_proxy_known.get("hit_accuracy", 0.0)),
            "known_positive_test_hit_count": int(patient_proxy_known.get("hit_count", 0)),
            "known_positive_test_n": int(patient_proxy_known.get("n_positive_patients", 0)),
            "thresholded_positive_priority_test_hit_accuracy": float(patient_proxy_thresholded.get("hit_accuracy", 0.0)),
            "thresholded_positive_priority_test_hit_count": int(patient_proxy_thresholded.get("hit_count", 0)),
            "thresholded_positive_priority_test_n": int(patient_proxy_thresholded.get("n_patients", 0)),
        },
        "ied_positive_oracle_localization": {
            "test_window_metrics": pos_loc_window,
            "test_patient_hit_accuracy": float(pos_loc_patient.get("hit_accuracy", 0.0)),
            "test_patient_hit_count": int(pos_loc_patient.get("hit_count", 0)),
            "test_positive_patients": int(pos_loc_patient.get("n_positive_patients", 0)),
            "scope": "oracle true IED-positive windows only; not deployable clinical SOZ localization",
        },
        "requirement_checks": {
            "patient_disjoint_split": no_overlap,
            "lineage_requirements_met": lineage_ok,
            "accuracy_beats_majority_baseline": bool(
                baseline_claim.get("claim_checks", {}).get("accuracy_beats_majority", False)
            ),
            "macro_f1_beats_majority_baseline": bool(
                baseline_claim.get("claim_checks", {}).get("macro_f1_beats_majority", False)
            ),
            "test_accuracy_ge_threshold": accuracy_ok,
            "test_macro_f1_ge_threshold": macro_ok,
            "proxy_requirements_met": bool(no_overlap and lineage_ok and accuracy_ok and macro_ok),
            "clinical_soz_sota_claim_supported": False,
            "reason_clinical_soz_sota_not_supported": (
                "VEPiSet labels are IED spatial-distribution labels, not clinical SOZ ground truth."
            ),
        },
        "recommended_claim": (
            "Strict patient-disjoint VEPiSet IED spatial-distribution proxy performance: "
            f"accuracy {test_metrics['accuracy']:.4f}, macro-F1 {test_metrics['macro_f1']:.4f}, "
            f"weighted-F1 {test_metrics['weighted_f1']:.4f}. Do not claim clinical scalp-EEG SOZ SOTA."
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "patient_disjoint_split": no_overlap,
        "lineage_requirements_met": lineage_ok,
        "proxy_requirements_met": summary["requirement_checks"]["proxy_requirements_met"],
        "clinical_soz_sota_claim_supported": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
