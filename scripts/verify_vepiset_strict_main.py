#!/usr/bin/env python3
"""One-command verification for the current strict VEPiSet main result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_RUN_DIR = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87")
DEFAULT_BASE_RUN_DIR = Path("outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20")
DEFAULT_POSITIVE_SPATIAL_BIAS_RUN = Path("outputs/vepiset_ied_v2_full6_patientclasssplit_main_positive_spatial_bias_weighted_tiny")


def run_command(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--base-run-dir", default=str(DEFAULT_BASE_RUN_DIR))
    parser.add_argument("--positive-spatial-bias-run", default=str(DEFAULT_POSITIVE_SPATIAL_BIAS_RUN))
    parser.add_argument("--min-test-accuracy", type=float, default=0.80)
    parser.add_argument("--min-test-macro-f1", type=float, default=0.40)
    parser.add_argument("--skip-derived", action="store_true", help="Only rerun strict audit and summary.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_run_dir = Path(args.base_run_dir)
    positive_spatial_bias_run = Path(args.positive_spatial_bias_run)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    python = sys.executable

    split_summary = base_run_dir / "split_summary.json"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not split_summary.exists():
        raise FileNotFoundError(split_summary)

    run_command([
        python,
        str(script_dir / "audit_vepiset_strict_lineage.py"),
        "--base-run",
        str(base_run_dir),
        "--main-run",
        str(run_dir),
        "--output-json",
        str(run_dir / "strict_lineage_audit.json"),
    ])

    run_command([
        python,
        str(script_dir / "audit_vepiset_strict_result.py"),
        "--run-dir",
        str(run_dir),
        "--split-summary",
        str(split_summary),
        "--require-no-patient-overlap",
        "--min-test-accuracy",
        str(args.min_test_accuracy),
        "--min-test-macro-f1",
        str(args.min_test_macro_f1),
    ])

    if not args.skip_derived:
        run_command([
            python,
            str(script_dir / "evaluate_vepiset_patient_proxy.py"),
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(run_dir / "patient_proxy_metrics_mean.json"),
            "--aggregation",
            "mean",
            "--selector",
            "balanced_patient_accuracy",
        ])
        run_command([
            python,
            str(script_dir / "evaluate_vepiset_patient_proxy.py"),
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(run_dir / "patient_proxy_metrics_top20.json"),
            "--aggregation",
            "top_frac",
            "--top-frac",
            "0.2",
            "--selector",
            "balanced_patient_accuracy",
        ])
        run_command([
            python,
            str(script_dir / "evaluate_vepiset_patient_proxy.py"),
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(run_dir / "patient_proxy_metrics_mean_positive_selector.json"),
            "--aggregation",
            "mean",
            "--selector",
            "positive_hit_accuracy",
        ])
        run_command([
            python,
            str(script_dir / "evaluate_vepiset_positive_localization.py"),
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(run_dir / "positive_localization_metrics.json"),
        ])
        run_command([
            python,
            str(project_root / "code" / "models" / "calibrate_vepiset_positive_spatial_bias.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(positive_spatial_bias_run),
            "--selector",
            "weighted_f1",
            "--selection-aggregation",
            "loo_mean",
            "--min-val-accuracy",
            "0.70",
            "--min-val-macro-f1",
            "0.70",
            "--max-abs-bias",
            "0.1",
            "--steps",
            "0.05,0.025,0.01",
            "--passes-per-step",
            "8",
        ])
        run_command([
            python,
            str(script_dir / "evaluate_vepiset_positive_localization.py"),
            "--run-dir",
            str(positive_spatial_bias_run),
            "--output-json",
            str(positive_spatial_bias_run / "positive_localization_metrics.json"),
        ])
        run_command([
            python,
            str(script_dir / "audit_vepiset_strict_result.py"),
            "--run-dir",
            str(positive_spatial_bias_run),
            "--split-summary",
            str(split_summary),
            "--require-no-patient-overlap",
            "--min-test-accuracy",
            "0.0",
            "--min-test-macro-f1",
            "0.0",
        ])

    run_command([
        python,
        str(script_dir / "audit_vepiset_baseline_claims.py"),
        "--main-run",
        str(run_dir),
        "--output-json",
        str(run_dir / "baseline_claim_audit.json"),
        "--min-accuracy",
        str(args.min_test_accuracy),
        "--min-macro-f1",
        str(args.min_test_macro_f1),
    ])

    run_command([
        python,
        str(script_dir / "summarize_vepiset_strict_main.py"),
        "--run-dir",
        str(run_dir),
        "--base-run-dir",
        str(base_run_dir),
        "--output-json",
        str(run_dir / "strict_main_summary.json"),
        "--positive-spatial-bias-run",
        str(positive_spatial_bias_run),
        "--min-test-accuracy",
        str(args.min_test_accuracy),
        "--min-test-macro-f1",
        str(args.min_test_macro_f1),
    ])

    summary_path = run_dir / "strict_main_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = summary.get("requirement_checks", {})
    passed = bool(checks.get("proxy_requirements_met", False))
    print(json.dumps({
        "summary": str(summary_path),
        "test_window_metrics": summary.get("test_window_metrics", {}),
        "patient_level_proxy": summary.get("patient_level_proxy", {}),
        "ied_positive_oracle_localization": summary.get("ied_positive_oracle_localization", {}),
        "conditional_positive_spatial_bias": summary.get("secondary_operating_points", {}).get(
            "conditional_positive_spatial_bias",
            {},
        ),
        "requirement_checks": checks,
    }, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
