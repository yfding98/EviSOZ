#!/usr/bin/env python3
"""Machine-readable audit for VEPiSet proxy vs scalp-EEG SOZ claims.

This script intentionally does not certify clinical SOZ SOTA.  It records the
current VEPiSet proxy metrics, a small set of scalp-EEG SOZ localization
reference results, and the task mismatch that prevents an apples-to-apples
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SUMMARY = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87"
    "/strict_main_summary.json"
)
DEFAULT_OUTPUT = Path("outputs/vepiset_sota_comparability_audit.json")


SCALP_SOZ_REFERENCES: List[Dict[str, Any]] = [
    {
        "name": "DeepSOZ",
        "citation": (
            "Shama et al., DeepSOZ: A Robust Deep Model for Joint Temporal and "
            "Spatial Seizure Onset Localization from Multichannel EEG Data, MICCAI 2023"
        ),
        "url": "https://link.springer.com/chapter/10.1007/978-3-031-43993-3_18",
        "task": "clinical scalp-EEG seizure onset localization from seizure recordings",
        "dataset": "Temple University Hospital corpus subset, 120 patients",
        "metrics": {
            "seizure_level_accuracy": 0.731,
            "patient_level_accuracy": 0.744,
        },
        "notes": "Reported in the project audit from the paper's SOZ localization table.",
    },
    {
        "name": "SZLoc",
        "citation": (
            "Craley et al., SZLoc: A Multi-resolution Architecture for Automated "
            "Epileptic Seizure Localization from Scalp EEG, MIDL 2022"
        ),
        "url": "https://proceedings.mlr.press/v172/craley22a.html",
        "task": "clinical scalp-EEG seizure localization from seizure recordings",
        "dataset": "JHH 34 focal epilepsy patients; UWM 16-patient generalization cohort",
        "metrics": {
            "jhh_patient_level_accuracy": 24.2 / 34.0,
            "jhh_seizure_level_accuracy": 109.6 / 201.0,
            "uwm_patient_level_accuracy_all_loss": 6.4 / 16.0,
            "uwm_patient_level_accuracy_electrode_loss": 6.7 / 16.0,
        },
        "notes": "Reported in the project audit from the paper's localization tables.",
    },
]


def _all_reference_accuracies() -> Dict[str, float]:
    values: Dict[str, float] = {}
    for ref in SCALP_SOZ_REFERENCES:
        for key, value in ref["metrics"].items():
            if key.endswith("accuracy"):
                values[f"{ref['name']}::{key}"] = float(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--require-clinical-sota",
        action="store_true",
        help="Exit non-zero unless the current result can support a clinical SOZ SOTA claim.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    metrics = summary.get("test_window_metrics", {})
    checks = summary.get("requirement_checks", {})
    baseline = summary.get("baseline_claim_audit", {}).get("majority_baseline", {})
    patient_proxy = summary.get("patient_level_proxy", {})
    positive_oracle = summary.get("ied_positive_oracle_localization", {})

    vep_accuracy = float(metrics.get("accuracy", 0.0))
    reference_accuracies = _all_reference_accuracies()
    max_reference_accuracy = max(reference_accuracies.values()) if reference_accuracies else 0.0

    task_mismatch_reasons = [
        "VEPiSet labels are interictal IED spatial-distribution labels, not clinical SOZ ground truth.",
        "VEPiSet evaluates 4-second interictal windows with a dominant Non-IED class.",
        "Clinical SOZ references evaluate seizure-level or patient-level localization from ictal scalp EEG.",
        "The VEPiSet main raw accuracy does not beat the all-Non-IED majority baseline on the strict test split.",
        "Positive-window localization audits condition on true IED-positive windows and are not deployable clinical SOZ metrics.",
    ]

    clinical_sota_supported = False
    audit = {
        "summary": str(summary_path),
        "vepiset_task": "six-class VEPiSet IED spatial-distribution proxy classification",
        "vepiset_metrics": metrics,
        "vepiset_majority_baseline": baseline,
        "vepiset_patient_proxy": patient_proxy,
        "vepiset_positive_oracle": positive_oracle,
        "references": SCALP_SOZ_REFERENCES,
        "reference_accuracy_values": reference_accuracies,
        "numeric_accuracy_above_max_reference_accuracy": vep_accuracy > max_reference_accuracy,
        "max_reference_accuracy": max_reference_accuracy,
        "apples_to_apples_comparison": False,
        "clinical_soz_sota_claim_supported": clinical_sota_supported,
        "clinical_soz_sota_claim_supported_by_summary": bool(
            checks.get("clinical_soz_sota_claim_supported", False)
        ),
        "task_mismatch_reasons": task_mismatch_reasons,
        "allowed_claim": (
            "Strict patient-disjoint VEPiSet IED spatial-distribution proxy performance "
            f"with accuracy={vep_accuracy:.4f}, macro-F1={float(metrics.get('macro_f1', 0.0)):.4f}, "
            f"weighted-F1={float(metrics.get('weighted_f1', 0.0)):.4f}."
        ),
        "forbidden_claims": [
            "This is scalp-EEG clinical SOZ SOTA.",
            "The model beats DeepSOZ/SZLoc on clinical SOZ localization.",
            "VEPiSet proves clinical seizure onset zone localization performance.",
        ],
    }

    output_path = Path(args.output_json)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.require_clinical_sota and not clinical_sota_supported:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
