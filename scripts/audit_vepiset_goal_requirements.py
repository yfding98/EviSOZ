#!/usr/bin/env python3
"""Audit the original VEPiSet goal against the strict main summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_SUMMARY = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_main_patientprior_conservative_macro_valacc87"
    "/strict_main_summary.json"
)
DEFAULT_BACKBONE_INIT = Path(
    "outputs/vepiset_ied_v2_full6_seed2026_patientclasssplit_logitadj025_macroselect_noamp20"
    "/vepiset_v2_backbone_init.pt"
)
DEFAULT_SOZ_DRYRUN = Path("outputs/vepiset_v2_backbone_soz_dryrun.json")
DEFAULT_TRAINER_SMOKE = Path("outputs/vepiset_v2_soz_trainer_smoke")
DEFAULT_TRAINER_SMOKE_SCRIPT = Path("scripts/run_vepiset_v2_soz_trainer_smoke.sh")
DEFAULT_SOTA_COMPARABILITY = Path("outputs/vepiset_sota_comparability_audit.json")


def _status(condition: bool, evidence: str) -> Dict[str, Any]:
    return {
        "met": bool(condition),
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--backbone-init", default=str(DEFAULT_BACKBONE_INIT))
    parser.add_argument("--soz-dryrun", default=str(DEFAULT_SOZ_DRYRUN))
    parser.add_argument("--trainer-smoke-dir", default=str(DEFAULT_TRAINER_SMOKE))
    parser.add_argument("--trainer-smoke-script", default=str(DEFAULT_TRAINER_SMOKE_SCRIPT))
    parser.add_argument("--sota-comparability", default=str(DEFAULT_SOTA_COMPARABILITY))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument(
        "--require-full-original-goal",
        action="store_true",
        help="Exit non-zero unless the clinical SOZ SOTA portion is also supported.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    metrics = summary.get("test_window_metrics", {})
    checks = summary.get("requirement_checks", {})
    baseline = summary.get("baseline_claim_audit", {}).get("majority_baseline", {})
    secondary = summary.get("secondary_operating_points", {}).get("recommended_high_accuracy", {})
    backbone_init = Path(args.backbone_init)
    soz_dryrun = Path(args.soz_dryrun)
    trainer_smoke_dir = Path(args.trainer_smoke_dir)
    trainer_smoke_script = Path(args.trainer_smoke_script)
    sota_comparability_path = Path(args.sota_comparability)
    dryrun_payload: Dict[str, Any] = {}
    if soz_dryrun.exists():
        dryrun_payload = json.loads(soz_dryrun.read_text(encoding="utf-8"))
    trainer_smoke_config: Dict[str, Any] = {}
    trainer_smoke_log = trainer_smoke_dir / "train.log"
    if (trainer_smoke_dir / "config.json").exists():
        trainer_smoke_config = json.loads((trainer_smoke_dir / "config.json").read_text(encoding="utf-8"))
    trainer_smoke_log_text = trainer_smoke_log.read_text(encoding="utf-8") if trainer_smoke_log.exists() else ""
    trainer_smoke_passed = (
        trainer_smoke_config.get("model_arch") == "v2_lightweight"
        and bool(trainer_smoke_config.get("init_soz_ckpt"))
        and "loaded=166" in trainer_smoke_log_text
        and "unexpected=0" in trainer_smoke_log_text
        and "Done." in trainer_smoke_log_text
    )
    sota_comparability: Dict[str, Any] = {}
    if sota_comparability_path.exists():
        sota_comparability = json.loads(sota_comparability_path.read_text(encoding="utf-8"))

    accuracy = float(metrics.get("accuracy", 0.0))
    patient_disjoint = bool(checks.get("patient_disjoint_split", False))
    lineage_ok = bool(checks.get("lineage_requirements_met", False))
    proxy_met = bool(checks.get("proxy_requirements_met", False))
    clinical_sota_supported = bool(checks.get("clinical_soz_sota_claim_supported", False))

    requirements = {
        "uses_integration_model_v2_core": _status(
            str(summary.get("model_core", "")) == "code/models/integration_model_v2.py",
            str(summary.get("model_core", "")),
        ),
        "uses_requested_vepiset_dataset": _status(
            str(summary.get("dataset_root", ""))
            == "/mnt/hd1/dyf/dataset/vepiset-dataset/opensource-dataset",
            str(summary.get("dataset_root", "")),
        ),
        "task_is_ied_classification_proxy": _status(
            str(summary.get("training_script", "")).endswith("train_vepiset_ied_v2.py"),
            "six-class VEPiSet IED spatial-distribution proxy classification",
        ),
        "test_accuracy_at_least_threshold": _status(
            accuracy >= float(args.min_accuracy),
            f"test accuracy={accuracy:.4f}, threshold={float(args.min_accuracy):.4f}",
        ),
        "no_patient_leakage": _status(
            patient_disjoint,
            json.dumps(summary.get("patient_overlap", {}), ensure_ascii=False),
        ),
        "validation_only_selection_and_calibration": _status(
            lineage_ok,
            json.dumps(summary.get("validation_selection", {}), ensure_ascii=False),
        ),
        "clinical_scalp_soz_sota_supported": _status(
            clinical_sota_supported,
            checks.get(
                "reason_clinical_soz_sota_not_supported",
                "No clinical SOZ ground-truth support recorded.",
            ),
        ),
    }

    audit = {
        "summary": str(summary_path),
        "strict_main_run": summary.get("run_dir", ""),
        "test_window_metrics": metrics,
        "majority_baseline": baseline,
        "secondary_high_accuracy_operating_point": secondary,
        "clinical_transfer_readiness": {
            "v2_backbone_init_exists": backbone_init.exists(),
            "v2_backbone_init": str(backbone_init),
            "soz_interface_dryrun_exists": soz_dryrun.exists(),
            "soz_interface_dryrun": str(soz_dryrun),
            "soz_interface_dryrun_passed": bool(dryrun_payload.get("dryrun_passed", False)),
            "v2_trainer_smoke_exists": trainer_smoke_dir.exists(),
            "v2_trainer_smoke": str(trainer_smoke_dir),
            "v2_trainer_smoke_script_exists": trainer_smoke_script.exists(),
            "v2_trainer_smoke_script": str(trainer_smoke_script),
            "v2_trainer_smoke_passed": bool(trainer_smoke_passed),
            "clinical_soz_claim_supported_by_dryrun": bool(
                dryrun_payload.get("clinical_soz_claim_supported", False)
            ),
        },
        "sota_comparability_audit": {
            "exists": sota_comparability_path.exists(),
            "path": str(sota_comparability_path),
            "numeric_accuracy_above_max_reference_accuracy": bool(
                sota_comparability.get("numeric_accuracy_above_max_reference_accuracy", False)
            ),
            "apples_to_apples_comparison": bool(
                sota_comparability.get("apples_to_apples_comparison", False)
            ),
            "clinical_soz_sota_claim_supported": bool(
                sota_comparability.get("clinical_soz_sota_claim_supported", False)
            ),
            "max_reference_accuracy": sota_comparability.get("max_reference_accuracy"),
        },
        "requirements": requirements,
        "proxy_goal_met": proxy_met,
        "full_original_goal_met": all(item["met"] for item in requirements.values()),
        "recommended_claim": summary.get("recommended_claim", ""),
        "clinical_claim_boundary": {
            "supported": clinical_sota_supported,
            "reason": checks.get("reason_clinical_soz_sota_not_supported", ""),
        },
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.require_full_original_goal and not audit["full_original_goal_met"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
