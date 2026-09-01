#!/usr/bin/env python3
"""Machine-audit the current-data trustworthy SOZ submission package.

The audit verifies published result identities, mandatory negative findings,
private-report coverage, and the distinction between a complete developmental
method-audit package and missing clinical confirmation.  It performs no model
training, target evaluation, thresholding, report generation, or claim repair.
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
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "trustworthy_soz_current_data_submission_package_v45"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_current_data_submission_package_v45_20260816"


PATHS = {
    "private_v36": ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816/result.json",
    "method_v37": ROOT / "outputs/trustworthy_soz_current_data_method_audit_v37_20260816/result.json",
    "D_stress_v38": ROOT / "outputs/trustworthy_soz_labram_v29_token_stress_v38_20260816/result.json",
    "reports_v39": ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816/manifest.json",
    "cases_v40": ROOT / "outputs/trustworthy_soz_private_case_audit_v40_20260816/manifest.json",
    "report_mutation_v41": ROOT / "outputs/trustworthy_soz_reporting_mutation_audit_v41_20260816/result.json",
    "margin_v42": ROOT / "outputs/trustworthy_soz_v29_margin_transport_v42_20260816/result.json",
    "H_stress_v43": ROOT / "outputs/trustworthy_soz_labram_v29_h_carrier_stress_v43_20260816/result.json",
    "candidate_reliance_v44": ROOT / "outputs/trustworthy_soz_v29_candidate_channel_reliance_v44_20260816/result.json",
    "M_threshold": ROOT / "outputs/tuev_morphology_oof_thresholds_v1_20260810/threshold_selection.json",
    "M_recovery": ROOT / "outputs/labram_morphology_hierarchical_recovery_oof_v1_20260810/paired_development_summary_v1/development_summary.json",
    "I_native": ROOT / "outputs/tusz_ictal_formal_v5_i_dev_20260810/i_dev_result.json",
    "V_F_native": ROOT / "outputs/labram_temporal_future_qualification_v1_20260816/result.json",
    "top_tier_readiness_v35": ROOT / "outputs/trustworthy_soz_top_tier_readiness_v35_20260816/result.json",
    "carrier_figure_v44": ROOT / "figures/trustworthy_soz_carrier_reliance_v44_20260816.pdf",
    "private_figure_v36": ROOT / "figures/trustworthy_soz_private_transfer_v36_20260816.pdf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _check(condition: bool, name: str, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(condition), "detail": detail}


def audit() -> tuple[dict[str, object], list[dict[str, object]]]:
    for path in PATHS.values():
        path.resolve(strict=True)
    payload = {
        name: _json(path)
        for name, path in PATHS.items()
        if path.suffix == ".json"
    }
    v36 = payload["private_v36"]
    v37 = payload["method_v37"]
    v38 = payload["D_stress_v38"]
    v39 = payload["reports_v39"]
    v40 = payload["cases_v40"]
    v41 = payload["report_mutation_v41"]
    v42 = payload["margin_v42"]
    v43 = payload["H_stress_v43"]
    v44 = payload["candidate_reliance_v44"]
    m_threshold = payload["M_threshold"]
    m_recovery = payload["M_recovery"]
    i_native = payload["I_native"]
    vf_native = payload["V_F_native"]
    readiness = payload["top_tier_readiness_v35"]

    private_summary = v36["frozen_arms"]["v29_equal_H_D"]
    checks = [
        _check(
            v36.get("status")
            == "completed_frozen_post_open_private_publication_audit",
            "private_frozen_transport_status",
            "v36 remains post-open descriptive transport",
        ),
        _check(
            private_summary["event_count"] == 51
            and private_summary["patient_count"] == 23
            and private_summary["event_micro"]["strict"] == 25 / 51
            and private_summary["event_micro"]["relaxed"] == 38 / 51,
            "private_primary_counts_and_metrics",
            "51 events/23 clusters; strict 25/51; neighborhood-4 38/51",
        ),
        _check(
            v37.get("status") == "completed_frozen_posthoc_development_method_audit"
            and v37.get("public_patient_count") == 102,
            "public_method_audit_status",
            "v37 is the same-roster 102-patient consumed-development audit",
        ),
        _check(
            v38.get("status") == "completed_frozen_D_token_reliance_stress"
            and v38["interpretation_boundary"]["H_carrier_perturbed"] is False
            and v38["interpretation_boundary"]["raw_EEG_causal_intervention"] is False,
            "D_stress_scope",
            "v38 perturbs cached D only and is not a raw-EEG explanation",
        ),
        _check(
            v39.get("report_count") == 88
            and v39.get("candidate_profile") == "v29_equal_H_D_probability_ensemble"
            and v39["access_receipt"]["private_target_or_error_audit_loaded"] is False,
            "v29_report_binding",
            "all 88 target-blind private events use the v29 candidate profile",
        ),
        _check(
            v40.get("case_count") == 4
            and set(v40.get("strata", ()))
            == {"exact", "neighbor_only", "contralateral_far", "known_spread_top1"}
            and v40["claim_boundary"]["target_blind_case_sample"] is False,
            "private_failure_case_coverage",
            "four outcome-stratified private cases include success and dangerous failures",
        ),
        _check(
            v41.get("mutation_attempts") == 2280
            and v41.get("unsafe_escape_count") == 0
            and v41.get("status") == "PASS_ALL_PRESPECIFIED_MUTATIONS_REJECTED",
            "machine_report_mutation_safety",
            "0 unsafe publishes among 2,280 prespecified mutations",
        ),
        _check(
            v42.get("status") == "NO_CLINICAL_RISK_QUALIFICATION"
            and v42["qualification_checks"]["private_margin_transport_qualified"]
            is False,
            "margin_negative_qualification",
            "margin remains unavailable as clinical confidence or transportable abstention",
        ),
        _check(
            v43.get("status")
            == "completed_frozen_H_carrier_public_private_reliance_stress"
            and v43["interpretation_boundary"]["clinical_phase_concept_qualified"]
            is False
            and max(
                v43["identity_replay_max_absolute_probability_difference"].values()
            )
            < 1e-6,
            "H_stress_scope_and_replay",
            "v43 replays public/private identity and does not qualify phase concepts",
        ),
        _check(
            v44.get("status")
            == "completed_candidate_specific_cached_carrier_reliance_audit"
            and math.isclose(
                v44["public"]["top1_content_removed_stability"]["top1_retention"],
                8 / 102,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            and math.isclose(
                v44["private"]["top1_content_removed_stability_all_88"][
                    "top1_retention"
                ],
                3 / 88,
                rel_tol=0.0,
                abs_tol=1e-7,
            ),
            "candidate_specific_reliance",
            "Top-1 content removal retains 8/102 public and 3/88 private rankings",
        ),
        _check(
            m_threshold.get("selection_status") == "NO_GO"
            and m_recovery["metrics"]["recovery_decision"]["decision"] == "NO_GO"
            and m_recovery.get("soz_reasoner_authorized") is False,
            "M_structural_absence",
            "morphology native threshold and one recovery both remain NO_GO",
        ),
        _check(
            i_native["decision"]["passed"] is False
            and i_native["decision"]["selected_head"] is None
            and i_native["decision"]["failure_consequence"]
            == "remove_ictal_family_no_third_iteration",
            "I_structural_absence",
            "ictal-involvement native promotion remains failed and absent",
        ),
        _check(
            vf_native.get("status") == "NO_GO"
            and vf_native["safety"]["soz_targets_loaded"] is False
            and vf_native["safety"]["private_data_loaded"] is False
            and vf_native["downstream_authorization"][
                "soz_reasoner_ingestion_authorized"
            ]
            is False,
            "V_F_structural_absence",
            "learned temporal-future concept remains NO_GO without SOZ target access",
        ),
        _check(
            readiness.get("top_tier_submission_ready") is False
            and readiness.get("submission_status") == "NOT_READY"
            and set(readiness.get("blocking_gates", ()))
            >= {
                "independent_clinician_report_qualification",
                "clinical_workflow_and_automation_bias_evidence",
                "label_fresh_risk_calibration",
                "label_fresh_one_shot_confirmation",
            },
            "mandatory_clinical_limitations_preserved",
            "fresh confirmation and clinician evidence remain missing/pending",
        ),
    ]

    report_root = ROOT / "outputs/trustworthy_soz_v29_research_reports_v39_20260816"
    html_count = len(list((report_root / "html/private_event").glob("*.html")))
    waveform_count = len(
        list((report_root / "waveforms/private_event").glob("*.png"))
    )
    checks.append(
        _check(
            html_count == 88 and waveform_count == 88,
            "private_report_and_waveform_materialization",
            f"private HTML={html_count}; waveform PNG={waveform_count}",
        )
    )
    if not all(bool(row["passed"]) for row in checks):
        failed = [row["check"] for row in checks if not bool(row["passed"])]
        raise RuntimeError(f"submission package audit failed: {failed}")

    artifacts = [
        {
            "claim_id": "C1_PRIVATE_TRANSPORT",
            "role": "main_private_result",
            "artifact": "private_v36",
            "allowed_wording": "frozen post-open cross-domain/cross-granularity transport",
            "forbidden_wording": "fresh external confirmation",
        },
        {
            "claim_id": "C2_PUBLIC_INCREMENT",
            "role": "consumed_development_method_audit",
            "artifact": "method_v37",
            "allowed_wording": "v29 exceeds a fold-local signal-free spatial prior",
            "forbidden_wording": "confirmed superiority over DeepSOZ",
        },
        {
            "claim_id": "C3_D_RELIANCE",
            "role": "cached_D_token_stress",
            "artifact": "D_stress_v38",
            "allowed_wording": "D uses channel-local and lateralized cached token content",
            "forbidden_wording": "D is a qualified temporal-evolution concept",
        },
        {
            "claim_id": "C4_V29_REPORTS",
            "role": "private_research_communication",
            "artifact": "reports_v39",
            "allowed_wording": "88 v29-bound target-blind research reports with waveforms",
            "forbidden_wording": "clinically validated automated diagnostic reports",
        },
        {
            "claim_id": "C5_PRIVATE_FAILURES",
            "role": "reviewer_facing_post_open_failure_analysis",
            "artifact": "cases_v40",
            "allowed_wording": "outcome-stratified exact/neighbor/contra-far/spread cases",
            "forbidden_wording": "target-blind representative case sample",
        },
        {
            "claim_id": "C6_REPORT_FIREWALL",
            "role": "machine_safety_audit",
            "artifact": "report_mutation_v41",
            "allowed_wording": "0/2,280 prespecified mutations escaped the validator",
            "forbidden_wording": "zero clinical hallucination risk",
        },
        {
            "claim_id": "C7_MARGIN_NO_GO",
            "role": "negative_risk_qualification",
            "artifact": "margin_v42",
            "allowed_wording": "margin did not transport as a qualified risk selector",
            "forbidden_wording": "calibrated clinical confidence or safe abstention",
        },
        {
            "claim_id": "C8_H_RELIANCE",
            "role": "public_private_cached_H_stress",
            "artifact": "H_stress_v43",
            "allowed_wording": "H uses channel-local lateralized representation content",
            "forbidden_wording": "H phase blocks are qualified onset/evolution concepts",
        },
        {
            "claim_id": "C9_CANDIDATE_RELIANCE",
            "role": "candidate_specific_cached_representation_intervention",
            "artifact": "candidate_reliance_v44",
            "allowed_wording": "ranking relies on the selected channel's cached H/D content",
            "forbidden_wording": "specific raw waveform intervals causally explain the ranking",
        },
        {
            "claim_id": "C10_NEGATIVE_CONCEPTS",
            "role": "qualification_fail_closed_evidence",
            "artifact": "M_threshold+M_recovery+I_native+V_F_native",
            "allowed_wording": "M/I/V_F failed native qualification and are structurally absent",
            "forbidden_wording": "complete three-concept SOZ reasoning",
        },
        {
            "claim_id": "C11_CLINICAL_CEILING",
            "role": "mandatory_limitation",
            "artifact": "top_tier_readiness_v35",
            "allowed_wording": "development-stage method audit",
            "forbidden_wording": "clinically validated trustworthy SOZ system",
        },
    ]
    for row in artifacts:
        names = str(row["artifact"]).split("+")
        paths = [PATHS[name] for name in names]
        row["paths"] = " | ".join(str(path.relative_to(ROOT)) for path in paths)
        row["sha256"] = " | ".join(_sha256(path) for path in paths)

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "DEVELOPMENT_METHOD_AUDIT_PACKAGE_READY_WITH_MANDATORY_LIMITATIONS",
        "current_data_method_audit_package_ready": True,
        "top_tier_clinical_validation_ready": False,
        "private_result_is_main_section": True,
        "submission_identity": (
            "qualification-aware fail-closed scalp-electrode SOZ-reference "
            "candidate ranking with post-open private transport and fact-locked reporting"
        ),
        "checks": checks,
        "artifact_count": len(artifacts),
        "blocking_evidence_not_fabricated": {
            "label_fresh_same_endpoint_confirmation": "MISSING",
            "label_fresh_risk_calibration": "MISSING",
            "independent_clinician_report_qualification": "PENDING",
            "clinical_workflow_automation_bias_outcomes": "PENDING",
            "raw_EEG_candidate_specific_causal_intervention": "NOT_PERFORMED",
        },
        "claim_boundary": {
            "public_is_consumed_development": True,
            "private_is_post_open_transport": True,
            "M_I_V_F_are_structurally_absent": True,
            "direct_V_is_description_only": True,
            "margin_is_clinically_qualified": False,
            "reports_are_clinically_validated": False,
            "output_is_cortical_SOZ_EZ_or_surgical_target": False,
        },
        "access_receipt": {
            "existing_published_results_loaded": True,
            "raw_EEG_loaded": False,
            "model_or_target_tensor_loaded": False,
            "training_or_inference_performed": False,
            "threshold_model_report_or_claim_selected": False,
            "reader_or_label_evidence_generated": False,
        },
    }
    return result, artifacts


def publish(
    output: Path,
    result: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
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
        with (staging / "claim_artifact_index.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(artifacts[0]))
            writer.writeheader()
            writer.writerows(artifacts)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result, artifacts = audit()
    output = publish(args.output, result, artifacts)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "method_audit_ready": result["current_data_method_audit_package_ready"],
                "clinical_validation_ready": result["top_tier_clinical_validation_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
