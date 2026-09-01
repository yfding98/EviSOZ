from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .label_fresh_confirmation import (
    ConfirmationContractError,
    validate_a5_result,
    validate_prediction_seal,
    validate_s1c_receipt,
)


class ReadinessContractError(ValueError):
    """Raised when the frozen publication contract is internally inconsistent."""


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: str
    evidence: Mapping[str, Any]
    consequence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "evidence": dict(self.evidence),
            "consequence": self.consequence,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReadinessContractError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float, *, tolerance: float = 1e-9) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def _reader_completion(path: Path) -> dict[str, int]:
    total = 0
    completed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get("review_status") == "completed" and row.get("review_completed_at"):
                completed += 1
    return {"completed": completed, "total": total}


def _require_contract_shape(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "trustworthy_soz_top_tier_confirmation_contract_v35":
        raise ReadinessContractError("Unexpected top-tier confirmation contract schema")

    roles = contract.get("dataset_roles", {})
    if roles.get("public_102") != "consumed_adaptive_development_only":
        raise ReadinessContractError("The consumed public cohort cannot be relabelled as confirmation")
    if roles.get("private_51_events_23_patients") != "post_open_exploratory_transport_only":
        raise ReadinessContractError("The opened private cohort cannot be relabelled as confirmation")

    primary = contract.get("endpoints", {}).get("primary", {})
    if primary.get("name") != "strict_positive_set_top1" or primary.get("unit") != "patient_equal":
        raise ReadinessContractError("Strict patient-equal Top-1 must remain the primary endpoint")

    concepts = contract.get("concept_state_machine", {}).get("families", {})
    for family in ("M_morphology", "I_ictal_involvement", "V_F_learned_future"):
        if concepts.get(family, {}).get("localization_access") is not False:
            raise ReadinessContractError(f"Failed concept may not access localization: {family}")
    if concepts.get("V_direct_observable", {}).get("localization_access") is not False:
        raise ReadinessContractError("Descriptive V may not be presented as a localization input")

    reporting = contract.get("reporting_contract", {})
    if reporting.get("patient_facts_mutable_by_llm") is not False:
        raise ReadinessContractError("LLM may not mutate patient facts")
    if reporting.get("localization_mutable_by_llm") is not False:
        raise ReadinessContractError("LLM may not mutate localization")


def audit_top_tier_readiness(
    *,
    workspace: Path,
    contract_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    contract_path = contract_path.resolve()
    contract = _load_json(contract_path)
    _require_contract_shape(contract)
    evidence_paths = contract["readiness_evidence"]

    def resolve(key: str) -> Path | None:
        raw = evidence_paths.get(key)
        return None if raw is None else (workspace / raw).resolve()

    gates: list[GateResult] = []

    v29_path = resolve("development_ranker")
    if v29_path is None or not v29_path.is_file():
        raise ReadinessContractError("Missing frozen v29 development artifact")
    v29 = _load_json(v29_path)
    v29_top1 = v29["metrics"]["portable_equal_ensemble"]["top1"]
    v29_ranking = v29["metrics"]["portable_equal_ensemble"]["ranking"]
    expected = contract["frozen_profiles"]["development_ranking"]
    v29_ok = (
        v29.get("status") == "completed_public_adaptive_exploratory_oof_freeze"
        and int(v29_ranking["n_patients"]) == int(expected["n_patients"])
        and _close(v29_top1["strict_accuracy"], expected["strict_top1"])
        and _close(v29_top1["relaxed_accuracy"], expected["neighborhood4_top1"])
    )
    gates.append(
        GateResult(
            "development_ranker_replay",
            "PASS" if v29_ok else "FAIL",
            {
                "artifact": str(v29_path.relative_to(workspace)),
                "artifact_sha256": _sha256(v29_path),
                "n_patients": int(v29_ranking["n_patients"]),
                "strict_top1": float(v29_top1["strict_accuracy"]),
                "neighborhood4_top1": float(v29_top1["relaxed_accuracy"]),
                "role": "consumed_development_only",
            },
            "retain_as_development_result_only" if v29_ok else "stop_release_and_reconcile_v29_lineage",
        )
    )

    private_path = resolve("private_exploratory_transport")
    if private_path is None or not private_path.is_file():
        raise ReadinessContractError("Missing private exploratory result")
    private = _load_json(private_path)
    private_ok = (
        private.get("status") == "completed_post_open_exploratory_private_evaluation"
        and private.get("claim_boundary", {}).get("private_has_been_opened_in_prior_project_iterations") is True
        and private.get("claim_boundary", {}).get("fresh_external_confirmation") is False
        and private.get("access_receipt", {}).get("private_used_for_training_weight_threshold_fold_or_model_selection") is False
    )
    private_metrics = private["private"]["metrics"]
    gates.append(
        GateResult(
            "private_role_integrity",
            "PASS" if private_ok else "FAIL",
            {
                "artifact": str(private_path.relative_to(workspace)),
                "event_count": int(private_metrics["event_count"]),
                "patient_count": int(private_metrics["patient_count"]),
                "role": "post_open_exploratory_transport_only",
            },
            "exclude_from_confirmation_claims" if private_ok else "stop_release_and_audit_private_access",
        )
    )

    morphology_path = resolve("morphology_native_evaluation")
    ictal_path = resolve("ictal_native_qualification")
    future_path = resolve("temporal_future_qualification")
    if any(path is None or not path.is_file() for path in (morphology_path, ictal_path, future_path)):
        raise ReadinessContractError("Missing at least one native concept qualification artifact")
    assert morphology_path is not None and ictal_path is not None and future_path is not None
    morphology = _load_json(morphology_path)
    ictal = _load_json(ictal_path)
    future = _load_json(future_path)
    morphology_spsw_f1 = float(dict((row[0], row[4]) for row in morphology["class_metrics"])["SPSW"])
    concept_ok = (
        morphology_spsw_f1 < 0.8
        and ictal.get("decision", {}).get("passed") is False
        and ictal.get("current_soz_reasoner_I_authorized") is False
        and future.get("status") == "NO_GO"
    )
    gates.append(
        GateResult(
            "failed_concept_structural_isolation",
            "PASS" if concept_ok else "FAIL",
            {
                "M_native_SPSW_f1": morphology_spsw_f1,
                "I_native_passed": ictal.get("decision", {}).get("passed"),
                "I_localization_authorized": ictal.get("current_soz_reasoner_I_authorized"),
                "V_F_status": future.get("status"),
                "failed_branches_in_localization": [],
            },
            "retain_M_I_V_F_as_structurally_absent" if concept_ok else "stop_release_and_reconcile_concept_authorization",
        )
    )

    llm_path = resolve("constrained_llm_manifest")
    if llm_path is None or not llm_path.is_file():
        raise ReadinessContractError("Missing constrained-LLM release manifest")
    llm = _load_json(llm_path)
    llm_counts = llm.get("counts", {})
    llm_access = llm.get("access_receipt", {})
    llm_ok = (
        llm.get("status") == "completed_optional_constrained_language_layer"
        and int(llm_counts.get("qwen3.6_constrained_language_only", -1))
        + int(llm_counts.get("deterministic_fallback", -1))
        == 190
        and llm_access.get("soz_gold_labels_loaded") is False
        and llm_access.get("localization_changed") is False
    )
    gates.append(
        GateResult(
            "language_layer_machine_safety",
            "PASS" if llm_ok else "FAIL",
            {
                "qwen_published": llm_counts.get("qwen3.6_constrained_language_only"),
                "deterministic_fallback": llm_counts.get("deterministic_fallback"),
                "patient_fact_clinical_validation": "pending_reader_study",
            },
            "language_layer_remains_research_only_until_reader_qualification",
        )
    )

    reader_manifest_path = resolve("reader_study_manifest")
    reader_a_path = resolve("reader_a_annotations")
    reader_b_path = resolve("reader_b_annotations")
    if any(path is None or not path.is_file() for path in (reader_manifest_path, reader_a_path, reader_b_path)):
        raise ReadinessContractError("Missing reader-study preflight artifact")
    assert reader_manifest_path is not None and reader_a_path is not None and reader_b_path is not None
    reader_manifest = _load_json(reader_manifest_path)
    reader_a = _reader_completion(reader_a_path)
    reader_b = _reader_completion(reader_b_path)
    reader_complete = (
        reader_a["total"] > 0
        and reader_a["completed"] == reader_a["total"]
        and reader_b["completed"] == reader_b["total"]
        and reader_manifest.get("analysis_boundary", {}).get("independent_clinician_results_pending") is False
    )
    gates.append(
        GateResult(
            "independent_clinician_report_qualification",
            "PASS" if reader_complete else "PENDING",
            {"reader_a": reader_a, "reader_b": reader_b},
            "do_not_claim_clinically_qualified_report" if not reader_complete else "apply_frozen_report_gates",
        )
    )

    language_manifest_path = resolve("template_vs_qwen_reader_manifest")
    language_reader_a_path = resolve("template_vs_qwen_reader_a_annotations")
    language_reader_b_path = resolve("template_vs_qwen_reader_b_annotations")
    if any(
        path is None or not path.is_file()
        for path in (language_manifest_path, language_reader_a_path, language_reader_b_path)
    ):
        raise ReadinessContractError("Missing template-vs-Qwen reader-study preflight artifact")
    assert language_manifest_path is not None
    assert language_reader_a_path is not None
    assert language_reader_b_path is not None
    language_manifest = _load_json(language_manifest_path)
    language_access = language_manifest.get("access_receipt", {})
    language_contract = language_manifest.get("comparison_contract", {})
    language_preflight_ok = (
        language_manifest.get("status") == "empty_target_blind_two_reader_language_comparison_pack_ready"
        and language_manifest.get("counts", {}).get("cases") == 40
        and language_manifest.get("counts", {}).get("variant_a_template") == 20
        and language_manifest.get("counts", {}).get("variant_a_qwen") == 20
        and language_contract.get("same_locked_patient_facts") is True
        and language_contract.get("same_localization") is True
        and language_contract.get("same_knowledge_source_ids") is True
        and language_access.get("soz_gold_loaded") is False
        and language_access.get("prediction_correctness_loaded") is False
    )
    gates.append(
        GateResult(
            "template_vs_qwen_study_preflight",
            "PASS" if language_preflight_ok else "FAIL",
            {
                "cases": language_manifest.get("counts", {}).get("cases"),
                "variant_a_template": language_manifest.get("counts", {}).get("variant_a_template"),
                "variant_a_qwen": language_manifest.get("counts", {}).get("variant_a_qwen"),
                "target_or_correctness_used_for_sampling": False,
            },
            "obtain_two_independent_blinded_reader_results" if language_preflight_ok else "stop_and_rebuild_language_comparison_pack",
        )
    )
    language_reader_a = _reader_completion(language_reader_a_path)
    language_reader_b = _reader_completion(language_reader_b_path)
    language_result_path = resolve("template_vs_qwen_reader_result")
    language_result_complete = (
        language_reader_a["total"] > 0
        and language_reader_a["completed"] == language_reader_a["total"]
        and language_reader_b["completed"] == language_reader_b["total"]
        and language_result_path is not None
        and language_result_path.is_file()
    )
    gates.append(
        GateResult(
            "template_vs_qwen_clinician_evidence",
            "PASS" if language_result_complete else "PENDING",
            {
                "reader_a": language_reader_a,
                "reader_b": language_reader_b,
                "result": None if language_result_path is None else str(language_result_path),
            },
            "do_not_claim_Qwen_clinical_increment" if not language_result_complete else "apply_prespecified_paired_language_analysis",
        )
    )

    workflow_manifest_path = resolve("workflow_mrmc_manifest")
    workflow_annotation_paths = [
        resolve("workflow_mrmc_reader_a_annotations"),
        resolve("workflow_mrmc_reader_b_annotations"),
        resolve("workflow_mrmc_reader_c_annotations"),
    ]
    workflow_preflight_receipt_path = resolve("workflow_mrmc_preflight_receipt")
    workflow_server_receipt_path = resolve("workflow_mrmc_server_preflight_receipt")
    if workflow_manifest_path is None or not workflow_manifest_path.is_file() or any(
        path is None or not path.is_file() for path in workflow_annotation_paths
    ) or workflow_preflight_receipt_path is None or not workflow_preflight_receipt_path.is_file() or workflow_server_receipt_path is None or not workflow_server_receipt_path.is_file():
        raise ReadinessContractError("Missing workflow MRMC preflight artifact")
    workflow_manifest = _load_json(workflow_manifest_path)
    workflow_receipt = _load_json(workflow_preflight_receipt_path)
    workflow_server_receipt = _load_json(workflow_server_receipt_path)
    workflow_design = workflow_manifest.get("design", {})
    workflow_access = workflow_manifest.get("access_receipt", {})
    workflow_preflight_ok = (
        workflow_manifest.get("status") == "empty_three_reader_three_arm_workflow_pack_ready"
        and workflow_manifest.get("counts", {}).get("cases") == 27
        and workflow_manifest.get("counts", {}).get("readers") == 3
        and workflow_design.get("each_case_receives_all_three_arms_across_readers") is True
        and workflow_design.get("raw_phase_locked_before_intervention") is True
        and workflow_design.get("factuality_and_language_study_patient_overlap") is False
        and workflow_access.get("target_used_for_training_calibration_or_model_selection") is False
        and workflow_access.get("candidate_or_report_changed_after_target_read") is False
        and workflow_receipt.get("preflight_passed") is True
        and workflow_receipt.get("outcome_metrics_computed") is False
        and workflow_server_receipt.get("status") == "PASS"
        and workflow_server_receipt.get("server_enforced_raw_phase_lock") is True
        and workflow_server_receipt.get("target_bearing_allocation_absent_from_server_view") is True
    )
    gates.append(
        GateResult(
            "clinical_workflow_study_preflight",
            "PASS" if workflow_preflight_ok else "FAIL",
            {
                "cases": workflow_manifest.get("counts", {}).get("cases"),
                "readers": workflow_manifest.get("counts", {}).get("readers"),
                "linked_events": workflow_manifest.get("counts", {}).get("linked_events"),
                "per_reader_arm_counts": workflow_manifest.get("counts", {}).get("per_reader_arm_counts"),
                "server_enforced_raw_phase_lock": workflow_server_receipt.get("server_enforced_raw_phase_lock"),
            },
            "obtain_three_reader_phase_locked_results" if workflow_preflight_ok else "stop_and_rebuild_workflow_pack",
        )
    )
    workflow_completion = [
        _reader_completion(path) for path in workflow_annotation_paths if path is not None
    ]
    clinical_utility_result = resolve("clinical_utility_reader_result")
    clinical_utility_complete = (
        len(workflow_completion) == 3
        and all(item["total"] > 0 and item["completed"] == item["total"] for item in workflow_completion)
        and clinical_utility_result is not None
        and clinical_utility_result.is_file()
    )
    gates.append(
        GateResult(
            "clinical_workflow_and_automation_bias_evidence",
            "PASS" if clinical_utility_complete else "PENDING",
            {
                "readers": workflow_completion,
                "result": None if clinical_utility_result is None else str(clinical_utility_result),
            },
            "do_not_claim_time_saving_decision_improvement_or_safe_reliance" if not clinical_utility_complete else "report_prespecified_MRMC_results",
        )
    )

    s1c = resolve("S1_C_calibration_receipt")
    a5_sealed = resolve("A5_sealed_prediction_receipt")
    a5_result = resolve("A5_opened_confirmation_result")
    calibration_complete = False
    calibration_status = "MISSING"
    calibration_evidence: dict[str, Any] = {
        "S1_C_receipt": None if s1c is None else str(s1c)
    }
    s1c_value: dict[str, Any] | None = None
    if s1c is not None and s1c.is_file():
        try:
            s1c_value = _load_json(s1c)
            s1c_payload = validate_s1c_receipt(s1c_value, require_real=True)
            calibration_complete = s1c_value.get("status") == "QUALIFIED"
            calibration_status = "PASS" if calibration_complete else "FAIL"
            calibration_evidence.update(
                {
                    "artifact_sha256": _sha256(s1c),
                    "receipt_payload_sha256": s1c_value.get("receipt_payload_sha256"),
                    "evidence_class": s1c_payload.get("evidence_class"),
                    "calibration_status": s1c_value.get("status"),
                    "patient_count": s1c_payload.get("enrolled_patient_count"),
                }
            )
        except (ConfirmationContractError, KeyError, TypeError, ValueError) as exc:
            calibration_status = "FAIL"
            calibration_evidence["validation_error"] = str(exc)

    confirmation_complete = False
    confirmation_status = "MISSING"
    confirmation_evidence: dict[str, Any] = {
        "sealed_prediction_receipt": None if a5_sealed is None else str(a5_sealed),
        "opened_result": None if a5_result is None else str(a5_result),
    }
    if (
        a5_sealed is not None
        and a5_sealed.is_file()
        and a5_result is not None
        and a5_result.is_file()
        and s1c_value is not None
    ):
        try:
            a5_seal_value = _load_json(a5_sealed)
            a5_result_value = _load_json(a5_result)
            seal_payload = validate_prediction_seal(
                a5_seal_value, expected_role="A5", require_real=True
            )
            opened_payload = validate_a5_result(
                a5_result_value,
                prediction_seal=a5_seal_value,
                s1c_receipt=s1c_value,
                require_real=True,
            )
            confirmation_complete = calibration_complete
            confirmation_status = "PASS" if confirmation_complete else "FAIL"
            confirmation_evidence.update(
                {
                    "sealed_artifact_sha256": _sha256(a5_sealed),
                    "result_artifact_sha256": _sha256(a5_result),
                    "seal_payload_sha256": a5_seal_value.get("seal_payload_sha256"),
                    "result_payload_sha256": a5_result_value.get("result_payload_sha256"),
                    "evidence_class": seal_payload.get("evidence_class"),
                    "patient_count": seal_payload.get("patient_count"),
                    "primary_endpoint": opened_payload.get("primary_endpoint"),
                }
            )
        except (ConfirmationContractError, KeyError, TypeError, ValueError) as exc:
            confirmation_status = "FAIL"
            confirmation_evidence["validation_error"] = str(exc)
    elif any(path is not None and path.is_file() for path in (a5_sealed, a5_result)):
        confirmation_status = "FAIL"
        confirmation_evidence["validation_error"] = (
            "A5 readiness requires sealed predictions, an opened result, and a valid S1-C receipt together"
        )
    gates.append(
        GateResult(
            "label_fresh_risk_calibration",
            calibration_status,
            calibration_evidence,
            "v29_has_no_clinical_abstention_or_prediction_set_guarantee" if not calibration_complete else "use_only_sealed_S1_C_policy",
        )
    )
    gates.append(
        GateResult(
            "label_fresh_one_shot_confirmation",
            confirmation_status,
            confirmation_evidence,
            "do_not_claim_independent_performance_or_superiority" if not confirmation_complete else "run_prespecified_paired_analysis_only",
        )
    )

    statuses = {gate.gate_id: gate.status for gate in gates}
    development_evidence_complete = all(
        statuses[key] == "PASS"
        for key in (
            "development_ranker_replay",
            "private_role_integrity",
            "failed_concept_structural_isolation",
            "language_layer_machine_safety",
            "template_vs_qwen_study_preflight",
            "clinical_workflow_study_preflight",
        )
    )
    top_tier_ready = all(gate.status == "PASS" for gate in gates)
    blockers = [gate.gate_id for gate in gates if gate.status != "PASS"]

    return {
        "schema_version": "trustworthy_soz_top_tier_readiness_result_v35",
        "contract": str(contract_path.relative_to(workspace)),
        "contract_sha256": _sha256(contract_path),
        "development_evidence_complete": development_evidence_complete,
        "top_tier_submission_ready": top_tier_ready,
        "submission_status": "READY" if top_tier_ready else "NOT_READY",
        "gates": [gate.as_dict() for gate in gates],
        "blocking_gates": blockers,
        "claim_ceiling": (
            "independently_confirmed_trustworthy_candidate_system"
            if top_tier_ready
            else "developmental_fail_closed_candidate_system_with_machine_audited_reporting"
        ),
    }


def write_readiness_result(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(result), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def statuses(result: Mapping[str, Any]) -> Sequence[str]:
    return tuple(str(gate["status"]) for gate in result.get("gates", []))
