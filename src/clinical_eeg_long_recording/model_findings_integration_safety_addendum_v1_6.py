"""Validate the additive model--Findings integration safety addendum v1.6.

This is an interface and scientific-authority validator.  It deliberately
fails closed while the full-stack exposure graph, trained checkpoints,
runtime registries, native evidence ledger, and clinical qualification are
missing.  Passing validation is not model, SOZ, Finding, or clinical
performance evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADDENDUM_PATH = (
    ROOT
    / "configs"
    / "clinical_eeg_model_findings_integration_safety_addendum_v1_6.json"
)
SCHEMA_VERSION = "clinical_eeg_model_findings_integration_safety_addendum_v1_6"
ADDENDUM_ID = (
    "CLINICAL-EEG-MODEL-FINDINGS-INTEGRATION-SAFETY-ADDENDUM-V1.6-20260824"
)
DEFAULT_ADDENDUM_SHA256 = (
    "ea94f89b9d103a1ab36e9664ebdd89fbba0adc8915fa6f80456acf054015cfb4"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_parent_bindings(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("integration parent binding roster drifted")
    expected_roles = {
        "base_model_findings_architecture_decision",
        "detector_cleanroom_execution_protocol",
        "findings_native_evidence_interface",
        "detector_scorer_denominator_and_efficiency_semantics",
    }
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("integration parent binding must be an object")
        role = str(row.get("role", ""))
        if role in by_role:
            raise ValueError("duplicate integration parent role")
        by_role[role] = row
        path = ROOT / str(row.get("path", ""))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"bound parent missing or symlinked: {path}")
        if _file_sha256(path) != row.get("file_sha256"):
            raise ValueError(f"bound parent hash drifted: {path}")
    if set(by_role) != expected_roles:
        raise ValueError("integration parent roles drifted")

    detector = json.loads(
        (ROOT / by_role["detector_cleanroom_execution_protocol"]["path"])
        .read_text(encoding="utf-8")
    )
    findings = json.loads(
        (ROOT / by_role["findings_native_evidence_interface"]["path"])
        .read_text(encoding="utf-8")
    )
    scorer = json.loads(
        (
            ROOT
            / by_role["detector_scorer_denominator_and_efficiency_semantics"][
                "path"
            ]
        ).read_text(encoding="utf-8")
    )
    if (
        detector.get("receipt_sha256")
        != by_role["detector_cleanroom_execution_protocol"].get(
            "semantic_receipt_sha256"
        )
    ):
        raise ValueError("detector semantic receipt drifted")
    if (
        findings.get("freeze_sha256")
        != by_role["findings_native_evidence_interface"].get(
            "semantic_receipt_sha256"
        )
    ):
        raise ValueError("Findings semantic receipt drifted")
    if (
        scorer.get("receipt_sha256")
        != by_role["detector_scorer_denominator_and_efficiency_semantics"].get(
            "semantic_receipt_sha256"
        )
    ):
        raise ValueError("detector scorer semantic receipt drifted")


def validate_model_findings_integration_safety_addendum_v1_6(
    value: Mapping[str, Any],
    *,
    trusted_addendum_sha256: str = DEFAULT_ADDENDUM_SHA256,
) -> dict[str, Any]:
    """Return a defensive copy after replaying every safety invariant."""

    addendum = deepcopy(dict(value))
    if addendum.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("integration safety schema drifted")
    if addendum.get("addendum_id") != ADDENDUM_ID:
        raise ValueError("integration safety identifier drifted")
    if addendum.get("status") != (
        "additive_interface_freeze_end_to_end_execution_blocked_not_trained_"
        "not_performance_admitted"
    ):
        raise ValueError("integration safety status drifted")

    observed_hash = addendum.get("addendum_sha256")
    body = deepcopy(addendum)
    body.pop("addendum_sha256", None)
    replayed_hash = _canonical_sha256(body)
    if observed_hash != replayed_hash or observed_hash != trusted_addendum_sha256:
        raise ValueError("integration safety addendum does not replay exactly")

    _validate_parent_bindings(addendum.get("integration_parent_bindings"))

    precedence = addendum["precedence"]
    if precedence["frozen_v1_4_or_child_freeze_files_modified"] is not False:
        raise ValueError("additive precedence drifted")
    if precedence["end_to_end_integration_claimed"] is not False:
        raise ValueError("unearned integration claim opened")

    firewall = addendum["forward_source_firewall"]
    if firewall["allowlisted_acquisition_field_registry"] != (
        "required_not_materialized"
    ):
        raise ValueError("acquisition header allowlist was overstated")
    for forbidden in (
        "EDF_annotations",
        "Excel_or_spreadsheet_fields",
        "doctor_labels_or_reports",
        "LLM_output_as_evidence",
    ):
        if forbidden not in firewall["forbidden"]:
            raise ValueError(f"forward source firewall drifted: {forbidden}")

    exposure = addendum["full_stack_exposure_graph"]
    if exposure["status"] != "required_not_materialized":
        raise ValueError("full-stack exposure status drifted")
    if exposure["split_unit"] != "patient":
        raise ValueError("full-stack split unit drifted")
    if exposure["detector_OOF_alone_is_sufficient_for_full_stack_evaluation"]:
        raise ValueError("detector OOF was incorrectly promoted to full-stack OOF")
    if exposure["cross_dataset_patient_overlap_ledger"]["status"] != (
        "not_materialized"
    ):
        raise ValueError("cross-dataset overlap status was overstated")
    if exposure["cross_dataset_patient_overlap_ledger"]["required_datasets"] != [
        "TUSZ",
        "SzCORE",
        "DeepSOZ",
        "TUEV",
        "TUAR",
    ]:
        raise ValueError("cross-dataset exposure roster drifted")
    if exposure["cross_dataset_patient_overlap_ledger"][
        "same_DeepSOZ_patient_may_supply_training_label_and_final_GT"
    ]:
        raise ValueError("DeepSOZ training/evaluation role leakage opened")
    if exposure["five_detector_checkpoints_per_arm_are_sufficient_for_full_stack_OOF"]:
        raise ValueError("detector checkpoints were promoted to full-stack OOF")

    phase = addendum["phase_specific_detector_prediction_authority"]
    if "inner_patient_cross_fitted" not in phase["source_train_downstream_rows"]:
        raise ValueError("source-train upstream authority is not inner cross-fitted")
    if "five_checkpoint" not in phase["source_dev"]:
        raise ValueError("source-dev upstream authority drifted")
    if "reference_free" not in phase["source_eval"]:
        raise ValueError("source-eval prediction-first authority drifted")
    if phase["current_status"] != "not_materialized":
        raise ValueError("phase-specific authority was overstated")
    if phase["ensemble_scope"] != (
        "one_separate_five_checkpoint_ensemble_per_provider_never_one_mixed_"
        "provider_ensemble"
    ):
        raise ValueError("provider-specific ensemble scope drifted")
    if phase["OP_ALARM_and_OP_NAVIGATION_may_select_different_provider_and_policy"] is not True:
        raise ValueError("detector lane-specific provider authority drifted")

    final_stack = addendum["final_frozen_stack_construction"]
    if final_stack["status"] != "required_not_materialized":
        raise ValueError("final stack construction was overstated")
    if final_stack[
        "source_dev_signal_or_reference_used_to_refit_detector_core_ranker_or_router"
    ]:
        raise ValueError("source-dev full-stack refit leakage opened")
    if final_stack["source_eval_or_private_reference_used_for_any_refit"]:
        raise ValueError("source-eval/private refit leakage opened")

    scoring = addendum["detector_selection_and_scoring_resolution"]
    if scoring["accuracy_primary_before_source_eval"] is not None:
        raise ValueError("accuracy primary opened before source-eval")
    if scoring["source_dev_role"] != (
        "freeze_decoder_threshold_policy_and_accuracy_primary_candidate_only"
    ):
        raise ValueError("source-dev selection authority drifted")
    if scoring["navigation_operating_point_may_fill_accuracy_primary"]:
        raise ValueError("navigation result was allowed to fill accuracy primary")
    if scoring["failure_duration_may_dilute_RTF"]:
        raise ValueError("failure duration was allowed to dilute RTF")

    nscec = addendum["NS_CEC_execution_semantics"]
    if nscec["real_native_EEG_multi_step_rollout"] is not False:
        raise ValueError("NS-CEC real rollout was overstated")
    if nscec["fixed_minus12_plus48_is_default_q0"]:
        raise ValueError("fixed window silently became q0")
    if nscec["left_right_action_seconds"] != [2, 4, 8, 16, 32]:
        raise ValueError("NS-CEC action grid drifted")
    if nscec["right_course_state_may_update_left_closure_or_positive_rank"]:
        raise ValueError("right-course future path to onset rank opened")

    causal = addendum["causal_claim_boundary"]
    if causal["end_to_end_online_causal_onset_claim_authorized"]:
        raise ValueError("end-to-end causal claim opened")
    if causal["query_count_support_geometry_stop_reason_or_right_course_state_may_enter_positive_rank"]:
        raise ValueError("control/course shortcut to rank opened")

    findings = addendum["findings_and_spatial_output_resolution"]
    per_reference = findings["per_reference_family_output"]
    if per_reference[
        "electrode_and_whole_bipolar_lead_share_one_probability_simplex"
    ]:
        raise ValueError("incompatible typed units share one simplex")
    if per_reference[
        "directed_whole_bipolar_lead_may_be_split_to_endpoint_probabilities"
    ]:
        raise ValueError("whole bipolar lead endpoint attribution opened")
    if findings["resolution_backoff"] != [
        "typed_unit_within_reference_family",
        "region",
        "laterality",
        "unresolved",
    ]:
        raise ValueError("resolution backoff drifted")
    broad = findings["broad_pattern"]
    if broad["is_a_resolution_backoff_state"]:
        raise ValueError("broad bilateral pattern became a fallback")
    if not broad[
        "requires_independent_positive_early_bilateral_near_synchrony_field_"
        "reference_group_deletion_and_late_invariance_gates"
    ]:
        raise ValueError("broad bilateral positive gate weakened")
    if findings["clinical_term_allowlist"]:
        raise ValueError("unqualified clinical term allowlist opened")
    if "equals_one" not in per_reference["normalization"]:
        raise ValueError("reference-conditioned simplex normalization drifted")
    if "unresolved" not in per_reference["no_qualified_trigger_binding_policy"]:
        raise ValueError("unbound top-1 may be promoted")

    aggregation = addendum["multi_event_aggregation_resolution"]
    if aggregation["v1_4_primary_preserved"] != (
        "equal_occurrence_capped_log_mean_exp"
    ):
        raise ValueError("v1.4 aggregation primary silently drifted")
    if aggregation["silent_primary_replacement_allowed"]:
        raise ValueError("silent aggregation replacement opened")
    if aggregation["real_complete_ITA_implementation"]:
        raise ValueError("complete ITA implementation was overstated")

    denominators = addendum["evaluation_denominators"]
    if denominators["candidate_conditional_may_replace_end_to_end"]:
        raise ValueError("candidate-conditional endpoint replaced end-to-end")
    if denominators["unlabelled_SOZ_channels_are_negative"]:
        raise ValueError("unlabelled SOZ channels became negatives")
    joint = denominators["end_to_end_joint_scoring_mapping"]
    if not joint["joint_Hit_at_k"].startswith("D_e_times"):
        raise ValueError("joint Hit@k mapping drifted")
    if "zero_for_D_e_equals_zero" not in joint["joint_reciprocal_rank"]:
        raise ValueError("joint MRR mapping drifted")
    if "zero_for_D_e_equals_zero" not in joint["joint_graded_nDCG"]:
        raise ValueError("joint nDCG mapping drifted")
    if "all_unknown_prediction" not in joint["proper_spatial_score_on_miss_or_failure"]:
        raise ValueError("miss/failure spatial calibration mapping drifted")
    if denominators["localization_scoring_registry"] != "required_not_materialized":
        raise ValueError("localization scoring registry was overstated")

    gate = addendum["runtime_integration_manifest_gate"]
    if gate["status"] != "not_materialized_execution_blocked":
        raise ValueError("runtime integration status drifted")
    if gate["five_detector_checkpoints_close_detector_only_not_full_stack_OOF"] is not True:
        raise ValueError("runtime detector/full-stack distinction drifted")
    for field in (
        "all_required_artifacts_materialized",
        "large_scale_detector_training_authorized",
        "end_to_end_long_EDF_inference_authorized",
        "source_eval_reference_open_authorized",
        "clinical_or_production_use_authorized",
    ):
        if gate[field] is not False:
            raise ValueError(f"runtime safety gate opened: {field}")

    current = addendum["current_effect_snapshot"]
    if current["accuracy_primary"] is not None:
        raise ValueError("current accuracy primary was overstated")
    for field in (
        "legal_full_stack_OOF_inventory_exists",
        "hidden64_checkpoint_exists",
        "real_NS_CEC_rollout_exists",
        "patient_held_out_SOZ_performance_exists",
        "performance_or_medical_effect_claimed",
    ):
        if current[field] is not False:
            raise ValueError(f"current effect was overstated: {field}")

    report = ROOT / addendum["primary_human_report"]
    if not report.is_file() or report.is_symlink():
        raise ValueError("primary integration report missing or symlinked")
    return addendum


def load_model_findings_integration_safety_addendum_v1_6(
    path: str | Path = DEFAULT_ADDENDUM_PATH,
) -> dict[str, Any]:
    source = Path(path)
    return validate_model_findings_integration_safety_addendum_v1_6(
        json.loads(source.read_text(encoding="utf-8"))
    )


__all__ = [
    "ADDENDUM_ID",
    "DEFAULT_ADDENDUM_PATH",
    "DEFAULT_ADDENDUM_SHA256",
    "SCHEMA_VERSION",
    "load_model_findings_integration_safety_addendum_v1_6",
    "validate_model_findings_integration_safety_addendum_v1_6",
]
