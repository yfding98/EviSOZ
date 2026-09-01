#!/usr/bin/env python3
"""Content-bind and fail-closed audit the clinical EEG v5 architecture overlay."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/clinical_eeg_top_tier_method_architecture_v5.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/clinical_eeg_top_tier_method_architecture_v5_20260825/receipt.json"
)
COMMON17 = (
    "FP1", "FP2", "F7", "F3", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "P4", "P8", "O1", "O2",
)
REQUIRED_FORBIDDEN = {
    "EDF_annotations", "Excel_onset_fields", "doctor_labels_or_text",
    "clinical_history", "video_or_behavior", "sleep_staging", "provocation",
    "ECG_EMG_EOG",
}
EXPECTED_EVIDENCE_KEYS = {
    "common17_two_level_validation",
    "common17_two_level_report",
    "ST16_source_dev_evaluation",
    "ST16_training_coverage",
    "adaptive_v2_real_smoke",
    "adaptive_v3_design_contract",
    "adaptive_v3_synthetic_audit",
    "findings_v3_schema",
    "restricted_findings_real_smoke",
    "record_aggregation_core",
    "legacy_adaptive_record_adapter",
    "detector_anchor_bridge_coverage",
    "detector_cleanroom_physical_isolation",
    "ST16_cleanroom_training_dry_run",
    "ST16_cleanroom_dev_prediction_dry_run",
    "ST16_cleanroom_formal_entry_audit",
    "continuous_coarse_sentinel_implementation",
    "continuous_coarse_sentinel_real_EDF_smoke",
    "findings_v3_record_adapter_implementation",
    "findings_v3_record_adapter_report",
    "record_complete_denominator_implementation",
    "record_complete_denominator_report",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def _bound_file(binding: Mapping[str, object]) -> tuple[Path, dict[str, str]]:
    relative = Path(str(binding["path"]))
    path = (ROOT / relative).resolve(strict=True)
    if ROOT not in path.parents:
        raise ValueError(f"bound artifact escapes workspace: {relative}")
    expected = str(binding["sha256"])
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"content binding mismatch for {relative}: {observed} != {expected}")
    return path, {"path": str(relative), "sha256": observed}


def _bound_json(binding: Mapping[str, object]) -> tuple[dict[str, Any], dict[str, str]]:
    path, receipt = _bound_file(binding)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"bound JSON is not an object: {path}")
    return value, receipt


def _close(actual: object, expected: float, name: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{name} is not numeric")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} drifted: {actual} != {expected}")


def audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "clinical_eeg_top_tier_method_architecture_v5_overlay":
        raise ValueError("unexpected v5 architecture schema")
    if config.get("status") != (
        "active_architecture_overlay_no_qualified_detector_partial_implementation_not_end_to_end"
    ):
        raise ValueError("unexpected v5 architecture maturity status")

    inheritance = config["inheritance"]
    _, parent = _bound_file(
        {"path": inheritance["parent_path"], "sha256": inheritance["parent_sha256"]}
    )
    _, parent_markdown = _bound_file(
        {
            "path": inheritance["parent_markdown_path"],
            "sha256": inheritance["parent_markdown_sha256"],
        }
    )
    for name in (
        "may_weaken_inference_firewall",
        "may_relabel_development_conditional_synthetic_or_smoke_results",
        "may_claim_end_to_end_or_clinical_validity",
    ):
        if inheritance[name] is not False:
            raise ValueError(f"v5 inheritance guard opened: {name}")

    scope = config["frozen_scope"]
    if tuple(scope["canonical_directly_observed_channels"]) != COMMON17:
        raise ValueError("v5 common17 order drifted")
    if scope["FZ_or_PZ_signal_synthesis_interpolation_zero_fill_or_prediction_mapping"] is not False:
        raise ValueError("v5 permits FZ/PZ on signal or prediction side")
    if "GT_only" not in str(next(key for key in scope if key == "GT_only_mapping")):
        raise ValueError("v5 midline operation is not marked GT-only")
    if "CZ := CZ OR FZ OR PZ" not in scope["GT_only_mapping"]:
        raise ValueError("v5 GT-only midline mapping drifted")

    firewall = config["inference_firewall"]
    if not REQUIRED_FORBIDDEN <= set(firewall["forbidden"]):
        raise ValueError("v5 EEG-only inference firewall is incomplete")
    if firewall["labels_allowed_for_source_train_supervision_or_postfreeze_evaluation_only"] is not True:
        raise ValueError("v5 label access is not fail-closed")
    if firewall["Qwen_or_other_LLM_may_measure_EEG_or_create_facts"] is not False:
        raise ValueError("v5 lets an LLM measure EEG or create facts")

    evidence = config["content_bound_execution_evidence"]
    if set(evidence) != EXPECTED_EVIDENCE_KEYS:
        raise ValueError("v5 execution-evidence roster drifted")
    bindings: dict[str, dict[str, str]] = {}
    json_payloads: dict[str, dict[str, Any]] = {}
    for role, binding in evidence.items():
        path, verified = _bound_file(binding)
        bindings[role] = verified
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"bound JSON must be an object: {role}")
            json_payloads[role] = payload

    two_level = json_payloads["common17_two_level_validation"]
    if two_level["common17_signal_contract"]["canonical_records_missing_any_common17_channel"] != 0:
        raise ValueError("bound common17 audit contains missing common17 channels")
    if two_level["common17_signal_contract"]["FZ_or_PZ_signal_synthesized_or_interpolated"] is not False:
        raise ValueError("bound common17 audit synthesized FZ/PZ")
    det = two_level["detection"]["evaluation"]
    local_det = config["truthful_current_state"]["detector"]["EN17_3epoch"]
    for field in (
        "event_sensitivity", "event_precision", "event_f1",
        "false_alarms_per_24h", "onset_hit_at_10s",
    ):
        _close(det[field], float(local_det[field]), f"EN17 {field}")
    boundary = two_level["claim_boundary"]
    if boundary["source_eval_opened"] is not False:
        raise ValueError("source-eval was unexpectedly opened")
    if boundary["detector_to_soz_end_to_end_metric_available"] is not False:
        raise ValueError("bound two-level result claims end-to-end SOZ")
    if boundary["soz_uses_oracle_onset"] is not True:
        raise ValueError("bound SOZ result is no longer marked oracle-onset")

    st16 = json_payloads["ST16_source_dev_evaluation"]
    st_metrics = st16["selected_strict_zero_dilation_ordered_one_to_one"]
    local_st = config["truthful_current_state"]["detector"]["ST16_1epoch"]
    for receipt_field, config_field in (
        ("event_sensitivity", "event_sensitivity"),
        ("event_precision", "event_precision"),
        ("event_f1", "event_f1"),
        ("alarm_false_alarms_per_24h", "false_alarms_per_24h"),
    ):
        _close(st_metrics[receipt_field], float(local_st[config_field]), f"ST16 {receipt_field}")
    _close(
        st_metrics["onset_absolute_hit_rate"]["10s"]["rate"],
        float(local_st["onset_hit_at_10s"]),
        "ST16 onset Hit@10s",
    )
    if st16["architecture_promotable"] is not False or st16["source_eval_opened"] is not False:
        raise ValueError("bound ST16 result is promoted or opened source-eval")

    adaptive_v2 = json_payloads["adaptive_v2_real_smoke"]
    if adaptive_v2["claim_limits"]["adaptive_superiority_authorized"] is not False:
        raise ValueError("adaptive-v2 smoke authorizes superiority")
    if not adaptive_v2["records"] or any(
        row["base_vs_fixed_shadow_agreement_pass"] is not False
        for row in adaptive_v2["records"]
    ):
        raise ValueError("adaptive-v2 real NO-GO evidence drifted")

    adaptive_v3 = json_payloads["adaptive_v3_synthetic_audit"]
    if adaptive_v3["claim_limits"]["engineering_behavior_audit_only"] is not True:
        raise ValueError("adaptive-v3 is no longer marked engineering-only")
    if adaptive_v3["claim_limits"]["detector_or_localization_efficacy_estimated"] is not False:
        raise ValueError("adaptive-v3 synthetic audit claims efficacy")
    if adaptive_v3["checks"]["late_spread_cannot_change_candidate_locked_positive_rank"] is not True:
        raise ValueError("adaptive-v3 late-evidence firewall is not bound")

    findings_smoke = json_payloads["restricted_findings_real_smoke"]
    if findings_smoke["record_count"] != 2:
        raise ValueError("restricted Findings real-smoke denominator drifted")
    if findings_smoke["claim_limits"]["engineering_real_EDF_smoke_only"] is not True:
        raise ValueError("restricted Findings smoke is overclaimed")
    if findings_smoke["claim_limits"]["clinical_term_qualification_claimed"] is not False:
        raise ValueError("restricted Findings smoke claims clinical term qualification")

    bridge = json_payloads["detector_anchor_bridge_coverage"]
    if bridge["claim_boundary"]["102_patient_end_to_end_metric_available"] is not False:
        raise ValueError("partial detector-anchor bridge was relabelled end-to-end")
    if bridge["SOZ_inference_performed"] is not False:
        raise ValueError("coverage-only bridge claims SOZ inference")

    cleanroom = json_payloads["detector_cleanroom_physical_isolation"]
    cleanroom_audit = cleanroom["audit"]
    if cleanroom["status"] != "pass_physical_isolation_manifests_materialized_training_not_started":
        raise ValueError("detector clean-room isolation status drifted")
    if (
        cleanroom_audit["source_train_recording_count"] != 4664
        or cleanroom_audit["source_dev_recording_count"] != 1821
        or cleanroom_audit["source_dev_target_files_opened"] != 0
        or cleanroom_audit["source_eval_target_files_opened"] != 0
        or cleanroom_audit["common17_missing_or_duplicate_record_count"] != 0
        or cleanroom_audit["FZ_or_PZ_in_model_tensor"] is not False
        or cleanroom_audit["split_isolation"]["all_required_intersections_zero"] is not True
    ):
        raise ValueError("detector clean-room isolation evidence drifted")
    if (
        cleanroom["prediction_execution"]["detector_trained"] is not False
        or cleanroom["prediction_execution"]["performance_estimated"] is not False
        or cleanroom["claim_limits"]["performance_or_SOTA_claim_authorized"] is not False
    ):
        raise ValueError("detector clean-room manifest preparation was overclaimed")

    st16_train_dry = json_payloads["ST16_cleanroom_training_dry_run"]
    if (
        not st16_train_dry["status"].startswith("no_go_formal_gpu_training")
        or st16_train_dry["source_train_recording_count"] != 4664
        or st16_train_dry["record_terminal_admission_denominator_count"] != 4664
        or st16_train_dry["native_record_count"] != 4436
        or st16_train_dry["short_context_arm_record_count"] != 228
        or st16_train_dry["planned_gradient_contributing_record_count"] != 4664
        or st16_train_dry["actual_gradient_contributing_record_count"] != 0
        or st16_train_dry["short_record_real_edf_transform_integrated"] is not False
        or st16_train_dry["short_record_masked_loss_integrated_in_existing_trainer"] is not False
        or st16_train_dry["claim_limits"]["complete_training_claim_authorized"] is not False
        or st16_train_dry["claim_limits"]["performance_estimated"] is not False
    ):
        raise ValueError("ST16 clean-room training dry-run maturity drifted")
    st16_dev_dry = json_payloads["ST16_cleanroom_dev_prediction_dry_run"]
    if (
        st16_dev_dry["expected_recording_count"] != 1821
        or st16_dev_dry["target_bearing_field_or_value_count"] != 0
        or st16_dev_dry["permissions"]["source_dev_target_path_resolved"] is not False
        or st16_dev_dry["permissions"]["source_dev_target_opened"] is not False
        or st16_dev_dry["permissions"]["source_eval_opened"] is not False
        or st16_dev_dry["checkpoint_required_and_not_loaded_by_dry_run"] is not True
    ):
        raise ValueError("ST16 clean-room dev prediction dry-run drifted")
    st16_entry = json_payloads["ST16_cleanroom_formal_entry_audit"]
    if (
        st16_entry["status"]
        != "pass_dry_run_and_target_isolation_formal_gpu_training_remains_no_go"
        or st16_entry["formal_launch_gate"]["status"] != "no_go"
        or st16_entry["formal_launch_gate"][
            "may_call_4664_records_actual_gradient_contributors"
        ] is not False
        or st16_entry["audit"]["source_train_terminal_admission_count"] != 4664
        or st16_entry["audit"]["source_train_actual_gradient_contribution_count"] != 0
        or st16_entry["audit"]["source_dev_target_open_count"] != 0
        or st16_entry["claim_limits"]["performance_estimated"] is not False
    ):
        raise ValueError("ST16 clean-room formal-entry audit drifted")
    if (
        st16_entry["artifact_bindings"]["training_dry_run"]["file_sha256"]
        != bindings["ST16_cleanroom_training_dry_run"]["sha256"]
        or st16_entry["artifact_bindings"]["dev_prediction_dry_run"]["file_sha256"]
        != bindings["ST16_cleanroom_dev_prediction_dry_run"]["sha256"]
    ):
        raise ValueError("ST16 formal-entry child receipt binding drifted")

    sentinel_smoke = json_payloads["continuous_coarse_sentinel_real_EDF_smoke"]
    if (
        sentinel_smoke["summary"]["record_count"] != 2
        or sentinel_smoke["summary"]["all_records_completed"] is not True
        or sentinel_smoke["summary"]["all_1s_4s_16s_partitions_gap_free"] is not True
        or sentinel_smoke["summary"]["all_FZ_PZ_samples_unread"] is not True
        or sentinel_smoke["summary"]["all_forbidden_source_APIs_unopened"] is not True
        or sentinel_smoke["summary"]["all_proposals_query_only"] is not True
        or sentinel_smoke["summary"]["all_raw_and_QC_hashes_exactly_replayed"] is not True
    ):
        raise ValueError("continuous sentinel real-EDF engineering smoke drifted")
    if (
        sentinel_smoke["claim_limits"]["sentinel_detection_performance_claim_authorized"] is not False
        or sentinel_smoke["claim_limits"]["Findings_or_SOZ_efficacy_claim_authorized"] is not False
    ):
        raise ValueError("continuous sentinel real-EDF smoke was overclaimed")

    current = config["truthful_current_state"]
    if current["detector"]["qualified_provider"] is not None:
        raise ValueError("v5 currently has no qualified detector provider")
    if current["detector"]["cleanroom_physical_isolation"] != {
        "status": "implemented_and_audited_manifest_preparation_only",
        "source_train_labeled_records": 4664,
        "source_dev_EEG_only_records": 1821,
        "source_dev_or_eval_target_files_opened": 0,
        "detector_training_or_performance_produced": False,
    }:
        raise ValueError("v5 detector clean-room implementation state drifted")
    if current["detector"]["ST16_cleanroom_formal_entry"] != {
        "status": "NO_GO_GPU_training_pending_short_record_real_EDF_transform_and_masked_loss_integration",
        "terminal_admission_records": 4664,
        "native_length_records": 4436,
        "short_record_planned_arm_records": 228,
        "planned_gradient_contributing_records": 4664,
        "actual_gradient_contributing_records": 0,
        "source_dev_target_bearing_field_or_value_count": 0,
    }:
        raise ValueError("v5 ST16 clean-room formal-entry state drifted")
    if current["adaptive_acquisition"] != {
        "v1": "rejected_real_259_event_geometry_degeneracy",
        "v2": "real_EDF_smoke_NO_GO_base_shadow_mismatch",
        "v3": "synthetic_only_executable_not_real_efficacy",
        "v5": "continuous_coarse_sentinel_real_EDF_engineering_smoke_passed_occurrence_association_BCDC_K3_MAER_and_real_efficacy_pending",
    }:
        raise ValueError("v5 adaptive maturity ladder drifted")
    if current["findings"]["restricted_materializer_implemented"] is not True:
        raise ValueError("v5 hides the implemented restricted Findings materializer")
    if current["findings"]["complete_native_producer_coverage"] is not False:
        raise ValueError("v5 overclaims complete Findings producer coverage")
    if current["findings"]["canonical_findings_v3_to_record_adapter_implemented"] is not True:
        raise ValueError("v5 hides the implemented canonical Findings adapter")
    if current["findings"]["connected_to_adaptive_v3_or_v5"] is not False:
        raise ValueError("v5 overclaims a real adaptive Findings producer wire")
    record_state = current["record_inference"]
    if (
        record_state["canonical_findings_v3_record_interface_connected"] is not True
        or record_state["canonical_findings_v3_record_interface_real_EEG_efficacy_estimated"] is not False
        or record_state["zero_candidate_record_supported"] is not True
        or record_state["technical_failure_record_supported"] is not True
        or record_state["partial_coverage_record_supported"] is not False
    ):
        raise ValueError("v5 record interface or complete-denominator maturity drifted")

    detector = config["mature_common17_detector_benchmark_v5"]
    family_ids = {row["id"] for row in detector["primary_families"]}
    if family_ids != {"ST16_C17_LB16", "DSH17_C17_REF", "EN17_C17_REF"}:
        raise ValueError("v5 detector-family benchmark drifted")
    if detector["prediction_first_contract"]["source_eval_opened"] is not False:
        raise ValueError("v5 detector contract opens source-eval")
    if detector["hard_completeness_gate"] != {
        "technical_failure_count": 0,
        "partial_record_count": 0,
        "physical_record_coverage_fraction": 1.0,
    }:
        raise ValueError("v5 detector completeness gate drifted")
    if detector["onset_eligibility_gate"]["overlap_hit_is_not_onset_accuracy"] is not True:
        raise ValueError("v5 permits overlap hit to stand in for onset accuracy")
    if not detector["selection"]["no_passing_provider_policy"].startswith("keep_provider_null"):
        raise ValueError("v5 may backfill oracle onsets when detector admission fails")

    acquisition = config["continuous_sentinel_ABEA_v5"]
    sentinel = acquisition["continuous_coarse_sentinel"]
    if not sentinel["implementation_status"].startswith("implemented_and_real_EDF_engineering_smoke_validated"):
        raise ValueError("v5 hides or overclaims the continuous sentinel implementation")
    if sentinel["current_implementation_detector_posterior_or_anchor_used"] is not False:
        raise ValueError("v5 overclaims a detector-posterior sentinel wire")
    if sentinel["covers_every_legal_cell_without_sparse_probe_gaps"] is not True:
        raise ValueError("v5 continuous sentinel has probe gaps")
    if sentinel["may_directly_assert_finding_onset_or_SOZ"] is not False:
        raise ValueError("v5 lets coarse sentinel assert an onset fact")
    if acquisition["change_islands"]["earliest_island_wins_unconditionally"] is not False:
        raise ValueError("v5 lets an unrelated earliest island hijack occurrence onset")
    if acquisition["outer_acquisition_is_distinct_from_boundary_estimation_and_inner_tokenization"] is not True:
        raise ValueError("v5 conflates outer support, boundary and tokenization")
    k3 = acquisition["K3_onset_causal_permission_firewall"]
    if k3["lookahead_confirmation_samples_may_enter_positive_channel_rank"] is not False:
        raise ValueError("v5 K3 leaks confirmation lookahead")
    if k3["late_course_spread_offset_or_recovery_may_increase_positive_onset_rank"] is not False:
        raise ValueError("v5 K3 lets late course increase onset rank")
    if k3["late_suffix_mutation_must_leave_C2_and_primary_channel_rank_bitwise_invariant"] is not True:
        raise ValueError("v5 does not require a late-suffix invariance test")

    findings = config["event_findings_Q_C1_C2_C3_v5"]
    if findings["carrier_schema"] != "event_eeg_findings_v3" or findings["carrier_schema_replaced"] is not False:
        raise ValueError("v5 replaces or drifts the Findings v3 carrier")
    if set(findings["observation_states"]) != {
        "present", "absent_with_opportunity", "uncertain", "not_evaluable",
    }:
        raise ValueError("v5 four-state Findings semantics drifted")
    if findings["atom_lifecycle"] != [
        "proposal", "deterministic_native_measurement", "qualified_assertion_or_abstention",
    ]:
        raise ValueError("v5 proposal-measurement-qualification chain drifted")
    if "lexicalize_only" not in findings["LLM_permission"]:
        raise ValueError("v5 gives an LLM more than lexicalization permission")
    if findings["current_implementation"]["complete_Q_C1_C2_C3_native_coverage"] is not False:
        raise ValueError("v5 claims complete Q/C1/C2/C3 coverage")
    implementation = findings["current_implementation"]
    if (
        implementation["continuous_sentinel_ledger_available"] is not True
        or implementation["MAER_sidecar_available"] is not False
        or implementation["canonical_findings_v3_record_adapter_available"] is not True
        or implementation["canonical_adapter_real_EEG_efficacy_estimated"] is not False
    ):
        raise ValueError("v5 Findings implementation maturity drifted")

    record = config["record_level_scalp_onset_inference_v5"]
    if record["complete_denominator"]["silent_event_or_record_drop_allowed"] is not False:
        raise ValueError("v5 permits silent denominator loss")
    if record["broad_generalized_or_unresolved_mass_mapped_to_CZ"] is not False:
        raise ValueError("v5 maps a nonlocalized state to CZ")
    if record["calibration"]["uncalibrated_output_must_be_called_normalized_support_score_not_probability"] is not True:
        raise ValueError("v5 permits probability language before calibration")
    required_current_gap = {
        "canonical_v3_adapter_has_synthetic_interface_validation_only",
        "partial_coverage_requires_observed_support_unknown_tail_contract",
        "no_real_detector_anchor_complete_denominator_metric",
    }
    if not required_current_gap <= set(record["current_core_limitations"]):
        raise ValueError("v5 hides current record-core gaps")

    chain = config["canonical_evidence_chain_v5"]
    if chain["canonical_v3_findings_to_v5_record_adapter_available"] is not True:
        raise ValueError("v5 hides the implemented canonical adapter")
    if "not_real_EEG_efficacy" not in chain[
        "canonical_v3_findings_to_v5_record_adapter_validation_scope"
    ]:
        raise ValueError("v5 overclaims canonical adapter efficacy")
    if chain["report_graph_connected_to_record_aggregation_available"] is not False:
        raise ValueError("v5 claims a connected report graph that is not implemented")

    gates = config["end_to_end_claim_gate"]
    expected_gates = {
        "qualified_alarm_or_navigation_detector_available": False,
        "detector_train_dev_physical_isolation_available": True,
        "continuous_coarse_sentinel_cache_implemented": True,
        "real_target_blind_patient_separated_support_comparison_passed": False,
        "complete_Q_C1_C2_C3_native_producer_coverage": False,
        "canonical_v3_findings_to_record_adapter_available": True,
        "zero_candidate_and_technical_record_complete_denominator_supported": True,
        "patient_disjoint_event_and_record_calibration_available": False,
        "detector_anchor_complete_denominator_SOZ_metric_available": False,
        "fact_locked_report_graph_connected": False,
        "automatic_clinical_diagnosis_or_deployment_authorized": False,
    }
    if gates != expected_gates:
        raise ValueError("v5 component or end-to-end gate drifted")
    positive_release_requirements = tuple(
        key for key in expected_gates
        if key != "automatic_clinical_diagnosis_or_deployment_authorized"
    )
    if all(gates[key] for key in positive_release_requirements):
        raise ValueError("v5 unexpectedly claims an end-to-end completed method")

    return content_address(
        {
            "schema_version": "clinical_eeg_top_tier_method_architecture_v5_audit_receipt",
            "status": "pass_content_bound_truthful_partial_implementation_not_end_to_end",
            "config": {
                "path": str(config_path.relative_to(ROOT)),
                "sha256": sha256_file(config_path),
            },
            "bindings": {
                "parent_config": parent,
                "parent_markdown": parent_markdown,
                **bindings,
            },
            "verified": {
                "common17_and_GT_only_midline_mapping": True,
                "EEG_only_inference_and_LLM_firewall": True,
                "EN17_and_ST16_replayed_as_failed_not_pending": True,
                "adaptive_v2_real_NO_GO_and_v3_synthetic_only": True,
                "restricted_real_Findings_materializer_is_implemented_but_incomplete": True,
                "detector_train_dev_cleanroom_physical_isolation_is_implemented": True,
                "ST16_cleanroom_formal_entry_is_NO_GO_not_completed_training": True,
                "continuous_sentinel_cache_real_EDF_engineering_path_is_verified_but_full_ABEA_and_efficacy_are_pending": True,
                "Q_C1_C2_C3_proposal_measurement_qualification_contract": True,
                "canonical_v3_to_record_adapter_and_zero_technical_denominator_paths_are_implemented": True,
                "partial_coverage_calibration_detector_anchor_metric_and_report_graph_remain_missing": True,
                "end_to_end_and_clinical_claim_gates_closed": True,
            },
            "unresolved_implementation_gates": [
                key for key in positive_release_requirements if gates[key] is False
            ],
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    receipt = audit(config_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(receipt) + b"\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
