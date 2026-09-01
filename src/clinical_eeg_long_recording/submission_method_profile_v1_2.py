"""Fail-closed validator for the clinical EEG submission profile v1.2.

The profile is a research-method decision, not a trained-model or performance
receipt.  This validator binds the dual detector operating points, Safe-VOI
fallback rules, event-level Findings structure, temporal permissions, and the
explicitly incomplete execution state to their checked-in dependencies.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence


SUBMISSION_METHOD_PROFILE_SCHEMA_VERSION_V1_2: Final[str] = (
    "clinical_eeg_submission_method_profile_v1_2"
)
SUBMISSION_METHOD_PROFILE_ID_V1_2: Final[str] = (
    "CLINICAL-EEG-SUBMISSION-METHOD-PROFILE-V1.2-20260824"
)
TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2: Final[str] = (
    "0e7f9392525bc8ccbff96ae6a50b9d4f7ec9d11513a75f3d29cf3a81b4356652"
)

_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_SUBMISSION_METHOD_PROFILE_PATH_V1_2: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_submission_method_profile_v1_2.json"
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS: Final[set[str]] = {
    "schema_version",
    "profile_id",
    "status",
    "authoritative_document",
    "inherits",
    "scope",
    "detector_dual_operating_point",
    "outer_acquisition",
    "inner_event_encoder",
    "representations",
    "temporal_permissions",
    "findings",
    "localization",
    "training",
    "minimum_ablations",
    "execution_status",
    "source_firewall",
    "scientific_permissions",
    "receipt_sha256",
}

_INHERITED_BINDINGS: Final[dict[str, tuple[str, str]]] = {
    "ba_ieg_v1_core_contract_sha256": (
        "configs/clinical_eeg_ba_ieg_v1_core_freeze.json",
        "contract_sha256",
    ),
    "detector_admission_addendum_receipt_sha256": (
        "configs/clinical_eeg_detector_admission_addendum_v1_1.json",
        "receipt_sha256",
    ),
    "onset_identity_addendum_receipt_sha256": (
        "configs/clinical_eeg_ba_ieg_onset_identity_addendum_v1_1.json",
        "receipt_sha256",
    ),
    "event_qualification_registry_receipt_sha256": (
        "configs/clinical_eeg_ba_ieg_event_qualification_threshold_registry_v1.json",
        "receipt_sha256",
    ),
    "findings_core_profile_sha256": (
        "configs/clinical_eeg_findings_v1_core_release_profile.json",
        "profile_sha256",
    ),
    "findings_composer_closure_receipt_sha256": (
        "configs/clinical_eeg_findings_v1_composer_closure_addendum.json",
        "receipt_sha256",
    ),
    "findings_v1_2_semantic_addendum_receipt_sha256": (
        "configs/clinical_eeg_findings_v1_2_semantic_addendum.json",
        "receipt_sha256",
    ),
    "record_context_policy_receipt_sha256": (
        "configs/clinical_eeg_record_non_event_context_card_policy_v1.json",
        "policy_sha256",
    ),
}

_SOURCE_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "edf_annotations_used": False,
    "spreadsheets_used": False,
    "doctor_labels_or_reports_used": False,
    "clinical_text_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_information_used": False,
    "ecg_or_other_physiology_used": False,
}

_SCIENTIFIC_PERMISSIONS: Final[dict[str, bool]] = {
    "architecture_candidate_frozen": True,
    "trained_model_claim_authorized": False,
    "performance_claim_authorized": False,
    "sota_claim_authorized": False,
    "clinical_report_or_diagnosis_authorized": False,
    "report_language_stage_authorized": False,
}

_EXECUTION_STATUS: Final[dict[str, bool]] = {
    "complete_official_dev_detector_benchmark": False,
    "qualified_alarm_operating_point": False,
    "qualified_navigation_operating_point": False,
    "a0_native12_complete_318_records_908_events": False,
    "source_trained_ba_ieg_checkpoint": False,
    "real_multistep_safe_voi_rollout": False,
    "all_twelve_event_slots_have_real_native_producers": False,
    "held_out_findings_performance": False,
    "held_out_soz_performance": False,
    "report_language_optimization_active": False,
    "end_to_end_pipeline_complete": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def submission_method_profile_self_sha256_v1_2(
    value: Mapping[str, object],
) -> str:
    """Hash canonical profile content after deleting only its receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("submission method profile must be an object")
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{context} keys drifted; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return deepcopy(value)


def _safe_project_file(relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{context} must be a canonical relative path")
    root = _ROOT.resolve(strict=True)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    path = unresolved.resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must resolve to a regular file")
    return path


def _load_strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite token {token}")
        ),
    )
    if type(value) is not dict:
        raise TypeError(f"{context} must contain an object")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_inherited_bindings(value: object) -> dict[str, Any]:
    row = _strict_object(value, set(_INHERITED_BINDINGS), "inherits")
    for name, (relative_path, field) in _INHERITED_BINDINGS.items():
        expected = _sha256(row[name], f"inherits.{name}")
        source = _load_strict_json(
            _safe_project_file(relative_path, f"inherits.{name}.path"),
            f"inherits.{name}.source",
        )
        if source.get(field) != expected:
            raise ValueError(f"inherits.{name} dependency receipt drifted")
    return row


def _require_equal(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context} drifted")


def validate_submission_method_profile_v1_2(
    value: Mapping[str, Any],
    *,
    verify_dependencies: bool = True,
    require_trusted_receipt: bool = True,
) -> dict[str, Any]:
    """Validate the frozen candidate and keep all unearned claims disabled."""

    row = _strict_object(value, _TOP_LEVEL_KEYS, "submission method profile")
    _require_equal(
        row["schema_version"],
        SUBMISSION_METHOD_PROFILE_SCHEMA_VERSION_V1_2,
        "schema_version",
    )
    _require_equal(row["profile_id"], SUBMISSION_METHOD_PROFILE_ID_V1_2, "profile_id")
    _require_equal(
        row["status"],
        "frozen_research_candidate_not_trained_not_performance_qualified",
        "status",
    )
    _safe_project_file(row["authoritative_document"], "authoritative_document")
    if verify_dependencies:
        _validate_inherited_bindings(row["inherits"])
    else:
        _strict_object(row["inherits"], set(_INHERITED_BINDINGS), "inherits")

    receipt = _sha256(row["receipt_sha256"], "receipt_sha256")
    if receipt != submission_method_profile_self_sha256_v1_2(row):
        raise ValueError("submission method profile self-hash replay failed")
    if require_trusted_receipt and receipt != (
        TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2
    ):
        raise ValueError("submission method profile is not the trusted v1.2 receipt")

    detector = row["detector_dual_operating_point"]
    if type(detector) is not dict:
        raise TypeError("detector_dual_operating_point must be an object")
    _require_equal(detector.get("shared_reference_free_prediction_inventory_required"), True, "detector prediction-first gate")
    _require_equal(
        detector.get("selection_layers"),
        {
            "technical_eligibility": {
                "all_requirements_mandatory": True,
                "requirements": [
                    "immutable_artifact_and_license_identity",
                    "safe_checkpoint_or_isolated_container_loading",
                    "checkpoint_native_preprocessing_replay",
                    "complete_official_dev_prediction_first_inventory",
                    "one_terminal_outcome_per_expected_recording",
                ],
            },
            "descriptive_research_primary": {
                "selection_rule": (
                    "frozen_official_dev_accuracy_false_alarm_onset_query_cost_pareto"
                ),
                "false_alarm_budgets_per_24h": [1, 3, 6, 12],
                "required_accuracy_views": [
                    "pooled_event_sensitivity",
                    "patient_macro_event_sensitivity",
                    "false_alarm_budget_partial_auc",
                    "recall_at_candidate_budget",
                    "all_reference_onset_hit_at_1_3_5_10_seconds",
                    "queried_eeg_seconds",
                ],
                "paired_patient_bootstrap_required": True,
                "deployment_gate_required_for_selection": False,
                "failure_to_meet_deployment_gate_keeps_research_only_status": True,
                "current_provider_id": None,
            },
            "deployment_qualification": {
                "uses_alarm_operating_point_gate": True,
                "external_transport_required": True,
                "downstream_onset_findings_utility_required": True,
                "failure_status": "qualified_provider_null",
                "current_provider_id": None,
            },
        },
        "detector three-layer selection policy",
    )
    alarm = detector.get("alarm_operating_point")
    navigation = detector.get("navigation_operating_point")
    if type(alarm) is not dict or type(navigation) is not dict:
        raise TypeError("both detector operating points must be objects")
    _require_equal(
        {
            "pooled": alarm.get("pooled_event_sensitivity_minimum"),
            "macro": alarm.get("patient_macro_event_sensitivity_minimum"),
            "fa24": alarm.get("all_unmatched_alarms_per_24h_maximum"),
            "rtf": alarm.get("warm_end_to_end_rtf_maximum"),
            "permission": alarm.get("clinical_or_production_permission"),
        },
        {"pooled": 0.9, "macro": 0.85, "fa24": 12.0, "rtf": 0.05, "permission": False},
        "alarm operating point",
    )
    _require_equal(
        {
            "pooled": navigation.get("pooled_onset_search_envelope_recall_target"),
            "macro": navigation.get("patient_macro_onset_search_envelope_recall_target"),
            "candidate_grid": navigation.get("candidate_budget_per_native_recording_hour_grid"),
            "query_grid": navigation.get("fine_analysis_eeg_seconds_per_recording_hour_grid"),
            "permission": navigation.get("clinical_alarm_or_sota_permission"),
        },
        {
            "pooled": 0.98,
            "macro": 0.95,
            "candidate_grid": [1, 2, 4, 8, 16],
            "query_grid": [60, 120, 300, 600],
            "permission": False,
        },
        "navigation operating point",
    )
    fallback = detector.get("zero_candidate_known_seizure_cohort_fallback")
    if type(fallback) is not dict or fallback != {
        "route": "coarse_full_record_fallback",
        "counts_as_detector_hit": False,
        "may_rewrite_detection_metrics": False,
        "guarantees_clinical_soz_conclusion": False,
    }:
        raise ValueError("zero-candidate fallback authority drifted")
    provider_roles = detector.get("provider_roles")
    if type(provider_roles) is not dict:
        raise TypeError("provider_roles must be an object")
    if any(
        provider_roles.get(name) is not None
        for name in (
            "accuracy_primary",
            "descriptive_research_primary",
            "deployment_qualified_provider",
        )
    ):
        raise ValueError(
            "research or deployment primary cannot be preselected without benchmark evidence"
        )

    outer = row["outer_acquisition"]
    if type(outer) is not dict:
        raise TypeError("outer_acquisition must be an object")
    _require_equal(
        {
            "method": outer.get("method"),
            "core": outer.get("current_core"),
            "challenger": outer.get("submission_challenger"),
        },
        {
            "method": "posterior_tail_rule_core_with_safe_voi_challenger",
            "core": "posterior_tail_boundary_touch_v1",
            "challenger": "safe_value_of_information_asymmetric_active_acquisition",
        },
        "outer core/challenger hierarchy",
    )
    _require_equal(outer.get("hard_feasible_action_filter_required"), True, "outer hard action filter")
    _require_equal(outer.get("nested_patient_cross_fit_required"), True, "outer cross-fit")
    _require_equal(outer.get("learned_stop_may_override_unresolved_rule_boundary"), False, "outer learned-stop authority")
    _require_equal(outer.get("fallback"), "posterior_tail_rule", "outer fallback")
    _require_equal(
        outer.get("initial_support"),
        {
            "candidate_interval_left_guard_seconds": 12,
            "candidate_interval_right_guard_seconds": 10,
            "anchor_only_relative_interval_seconds": [-12, 10],
            "source_dev_calibration_required": True,
            "is_final_fixed_window": False,
        },
        "outer initial support",
    )
    _require_equal(
        outer.get("posterior_decision_semantics"),
        {
            "detector_and_boundary_posteriors_fused_as_probability": False,
            "extension": "logical_or_if_any_unresolved_tail_boundary_censoring_or_context_need",
            "stop": "logical_and_only_after_all_left_right_boundary_censoring_and_context_needs_resolved",
        },
        "outer posterior decision semantics",
    )
    _require_equal(
        outer.get("neighbor_candidate_policy"),
        "event_group_joint_refine_split_merge_review_no_midpoint_truncation",
        "neighbor candidate policy",
    )
    _require_equal(
        outer.get("temporal_evidence_slices"),
        {
            "k3_seconds": 3,
            "k5_shadow_seconds": 5,
            "full_course_may_rewrite_positive_onset_identity": False,
        },
        "outer temporal evidence slices",
    )

    inner = row["inner_event_encoder"]
    if type(inner) is not dict:
        raise TypeError("inner_event_encoder must be an object")
    _require_equal(inner.get("outer_support_only"), True, "inner support boundary")
    _require_equal(inner.get("channel_neutral_routing"), True, "inner channel neutrality")
    _require_equal(inner.get("lane_budget_transfer_allowed"), False, "inner lane budget isolation")
    _require_equal(inner.get("physical_time_scales_seconds"), [1, 4, 16], "inner physical scales")
    _require_equal(
        {
            "core": inner.get("current_core_routing"),
            "challenger": inner.get("submission_challenger"),
        },
        {
            "core": "fixed_complete_1_4_16_second_partition",
            "challenger": "learned_channel_neutral_ragged_router",
        },
        "inner core/challenger hierarchy",
    )

    temporal = row["temporal_permissions"]
    if type(temporal) is not dict:
        raise TypeError("temporal_permissions must be an object")
    _require_equal(temporal.get("causal_to_offline_edge"), "detached", "causal/offline edge")
    _require_equal(temporal.get("offline_or_late_evidence_may_create_positive_onset_identity"), False, "late evidence authority")
    _require_equal(
        temporal.get("offline_lane"),
        "course_later_recruitment_offset_return_and_counterevidence",
        "offline semantic ceiling",
    )

    findings = row["findings"]
    if type(findings) is not dict:
        raise TypeError("findings must be an object")
    _require_equal(findings.get("event_card_slot_count"), 12, "Event Card slot count")
    _require_equal(findings.get("record_context_card_slot_count"), 6, "Context Card slot count")
    _require_equal(findings.get("report_eligible_automated_allowlist"), [], "Findings automated allowlist")
    _require_equal(findings.get("ifcn_ied_six_item_wire_complete"), False, "IFCN IED wire status")
    _require_equal(
        {
            "chain": findings.get("execution_chain"),
            "addendum": findings.get("semantic_addendum_id"),
            "boundaries": findings.get(
                "detector_possible_and_qualified_onset_intervals_distinct"
            ),
            "ied_in_event": findings.get("interictal_ied_event_card_allowed"),
            "s10": findings.get("s10_default_semantics"),
            "s11": findings.get("s11_default_semantics"),
            "context_onset": findings.get(
                "context_may_create_or_reorder_primary_onset"
            ),
        },
        {
            "chain": [
                "four_proposal_families",
                "native_physical_deterministic_measurement",
                "factorized_event_grammar_and_explicit_composer",
                "per_term_qualification",
            ],
            "addendum": "CLINICAL-EEG-FINDINGS-V1.2-SEMANTIC-ADDENDUM-20260824",
            "boundaries": True,
            "ied_in_event": False,
            "s10": "scalp_topographic_recruitment_candidate_not_spread",
            "s11": "cessation_or_return_candidate_not_termination_or_postictal",
            "context_onset": False,
        },
        "Findings v1.2 semantic binding",
    )

    localization = row["localization"]
    if type(localization) is not dict:
        raise TypeError("localization must be an object")
    _require_equal(localization.get("bipolar_semantics"), "whole_lead_without_endpoint_attribution", "bipolar semantics")
    _require_equal(localization.get("multiple_mode_claim_requires_exhaustive_event_mode_gold"), True, "multiple-mode gate")

    _require_equal(row["source_firewall"], _SOURCE_FIREWALL, "source_firewall")
    _require_equal(row["scientific_permissions"], _SCIENTIFIC_PERMISSIONS, "scientific_permissions")
    _require_equal(row["execution_status"], _EXECUTION_STATUS, "execution_status")
    return row


def load_submission_method_profile_v1_2(
    path: Path = DEFAULT_SUBMISSION_METHOD_PROFILE_PATH_V1_2,
) -> dict[str, Any]:
    return validate_submission_method_profile_v1_2(
        _load_strict_json(path, "submission method profile")
    )


__all__ = [
    "DEFAULT_SUBMISSION_METHOD_PROFILE_PATH_V1_2",
    "SUBMISSION_METHOD_PROFILE_ID_V1_2",
    "SUBMISSION_METHOD_PROFILE_SCHEMA_VERSION_V1_2",
    "TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2",
    "load_submission_method_profile_v1_2",
    "submission_method_profile_self_sha256_v1_2",
    "validate_submission_method_profile_v1_2",
]
