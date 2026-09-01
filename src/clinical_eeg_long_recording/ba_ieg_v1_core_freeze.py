"""Fail-closed loader for the additive BA-IEG v1 architecture freeze.

The checked-in contract freezes a *method selection*, not a trained system.  It
now includes the external whole-record detector-navigation interface, a
resource-aware admission gate, and mutually exclusive A0 oracle-navigation and
A1 detector-frozen arms.  No detector currently holds a qualified operating
point, so A1 remains blocked and A0 remains a conditional downstream upper
bound only.

The in-memory seam from the causal trace through the shallow typed-unit head to
the record-local capped log-mean-exp aggregator is implemented and covered by
focused component tests.  A complete-record, training-only patient aggregation
bridge is also implemented, but none of these components is connected to a
full disk runner and no trained checkpoint exists.  The event-qualification
manifest is implemented, but the target-free provider, threshold and receipt
producer are not.  A separate
deterministic conservative region/laterality projection is now component-tested,
but remains untrained, uncalibrated and disconnected from the disk runner.
Loading this contract therefore cannot authorize end-to-end
training, inference, report generation, clinical use, or production deployment.

The trusted default is protected by three independent bindings:

* an exact SHA-256 of the checked-in schema bytes;
* a canonical content hash embedded in the contract, excluding only its own
  ``contract_sha256`` field; and
* an exact SHA-256 of the checked-in default JSON bytes compiled here.

Changing a method flag and recomputing the embedded hash is insufficient to
enable a forbidden route: semantic invariants and the compiled trusted-default
hash fail closed as well.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from .ba_ieg_capped_log_mean_exp_event_bag_v1 import (
    BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID,
    BA_IEG_EVENT_AGGREGATION_STATUSES,
)
from .ba_ieg_complete_patient_positive_set_bridge_v1 import (
    BA_IEG_COMPLETE_PATIENT_POSITIVE_SET_BRIDGE_ID_V1,
    BA_IEG_NAVIGATION_ARM_A0,
    BA_IEG_NAVIGATION_ARM_A1,
)
from .ba_ieg_inner_ragged_router_v1 import BA_IEG_INNER_RAGGED_ROUTER_MODEL_ID
from .ba_ieg_outer_active_acquisition_v2 import (
    BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID,
)
from .ba_ieg_physical_time_encoder import BA_IEG_PHYSICAL_TIME_ENCODER_ID
from .ba_ieg_record_spatial_resolution_projection_v1 import (
    BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_SHA256,
    BA_IEG_RECORD_SPATIAL_RESOLUTION_PROJECTION_ID,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID,
)
from .ba_ieg_training_contract import (
    BA_IEG_P0_IMPLEMENTATION_ID_NATIVE_12,
    BA_IEG_P0_TOKEN_FEATURES,
    BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
    BAIEGP0TokenizationPolicy,
)
from .mode_aware_hierarchical_positive_set_mil_v1 import (
    MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
)


BA_IEG_V1_CORE_FREEZE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_ba_ieg_v1_core_freeze_v1"
)
BA_IEG_V1_CORE_FREEZE_CONTRACT_ID: Final[str] = (
    "CLINICAL-EEG-BA-IEG-V1-CORE-FREEZE-20260823"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BA_IEG_V1_CORE_FREEZE_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_ba_ieg_v1_core_freeze.json"
)
DEFAULT_BA_IEG_V1_CORE_FREEZE_SCHEMA_PATH: Final[Path] = (
    _ROOT / "schemas" / "clinical_eeg_ba_ieg_v1_core_freeze.schema.json"
)

# These are trusted defaults, not values learned from the files at runtime.
TRUSTED_BA_IEG_V1_CORE_FREEZE_CONTRACT_SHA256: Final[str] = (
    "d02cb0044555195cb697b4e3e210f0dd6d55378d693c1e35f6a778420a09be91"
)
TRUSTED_BA_IEG_V1_CORE_FREEZE_FILE_SHA256: Final[str] = (
    "fb9d44c519f100a41aa5e0950484d09e3a0d4d5f2a1ea5c732865041005e5929"
)
TRUSTED_BA_IEG_V1_CORE_FREEZE_SCHEMA_SHA256: Final[str] = (
    "fa86baec3002782d28a7b85fd251a48e08b8063ddae7e41731f4ec4fdbb1f498"
)

_EXPECTED_CORE_ROUTE = (
    "external_whole_record_detector_navigation_contract",
    "rule_based_posterior_tail_acquisition",
    "deterministic_p0_16_feature_1_4_16_second_tokens",
    "permission_split_segmental_state_model",
    "shallow_causal_typed_unit_head",
    "capped_log_mean_exp_event_bag",
    "deterministic_conservative_record_spatial_projection",
)
_EXPECTED_SHADOW_IMPLEMENTATIONS = {
    "learned_outer_acquisition": BA_IEG_OUTER_ACTIVE_ACQUISITION_METHOD_ID_V2,
    "learned_inner_router": BA_IEG_INNER_RAGGED_ROUTER_MODEL_ID,
    "independent_physical_time_onset_encoder": BA_IEG_PHYSICAL_TIME_ENCODER_ID,
    "learned_mode_mixture": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
    "qwen_renderer": "qwen3.6_claim_locked_graph_to_text_renderer",
}
_EXPECTED_FIREWALL_FALSE = frozenset(
    {
        "edf_annotations_used",
        "excel_fields_used",
        "doctor_labels_used",
        "private_labels_used",
        "clinical_text_used",
        "patient_demographics_used",
        "video_or_behavior_used",
        "sleep_or_activation_labels_used",
        "ecg_or_other_polygraphy_used",
    }
)
_EXPECTED_BLOCKING_REASONS = (
    "qualified_detector_operating_point_and_runtime_receipt_missing",
    "adaptive_acquisition_threshold_receipt_missing",
    "target_free_event_qualification_provider_threshold_receipt_missing",
    "complete_patient_bridge_not_connected_to_disk_training_runner",
    "disk_runner_not_connected",
    "source_trained_core_checkpoint_missing",
    "performance_not_established",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ba_ieg_v1_core_freeze_contract_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical self hash, excluding only ``contract_sha256``."""

    if not isinstance(value, Mapping):
        raise TypeError("BA-IEG v1 core freeze must be a mapping")
    body = deepcopy(dict(value))
    body.pop("contract_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(body))


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _regular_file_bytes(path: Path, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path.read_bytes()


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema_bytes = _regular_file_bytes(
        DEFAULT_BA_IEG_V1_CORE_FREEZE_SCHEMA_PATH,
        "BA-IEG v1 core freeze schema",
    )
    if _sha256_bytes(schema_bytes) != TRUSTED_BA_IEG_V1_CORE_FREEZE_SCHEMA_SHA256:
        raise ValueError("trusted BA-IEG v1 core freeze schema hash drifted")
    try:
        schema = json.loads(schema_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("BA-IEG v1 core freeze schema is not valid UTF-8 JSON") from error
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _schema_error_path(error: object) -> str:
    path = getattr(error, "absolute_path", ())
    return "$" + "".join(
        f"[{item}]" if isinstance(item, int) else f".{item}" for item in path
    )


def _validate_code_bindings(candidate: Mapping[str, Any]) -> None:
    p0 = candidate["p0_tokenization"]
    if p0["implementation_id"] != BA_IEG_P0_IMPLEMENTATION_ID_NATIVE_12:
        raise ValueError("P0 implementation ID drifted from the registered native-12 code")
    if p0["view_profile"] != BA_IEG_P0_VIEW_PROFILE_NATIVE_12:
        raise ValueError("P0 view profile drifted from the registered native-12 profile")
    if tuple(p0["feature_names"]) != tuple(BA_IEG_P0_TOKEN_FEATURES):
        raise ValueError("P0 deterministic 16-feature order drifted")

    policy = BAIEGP0TokenizationPolicy().to_dict()
    if p0["scale_duration_seconds"] != policy["scale_duration_seconds"]:
        raise ValueError("P0 1/4/16-second duration policy drifted")
    if p0["scale_step_seconds"] != policy["scale_step_seconds"]:
        raise ValueError("P0 1/4/16-second step policy drifted")
    if p0["minimum_fine_samples"] != policy["minimum_fine_samples"]:
        raise ValueError("P0 minimum-fine-sample policy drifted")

    temporal = candidate["temporal_backbone"]
    if temporal["implementation_id"] != BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID:
        raise ValueError("permission-split segmental backbone implementation ID drifted")

    head = candidate["causal_typed_unit_head"]
    if head["head_id"] != BA_IEG_SHALLOW_CAUSAL_TYPED_UNIT_HEAD_ID:
        raise ValueError("shallow causal typed-unit head implementation ID drifted")

    aggregator = candidate["event_bag_aggregation"]
    if aggregator["aggregator_id"] != BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID:
        raise ValueError("record event-bag aggregator implementation ID drifted")
    qualification = aggregator["event_qualification_status_manifest"]
    if tuple(qualification["allowed_statuses"]) != tuple(
        BA_IEG_EVENT_AGGREGATION_STATUSES
    ):
        raise ValueError("event qualification status roster drifted from code")

    patient = candidate["training_only_patient_aggregation"]
    if patient["bridge_id"] != BA_IEG_COMPLETE_PATIENT_POSITIVE_SET_BRIDGE_ID_V1:
        raise ValueError("complete-patient training bridge implementation ID drifted")
    if tuple(patient["navigation_arms"]) != (
        BA_IEG_NAVIGATION_ARM_A0,
        BA_IEG_NAVIGATION_ARM_A1,
    ):
        raise ValueError("complete-patient navigation-arm roster drifted")

    projection = candidate["record_spatial_resolution_projection"]
    if projection["projection_id"] != (
        BA_IEG_RECORD_SPATIAL_RESOLUTION_PROJECTION_ID
    ):
        raise ValueError("record spatial projection implementation ID drifted")
    if projection["ontology_sha256"] != (
        BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_SHA256
    ):
        raise ValueError("record spatial projection ontology hash drifted")

    shadows = candidate["shadow_components"]
    for name, implementation_id in _EXPECTED_SHADOW_IMPLEMENTATIONS.items():
        if shadows[name]["implementation_id"] != implementation_id:
            raise ValueError(f"shadow component {name} implementation ID drifted")


def _validate_method_invariants(candidate: Mapping[str, Any]) -> None:
    if tuple(candidate["core_route"]) != _EXPECTED_CORE_ROUTE:
        raise ValueError("BA-IEG v1 core route drifted")

    navigation = candidate["detector_navigation"]
    if (
        navigation["formal_provider_status"] != "no_qualified_operating_point"
        or navigation["formal_provider_id"] is not None
        or navigation["a0_navigation"]["semantics"]
        != "conditional_on_seizure_interval_upper_bound"
        or navigation["a0_navigation"]["end_to_end_claim_authorized"]
        or navigation["a1_navigation"]["status"]
        != "blocked_no_qualified_detector_operating_point"
        or navigation["a1_navigation"][
            "reference_onset_used_for_candidate_selection"
        ]
        or not navigation["a1_navigation"]["required_for_end_to_end_claim"]
        or navigation["a1_navigation"][
            "typed_operating_point_receipt_validator_implemented"
        ]
        or navigation["a1_navigation"]["roster_instantiation_authorized"]
    ):
        raise ValueError("detector navigation A0/A1 boundary was weakened")
    if tuple(navigation["a1_navigation"]["required_typed_receipt_bindings"]) != (
        "detector_checkpoint_sha256",
        "prediction_bundle_sha256",
        "scorer_matching_contract_sha256",
        "official_split_receipt_sha256",
        "operating_point_thresholds",
        "pooled_event_sensitivity",
        "patient_macro_event_sensitivity",
        "background_only_false_alarms_per_24h",
        "warm_end_to_end_rtf",
        "runtime_hardware_receipt_sha256",
    ):
        raise ValueError("A1 typed detector operating-point receipt contract drifted")
    if not (
        navigation["pooled_event_sensitivity_minimum"] == 0.90
        and navigation["patient_macro_event_sensitivity_minimum"] == 0.85
        and navigation["background_only_false_alarms_per_24h_maximum"] == 12.0
        and navigation["warm_end_to_end_rtf_maximum"] == 0.05
        and navigation["runtime_hardware_receipt_status"]
        == "must_be_frozen_before_benchmark_not_yet_bound"
    ):
        raise ValueError("detector accuracy/false-alarm/runtime admission gate drifted")

    acquisition = candidate["acquisition"]
    if not (
        acquisition["posterior_tail_rule_based"]
        and acquisition["left_right_asymmetric"]
        and not acquisition["learned_policy"]
        and not acquisition["private_or_doctor_target_used"]
    ):
        raise ValueError("v1 acquisition must remain rule-based, asymmetric and target-free")
    if (
        acquisition["control_loop"]
        != "initial_support_then_iterative_acquire_p0_segmental_posterior_recompute"
        or acquisition["initial_posterior_provider"]
        != "detector_native_dense_posterior_or_candidate_credible_envelope"
        or acquisition["refinement_posterior_provider"]
        != "permission_split_segmental_state_model_boundary_marginals"
        or acquisition["threshold_receipt_available"]
        or acquisition["fallback_when_refinement_posterior_unavailable"]
        != "fixed_watchdog_support_and_adaptive_status_not_evaluable"
    ):
        raise ValueError("adaptive acquisition posterior loop or fallback drifted")

    head = candidate["causal_typed_unit_head"]
    if head["implementation_status"] != "implemented_focused_component_tests_only":
        raise ValueError("typed-unit head evidence scope cannot be promoted by config mutation")
    if not head["in_memory_causal_trace_seam_implemented"]:
        raise ValueError("typed-unit head must preserve the implemented causal-trace seam")
    if (
        head["disk_runner_connected"]
        or head["trained"]
        or head["trained_checkpoint_available"]
    ):
        raise ValueError("typed-unit head remains untrained and disconnected from disk runner")
    if tuple(head["output_unit_types"]) != (
        "physical_electrode",
        "bipolar_lead",
    ):
        raise ValueError("typed-unit head may output only electrode and whole bipolar lead")
    if head["bipolar_unit_semantics"] != (
        "whole_bipolar_lead_identity_without_endpoint_attribution"
    ):
        raise ValueError("typed-unit head bipolar identity semantics drifted")
    if head["bipolar_endpoint_attribution_authorized"]:
        raise ValueError("a bipolar lead cannot be attributed to an endpoint electrode")
    if head["late_spread_can_create_positive_onset"]:
        raise ValueError("late spread cannot create positive onset evidence")
    if head["region_laterality_projection_status"] != (
        "implemented_as_separate_downstream_component_focused_tests_only"
    ) or head["region_laterality_allowed_future_route"] != (
        "deterministic_conservative_projection_selected"
    ):
        raise ValueError("typed-unit head spatial projection binding drifted")

    aggregator = candidate["event_bag_aggregation"]
    if aggregator["implementation_status"] != (
        "implemented_focused_component_tests_only"
    ):
        raise ValueError("record event-bag evidence scope cannot be promoted by config mutation")
    if (
        aggregator["aggregation_level"] != "event_to_record"
        or not aggregator["record_local_only"]
        or aggregator["patient_level_aggregation_performed"]
    ):
        raise ValueError("v1 event bag must remain record-local, not patient-level")
    if (
        aggregator["disk_runner_connected"]
        or aggregator["trained"]
        or aggregator["trained_checkpoint_available"]
    ):
        raise ValueError("record event-bag remains untrained and disconnected from disk runner")
    if tuple(aggregator["record_level_unit_types"]) != (
        "physical_electrode",
        "bipolar_lead",
    ):
        raise ValueError("record event-bag may aggregate only electrode and whole bipolar lead")
    if aggregator["bipolar_unit_semantics"] != (
        "whole_bipolar_lead_identity_without_endpoint_attribution"
    ) or aggregator["bipolar_endpoint_attribution_authorized"]:
        raise ValueError("record event-bag must preserve whole bipolar lead identities")
    if aggregator["region_laterality_projection_status"] != (
        "implemented_as_separate_downstream_component_focused_tests_only"
    ) or aggregator["region_laterality_allowed_future_route"] != (
        "deterministic_conservative_projection_selected"
    ):
        raise ValueError("record-bag spatial projection binding drifted")
    if aggregator["exact_duplicate_events_increase_evidence"]:
        raise ValueError("copied events cannot increase record evidence")
    if aggregator["learned_mode_mixture_used"]:
        raise ValueError("learned mode mixture is outside the v1 core")
    qualification = aggregator["event_qualification_status_manifest"]
    if qualification["implementation_status"] != (
        "implemented_contract_and_focused_component_tests"
    ):
        raise ValueError("event qualification manifest evidence scope drifted")
    if any(
        qualification[key]
        for key in (
            "target_free_provider_implemented",
            "threshold_frozen",
            "qualification_receipt_provider_implemented",
            "manual_or_target_conditioned_status_authorized",
        )
    ):
        raise ValueError("event qualification provider/threshold/receipt remain blocked")

    patient = candidate["training_only_patient_aggregation"]
    if (
        patient["implementation_status"]
        != "implemented_focused_component_tests_not_connected_to_disk_runner"
        or patient["input_level"]
        != "complete_record_physical_electrode_logits_after_record_event_bag"
        or patient["aggregation_level"] != "record_to_patient_training_only"
        or tuple(patient["supervised_unit_types"]) != ("physical_electrode",)
        or patient["bipolar_lead_supervision_authorized"]
        or patient["patient_target_copied_to_record_or_event"]
        or patient["positive_set_loss_count_per_patient_per_step_maximum"] != 1
        or patient["a0_a1_mixed_in_one_run"]
        or patient["disk_runner_connected"]
        or patient["trained"]
        or patient["checkpoint_available"]
        or patient["inference_or_report_output_produced"]
    ):
        raise ValueError("training-only complete-patient bridge boundary drifted")
    if not (
        patient["same_temperature_and_clip_as_event_bag"]
        and patient["complete_record_roster_required"]
        and patient["zero_candidate_records_retained"]
        and patient["audit_denominator"]
        == "all_unique_complete_roster_records_including_zero_and_unqualified"
        and patient["numerical_denominator"]
        == "per_electrode_evaluable_record_mask_only"
        and patient["zero_or_unqualified_record_logit_imputation"] == "none"
        and not patient["detection_miss_penalty_in_localization_loss"]
        and not patient[
            "a1_typed_operating_point_receipt_validator_implemented"
        ]
        and not patient["a1_roster_instantiation_authorized"]
        and not patient["exact_container_aliases_increase_weight"]
    ):
        raise ValueError("complete-patient denominator or aggregation semantics drifted")

    projection = candidate["record_spatial_resolution_projection"]
    if projection["implementation_status"] != (
        "implemented_focused_component_tests_only"
    ) or projection["input_source"] != (
        "record_local_capped_lme_typed_unit_logits"
    ):
        raise ValueError("record spatial projection evidence scope drifted")
    if tuple(projection["axes"]) != (
        "laterality",
        "coarse_scalp_region",
        "joint_laterality_coarse_scalp_region",
    ):
        raise ValueError("record spatial projection axes drifted")
    if not projection["physical_electrode_precedence"] or projection[
        "whole_bipolar_lead_endpoint_attribution_authorized"
    ]:
        raise ValueError("record spatial projection source precedence drifted")
    if projection["cross_side_lead_projection"] != (
        "ineligible_for_laterality_region_and_joint_axes"
    ) or projection["same_side_cross_region_lead_projection"] != (
        "laterality_only"
    ):
        raise ValueError("record spatial projection lead policy drifted")
    if projection["bilateral_near_synchronous_phenotype_inferred"] or projection[
        "bilateral_near_synchronous_requirement"
    ] != "external_record_level_temporal_evidence":
        raise ValueError("spatial logits cannot create a bilateral timing phenotype")
    if projection["disk_runner_connected"] or projection["trained"] or projection[
        "outputs_are_clinical_probabilities"
    ]:
        raise ValueError("record spatial projection remains non-operational")

    for name, row in candidate["shadow_components"].items():
        if row["status"] != "shadow_off" or any(
            row[key]
            for key in (
                "enabled",
                "core_route_authorized",
                "checkpoint_authorized",
                "claim_support_authorized",
            )
        ):
            raise ValueError(f"shadow component {name} must remain fully off")


def _validate_firewall_report_performance(candidate: Mapping[str, Any]) -> None:
    firewall = candidate["inference_firewall"]
    if not firewall["eeg_samples_used"] or not firewall[
        "allowlisted_acquisition_metadata_used"
    ]:
        raise ValueError("v1 core requires EEG and allowlisted acquisition metadata")
    for name in _EXPECTED_FIREWALL_FALSE:
        if firewall[name] is not False:
            raise ValueError(f"forbidden inference input {name} was enabled")

    report = candidate["report_interface"]
    if report["report_eligible_term_allowlist"]:
        raise ValueError("v1 core report-eligible allowlist must remain empty")
    if any(
        report[key]
        for key in (
            "report_optimization_in_v1_core_scope",
            "report_route_connected",
            "qwen_enabled",
            "model_outputs_may_be_lexicalized_as_clinical_facts",
        )
    ):
        raise ValueError("report/Qwen route is outside the v1 model-core freeze")

    performance = candidate["performance_status"]
    if performance["state"] != "performance_not_established":
        raise ValueError("BA-IEG v1 performance has not been established")
    if any(
        performance[key]
        for key in (
            "end_to_end_checkpoint_available",
            "source_trained_core_checkpoint_available",
            "clinical_findings_performance_established",
            "soz_ranking_performance_established",
            "adaptive_window_added_value_established",
            "sota_claim_authorized",
            "clinical_use_claim_authorized",
        )
    ):
        raise ValueError("performance or clinical claim was enabled without evidence")

    authorization = candidate["execution_authorization"]
    if authorization["component_test_execution_authorized"] is not True:
        raise ValueError("focused component-test execution authorization drifted")
    if any(
        authorization[key]
        for key in (
            "end_to_end_training_route_connected",
            "disk_training_runner_connected",
            "end_to_end_inference_route_connected",
            "private_inference_route_connected",
            "report_generation_route_connected",
            "production_route_connected",
        )
    ):
        raise ValueError("incomplete v1 core cannot connect an execution route")
    if tuple(authorization["blocking_reason_codes"]) != _EXPECTED_BLOCKING_REASONS:
        raise ValueError("v1 core execution blocking reasons drifted")


def validate_clinical_eeg_ba_ieg_v1_core_freeze(
    value: object,
    *,
    require_trusted_contract: bool = False,
) -> dict[str, Any]:
    """Validate and defensively copy one BA-IEG v1 core freeze contract."""

    if type(value) is not dict:
        raise TypeError("BA-IEG v1 core freeze contract must be an object")
    candidate: dict[str, Any] = deepcopy(value)
    _reject_nonfinite(candidate)
    errors = sorted(
        _schema_validator().iter_errors(candidate),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        rendered = "; ".join(
            f"{_schema_error_path(error)}: {error.message}" for error in errors[:8]
        )
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(
            "clinical_eeg_ba_ieg_v1_core_freeze schema validation failed: "
            + rendered
        )

    if candidate["schema_version"] != BA_IEG_V1_CORE_FREEZE_SCHEMA_VERSION:
        raise ValueError("BA-IEG v1 core freeze schema version drifted")
    if candidate["contract_id"] != BA_IEG_V1_CORE_FREEZE_CONTRACT_ID:
        raise ValueError("BA-IEG v1 core freeze contract identity drifted")
    if (
        candidate["schema_binding"]["sha256"]
        != TRUSTED_BA_IEG_V1_CORE_FREEZE_SCHEMA_SHA256
    ):
        raise ValueError("BA-IEG v1 core freeze schema binding drifted")

    computed = ba_ieg_v1_core_freeze_contract_sha256(candidate)
    if candidate["contract_sha256"] != computed:
        raise ValueError("BA-IEG v1 core freeze content hash mismatch")
    if (
        require_trusted_contract
        and computed != TRUSTED_BA_IEG_V1_CORE_FREEZE_CONTRACT_SHA256
    ):
        raise ValueError("BA-IEG v1 core freeze is not the trusted default contract")

    _validate_code_bindings(candidate)
    _validate_method_invariants(candidate)
    _validate_firewall_report_performance(candidate)
    return candidate


def load_clinical_eeg_ba_ieg_v1_core_freeze(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a contract; the checked-in default additionally requires byte identity."""

    resolved = DEFAULT_BA_IEG_V1_CORE_FREEZE_PATH if path is None else Path(path)
    payload_bytes = _regular_file_bytes(resolved, "BA-IEG v1 core freeze config")
    trusted_default = path is None or resolved.resolve() == (
        DEFAULT_BA_IEG_V1_CORE_FREEZE_PATH.resolve()
    )
    if (
        trusted_default
        and _sha256_bytes(payload_bytes) != TRUSTED_BA_IEG_V1_CORE_FREEZE_FILE_SHA256
    ):
        raise ValueError("trusted BA-IEG v1 core freeze config file hash drifted")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("BA-IEG v1 core freeze config is not valid UTF-8 JSON") from error
    return validate_clinical_eeg_ba_ieg_v1_core_freeze(
        payload,
        require_trusted_contract=trusted_default,
    )


def assert_clinical_eeg_ba_ieg_v1_core_allows_end_to_end_inference(
    value: object,
) -> dict[str, Any]:
    """Fail closed until every required component is implemented and authorized."""

    candidate = validate_clinical_eeg_ba_ieg_v1_core_freeze(value)
    authorization = candidate["execution_authorization"]
    if not authorization["end_to_end_inference_route_connected"]:
        reasons = ", ".join(authorization["blocking_reason_codes"])
        raise ValueError(f"BA-IEG v1 core end-to-end inference is blocked: {reasons}")
    return candidate  # pragma: no cover - v1 checked-in contract is deliberately blocked


def assert_clinical_eeg_ba_ieg_v1_core_allows_end_to_end_training(
    value: object,
) -> dict[str, Any]:
    """Fail closed while the patient bridge is disconnected and no runner exists."""

    candidate = validate_clinical_eeg_ba_ieg_v1_core_freeze(value)
    authorization = candidate["execution_authorization"]
    if not authorization["end_to_end_training_route_connected"]:
        reasons = ", ".join(authorization["blocking_reason_codes"])
        raise ValueError(f"BA-IEG v1 core end-to-end training is blocked: {reasons}")
    return candidate  # pragma: no cover - v1 checked-in contract is deliberately blocked


def assert_clinical_eeg_ba_ieg_v1_core_allows_report_generation(
    value: object,
) -> dict[str, Any]:
    """Fail closed while the model-core contract has no report route."""

    candidate = validate_clinical_eeg_ba_ieg_v1_core_freeze(value)
    authorization = candidate["execution_authorization"]
    report = candidate["report_interface"]
    if not authorization["report_generation_route_connected"] or not report[
        "report_route_connected"
    ]:
        reasons = ", ".join(authorization["blocking_reason_codes"])
        raise ValueError(f"BA-IEG v1 core report generation is blocked: {reasons}")
    return candidate  # pragma: no cover - v1 checked-in contract is deliberately blocked


__all__ = [
    "BA_IEG_V1_CORE_FREEZE_CONTRACT_ID",
    "BA_IEG_V1_CORE_FREEZE_SCHEMA_VERSION",
    "DEFAULT_BA_IEG_V1_CORE_FREEZE_PATH",
    "DEFAULT_BA_IEG_V1_CORE_FREEZE_SCHEMA_PATH",
    "TRUSTED_BA_IEG_V1_CORE_FREEZE_CONTRACT_SHA256",
    "TRUSTED_BA_IEG_V1_CORE_FREEZE_FILE_SHA256",
    "TRUSTED_BA_IEG_V1_CORE_FREEZE_SCHEMA_SHA256",
    "assert_clinical_eeg_ba_ieg_v1_core_allows_end_to_end_training",
    "assert_clinical_eeg_ba_ieg_v1_core_allows_end_to_end_inference",
    "assert_clinical_eeg_ba_ieg_v1_core_allows_report_generation",
    "ba_ieg_v1_core_freeze_contract_sha256",
    "load_clinical_eeg_ba_ieg_v1_core_freeze",
    "validate_clinical_eeg_ba_ieg_v1_core_freeze",
]
