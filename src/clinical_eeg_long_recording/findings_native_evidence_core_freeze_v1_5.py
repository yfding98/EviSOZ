"""Validator for the additive EEG-native Findings core freeze v1.5.

This module validates an architecture and evidence-authority contract.  It
does not implement a trained detector, a clinical term classifier, a SOZ
ranker, or a report generator, and it must not be used as performance evidence.
The validator intentionally binds the additive freeze to the exact parent
files that were audited without modifying the frozen v1.4 contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE_PATH = (
    ROOT / "configs" / "clinical_eeg_findings_native_evidence_core_freeze_v1_5.json"
)
SCHEMA_VERSION = "clinical_eeg_findings_native_evidence_core_freeze_v1_5"
FREEZE_ID = "CLINICAL-EEG-FINDINGS-NATIVE-EVIDENCE-CORE-FREEZE-V1.5-20260824"
DEFAULT_FREEZE_SHA256 = (
    "d460dafdfe5e76a90369ec0939becb7f92e9beb28ee40e12b72eccdfbcdc1ed1"
)

_FOUR_STATES = [
    "present",
    "uncertain",
    "not_evaluable",
    "absent_with_opportunity",
]
_NAMESPACES = ["proposal", "measurement", "qualified_assertion"]
_FACT_CHAIN = ["proposal", "native_measurement", "qualified_assertion"]
_PERMISSION_LANES = [
    "onset_causal",
    "course_offline",
    "matched_context",
    "limitation",
]
_PROPOSAL_FAMILIES = [
    "P01_QC_ARTIFACT_OPPORTUNITY",
    "P02_EVENT_STATE_BOUNDARY",
    "P03_COMPONENT_CYCLE_CHANGE_POINT",
    "P04_PER_UNIT_INVOLVEMENT_EARLIEST_FIELD",
]
_QUERY_TRANSITIONS = [
    "first_observed",
    "first_observed_and_stabilized",
    "updated_unstable",
    "stabilized",
    "changed_after_stabilization",
    "invalidated",
]
_TRAINING_STAGES = [
    "T0_DETERMINISTIC_MEASUREMENT",
    "T1_SOURCE_ONLY_SELF_SUPERVISION",
    "T2_SCOPE_LIMITED_PUBLIC_WEAK_SUPERVISION",
    "T3_CROSSFITTED_QUERY_ROUTER",
    "T4_FRESH_TERM_QUALIFICATION",
]


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


def _exact_list(value: object, expected: list[str], context: str) -> None:
    if value != expected:
        raise ValueError(f"{context} drifted")


def _unique_strings(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{context} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{context} contains duplicates")
    return list(value)


def _validate_parent_bindings(value: object) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("parent_bindings must contain the four audited parents")
    roles: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise TypeError(f"parent_bindings[{index}] must be an object")
        path = ROOT / str(row.get("path", ""))
        role = str(row.get("role", ""))
        expected = str(row.get("file_sha256", ""))
        if not path.is_file():
            raise ValueError(f"parent binding does not exist: {path}")
        if len(expected) != 64 or _file_sha256(path) != expected:
            raise ValueError(f"parent binding hash drifted: {path}")
        if not role or role in roles:
            raise ValueError("parent binding roles must be non-empty and unique")
        roles.add(role)


def validate_findings_native_evidence_core_freeze_v1_5(
    value: Mapping[str, Any],
    *,
    trusted_freeze_sha256: str = DEFAULT_FREEZE_SHA256,
) -> dict[str, Any]:
    """Validate and return a defensive copy of the additive freeze."""

    freeze = deepcopy(dict(value))
    if freeze.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("native Findings freeze schema drifted")
    if freeze.get("freeze_id") != FREEZE_ID:
        raise ValueError("native Findings freeze identifier drifted")
    if freeze.get("freeze_status") != (
        "additive_interface_frozen_implementation_partial_not_trained_"
        "not_performance_admitted"
    ):
        raise ValueError("native Findings evidence status drifted")

    observed_hash = freeze.get("freeze_sha256")
    body = deepcopy(freeze)
    body.pop("freeze_sha256", None)
    replayed_hash = _canonical_sha256(body)
    if observed_hash != replayed_hash or observed_hash != trusted_freeze_sha256:
        raise ValueError("native Findings freeze does not replay exactly")

    scope = freeze["scope"]
    for false_claim in (
        "not_cortical_soz_ez_or_surgical_target",
        "not_12_independent_clinical_heads",
        "report_language_optimization_paused",
    ):
        if scope.get(false_claim) is not True:
            raise ValueError(f"scope firewall drifted: {false_claim}")
    if scope.get("clinical_or_production_use") is not False:
        raise ValueError("clinical or production permission cannot be opened")

    _validate_parent_bindings(freeze["parent_bindings"])

    firewall = freeze["forward_source_firewall"]
    if firewall.get("EEG_samples_used") is not True:
        raise ValueError("EEG samples must remain the forward signal source")
    if firewall.get("allowlisted_acquisition_metadata_used") is not True:
        raise ValueError("allowlisted acquisition metadata must remain explicit")
    for key, enabled in firewall.items():
        if key in {"EEG_samples_used", "allowlisted_acquisition_metadata_used"}:
            continue
        if enabled is not False:
            raise ValueError(f"non-EEG forward source permission opened: {key}")

    bottom = freeze["minimal_bottom_layer"]
    proposal = bottom["proposal_contract"]
    proposal_ids = [row["proposal_family_id"] for row in proposal["families"]]
    _exact_list(proposal_ids, _PROPOSAL_FAMILIES, "proposal family roster")
    _unique_strings(proposal["mandatory_fields"], "proposal mandatory fields")
    if not {
        "attention_to_finding",
        "saliency_to_finding",
        "detector_score_to_positive_onset_rank",
    }.issubset(set(proposal["forbidden_promotions"])):
        raise ValueError("proposal-to-fact firewalls are incomplete")

    measurement = bottom["native_measurement_contract"]
    _exact_list(measurement["fact_chain"], _FACT_CHAIN, "fact chain")
    if measurement["deterministic_native_remeasurement_required"] is not True:
        raise ValueError("native remeasurement cannot be disabled")
    _unique_strings(measurement["mandatory_fields"], "measurement mandatory fields")
    if measurement["physical_unit_requirements"][
        "whole_bipolar_lead_identity_preserved"
    ] is not True:
        raise ValueError("whole bipolar lead identity must be preserved")
    if measurement["physical_unit_requirements"][
        "bipolar_endpoint_attribution_forbidden"
    ] is not True:
        raise ValueError("bipolar endpoint attribution firewall drifted")

    ledger = bottom["shared_physical_time_ledger_contract"]
    if ledger["append_only"] is not True:
        raise ValueError("the shared native ledger must be append-only")
    _exact_list(ledger["status_vocabulary"], _FOUR_STATES, "ledger states")
    _exact_list(ledger["namespace_vocabulary"], _NAMESPACES, "ledger namespaces")
    _exact_list(ledger["permission_lanes"], _PERMISSION_LANES, "permission lanes")
    if ledger["current_unified_implementation_status"] != (
        "not_implemented_existing_components_cover_binding_query_transition_"
        "and_surface_projection_separately"
    ):
        raise ValueError("unified ledger implementation truth drifted")
    for path_value in ledger["existing_components_reused"]:
        if not (ROOT / path_value).is_file():
            raise ValueError(f"reused ledger component is missing: {path_value}")

    four_state = freeze["four_state_decision_rule"]
    _exact_list(
        [row["emit"] for row in four_state["ordered_evaluation"]],
        [
            "not_evaluable",
            "present",
            "absent_with_opportunity",
            "uncertain",
        ],
        "four-state decision order",
    )
    for guard in (
        "not_evaluable_is_negative",
        "technical_failure_is_negative",
        "missing_candidate_is_negative",
        "clinical_absent_with_opportunity_currently_authorized",
    ):
        if four_state[guard] is not False:
            raise ValueError(f"four-state fail-closed guard drifted: {guard}")
    if four_state["automated_clinical_term_allowlist"] != []:
        raise ValueError("automated clinical term allowlist must remain empty")

    slots = freeze["slot_contract"]
    if [row["slot_id"] for row in slots] != [f"S{i:02d}" for i in range(1, 13)]:
        raise ValueError("slot roster must be exactly ordered S01-S12")
    for row in slots:
        for producer in row["primary_producers"]:
            if not (ROOT / producer).is_file():
                raise ValueError(f"slot producer is missing: {producer}")
        if row["current_positive_onset_authorized"] is not False:
            raise ValueError("current positive onset authority cannot be claimed")
        if row["late_course_may_promote_positive_onset"] is not False:
            raise ValueError("late course cannot promote positive onset")
        if row["clinical_absence_authorized"] is not False:
            raise ValueError("clinical absence cannot be authorized")
        if row["report_promotion_authorized"] is not False:
            raise ValueError("report promotion cannot be authorized")
        if row["uncertain_authorized"] is not True or row["not_evaluable_required"] is not True:
            raise ValueError("each slot must retain uncertain and not-evaluable states")
    for row in slots[9:]:
        if row["future_positive_onset_permission"] != "forbidden":
            raise ValueError("S10-S12 may never promote positive onset")
    if not all("trigger_atom" in row["positive_onset_role"] for row in slots[2:6]):
        raise ValueError("S03-S06 must remain conditional trigger atoms")
    if slots[6]["positive_onset_role"] != "primary_temporal_typed_onset_hypothesis":
        raise ValueError("S07 temporal onset role drifted")
    if slots[7]["positive_onset_role"] != "primary_spatial_typed_onset_hypothesis":
        raise ValueError("S08 spatial onset role drifted")

    algorithms = freeze["cross_slot_algorithms"]
    onset = algorithms["onset_trigger_attribution"]
    _exact_list(onset["candidate_scope"], ["S07", "S08"], "onset candidate scope")
    _exact_list(
        onset["eligible_trigger_slots"],
        ["S03", "S04", "S05", "S06"],
        "onset trigger slots",
    )
    if onset["implementation_status"] != "not_implemented_contract_frozen":
        raise ValueError("onset attribution implementation status drifted")
    if onset["attention_or_saliency_accepted"] is not False:
        raise ValueError("attention or saliency cannot be onset evidence")
    if onset["thresholds"]["threshold_registry_materialized"] is not False:
        raise ValueError("unmaterialized onset thresholds cannot be claimed frozen")

    closure = algorithms["query_indexed_evidence_closure"]
    _exact_list(closure["atom_transitions"], _QUERY_TRANSITIONS, "query transitions")
    if closure["implementation_status"] != (
        "TEST_contract_implemented_no_real_native_EEG_multistep_rollout"
    ):
        raise ValueError("query-closure implementation status drifted")
    if not (ROOT / closure["existing_component"]).is_file():
        raise ValueError("query-closure component is missing")
    if closure["trajectory_delta_is_additive_across_steps"] is not False:
        raise ValueError("query trajectory deltas are non-additive")
    if closure["each_next_state_requires_actual_reveal_and_full_recompute"] is not True:
        raise ValueError("query closure must recompute after actual reveal")
    if closure["attention_or_action_value_is_a_finding"] is not False:
        raise ValueError("action value cannot become a Finding")
    seed = closure["initial_support_policy"]
    if seed["default_minus12_plus48_forbidden"] is not True:
        raise ValueError("A1 initial support cannot default to minus12/plus48")
    if seed["exact_seconds_frozen_or_optimal"] is not False:
        raise ValueError("no initial seed duration has been established as optimal")
    if not seed["profile_selection"].startswith("pre_register_on_source_train_inner_dev"):
        raise ValueError("initial seed profile must be selected without evaluation leakage")
    if "candidate_centered_minus12_plus48_comparator" not in seed[
        "required_seed_length_ablation"
    ]:
        raise ValueError("minus12/plus48 must remain a paired seed comparator")
    if not {
        "reference_onset_annotation",
        "EDF_annotation",
        "Excel_or_spreadsheet",
        "doctor_label_or_report",
        "SOZ_label",
    }.issubset(set(seed["forbidden_seed_inputs"])):
        raise ValueError("initial seed source firewall is incomplete")
    if "query_index_zero" not in seed["q0_semantics"]:
        raise ValueError("the registered initial seed must define q0")
    if "query_more_EEG" not in seed["insufficient_seed_action"]:
        raise ValueError("an insufficient seed must remain open rather than fail")

    rank = algorithms["rank_contribution_counterfactual"]
    if rank["implementation_status"] != "not_implemented_contract_frozen":
        raise ValueError("rank counterfactual implementation status drifted")
    if rank["single_atom_deltas_may_be_summed"] is not False:
        raise ValueError("single-atom deltas cannot be summed")
    if rank["Shapley_or_causal_physiology_claimed"] is not False:
        raise ValueError("rank attribution is not a physiological causal claim")
    if rank["thresholds"]["threshold_registry_materialized"] is not False:
        raise ValueError("unmaterialized rank thresholds cannot be claimed frozen")
    required_counterfactual_fragments = {
        "without_that_atom",
        "single_atom_insertion",
        "joint_deletion",
        "after_the_locked_prefix",
        "every_legal_reference",
        "back_off",
    }
    joined_steps = " ".join(rank["exact_steps"])
    if not all(fragment in joined_steps for fragment in required_counterfactual_fragments):
        raise ValueError("rank counterfactual roster is incomplete")

    mandatory_output = freeze["mandatory_non_abstaining_spatial_output"]
    if mandatory_output["research_broad_pattern_id"] != (
        "widespread_bilateral_near_synchronous_scalp_onset"
    ):
        raise ValueError("research broad-pattern identifier drifted")
    required_outputs = set(mandatory_output["required_event_outputs"])
    if not {
        "laterality_probability_distribution_with_uncertainty_mass",
        "region_probability_distribution_with_resolution_status",
        "typed_electrode_or_whole_bipolar_lead_probability_distribution_over_all_evaluable_units",
        "positive_trigger_atom_bindings_or_typed_missing_binding_reason",
    }.issubset(required_outputs):
        raise ValueError("mandatory non-abstaining spatial distributions are incomplete")
    if "may_not_create_rescue_or_reorder" not in mandatory_output["S10_role"]:
        raise ValueError("S10 cannot create or rescue a broad onset candidate")
    forbidden_defaults = set(mandatory_output["forbidden_default_fallbacks"])
    if not {"diffuse_onset", "generalized_onset", "generalized_epilepsy"}.issubset(
        forbidden_defaults
    ):
        raise ValueError("diffuse/generalized default fallbacks are not fully closed")
    ita = mandatory_output["complete_ITA_alignment"]
    if ita["one_distribution_per_detected_occurrence"] is not True:
        raise ValueError("complete ITA requires one distribution per occurrence")
    if ita["top_N_truncation_before_record_aggregation"] is not False:
        raise ValueError("complete ITA forbids pre-aggregation top-N truncation")
    if ita["qualified_only_secondary_cannot_replace_primary"] is not True:
        raise ValueError("qualified-only aggregation cannot replace complete ITA")
    if mandatory_output["clinical_generalized_or_diffuse_diagnosis_authorized"] is not False:
        raise ValueError("a research broad pattern is not a clinical generalized diagnosis")

    terms = freeze["clinical_term_firewall"]
    if terms["automated_allowlist"] != [] or terms["report_text_authorized"] is not False:
        raise ValueError("clinical term/report firewall opened")
    protected = set(terms["protected_or_forbidden_examples"])
    if not {
        "spike",
        "IED",
        "rhythmic_theta_activity",
        "definite_evolution",
        "low_voltage_fast_activity",
        "cortical_SOZ",
        "epileptogenic_zone",
        "sleep_stage_or_sleep_activation",
    }.issubset(protected):
        raise ValueError("protected clinical term examples are incomplete")
    if terms["protected_terms_may_be_new_parent_slots"] is not False:
        raise ValueError("protected terms cannot become new parent slots")
    if terms["protected_terms_must_be_conditional_composites_of_S03_to_S06"] is not True:
        raise ValueError("protected term composition boundary drifted")

    ladder = freeze["evaluation_ladder"]
    _exact_list([row["level_id"] for row in ladder], [f"F{i}" for i in range(5)], "F0-F4 ladder")
    if any(row["requires_dense_clinical_GT"] for row in ladder[:4]):
        raise ValueError("F0-F3 must not require dense clinical GT")
    if ladder[4]["requires_dense_clinical_GT"] is not True:
        raise ValueError("F4 must be the fresh expert/clinical endpoint")
    if ladder[3]["current_status"] != "not_implemented":
        raise ValueError("F3 cannot be claimed implemented")

    training = freeze["training_without_dense_GT"]
    _exact_list(
        [row["stage_id"] for row in training["stages"]],
        _TRAINING_STAGES,
        "training stage roster",
    )
    weak = training["stages"][2]
    if weak["unlabelled_channels_are_negative"] is not False:
        raise ValueError("unlabelled channels cannot be trained as negatives")
    if weak["spread_labels_are_separate_soft_relevance"] is not True:
        raise ValueError("spread labels must remain separate soft relevance")
    router = training["stages"][3]
    if router["self_confidence_entropy_attention_or_saliency_as_target"] is not False:
        raise ValueError("router self-certification is forbidden")
    expert = training["stages"][4]
    if expert["currently_available"] is not False:
        raise ValueError("fresh expert qualification cannot be claimed available")

    truth = freeze["implementation_truth"]
    for key in (
        "complete_real_12_slot_event_card",
        "clinical_term_qualification",
        "patient_held_out_SOZ_performance",
        "performance_or_medical_effect_claimed",
    ):
        if truth[key] is not False:
            raise ValueError(f"implementation truth overclaim: {key}")
    for key, permitted in freeze["scientific_permissions"].items():
        if permitted is not False:
            raise ValueError(f"scientific permission cannot be opened: {key}")

    return freeze


def load_findings_native_evidence_core_freeze_v1_5(
    path: Path | str = DEFAULT_FREEZE_PATH,
) -> dict[str, Any]:
    """Load and validate the default additive freeze."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("native Findings freeze must be a JSON object")
    return validate_findings_native_evidence_core_freeze_v1_5(payload)


__all__ = [
    "DEFAULT_FREEZE_PATH",
    "DEFAULT_FREEZE_SHA256",
    "FREEZE_ID",
    "SCHEMA_VERSION",
    "load_findings_native_evidence_core_freeze_v1_5",
    "validate_findings_native_evidence_core_freeze_v1_5",
]
