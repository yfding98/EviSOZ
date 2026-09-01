"""Fail-closed loader for the additive clinical EEG v1.3-min addendum.

The v1.2 submission profile and all of its frozen receipts remain immutable.
This addendum only narrows the future primary experiment after the adversarial
freeze audit: it separates A0/A1 candidate arms, makes single-bout topology the
primary, requires an earliest-prefix K3 lock, isolates the three loss gradient
authorities, preserves both ITA and qualified-only ranking tracks, and assigns
the 12-slot Findings API to core/nonblocking/deferred tiers.

Loading this method contract does not authorize roster materialization,
training, evaluation, inference, report generation, performance claims, or
clinical use.  Every such permission is deliberately false.  Gate status may
only be advanced by a future additive receipt, never by editing or rehashing
this file.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence

from .submission_method_profile_v1_2 import (
    DEFAULT_SUBMISSION_METHOD_PROFILE_PATH_V1_2,
    SUBMISSION_METHOD_PROFILE_ID_V1_2,
    TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2,
    load_submission_method_profile_v1_2,
)


V1_3_MIN_ADVERSARIAL_ADDENDUM_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_v1_3_min_adversarial_addendum_v1"
)
V1_3_MIN_ADVERSARIAL_ADDENDUM_ID: Final[str] = (
    "CLINICAL-EEG-V1.3-MIN-ADVERSARIAL-ADDENDUM-20260824"
)

_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_V1_3_MIN_ADVERSARIAL_ADDENDUM_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_v1_3_min_adversarial_addendum.json"
)
V1_3_MIN_ADVERSARIAL_AUDIT_PATH: Final[Path] = (
    _ROOT
    / "research"
    / "02_method"
    / "clinical_eeg_v1_2_variable_support_v1_4_adversarial_freeze_audit_20260824_zh.md"
)

# Compiled trust anchors.  They are intentionally not learned from the files at
# runtime.  Filled from the checked-in canonical bytes after receipt creation.
TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_RECEIPT_SHA256: Final[str] = (
    "0f7a3054fb8087842f6f7afde6e2c5a7320f28ad70fdc0c3c6605538451fb05c"
)
TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_FILE_SHA256: Final[str] = (
    "d771a8759172453731b07d6fb29de2908dc935f00010783d6b6f3486c88ebf4a"
)
TRUSTED_V1_3_MIN_ADVERSARIAL_AUDIT_FILE_SHA256: Final[str] = (
    "35c9f3af0605fec7266aa8e4f05fde85d5ea885648a561e7f68e1540430a9452"
)
TRUSTED_V1_2_SUBMISSION_PROFILE_FILE_SHA256: Final[str] = (
    "c3e37d170be6cf355faba4835e76c5e41435220cc4c52775ae387d674a46648e"
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_SECTION_HEADING: Final[str] = (
    "## 15. revised primary 的最小机器可实现规格"
)
_TOP_LEVEL_KEYS: Final[set[str]] = {
    "schema_version",
    "addendum_id",
    "status",
    "additive_bindings",
    "candidate_arms",
    "outer_rule_primary",
    "temporal_primary",
    "onset_identity_gate",
    "loss_authority",
    "record_ranking",
    "findings_tiers",
    "promotion_gates",
    "source_firewall",
    "execution_status",
    "execution_permissions",
    "scientific_permissions",
    "receipt_sha256",
}
_CANDIDATE_ARM_ORDER: Final[tuple[str, ...]] = (
    "A0_oracle_upper_bound",
    "A0_jitter_censor_development",
    "A1_oof_train",
    "A1_frozen_dev_eval",
)
_PRIMARY_TRANSITION_EDGES: Final[tuple[str, ...]] = (
    "S0_to_S1",
    "S1_to_S2",
    "S2_to_S3",
    "S0_to_S2_short_event",
    "S1_to_S3_short_return",
)
_DISABLED_PRIMARY_EDGES: Final[tuple[str, ...]] = (
    "S3_to_S0_clean_return",
    "S3_to_S1_reentry",
    "S3_to_S2_reentry_short",
)
_EXPECTED_LOSS_UPDATES: Final[dict[str, tuple[str, ...]]] = {
    "L_segmental": (
        "causal_global_boundary",
        "detached_offline_course",
    ),
    "L_typed_boundary_MIL": ("typed_boundary_proposal",),
    "L_patient_positive_set": ("identity_adapter", "rank_head"),
}
_GATE_ORDER: Final[tuple[str, ...]] = (
    "G0",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
)
_DEFERRED_CLINICAL_TERMS: Final[tuple[str, ...]] = (
    "spike_sharp_wave_ied",
    "pathological_theta_or_focal_slowing",
    "clinical_rhythmic_or_periodic_discharge",
    "acns_definite_evolution_or_electrographic_seizure",
    "spread_or_propagation",
    "termination_or_postictal",
    "lvfa_hfo_or_dc_shift",
    "generalized_or_diffuse_onset",
    "cortical_soz_or_ez",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def v1_3_min_adversarial_addendum_self_sha256(
    value: Mapping[str, object],
) -> str:
    """Return the canonical self-hash, excluding only ``receipt_sha256``."""

    if not isinstance(value, Mapping):
        raise TypeError("v1.3-min adversarial addendum must be an object")
    body = deepcopy(dict(value))
    body.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _regular_file_bytes(path: Path, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path.read_bytes()


def _load_strict_json(path: Path, context: str) -> dict[str, Any]:
    raw = _regular_file_bytes(path, context)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite JSON token {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _safe_project_file(relative: object, context: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{context} must be a relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{context} is not a canonical relative path")
    root = _ROOT.resolve(strict=True)
    unresolved = root.joinpath(*pure.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    candidate = unresolved.resolve(strict=True)
    candidate.relative_to(root)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{context} must resolve to a regular non-symlink file")
    return candidate


def _require_equal(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context} drifted")


def _require_false_mapping(value: object, context: str) -> dict[str, bool]:
    if type(value) is not dict or not value:
        raise TypeError(f"{context} must be a non-empty object")
    if any(type(item) is not bool or item for item in value.values()):
        raise ValueError(f"every {context} flag must remain false")
    return value


@lru_cache(maxsize=1)
def _trusted_default() -> dict[str, Any]:
    raw = _regular_file_bytes(
        DEFAULT_V1_3_MIN_ADVERSARIAL_ADDENDUM_PATH,
        "trusted v1.3-min adversarial addendum",
    )
    if _sha256_bytes(raw) != TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_FILE_SHA256:
        raise ValueError("trusted v1.3-min adversarial addendum file hash drifted")
    return _load_strict_json(
        DEFAULT_V1_3_MIN_ADVERSARIAL_ADDENDUM_PATH,
        "trusted v1.3-min adversarial addendum",
    )


def _validate_additive_bindings(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "base_profile",
        "adversarial_audit",
        "modifies_existing_frozen_profile_or_hash",
        "precedence",
    }:
        raise ValueError("additive_bindings keys drifted")
    if value["modifies_existing_frozen_profile_or_hash"] is not False:
        raise ValueError("v1.3-min must not mutate any existing frozen profile or hash")
    _require_equal(
        value["precedence"],
        "narrows_v1_2_for_v1_3_min_without_promoting_any_existing_permission",
        "additive precedence",
    )

    base = value["base_profile"]
    expected_base = {
        "path": "configs/clinical_eeg_submission_method_profile_v1_2.json",
        "profile_id": SUBMISSION_METHOD_PROFILE_ID_V1_2,
        "receipt_sha256": TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2,
        "file_sha256": TRUSTED_V1_2_SUBMISSION_PROFILE_FILE_SHA256,
        "mutation_required": False,
    }
    _require_equal(base, expected_base, "base v1.2 profile binding")
    base_path = _safe_project_file(base["path"], "base_profile.path")
    if base_path != DEFAULT_SUBMISSION_METHOD_PROFILE_PATH_V1_2.resolve(strict=True):
        raise ValueError("base profile path does not resolve to the frozen v1.2 profile")
    if _sha256_bytes(_regular_file_bytes(base_path, "base v1.2 profile")) != (
        TRUSTED_V1_2_SUBMISSION_PROFILE_FILE_SHA256
    ):
        raise ValueError("base v1.2 profile file hash drifted")
    profile = load_submission_method_profile_v1_2()
    if profile["receipt_sha256"] != (
        TRUSTED_SUBMISSION_METHOD_PROFILE_RECEIPT_SHA256_V1_2
    ):
        raise ValueError("base v1.2 profile receipt replay failed")

    audit = value["adversarial_audit"]
    expected_audit = {
        "path": (
            "research/02_method/"
            "clinical_eeg_v1_2_variable_support_v1_4_"
            "adversarial_freeze_audit_20260824_zh.md"
        ),
        "file_sha256": TRUSTED_V1_3_MIN_ADVERSARIAL_AUDIT_FILE_SHA256,
        "normative_section_heading": _AUDIT_SECTION_HEADING,
        "role": "normative_source_for_this_narrowing_addendum",
    }
    _require_equal(audit, expected_audit, "adversarial audit binding")
    audit_path = _safe_project_file(audit["path"], "adversarial_audit.path")
    if audit_path != V1_3_MIN_ADVERSARIAL_AUDIT_PATH.resolve(strict=True):
        raise ValueError("adversarial audit path drifted")
    audit_raw = _regular_file_bytes(audit_path, "adversarial audit")
    if _sha256_bytes(audit_raw) != TRUSTED_V1_3_MIN_ADVERSARIAL_AUDIT_FILE_SHA256:
        raise ValueError("adversarial audit file hash drifted")
    try:
        audit_text = audit_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("adversarial audit is not valid UTF-8") from error
    if _AUDIT_SECTION_HEADING not in audit_text:
        raise ValueError("adversarial audit normative section is missing")


def _validate_candidate_arms(value: object) -> None:
    if type(value) is not dict or tuple(value) != _CANDIDATE_ARM_ORDER:
        raise ValueError("candidate arm roster or order drifted")
    if any(arm.get("current_execution_authorized") is not False for arm in value.values()):
        raise ValueError("candidate-arm execution permission must remain false")
    a0 = value["A0_oracle_upper_bound"]
    if not (
        a0["reference_interval_defines_support"] is True
        and a0["train_primary_boundary"] is False
        and a0["end_to_end_claim"] is False
        and a0["permitted_endpoint"] == "conditional_localization_upper_bound"
    ):
        raise ValueError("A0 oracle upper-bound authority drifted")
    jitter = value["A0_jitter_censor_development"]
    _require_equal(
        jitter["required_example_classes"],
        [
            "observed_onset",
            "left_censored",
            "right_censored",
            "no_onset_in_support",
        ],
        "A0-jitter censor class roster",
    )
    _require_equal(
        jitter["shortcut_controls"],
        [
            "position_only_baseline",
            "eeg_shuffled_with_position_preserved_baseline",
        ],
        "A0-jitter shortcut controls",
    )
    a1 = value["A1_oof_train"]
    if not (
        a1["provider_predictions"] == "patient_oof_reference_free"
        and a1["prediction_freeze_before_target_join"] == "required"
        and a1["zero_candidate_records_retained"] is True
    ):
        raise ValueError("A1-OOF prediction-first boundary drifted")
    a1_eval = value["A1_frozen_dev_eval"]
    if not (
        a1_eval["provider_checkpoint_and_decoder_frozen"] == "required"
        and a1_eval["reference_join_after_prediction_freeze"] == "required"
        and a1_eval["threshold_selection"] == "source_dev_only"
        and a1_eval["one_shot_source_eval"] is True
    ):
        raise ValueError("A1 frozen dev/eval boundary drifted")


def _validate_outer_temporal_and_onset(candidate: Mapping[str, Any]) -> None:
    outer = candidate["outer_rule_primary"]
    _require_equal(outer["geometric_chunk_seconds"], [2, 4, 8, 16, 32], "outer geometric chunks")
    _require_equal(outer["extension_logic"], "logical_or_any_unresolved_risk", "outer OR extension")
    _require_equal(
        outer["normal_stop_logic"],
        "logical_and_all_risks_resolved_for_calibrated_consecutive_recomputations",
        "outer AND stop",
    )
    if not (
        outer["distant_context_may_create_or_reorder_positive_onset"] is False
        and outer["hard_cap_receipt_available"] is False
        and outer["two_recompute_stability_threshold_receipt_available"] is False
        and outer["current_execution_authorized"] is False
    ):
        raise ValueError("outer rule readiness or onset authority was promoted")
    forbidden_actions = {
        "soz_or_typed_unit_rank",
        "doctor_or_private_target",
        "earliest_field_downstream_utility",
        "course_or_return_downstream_utility",
        "cross_reference_downstream_utility",
    }
    if set(outer["excluded_action_features_v1"]) != forbidden_actions:
        raise ValueError("outer excluded action-feature roster drifted")

    temporal = candidate["temporal_primary"]
    _require_equal(temporal["p0_scales_seconds"], [1, 4, 16], "P0 physical-time scales")
    _require_equal(temporal["semimarkov_states"], ["S0", "S1", "S2", "S3"], "semi-Markov states")
    _require_equal(temporal["primary_transition_edges"], list(_PRIMARY_TRANSITION_EDGES), "single-bout primary edges")
    _require_equal(temporal["disabled_primary_edges"], list(_DISABLED_PRIMARY_EDGES), "disabled recurrent primary edges")
    if not (
        temporal["onset_encoder"] == "causal_gru_hidden64"
        and temporal["course_encoder"] == "detached_offline_bigru_hidden64"
        and temporal["recurrent_or_multiple_bout_topology_role"] == "shadow_only"
        and temporal["left_right_censoring"] == "exact"
        and temporal["primary_checkpoint_admitted"] is False
        and temporal["current_execution_authorized"] is False
    ):
        raise ValueError("temporal primary or single-bout authority drifted")

    gate = candidate["onset_identity_gate"]
    decision = gate["onset_decision"]
    if not (
        decision["type"] == "earliest_prefix_locked_decision"
        and decision["source"] == "global_causal_hazard_only"
        and decision["threshold_receipt_available"] is False
        and decision["maximum_interval_width_receipt_available"] is False
        and decision["late_revision_of_locked_positive_interval"] == "forbidden"
        and decision["never_locked_route"] == "ita_low_confidence_only_not_evidence_grade"
    ):
        raise ValueError("earliest-prefix onset lock authority drifted")
    if not (
        gate["K3_primary"]["horizon_seconds"] == 3
        and gate["K5_shadow"]["horizon_seconds"] == 5
        and gate["K5_shadow"]["positive_primary_rank"] == "forbidden"
        and gate["full_course"]["positive_primary_rank"] == "forbidden"
        and gate["counterfactual_receipt_available"] is False
        and gate["current_execution_authorized"] is False
    ):
        raise ValueError("K3/K5/full-course permission split drifted")


def _validate_loss_and_ranking(candidate: Mapping[str, Any]) -> None:
    losses = candidate["loss_authority"]
    for name, updates in _EXPECTED_LOSS_UPDATES.items():
        _require_equal(losses[name]["may_update"], list(updates), f"{name} gradient authority")
    if not (
        losses["L_typed_boundary_MIL"]["event_positive_copied_to_each_typed_unit"] is False
        and losses["L_patient_positive_set"]["application_count_per_complete_patient"] == 1
        and losses["L_patient_positive_set"]["unlabelled_channel_semantics"] == "unknown_not_negative"
        and losses["registered_in_one_composite_trainable_module"] is False
        and losses["per_parameter_gradient_audit_receipt_available"] is False
        and losses["optimizer_execution_authorized"] is False
    ):
        raise ValueError("three-loss gradient or execution authority drifted")
    forbidden_patient_updates = {
        "global_onset_gate",
        "global_boundary",
        "causal_backbone",
        "offline_backbone",
    }
    if set(losses["L_patient_positive_set"]["may_not_update"]) != forbidden_patient_updates:
        raise ValueError("patient-positive-set forbidden gradient roster drifted")

    ranking = candidate["record_ranking"]
    if not (
        ranking["occurrence_deduplication"] == "required"
        and ranking["base_pool"] == "equal_event_capped_log_mean_exp"
        and ranking["intention_to_analyze_primary_research"]["candidate_scope"]
        == "all_eeg_evaluable_candidate_occurrences"
        and ranking["intention_to_analyze_primary_research"]
        ["event_presence_and_quality_uncertainty_retained"] is True
        and ranking["intention_to_analyze_primary_research"]
        ["hard_qualification_threshold_may_silently_drop_candidates"] is False
        and ranking["qualified_only_safety_secondary"]["candidate_scope"]
        == "qualified_occurrences_only"
        and ranking["whole_bipolar_endpoint_attribution"] == "forbidden"
        and ranking["performance_evaluation_authorized"] is False
    ):
        raise ValueError("ITA/qualified ranking authority drifted")


def _validate_findings_gates_and_firewall(candidate: Mapping[str, Any]) -> None:
    findings = candidate["findings_tiers"]
    if not (
        len(findings["core"]) == 10
        and len(findings["nonblocking_candidate"]) == 3
        and tuple(findings["deferred_clinical_terms"]) == _DEFERRED_CLINICAL_TERMS
        and findings["twelve_slot_api_is_twelve_trained_heads"] is False
        and findings["native_measurement_may_promote_deferred_clinical_term"] is False
        and findings["current_real_native_g5_admission"] is False
    ):
        raise ValueError("Findings tier or clinical-term authority drifted")

    gates = candidate["promotion_gates"]
    _require_equal(gates["order"], list(_GATE_ORDER), "G0-G6 order")
    for gate_id in _GATE_ORDER[:6]:
        if gates[gate_id]["status"] != "not_passed":
            raise ValueError(f"{gate_id} must remain not_passed")
    if gates["G6"]["status"] != "blocked_by_G0_through_G5":
        raise ValueError("G6 must remain blocked by G0 through G5")
    if not (
        gates["gate_transition_authorized_by_this_addendum"] is False
        and gates["automatic_promotion_from_test_pass_count"] is False
    ):
        raise ValueError("promotion-gate transition authority was opened")

    firewall = candidate["source_firewall"]
    if not (
        firewall["eeg_samples_used_in_inference"] is True
        and firewall["allowlisted_acquisition_metadata_used_in_inference"] is True
        and firewall[
            "reference_labels_may_join_only_after_prediction_freeze_for_training_targets_or_evaluation"
        ] is True
        and firewall[
            "reference_labels_may_define_provider_features_candidate_scores_or_inference_inputs"
        ] is False
    ):
        raise ValueError("EEG-only prediction-first firewall drifted")
    prohibited = (
        "edf_annotations_used_in_inference",
        "spreadsheets_used_in_inference",
        "doctor_labels_or_reports_used_in_inference",
        "private_targets_used_in_model_forward",
        "clinical_text_used_in_inference",
        "video_or_behavior_used_in_inference",
        "sleep_or_activation_information_used_in_inference",
        "ecg_or_other_physiology_used_in_inference",
    )
    if any(firewall[field] is not False for field in prohibited):
        raise ValueError("a prohibited non-EEG inference source was enabled")


def validate_v1_3_min_adversarial_addendum(
    value: Mapping[str, object],
    *,
    verify_dependencies: bool = True,
    require_trusted_receipt: bool = True,
) -> dict[str, Any]:
    """Validate the immutable additive contract and keep every gate closed."""

    if type(value) is not dict:
        raise TypeError("v1.3-min adversarial addendum must be a JSON object")
    candidate = deepcopy(value)
    if set(candidate) != _TOP_LEVEL_KEYS:
        missing = _TOP_LEVEL_KEYS - set(candidate)
        unknown = set(candidate) - _TOP_LEVEL_KEYS
        raise ValueError(
            "v1.3-min addendum keys drifted; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    _require_equal(
        candidate["schema_version"],
        V1_3_MIN_ADVERSARIAL_ADDENDUM_SCHEMA_VERSION,
        "schema_version",
    )
    _require_equal(candidate["addendum_id"], V1_3_MIN_ADVERSARIAL_ADDENDUM_ID, "addendum_id")
    _require_equal(
        candidate["status"],
        "additive_method_contract_only_not_implemented_not_trained_not_performance_qualified",
        "status",
    )

    receipt = candidate["receipt_sha256"]
    if not isinstance(receipt, str) or _SHA256_RE.fullmatch(receipt) is None:
        raise ValueError("receipt_sha256 must be a lowercase SHA-256")
    if receipt != v1_3_min_adversarial_addendum_self_sha256(candidate):
        raise ValueError("v1.3-min adversarial addendum self-hash replay failed")

    if verify_dependencies:
        _validate_additive_bindings(candidate["additive_bindings"])
    _validate_candidate_arms(candidate["candidate_arms"])
    _validate_outer_temporal_and_onset(candidate)
    _validate_loss_and_ranking(candidate)
    _validate_findings_gates_and_firewall(candidate)
    _require_false_mapping(candidate["execution_status"], "execution_status")
    _require_false_mapping(candidate["execution_permissions"], "execution_permissions")
    _require_false_mapping(candidate["scientific_permissions"], "scientific_permissions")

    # Exact checked-in default comparison closes fields not separately named in
    # the semantic assertions above.  A caller cannot add a permissive nested
    # field and merely recompute the canonical receipt.
    if candidate != _trusted_default():
        raise ValueError("v1.3-min candidate differs from the trusted additive default")
    if require_trusted_receipt and receipt != (
        TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_RECEIPT_SHA256
    ):
        raise ValueError("v1.3-min adversarial addendum receipt is not trusted")
    return candidate


def load_v1_3_min_adversarial_addendum(
    path: str | Path = DEFAULT_V1_3_MIN_ADVERSARIAL_ADDENDUM_PATH,
) -> dict[str, Any]:
    """Load a strict JSON copy of the trusted v1.3-min addendum."""

    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = (_ROOT / candidate_path).resolve()
    return validate_v1_3_min_adversarial_addendum(
        _load_strict_json(candidate_path, "v1.3-min adversarial addendum")
    )


def assert_v1_3_min_execution_authorized() -> None:
    """Always fail: this addendum registers method restrictions, not execution."""

    load_v1_3_min_adversarial_addendum()
    raise RuntimeError(
        "v1.3-min execution is not authorized; G0-G5 receipts and a future "
        "additive admission contract are required"
    )


def assert_v1_3_min_performance_claim_authorized() -> None:
    """Always fail until a future immutable performance receipt is admitted."""

    load_v1_3_min_adversarial_addendum()
    raise RuntimeError(
        "v1.3-min performance claims are not authorized by a method contract"
    )


__all__ = [
    "DEFAULT_V1_3_MIN_ADVERSARIAL_ADDENDUM_PATH",
    "TRUSTED_V1_2_SUBMISSION_PROFILE_FILE_SHA256",
    "TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_FILE_SHA256",
    "TRUSTED_V1_3_MIN_ADVERSARIAL_ADDENDUM_RECEIPT_SHA256",
    "TRUSTED_V1_3_MIN_ADVERSARIAL_AUDIT_FILE_SHA256",
    "V1_3_MIN_ADVERSARIAL_ADDENDUM_ID",
    "V1_3_MIN_ADVERSARIAL_ADDENDUM_SCHEMA_VERSION",
    "assert_v1_3_min_execution_authorized",
    "assert_v1_3_min_performance_claim_authorized",
    "load_v1_3_min_adversarial_addendum",
    "v1_3_min_adversarial_addendum_self_sha256",
    "validate_v1_3_min_adversarial_addendum",
]
