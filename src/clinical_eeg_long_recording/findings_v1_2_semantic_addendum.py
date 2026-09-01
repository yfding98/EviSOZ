"""Fail-closed semantic addendum for Findings v1.2.

The immutable v1 Findings profile, composer-closure receipt, and record-level
Context policy remain unchanged.  This additive contract narrows ambiguous
Event Card semantics without claiming that a native producer, clinical term
qualifier, trained model, or target-domain performance now exists.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, Sequence

from .findings_v1_composer_closure_addendum_v1 import (
    DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SHA256,
    FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID,
    load_findings_v1_composer_closure_addendum_v1,
)
from .findings_v1_core_release_profile import (
    DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256,
    FINDINGS_V1_CORE_RELEASE_PROFILE_ID,
    load_findings_v1_core_release_profile,
)
from .record_non_event_context_card_v1 import (
    DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1,
    RECORD_NON_EVENT_CONTEXT_CARD_POLICY_ID_V1,
    load_record_non_event_context_card_policy_v1,
)


FINDINGS_V1_2_SEMANTIC_ADDENDUM_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_v1_2_semantic_addendum_v1"
)
FINDINGS_V1_2_SEMANTIC_ADDENDUM_ID: Final[str] = (
    "CLINICAL-EEG-FINDINGS-V1.2-SEMANTIC-ADDENDUM-20260824"
)
TRUSTED_FINDINGS_V1_2_SEMANTIC_ADDENDUM_RECEIPT_SHA256: Final[str] = (
    "bd7d4a2b3dd0a4b7af75d391ee3ac7e9f1066af5404e1a85d7bc525e47ce6717"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINDINGS_V1_2_SEMANTIC_ADDENDUM_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_findings_v1_2_semantic_addendum.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "addendum_id",
    "status",
    "immutable_dependency_bindings",
    "event_card_slot_semantics",
    "context_onset_noninterference",
    "assertion_and_term_permissions",
    "source_firewall",
    "required_counterexample_contracts",
    "scientific_permissions",
    "receipt_sha256",
}

_EXPECTED_DEPENDENCIES: Mapping[str, object] = {
    "v1_core_release_profile": {
        "path": "configs/clinical_eeg_findings_v1_core_release_profile.json",
        "identifier": FINDINGS_V1_CORE_RELEASE_PROFILE_ID,
        "receipt_sha256": DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256,
        "file_sha256": (
            "bbf43e69c71b8c08dd18ed56f4ed2ae775abdfd793453864a77da98644cf9793"
        ),
        "mutation_required": False,
    },
    "v1_composer_closure_addendum": {
        "path": "configs/clinical_eeg_findings_v1_composer_closure_addendum.json",
        "identifier": FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_ID,
        "receipt_sha256": DEFAULT_FINDINGS_V1_COMPOSER_CLOSURE_ADDENDUM_SHA256,
        "file_sha256": (
            "d156f878342fd524b13aacfba72d80603947c1aaf50b35b5dddb43613d2ce28e"
        ),
        "mutation_required": False,
    },
    "v1_record_non_event_context_policy": {
        "path": "configs/clinical_eeg_record_non_event_context_card_policy_v1.json",
        "identifier": RECORD_NON_EVENT_CONTEXT_CARD_POLICY_ID_V1,
        "receipt_sha256": DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1,
        "file_sha256": (
            "7baaf13739a7f127dce03c5c15525d56d4bff7b17b471d1891777081944c30ac"
        ),
        "mutation_required": False,
    },
}

_EXPECTED_SLOT_SEMANTICS: Mapping[str, object] = {
    "S02_EVENT_BOUNDARY": {
        "required_distinct_interval_fields": [
            "detector_candidate_interval",
            "possible_onset_interval",
            "qualified_onset_interval",
        ],
        "detector_candidate_semantics": "navigation_only_not_onset_fact",
        "possible_onset_semantics": "model_candidate_not_qualified_onset",
        "qualified_onset_requires_independent_target_domain_receipt": True,
        "detector_candidate_can_be_promoted_without_qualification": False,
        "possible_onset_can_be_promoted_without_qualification": False,
    },
    "S05_WAVEFORM_MORPHOLOGY": {
        "event_scope_allowed": [
            "waveform_morphology_primitive",
            "ictal_sharp_contoured_component_candidate",
        ],
        "event_scope_forbidden": [
            "interictal_ied_instance",
            "interictal_epileptiform_discharge",
            "sharp_wave",
            "spike",
        ],
        "interictal_ied_storage_scope": "record_non_event_context_card_only",
        "event_to_context_binding": "context_card_id_only_no_payload_copy",
        "ictal_primitive_can_be_renamed_interictal_ied": False,
    },
    "S06_RHYTHMICITY_PERIODICITY": {
        "required_instance_ledgers": [
            "component_instance_ledger",
            "cycle_instance_ledger",
            "element_instance_ledger",
        ],
        "summary_score_without_instance_ledger_sufficient": False,
        "single_fft_or_autocorrelation_peak_sufficient_for_rhythmicity": False,
        "single_fft_or_autocorrelation_peak_sufficient_for_periodicity": False,
        "clinical_rhythmic_or_periodic_term_requires_independent_qualification": True,
    },
    "S09_CHANGE_POINTS_EVOLUTION": {
        "primary_trajectory_axes": ["frequency", "morphology", "location"],
        "amplitude_trajectory_storage": "separate_nonqualifying_axis",
        "distribution_trajectory_storage": "separate_nonqualifying_axis",
        "amplitude_alone_can_qualify_evolution": False,
        "distribution_alone_can_qualify_location_evolution": False,
        "course_information_can_create_positive_onset_evidence": False,
    },
    "S10_LATER_INVOLVEMENT": {
        "default_term": "scalp_topographic_recruitment_candidate",
        "allowed_measurements": [
            "later_involvement_interval",
            "lead_lag_interval",
            "partial_order",
        ],
        "forbidden_automated_terms": ["ictal_spread", "propagation", "spread"],
        "late_involvement_can_create_or_raise_primary_onset_rank": False,
        "spread_term_requires_independent_qualification": True,
    },
    "S11_CESSATION_RETURN": {
        "default_term": "cessation_or_return_candidate",
        "required_interval_fields": [
            "last_unequivocal_ictal_time",
            "first_stable_return_to_context_time",
        ],
        "per_unit_asynchrony_preserved": True,
        "right_censoring_preserved": True,
        "forbidden_automated_terms": [
            "ictal_termination",
            "postictal",
            "termination",
        ],
        "record_end_or_score_drop_proves_termination": False,
    },
}

_EXPECTED_CONTEXT_NONINTERFERENCE: Mapping[str, object] = {
    "interictal_ied_instances_live_in_context_only": True,
    "event_card_context_reference_is_id_only": True,
    "interictal_concordance_or_conflict_may_be_recorded": True,
    "context_can_create_positive_onset_evidence": False,
    "context_can_change_primary_onset_rank": False,
    "context_can_create_onset_candidate_membership": False,
    "interictal_findings_may_only_support_separate_concordance_analysis": True,
}

_EXPECTED_ASSERTION_PERMISSIONS: Mapping[str, object] = {
    "assertion_vocabulary": [
        "measured",
        "model_candidate",
        "report_eligible_automated",
    ],
    "status_vocabulary": [
        "present",
        "absent_with_opportunity",
        "uncertain",
        "not_evaluable",
    ],
    "measured_terms": [
        "component_cycle_or_element_interval",
        "count_or_burden",
        "duration_ms_or_s",
        "frequency_hz",
        "lead_lag_interval",
        "partial_order",
        "physical_amplitude_uv",
        "quality_or_opportunity",
        "typed_electrode_or_whole_bipolar_lead",
    ],
    "model_candidate_only_terms": [
        "band_limited_low_amplitude_fast_activity_candidate",
        "cessation_or_return_candidate",
        "frequency_morphology_or_location_trajectory_candidate",
        "phase_reversal_or_reference_specific_field_candidate",
        "rhythmic_or_periodic_candidate",
        "scalp_topographic_recruitment_candidate",
        "sharp_contoured_component_candidate",
    ],
    "forbidden_automated_terms_without_independent_qualification": [
        "acns_definite_evolution",
        "cortical_soz_or_ez",
        "diffuse_or_generalized_onset",
        "electrographic_seizure",
        "interictal_epileptiform_discharge",
        "ictal_spread_or_propagation",
        "ictal_termination_or_postictal_state",
        "lvfa_on_scalp_eeg",
        "pathological_theta_or_focal_slowing",
        "sharp_wave",
        "spike",
    ],
    "report_eligible_automated_allowlist": [],
    "not_evaluable_is_negative": False,
    "detector_negative_is_interictal": False,
    "absence_requires_complete_opportunity_and_target_domain_sensitivity_receipt": True,
}

_EXPECTED_SOURCE_FIREWALL: Mapping[str, bool] = {
    "eeg_signal_and_allowlisted_acquisition_metadata_used": True,
    "private_eeg_signal_permitted_under_same_eeg_only_contract": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_or_report_used": False,
    "patient_demographics_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}

_EXPECTED_COUNTEREXAMPLES = (
    "context_concordance_cannot_create_or_reorder_primary_onset_candidates",
    "detector_candidate_cannot_be_relabelled_qualified_onset",
    "event_card_cannot_embed_interictal_ied_payload",
    "late_involvement_cannot_be_relabelled_spread_or_raise_onset_rank",
    "nonempty_report_allowlist_fails_closed",
    "rehash_after_dependency_receipt_tamper_still_fails_closed",
    "s11_score_drop_cannot_be_relabelled_termination_or_postictal",
    "s6_summary_without_component_cycle_element_ledger_is_insufficient",
    "s9_amplitude_or_distribution_alone_cannot_qualify_evolution",
)

_EXPECTED_SCIENTIFIC_PERMISSIONS: Mapping[str, bool] = {
    "old_v1_contract_mutation_authorized": False,
    "semantic_contract_is_implementation_result": False,
    "software_contract_tests_are_model_performance": False,
    "target_domain_finding_accuracy_claim_authorized": False,
    "soz_accuracy_claim_authorized": False,
    "clinical_or_production_use_authorized": False,
    "report_or_qwen_route_connected": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def findings_v1_2_semantic_addendum_self_sha256(
    value: Mapping[str, object],
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("Findings v1.2 semantic addendum must be an object")
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


def _load_strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
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


def _strict_object(
    value: object, expected: Mapping[str, object], context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != set(expected):
        missing = set(expected) - set(value)
        unknown = set(value) - set(expected)
        raise ValueError(
            f"{context} keys drifted; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if value != expected:
        raise ValueError(f"{context} semantic policy drifted")
    return deepcopy(value)


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_immutable_dependencies(value: object) -> dict[str, Any]:
    rows = _strict_object(
        value, _EXPECTED_DEPENDENCIES, "immutable_dependency_bindings"
    )
    for name, row in rows.items():
        path = _safe_project_file(row["path"], f"{name}.path")
        if _file_sha256(path) != row["file_sha256"]:
            raise ValueError(f"{name} immutable file content drifted")

    core_row = rows["v1_core_release_profile"]
    core = load_findings_v1_core_release_profile(
        _safe_project_file(core_row["path"], "v1 core path"),
        trusted_profile_sha256=core_row["receipt_sha256"],
    )
    if core["profile_id"] != core_row["identifier"]:
        raise ValueError("v1 core profile identity drifted")

    composer_row = rows["v1_composer_closure_addendum"]
    composer = load_findings_v1_composer_closure_addendum_v1(
        _safe_project_file(composer_row["path"], "v1 composer path"),
        trusted_receipt_sha256=composer_row["receipt_sha256"],
    )
    if composer["addendum_id"] != composer_row["identifier"]:
        raise ValueError("v1 composer closure identity drifted")

    context_row = rows["v1_record_non_event_context_policy"]
    context = load_record_non_event_context_card_policy_v1(
        _safe_project_file(context_row["path"], "v1 context path"),
        trusted_policy_sha256=context_row["receipt_sha256"],
    )
    if context["policy_id"] != context_row["identifier"]:
        raise ValueError("v1 record Context policy identity drifted")
    return rows


def validate_clinical_eeg_findings_v1_2_semantic_addendum(
    value: Mapping[str, object],
) -> dict[str, Any]:
    """Validate exact semantic ceilings and immutable v1 dependencies."""

    if type(value) is not dict:
        raise TypeError("Findings v1.2 semantic addendum must be a JSON object")
    candidate = deepcopy(value)
    if set(candidate) != _TOP_LEVEL_KEYS:
        missing = _TOP_LEVEL_KEYS - set(candidate)
        unknown = set(candidate) - _TOP_LEVEL_KEYS
        raise ValueError(
            f"Findings v1.2 addendum fields drifted; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if candidate["schema_version"] != FINDINGS_V1_2_SEMANTIC_ADDENDUM_SCHEMA_VERSION:
        raise ValueError("Findings v1.2 addendum schema identity drifted")
    if candidate["addendum_id"] != FINDINGS_V1_2_SEMANTIC_ADDENDUM_ID:
        raise ValueError("Findings v1.2 addendum identity drifted")
    if candidate["status"] != (
        "semantic_contract_frozen_implementation_and_performance_not_established"
    ):
        raise ValueError("Findings v1.2 semantic contract status was promoted")
    receipt = _require_sha256(candidate["receipt_sha256"], "receipt_sha256")
    if receipt != findings_v1_2_semantic_addendum_self_sha256(candidate):
        raise ValueError("Findings v1.2 semantic addendum self-hash mismatch")

    _validate_immutable_dependencies(candidate["immutable_dependency_bindings"])
    _strict_object(
        candidate["event_card_slot_semantics"],
        _EXPECTED_SLOT_SEMANTICS,
        "event_card_slot_semantics",
    )
    _strict_object(
        candidate["context_onset_noninterference"],
        _EXPECTED_CONTEXT_NONINTERFERENCE,
        "context_onset_noninterference",
    )
    _strict_object(
        candidate["assertion_and_term_permissions"],
        _EXPECTED_ASSERTION_PERMISSIONS,
        "assertion_and_term_permissions",
    )
    _strict_object(candidate["source_firewall"], _EXPECTED_SOURCE_FIREWALL, "source_firewall")
    if tuple(candidate["required_counterexample_contracts"]) != _EXPECTED_COUNTEREXAMPLES:
        raise ValueError("Findings v1.2 required counterexamples drifted")
    _strict_object(
        candidate["scientific_permissions"],
        _EXPECTED_SCIENTIFIC_PERMISSIONS,
        "scientific_permissions",
    )
    if receipt != TRUSTED_FINDINGS_V1_2_SEMANTIC_ADDENDUM_RECEIPT_SHA256:
        raise ValueError("Findings v1.2 semantic addendum receipt is not host trusted")
    return candidate


def load_clinical_eeg_findings_v1_2_semantic_addendum(
    path: str | Path = DEFAULT_FINDINGS_V1_2_SEMANTIC_ADDENDUM_PATH,
) -> dict[str, Any]:
    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = (_ROOT / candidate_path).resolve()
    return validate_clinical_eeg_findings_v1_2_semantic_addendum(
        _load_strict_json(candidate_path, "Findings v1.2 semantic addendum")
    )


__all__ = [
    "DEFAULT_FINDINGS_V1_2_SEMANTIC_ADDENDUM_PATH",
    "FINDINGS_V1_2_SEMANTIC_ADDENDUM_ID",
    "FINDINGS_V1_2_SEMANTIC_ADDENDUM_SCHEMA_VERSION",
    "TRUSTED_FINDINGS_V1_2_SEMANTIC_ADDENDUM_RECEIPT_SHA256",
    "findings_v1_2_semantic_addendum_self_sha256",
    "load_clinical_eeg_findings_v1_2_semantic_addendum",
    "validate_clinical_eeg_findings_v1_2_semantic_addendum",
]
