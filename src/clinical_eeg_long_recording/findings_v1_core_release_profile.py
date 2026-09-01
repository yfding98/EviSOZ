"""Fail-closed capability audit for the frozen EEG-only Findings v1-core.

The 12-slot Event Card, six-slot record Context Card, and record Event
Aggregate are structural destinations.  Their existence does not prove that
the atomic evidence producers underneath them are complete.  This module
therefore binds those surfaces to the closed 28-core-atom + 12-child-roster +
41-term-query denominator and emits an explicit readiness receipt.

The receipt is a *software capability* audit, never a patient Finding or a
model-performance result.  It preserves three distinct dispositions:

* ``implemented_now`` -- a v1-core-required shadow query is wired today;
* ``required_gap`` -- a replayable measurement/candidate required by the
  frozen minimal profile is still not closed end to end; and
* ``deferred_term`` -- clinical/acquisition-sensitive qualification was
  deliberately left outside v1-core and remains not evaluable.

No disposition authorizes report text.  The automated report allowlist is
empty, and ``not_evaluable``, technical failure, or a missing candidate can
never be rewritten as a negative.  A clinical absence would additionally
need complete opportunity, completed processing, a matching target-domain
sensitivity/NPV receipt, and an explicit term decision; this profile itself
still does not authorize that absence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_atom_roster import (
    load_event_findings_atom_roster_policy,
    validate_event_findings_atom_roster_policy,
)
from .event_findings_term_query_denominator_v2 import (
    load_event_findings_term_query_denominator_policy_v2,
    validate_event_findings_term_query_denominator_policy_v2,
)
from .minimum_event_evidence_card_registry_v1 import (
    DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1,
    load_minimum_event_evidence_card_registry_v1,
)
from .record_event_aggregate_v1_1 import (
    RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1,
    RECORD_EVENT_AGGREGATE_SCHEMA_PATH_V1_1,
    RECORD_EVENT_AGGREGATE_SCHEMA_VERSION_V1_1,
)
from .record_non_event_context_card_v1 import (
    DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1,
    load_record_non_event_context_card_policy_v1,
)


FINDINGS_V1_CORE_RELEASE_PROFILE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_v1_core_release_profile_v1"
)
FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_findings_v1_core_readiness_receipt_v1"
)
FINDINGS_V1_CORE_RELEASE_PROFILE_ID: Final[str] = (
    "CLINICAL-EEG-FINDINGS-V1-CORE-RELEASE-PROFILE-V1"
)
FINDINGS_V1_CORE_READINESS_METHOD_ID: Final[str] = (
    "CLINICAL-EEG-FINDINGS-V1-CORE-READINESS-AUDIT-V1"
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH: Final[Path] = (
    _ROOT / "configs" / "clinical_eeg_findings_v1_core_release_profile.json"
)
FINDINGS_V1_CORE_RELEASE_PROFILE_SCHEMA_PATH: Final[Path] = (
    _ROOT
    / "schemas"
    / "clinical_eeg_findings_v1_core_release_profile.schema.json"
)
FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_PATH: Final[Path] = (
    _ROOT
    / "schemas"
    / "clinical_eeg_findings_v1_core_readiness_receipt.schema.json"
)
DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256: Final[str] = (
    "6307f421ac90cdf3f8bb48b1985c62f02fe6bed65141570f0261fd2943af7c76"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IMPLEMENTED_PREFIX = "implemented_"
_UNIMPLEMENTED = "unimplemented_not_evaluable"
_DISPOSITIONS = ("implemented_now", "required_gap", "deferred_term")
_CLOSURE_STAGES = (
    "proposal",
    "deterministic_measurement",
    "rule_composer",
)

_ATOM_IMPLEMENTED_NOW = frozenset(
    {
        "c1_composite_pattern_instances",
        "c1_frequency_spectral_profile",
        "c1_occurrence_burden_variability",
        "c1_rhythmicity_profile",
        "c1_waveform_morphology_profile",
        "c2_cross_reference_stability_resolution",
        "c2_earliest_distinguishable_set",
        "c2_involvement_partial_order_near_synchrony",
        "c2_onset_boundary_decision_availability",
        "c2_per_unit_involvement_interval",
        "c2_reference_specific_field_polarity",
        "c2_spatial_involvement_instances",
        "c3_later_involvement_order",
        "c3_offset_cessation_censoring",
        "c3_post_event_change",
        "q_artifact_interval_instances",
        "q_artifact_usable_support",
        "q_background_comparability",
        "q_boundary_censoring_ownership",
        "q_competing_signal_hypotheses_event_outcome",
        "q_signal_clock_channel_coverage",
        "q_view_reference_bandwidth_capability",
    }
)
_ATOM_REQUIRED_GAPS = frozenset(
    {
        "c1_ictal_sharp_component_instances",
        "c1_periodic_element_instances",
        "c1_periodicity_element_interval_profile",
        "c1_physical_amplitude_profile",
        "c1_rhythmic_run_instances",
        "c3_amplitude_course",
        "c3_evolution_transition_instances",
        "c3_frequency_evolution",
        "c3_location_distribution_evolution",
        "c3_morphology_evolution",
        "c3_return_to_comparable_background",
        "c3_rhythmicity_course",
    }
)
_ATOM_DEFERRED = frozenset(
    {
        "a_dc_shift_pattern_instances",
        "a_hfo_pattern_instances",
        "a_lvfa_pattern_instances",
        "a_very_slow_activity_pattern_instances",
        "c1_interictal_ied_instances",
        "c1_interictal_ied_morphology_profile",
    }
)

_QUERY_IMPLEMENTED_NOW = frozenset(
    {
        "TQ-ARTIFACT-INTERVAL",
        "TQ-BACKGROUND-SPECTRAL-PROFILE",
        "TQ-EVENT-FREQUENCY-SPECTRAL-PROFILE",
        "TQ-ONSET-BOUNDARY",
        "TQ-ONSET-CROSS-REFERENCE-RESOLUTION",
        "TQ-ONSET-EARLIEST-SET-MEMBERSHIP",
        "TQ-ONSET-INVOLVEMENT-ORDER-NEAR-SYNCHRONY",
        "TQ-ONSET-LATER-INVOLVEMENT",
        "TQ-ONSET-PER-UNIT-INVOLVEMENT",
        "TQ-ONSET-REFERENCE-FIELD-POLARITY",
        "TQ-ONSET-RESEARCH-SCALP-VISIBLE-HYPOTHESIS",
        "TQ-POST-EVENT-ATTENUATION",
        "TQ-POST-EVENT-SLOWING",
        "TQ-RECOVERY-OFFSET",
        "TQ-RHYTHMICITY-PROFILE",
        "TQ-SIGNAL-USABLE-FRACTION",
    }
)
_QUERY_REQUIRED_GAPS = frozenset(
    {
        "TQ-EVENT-AMPLITUDE-COURSE",
        "TQ-EVENT-RHYTHMICITY-COURSE",
        "TQ-EVOLUTION-FREQUENCY",
        "TQ-EVOLUTION-LOCATION",
        "TQ-EVOLUTION-MORPHOLOGY",
        "TQ-PERIODIC-ELEMENT-INSTANCE",
        "TQ-PHYSICAL-AMPLITUDE-PROFILE",
        "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND",
        "TQ-RHYTHMIC-RUN-INSTANCE",
        "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE",
    }
)
_QUERY_DEFERRED = frozenset(
    {
        "TQ-ACQUISITION-DC-SHIFT",
        "TQ-ACQUISITION-HFO",
        "TQ-ACQUISITION-LVFA",
        "TQ-ACQUISITION-VERY-SLOW",
        "TQ-DEFINITE-EVOLUTION-FREQUENCY",
        "TQ-DEFINITE-EVOLUTION-LOCATION",
        "TQ-DEFINITE-EVOLUTION-MORPHOLOGY",
        "TQ-ELECTROGRAPHIC-SEIZURE",
        "TQ-IED-RECORD-SUMMARY",
        "TQ-PERIODIC-DISCHARGE-SUMMARY",
        "TQ-RHYTHMIC-THETA-SUMMARY",
        "TQ-SHARP-WAVE-INTERICTAL-INSTANCE",
        "TQ-SHARP-WAVE-INTERICTAL-SUMMARY",
        "TQ-SPIKE-INTERICTAL-INSTANCE",
        "TQ-SPIKE-INTERICTAL-SUMMARY",
    }
)

_EXPECTED_GAP_LAYERS: Final[dict[str, tuple[str, ...]]] = {
    "TQ-EVENT-AMPLITUDE-COURSE": (
        "deterministic_measurement",
        "rule_composer",
    ),
    "TQ-EVENT-RHYTHMICITY-COURSE": (
        "deterministic_measurement",
        "rule_composer",
    ),
    "TQ-EVOLUTION-FREQUENCY": ("proposal", "rule_composer"),
    "TQ-EVOLUTION-LOCATION": ("proposal", "rule_composer"),
    "TQ-EVOLUTION-MORPHOLOGY": ("proposal", "rule_composer"),
    "TQ-PERIODIC-ELEMENT-INSTANCE": ("rule_composer",),
    "TQ-PHYSICAL-AMPLITUDE-PROFILE": ("rule_composer",),
    "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND": ("rule_composer",),
    "TQ-RHYTHMIC-RUN-INSTANCE": (
        "proposal",
        "deterministic_measurement",
        "rule_composer",
    ),
    "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE": (
        "proposal",
        "rule_composer",
    ),
}

_DEFERRED_TERM_IDS = frozenset(
    {
        "dc_shift_candidate",
        "definite_evolution",
        "electrographic_seizure",
        "high_frequency_oscillation",
        "interictal_epileptiform_discharge",
        "low_voltage_fast_activity",
        "periodic_discharge",
        "rhythmic_theta_activity",
        "sharp_wave",
        "spike",
        "very_slow_activity_candidate",
    }
)

_ABSENCE_POLICY = {
    "profile_authorizes_clinical_absence": False,
    "complete_opportunity_required": True,
    "completed_processing_required": True,
    "matching_target_domain_sensitivity_or_npv_required": True,
    "explicit_term_decision_required": True,
    "not_evaluable_is_negative": False,
    "technical_failure_is_negative": False,
    "missing_candidate_is_negative": False,
}
_SEMANTIC_PERMISSIONS = {
    "eeg_signal_measurement_authorized": True,
    "scalp_visible_onset_candidate_authorized": True,
    "offline_context_may_create_positive_onset": False,
    "late_spread_may_create_positive_onset": False,
    "tcp_edge_may_be_attributed_to_an_endpoint": False,
    "cortical_soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
}
_SOURCE_FIREWALL = {
    "eeg_samples_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "knowledge_base_used_as_patient_fact": False,
    "llm_used": False,
    "production_report_route_used": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _schema_errors(value: object, path: Path) -> list[str]:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.path),
    )
    rendered: list[str] = []
    for error in errors[:16]:
        pointer = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{pointer}: {error.message}")
    if len(errors) > 16:
        rendered.append(f"... {len(errors) - 16} more error(s)")
    return rendered


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_sorted_unique_ids(value: object, context: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be an ID array")
    result = [str(item) for item in value]
    if any(_ID_RE.fullmatch(item) is None for item in result):
        raise ValueError(f"{context} contains an invalid ID")
    if result != sorted(result) or len(result) != len(set(result)):
        raise ValueError(f"{context} must be sorted and unique")
    return result


def _partition_map(
    value: Mapping[str, object],
    *,
    context: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for disposition in _DISPOSITIONS:
        ids = _require_sorted_unique_ids(
            value[disposition], f"{context}.{disposition}"
        )
        for item_id in ids:
            if item_id in result:
                raise ValueError(f"{context} partitions overlap at {item_id}")
            result[item_id] = disposition
    return result


def _expected_partition_map(
    implemented: frozenset[str],
    gaps: frozenset[str],
    deferred: frozenset[str],
) -> dict[str, str]:
    if implemented & gaps or implemented & deferred or gaps & deferred:
        raise AssertionError("frozen readiness partitions overlap")
    return {
        **{item: "implemented_now" for item in implemented},
        **{item: "required_gap" for item in gaps},
        **{item: "deferred_term" for item in deferred},
    }


_EXPECTED_ATOM_PARTITION = _expected_partition_map(
    _ATOM_IMPLEMENTED_NOW, _ATOM_REQUIRED_GAPS, _ATOM_DEFERRED
)
_EXPECTED_QUERY_PARTITION = _expected_partition_map(
    _QUERY_IMPLEMENTED_NOW, _QUERY_REQUIRED_GAPS, _QUERY_DEFERRED
)


def validate_findings_v1_core_release_profile(
    value: object,
    *,
    trusted_profile_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the frozen profile, its self-hash, and semantic ceilings."""

    if type(value) is not dict:
        raise TypeError("Findings v1-core release profile must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, FINDINGS_V1_CORE_RELEASE_PROFILE_SCHEMA_PATH)
    if errors:
        raise ValueError("release profile schema validation failed: " + "; ".join(errors))
    expected_hash = _self_hash(candidate, "profile_sha256")
    if candidate["profile_sha256"] != expected_hash:
        raise ValueError("Findings v1-core release profile SHA-256 mismatch")
    if trusted_profile_sha256 is not None and expected_hash != _require_sha256(
        trusted_profile_sha256, "trusted_profile_sha256"
    ):
        raise ValueError("Findings v1-core release profile is not host trusted")
    if candidate["profile_id"] != FINDINGS_V1_CORE_RELEASE_PROFILE_ID:
        raise ValueError("Findings v1-core release profile ID drifted")

    partition = candidate["partition_policy"]
    atom_map = _partition_map(
        partition["atom_roster_partitions"], context="atom_roster_partitions"
    )
    query_map = _partition_map(
        partition["term_query_partitions"], context="term_query_partitions"
    )
    if atom_map != _EXPECTED_ATOM_PARTITION:
        raise ValueError("atom-roster v1-core readiness partition drifted")
    if query_map != _EXPECTED_QUERY_PARTITION:
        raise ValueError("term-query v1-core readiness partition drifted")
    if len(atom_map) != 40 or len(query_map) != 41:
        raise ValueError("v1-core release profile denominator is not closed")

    gap_specs = list(candidate["required_query_gap_specs"])
    gap_ids = [str(row["term_query_id"]) for row in gap_specs]
    if gap_ids != sorted(_QUERY_REQUIRED_GAPS):
        raise ValueError("required query gap specs are not the exact frozen set")
    for row in gap_specs:
        query_id = str(row["term_query_id"])
        layers = tuple(str(item) for item in row["blocking_closure_layers"])
        if layers != _EXPECTED_GAP_LAYERS[query_id]:
            raise ValueError(f"{query_id}: blocking closure layers drifted")
        evidence = _require_sorted_unique_ids(
            row["existing_component_evidence"],
            f"{query_id}.existing_component_evidence",
        )
        if not evidence:
            raise ValueError(f"{query_id}: component-level audit evidence is empty")

    boundary = candidate["qualification_boundary"]
    if boundary["effective_assertion_levels"] != ["measured", "model_candidate"]:
        raise ValueError("v1-core assertion ceiling drifted")
    if boundary["report_eligible_automated_allowlist"] != []:
        raise ValueError("automated report allowlist must remain empty")
    if boundary["positive_clinical_qualification_enabled"] is not False:
        raise ValueError("positive clinical qualification must remain disabled")
    deferred_terms = _require_sorted_unique_ids(
        boundary["clinical_or_acquisition_deferred_term_ids"],
        "clinical_or_acquisition_deferred_term_ids",
    )
    if set(deferred_terms) != set(_DEFERRED_TERM_IDS):
        raise ValueError("deferred clinical/acquisition term set drifted")
    if boundary["absence_policy"] != _ABSENCE_POLICY:
        raise ValueError("absence/opportunity/NPV fail-closed policy drifted")
    if boundary["semantic_permissions"] != _SEMANTIC_PERMISSIONS:
        raise ValueError("EEG-only semantic permission boundary drifted")
    if candidate["source_firewall"] != _SOURCE_FIREWALL:
        raise ValueError("EEG-only patient-fact source firewall drifted")
    return candidate


def load_findings_v1_core_release_profile(
    path: str | Path = DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH,
    *,
    trusted_profile_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the checked-in profile under its host trust anchor."""

    resolved = Path(path)
    if trusted_profile_sha256 is None:
        if resolved.resolve() != DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH.resolve():
            raise ValueError("a non-default release profile requires a trust anchor")
        trusted_profile_sha256 = DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256
    return validate_findings_v1_core_release_profile(
        _read_json(resolved), trusted_profile_sha256=trusted_profile_sha256
    )


def _validated_atom_policy(
    value: Mapping[str, object] | None,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if value is None:
        result = load_event_findings_atom_roster_policy(
            trusted_policy_sha256=expected_sha256
        )
    else:
        result = validate_event_findings_atom_roster_policy(
            dict(value), trusted_policy_sha256=expected_sha256
        )
    if result["policy_sha256"] != expected_sha256:
        raise ValueError("atom-roster policy differs from release profile binding")
    return result


def _validated_query_policy(
    value: Mapping[str, object] | None,
    *,
    expected_sha256: str,
    atom_policy: Mapping[str, object],
) -> dict[str, Any]:
    kwargs = {
        "atom_roster_policy": atom_policy,
        "trusted_atom_roster_policy_sha256": str(atom_policy["policy_sha256"]),
    }
    if value is None:
        result = load_event_findings_term_query_denominator_policy_v2(
            trusted_policy_sha256=expected_sha256,
            **kwargs,
        )
    else:
        result = validate_event_findings_term_query_denominator_policy_v2(
            dict(value), trusted_policy_sha256=expected_sha256, **kwargs
        )
    if result["policy_sha256"] != expected_sha256:
        raise ValueError("term-query policy differs from release profile binding")
    return result


def _observed_surfaces(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_registry = load_minimum_event_evidence_card_registry_v1(
        trusted_registry_sha256=(
            DEFAULT_MINIMUM_EVENT_EVIDENCE_CARD_REGISTRY_SHA256_V1
        )
    )
    context_policy = load_record_non_event_context_card_policy_v1(
        trusted_policy_sha256=(
            DEFAULT_RECORD_NON_EVENT_CONTEXT_CARD_POLICY_SHA256_V1
        )
    )
    aggregate_schema_sha256 = hashlib.sha256(
        RECORD_EVENT_AGGREGATE_SCHEMA_PATH_V1_1.read_bytes()
    ).hexdigest()
    observed = {
        "event_card_12_slot": {
            "contract_id": str(event_registry["registry_id"]),
            "contract_sha256": str(event_registry["registry_sha256"]),
            "observed_slot_or_query_count": len(event_registry["slots"]),
        },
        "record_context_card_6_slot": {
            "contract_id": str(context_policy["policy_id"]),
            "contract_sha256": str(context_policy["policy_sha256"]),
            "observed_slot_or_query_count": len(context_policy["slots"]),
        },
        "record_event_aggregate": {
            "contract_id": RECORD_EVENT_AGGREGATE_SCHEMA_VERSION_V1_1,
            "contract_sha256": aggregate_schema_sha256,
            "observed_slot_or_query_count": len(RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1),
        },
    }
    result: list[dict[str, Any]] = []
    for expected in profile["surface_contracts"]:
        surface_id = str(expected["surface_id"])
        if surface_id not in observed:
            raise ValueError(f"unknown surface contract {surface_id}")
        row = observed[surface_id]
        if row["contract_id"] != expected["contract_id"]:
            raise ValueError(f"{surface_id}: contract ID differs from profile")
        if row["contract_sha256"] != expected["contract_sha256"]:
            raise ValueError(f"{surface_id}: contract hash differs from profile")
        if (
            row["observed_slot_or_query_count"]
            != expected["expected_slot_or_query_count"]
        ):
            raise ValueError(f"{surface_id}: slot/query count differs from profile")
        result.append(
            {
                "surface_id": surface_id,
                **row,
                "status": "implemented_structural_shadow",
                "closes_atomic_query_gaps": False,
            }
        )
    return sorted(result, key=lambda item: str(item["surface_id"]))


def _source_roster_rows(
    atom_policy: Mapping[str, Any],
) -> tuple[list[tuple[str, Mapping[str, Any]]], dict[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in atom_policy["core_atom_specs"]:
        row = dict(raw)
        item_id = str(row["atom_id"])
        rows.append(("core_atom", row))
        by_id[item_id] = row
    for raw in atom_policy["child_roster_specs"]:
        row = dict(raw)
        item_id = str(row["child_roster_id"])
        rows.append(("child_roster", row))
        by_id[item_id] = row
    if set(by_id) != set(_EXPECTED_ATOM_PARTITION):
        raise ValueError("observed atom/child roster differs from release denominator")
    return rows, by_id


def _readiness(disposition: str) -> str:
    return {
        "implemented_now": "implemented_now",
        "required_gap": "required_gap",
        "deferred_term": "intentionally_deferred",
    }[disposition]


def _validate_observed_implementation(
    item_id: str,
    implementation: str,
    disposition: str,
) -> None:
    if disposition == "implemented_now":
        if not implementation.startswith(_IMPLEMENTED_PREFIX):
            raise ValueError(
                f"{item_id}: profile says implemented_now but source says {implementation}"
            )
        return
    if implementation != _UNIMPLEMENTED:
        raise ValueError(
            f"{item_id}: frozen gap/deferred status changed; refresh the release profile"
        )


def _atom_receipt_rows(
    atom_policy: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    source_rows, by_id = _source_roster_rows(atom_policy)
    allowed_assertions = list(profile["qualification_boundary"]["effective_assertion_levels"])
    deferred_terms = set(
        profile["qualification_boundary"][
            "clinical_or_acquisition_deferred_term_ids"
        ]
    )
    result: list[dict[str, Any]] = []
    for kind, source in source_rows:
        item_id = str(
            source["atom_id"] if kind == "core_atom" else source["child_roster_id"]
        )
        disposition = _EXPECTED_ATOM_PARTITION[item_id]
        implementation = str(source["current_implementation_status"])
        _validate_observed_implementation(item_id, implementation, disposition)
        source_assertions = [str(item) for item in source["allowed_assertion_levels"]]
        effective = (
            []
            if disposition == "deferred_term"
            else [item for item in allowed_assertions if item in source_assertions]
        )
        source_terms = sorted(str(item) for item in source["allowed_term_ids"])
        permitted_terms = (
            []
            if disposition == "deferred_term"
            else sorted(item for item in source_terms if item not in deferred_terms)
        )
        result.append(
            {
                "roster_item_kind": kind,
                "roster_item_id": item_id,
                "observed_implementation_status": implementation,
                "release_disposition": disposition,
                "v1_core_required": disposition != "deferred_term",
                "current_readiness": _readiness(disposition),
                "source_allowed_assertion_levels": source_assertions,
                "effective_assertion_levels": effective,
                "source_allowed_term_ids": source_terms,
                "v1_permitted_term_ids": permitted_terms,
                "report_promotion_authorized": False,
                "clinical_absence_authorized": False,
            }
        )
    return sorted(result, key=lambda item: str(item["roster_item_id"])), by_id


def _query_receipt_rows(
    query_policy: Mapping[str, Any],
    atom_by_id: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gap_layers = {
        str(row["term_query_id"]): list(row["blocking_closure_layers"])
        for row in profile["required_query_gap_specs"]
    }
    allowed_assertions = list(profile["qualification_boundary"]["effective_assertion_levels"])
    result: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for raw in query_policy["query_specs"]:
        source = dict(raw)
        query_id = str(source["term_query_id"])
        observed_ids.add(query_id)
        disposition = _EXPECTED_QUERY_PARTITION[query_id]
        implementation = str(source["implementation_status"])
        _validate_observed_implementation(query_id, implementation, disposition)
        primary = source["primary_roster_item"]
        primary_id = str(primary["roster_item_id"])
        primary_source = atom_by_id[primary_id]
        source_assertions = [
            str(item) for item in primary_source["allowed_assertion_levels"]
        ]
        effective = (
            []
            if disposition == "deferred_term"
            else [item for item in allowed_assertions if item in source_assertions]
        )
        if source["report_promotion_authorized"] is not False:
            raise ValueError(f"{query_id}: source query unexpectedly authorizes report promotion")
        result.append(
            {
                "term_query_id": query_id,
                "term_id": str(source["term_id"]),
                "claim_kind": str(source["claim_kind"]),
                "primary_roster_item_kind": str(primary["roster_item_kind"]),
                "primary_roster_item_id": primary_id,
                "observed_implementation_status": implementation,
                "release_disposition": disposition,
                "v1_core_required": disposition != "deferred_term",
                "current_readiness": _readiness(disposition),
                "blocking_closure_layers": gap_layers.get(query_id, []),
                "effective_assertion_levels": effective,
                "negative_semantics": str(source["negative_semantics"]),
                "onset_support_permission": str(source["onset_support_permission"]),
                "report_promotion_authorized": False,
                "positive_clinical_qualification_status": (
                    "deferred_fail_closed_no_report_allowlist"
                ),
                "clinical_absence_authorized": False,
            }
        )
    if observed_ids != set(_EXPECTED_QUERY_PARTITION):
        raise ValueError("observed term-query roster differs from release denominator")
    return sorted(result, key=lambda item: str(item["term_query_id"]))


def _summary(partition: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    implemented = sorted(str(item) for item in partition["implemented_now"])
    gaps = sorted(str(item) for item in partition["required_gap"])
    deferred = sorted(str(item) for item in partition["deferred_term"])
    return {
        "total_count": len(implemented) + len(gaps) + len(deferred),
        "implemented_now_count": len(implemented),
        "required_gap_count": len(gaps),
        "deferred_term_count": len(deferred),
        "implemented_now_ids": implemented,
        "required_gap_ids": gaps,
        "deferred_term_ids": deferred,
    }


def _stage_readiness(
    profile: Mapping[str, Any],
    query_ids: Sequence[str],
) -> list[dict[str, Any]]:
    gap_specs = list(profile["required_query_gap_specs"])
    result: list[dict[str, Any]] = []
    for stage in _CLOSURE_STAGES:
        gaps = sorted(
            str(row["term_query_id"])
            for row in gap_specs
            if stage in row["blocking_closure_layers"]
        )
        result.append(
            {
                "stage": stage,
                "status": "required_gap",
                "required_gap_query_ids": gaps,
                "positive_qualification_deferred_query_ids": [],
            }
        )
    result.append(
        {
            "stage": "qualification",
            "status": "implemented_fail_closed_positive_qualification_deferred",
            "required_gap_query_ids": [],
            "positive_qualification_deferred_query_ids": sorted(query_ids),
        }
    )
    return result


def _build_readiness_receipt(
    profile: Mapping[str, Any],
    atom_policy: Mapping[str, Any],
    query_policy: Mapping[str, Any],
) -> dict[str, Any]:
    surfaces = _observed_surfaces(profile)
    atom_rows, atom_by_id = _atom_receipt_rows(atom_policy, profile)
    query_rows = _query_receipt_rows(query_policy, atom_by_id, profile)
    partitions = profile["partition_policy"]
    atom_summary = _summary(partitions["atom_roster_partitions"])
    query_summary = _summary(partitions["term_query_partitions"])
    blocker_ids = sorted(
        set(atom_summary["required_gap_ids"]) | set(query_summary["required_gap_ids"])
    )
    source_bindings = {
        "atom_roster_policy_id": str(atom_policy["roster_id"]),
        "atom_roster_policy_sha256": str(atom_policy["policy_sha256"]),
        "term_query_policy_id": str(query_policy["policy_id"]),
        "term_query_policy_sha256": str(query_policy["policy_sha256"]),
    }
    receipt_seed = _canonical_sha256(
        {
            "profile_sha256": profile["profile_sha256"],
            "source_contract_bindings": source_bindings,
            "surface_readiness": surfaces,
        }
    )[:24]
    release_ready = not blocker_ids
    body: dict[str, Any] = {
        "schema_version": FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_VERSION,
        "method_id": FINDINGS_V1_CORE_READINESS_METHOD_ID,
        "receipt_id": f"FINDINGS-V1-CORE-READINESS-{receipt_seed}",
        "profile_binding": {
            "profile_id": str(profile["profile_id"]),
            "profile_sha256": str(profile["profile_sha256"]),
        },
        "source_contract_bindings": source_bindings,
        "surface_readiness": surfaces,
        "atom_roster_rows": atom_rows,
        "term_query_rows": query_rows,
        "readiness_summary": {
            "atom_roster": atom_summary,
            "term_queries": query_summary,
        },
        "stage_readiness": _stage_readiness(
            profile, [str(row["term_query_id"]) for row in query_rows]
        ),
        "qualification_boundary": deepcopy(profile["qualification_boundary"]),
        "source_firewall": deepcopy(profile["source_firewall"]),
        "release_ready": release_ready,
        "readiness_status": (
            "ready_v1_core" if release_ready else "not_ready_required_core_gaps"
        ),
        "blocker_ids": blocker_ids,
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def materialize_findings_v1_core_readiness_receipt(
    *,
    profile: Mapping[str, object] | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    term_query_policy: Mapping[str, object] | None = None,
    trusted_profile_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize the source-bound, non-patient v1-core readiness receipt."""

    if profile is None:
        checked_profile = load_findings_v1_core_release_profile(
            trusted_profile_sha256=trusted_profile_sha256
        )
    else:
        if trusted_profile_sha256 is None:
            trusted_profile_sha256 = DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256
        checked_profile = validate_findings_v1_core_release_profile(
            dict(profile), trusted_profile_sha256=trusted_profile_sha256
        )
    sources = checked_profile["source_contracts"]
    atom = _validated_atom_policy(
        atom_roster_policy,
        expected_sha256=str(sources["atom_roster_policy_sha256"]),
    )
    query = _validated_query_policy(
        term_query_policy,
        expected_sha256=str(sources["term_query_policy_sha256"]),
        atom_policy=atom,
    )
    if atom["roster_id"] != sources["atom_roster_policy_id"]:
        raise ValueError("atom-roster policy ID differs from release profile")
    if query["policy_id"] != sources["term_query_policy_id"]:
        raise ValueError("term-query policy ID differs from release profile")
    receipt = _build_readiness_receipt(checked_profile, atom, query)
    errors = _schema_errors(receipt, FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_PATH)
    if errors:
        raise ValueError("readiness receipt schema validation failed: " + "; ".join(errors))
    return receipt


def validate_findings_v1_core_readiness_receipt(
    value: object,
    *,
    profile: Mapping[str, object] | None = None,
    atom_roster_policy: Mapping[str, object] | None = None,
    term_query_policy: Mapping[str, object] | None = None,
    trusted_profile_sha256: str | None = None,
    trusted_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay the readiness audit and reject any mutated or stale receipt."""

    if type(value) is not dict:
        raise TypeError("Findings v1-core readiness receipt must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate, FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_PATH)
    if errors:
        raise ValueError("readiness receipt schema validation failed: " + "; ".join(errors))
    expected_hash = _self_hash(candidate, "receipt_sha256")
    if candidate["receipt_sha256"] != expected_hash:
        raise ValueError("Findings v1-core readiness receipt SHA-256 mismatch")
    if trusted_receipt_sha256 is not None and expected_hash != _require_sha256(
        trusted_receipt_sha256, "trusted_receipt_sha256"
    ):
        raise ValueError("Findings v1-core readiness receipt is not host trusted")
    expected = materialize_findings_v1_core_readiness_receipt(
        profile=profile,
        atom_roster_policy=atom_roster_policy,
        term_query_policy=term_query_policy,
        trusted_profile_sha256=trusted_profile_sha256,
    )
    if candidate != expected:
        raise ValueError("Findings v1-core readiness receipt does not replay exactly")
    return candidate


__all__ = [
    "DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_PATH",
    "DEFAULT_FINDINGS_V1_CORE_RELEASE_PROFILE_SHA256",
    "FINDINGS_V1_CORE_READINESS_METHOD_ID",
    "FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_PATH",
    "FINDINGS_V1_CORE_READINESS_RECEIPT_SCHEMA_VERSION",
    "FINDINGS_V1_CORE_RELEASE_PROFILE_ID",
    "FINDINGS_V1_CORE_RELEASE_PROFILE_SCHEMA_PATH",
    "FINDINGS_V1_CORE_RELEASE_PROFILE_SCHEMA_VERSION",
    "load_findings_v1_core_release_profile",
    "materialize_findings_v1_core_readiness_receipt",
    "validate_findings_v1_core_readiness_receipt",
    "validate_findings_v1_core_release_profile",
]
